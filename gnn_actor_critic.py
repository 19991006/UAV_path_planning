"""
Graph actor-critic modules for strict CTDE GNN-MAPPO / DA-MAPPO.

Design:
- Actor is a shared ego-graph policy. For each UAV i, it receives only ego_graph_i:
  center node = UAV i; neighbor nodes = UAVs within communication range; non-neighbor
  padded slots are zeroed and have edge_mask=0. It outputs only the center action.
- Critic is centralized. It receives the full UAV graph and outputs scalar V(s).
- No PyTorch Geometric dependency is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class GraphNetworkConfig:
    node_dim: int
    edge_dim: int
    action_dim: int = 2
    hidden_dim: int = 256
    num_gnn_layers: int = 3
    activation: str = "relu"
    log_std_init: float = -0.5
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    use_orthogonal_init: bool = False


class TanhNormal:
    """Tanh-squashed Normal distribution for actions in [-1, 1]."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = log_std
        self.std = torch.exp(log_std)
        self.normal = Normal(mean, self.std)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_action = self.normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = self.log_prob_from_raw(raw_action, action)
        return action, log_prob

    def deterministic(self) -> torch.Tensor:
        return torch.tanh(self.mean)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action, -0.999999, 0.999999)
        raw_action = torch.atanh(action)
        return self.log_prob_from_raw(raw_action, action)

    def log_prob_from_raw(self, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        log_prob = self.normal.log_prob(raw_action)
        correction = torch.log(1.0 - action.pow(2) + 1e-6)
        return (log_prob - correction).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.normal.entropy().sum(dim=-1)


def get_activation(name: str) -> type[nn.Module]:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name == "elu":
        return nn.ELU
    raise ValueError(f"Unknown activation: {name}")


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, activation: str) -> nn.Sequential:
    act = get_activation(activation)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        act(),
        nn.Linear(hidden_dim, output_dim),
        act(),
    )


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class GraphMessagePassingLayer(nn.Module):
    """Directed message passing with edge features and an edge mask.

    Supports:
        h:          [N, H] or [B, N, H]
        edge_index: [2, E]
        edge_attr:  [E, F] or [B, E, F]

    The last edge feature is edge_mask in [0, 1]. mask=0 forces the edge to send
    no message. This is used by the ego actor to block padded / non-neighbor slots.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, activation: str = "relu"):
        super().__init__()
        act = get_activation(activation)
        self.edge_dim = edge_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze_batch = True
        elif h.dim() != 3:
            raise ValueError(f"Expected h dim 2 or 3, got {tuple(h.shape)}")

        if edge_attr.dim() == 2:
            edge_attr = edge_attr.unsqueeze(0).expand(h.shape[0], -1, -1)
        elif edge_attr.dim() != 3:
            raise ValueError(f"Expected edge_attr dim 2 or 3, got {tuple(edge_attr.shape)}")

        edge_index = edge_index.long().to(h.device)
        edge_attr = edge_attr.to(device=h.device, dtype=h.dtype)
        src, dst = edge_index[0], edge_index[1]

        if edge_attr.shape[-1] != self.edge_dim:
            raise ValueError(f"edge_attr last dim {edge_attr.shape[-1]}, expected {self.edge_dim}")

        if src.numel() == 0:
            agg = torch.zeros_like(h)
        else:
            edge_mask = edge_attr[..., -1:].clamp(0.0, 1.0)  # [B, E, 1]
            h_src = h[:, src, :]
            h_dst = h[:, dst, :]
            msg_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
            messages = self.message_mlp(msg_input) * edge_mask

            agg = torch.zeros_like(h)
            agg.index_add_(1, dst, messages)

            deg = torch.zeros((h.shape[0], h.shape[1]), device=h.device, dtype=h.dtype)
            deg.index_add_(1, dst, edge_mask.squeeze(-1))
            agg = agg / deg.clamp_min(1.0).unsqueeze(-1)

        updated = self.update_mlp(torch.cat([h, agg], dim=-1))
        h = h + updated
        if squeeze_batch:
            h = h.squeeze(0)
        return h


class EgoGraphActor(nn.Module):
    """Shared decentralized actor over padded ego graphs.

    Input shapes:
        ego_node_features: [M, D], [G, M, D], or [B, N, M, D]
        ego_edge_index:    [2, E] using local ego-node indices
        ego_edge_attr:     [E, F], [G, E, F], or [B, N, E, F]

    Output distribution mean shapes:
        [action_dim], [G, action_dim], or [B, N, action_dim]

    Only node 0, the ego center node, is used for the action output.
    """

    def __init__(self, config: GraphNetworkConfig):
        super().__init__()
        self.config = config
        self.node_encoder = make_mlp(config.node_dim, config.hidden_dim, config.hidden_dim, config.activation)
        self.gnn_layers = nn.ModuleList(
            [GraphMessagePassingLayer(config.hidden_dim, config.edge_dim, config.activation)
             for _ in range(config.num_gnn_layers)]
        )
        self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Parameter(torch.full((config.action_dim,), config.log_std_init))

        if config.use_orthogonal_init:
            orthogonal_init(self)
            nn.init.constant_(self.mean_head.bias, 0.0)

    def _flatten_ego_batch(self, ego_node_features: torch.Tensor, ego_edge_attr: torch.Tensor):
        if ego_node_features.dim() == 2:
            leading_shape = ()
            flat_nodes = ego_node_features.unsqueeze(0)
        elif ego_node_features.dim() >= 3:
            leading_shape = tuple(ego_node_features.shape[:-2])
            flat_nodes = ego_node_features.reshape(-1, ego_node_features.shape[-2], ego_node_features.shape[-1])
        else:
            raise ValueError(f"Invalid ego_node_features shape {tuple(ego_node_features.shape)}")

        num_graphs = flat_nodes.shape[0]
        if ego_edge_attr.dim() == 2:
            flat_edges = ego_edge_attr.unsqueeze(0).expand(num_graphs, -1, -1)
        elif ego_edge_attr.dim() >= 3:
            flat_edges = ego_edge_attr.reshape(-1, ego_edge_attr.shape[-2], ego_edge_attr.shape[-1])
            if flat_edges.shape[0] != num_graphs:
                raise ValueError(
                    f"ego_edge_attr leading graphs {flat_edges.shape[0]} != node graphs {num_graphs}"
                )
        else:
            raise ValueError(f"Invalid ego_edge_attr shape {tuple(ego_edge_attr.shape)}")
        return leading_shape, flat_nodes, flat_edges

    def forward(
        self,
        ego_node_features: torch.Tensor,
        ego_edge_index: torch.Tensor,
        ego_edge_attr: torch.Tensor,
    ) -> TanhNormal:
        leading_shape, flat_nodes, flat_edges = self._flatten_ego_batch(ego_node_features, ego_edge_attr)

        h = self.node_encoder(flat_nodes)
        for layer in self.gnn_layers:
            h = layer(h, ego_edge_index, flat_edges)

        center_h = h[:, 0, :]  # node 0 is the controlled ego UAV
        mean = self.mean_head(center_h)
        if leading_shape:
            mean = mean.reshape(*leading_shape, self.config.action_dim)
        else:
            mean = mean.squeeze(0)

        log_std = torch.clamp(self.log_std, self.config.min_log_std, self.config.max_log_std)
        log_std = log_std.expand_as(mean)
        return TanhNormal(mean, log_std)

    def act(
        self,
        ego_node_features: torch.Tensor,
        ego_edge_index: torch.Tensor,
        ego_edge_attr: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(ego_node_features, ego_edge_index, ego_edge_attr)
        if deterministic:
            actions = dist.deterministic()
            log_probs = dist.log_prob(actions)
        else:
            actions, log_probs = dist.sample()
        return actions, log_probs

    def evaluate_actions(
        self,
        ego_node_features: torch.Tensor,
        ego_edge_index: torch.Tensor,
        ego_edge_attr: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(ego_node_features, ego_edge_index, ego_edge_attr)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class GraphCritic(nn.Module):
    """Graph-level centralized critic with mean pooling over UAV nodes."""

    def __init__(self, config: GraphNetworkConfig):
        super().__init__()
        self.config = config
        self.node_encoder = make_mlp(config.node_dim, config.hidden_dim, config.hidden_dim, config.activation)
        self.gnn_layers = nn.ModuleList(
            [GraphMessagePassingLayer(config.hidden_dim, config.edge_dim, config.activation)
             for _ in range(config.num_gnn_layers)]
        )
        act = get_activation(config.activation)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            act(),
            nn.Linear(config.hidden_dim, 1),
        )

        if config.use_orthogonal_init:
            orthogonal_init(self)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            squeeze_batch = True
        elif node_features.dim() != 3:
            raise ValueError(f"Expected node_features dim 2 or 3, got {tuple(node_features.shape)}")

        if edge_attr.dim() == 2:
            edge_attr = edge_attr.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        elif edge_attr.dim() != 3:
            raise ValueError(f"Expected edge_attr dim 2 or 3, got {tuple(edge_attr.shape)}")

        h = self.node_encoder(node_features)
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_attr)

        graph_embedding = h.mean(dim=1)
        value = self.value_head(graph_embedding).squeeze(-1)
        if squeeze_batch:
            value = value.squeeze(0)
        return value


class GraphActorCritic(nn.Module):
    """Shared ego-graph actor + full-graph centralized critic."""

    def __init__(self, config: GraphNetworkConfig):
        super().__init__()
        self.actor = EgoGraphActor(config)
        self.critic = GraphCritic(config)

    def act(
        self,
        actor_ego_node_features: torch.Tensor,
        actor_ego_edge_index: torch.Tensor,
        actor_ego_edge_attr: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor.act(
            actor_ego_node_features,
            actor_ego_edge_index,
            actor_ego_edge_attr,
            deterministic=deterministic,
        )

    def value(
        self,
        critic_node_features: torch.Tensor,
        critic_edge_index: torch.Tensor,
        critic_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic(critic_node_features, critic_edge_index, critic_edge_attr)

    def evaluate_actions(
        self,
        actor_ego_node_features: torch.Tensor,
        actor_ego_edge_index: torch.Tensor,
        actor_ego_edge_attr: torch.Tensor,
        critic_node_features: torch.Tensor,
        critic_edge_index: torch.Tensor,
        critic_edge_attr: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, entropy = self.actor.evaluate_actions(
            actor_ego_node_features,
            actor_ego_edge_index,
            actor_ego_edge_attr,
            actions,
        )
        values = self.critic(critic_node_features, critic_edge_index, critic_edge_attr)
        return log_probs, entropy, values
