"""
Graph actor-critic modules for GNN-MAPPO / DA-MAPPO.

This version is designed for agent-count generalization:
- Each UAV is represented as one graph node.
- Teammate information is represented as directed edge features.
- Actor produces one continuous action distribution per node.
- Critic pools node embeddings into one graph-level V(s).

No PyTorch Geometric dependency is required. The message-passing layer is
implemented with native PyTorch operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class GraphNetworkConfig:
    """GNN actor-critic hyperparameters."""

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
        # Practical approximation: entropy of pre-squash Gaussian.
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
    """One directed message-passing layer with edge features.

    Always operates on 2D tensors. Multiple graphs are handled by
    concatenating them into one large graph with offset edge indices.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, activation: str = "relu"):
        super().__init__()
        act = get_activation(activation)
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

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # h:         [N_total, H]
        # edge_index: [2, E_total]
        # edge_attr:  [E_total, F]
        edge_index = edge_index.long().to(h.device)
        edge_attr = edge_attr.to(device=h.device, dtype=h.dtype)
        src, dst = edge_index[0], edge_index[1]

        h_src = h[src]   # [E_total, H]
        h_dst = h[dst]   # [E_total, H]
        msg_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        messages = self.message_mlp(msg_input)   # [E_total, H]

        agg = torch.zeros_like(h)                # [N_total, H]
        agg.index_add_(0, dst, messages)

        deg = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        ones = torch.ones(dst.shape[0], device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, ones)
        agg = agg / deg.clamp_min(1.0).view(-1, 1)

        updated = self.update_mlp(torch.cat([h, agg], dim=-1))
        h = h + updated   # residual connection
        return h


class GraphActor(nn.Module):
    """Shared graph actor. Produces one action distribution per UAV node."""

    def __init__(self, config: GraphNetworkConfig):
        super().__init__()
        self.config = config
        self.node_encoder = make_mlp(config.node_dim, config.hidden_dim, config.hidden_dim, config.activation)
        self.gnn_layers = nn.ModuleList(
            [
                GraphMessagePassingLayer(config.hidden_dim, config.edge_dim, config.activation)
                for _ in range(config.num_gnn_layers)
            ]
        )
        self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Parameter(torch.full((config.action_dim,), config.log_std_init))

        if config.use_orthogonal_init:
            orthogonal_init(self)
            nn.init.constant_(self.mean_head.bias, 0.0)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> TanhNormal:
        """node_features: [N_total, node_dim], returns TanhNormal with mean [N_total, action_dim]."""
        h = self.node_encoder(node_features)
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_attr)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std, self.config.min_log_std, self.config.max_log_std)
        log_std = log_std.expand_as(mean)
        return TanhNormal(mean, log_std)

    def act(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(node_features, edge_index, edge_attr)
        if deterministic:
            actions = dist.deterministic()
            log_probs = dist.log_prob(actions)
        else:
            actions, log_probs = dist.sample()
        return actions, log_probs

    def evaluate_actions(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(node_features, edge_index, edge_attr)
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
            [
                GraphMessagePassingLayer(config.hidden_dim, config.edge_dim, config.activation)
                for _ in range(config.num_gnn_layers)
            ]
        )
        act = get_activation(config.activation)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            act(),
            nn.Linear(config.hidden_dim, 1),
        )

        if config.use_orthogonal_init:
            orthogonal_init(self)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        squeeze_batch = False
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            if edge_attr.dim() == 2:
                edge_attr = edge_attr.unsqueeze(0)
            squeeze_batch = True

        h = self.node_encoder(node_features)
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_attr)

        graph_embedding = h.mean(dim=1)  # [B, H], stable across different N.
        value = self.value_head(graph_embedding).squeeze(-1)
        if squeeze_batch:
            value = value.squeeze(0)
        return value


class GraphActorCritic(nn.Module):
    """Convenience wrapper holding graph actor and graph critic."""

    def __init__(self, config: GraphNetworkConfig):
        super().__init__()
        self.actor = GraphActor(config)
        self.critic = GraphCritic(config)

    def act(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor.act(node_features, edge_index, edge_attr, deterministic=deterministic)

    def value(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic(node_features, edge_index, edge_attr)

    def evaluate_actions(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, entropy = self.actor.evaluate_actions(node_features, edge_index, edge_attr, actions)
        values = self.critic(node_features, edge_index, edge_attr)
        return log_probs, entropy, values
