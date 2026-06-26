"""Graph rollout buffer for fixed-N strict CTDE GNN-MAPPO training.

Actor graph:
- One padded ego graph per agent.
- actor_ego_node_features: [T, N, M, node_dim]
- actor_ego_edge_attrs:    [T, N, E_ego, edge_dim]

Critic graph:
- One full/global UAV graph per timestep.
- critic_node_features: [T, N, node_dim]
- critic_edge_attrs:    [T, E_global, edge_dim]

The shared actor network is applied to all N ego graphs. The critic is applied to
one global graph per time step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator

import numpy as np
import torch


@dataclass
class GraphBufferConfig:
    rollout_steps: int
    num_agents: int
    node_dim: int
    edge_dim: int
    actor_ego_num_nodes: int
    actor_ego_num_edges: int
    critic_num_edges: int
    action_dim: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cpu"


class GraphRolloutBuffer:
    """Fixed-size rollout buffer storing actor ego graphs and critic full graphs."""

    def __init__(
        self,
        config: GraphBufferConfig,
        actor_ego_edge_index: np.ndarray,
        critic_edge_index: np.ndarray,
    ):
        self.cfg = config
        self.rollout_steps = config.rollout_steps
        self.num_agents = config.num_agents
        self.node_dim = config.node_dim
        self.edge_dim = config.edge_dim
        self.actor_ego_num_nodes = config.actor_ego_num_nodes
        self.actor_ego_num_edges = config.actor_ego_num_edges
        self.critic_num_edges = config.critic_num_edges
        self.action_dim = config.action_dim
        self.device = torch.device(config.device)

        actor_ego_edge_index = np.asarray(actor_ego_edge_index, dtype=np.int64)
        critic_edge_index = np.asarray(critic_edge_index, dtype=np.int64)
        if actor_ego_edge_index.shape != (2, self.actor_ego_num_edges):
            raise ValueError(
                f"actor_ego_edge_index shape {actor_ego_edge_index.shape}, "
                f"expected {(2, self.actor_ego_num_edges)}"
            )
        if critic_edge_index.shape != (2, self.critic_num_edges):
            raise ValueError(
                f"critic_edge_index shape {critic_edge_index.shape}, expected {(2, self.critic_num_edges)}"
            )

        self.actor_ego_edge_index_np = actor_ego_edge_index
        self.critic_edge_index_np = critic_edge_index
        self.actor_ego_edge_index = torch.as_tensor(actor_ego_edge_index, dtype=torch.long, device=self.device)
        self.critic_edge_index = torch.as_tensor(critic_edge_index, dtype=torch.long, device=self.device)

        self.actor_ego_node_features = np.zeros(
            (self.rollout_steps, self.num_agents, self.actor_ego_num_nodes, self.node_dim),
            dtype=np.float32,
        )
        self.actor_ego_edge_attrs = np.zeros(
            (self.rollout_steps, self.num_agents, self.actor_ego_num_edges, self.edge_dim),
            dtype=np.float32,
        )
        self.critic_node_features = np.zeros((self.rollout_steps, self.num_agents, self.node_dim), dtype=np.float32)
        self.critic_edge_attrs = np.zeros((self.rollout_steps, self.critic_num_edges, self.edge_dim), dtype=np.float32)
        self.actions = np.zeros((self.rollout_steps, self.num_agents, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        self.dones = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        self.values = np.zeros((self.rollout_steps,), dtype=np.float32)

        self.advantages = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        self.returns = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

    def add(
        self,
        actor_ego_node_features: np.ndarray,
        actor_ego_edge_attr: np.ndarray,
        critic_node_features: np.ndarray,
        critic_edge_attr: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        value: float,
    ) -> None:
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("GraphRolloutBuffer is full. Call reset() before adding more data.")

        actor_ego_node_features = np.asarray(actor_ego_node_features, dtype=np.float32)
        actor_ego_edge_attr = np.asarray(actor_ego_edge_attr, dtype=np.float32)
        critic_node_features = np.asarray(critic_node_features, dtype=np.float32)
        critic_edge_attr = np.asarray(critic_edge_attr, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        log_probs = np.asarray(log_probs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        expected_actor_nodes = (self.num_agents, self.actor_ego_num_nodes, self.node_dim)
        expected_actor_edges = (self.num_agents, self.actor_ego_num_edges, self.edge_dim)
        if actor_ego_node_features.shape != expected_actor_nodes:
            raise ValueError(f"actor_ego_node_features shape {actor_ego_node_features.shape}, expected {expected_actor_nodes}")
        if actor_ego_edge_attr.shape != expected_actor_edges:
            raise ValueError(f"actor_ego_edge_attr shape {actor_ego_edge_attr.shape}, expected {expected_actor_edges}")
        if critic_node_features.shape != (self.num_agents, self.node_dim):
            raise ValueError(f"critic_node_features shape {critic_node_features.shape}, expected {(self.num_agents, self.node_dim)}")
        if critic_edge_attr.shape != (self.critic_num_edges, self.edge_dim):
            raise ValueError(f"critic_edge_attr shape {critic_edge_attr.shape}, expected {(self.critic_num_edges, self.edge_dim)}")
        if actions.shape != (self.num_agents, self.action_dim):
            raise ValueError(f"actions shape {actions.shape}, expected {(self.num_agents, self.action_dim)}")
        if log_probs.shape != (self.num_agents,):
            raise ValueError(f"log_probs shape {log_probs.shape}, expected {(self.num_agents,)}")
        if rewards.shape != (self.num_agents,):
            raise ValueError(f"rewards shape {rewards.shape}, expected {(self.num_agents,)}")
        if dones.shape == ():
            dones = np.full((self.num_agents,), float(dones), dtype=np.float32)
        if dones.shape != (self.num_agents,):
            raise ValueError(f"dones shape {dones.shape}, expected {(self.num_agents,)}")

        self.actor_ego_node_features[self.ptr] = actor_ego_node_features
        self.actor_ego_edge_attrs[self.ptr] = actor_ego_edge_attr
        self.critic_node_features[self.ptr] = critic_node_features
        self.critic_edge_attrs[self.ptr] = critic_edge_attr
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = float(value)

        self.ptr += 1
        self.full = self.ptr == self.rollout_steps

    def compute_returns_and_advantages(self, last_value: float, last_done: bool | np.ndarray) -> None:
        valid_steps = self.ptr
        if valid_steps == 0:
            raise RuntimeError("Cannot compute returns on an empty buffer.")

        if isinstance(last_done, np.ndarray):
            last_done_float = float(np.all(last_done))
        else:
            last_done_float = float(last_done)

        gae = np.zeros((self.num_agents,), dtype=np.float32)

        for step in reversed(range(valid_steps)):
            if step == valid_steps - 1:
                next_value = float(last_value)
                next_nonterminal = 1.0 - last_done_float
            else:
                next_value = self.values[step + 1]
                next_nonterminal = 1.0 - float(np.all(self.dones[step]))

            delta = self.rewards[step] + self.cfg.gamma * next_value * next_nonterminal - self.values[step]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]

    def iter_minibatches(
        self,
        batch_size: int,
        shuffle: bool = True,
        normalize_advantages: bool = True,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        valid_steps = self.ptr
        if valid_steps == 0:
            raise RuntimeError("Cannot iterate minibatches from an empty buffer.")

        indices = np.arange(valid_steps)
        if shuffle:
            np.random.shuffle(indices)

        advantages = self.advantages[:valid_steps].copy()
        if normalize_advantages:
            mean = advantages.mean()
            std = advantages.std() + 1e-8
            advantages = (advantages - mean) / std

        for start in range(0, valid_steps, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "actor_ego_node_features": torch.as_tensor(self.actor_ego_node_features[idx], dtype=torch.float32, device=self.device),
                "actor_ego_edge_attrs": torch.as_tensor(self.actor_ego_edge_attrs[idx], dtype=torch.float32, device=self.device),
                "actor_ego_edge_index": self.actor_ego_edge_index,
                "critic_node_features": torch.as_tensor(self.critic_node_features[idx], dtype=torch.float32, device=self.device),
                "critic_edge_attrs": torch.as_tensor(self.critic_edge_attrs[idx], dtype=torch.float32, device=self.device),
                "critic_edge_index": self.critic_edge_index,
                "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=self.device),
                "old_log_probs": torch.as_tensor(self.log_probs[idx], dtype=torch.float32, device=self.device),
                "advantages": torch.as_tensor(advantages[idx], dtype=torch.float32, device=self.device),
                "returns": torch.as_tensor(self.returns[idx], dtype=torch.float32, device=self.device),
                "old_values": torch.as_tensor(self.values[idx], dtype=torch.float32, device=self.device),
            }

    def summary(self) -> Dict[str, float]:
        valid_steps = self.ptr
        if valid_steps == 0:
            return {}
        return {
            "mean_reward": float(np.mean(self.rewards[:valid_steps])),
            "std_reward": float(np.std(self.rewards[:valid_steps])),
            "mean_value": float(np.mean(self.values[:valid_steps])),
            "std_value": float(np.std(self.values[:valid_steps])),
            "mean_advantage": float(np.mean(self.advantages[:valid_steps])),
            "std_advantage": float(np.std(self.advantages[:valid_steps])),
            "mean_return": float(np.mean(self.returns[:valid_steps])),
            "std_return": float(np.std(self.returns[:valid_steps])),
            "buffer_steps": float(valid_steps),
        }
