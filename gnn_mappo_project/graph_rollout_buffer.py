"""Graph rollout buffer for fixed-N GNN-MAPPO training.

This first GNN version still trains with a fixed number of agents per run, but the
model parameters do not depend on num_agents. Therefore a checkpoint trained at
N=5 can be loaded and executed at N=3 or N=10.
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
    num_edges: int
    action_dim: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cpu"


class GraphRolloutBuffer:
    """Fixed-size rollout buffer storing complete graphs per time step."""

    def __init__(self, config: GraphBufferConfig, edge_index: np.ndarray):
        self.cfg = config
        self.rollout_steps = config.rollout_steps
        self.num_agents = config.num_agents
        self.node_dim = config.node_dim
        self.edge_dim = config.edge_dim
        self.num_edges = config.num_edges
        self.action_dim = config.action_dim
        self.device = torch.device(config.device)

        edge_index = np.asarray(edge_index, dtype=np.int64)
        if edge_index.shape != (2, self.num_edges):
            raise ValueError(f"edge_index shape {edge_index.shape}, expected {(2, self.num_edges)}")
        self.edge_index_np = edge_index
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)

        self.node_features = np.zeros((self.rollout_steps, self.num_agents, self.node_dim), dtype=np.float32)
        self.edge_attrs = np.zeros((self.rollout_steps, self.num_edges, self.edge_dim), dtype=np.float32)
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
        node_features: np.ndarray,
        edge_attr: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        value: float,
    ) -> None:
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("GraphRolloutBuffer is full. Call reset() before adding more data.")

        node_features = np.asarray(node_features, dtype=np.float32)
        edge_attr = np.asarray(edge_attr, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        log_probs = np.asarray(log_probs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        if node_features.shape != (self.num_agents, self.node_dim):
            raise ValueError(f"node_features shape {node_features.shape}, expected {(self.num_agents, self.node_dim)}")
        if edge_attr.shape != (self.num_edges, self.edge_dim):
            raise ValueError(f"edge_attr shape {edge_attr.shape}, expected {(self.num_edges, self.edge_dim)}")
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

        self.node_features[self.ptr] = node_features
        self.edge_attrs[self.ptr] = edge_attr
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
        """Yield graph-time minibatches.

        batch_size is the number of complete graph time steps, not the number of
        flattened agent samples.
        """
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
                "node_features": torch.as_tensor(self.node_features[idx], dtype=torch.float32, device=self.device),
                "edge_attrs": torch.as_tensor(self.edge_attrs[idx], dtype=torch.float32, device=self.device),
                "edge_index": self.edge_index,
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
