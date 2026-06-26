"""GNN-MAPPO trainer for DA-MAPPO with agent-count-generalizable policies."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gnn_actor_critic import GraphActorCritic, GraphNetworkConfig
from graph_rollout_buffer import GraphBufferConfig, GraphRolloutBuffer
from mappo import MAPPOConfig


class GraphMAPPOAgent:
    """MAPPO trainer using a graph actor and graph critic.

    This first version assumes a fixed number of agents inside one training run.
    Checkpoints can still be loaded for evaluation with a different number of
    agents because the neural network parameters depend on node_dim and edge_dim,
    not num_agents.
    """

    def __init__(self, env, config: Optional[MAPPOConfig] = None):
        self.env = env
        self.cfg = config or MAPPOConfig()
        self.device = self._resolve_device(self.cfg.device)

        self.env.reset()
        node_features, edge_index, edge_attr = self.env.get_graph_obs()

        self.num_agents = self.env.num_agents
        self.node_dim = int(node_features.shape[1])
        self.edge_dim = int(edge_attr.shape[1])
        self.action_dim = self.env.action_dim

        network_config = GraphNetworkConfig(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            action_dim=self.action_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_gnn_layers=self.cfg.num_hidden_layers,
            activation=self.cfg.activation,
            log_std_init=self.cfg.log_std_init,
        )
        self.model = GraphActorCritic(network_config).to(self.device)

        self.actor_optimizer = optim.Adam(
            self.model.actor.parameters(),
            lr=self.cfg.actor_lr,
            eps=self.cfg.eps,
        )
        self.critic_optimizer = optim.Adam(
            self.model.critic.parameters(),
            lr=self.cfg.critic_lr,
            eps=self.cfg.eps,
        )

        buffer_config = GraphBufferConfig(
            rollout_steps=self.cfg.rollout_steps,
            num_agents=self.num_agents,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            action_dim=self.action_dim,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            device=str(self.device),
        )
        self.buffer = GraphRolloutBuffer(buffer_config)

        self.total_env_steps = 0
        self.num_updates = 0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _graph_to_tensors(self, node_features, edge_index, edge_attr):
        node_features_t = torch.as_tensor(node_features, dtype=torch.float32, device=self.device)
        edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr_t = torch.as_tensor(edge_attr, dtype=torch.float32, device=self.device)
        return node_features_t, edge_index_t, edge_attr_t

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def collect_rollout(self) -> Dict[str, float]:
        self.buffer.reset()
        self.env.reset()

        episode_returns = []
        episode_lengths = []
        current_episode_return = np.zeros((self.num_agents,), dtype=np.float32)
        current_episode_length = 0
        last_transition_done = False

        for _ in range(self.cfg.rollout_steps):
            node_features, edge_index, edge_attr = self.env.get_graph_obs()
            node_features_t, edge_index_t, edge_attr_t = self._graph_to_tensors(
                node_features, edge_index, edge_attr
            )

            with torch.no_grad():
                actions_t, log_probs_t = self.model.act(
                    node_features_t,
                    edge_index_t,
                    edge_attr_t,
                    deterministic=False,
                )
                # Single-graph forward: batch = zeros(N)
                batch_t = torch.zeros(node_features_t.shape[0], dtype=torch.long, device=self.device)
                value_t = self.model.value(node_features_t, edge_index_t, edge_attr_t, batch_t)

            actions = actions_t.cpu().numpy().astype(np.float32)
            log_probs = log_probs_t.cpu().numpy().astype(np.float32)
            value = float(value_t.item())

            _, rewards, dones, _ = self.env.step(actions)
            last_transition_done = bool(np.all(dones))

            self.buffer.add(
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                actions=actions,
                log_probs=log_probs,
                rewards=rewards,
                dones=dones,
                value=value,
            )

            self.total_env_steps += self.num_agents
            current_episode_return += rewards
            current_episode_length += 1

            if last_transition_done:
                episode_returns.append(float(np.mean(current_episode_return)))
                episode_lengths.append(float(current_episode_length))
                self.env.reset()
                current_episode_return[:] = 0.0
                current_episode_length = 0

        if last_transition_done:
            last_value = 0.0
        else:
            node_features, edge_index, edge_attr = self.env.get_graph_obs()
            node_features_t, edge_index_t, edge_attr_t = self._graph_to_tensors(
                node_features, edge_index, edge_attr
            )
            batch_t = torch.zeros(node_features_t.shape[0], dtype=torch.long, device=self.device)
            with torch.no_grad():
                last_value = float(self.model.value(node_features_t, edge_index_t, edge_attr_t, batch_t).item())

        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=last_transition_done,
        )

        diagnostics = self.buffer.summary()
        diagnostics.update(
            {
                "episodes_finished": float(len(episode_returns)),
                "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
                "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                "total_env_steps": float(self.total_env_steps),
            }
        )
        return diagnostics

    # ------------------------------------------------------------------
    # PPO / MAPPO update
    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        actor_losses = []
        critic_losses = []
        entropy_values = []
        approx_kls = []
        clip_fractions = []
        total_losses = []

        # For graph batches, minibatch_size means number of graph time steps.
        graph_batch_size = min(self.cfg.minibatch_size, max(1, self.buffer.ptr))

        for _ in range(self.cfg.ppo_epochs):
            for batch in self.buffer.iter_minibatches(
                batch_size=graph_batch_size,
                shuffle=True,
                normalize_advantages=self.cfg.normalize_advantages,
            ):
                metrics = self._update_minibatch(batch)
                actor_losses.append(metrics["actor_loss"])
                critic_losses.append(metrics["critic_loss"])
                entropy_values.append(metrics["entropy"])
                approx_kls.append(metrics["approx_kl"])
                clip_fractions.append(metrics["clip_fraction"])
                total_losses.append(metrics["total_loss"])

        self.num_updates += 1
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropy_values)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "total_loss": float(np.mean(total_losses)),
            "num_updates": float(self.num_updates),
        }

    def _update_minibatch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        node_features = batch["node_features"]  # [N_total, H]
        edge_index = batch["edge_index"]         # [2, E_total]
        edge_attrs = batch["edge_attrs"]         # [E_total, F]
        batch_vec = batch["batch"]                # [N_total]
        actions = batch["actions"]                # [B, N, action_dim]
        old_log_probs = batch["old_log_probs"]    # [B, N]
        advantages = batch["advantages"]          # [B, N]
        returns = batch["returns"]                # [B, N]
        old_values = batch["old_values"]          # [B]

        B = old_log_probs.shape[0]

        # Evaluate: Actor takes flat actions [N_total, action_dim]
        new_log_probs_flat, entropy_flat, new_values = self.model.evaluate_actions(
            node_features,
            edge_index,
            edge_attrs,
            actions.view(-1, self.action_dim),
            batch_vec,
        )
        new_log_probs = new_log_probs_flat.view(B, -1)  # [B, N]
        entropy = entropy_flat.view(B, -1)               # [B, N]

        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        unclipped_policy_loss = -advantages * ratio
        clipped_policy_loss = -advantages * torch.clamp(
            ratio,
            1.0 - self.cfg.clip_coef,
            1.0 + self.cfg.clip_coef,
        )
        actor_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()

        # One graph-level value is trained against all per-agent returns.
        value_pred = new_values.unsqueeze(-1).expand_as(returns)
        old_value_pred = old_values.unsqueeze(-1).expand_as(returns)
        if self.cfg.use_value_clipping:
            value_pred_clipped = old_value_pred + torch.clamp(
                value_pred - old_value_pred,
                -self.cfg.value_clip_coef,
                self.cfg.value_clip_coef,
            )
            value_losses = (value_pred - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            critic_loss = 0.5 * (value_pred - returns).pow(2).mean()

        entropy_loss = entropy.mean()
        total_loss = (
            actor_loss
            + self.cfg.value_loss_coef * critic_loss
            - self.cfg.entropy_coef * entropy_loss
        )

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (torch.abs(ratio - 1.0) > self.cfg.clip_coef).float().mean()

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "entropy": float(entropy_loss.detach().cpu().item()),
            "approx_kl": float(approx_kl.detach().cpu().item()),
            "clip_fraction": float(clip_fraction.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def train_one_update(self) -> Dict[str, float]:
        rollout_metrics = self.collect_rollout()
        update_metrics = self.update()
        return {**rollout_metrics, **update_metrics}

    def act(self, obs: Optional[np.ndarray] = None, deterministic: bool = True) -> np.ndarray:
        """Get actions for evaluation.

        The obs argument is ignored and kept only for compatibility with the old
        evaluation loop. The graph policy always reads env.get_graph_obs().
        """
        node_features, edge_index, edge_attr = self.env.get_graph_obs()
        node_features_t, edge_index_t, edge_attr_t = self._graph_to_tensors(
            node_features, edge_index, edge_attr
        )
        with torch.no_grad():
            actions_t, _ = self.model.act(
                node_features_t,
                edge_index_t,
                edge_attr_t,
                deterministic=deterministic,
            )
        return actions_t.cpu().numpy().astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "config": self.cfg,
                "model_type": "gnn",
                "total_env_steps": self.total_env_steps,
                "num_updates": self.num_updates,
                "node_dim": self.node_dim,
                "edge_dim": self.edge_dim,
                "action_dim": self.action_dim,
                "train_num_agents": self.num_agents,
            },
            path,
        )

    def load(
        self,
        path: str | Path,
        map_location: Optional[str] = None,
        load_optimizer: bool = True,
    ) -> None:
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if load_optimizer:
            if "actor_optimizer" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_env_steps = int(checkpoint.get("total_env_steps", 0))
        self.num_updates = int(checkpoint.get("num_updates", 0))
