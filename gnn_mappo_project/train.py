"""Training entry point for DA-MAPPO and GNN-DA-MAPPO.

Examples:
    # Original MLP MAPPO
    python train.py --total-updates 20 --rollout-steps 128 --num-agents 5

    # GNN MAPPO, fixed N training but cross-N checkpoint execution
    python train.py --use-gnn --total-updates 20 --rollout-steps 128 --num-agents 5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[no-redef]
        """Fallback no-op writer when tensorboard is not installed."""
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from env import MultiUAV2DEnv, UAVEnvConfig
from mappo import MAPPOAgent, MAPPOConfig
from gnn_mappo import GraphMAPPOAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DA-MAPPO / GNN-DA-MAPPO on the multi-UAV environment.")

    # General.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-name", type=str, default="stage1_gnn")
    parser.add_argument("--save-dir", type=str, default="runs")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Optional checkpoint for continuing training.")
    # parser.add_argument("--resume-checkpoint", type=str, default="runs/stage1_gnn_gnn_N5_O5_hungarian_S42/checkpoints/update_150.pt", help="Optional checkpoint for continuing training.")
    parser.add_argument("--use-gnn", action="store_true", default=True, help="Use GNN MAPPO agent")
    parser.add_argument("--torch-num-threads", type=int, default=1,
                        help="Set torch CPU threads. Default 1 avoids CPU oversubscription on small GNN batches.")

    # Environment.
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--num-obstacles", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--assigner-name", type=str, default="hungarian", choices=["hungarian", "greedy", "fixed", "cross"])
    parser.add_argument("--layout-mode", type=str, default="same_side", choices=["same_side", "cross"])
    parser.add_argument("--lidar-num-rays", type=int, default=35)
    parser.add_argument("--lidar-range", type=float, default=5.0)
    parser.add_argument("--freeze-arrived-uavs", action="store_true", help="Freeze UAV commands after arrival bonus is collected.")

    # Optional dynamic-target settings, kept disabled by default.
    parser.add_argument("--dynamic-targets", action="store_true")
    parser.add_argument("--target-motion-mode", type=str, default="none", choices=["none", "swap", "linear", "linear_swap"])
    parser.add_argument("--target-swap-interval", type=int, default=100)
    parser.add_argument("--target-swap-start-step", type=int, default=100)
    parser.add_argument("--target-speed", type=float, default=0.2)

    # Training.
    parser.add_argument("--total-updates", type=int, default=5000)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512,
                        help="MLP: flattened samples; GNN: graph time steps.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Network.
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=3,
                        help="MLP hidden layers, or GNN message-passing layers when --use-gnn is set.")
    parser.add_argument("--activation", type=str, default="relu", choices=["tanh", "relu", "gelu", "elu"])
    parser.add_argument("--log-std-init", type=float, default=-0.5)

    # Logging / saving / evaluation.
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--deterministic-eval", action="store_true", default=True)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(args: argparse.Namespace, seed_offset: int = 0, num_agents: int | None = None) -> MultiUAV2DEnv:
    cfg = UAVEnvConfig(
        num_agents=args.num_agents if num_agents is None else num_agents,
        num_obstacles=args.num_obstacles,
        max_steps=args.max_steps,
        assigner_name=args.assigner_name,
        layout_mode=args.layout_mode,
        lidar_num_rays=args.lidar_num_rays,
        lidar_range=args.lidar_range,
        freeze_arrived_uavs=args.freeze_arrived_uavs,
        dynamic_targets=args.dynamic_targets,
        target_motion_mode=args.target_motion_mode,
        target_swap_interval=args.target_swap_interval,
        target_swap_start_step=args.target_swap_start_step,
        target_speed=args.target_speed,
        seed=args.seed + seed_offset,
    )
    return MultiUAV2DEnv(cfg)


def make_agent(env: MultiUAV2DEnv, args: argparse.Namespace):
    cfg = MAPPOConfig(
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_coef=args.clip_coef,
        value_clip_coef=args.value_clip_coef,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        activation=args.activation,
        log_std_init=args.log_std_init,
        device=args.device,
    )
    if args.use_gnn:
        return GraphMAPPOAgent(env, cfg)
    return MAPPOAgent(env, cfg)


def evaluate_policy(agent, env: MultiUAV2DEnv, num_episodes: int = 5, deterministic: bool = True) -> Dict[str, float]:
    episode_returns = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    timeout_count = 0

    for ep in range(num_episodes):
        obs = env.reset(seed=env.cfg.seed + 10_000 + ep if env.cfg.seed is not None else None)
        done = False
        ep_return = np.zeros(env.num_agents, dtype=np.float32)
        ep_len = 0
        info = {}

        while not done:
            actions = agent.act(obs, deterministic=deterministic)
            obs, rewards, dones, info = env.step(actions)
            ep_return += rewards
            ep_len += 1
            done = bool(np.all(dones))

        reason = info.get("termination_reason", "")
        if "success" in reason:
            success_count += 1
        if "collision" in reason or "boundary_violation" in reason:
            collision_count += 1
        if "timeout" in reason:
            timeout_count += 1

        episode_returns.append(float(np.mean(ep_return)))
        episode_lengths.append(float(ep_len))

    return {
        "eval_mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "eval_std_return": float(np.std(episode_returns)) if episode_returns else 0.0,
        "eval_mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "eval_success_rate": float(success_count / max(num_episodes, 1)),
        "eval_collision_rate": float(collision_count / max(num_episodes, 1)),
        "eval_timeout_rate": float(timeout_count / max(num_episodes, 1)),
    }


def write_csv_row(csv_path: Path, row: Dict[str, float]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = [
        "update",
        "total_env_steps",
        "mean_reward",
        "mean_episode_return",
        "episodes_finished",
        "actor_loss",
        "critic_loss",
        "entropy",
        "approx_kl",
        "eval_mean_return",
        "eval_success_rate",
    ]
    parts = []
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if key == "update":
                parts.append(f"{key}={int(value)}")
            else:
                parts.append(f"{key}={value:.4f}")
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    set_seed(args.seed)

    algo = "gnn" if args.use_gnn else "mlp"
    tag = f"{args.run_name}_{algo}_N{args.num_agents}_O{args.num_obstacles}_{args.assigner_name}_S{args.seed}"
    run_dir = Path(args.save_dir) / tag
    checkpoint_dir = run_dir / "checkpoints"
    csv_path = run_dir / "metrics.csv"
    log_dir = run_dir / "tensorboard"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to: {config_path}")

    writer = SummaryWriter(log_dir)

    env = make_env(args, seed_offset=0)
    eval_env = make_env(args, seed_offset=999)
    agent = make_agent(env, args)

    if args.resume_checkpoint:
        print(f"Loading checkpoint from: {args.resume_checkpoint}")
        agent.load(args.resume_checkpoint)
        print(
            f"Resume training from update={agent.num_updates}, "
            f"env_steps={agent.total_env_steps}"
        )

    print("DA-MAPPO training started")
    print(f"algorithm: {'GNN-MAPPO' if args.use_gnn else 'MLP-MAPPO'}")
    print(f"run_dir: {run_dir}")
    print(f"device: {agent.device}")
    print(f"num_agents: {agent.num_agents}")

    if args.use_gnn:
        print(f"node_dim: {agent.node_dim}")
        print(f"edge_dim: {agent.edge_dim}")
        print(f"num_edges: {agent.num_edges}")
    else:
        print(f"obs_dim: {agent.obs_dim}")
        print(f"state_dim: {agent.state_dim}")

    print(f"action_dim: {agent.action_dim}")
    print(f"rollout_steps: {args.rollout_steps}")
    print(f"minibatch_size: {args.minibatch_size}")
    print(f"num_obstacles: {args.num_obstacles}")
    print("-" * 80)

    start_time = time.time()

    # best.pt 保存依据：
    # 当前训练 rollout 中 mean_episode_return 最大。
    best_mean_episode_return = -float("inf")
    best_update = 0

    for update in range(1, args.total_updates + 1):
        train_metrics = agent.train_one_update()

        metrics: Dict[str, float] = {
            "update": float(update),
            "time_sec": float(time.time() - start_time),
            **train_metrics,
        }

        # 默认本轮没有保存 best。
        metrics["saved_best"] = 0.0

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, update)

        # ------------------------------------------------------------
        # 按训练 rollout 的 mean_episode_return 保存 best.pt
        # ------------------------------------------------------------
        current_mean_episode_return = float(
            metrics.get("mean_episode_return", -float("inf"))
        )
        episodes_finished = float(metrics.get("episodes_finished", 0.0))

        # 只有当前 rollout 里真的结束了 episode，才参与 best 判断。
        # 如果你想让 episodes_finished=0 时也参与比较，可以删掉 episodes_finished > 0 这个条件。
        if (
            episodes_finished > 0
            and current_mean_episode_return > best_mean_episode_return
        ):
            best_mean_episode_return = current_mean_episode_return
            best_update = update
            agent.save(checkpoint_dir / "best.pt")
            metrics["saved_best"] = 1.0
            metrics["best_mean_episode_return"] = best_mean_episode_return

            print(
                "[best] saved best.pt | "
                f"update={update}, "
                f"mean_episode_return={best_mean_episode_return:.4f}, "
                f"episodes_finished={episodes_finished:.0f}"
            )
        else:
            metrics["best_mean_episode_return"] = best_mean_episode_return

        writer.add_scalar(
            "train/best_mean_episode_return",
            best_mean_episode_return,
            update,
        )

        # ------------------------------------------------------------
        # Evaluation 仍然保留，但不再负责保存 best.pt
        # ------------------------------------------------------------
        should_eval = args.eval_interval > 0 and update % args.eval_interval == 0
        if should_eval:
            eval_metrics = evaluate_policy(
                agent=agent,
                env=eval_env,
                num_episodes=args.eval_episodes,
                deterministic=args.deterministic_eval,
            )
            metrics.update(eval_metrics)

            for key, value in eval_metrics.items():
                writer.add_scalar(f"eval/{key}", value, update)

        # ------------------------------------------------------------
        # 定期保存普通 checkpoint
        # ------------------------------------------------------------
        should_save = args.save_interval > 0 and update % args.save_interval == 0
        if should_save:
            agent.save(checkpoint_dir / f"update_{update}.pt")

        write_csv_row(csv_path, metrics)

        if args.log_interval > 0 and update % args.log_interval == 0:
            print(format_metrics(metrics))

    agent.save(checkpoint_dir / "final.pt")
    writer.close()

    print("-" * 80)
    print("Training finished")
    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"Best update: {best_update}")
    print(f"Best mean_episode_return: {best_mean_episode_return:.4f}")
    print(f"Final checkpoint: {checkpoint_dir / 'final.pt'}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print(f"Metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
