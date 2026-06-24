"""
Training entry point for DA-MAPPO.

Required project files:
    target_assignment.py
    env.py
    actor_critic.py
    rollout_buffer.py
    mappo.py

Quick test:
    python train.py --total-updates 20 --rollout-steps 128 --num-obstacles 5

Full run:
    python train.py --total-updates 1000 --rollout-steps 512 --num-obstacles 10
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env import MultiUAV2DEnv, UAVEnvConfig
from mappo import MAPPOAgent, MAPPOConfig
from gnn_mappo import GraphMAPPOAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DA-MAPPO on the multi-UAV target assignment environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General.
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Torch device: auto selects CUDA if available, else CPU")
    parser.add_argument("--run-name", type=str, default="mappo",
                        help="Short identifier for this run (used in output directory name)")
    parser.add_argument("--save-dir", type=str, default="runs",
                        help="Parent directory for all run outputs")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to checkpoint.pt for resuming training")

    # Environment.
    parser.add_argument("--num-agents", type=int, default=3,
                        help="Number of UAVs (target count always equals agent count)")
    parser.add_argument("--num-obstacles", type=int, default=20,
                        help="Number of circular obstacles placed in the arena")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="Max environment steps per episode before timeout")
    parser.add_argument("--assigner-name", type=str, default="fixed",
                        choices=["hungarian", "greedy", "fixed", "cross", "cbba"],
                        help="Online target assignment algorithm")
    parser.add_argument("--reassign-interval", type=int, default=10,
                        help="Steps between full reassignments (1 = every step)")
    parser.add_argument("--lidar-num-rays", type=int, default=35,
                        help="Number of LiDAR rays per UAV for obstacle sensing")
    parser.add_argument("--lidar-range", type=float, default=5.0,
                        help="Maximum LiDAR detection range (meters)")
    parser.add_argument("--layout-mode", type=str, default="same_side",
                        choices=["same_side", "cross"],
                        help="Initial layout of agent and target spawn regions")

    # Dynamic targets.
    parser.add_argument("--dynamic-targets", action="store_true", default=False,
                        help="Enable dynamic target motion")
    parser.add_argument("--target-motion-mode", type=str, default="none",
                        choices=["none", "swap", "linear", "linear_swap", "racetrack"],
                        help="Target motion mode")
    parser.add_argument("--target-speed", type=float, default=0.2,
                        help="Target movement speed for linear motion (m/s)")
    parser.add_argument("--racetrack-turn-radius", type=float, default=0.5,
                        help="Racetrack U-turn arc radius")
    parser.add_argument("--racetrack-straight-half-length", type=float, default=9.0,
                        help="Racetrack straight segment half-length")
    parser.add_argument("--racetrack-front-speed", type=float, default=0.2,
                        help="Racetrack speed on front (left) side (m/s)")
    parser.add_argument("--racetrack-back-speed", type=float, default=2.0,
                        help="Racetrack speed on back (right) side (m/s)")

    # Training.
    parser.add_argument("--total-updates", type=int, default=1000,
                        help="Number of PPO update iterations to run")
    parser.add_argument("--rollout-steps", type=int, default=512,
                        help="Environment steps collected per rollout before each update")
    parser.add_argument("--ppo-epochs", type=int, default=10,
                        help="Number of PPO epochs per update (K in the paper)")
    parser.add_argument("--minibatch-size", type=int, default=64,
                        help="Minibatch size for PPO gradient steps")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for cumulative return")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE trace-decay parameter for advantage estimation")
    parser.add_argument("--actor-lr", type=float, default=3e-5,
                        help="Learning rate for the actor network")
    parser.add_argument("--critic-lr", type=float, default=3e-4,
                        help="Learning rate for the critic network")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="PPO surrogate objective clipping epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient for exploration")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="Weight of the value loss in the combined PPO loss")
    parser.add_argument("--target-kl", type=float, default=0.02,
                        help="Early-stop PPO epoch if mean KL exceeds this (0 = disabled)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum gradient norm for gradient clipping")

    # Network.
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden layer dimension in actor/critic MLPs (or GNN node embedding dim)")
    parser.add_argument("--num-hidden-layers", type=int, default=2,
                        help="Number of hidden layers (or GNN message-passing rounds)")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["tanh", "relu", "gelu", "elu"],
                        help="Activation function for hidden layers")

    # GNN.
    parser.add_argument("--use-gnn", action="store_true", default=False,
                        help="Use GNN MAPPO agent instead of MLP (agent-count generalizable)")
    parser.add_argument("--torch-num-threads", type=int, default=1,
                        help="Torch CPU thread count (1 avoids oversubscription on small GNN batches)")

    # Logging / saving / evaluation.
    parser.add_argument("--log-interval", type=int, default=1,
                        help="Log training metrics to CSV and TensorBoard every N updates")
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Run deterministic evaluation every N updates (0 = skip)")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save periodic checkpoints every N updates")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of episodes per evaluation run")
    parser.add_argument("--deterministic-eval", action="store_true", default=True,
                        help="Use deterministic action selection during evaluation")



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
        reassign_interval=args.reassign_interval,
        lidar_num_rays=args.lidar_num_rays,
        lidar_range=args.lidar_range,
        layout_mode=args.layout_mode,
        dynamic_targets=args.dynamic_targets,
        target_motion_mode=args.target_motion_mode,
        target_speed=args.target_speed,
        racetrack_turn_radius=args.racetrack_turn_radius,
        racetrack_straight_half_length=args.racetrack_straight_half_length,
        racetrack_front_speed=args.racetrack_front_speed,
        racetrack_back_speed=args.racetrack_back_speed,
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
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        activation=args.activation,
        device=args.device,
    )
    if args.use_gnn:
        return GraphMAPPOAgent(env, cfg)
    return MAPPOAgent(env, cfg)


def evaluate_policy(
    agent: MAPPOAgent,
    env: MultiUAV2DEnv,
    num_episodes: int = 5,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Run evaluation episodes without training."""
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
    """Append one metrics row to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format selected metrics for console output."""
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
    set_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    policy = "GNN" if args.use_gnn else "MLP"
    tag = f"{timestamp}_{policy}_N{args.num_agents}_O{args.num_obstacles}_{args.assigner_name}_S{args.seed}_{args.run_name}"
    run_dir = Path(args.save_dir) / tag
    checkpoint_dir = run_dir / "checkpoints"
    csv_path = run_dir / "metrics.csv"
    log_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    config = vars(args)
    config["cli_command"] = "python " + " ".join(sys.argv)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")

    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    writer = SummaryWriter(log_dir)

    env = make_env(args, seed_offset=0)
    eval_env = make_env(args, seed_offset=999)
    agent = make_agent(env, args)
    if args.resume_checkpoint:
        print(f"Loading checkpoint from: {args.resume_checkpoint}")
        agent.load(args.resume_checkpoint)
        print(f"Resume training from update={agent.num_updates}, env_steps={agent.total_env_steps}")

    algo = "GNN-MAPPO" if args.use_gnn else "MLP-MAPPO"
    print(f"DA-MAPPO training started [{algo}]")
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
    print(f"num_obstacles: {args.num_obstacles}")
    print("-" * 80)

    start_time = time.time()
    best_eval_return = -float("inf")

    for update in range(1, args.total_updates + 1):
        train_metrics = agent.train_one_update()

        metrics: Dict[str, float] = {
            "update": float(update),
            "time_sec": float(time.time() - start_time),
            **train_metrics,
        }

        # TensorBoard: log every update
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, update)

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

            if eval_metrics["eval_mean_return"] > best_eval_return:
                best_eval_return = eval_metrics["eval_mean_return"]
                agent.save(checkpoint_dir / "best.pt")
                metrics["saved_best"] = 1.0
            else:
                metrics["saved_best"] = 0.0

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
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print(f"Final checkpoint: {checkpoint_dir / 'final.pt'}")
    print(f"Metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
