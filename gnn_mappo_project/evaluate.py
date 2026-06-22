"""Evaluation and visualization for DA-MAPPO / GNN-DA-MAPPO.

Examples:
    # Evaluate with the training N
    python evaluate.py runs/stage1_gnn_gnn_N5_O5_hungarian_S42 --episodes 10

    # Cross-N GNN evaluation: load N=5 checkpoint and execute in N=10 env
    python evaluate.py runs/stage1_gnn_gnn_N5_O20_hungarian_S42 --episodes 10 --checkpoint update_1500.pt --num-obstacles 50 --eval-num-agents 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from env import MultiUAV2DEnv, UAVEnvConfig
from mappo import MAPPOAgent, MAPPOConfig
from gnn_mappo import GraphMAPPOAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DA-MAPPO / GNN-DA-MAPPO policy.")
    parser.add_argument("run_dir", type=str, help="Path to the training run directory")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0, help="Override eval seed; 0 means use training seed")
    parser.add_argument("--output-dir", type=str, default="eval_outputs")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--num-obstacles", type=int, default=None, help="Override num_obstacles from training config")
    parser.add_argument("--eval-num-agents", type=int, default=None,
                        help="Override num_agents for evaluation. Intended for GNN cross-N tests.")
    parser.add_argument("--use-gnn", action="store_true",
                        help="Force GNN agent. If omitted, config.json use_gnn is used.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-video", action="store_true", help="Save MP4 video for the best-return episode")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--torch-num-threads", type=int, default=1,
                        help="Set torch CPU threads. Default 1 avoids CPU oversubscription on small GNN batches.")
    return parser.parse_args()


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with open(config_path) as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env_from_config(
    train_args: dict,
    seed: int,
    eval_num_agents: int | None = None,
    num_obstacles: int | None = None,
) -> MultiUAV2DEnv:
    cfg = UAVEnvConfig(
        num_agents=eval_num_agents if eval_num_agents is not None else train_args["num_agents"],
        num_obstacles=num_obstacles if num_obstacles is not None else train_args["num_obstacles"],
        max_steps=train_args["max_steps"],
        assigner_name=train_args["assigner_name"],
        layout_mode=train_args.get("layout_mode", "same_side"),
        lidar_num_rays=train_args["lidar_num_rays"],
        lidar_range=train_args["lidar_range"],
        freeze_arrived_uavs=train_args.get("freeze_arrived_uavs", False),
        dynamic_targets=train_args.get("dynamic_targets", False),
        target_motion_mode=train_args.get("target_motion_mode", "none"),
        target_swap_interval=train_args.get("target_swap_interval", 100),
        target_swap_start_step=train_args.get("target_swap_start_step", 100),
        target_speed=train_args.get("target_speed", 0.2),
        seed=seed,
    )
    return MultiUAV2DEnv(cfg)


def make_mappo_config(train_args: dict, device: str) -> MAPPOConfig:
    return MAPPOConfig(
        rollout_steps=train_args["rollout_steps"],
        gamma=train_args.get("gamma", 0.99),
        gae_lambda=train_args.get("gae_lambda", 0.95),
        ppo_epochs=train_args["ppo_epochs"],
        minibatch_size=train_args["minibatch_size"],
        clip_coef=train_args.get("clip_coef", 0.2),
        value_clip_coef=train_args.get("value_clip_coef", 0.2),
        entropy_coef=train_args.get("entropy_coef", 0.01),
        value_loss_coef=train_args.get("value_loss_coef", 0.5),
        max_grad_norm=train_args.get("max_grad_norm", 0.5),
        actor_lr=train_args.get("actor_lr", 3e-4),
        critic_lr=train_args.get("critic_lr", 3e-4),
        hidden_dim=train_args["hidden_dim"],
        num_hidden_layers=train_args["num_hidden_layers"],
        activation=train_args["activation"],
        log_std_init=train_args.get("log_std_init", -0.5),
        device=device,
    )


def load_agent_from_run(
    env: MultiUAV2DEnv,
    run_dir: Path,
    train_args: dict,
    device: str,
    checkpoint: str = "best.pt",
    use_gnn: bool = False,
):
    cfg = make_mappo_config(train_args, device)
    agent = GraphMAPPOAgent(env, cfg) if use_gnn else MAPPOAgent(env, cfg)

    checkpoint_path = run_dir / "checkpoints" / checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if use_gnn:
        agent.load(checkpoint_path, load_optimizer=False)
    else:
        agent.load(checkpoint_path)
    agent.model.eval()
    return agent


def run_episode(agent, env: MultiUAV2DEnv, seed: int | None, deterministic: bool = True) -> Tuple[Dict[str, float], Dict]:
    obs = env.reset(seed=seed)
    done = False

    positions_history: List[np.ndarray] = [env.positions.copy()]
    assignments_history: List[np.ndarray] = [env.assignments.copy()]
    arrived_history: List[np.ndarray] = [env.arrived.copy()]
    reward_history: List[np.ndarray] = []

    episode_return = np.zeros(env.num_agents, dtype=np.float32)
    episode_length = 0
    info: Dict = {}

    while not done:
        actions = agent.act(obs, deterministic=deterministic)
        obs, rewards, dones, info = env.step(actions)

        episode_return += rewards
        episode_length += 1
        reward_history.append(rewards.copy())
        positions_history.append(env.positions.copy())
        assignments_history.append(env.assignments.copy())
        arrived_history.append(env.arrived.copy())
        done = bool(np.all(dones))

    reason = info.get("termination_reason", "")
    metrics = {
        "episode_return_mean": float(np.mean(episode_return)),
        "episode_return_sum": float(np.sum(episode_return)),
        "episode_length": float(episode_length),
        "success": float("success" in reason),
        "collision_or_boundary": float("collision" in reason or "boundary_violation" in reason),
        "timeout": float("timeout" in reason),
    }

    trajectory_data = {
        "positions_history": np.asarray(positions_history, dtype=np.float32),
        "assignments_history": np.asarray(assignments_history, dtype=np.int64),
        "arrived_history": np.asarray(arrived_history, dtype=bool),
        "reward_history": np.asarray(reward_history, dtype=np.float32),
        "target_positions": env.target_positions.copy(),
        "obstacle_centers": env.obstacle_centers.copy(),
        "obstacle_radii": env.obstacle_radii.copy(),
        "final_assignments": env.assignments.copy(),
        "termination_reason": reason,
    }
    return metrics, trajectory_data


def plot_trajectory(
    trajectory_data: Dict,
    env_cfg: UAVEnvConfig,
    output_path: Path | None = None,
    show: bool = False,
    title: str = "DA-MAPPO Evaluation Trajectory",
) -> None:
    import matplotlib.pyplot as plt

    positions_history = trajectory_data["positions_history"]
    target_positions = trajectory_data["target_positions"]
    obstacle_centers = trajectory_data["obstacle_centers"]
    obstacle_radii = trajectory_data["obstacle_radii"]
    final_assignments = trajectory_data["final_assignments"]
    reason = trajectory_data["termination_reason"]

    num_agents = positions_history.shape[1]
    half_size = env_cfg.world_size / 2.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-half_size, half_size)
    ax.set_ylim(-half_size, half_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\ntermination: {reason}")
    ax.grid(True)

    for center, radius in zip(obstacle_centers, obstacle_radii):
        circle = plt.Circle(center, radius, fill=True, alpha=0.45)
        ax.add_patch(circle)

    ax.scatter(target_positions[:, 0], target_positions[:, 1], marker="*", s=180, label="Targets")
    for t_id, target in enumerate(target_positions):
        ax.text(target[0], target[1], f"T{t_id}")

    arrived_history = trajectory_data.get("arrived_history")
    for i in range(num_agents):
        full_traj = positions_history[:, i, :]
        if arrived_history is not None and arrived_history.shape[0] > 0:
            arrived_indices = np.where(arrived_history[:, i])[0]
            end_idx = int(arrived_indices[0]) if len(arrived_indices) > 0 else full_traj.shape[0] - 1
        else:
            end_idx = full_traj.shape[0] - 1

        traj = full_traj[: end_idx + 1]
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, label=f"UAV {i}")
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=80)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=100)

        assigned_target = target_positions[final_assignments[i]]
        ax.plot([traj[-1, 0], assigned_target[0]], [traj[-1, 1], assigned_target[1]], linestyle="--", linewidth=1)
        ax.text(traj[-1, 0], traj[-1, 1], f"U{i}")

    ax.legend(loc="upper left")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_trajectory(trajectory_data: Dict, env_cfg: UAVEnvConfig, output_path: Path, fps: int = 10) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    positions_history = trajectory_data["positions_history"]
    target_positions = trajectory_data["target_positions"]
    obstacle_centers = trajectory_data["obstacle_centers"]
    obstacle_radii = trajectory_data["obstacle_radii"]
    final_assignments = trajectory_data["final_assignments"]
    reason = trajectory_data["termination_reason"]

    num_agents = positions_history.shape[1]
    num_steps = positions_history.shape[0]
    half_size = env_cfg.world_size / 2.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-half_size, half_size)
    ax.set_ylim(-half_size, half_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    for center, radius in zip(obstacle_centers, obstacle_radii):
        ax.add_patch(plt.Circle(center, radius, fill=True, alpha=0.45))
    ax.scatter(target_positions[:, 0], target_positions[:, 1], marker="*", s=180, label="Targets")

    lines = [ax.plot([], [], linewidth=2, label=f"UAV {i}")[0] for i in range(num_agents)]
    dots = ax.plot([], [], "ko", ms=6)[0]
    assign_lines = [ax.plot([], [], "--", linewidth=1, alpha=0.5)[0] for _ in range(num_agents)]
    ax.legend(loc="upper left")

    def update(frame: int):
        end = frame + 1
        for i in range(num_agents):
            traj = positions_history[:end, i, :]
            lines[i].set_data(traj[:, 0], traj[:, 1])
            x, y = positions_history[end - 1, i]
            tx, ty = target_positions[final_assignments[i]]
            assign_lines[i].set_data([x, tx], [y, ty])
        dots.set_data(positions_history[end - 1, :, 0], positions_history[end - 1, :, 1])
        ax.set_title(f"step {end}/{num_steps}\ntermination: {reason}")
        return lines + [dots] + assign_lines

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani = FuncAnimation(fig, update, frames=num_steps, interval=1000 // fps, blit=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # matplotlib may not find ffmpeg automatically; check common conda locations.
    if not writers.is_available("ffmpeg"):
        candidates = [
            Path("D:/ffmepg/ffmpeg-2026-04-19-git-de18feb0f0-essentials_build/bin/ffmpeg.exe"),
            Path("D:/anaconda3/Library/bin/ffmpeg.exe"),
        ]
        for p in candidates:
            if p.exists():
                plt.rcParams["animation.ffmpeg_path"] = str(p)
                break
    ani.save(str(output_path), writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    aggregated = {}
    for key in keys:
        values = np.array([m[key] for m in metrics_list], dtype=np.float32)
        aggregated[f"mean_{key}"] = float(np.mean(values))
        aggregated[f"std_{key}"] = float(np.std(values))
    return aggregated


def main() -> None:
    args = parse_args()
    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    run_dir = Path(args.run_dir)
    train_args = load_run_config(run_dir)
    use_gnn = bool(args.use_gnn or train_args.get("use_gnn", False))

    run_tag = run_dir.name
    run_agent_numble = args.eval_num_agents
    output_dir = Path(args.output_dir) / run_tag /str(run_agent_numble)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_seed = args.seed if args.seed != 0 else train_args.get("seed", 42)
    set_seed(eval_seed)

    env = make_env_from_config(
        train_args,
        seed=eval_seed,
        eval_num_agents=args.eval_num_agents,
        num_obstacles=args.num_obstacles,
    )
    agent = load_agent_from_run(env, run_dir, train_args, args.device, args.checkpoint, use_gnn=use_gnn)

    print("DA-MAPPO evaluation started")
    print(f"algorithm: {'GNN-MAPPO' if use_gnn else 'MLP-MAPPO'}")
    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {run_dir / 'checkpoints' / args.checkpoint}")
    print(f"device: {agent.device}")
    print(f"episodes: {args.episodes}")
    print(f"num_agents: {env.num_agents}")
    print(f"num_obstacles: {env.cfg.num_obstacles}")
    print(f"assigner: {env.cfg.assigner_name}")
    if use_gnn:
        print(f"node_dim: {agent.node_dim}")
        print(f"edge_dim: {agent.edge_dim}")
        print(f"num_edges: {agent.num_edges}")
    else:
        print(f"obs_dim: {agent.obs_dim}")
        print(f"state_dim: {agent.state_dim}")
    print("-" * 80)

    all_metrics: List[Dict[str, float]] = []
    all_trajectories: List[Dict] = []

    for ep in range(args.episodes):
        metrics, trajectory_data = run_episode(
            agent=agent,
            env=env,
            seed=eval_seed + ep,
            deterministic=args.deterministic,
        )
        all_metrics.append(metrics)
        all_trajectories.append(trajectory_data)

        print(
            f"episode={ep + 1:03d} | "
            f"return_mean={metrics['episode_return_mean']:.3f} | "
            f"length={metrics['episode_length']:.0f} | "
            f"success={metrics['success']:.0f} | "
            f"reason={trajectory_data['termination_reason']}"
        )

        if not args.no_plots or args.show:
            plot_path = output_dir / f"trajectory_ep_{ep + 1:03d}.png" if not args.no_plots else None
            plot_trajectory(
                trajectory_data=trajectory_data,
                env_cfg=env.cfg,
                output_path=plot_path,
                show=args.show,
                title=f"DA-MAPPO Evaluation Episode {ep + 1}",
            )

    aggregated = aggregate_metrics(all_metrics)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print("-" * 80)
    print("Aggregated metrics:")
    for key, value in aggregated.items():
        print(f"{key}: {value:.6f}")

    if not args.no_plots:
        print(f"Trajectory plots saved to: {output_dir.resolve()}")
    print(f"Metrics JSON: {output_dir / 'metrics.json'}")

    if args.save_video and all_trajectories:
        best_idx = int(np.argmax([m["episode_return_mean"] for m in all_metrics]))
        video_path = output_dir / f"trajectory_best_ep_{best_idx + 1:03d}.mp4"
        print(f"Saving video for best-return episode {best_idx + 1} ...")
        animate_trajectory(all_trajectories[best_idx], env.cfg, video_path, fps=args.video_fps)
        print(f"Video saved to: {video_path}")


if __name__ == "__main__":
    main()
