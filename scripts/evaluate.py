"""
Evaluation and visualization script.

Usage:
    python evaluate.py <run_dir> [--episodes 10] [--no-plots] [--show]
    python evaluate.py runs/mappo_N3_O20_fixed_S42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path for `rl_path_planning` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Tuple

import numpy as np
import torch

from rl_path_planning.env import MultiUAV2DEnv, UAVEnvConfig
from rl_path_planning.mappo import MAPPOAgent, MAPPOConfig
from rl_path_planning.gnn_mappo import GraphMAPPOAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DA-MAPPO policy.")
    parser.add_argument("run_dir", type=str,
                        help="Path to the training run directory")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument(
        "--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0,
                        help="Override eval seed (0 = use training seed)")
    parser.add_argument("--output-dir", type=str, default="eval_outputs")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Custom tag for output subdirectory (default: use run_dir name)")
    parser.add_argument("--checkpoint", type=str, default="best.pt",
                        help="Checkpoint filename in run_dir/checkpoints/ (default: best.pt)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively")
    parser.add_argument("--num-obstacles", type=int, default=None,
                        help="Override num_obstacles from training config")
    parser.add_argument("--assigner-name", type=str, default=None,
                        help="Override assigner_name from training config (e.g. 'hungarian', 'greedy', 'fixed', 'cbba')")
    parser.add_argument("--layout-mode", type=str, default=None,
                        help="Override layout_mode from training config (e.g. 'same_side', 'cross')")
    parser.add_argument("--dynamic-targets", type=bool, default=None,
                        help="Override dynamic_targets from training config (True/False)")
    parser.add_argument("--target-motion-mode", type=str, default=None,
                        help="Override target_motion_mode from training config")
    parser.add_argument("--racetrack-turn-radius", type=float, default=None,
                        help="Override racetrack_turn_radius")
    parser.add_argument("--racetrack-straight-half-length", type=float, default=None,
                        help="Override racetrack_straight_half_length")
    parser.add_argument("--racetrack-front-speed", type=float, default=None,
                        help="Override racetrack_front_speed")
    parser.add_argument("--racetrack-back-speed", type=float, default=None,
                        help="Override racetrack_back_speed")
    parser.add_argument("--save-video", action="store_true",
                        help="Save MP4 video of the best episode")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="FPS for saved video (default: 30)")

    # GNN.
    parser.add_argument("--eval-num-agents", type=int, default=None,
                        help="Override num_agents for evaluation (GNN cross-N tests)")
    parser.add_argument("--use-gnn", action="store_true",
                        help="Force GNN agent. If omitted, config.json use_gnn is used.")
    parser.add_argument("--torch-num-threads", type=int, default=1,
                        help="Set torch CPU threads (default 1 for small GNN batches)")
    return parser.parse_args()


# =============================================================================
# Utilities
# =============================================================================
def load_run_config(run_dir: Path) -> dict:
    """Read config.json from a run directory."""
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


def make_env_from_config(train_args: dict, seed_offset: int = 0,
                         num_obstacles: int | None = None,
                         assigner_name: str | None = None,
                         layout_mode: str | None = None,
                         dynamic_targets: bool | None = None,
                         target_motion_mode: str | None = None,
                         racetrack_turn_radius: float | None = None,
                         racetrack_straight_half_length: float | None = None,
                         racetrack_front_speed: float | None = None,
                         racetrack_back_speed: float | None = None,
                         eval_num_agents: int | None = None) -> MultiUAV2DEnv:
    """Build env from saved training config."""
    cfg = UAVEnvConfig(
        num_agents=eval_num_agents if eval_num_agents is not None else train_args["num_agents"],
        num_obstacles=num_obstacles if num_obstacles is not None else train_args[
            "num_obstacles"],
        max_steps=train_args["max_steps"],
        assigner_name=(
            assigner_name
            if assigner_name is not None else
            train_args["assigner_name"]
        ),
        reassign_interval=train_args.get("reassign_interval", 10),
        lidar_num_rays=train_args["lidar_num_rays"],
        lidar_range=train_args["lidar_range"],
        layout_mode=(
            layout_mode
            if layout_mode is not None else
            train_args.get("layout_mode", "same_side")
        ),
        dynamic_targets=(
            dynamic_targets
            if dynamic_targets is not None else
            train_args.get("dynamic_targets", False)
        ),
        target_motion_mode=(
            target_motion_mode
            if target_motion_mode is not None else
            train_args.get("target_motion_mode", "none")
        ),
        target_speed=train_args.get("target_speed", 0.2),
        racetrack_turn_radius=(
            racetrack_turn_radius
            if racetrack_turn_radius is not None else
            train_args.get("racetrack_turn_radius", 0.5)
        ),
        racetrack_straight_half_length=(
            racetrack_straight_half_length
            if racetrack_straight_half_length is not None else
            train_args.get("racetrack_straight_half_length", 9.0)
        ),
        racetrack_front_speed=(
            racetrack_front_speed
            if racetrack_front_speed is not None else
            train_args.get("racetrack_front_speed", 0.2)
        ),
        racetrack_back_speed=(
            racetrack_back_speed
            if racetrack_back_speed is not None else
            train_args.get("racetrack_back_speed", 2.0)
        ),
        seed=train_args["seed"] + seed_offset,
    )
    return MultiUAV2DEnv(cfg)


def load_agent_from_run(env: MultiUAV2DEnv, run_dir: Path, train_args: dict, device: str,
                        checkpoint: str = "best.pt",
                        use_gnn: bool = False):
    """Build agent from saved config and load checkpoint."""
    mappo_cfg = MAPPOConfig(
        rollout_steps=train_args["rollout_steps"],
        ppo_epochs=train_args["ppo_epochs"],
        minibatch_size=train_args["minibatch_size"],
        hidden_dim=train_args["hidden_dim"],
        num_hidden_layers=train_args["num_hidden_layers"],
        activation=train_args["activation"],
        device=device,
    )
    agent = GraphMAPPOAgent(env, mappo_cfg) if use_gnn else MAPPOAgent(env, mappo_cfg)

    checkpoint_path = run_dir / "checkpoints" / checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if use_gnn:
        agent.load(checkpoint_path, load_optimizer=False)
    else:
        agent.load(checkpoint_path)
    agent.model.eval()
    return agent


def run_episode(
    agent: MAPPOAgent,
    env: MultiUAV2DEnv,
    seed: int | None,
    deterministic: bool = True,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray | str]]:
    """Run one evaluation episode and collect trajectory data."""
    obs = env.reset(seed=seed)
    done = False

    positions_history: List[np.ndarray] = [env.positions.copy()]
    target_positions_history: List[np.ndarray] = [env.target_positions.copy()]
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
        target_positions_history.append(env.target_positions.copy())
        assignments_history.append(env.assignments.copy())
        arrived_history.append(env.arrived.copy())

        done = bool(np.all(dones))

    reason = info.get("termination_reason", "")
    success = float("success" in reason)
    collision_or_boundary = float(
        "collision" in reason or "boundary_violation" in reason)
    timeout = float("timeout" in reason)

    metrics = {
        "episode_return_mean": float(np.mean(episode_return)),
        "episode_return_sum": float(np.sum(episode_return)),
        "episode_length": float(episode_length),
        "success": success,
        "collision_or_boundary": collision_or_boundary,
        "timeout": timeout,
    }

    trajectory_data = {
        "positions_history": np.asarray(positions_history, dtype=np.float32),
        "target_positions_history": np.asarray(target_positions_history, dtype=np.float32),
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
    trajectory_data: Dict[str, np.ndarray | str],
    env_cfg: UAVEnvConfig,
    output_path: Path | None = None,
    show: bool = False,
    title: str = "DA-MAPPO Evaluation Trajectory",
) -> None:
    """Plot UAV trajectories, targets, obstacles, and final assignments."""
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

    # Obstacles.
    for center, radius in zip(obstacle_centers, obstacle_radii):
        circle = plt.Circle(center, radius, fill=True, alpha=0.45)
        ax.add_patch(circle)

    # Targets (final positions + trajectory if available).
    target_positions_history = trajectory_data.get("target_positions_history")
    if target_positions_history is not None:
        for i in range(target_positions.shape[0]):
            t_traj = target_positions_history[:, i, :]
            ax.plot(t_traj[:, 0], t_traj[:, 1], linewidth=0.5, alpha=0.4)
            ax.scatter(t_traj[0, 0], t_traj[0, 1], marker="s", s=40)
    ax.scatter(target_positions[:, 0], target_positions[:,
               1], marker="*", s=180, label="Targets")
    for t_id, target in enumerate(target_positions):
        ax.text(target[0], target[1], f"T{t_id}")

    # UAV trajectories (full, since agents do not freeze after arrival).
    for i in range(num_agents):
        full_traj = positions_history[:, i, :]
        traj = full_traj
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, label=f"UAV {i}")
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=80)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=100)

        assigned_target = target_positions[final_assignments[i]]
        ax.plot(
            [traj[-1, 0], assigned_target[0]],
            [traj[-1, 1], assigned_target[1]],
            linestyle="--",
            linewidth=1,
        )
        ax.text(traj[-1, 0], traj[-1, 1], f"U{i}")

    ax.legend(loc="upper left")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_trajectory(
    trajectory_data: Dict[str, np.ndarray | str],
    env_cfg: UAVEnvConfig,
    output_path: Path,
    fps: int = 10,
    title: str = "DA-MAPPO Evaluation Trajectory",
) -> None:
    """Save an MP4 video of the trajectory animation."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, writers

    positions_history = trajectory_data["positions_history"]
    target_positions_history = trajectory_data.get(
        "target_positions_history",
        np.tile(trajectory_data["target_positions"], (positions_history.shape[0], 1, 1)),
    )
    obstacle_centers = trajectory_data["obstacle_centers"]
    obstacle_radii = trajectory_data["obstacle_radii"]
    assignments_history = trajectory_data.get("assignments_history")
    if assignments_history is None:
        assignments_history = np.tile(
            trajectory_data["final_assignments"], (positions_history.shape[0], 1))
    reward_history = trajectory_data["reward_history"]
    reason = trajectory_data["termination_reason"]

    num_agents = positions_history.shape[1]
    num_steps = positions_history.shape[0]
    half_size = env_cfg.world_size / 2.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-half_size, half_size)
    ax.set_ylim(-half_size, half_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # Static elements.
    for center, radius in zip(obstacle_centers, obstacle_radii):
        circle = plt.Circle(center, radius, fill=True, alpha=0.45)
        ax.add_patch(circle)

    # Animated elements.
    target_scatter = ax.scatter([], [], marker="*", s=180, label="Targets")
    target_labels = [ax.text(0, 0, f"T{i}", visible=False)
                     for i in range(num_agents)]
    trail_lines = [ax.plot([], [], linewidth=2, label=f"UAV {i}")[
        0] for i in range(num_agents)]
    pos_dots = ax.plot([], [], "ko", ms=6)[0]
    uav_labels = [ax.text(0, 0, f"U{i}", visible=False)
                  for i in range(num_agents)]
    assign_lines = [ax.plot([], [], "--", linewidth=1, alpha=0.5)[0]
                    for i in range(num_agents)]
    ax.legend(loc="upper left")

    info_text = ax.text(0.98, 0.98, "", transform=ax.transAxes,
                        va="top", ha="right", fontsize=9)

    def update(frame: int):
        end = frame + 1
        # Animate targets
        t_pos = target_positions_history[end - 1]
        target_scatter.set_offsets(t_pos)
        for i in range(num_agents):
            target_labels[i].set_position((t_pos[i, 0], t_pos[i, 1]))
            target_labels[i].set_visible(True)

        for i in range(num_agents):
            traj = positions_history[:end, i, :]
            trail_lines[i].set_data(traj[:, 0], traj[:, 1])
            x, y = positions_history[end - 1, i]
            uav_labels[i].set_position((x, y))
            uav_labels[i].set_visible(True)
            tx, ty = t_pos[assignments_history[end - 1, i]]
            assign_lines[i].set_data([x, tx], [y, ty])

        pos_dots.set_data(
            positions_history[end - 1, :, 0],
            positions_history[end - 1, :, 1]
        )

        parts = [f"step {end}/{num_steps}"]
        if frame < len(reward_history):
            parts.append(
                f"reward mean: {float(np.mean(reward_history[frame])):.3f}"
            )
        info_text.set_text("  ".join(parts))

        ax.set_title(f"{title}\ntermination: {reason}")
        return ([target_scatter] + trail_lines + [pos_dots] + target_labels +
                uav_labels + assign_lines + [info_text])

    ani = FuncAnimation(fig,
                        update,
                        frames=num_steps,
                        interval=1000 // fps,
                        blit=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # matplotlib may not find ffmpeg automatically; check common conda locations.
    if not writers.is_available("ffmpeg"):
        candidates = [
            Path("D:/anaconda3/envs/deepRL/Library/bin/ffmpeg.exe"),
            Path("D:/anaconda3/Library/bin/ffmpeg.exe"),
        ]
        for p in candidates:
            if p.exists():
                plt.rcParams["animation.ffmpeg_path"] = str(p)
                break

    ani.save(str(output_path), writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate episode metrics."""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    aggregated = {}
    for key in keys:
        values = np.array([m[key] for m in metrics_list], dtype=np.float32)
        aggregated[f"mean_{key}"] = float(np.mean(values))
        aggregated[f"std_{key}"] = float(np.std(values))
    return aggregated


# =============================================================================
# Main evaluation
# =============================================================================
def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    run_tag = (
        args.run_tag
        if args.run_tag is not None else
        args.run_dir.strip("\\").split("\\")[-1]
    )
    train_args = load_run_config(run_dir)
    use_gnn = bool(args.use_gnn or train_args.get("use_gnn", False))
    print(f"Loaded config from: {run_dir / 'config.json'}")

    output_dir = Path(args.output_dir + f"/{run_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_seed = args.seed if args.seed != 0 else train_args.get("seed", 42)
    set_seed(eval_seed)

    env = make_env_from_config(
        train_args,
        seed_offset=0,
        num_obstacles=args.num_obstacles,
        assigner_name=args.assigner_name,
        layout_mode=args.layout_mode,
        dynamic_targets=args.dynamic_targets,
        target_motion_mode=args.target_motion_mode,
        racetrack_turn_radius=args.racetrack_turn_radius,
        racetrack_straight_half_length=args.racetrack_straight_half_length,
        racetrack_front_speed=args.racetrack_front_speed,
        racetrack_back_speed=args.racetrack_back_speed,
        eval_num_agents=args.eval_num_agents,
    )
    agent = load_agent_from_run(
        env, run_dir, train_args, args.device, args.checkpoint,
        use_gnn=use_gnn)

    algo = "GNN-MAPPO" if use_gnn else "MLP-MAPPO"
    print(f"DA-MAPPO evaluation started [{algo}]")
    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {run_dir / 'checkpoints' / args.checkpoint}")
    print(f"device: {agent.device}")
    print(f"episodes: {args.episodes}")
    print(f"num_agents: {env.num_agents}")
    print(f"num_obstacles: {train_args['num_obstacles']}")
    print(f"assigner: {train_args['assigner_name']}")
    print(f"dynamic_targets: {env.cfg.dynamic_targets}  mode: {env.cfg.target_motion_mode}")
    if use_gnn:
        print(f"node_dim: {agent.node_dim}")
        print(f"edge_dim: {agent.edge_dim}")
        print(f"num_edges: dynamic (distance-gated per step)")
    else:
        print(f"obs_dim: {agent.obs_dim}")
        print(f"state_dim: {agent.state_dim}")
    print("-" * 80)

    all_metrics: List[Dict[str, float]] = []
    all_trajectories: List[Dict[str, np.ndarray | str]] = []

    for ep in range(args.episodes):
        episode_seed = eval_seed + ep
        metrics, trajectory_data = run_episode(
            agent=agent,
            env=env,
            seed=episode_seed,
            deterministic=args.deterministic,
        )
        all_metrics.append(metrics)
        all_trajectories.append(trajectory_data)

        reason = trajectory_data["termination_reason"]
        print(
            f"episode={ep + 1:03d} | "
            f"return_mean={metrics['episode_return_mean']:.3f} | "
            f"length={metrics['episode_length']:.0f} | "
            f"success={metrics['success']:.0f} | "
            f"reason={reason}"
        )

        if not args.no_plots or args.show:
            plot_path = output_dir / \
                f"trajectory_ep_{ep + 1:03d}.png" if not args.no_plots else None
            plot_trajectory(
                trajectory_data=trajectory_data,
                env_cfg=env.cfg,
                output_path=plot_path,
                show=args.show,
                title=f"DA-MAPPO Evaluation Episode {ep + 1}",
            )

    aggregated = aggregate_metrics(all_metrics)
    print("-" * 80)
    print("Aggregated metrics:")
    for key, value in aggregated.items():
        print(f"{key}: {value:.6f}")

    if not args.no_plots:
        print(f"Trajectory plots saved to: {output_dir.resolve()}")

    if args.save_video:
        print("-" * 80)
        print("Episode summary:")
        for ep in range(args.episodes):
            m = all_metrics[ep]
            print(f"  ep={ep + 1:03d}  return_mean={m['episode_return_mean']:.3f}  "
                  f"success={m['success']:.0f}  reason={all_trajectories[ep]['termination_reason']}")

        raw = input(
            "Enter episode numbers to save (e.g. 1,3,5), or press Enter to skip: ").strip()
        if raw:
            selected = [int(x.strip()) - 1 for x in raw.split(",")
                        if x.strip().isdigit()]
            for idx in selected:
                if 0 <= idx < args.episodes:
                    video_path = output_dir / \
                        f"trajectory_ep_{idx + 1:03d}.mp4"
                    print(f"Saving video for episode {idx + 1} ...")
                    animate_trajectory(
                        trajectory_data=all_trajectories[idx],
                        env_cfg=env.cfg,
                        output_path=video_path,
                        fps=args.video_fps,
                        title=f"DA-MAPPO Episode {idx + 1}",
                    )
                    print(f"  saved to: {video_path}")
        else:
            print("Skipped video save.")


if __name__ == "__main__":
    main()
