#!/usr/bin/env python3
"""Experiment 1: N=3, 4 assigners x static/dynamic x 30/50/80 obstacles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from scripts.evaluate import (
    load_run_config,
    set_seed,
    make_env_from_config,
    load_agent_from_run,
    run_episode,
)


ASSIGNERS = ["fixed", "cbaa", "hungarian", "egtap"]
TARGET_MODES = ["static", "dynamic"]
OBSTACLE_COUNTS = [30, 50, 80]
EPISODES = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 1 batch evaluation")
    p.add_argument("run_dir", type=str, help="Path to T0 checkpoint run directory")
    p.add_argument("--checkpoint", type=str, default="best.pt")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="eval_outputs/exp1")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    train_args = load_run_config(run_dir)
    output_dir = Path(args.output_dir) / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for assigner in ASSIGNERS:
        for target_mode in TARGET_MODES:
            for n_obs in OBSTACLE_COUNTS:
                cell_tag = f"{assigner}_{target_mode}_O{n_obs}"
                print(f"\n{'='*60}\nExp1: {cell_tag}\n{'='*60}")

                set_seed(args.seed)
                env = make_env_from_config(
                    train_args, seed_offset=0,
                    assigner_name=assigner,
                    num_obstacles=n_obs,
                    dynamic_targets=(target_mode == "dynamic"),
                    target_motion_mode="swap" if target_mode == "dynamic" else "none",
                    reassign_interval=1,
                    eval_num_agents=3,
                )
                agent = load_agent_from_run(
                    env, run_dir, train_args, args.device, args.checkpoint, use_gnn=True,
                )
                agent.model.eval()

                cell_metrics = []
                for ep in range(EPISODES):
                    metrics, _traj = run_episode(agent, env, seed=args.seed + ep, deterministic=True)
                    cell_metrics.append(metrics)

                agg = {}
                for key in cell_metrics[0].keys():
                    vals = np.array([m[key] for m in cell_metrics], dtype=np.float32)
                    agg[f"mean_{key}"] = float(np.mean(vals))
                    agg[f"std_{key}"] = float(np.std(vals))
                agg["cell"] = cell_tag
                agg["assigner"] = assigner
                agg["target_mode"] = target_mode
                agg["num_obstacles"] = n_obs
                agg["num_agents"] = 3
                agg["episodes"] = EPISODES
                all_results.append(agg)

                cell_dir = output_dir / cell_tag
                cell_dir.mkdir(parents=True, exist_ok=True)
                with open(cell_dir / "metrics.json", "w") as f:
                    json.dump(agg, f, indent=2)
                print(f"  success={agg.get('mean_success', 'N/A'):.3f}  "
                      f"collision={agg.get('mean_collision_or_boundary', 'N/A'):.3f}")

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
