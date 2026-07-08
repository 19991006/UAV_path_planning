#!/usr/bin/env python3
"""Experiment 2: 2x5x5 generalization heatmap (static + dynamic)."""

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

MODELS = {
    (3, "static"): "T0",
    (5, "static"): "S5",
    (10, "static"): "S10",
    (15, "static"): "S15",
    (20, "static"): "S20",
    (3, "dynamic"): "D3",
    (5, "dynamic"): "D5",
    (10, "dynamic"): "D10",
    (15, "dynamic"): "D15",
    (20, "dynamic"): "D20",
}
EVAL_N_VALUES = [3, 5, 10, 15, 20]
OBSTACLE_COUNTS = [30, 50, 80]
EPISODES = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 2 heatmap evaluation")
    p.add_argument("--model-map", type=str, required=True,
                   help="JSON file mapping model ID -> run_dir path")
    p.add_argument("--checkpoint", type=str, default="best.pt")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="eval_outputs/exp2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-mode", type=str, default="both",
                   choices=["static", "dynamic", "both"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.model_map) as f:
        model_map = json.load(f)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    modes = ["static", "dynamic"] if args.target_mode == "both" else [args.target_mode]

    for mode in modes:
        for (train_n, target_mode), model_id in MODELS.items():
            if target_mode != mode:
                continue
            if model_id not in model_map:
                print(f"WARNING: {model_id} not in model_map, skipping")
                continue
            run_dir = Path(model_map[model_id])
            train_args = load_run_config(run_dir)

            for eval_n in EVAL_N_VALUES:
                for n_obs in OBSTACLE_COUNTS:
                    tag = f"train{train_n}_{mode}_eval{eval_n}_O{n_obs}"
                    print(f"Exp2: {tag} (model={model_id})")

                    set_seed(args.seed)
                    env = make_env_from_config(
                        train_args, seed_offset=0,
                        num_obstacles=n_obs,
                        assigner_name="egtap",
                        dynamic_targets=(mode == "dynamic"),
                        target_motion_mode="swap" if mode == "dynamic" else "none",
                        reassign_interval=1,
                        eval_num_agents=eval_n,
                    )
                    agent = load_agent_from_run(
                        env, run_dir, train_args, args.device, args.checkpoint, use_gnn=True,
                    )
                    agent.model.eval()

                    cell_metrics = []
                    for ep in range(EPISODES):
                        m, _ = run_episode(agent, env, seed=args.seed + ep, deterministic=True)
                        cell_metrics.append(m)

                    agg = {}
                    for key in cell_metrics[0].keys():
                        vals = np.array([m[key] for m in cell_metrics], dtype=np.float32)
                        agg[f"mean_{key}"] = float(np.mean(vals))
                        agg[f"std_{key}"] = float(np.std(vals))
                    agg["train_n"] = train_n
                    agg["eval_n"] = eval_n
                    agg["target_mode"] = mode
                    agg["num_obstacles"] = n_obs
                    agg["model_id"] = model_id
                    all_results.append(agg)

                    cell_dir = output_dir / tag
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    with open(cell_dir / "metrics.json", "w") as f:
                        json.dump(agg, f, indent=2)
                    print(f"  success={agg.get('mean_success', 'N/A'):.3f}")

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
