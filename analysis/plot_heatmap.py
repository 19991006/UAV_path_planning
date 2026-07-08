#!/usr/bin/env python3
"""Generate figures: Exp1 bar chart, Exp2 heatmaps, Exp3 timing curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def plot_exp1(csv_path: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path)))
    assigners = ["fixed", "cbaa", "hungarian", "egtap"]
    modes = ["static", "dynamic"]
    obs = [30, 50, 80]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for idx, n_obs in enumerate(obs):
        ax = axes[idx]
        x = np.arange(len(assigners))
        w = 0.35
        for mi, mode in enumerate(modes):
            vals = []
            for a in assigners:
                match = [r for r in rows
                         if r.get("assigner") == a and r.get("target_mode") == mode
                         and int(r.get("num_obstacles", 0)) == n_obs]
                vals.append(float(match[0]["mean_success"]) * 100 if match else 0)
            off = w * (mi - 0.5) + w / 2
            bars = ax.bar(x + off, vals, w * 0.9, label=mode)
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                            f"{v:.0f}", ha="center", fontsize=8)
        ax.set_title(f"{n_obs} obstacles")
        ax.set_xticks(x)
        ax.set_xticklabels(assigners)
        if idx == 0:
            ax.legend()
        ax.set_ylabel("Success Rate (%)")
    fig.suptitle("Experiment 1: When does assignment matter?")
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_bar.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'exp1_bar.png'}")


def plot_exp2_heatmap(csv_path: Path, out_dir: Path, mode: str, n_obs: int) -> None:
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path)))
    ns = [3, 5, 10, 15, 20]
    mat = np.zeros((len(ns), len(ns)))
    for i, tn in enumerate(ns):
        for j, en in enumerate(ns):
            match = [r for r in rows
                     if int(r.get("train_n", 0)) == tn
                     and int(r.get("eval_n", 0)) == en
                     and r.get("target_mode") == mode
                     and int(r.get("num_obstacles", 0)) == n_obs]
            mat[i, j] = float(match[0]["mean_success"]) * 100 if match else 0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    for i in range(len(ns)):
        for j in range(len(ns)):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns)
    ax.set_yticks(range(len(ns))); ax.set_yticklabels(ns)
    ax.set_xlabel("Evaluation N"); ax.set_ylabel("Training N")
    ax.set_title(f"Success Rate - {mode}, {n_obs} obstacles")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"exp2_heatmap_{mode}_O{n_obs}.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / f'exp2_heatmap_{mode}_O{n_obs}.png'}")


def plot_exp3(csv_path: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path)))
    eg = sorted([(int(r["N"]), float(r["mean_ms"])) for r in rows
                 if r.get("component") == "egtap"])
    hu = sorted([(int(r["N"]), float(r["mean_ms"])) for r in rows
                 if r.get("component") == "hungarian"])
    gnn = sorted([(int(r["N"]), float(r["mean_ms"])) for r in rows
                  if r.get("component") == "gnn_inference"])

    fig, ax = plt.subplots(figsize=(7, 5))
    if eg: ax.plot(*zip(*eg), "o-", label="EG-TAP", lw=2)
    if hu: ax.plot(*zip(*hu), "s-", label="Hungarian", lw=2)
    if gnn: ax.plot(*zip(*gnn), "^-", label="GNN inference", lw=2)
    ax.set_xlabel("N"); ax.set_ylabel("Time (ms)")
    ax.set_title("Experiment 3: Per-Step Computation Time vs Swarm Size")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_timing.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'exp3_timing.png'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate experiment figures")
    p.add_argument("--exp1-csv", type=str, default="eval_outputs/exp1_summary.csv")
    p.add_argument("--exp2-csv", type=str, default="eval_outputs/exp2_summary.csv")
    p.add_argument("--exp3-csv", type=str, default="eval_outputs/exp3/timing_results.csv")
    p.add_argument("--output-dir", type=str, default="eval_outputs/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if Path(args.exp1_csv).exists():
        plot_exp1(Path(args.exp1_csv), out)
    if Path(args.exp2_csv).exists():
        for m in ["static", "dynamic"]:
            for o in [30, 50, 80]:
                plot_exp2_heatmap(Path(args.exp2_csv), out, m, o)
    if Path(args.exp3_csv).exists():
        plot_exp3(Path(args.exp3_csv), out)


if __name__ == "__main__":
    main()
