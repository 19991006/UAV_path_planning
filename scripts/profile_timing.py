#!/usr/bin/env python3
"""Experiment 3: Pure assignment solve-time and inference-time profiling vs N."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from rl_path_planning.target_assignment import build_assigner
from rl_path_planning.gnn_actor_critic import GraphActorCritic, GraphNetworkConfig


N_VALUES = [3, 5, 10, 15, 20, 30, 50]
N_ITERS = 1000
NODE_DIM = 35 + 4 + 2 + 12  # lidar + ego + target + comm(2*(7-1) for N=7 compat)
EDGE_DIM = 4


def random_positions(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-9.0, 9.0, size=(n, 2)).astype(np.float32)


def full_adjacency(n: int) -> np.ndarray:
    adj = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def profile_assigner(name: str, n_vals: list[int], n_iters: int,
                     rng: np.random.Generator) -> list[dict]:
    assigner = build_assigner(name)
    results = []
    for n in n_vals:
        agents = random_positions(n, rng)
        targets = random_positions(n, rng)
        comm = full_adjacency(n)
        for _ in range(50):
            assigner.assign(agents, targets, communication_graph=comm)
        times = []
        for _ in range(n_iters):
            agents = random_positions(n, rng)
            targets = random_positions(n, rng)
            t0 = time.perf_counter()
            assigner.assign(agents, targets, communication_graph=comm)
            times.append(time.perf_counter() - t0)
        arr = np.array(times)
        r = {"component": name, "N": n, "mean_ms": float(arr.mean() * 1000),
             "std_ms": float(arr.std() * 1000), "min_ms": float(arr.min() * 1000),
             "max_ms": float(arr.max() * 1000)}
        results.append(r)
        print(f"  {name} N={n:3d}: {arr.mean()*1000:.3f} +/- {arr.std()*1000:.3f} ms")
    return results


def profile_inference(n_vals: list[int], n_iters: int, device: str = "cpu") -> list[dict]:
    config = GraphNetworkConfig(
        node_dim=NODE_DIM, edge_dim=EDGE_DIM, action_dim=2,
        hidden_dim=128, num_gnn_layers=2, activation="relu",
    )
    model = GraphActorCritic(config).to(device)
    model.eval()
    results = []
    for n in n_vals:
        nodes = torch.randn(n, NODE_DIM, device=device)
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append([i, j])
        ei = torch.tensor(edges, dtype=torch.long, device=device).t()
        ea = torch.randn(ei.shape[1], EDGE_DIM, device=device)
        for _ in range(20):
            with torch.no_grad():
                model(nodes, ei, ea)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(nodes, ei, ea)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        arr = np.array(times)
        r = {"component": "gnn_inference", "N": n, "device": device,
             "mean_ms": float(arr.mean() * 1000), "std_ms": float(arr.std() * 1000),
             "min_ms": float(arr.min() * 1000), "max_ms": float(arr.max() * 1000)}
        results.append(r)
        print(f"  GNN N={n:3d} [{device}]: {arr.mean()*1000:.3f} +/- {arr.std()*1000:.3f} ms")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3: timing profiling")
    p.add_argument("--output-dir", type=str, default="eval_outputs/exp3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_results = []

    print("EG-TAP profiling...")
    all_results.extend(profile_assigner("egtap", N_VALUES, N_ITERS, rng))
    print("Hungarian profiling...")
    all_results.extend(profile_assigner("hungarian", N_VALUES, N_ITERS, rng))
    print(f"GNN inference profiling on {args.device}...")
    all_results.extend(profile_inference(N_VALUES, N_ITERS, args.device))

    with open(out / "timing_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out / "timing_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "N", "device",
                            "mean_ms", "std_ms", "min_ms", "max_ms"])
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"\nSaved: {out.resolve()}")


if __name__ == "__main__":
    main()
