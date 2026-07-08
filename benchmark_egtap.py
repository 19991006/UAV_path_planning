"""
Benchmark EG-TAP convergence with perturbation (Algorithm 3) enabled.

Uses the same initial layout and distance-based communication graph as env.py.
Tests native convergence (no fallback) across N and α.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np

from target_assignment import EGTAPTargetAssigner, HungarianTargetAssigner


# ---------------------------------------------------------------------------
# Layout helpers — identical logic to UAVEnvConfig / MultiUAV2DEnv
# ---------------------------------------------------------------------------

def _scale_factor(num_agents: int) -> float:
    return num_agents / 5.0


def build_initial_positions(
    num_agents: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Deterministic layout: agents left, targets right (matching env.py).

    No position perturbation — symmetry breaking is handled by EG-TAP's
    cost perturbation (Algorithm 3).
    """
    sf = _scale_factor(num_agents)
    world_size = 20.0 * sf
    start_x = -8.0 * sf
    target_x = 8.0 * sf
    gap = 4.0
    comm_range = 8.0

    center = (num_agents - 1) / 2.0
    agents = np.zeros((num_agents, 2), dtype=np.float32)
    targets = np.zeros((num_agents, 2), dtype=np.float32)
    for i in range(num_agents):
        y = (i - center) * gap
        agents[i, 0] = start_x
        agents[i, 1] = y
        targets[i, 0] = target_x
        targets[i, 1] = y

    return agents, targets, world_size, comm_range


def build_comm_adj(positions: np.ndarray, comm_range: float) -> np.ndarray:
    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    adj = (dists <= comm_range) & ~np.eye(positions.shape[0], dtype=bool)
    return adj


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    num_agents: int,
    step_size: float,
    max_iterations: int = 50000,
    trials: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)

    hungarian = HungarianTargetAssigner(squared_distance=True)
    egtap = EGTAPTargetAssigner(
        step_size=step_size, max_iterations=max_iterations,
        squared_distance=True, perturbation_scale=1e-6, check_interval=1,
        seed=int(rng.integers(0, 2**31)),
    )

    native_cf_count = 0
    iters_list = []
    cost_ratios = []
    lap_norms = []
    failures = 0

    for _ in range(trials):
        agents, targets, world_size, comm_range = build_initial_positions(num_agents)
        comm_adj = build_comm_adj(agents, comm_range)

        # Hungarian oracle
        h_assign, h_cost, _ = hungarian.assign(agents.copy(), targets.copy())
        h_total = float(np.sum(h_cost[np.arange(num_agents), h_assign]))

        # Laplacian norm
        degree = comm_adj.sum(axis=1).astype(np.float32)
        L_mat = np.diag(degree) - comm_adj.astype(np.float32)
        eigvals = np.linalg.eigvalsh(L_mat)
        lap_norms.append(float(eigvals[-1]))

        # EG-TAP with perturbation
        try:
            egtap_assign, _, egtap_info = egtap.assign(
                agents.copy(), targets.copy(), communication_graph=comm_adj,
            )
        except RuntimeError:
            failures += 1
            continue

        # conflict-free check
        if len(set(egtap_assign)) == num_agents:
            native_cf_count += 1

        iters_list.append(egtap_info["iterations"])
        cost = float(np.sum(np.linalg.norm(agents - targets[egtap_assign], axis=1) ** 2))
        cost_ratios.append(cost / max(h_total, 1e-8))

    return {
        "N": num_agents,
        "alpha": step_size,
        "trials": trials,
        "native_cf_pct": 100.0 * native_cf_count / trials,
        "failures": failures,
        "median_iters": float(np.median(iters_list)) if iters_list else None,
        "max_iters": float(np.max(iters_list)) if iters_list else None,
        "mean_cost_ratio": float(np.mean(cost_ratios)) if cost_ratios else None,
        "lap_norm_max": float(np.max(lap_norms)),
    }


# ---------------------------------------------------------------------------
# Grid scan
# ---------------------------------------------------------------------------

def run_grid(
    N_values=(3, 5, 10, 20),
    alpha_values=(0.01, 0.02, 0.05, 0.1, 0.2),
    trials: int = 10,
    max_iterations: int = 50000,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    header = f"{'N':>4} {'α':>8} {'cf%':>8} {'fail':>6} {'med_iter':>10} {'max_iter':>10} {'cost':>8} {'||L||':>8}"
    print(header)
    print("-" * len(header))

    results = []

    for n in N_values:
        for alpha in alpha_values:
            r = evaluate_one(n, alpha, max_iterations=max_iterations, trials=trials, rng=rng)
            results.append(r)

            iters_str = f"{r['median_iters']:.0f}" if r['median_iters'] is not None else "FAIL"
            max_it_str = f"{r['max_iters']:.0f}" if r['max_iters'] is not None else "FAIL"
            cr_str = f"{r['mean_cost_ratio']:.4f}" if r['mean_cost_ratio'] is not None else "FAIL"

            print(
                f"{r['N']:>4} {r['alpha']:>8.4f} {r['native_cf_pct']:>7.1f}% "
                f"{r['failures']:>5} "
                f"{iters_str:>10} {max_it_str:>10} {cr_str:>8} {r['lap_norm_max']:>8.2f}"
            )
        print()

    return results


def summarize(results: list[dict]):
    print("=" * 70)
    print("Best α per N (100% cf, lowest median_iters):")

    by_n = {}
    for r in results:
        by_n.setdefault(r["N"], []).append(r)

    for n, entries in sorted(by_n.items()):
        entries.sort(key=lambda x: (
            x["native_cf_pct"],
            x["failures"],
            -(x["median_iters"] or 1e9)
        ), reverse=True)
        best = entries[0]
        print(f"  N={n:>2}: α={best['alpha']:.4f}, cf={best['native_cf_pct']:.0f}%, "
              f"med_iter={best['median_iters']:.0f}, max_iter={best['max_iters']:.0f}, "
              f"cost={best['mean_cost_ratio']:.4f}, ||L||={best['lap_norm_max']:.2f}")

    print()
    print("Theoretical upper bound α < 1/||L||_max (Corollary 1):")
    for n, entries in sorted(by_n.items()):
        lap = entries[0]["lap_norm_max"]
        print(f"  N={n:>2}: α < {1.0/lap:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark EG-TAP with perturbation")
    ap.add_argument("--N", type=int, nargs="+", default=[3, 5, 10, 20])
    ap.add_argument("--alpha", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--max-iters", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results = run_grid(
        N_values=args.N,
        alpha_values=args.alpha,
        trials=args.trials,
        max_iterations=args.max_iters,
        seed=args.seed,
    )
    summarize(results)
