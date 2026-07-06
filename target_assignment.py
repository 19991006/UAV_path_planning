"""
Pluggable online target assignment module for DA-MAPPO reproduction.

Save as:
    assignment/target_assignment.py

Design goal:
- The environment should not depend on a specific assignment algorithm.
- Any online assignment method only needs to implement BaseTargetAssigner.assign().
- Hungarian assignment is provided as the default method.
- Later you can replace it with auction-based assignment, greedy assignment,
  learned assignment, graph matching, attention-based assignment, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError(
        "HungarianTargetAssigner requires scipy. Install it with: pip install scipy"
    ) from exc


AssignmentResult = Tuple[np.ndarray, np.ndarray, Dict]


@dataclass
class BaseTargetAssigner(ABC):
    """
    Abstract base class for online target assignment.

    Required input:
        agent_positions: np.ndarray, shape [num_agents, 2]
        target_positions: np.ndarray, shape [num_targets, 2]

    Required output:
        assignments: np.ndarray, shape [num_agents]
            assignments[i] = target index assigned to agent i

        cost_matrix: np.ndarray, shape [num_agents, num_targets]
            cost matrix used by the assigner. If a method does not explicitly
            use a cost matrix, it can still return a diagnostic pseudo-cost.

        info: dict
            extra diagnostic information.
    """

    @abstractmethod
    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        raise NotImplementedError


@dataclass
class HungarianTargetAssigner(BaseTargetAssigner):
    """
    Minimum-cost one-to-one assignment using Hungarian algorithm.

    This corresponds to the assignment module used in the paper:
        C[i, j] = ||p_i - q_j||_2^2

    Args:
        squared_distance: whether to use squared Euclidean distance.
                          The paper uses squared Euclidean distance.
    """

    squared_distance: bool = True

    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        self._validate_inputs(agent_positions, target_positions)

        cost_matrix = self.build_cost_matrix(agent_positions, target_positions)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        num_agents = agent_positions.shape[0]
        assignments = np.full((num_agents,), -1, dtype=np.int64)
        for agent_id, target_id in zip(row_ind, col_ind):
            assignments[agent_id] = target_id

        if np.any(assignments < 0):
            raise RuntimeError(
                "Hungarian assignment failed to assign every agent. "
                "Check that num_targets >= num_agents."
            )

        total_cost = float(cost_matrix[row_ind, col_ind].sum())
        info = {
            "assigner": "hungarian",
            "total_assignment_cost": total_cost,
        }
        return assignments, cost_matrix.astype(np.float32), info

    def build_cost_matrix(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
    ) -> np.ndarray:
        diff = agent_positions[:, None, :] - target_positions[None, :, :]
        distances_sq = np.sum(diff ** 2, axis=-1)
        if self.squared_distance:
            return distances_sq.astype(np.float32)
        return np.sqrt(distances_sq + 1e-8).astype(np.float32)

    @staticmethod
    def _validate_inputs(
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
    ) -> None:
        if agent_positions.ndim != 2 or agent_positions.shape[1] != 2:
            raise ValueError(
                f"agent_positions must have shape [num_agents, 2], got {agent_positions.shape}."
            )
        if target_positions.ndim != 2 or target_positions.shape[1] != 2:
            raise ValueError(
                f"target_positions must have shape [num_targets, 2], got {target_positions.shape}."
            )
        if target_positions.shape[0] < agent_positions.shape[0]:
            raise ValueError(
                "num_targets must be >= num_agents for one-to-one assignment. "
                f"Got num_targets={target_positions.shape[0]}, "
                f"num_agents={agent_positions.shape[0]}."
            )


@dataclass
class GreedyNearestTargetAssigner(BaseTargetAssigner):
    """
    Simple greedy nearest-target assigner.

    This is mainly useful as a quick replaceable baseline.
    It sorts all agent-target pairs by distance and greedily picks non-conflicting pairs.
    It is not globally optimal, but it is fast and easy to compare against Hungarian.
    """

    squared_distance: bool = True

    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        HungarianTargetAssigner._validate_inputs(agent_positions, target_positions)

        diff = agent_positions[:, None, :] - target_positions[None, :, :]
        distances_sq = np.sum(diff ** 2, axis=-1)
        cost_matrix = distances_sq if self.squared_distance else np.sqrt(distances_sq + 1e-8)

        num_agents, num_targets = cost_matrix.shape
        assignments = np.full((num_agents,), -1, dtype=np.int64)
        used_targets = set()

        pairs = [
            (float(cost_matrix[i, j]), i, j)
            for i in range(num_agents)
            for j in range(num_targets)
        ]
        pairs.sort(key=lambda x: x[0])

        for _, agent_id, target_id in pairs:
            if assignments[agent_id] != -1:
                continue
            if target_id in used_targets:
                continue
            assignments[agent_id] = target_id
            used_targets.add(target_id)
            if np.all(assignments >= 0):
                break

        if np.any(assignments < 0):
            raise RuntimeError("Greedy assignment failed to assign every agent.")

        total_cost = float(sum(cost_matrix[i, assignments[i]] for i in range(num_agents)))
        info = {
            "assigner": "greedy_nearest",
            "total_assignment_cost": total_cost,
        }
        return assignments, cost_matrix.astype(np.float32), info


class FixedTargetAssigner(BaseTargetAssigner):
    """
    Fixed assignment baseline: agent i -> target i.

    Useful for ablation:
    - FixedTargetAssigner + MAPPO approximates ordinary MAPPO with fixed targets.
    - HungarianTargetAssigner + MAPPO is DA-MAPPO-style dynamic assignment.
    """

    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        HungarianTargetAssigner._validate_inputs(agent_positions, target_positions)

        num_agents = agent_positions.shape[0]
        assignments = np.arange(num_agents, dtype=np.int64)

        diff = agent_positions[:, None, :] - target_positions[None, :, :]
        cost_matrix = np.sum(diff ** 2, axis=-1).astype(np.float32)

        info = {
            "assigner": "fixed",
            "total_assignment_cost": float(sum(cost_matrix[i, assignments[i]] for i in range(num_agents))),
        }
        return assignments, cost_matrix, info


class CrossTargetAssigner(BaseTargetAssigner):
    """
    Cross assignment baseline: agent i -> target (N-1-i).

    Maps agents to targets in reverse order. For 3 agents:
        agent 0 -> target 2, agent 1 -> target 1, agent 2 -> target 0.

    Useful for testing whether the policy can handle crossed paths.
    """

    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        HungarianTargetAssigner._validate_inputs(agent_positions, target_positions)

        num_agents = agent_positions.shape[0]
        assignments = np.array([num_agents - 1 - i for i in range(num_agents)], dtype=np.int64)

        diff = agent_positions[:, None, :] - target_positions[None, :, :]
        cost_matrix = np.sum(diff ** 2, axis=-1).astype(np.float32)

        info = {
            "assigner": "cross",
            "total_assignment_cost": float(sum(cost_matrix[i, assignments[i]] for i in range(num_agents))),
        }
        return assignments, cost_matrix, info


class CBAATargetAssigner(BaseTargetAssigner):
    """Consensus-Based Auction Algorithm (Choi, Brunet & How 2009, Section III).

    Single-task assignment: each agent is assigned exactly one target.
    Iterates between two phases until a conflict-free assignment is reached.

    Phase 1 - Auction (Algorithm 1):
        Each *unassigned* agent picks the task with the highest bid that
        exceeds the current winning bid known to that agent:
            h_ij = I(c_ij > y_ij)          # Eq (2): valid tasks
            J_i  = argmax_j  h_ij * c_ij    # best valid task
        Then sets x_i,J_i = 1 and y_i,J_i = c_i,J_i.

    Phase 2 - Consensus (Algorithm 2):
        (a) Max-consensus: every agent replaces its y vector with the
            element-wise maximum over its neighbours.
        (b) Winner check: for each assigned agent, if a neighbour has a
            strictly higher bid on the agent's task (or an equal bid from
            a lower-ID agent), the agent releases the task.

    State per agent i:
        x[i]  - binary vector [N_t];  x[i][j] = 1  iff agent i holds task j
        y[i]  - float vector  [N_t];  y[i][j] = highest bid agent i knows for task j

    Convergence: Theorem 1 guarantees a conflict-free assignment in at most
    N_min * D iterations on a connected static graph with DMG scoring.
    """

    def __init__(self, max_iterations: int = 100, squared_distance: bool = True):
        self.max_iterations = max_iterations
        self.squared_distance = squared_distance

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        HungarianTargetAssigner._validate_inputs(agent_positions, target_positions)

        num_agents = agent_positions.shape[0]
        num_targets = target_positions.shape[0]

        # -- bid matrix  c[i][j] >= 0 ---------------------------------
        diff = agent_positions[:, None, :] - target_positions[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        max_dist_sq = float(np.max(dist_sq))
        if self.squared_distance:
            bid = (max_dist_sq - dist_sq).astype(np.float32)
        else:
            bid = (max_dist_sq - np.sqrt(dist_sq + 1e-8)).astype(np.float32)

        # -- communication graph --------------------------------------
        if ("communication_graph" in kwargs
                and kwargs["communication_graph"] is not None):
            adj = np.asarray(kwargs["communication_graph"], dtype=bool)
        else:
            adj = np.ones((num_agents, num_agents), dtype=bool)
            np.fill_diagonal(adj, False)

        # -- initialise state -----------------------------------------
        x = np.zeros((num_agents, num_targets), dtype=bool)     # x[i]
        y = np.zeros((num_agents, num_targets), dtype=np.float32)  # y[i]

        # -- Phase 1 / Phase 2 iteration ------------------------------
        for iteration in range(self.max_iterations):

            # ==========================================================
            # Phase 1 - Auction (Algorithm 1)
            # ==========================================================
            # Only *unassigned* agents place bids.
            for i in range(num_agents):
                if np.any(x[i]):            # sum_j x_ij != 0  ->  skip
                    continue

                # h_ij = I(c_ij > y_ij)                               Eq (2)
                h = bid[i] > y[i]
                if not np.any(h):
                    continue                # no task with bid > known max

                # J_i = argmax_j  h_ij * c_ij
                masked = np.where(h, bid[i], -np.inf)
                J_i = int(np.argmax(masked))

                # place bid
                x[i, J_i] = True
                y[i, J_i] = bid[i, J_i]

            # ==========================================================
            # Phase 2 - Consensus (Algorithm 2)
            # ==========================================================

            # Snapshot pre-consensus state so that max-consensus and
            # winner check use the same synchronised values.
            y_snap = y.copy()
            x_snap = x.copy()

            # --- Line 4: max-consensus ---------------------------------
            for i in range(num_agents):
                nb = np.where(adj[i])[0]
                if len(nb) > 0:
                    y[i] = np.maximum(y_snap[i], y_snap[nb].max(axis=0))
                else:
                    y[i] = y_snap[i].copy()

            # --- Lines 5-8: winner check -------------------------------
            for i in range(num_agents):
                if not np.any(x[i]):        # unassigned -> skip
                    continue

                J_i = int(np.argmax(x[i]))  # task held by agent i

                # z_i,J_i = argmax_{k : g_ik=1 or k==i}  y_snap[k][J_i]
                #
                # Two rules:
                # 1. Strictly higher bid → always wins (lets max-consensus
                #    propagated values resolve non-neighbour conflicts).
                # 2. Equal bid → prefer the agent that actually *holds*
                #    the task.  Between two holders, lower ID wins.
                best_agent = i
                best_bid = y_snap[i, J_i]
                for k in range(num_agents):
                    if k == i or not adj[i, k]:
                        continue
                    val = y_snap[k, J_i]
                    if val > best_bid:
                        best_bid = val
                        best_agent = k
                    elif val == best_bid:
                        k_holds = x_snap[k, J_i]
                        best_holds = x_snap[best_agent, J_i]
                        if k_holds and not best_holds:
                            best_agent = k
                        elif k_holds == best_holds and k < best_agent:
                            best_agent = k

                if best_agent != i:
                    # outbid - release task.
                    # y is NOT cleared: max-consensus has already set y[i][J_i]
                    # to the global winning bid, which correctly prevents
                    # this agent from re-bidding on the same task next round.
                    x[i, :] = False

            # ==========================================================
            # Convergence check
            # ==========================================================
            assigned = np.any(x, axis=1)
            if not np.all(assigned):
                continue

            # No task may be assigned to more than one agent.
            if np.any(np.sum(x, axis=0) > 1):
                continue

            break   # converged

        # -- build output assignments ---------------------------------
        assignments = np.full(num_agents, -1, dtype=np.int64)
        for i in range(num_agents):
            if np.any(x[i]):
                assignments[i] = int(np.argmax(x[i]))

        if np.any(assignments < 0):
            unassigned = [i for i in range(num_agents) if assignments[i] < 0]
            raise RuntimeError(
                f"CBAA did not converge within {self.max_iterations} iterations. "
                f"Unassigned agents: {unassigned}. "
                f"Increase max_iterations or check graph connectivity."
            )

        if np.any(np.bincount(assignments[assignments >= 0]) > 1):
            raise RuntimeError(
                "CBAA converged with task conflicts - this should not happen."
            )

        # -- diagnostics ----------------------------------------------
        cost_matrix = dist_sq.astype(np.float32)
        total_cost = float(sum(cost_matrix[i, assignments[i]]
                               for i in range(num_agents)))
        info = {
            "assigner": "cbaa",
            "total_assignment_cost": total_cost,
            "iterations": iteration + 1,
            "converged": True,
        }
        return assignments, cost_matrix, info


def build_assigner(name: str, **kwargs) -> BaseTargetAssigner:
    """
    Factory function for convenient config-based construction.

    Example:
        assigner = build_assigner("hungarian")
        assigner = build_assigner("greedy")
        assigner = build_assigner("cbaa")
    """
    name = name.lower()
    if name in {"hungarian", "hungarian_target", "min_cost"}:
        return HungarianTargetAssigner(**kwargs)
    if name in {"greedy", "greedy_nearest"}:
        return GreedyNearestTargetAssigner(**kwargs)
    if name in {"fixed", "identity"}:
        return FixedTargetAssigner()
    if name in {"cross", "reverse"}:
        return CrossTargetAssigner()
    if name in {"cbaa", "cbba", "consensus_bundle", "auction"}:
        return CBAATargetAssigner(**kwargs)
    raise ValueError(f"Unknown assigner name: {name}")


if __name__ == "__main__":
    agents = np.array([
        [-8.0, -3.0],
        [-8.0, 0.0],
        [-8.0, 3.0],
    ], dtype=np.float32)

    targets = np.array([
        [8.0, -4.0],
        [8.0, 0.0],
        [8.0, 4.0],
    ], dtype=np.float32)

    for method in ["fixed", "greedy", "hungarian", "cbaa"]:
        assigner = build_assigner(method)
        assignments, cost_matrix, info = assigner.assign(agents, targets)
        print(f"\nMethod: {method}")
        print("Assignments:", assignments)
        print("Cost matrix:\n", cost_matrix)
        print("Info:", info)
