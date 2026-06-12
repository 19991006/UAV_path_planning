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
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError(
        "HungarianTargetAssigner requires scipy. Install it with: pip install scipy"
    ) from exc


AssignmentResult = Tuple[np.ndarray, np.ndarray, Dict]


@dataclass
class CBBAConfig:
    """Configuration for CBBATargetAssigner.

    Args:
        L_t: Maximum number of tasks each agent can hold (default 1 = single-assignment).
        max_iterations: Maximum Phase-1/Phase-2 alternations before forced return.
        use_timestamps: If True, use full Table I with timestamp-based freshness checks.
        communication_graph: (N_u, N_u) adjacency matrix. None means fully connected.
        squared_distance: Use squared Euclidean distance for score computation.
        max_dist_sq: Maximum squared distance between any agent-target pair.
            Used to convert distances to non-negative rewards. If None, computed
            automatically from agent/target positions at assign() time.
    """
    L_t: int = 1
    max_iterations: int = 100
    use_timestamps: bool = False
    communication_graph: Optional[np.ndarray] = None
    squared_distance: bool = True
    max_dist_sq: Optional[float] = None


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


class CBBAAgent:
    """Single agent in the CBBA auction.

    Holds the five local state vectors per Choi et al. (2009):
        b: bundle (ordered by time of addition)
        p: path (ordered by execution sequence)
        y: winning bid list (length N_t)
        z: winning agent list (length N_t, -1 = none)
        s: timestamp list (length N_u)
    """

    def __init__(
        self,
        agent_id: int,
        position: np.ndarray,
        target_positions: np.ndarray,
        config: CBBAConfig,
        max_dist_sq: Optional[float] = None,
    ):
        self.agent_id = agent_id
        self.position = position.astype(np.float32)
        self.targets = target_positions.astype(np.float32)
        self.cfg = config
        self.max_dist_sq = max_dist_sq

        self.num_targets = target_positions.shape[0]
        self.num_agents_for_s = 0  # set externally after initialization

        self.b: list[int] = []
        self.p: list[int] = []
        self.y = np.zeros(self.num_targets, dtype=np.float32)
        self.z = np.full(self.num_targets, -1, dtype=np.int64)
        self.s = np.zeros(1, dtype=np.float32)

        self.total_score = 0.0

    @staticmethod
    def _distance_sq(a: np.ndarray, b: np.ndarray) -> float:
        """Squared Euclidean distance between two points."""
        return float(np.sum((a - b) ** 2))

    def _path_score(self, path: list[int]) -> float:
        """S_i^{p_i}: total reward along a path = sum (max_dist_sq - dist_sq).

        Converts distances to non-negative rewards so that CBBA marginal gains
        (Equation 3) are always >= 0 as required by c_ij >= 0 in the paper.
        """
        if not path:
            return 0.0
        score = 0.0
        prev = self.position
        for task_id in path:
            target = self.targets[task_id]
            dist_sq = self._distance_sq(prev, target)
            if self.cfg.squared_distance:
                score += self.max_dist_sq - dist_sq
            else:
                score += self.max_dist_sq - float(np.sqrt(dist_sq + 1e-8))
            prev = target
        return score

    def _marginal_gain(self, task_j: int) -> Tuple[float, int]:
        """Equation (3): max score improvement from inserting task_j into current path.

        Returns:
            (gain, best_insert_position): gain = max_n (S^{p ⊕_n {j}} - S^p),
            best_insert_position is the n that achieves the max.
        """
        current_score = self.total_score
        best_score = -float('inf')
        best_pos = 0

        for n in range(len(self.p) + 1):
            candidate = list(self.p)
            candidate.insert(n, task_j)
            score = self._path_score(candidate)
            if score > best_score:
                best_score = score
                best_pos = n

        gain = best_score - current_score
        return gain, best_pos

    def build_bundle(self) -> bool:
        """Phase 1: Bundle construction (Algorithm 3 in the paper).

        Greedily adds tasks to the bundle until no more profitable tasks exist
        or the bundle reaches L_t capacity.

        Returns:
            True if the bundle changed, False otherwise.
        """
        changed = False

        while len(self.b) < self.cfg.L_t:
            best_gain = -float('inf')
            best_task = -1
            best_pos = -1

            for j in range(self.num_targets):
                if j in self.b:
                    continue
                gain, insert_pos = self._marginal_gain(j)

                # Equation (2): h_ij = I(c_ij > y_ij)
                if gain <= 0.0 or gain <= self.y[j]:
                    continue

                if gain > best_gain:
                    best_gain = gain
                    best_task = j
                    best_pos = insert_pos

            if best_task == -1:
                break

            # Equation (4): update bundle, path, y, z
            self.b.append(best_task)
            self.p.insert(best_pos, best_task)
            self.y[best_task] = best_gain
            self.z[best_task] = self.agent_id
            self.total_score += best_gain
            changed = True

        return changed

    def receive_from(self, sender: "CBBAAgent") -> None:
        """Phase 2: Process a message from a neighboring agent (Table I)."""
        for j in range(self.num_targets):
            z_s = sender.z[j]
            z_r = self.z[j]
            y_s = sender.y[j]
            y_r = self.y[j]

            action = self._resolve_action(z_s, z_r, y_s, y_r, sender)
            self._apply_action(j, action, y_s, z_s)

    def _resolve_action(
        self, z_s: int, z_r: int, y_s: float, y_r: float, sender: "CBBAAgent"
    ) -> str:
        """Determine update/reset/leave per Table I."""
        sid = sender.agent_id
        rid = self.agent_id

        # Case: sender thinks it is the winner
        if z_s == sid:
            if z_r == rid:
                return "update" if y_s > y_r else "leave"
            if z_r == sid:
                return "update"
            if z_r == -1:
                return "update"
            # z_r is third party m
            if self.cfg.use_timestamps:
                if sender.s[z_r] > self.s[z_r] or y_s > y_r:
                    return "update"
                return "leave"
            return "update"

        # Case: sender thinks receiver is the winner
        if z_s == rid:
            if z_r == rid:
                return "leave"
            if z_r == sid:
                return "reset"
            if z_r == -1:
                return "leave"
            # z_r is third party m
            if self.cfg.use_timestamps:
                if sender.s[z_r] > self.s[z_r]:
                    return "reset"
                return "leave"
            return "leave"

        # Case: sender thinks a third party m is the winner
        if z_s != -1 and z_s != sid and z_s != rid:
            if z_r == rid:
                if self.cfg.use_timestamps:
                    if sender.s[z_s] > self.s[z_s] and y_s > y_r:
                        return "update"
                    return "leave"
                return "update" if y_s > y_r else "leave"
            if z_r == sid:
                if self.cfg.use_timestamps:
                    if sender.s[z_s] > self.s[z_s]:
                        return "update"
                    return "reset"
                return "reset"
            if z_r == z_s:
                if self.cfg.use_timestamps:
                    if sender.s[z_s] > self.s[z_s]:
                        return "update"
                    return "leave"
                return "update" if y_s > y_r else "leave"
            if z_r == -1:
                if self.cfg.use_timestamps:
                    if sender.s[z_s] > self.s[z_s]:
                        return "update"
                    return "leave"
                return "update"
            # z_r is different third party n
            if self.cfg.use_timestamps:
                if sender.s[z_s] > self.s[z_s] and sender.s[z_r] > self.s[z_r]:
                    return "update"
                if sender.s[z_s] > self.s[z_s] and y_s > y_r:
                    return "update"
                if sender.s[z_r] > self.s[z_r] and self.s[z_s] > sender.s[z_s]:
                    return "reset"
                return "leave"
            return "update" if y_s > y_r else "leave"

        # Case: sender thinks no one has won the task
        if z_s == -1:
            if z_r == rid:
                return "leave"
            if z_r == sid:
                return "update"
            if z_r == -1:
                return "leave"
            # z_r is third party m
            if self.cfg.use_timestamps:
                if sender.s[z_r] > self.s[z_r]:
                    return "update"
                return "leave"
            return "leave"

        return "leave"

    def _apply_action(self, j: int, action: str, y_s: float, z_s: int) -> None:
        if action == "update":
            self.y[j] = y_s
            self.z[j] = z_s
        elif action == "reset":
            self.y[j] = 0.0
            self.z[j] = -1

    def update_timestamp(self, sender_id: int, time: float) -> None:
        """Record that we received fresh info from sender_id at the given time."""
        if sender_id < len(self.s):
            self.s[sender_id] = max(self.s[sender_id], time)

    def cascade_release(self) -> bool:
        """Equation (6): Release outbid tasks and all tasks added after them.

        Finds the first task in the bundle where z[task] != agent_id.
        Releases that task and all subsequent bundle entries.
        Rebuilds path from the remaining bundle entries.

        Returns:
            True if any tasks were released.
        """
        # Find first outbid position
        n_bar = len(self.b)
        for n, task in enumerate(self.b):
            if self.z[task] != self.agent_id:
                n_bar = n
                break

        if n_bar == len(self.b):
            return False

        # Release tasks from n_bar onward
        for n in range(n_bar, len(self.b)):
            task = self.b[n]
            self.y[task] = 0.0
            self.z[task] = -1

        # Truncate bundle
        self.b = self.b[:n_bar]

        # Rebuild path: keep only tasks still in bundle
        self.p = [t for t in self.p if t in self.b]

        # Recompute total score
        self.total_score = self._path_score(self.p)
        return True


class CBBATargetAssigner(BaseTargetAssigner):
    """Consensus-Based Bundle Algorithm (CBBA) for distributed task assignment.

    Implements the algorithm from Choi, Brunet & How (2009). Internally creates
    one CBBAAgent per UAV and alternates between Phase 1 (bundle construction)
    and Phase 2 (consensus-based conflict resolution) until convergence.

    Args:
        config: CBBAConfig with algorithm parameters.
    """

    def __init__(self, config: Optional[CBBAConfig] = None, **kwargs):
        self.cfg = config or CBBAConfig(**kwargs)

    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        HungarianTargetAssigner._validate_inputs(agent_positions, target_positions)

        num_agents = agent_positions.shape[0]
        num_targets = target_positions.shape[0]

        # Compute max_dist_sq for reward normalization if not provided
        max_dist_sq = self.cfg.max_dist_sq
        if max_dist_sq is None:
            diff = agent_positions[:, None, :] - target_positions[None, :, :]
            max_dist_sq = float(np.max(np.sum(diff ** 2, axis=-1)))

        # 1. Initialize agents
        agents = [
            CBBAAgent(i, agent_positions[i], target_positions, self.cfg, max_dist_sq)
            for i in range(num_agents)
        ]

        # Set timestamp vector sizes
        for agent in agents:
            agent.num_agents_for_s = num_agents
            agent.s = np.zeros(num_agents, dtype=np.float32)

        # Build communication graph: config > kwargs > fully-connected default
        if self.cfg.communication_graph is not None:
            adj = np.asarray(self.cfg.communication_graph)
        elif "communication_graph" in kwargs and kwargs["communication_graph"] is not None:
            adj = np.asarray(kwargs["communication_graph"])
        else:
            adj = np.ones((num_agents, num_agents), dtype=bool)
            np.fill_diagonal(adj, False)

        # 2. Iterate Phase 1 <-> Phase 2 until convergence
        iteration = 0
        for iteration in range(self.cfg.max_iterations):
            # Phase 1: Bundle construction
            for agent in agents:
                agent.build_bundle()

            # Phase 2: Consensus
            for i in range(num_agents):
                for k in range(num_agents):
                    if i == k or not adj[i, k]:
                        continue
                    agents[i].receive_from(agents[k])
                    if self.cfg.use_timestamps:
                        agents[i].update_timestamp(k, float(iteration))

            # Cascading release
            for agent in agents:
                agent.cascade_release()

            # Check convergence
            if self._is_converged(agents, num_agents):
                break

        # 3. Build output
        assignments = np.full(num_agents, -1, dtype=np.int64)
        for i, agent in enumerate(agents):
            if agent.b:
                assignments[i] = agent.b[0]

        # If any agent unassigned, greedy fallback
        if np.any(assignments < 0):
            unassigned = [i for i in range(num_agents) if assignments[i] < 0]
            unused = [j for j in range(num_targets) if j not in assignments]
            for idx, agent_idx in enumerate(unassigned):
                if idx < len(unused):
                    assignments[agent_idx] = unused[idx]

        if np.any(assignments < 0):
            raise RuntimeError(
                f"CBBA failed to assign all agents. "
                f"Assigned: {assignments.tolist()}, "
                f"converged at iteration {iteration + 1}."
            )

        # Build cost matrix for diagnostics
        diff = agent_positions[:, None, :] - target_positions[None, :, :]
        cost_matrix = np.sum(diff ** 2, axis=-1).astype(np.float32)

        total_cost = float(sum(cost_matrix[i, assignments[i]] for i in range(num_agents)))
        info = {
            "assigner": "cbba",
            "total_assignment_cost": total_cost,
            "iterations": iteration + 1,
            "converged": self._is_converged(agents, num_agents),
        }
        return assignments, cost_matrix, info

    @staticmethod
    def _is_converged(agents: list[CBBAAgent], num_agents: int) -> bool:
        """Check convergence: all z vectors agree and all agents assigned."""
        if not agents:
            return False
        for j in range(agents[0].num_targets):
            winner = agents[0].z[j]
            for agent in agents[1:]:
                if agent.z[j] != winner:
                    return False

        assigned_count = sum(1 for a in agents if len(a.b) > 0)
        return assigned_count >= num_agents


def build_assigner(name: str, **kwargs) -> BaseTargetAssigner:
    """
    Factory function for convenient config-based construction.

    Example:
        assigner = build_assigner("hungarian")
        assigner = build_assigner("greedy")
        assigner = build_assigner("fixed")
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
    if name in {"cbba", "consensus_bundle", "auction"}:
        return CBBATargetAssigner(**kwargs)
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

    for method in ["fixed", "greedy", "hungarian", "cbba"]:
        assigner = build_assigner(method)
        assignments, cost_matrix, info = assigner.assign(agents, targets)
        print(f"\nMethod: {method}")
        print("Assignments:", assignments)
        print("Cost matrix:\n", cost_matrix)
        print("Info:", info)
