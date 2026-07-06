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
        bid_strategy: "distance_reward" keeps the original distance-reward bid.
            "opportunity_auction" uses an auction-style opportunity-cost bid.
        auction_epsilon: Small positive increment for opportunity auction bids.
            If None, use a scale-aware default from max_dist_sq at assign() time.
    """
    L_t: int = 1
    max_iterations: int = 50
    use_timestamps: bool = False
    communication_graph: Optional[np.ndarray] = None
    squared_distance: bool = True
    max_dist_sq: Optional[float] = None
    bid_strategy: str = "distance_reward"
    auction_epsilon: Optional[float] = 1e-3


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

    def _single_task_utility(self, task_j: int) -> float:
        return self._path_score([task_j])

    def _effective_auction_epsilon(self) -> float:
        if self.cfg.auction_epsilon is not None:
            return float(self.cfg.auction_epsilon)

        reward_scale = max(float(self.max_dist_sq or 0.0), 1.0)
        return max(1e-3, 1e-3 * reward_scale)

    def _opportunity_auction_bid(self, task_j: int) -> Tuple[float, int]:
        """Auction bid using opportunity cost against the next-best target.

        y[j] acts as the current target price in this mode.  An agent bids only
        on its best net-value target, raising that target's price by the gap to
        its second-best option plus epsilon.
        """
        utilities = np.array(
            [self._single_task_utility(j) for j in range(self.num_targets)],
            dtype=np.float32,
        )
        net_values = utilities - self.y
        order = sorted(
            range(self.num_targets),
            key=lambda j: (-float(net_values[j]), int(j)),
        )
        if not order or task_j != order[0]:
            return -float("inf"), 0

        best_net = float(net_values[order[0]])
        second_net = float(net_values[order[1]]) if len(order) > 1 else best_net
        bid_increment = max(best_net - second_net, 0.0) + self._effective_auction_epsilon()
        return float(self.y[task_j]) + bid_increment, 0

    def _candidate_bid(self, task_j: int) -> Tuple[float, int]:
        if self.cfg.bid_strategy == "distance_reward":
            return self._marginal_gain(task_j)
        if self.cfg.bid_strategy == "opportunity_auction":
            return self._opportunity_auction_bid(task_j)
        raise ValueError(f"Unknown CBBA bid_strategy: {self.cfg.bid_strategy}")

    @staticmethod
    def _bid_wins(
        candidate_bid: float,
        candidate_agent: int,
        incumbent_bid: float,
        incumbent_agent: int,
        eps: float = 1e-6,
    ) -> bool:
        """Return True when a candidate bid should replace the incumbent."""
        if candidate_agent < 0:
            return False
        if incumbent_agent < 0:
            return True
        if candidate_bid > incumbent_bid + eps:
            return True
        if abs(candidate_bid - incumbent_bid) <= eps:
            return candidate_agent < incumbent_agent
        return False

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
                gain, insert_pos = self._candidate_bid(j)

                # Equation (2): h_ij = I(c_ij > y_ij), with deterministic
                # tie-breaking so equal bids do not keep conflicting forever.
                if gain < -1e-6 or not self._bid_wins(
                    gain,
                    self.agent_id,
                    float(self.y[j]),
                    int(self.z[j]),
                ):
                    continue

                if (
                    gain > best_gain + 1e-6
                    or (abs(gain - best_gain) <= 1e-6 and (best_task < 0 or j < best_task))
                ):
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
        if self.cfg.bid_strategy not in {"distance_reward", "opportunity_auction"}:
            raise ValueError(f"Unknown CBBA bid_strategy: {self.cfg.bid_strategy}")
        if self.cfg.auction_epsilon is not None and self.cfg.auction_epsilon <= 0:
            raise ValueError("auction_epsilon must be positive or None for adaptive scaling.")
        if self.cfg.bid_strategy == "opportunity_auction" and (
            self.cfg.L_t != 1 or self.cfg.use_timestamps
        ):
            raise ValueError(
                "opportunity_auction currently supports only L_t=1 and use_timestamps=False."
            )

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
            adj = np.asarray(self.cfg.communication_graph, dtype=bool)
        elif "communication_graph" in kwargs and kwargs["communication_graph"] is not None:
            adj = np.asarray(kwargs["communication_graph"], dtype=bool)
        else:
            adj = np.ones((num_agents, num_agents), dtype=bool)
            np.fill_diagonal(adj, False)
        if adj.shape != (num_agents, num_agents):
            raise ValueError(
                f"communication_graph must have shape {(num_agents, num_agents)}, got {adj.shape}."
            )
        np.fill_diagonal(adj, False)
        comm_connected = self._is_connected(adj)

        # 2. Iterate Phase 1 <-> Phase 2 until convergence
        iteration = 0
        for iteration in range(self.cfg.max_iterations):
            # Phase 1: Bundle construction
            bundle_changed = False
            for agent in agents:
                bundle_changed = agent.build_bundle() or bundle_changed

            if self.cfg.L_t == 1 and not self.cfg.use_timestamps:
                # CBAA Phase 2 — Algorithm 2 from Choi et al. (2009).
                #
                # Line 3-4: y_ij = max_{k: adj[i,k] or k==i} y_kj   ∀j
                # Line 5-8: z_i,J_i = argmax_k (y_k,J_i); release if z_i ≠ i.
                #
                # Snapshot all y vectors first so that the winner check uses
                # Phase-1 values, not values already updated by max-consensus.

                y_snapshot = [a.y.copy() for a in agents]
                z_snapshot = [a.z.copy() for a in agents]
                prev_bundles = [list(a.b) for a in agents]
                consensus_changed = False

                for i in range(num_agents):
                    for j in range(num_targets):
                        best_bid = float(y_snapshot[i][j])
                        best_agent = int(z_snapshot[i][j])
                        for k in range(num_agents):
                            if not adj[i, k]:
                                continue
                            candidate_bid = float(y_snapshot[k][j])
                            candidate_agent = int(z_snapshot[k][j])
                            if CBBAAgent._bid_wins(
                                candidate_bid,
                                candidate_agent,
                                best_bid,
                                best_agent,
                            ):
                                best_bid = candidate_bid
                                best_agent = candidate_agent
                        agents[i].y[j] = best_bid
                        agents[i].z[j] = best_agent
                        if (
                            abs(best_bid - float(y_snapshot[i][j])) > 1e-6
                            or best_agent != int(z_snapshot[i][j])
                        ):
                            consensus_changed = True

                released = False
                for i in range(num_agents):
                    if not agents[i].b:
                        continue
                    task_j = agents[i].b[0]
                    if agents[i].z[task_j] != i:
                        agents[i].b = []
                        agents[i].p = []
                        agents[i].total_score = 0.0
                        released = True

                # Convergence or early exit.
                if self._is_converged_cbaa(agents, num_agents):
                    break
                # Early exit after a stationary iteration: no agent released
                # and no new bids placed — the assignment is stable.  The
                # greedy fallback at the output stage handles remaining gaps.
                bundles_unchanged = all(
                    list(a.b) == prev_bundles[idx] for idx, a in enumerate(agents)
                )
                if bundles_unchanged and not bundle_changed and not released and not consensus_changed:
                    break
            else:
                # CBBA Phase 2 — Table I pairwise consensus.
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

        # 3. Build cost matrix
        diff = agent_positions[:, None, :] - target_positions[None, :, :]
        cost_matrix = np.sum(diff ** 2, axis=-1).astype(np.float32)

        algorithm_mode = (
            "cbaa_single_task"
            if self.cfg.L_t == 1 and not self.cfg.use_timestamps
            else "cbba_bundle"
        )
        converged = (
            self._is_converged_cbaa(agents, num_agents)
            if algorithm_mode == "cbaa_single_task"
            else self._is_converged(agents, num_agents)
        )

        # 4. Build output assignments
        assignments = np.full(num_agents, -1, dtype=np.int64)
        for i, agent in enumerate(agents):
            if agent.b:
                assignments[i] = agent.b[0]

        # Resolve remaining conflicts / unassigned agents greedily.
        assigned_mask = assignments >= 0
        _, counts = np.unique(assignments[assigned_mask], return_counts=True)
        has_conflict = bool(np.any(counts > 1))
        has_unassigned = bool(np.any(~assigned_mask))
        used_fallback = has_conflict or has_unassigned
        fallback_reason = None
        if used_fallback:
            fallback_reason = "conflict" if has_conflict else "unassigned"
            claimed = set()
            clean = np.full(num_agents, -1, dtype=np.int64)
            order = np.argsort(cost_matrix.min(axis=1))
            for i in order:
                available = [j for j in range(num_targets) if j not in claimed]
                if not available:
                    break
                best_j = available[int(np.argmin(cost_matrix[i, available]))]
                clean[i] = best_j
                claimed.add(best_j)
            assignments = clean

        if np.any(assignments < 0):
            raise RuntimeError(
                f"CBBA/CBAA failed to assign all agents. "
                f"Assigned: {assignments.tolist()}, "
                f"converged at iteration {iteration + 1}."
            )

        total_cost = float(sum(cost_matrix[i, assignments[i]] for i in range(num_agents)))
        effective_auction_epsilon = (
            agents[0]._effective_auction_epsilon()
            if self.cfg.bid_strategy == "opportunity_auction"
            else self.cfg.auction_epsilon
        )
        info = {
            "assigner": (
                "cbba_auction"
                if self.cfg.bid_strategy == "opportunity_auction"
                else "cbba"
            ),
            "algorithm_mode": algorithm_mode,
            "bid_strategy": self.cfg.bid_strategy,
            "auction_epsilon": (
                None
                if effective_auction_epsilon is None
                else float(effective_auction_epsilon)
            ),
            "auction_epsilon_adaptive": bool(
                self.cfg.bid_strategy == "opportunity_auction"
                and self.cfg.auction_epsilon is None
            ),
            "total_assignment_cost": total_cost,
            "iterations": iteration + 1,
            "converged": bool(converged and comm_connected),
            "comm_connected": bool(comm_connected),
            "used_fallback": bool(used_fallback),
            "fallback_reason": fallback_reason,
        }
        return assignments, cost_matrix, info

    @staticmethod
    def _is_connected(adj: np.ndarray) -> bool:
        n = adj.shape[0]
        if n <= 1:
            return True
        seen = {0}
        stack = [0]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(adj[i]):
                j = int(j)
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return len(seen) == n

    @staticmethod
    def _is_converged(agents: list[CBBAAgent], num_agents: int) -> bool:
        """Check convergence (CBBA): all z vectors agree and all agents assigned."""
        if not agents:
            return False
        for j in range(agents[0].num_targets):
            winner = agents[0].z[j]
            for agent in agents[1:]:
                if agent.z[j] != winner:
                    return False

        assigned_count = sum(1 for a in agents if len(a.b) > 0)
        return assigned_count >= num_agents

    @staticmethod
    def _is_converged_cbaa(agents: list[CBBAAgent], num_agents: int) -> bool:
        """Check convergence (CBAA): all agents assigned and no task conflicts."""
        if not agents:
            return False
        # Every agent must hold exactly one task (L_t = 1).
        if any(len(a.b) != 1 for a in agents):
            return False
        # No two agents may claim the same task.
        claimed = set()
        for i, a in enumerate(agents):
            task_j = a.b[0]
            if task_j in claimed:
                return False
            if a.z[task_j] != i:
                return False
            claimed.add(task_j)
        return True



@dataclass
class DistributedADMMConfig:
    """Configuration for DistributedADMMTargetAssigner.

    This assigner is intended for the N-UAV / N-target case.  It first runs a
    decentralized consensus-ADMM style continuous relaxation over local copies
    of the global assignment matrix, then converts the relaxed matrix into a
    discrete one-to-one assignment using distributed conflict resolution.

    Args:
        max_iterations: Number of inner ADMM message-passing iterations per
            environment reassignment.
        rho: ADMM penalty parameter. Larger values emphasize neighbor
            consistency; smaller values emphasize the local distance cost.
        sinkhorn_iterations: Number of row/column normalization steps used to
            project a matrix to the doubly stochastic relaxation.
        temperature: Softness of the Sinkhorn projection. Smaller values make
            the relaxed matrix closer to a permutation matrix, but may be less
            stable.
        conflict_rounds: Maximum number of winner/loser conflict-resolution
            rounds after ADMM.
        squared_distance: Whether to use squared Euclidean distance as the cost.
        use_warm_start: Reuse the previous relaxed assignment matrix when the
            environment calls assign() again. This is useful for dynamic targets.
        convergence_tol: Stop ADMM when local copies change by less than this value.
        allow_global_safety_repair: If True, repair invalid/disconnected results
            with a centralized safety pass so the simulator state remains valid.
        eps: Numerical stability constant.
    """

    max_iterations: int = 30
    rho: float = 2.0
    sinkhorn_iterations: int = 30
    temperature: float = 0.2
    conflict_rounds: int = 20
    squared_distance: bool = True
    use_warm_start: bool = True
    convergence_tol: float = 1e-4
    allow_global_safety_repair: bool = True
    eps: float = 1e-8


class DistributedADMMTargetAssigner(BaseTargetAssigner):
    """Distributed ADMM + distributed conflict resolution target assigner.

    The environment calls this class exactly like the existing Hungarian / CBBA
    assigners.  Internally, the method emulates one communication period of a
    decentralized algorithm:

    1. Each UAV i owns a local copy X_i of the full assignment matrix X.
    2. UAV i only puts its own cost row C[i, :] into the local objective.
    3. Neighboring UAVs enforce X_i = X_k with consensus-ADMM edge variables.
    4. The averaged relaxed assignment matrix is discretized by local target
       selection plus iterative conflict resolution.

    Notes:
        - This is a practical simulation implementation.  It uses the provided
          communication_graph to restrict ADMM message passing.
        - If the communication graph is disconnected, no purely local method can
          guarantee global one-to-one assignment across disconnected components.
          In that case, the final safety repair keeps the environment valid and
          reports used_global_repair=True in info.
    """

    def __init__(self, config: Optional[DistributedADMMConfig] = None, **kwargs):
        self.cfg = config or DistributedADMMConfig(**kwargs)
        self._warm_X: Optional[np.ndarray] = None

    def assign(
        self,
        agent_positions: np.ndarray,
        target_positions: np.ndarray,
        **kwargs,
    ) -> AssignmentResult:
        HungarianTargetAssigner._validate_inputs(agent_positions, target_positions)

        num_agents = agent_positions.shape[0]
        num_targets = target_positions.shape[0]
        if num_agents != num_targets:
            raise ValueError(
                "DistributedADMMTargetAssigner currently assumes N agents and N targets. "
                f"Got num_agents={num_agents}, num_targets={num_targets}."
            )

        cost_matrix = self._build_cost_matrix(agent_positions, target_positions)
        cost_norm = self._normalize_cost(cost_matrix)

        adj = self._build_adjacency(num_agents, kwargs.get("communication_graph", None))
        connected = self._is_connected(adj)

        relaxed_matrix, residual_trace, admm_converged = self._run_consensus_admm(cost_norm, adj)
        assignments, conflict_info = self._distributed_conflict_resolution(
            relaxed_matrix=relaxed_matrix,
            cost_matrix=cost_matrix,
            adj=adj,
        )

        distributed_assignment_valid = self._is_valid_assignment(assignments, num_agents)
        used_global_safety_repair = False
        repair_reason = None
        resolution_scope = "connected_graph" if connected else "component_local"
        if not connected:
            repair_reason = "disconnected_graph"
        elif not distributed_assignment_valid:
            repair_reason = "conflict_resolution_failed"

        if repair_reason is not None:
            if not self.cfg.allow_global_safety_repair:
                raise RuntimeError(
                    "Distributed ADMM could not produce a guaranteed global one-to-one "
                    f"assignment without safety repair. reason={repair_reason}."
                )
            assignments = self._global_safety_repair(relaxed_matrix, cost_matrix)
            used_global_safety_repair = True
            resolution_scope = "global_safety"

        if self.cfg.use_warm_start:
            # Store a permutation-like warm start matching the executable result.
            self._warm_X = np.zeros((num_agents, num_targets), dtype=np.float32)
            self._warm_X[np.arange(num_agents), assignments] = 1.0

        total_cost = float(sum(cost_matrix[i, assignments[i]] for i in range(num_agents)))
        info = {
            "assigner": "distributed_admm_conflict",
            "algorithm_mode": "distributed_admm_relaxation",
            "total_assignment_cost": total_cost,
            "admm_iterations": self.cfg.max_iterations,
            "admm_iterations_actual": len(residual_trace),
            "admm_converged": bool(admm_converged),
            "rho": float(self.cfg.rho),
            "temperature": float(self.cfg.temperature),
            "comm_connected": bool(connected),
            "used_distributed_conflict_resolution": True,
            "used_global_repair": bool(used_global_safety_repair),
            "used_global_safety_repair": bool(used_global_safety_repair),
            "repair_reason": repair_reason,
            "resolution_scope": resolution_scope,
            "residual_last": float(residual_trace[-1]) if residual_trace else 0.0,
            "residual_trace": residual_trace,
            **conflict_info,
        }
        return assignments.astype(np.int64), cost_matrix.astype(np.float32), info

    def _build_cost_matrix(self, agent_positions: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
        diff = agent_positions[:, None, :] - target_positions[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        if self.cfg.squared_distance:
            return dist_sq.astype(np.float32)
        return np.sqrt(dist_sq + self.cfg.eps).astype(np.float32)

    def _normalize_cost(self, cost_matrix: np.ndarray) -> np.ndarray:
        c_min = float(np.min(cost_matrix))
        c_max = float(np.max(cost_matrix))
        return ((cost_matrix - c_min) / (c_max - c_min + self.cfg.eps)).astype(np.float32)

    @staticmethod
    def _build_adjacency(num_agents: int, communication_graph: Optional[np.ndarray]) -> np.ndarray:
        if communication_graph is None:
            adj = np.ones((num_agents, num_agents), dtype=bool)
            np.fill_diagonal(adj, False)
            return adj
        adj = np.asarray(communication_graph, dtype=bool).copy()
        if adj.shape != (num_agents, num_agents):
            raise ValueError(
                f"communication_graph must have shape {(num_agents, num_agents)}, got {adj.shape}."
            )
        np.fill_diagonal(adj, False)
        # Treat the communication link as undirected for pairwise ADMM exchange.
        return adj | adj.T

    @staticmethod
    def _is_connected(adj: np.ndarray) -> bool:
        n = adj.shape[0]
        if n <= 1:
            return True
        seen = {0}
        stack = [0]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(adj[i]):
                if int(j) not in seen:
                    seen.add(int(j))
                    stack.append(int(j))
        return len(seen) == n

    @staticmethod
    def _connected_components(adj: np.ndarray) -> list[list[int]]:
        n = adj.shape[0]
        unseen = set(range(n))
        components: list[list[int]] = []
        while unseen:
            start = unseen.pop()
            component = [start]
            stack = [start]
            while stack:
                i = stack.pop()
                for j in np.flatnonzero(adj[i]):
                    j = int(j)
                    if j in unseen:
                        unseen.remove(j)
                        component.append(j)
                        stack.append(j)
            components.append(sorted(component))
        return components

    @staticmethod
    def _components_are_locally_valid(assignments: np.ndarray, components: list[list[int]]) -> bool:
        for component in components:
            local_targets = [int(assignments[i]) for i in component]
            if len(set(local_targets)) != len(local_targets):
                return False
        return True

    def _initial_assignment_matrix(self, num_agents: int, num_targets: int) -> np.ndarray:
        if (
            self.cfg.use_warm_start
            and self._warm_X is not None
            and self._warm_X.shape == (num_agents, num_targets)
        ):
            return self._warm_X.astype(np.float32).copy()
        return np.full((num_agents, num_targets), 1.0 / num_targets, dtype=np.float32)

    def _run_consensus_admm(self, cost_norm: np.ndarray, adj: np.ndarray) -> Tuple[np.ndarray, list[float], bool]:
        n, m = cost_norm.shape
        X = np.stack([self._initial_assignment_matrix(n, m) for _ in range(n)], axis=0)

        # Directed edge dual variables U[i, k] are only active when adj[i, k] is true.
        U = np.zeros((n, n, n, m), dtype=np.float32)
        Z = np.zeros((n, n, n, m), dtype=np.float32)
        for i in range(n):
            for k in np.flatnonzero(adj[i]):
                Z[i, k] = 0.5 * (X[i] + X[int(k)])

        residual_trace: list[float] = []
        rho = max(float(self.cfg.rho), self.cfg.eps)
        converged = False

        for _ in range(self.cfg.max_iterations):
            X_prev = X.copy()

            # X-update: each agent solves an approximate local ADMM subproblem.
            for i in range(n):
                neighbors = np.flatnonzero(adj[i])
                grad = np.zeros((n, m), dtype=np.float32)
                grad[i, :] = cost_norm[i, :]

                if len(neighbors) == 0:
                    center = X[i]
                    scale = rho
                else:
                    center = np.mean([Z[i, int(k)] - U[i, int(k)] for k in neighbors], axis=0)
                    scale = rho * float(len(neighbors))

                score = center - grad / max(scale, self.cfg.eps)
                X[i] = self._sinkhorn_project(score)

            # Z-update: pairwise consensus between neighbors.
            for i in range(n):
                for k in np.flatnonzero(adj[i]):
                    k = int(k)
                    if i < k:
                        z = 0.5 * (X[i] + U[i, k] + X[k] + U[k, i])
                        Z[i, k] = z
                        Z[k, i] = z

            # U-update: scaled dual ascent.
            primal_residual = 0.0
            edge_count = 0
            for i in range(n):
                for k in np.flatnonzero(adj[i]):
                    k = int(k)
                    U[i, k] += X[i] - Z[i, k]
                    primal_residual += float(np.linalg.norm(X[i] - Z[i, k]))
                    edge_count += 1

            residual_trace.append(primal_residual / max(edge_count, 1))

            if np.linalg.norm(X - X_prev) < self.cfg.convergence_tol:
                converged = True
                break

        return np.mean(X, axis=0).astype(np.float32), residual_trace, converged

    def _sinkhorn_project(self, score: np.ndarray) -> np.ndarray:
        """Map arbitrary scores to the doubly stochastic relaxation."""
        temp = max(float(self.cfg.temperature), self.cfg.eps)
        s = score / temp
        s = s - np.max(s)
        mat = np.exp(s).astype(np.float32) + self.cfg.eps

        for _ in range(self.cfg.sinkhorn_iterations):
            mat /= np.sum(mat, axis=1, keepdims=True) + self.cfg.eps
            mat /= np.sum(mat, axis=0, keepdims=True) + self.cfg.eps

        return mat.astype(np.float32)

    def _distributed_conflict_resolution(
        self,
        relaxed_matrix: np.ndarray,
        cost_matrix: np.ndarray,
        adj: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        n, m = relaxed_matrix.shape
        preference_order = np.argsort(-relaxed_matrix, axis=1)
        rank_ptr = np.zeros(n, dtype=np.int64)
        assignments = preference_order[:, 0].astype(np.int64)

        num_conflicts = 0
        rounds_used = 0
        components = self._connected_components(adj)
        for r in range(self.cfg.conflict_rounds):
            rounds_used = r + 1
            changed = False
            for component in components:
                component_set = set(component)
                for target_j in range(m):
                    claimers = [
                        int(i)
                        for i in component
                        if int(assignments[int(i)]) == target_j
                    ]
                    if len(claimers) <= 1:
                        continue

                    num_conflicts += 1
                    # Winner rule is evaluated only inside the communication
                    # component, avoiding cross-component global knowledge.
                    winner = min(
                        claimers,
                        key=lambda i: (
                            float(cost_matrix[i, target_j]),
                            -float(relaxed_matrix[i, target_j]),
                            int(i),
                        ),
                    )

                    for loser in claimers:
                        if loser == winner:
                            continue
                        rank_ptr[loser] += 1
                        while rank_ptr[loser] < m:
                            candidate = int(preference_order[loser, rank_ptr[loser]])
                            component_claimers = [
                                int(i)
                                for i in component_set
                                if int(assignments[int(i)]) == candidate
                                and int(i) != loser
                            ]
                            assignments[loser] = candidate
                            changed = True
                            if not component_claimers:
                                break
                            break

            if not changed or self._components_are_locally_valid(assignments, components):
                break

        return assignments.astype(np.int64), {
            "conflict_rounds_used": int(rounds_used),
            "num_conflict_events": int(num_conflicts),
            "num_communication_components": int(len(components)),
        }

    @staticmethod
    def _is_valid_assignment(assignments: np.ndarray, num_agents: int) -> bool:
        if assignments.shape != (num_agents,):
            return False
        if np.any(assignments < 0):
            return False
        return len(set(assignments.tolist())) == num_agents

    def _global_safety_repair(self, relaxed_matrix: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
        """Deterministic final repair to keep the simulator state valid.

        This is not the conceptual distributed step. It is only used when the
        communication graph is disconnected or conflict rounds ended before a
        valid one-to-one assignment was obtained.
        """
        n, m = cost_matrix.shape
        assignments = np.full(n, -1, dtype=np.int64)
        used_targets: set[int] = set()

        # Agents with a confident relaxed row choose first.
        confidence = np.max(relaxed_matrix, axis=1)
        order = np.argsort(-confidence)
        for i in order:
            i = int(i)
            for target_j in np.argsort(-relaxed_matrix[i]):
                target_j = int(target_j)
                if target_j not in used_targets:
                    assignments[i] = target_j
                    used_targets.add(target_j)
                    break

        # Any remaining gaps are filled by minimum distance.
        for i in range(n):
            if assignments[i] >= 0:
                continue
            available = [j for j in range(m) if j not in used_targets]
            if not available:
                break
            best_j = available[int(np.argmin(cost_matrix[i, available]))]
            assignments[i] = int(best_j)
            used_targets.add(int(best_j))

        return assignments.astype(np.int64)

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
    if name in {"cbba_auction", "opportunity_auction", "cbba_opportunity"}:
        cfg_kwargs = dict(kwargs)
        cfg_kwargs.setdefault("bid_strategy", "opportunity_auction")
        cfg_kwargs.setdefault("auction_epsilon", None)
        cfg_kwargs.setdefault("max_iterations", 500)
        return CBBATargetAssigner(**cfg_kwargs)
    if name in {"admm", "distributed_admm", "admm_conflict"}:
        return DistributedADMMTargetAssigner(**kwargs)
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

    for method in ["fixed", "greedy", "hungarian", "cbba", "cbba_auction", "admm"]:
        assigner = build_assigner(method)
        assignments, cost_matrix, info = assigner.assign(agents, targets)
        print(f"\nMethod: {method}")
        print("Assignments:", assignments)
        print("Cost matrix:\n", cost_matrix)
        print("Info:", info)
