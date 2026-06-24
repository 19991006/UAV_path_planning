"""
Multi-UAV 2D environment for DA-MAPPO with pluggable online target assignment.

Paper-style observation o_i = [z_i, u_i, g_i, q_i]:
    z_i: normalized 2D LiDAR distances
    u_i: [v_norm, omega_norm, a_v_norm, a_omega_norm]
    g_i: [target_distance_norm, target_bearing_norm]
    q_i: relative teammate positions normalized by world_size

Merged features:
- Pluggable online target assignment (Hungarian / greedy / fixed)
- Dynamic targets: swap, linear motion, or both
- Paper-style reward with hover, boundary, hard+soft LiDAR, arrival decay
- Centralized critic global state with joint obs, cost matrix, obstacle encoding
- Optional freeze-after-arrival (freeze_arrived_uavs config flag)
- 2D LiDAR obstacle sensing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from target_assignment import BaseTargetAssigner, build_assigner
except ImportError as exc:
    raise ImportError(
        "Could not import target_assignment. "
        "Please save the pluggable assignment module as target_assignment.py "
        "or update the import path to assignment.target_assignment."
    ) from exc


@dataclass
class UAVEnvConfig:
    """Configuration for the 2D UAV environment."""

    num_agents: int = 3
    world_size: float = 20.0
    dt: float = 0.1
    max_steps: int = 600

    # Action bounds.
    max_linear_velocity: float = 1.0
    min_linear_velocity: float = -0.3
    max_angular_velocity: float = 1.0

    # Initial UAV layout.
    start_x: float = -8.0
    start_y_gap: float = 4.0

    # Layout mode: "same_side" (all agents left, all targets right) or
    # "cross" (agents and targets interleaved on both sides).
    layout_mode: str = "same_side"

    # Target settings.
    target_x: float = 8.0
    target_y_gap: float = 4.0
    arrival_threshold: float = 0.5

    # Dynamic target settings.
    # target_motion_mode options:
    #   "none"        : static targets;
    #   "swap"        : periodically permute target positions;
    #   "linear"      : targets move with bouncing linear velocities;
    #   "linear_swap" : both linear motion and periodic swapping;
    #   "racetrack"   : stadium-shaped track, slow on left (front) / fast on right (back).
    dynamic_targets: bool = False
    target_motion_mode: str = "none"
    target_swap_interval: int = 100
    target_swap_start_step: int = 100
    target_speed: float = 0.2
    target_area_x_min: float = 6.0
    target_area_x_max: float = 9.0
    target_area_y_min: float = -8.0
    target_area_y_max: float = 8.0

    # Racetrack target motion.
    # Track is a vertical stadium: left side at target_x (front, slow),
    # right side at target_x + 2*R (back, fast), U-turn arcs at top/bottom.
    racetrack_turn_radius: float = 0.5
    racetrack_straight_half_length: float = 9.0
    racetrack_front_speed: float = 0.2
    # Auto-derived in __post_init__ from front_speed + geometry + target_y_gap.
    # Config value is ignored; kept for backward compatibility.
    racetrack_back_speed: float = 2.0

    # Assignment settings. Options: "hungarian", "greedy", "fixed".
    assigner_name: str = "hungarian"
    reassign_interval: int = 10  # steps between full reassignments (1 = every step)

    # LiDAR settings.
    lidar_num_rays: int = 35
    lidar_range: float = 5.0
    lidar_fov: float = 2.0 * np.pi

    # Communication / topology observation.
    communication_range: float = 8.0
    use_communication_range_mask: bool = False

    # Obstacle settings.
    num_obstacles: int = 20
    obstacle_radius_min: float = 0.2
    obstacle_radius_max: float = 0.2
    obstacle_area_x_min: float = -6.0
    obstacle_area_x_max: float = 6.0
    obstacle_area_y_min: float = -10.0
    obstacle_area_y_max: float = 10.0
    # obstacle_area_x_min: float = -4.0
    # obstacle_area_x_max: float = 5.0
    # obstacle_area_y_min: float = -8.0
    # obstacle_area_y_max: float = 8.0
    min_obstacle_spacing: float = 0.8

    # Collision / safety settings.
    uav_radius: float = 0.20
    obstacle_safety_margin: float = 0.05
    inter_agent_min_distance: float = 0.45

    # Arrival behavior.
    freeze_arrived_uavs: bool = False

    # Reward settings.
    progress_scale: float = 10.0
    all_arrived_bonus: float = 100.0
    step_penalty: float = -0.3
    collision_penalty: float = -100.0

    # Centralized critic global state settings.
    critic_include_joint_obs: bool = False
    critic_include_obstacles: bool = True
    critic_include_cost_matrix: bool = False

    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_agents <= 5:
            scale_factor = 1.0
        elif 5 < self.num_agents <= 10:
            scale_factor = 2.0
        else:
            scale_factor = 3.0
        self.world_size *= scale_factor
        self.start_x *= scale_factor
        self.target_x *= scale_factor
        self.obstacle_area_x_min *= scale_factor
        self.obstacle_area_x_max *= scale_factor

        if self.layout_mode == "cross" and self.assigner_name == "cross" and self.num_agents % 2 == 0:
            raise ValueError(
                f"layout_mode='cross' + assigner_name='cross' with even num_agents "
                f"({self.num_agents}) results in same-side assignments. "
                "Use odd num_agents or change one of the two settings."
            )
        self.obstacle_area_y_min *= scale_factor
        self.obstacle_area_y_max *= scale_factor
        self.target_area_x_min *= scale_factor
        self.target_area_x_max *= scale_factor
        self.racetrack_turn_radius *= scale_factor
        self.racetrack_straight_half_length *= scale_factor

        # Auto-derive back speed so that the rearmost target's loop-around time
        # equals the inter-target gap traversal time on the front.
        # v_back = v_front * (2πR + 2L) / target_y_gap
        R = self.racetrack_turn_radius
        L = self.racetrack_straight_half_length
        back_route_len = 2.0 * np.pi * R + 2.0 * L
        self.racetrack_back_speed = self.racetrack_front_speed * back_route_len / self.target_y_gap


class MultiUAV2DEnv:
    """
    Multi-UAV 2D environment with paper-style observations and online assignment.

    Action per UAV:
        [v, omega]

    Observation per UAV:
        o_i = [z_i, u_i, g_i, q_i]

    Specifically:
        z_i: normalized LiDAR distances, shape [D]
        u_i: [v_norm, omega_norm, a_v_norm, a_omega_norm]
        g_i: [target_distance_norm, target_bearing_norm]
        q_i: relative teammate positions normalized by world_size,
             shape [2 * (num_agents - 1)]
    """

    def __init__(
        self,
        config: Optional[UAVEnvConfig] = None,
        target_assigner: Optional[BaseTargetAssigner] = None,
    ):
        self.cfg = config or UAVEnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        if self.cfg.num_agents < 1:
            raise ValueError("num_agents must be >= 1.")

        self.num_agents = self.cfg.num_agents
        self.num_targets = self.cfg.num_agents
        self.action_dim = 2
        self.obs_dim = self.cfg.lidar_num_rays + 4 + 2 + 2 * (self.num_agents - 1)

        # Graph-observation dimensions for GNN MAPPO.
        # node_features_i = [lidar_i, ego_motion_i, assigned_target_i]
        # edge_attr_ij = [dx_ij, dy_ij, distance_ij, bearing_ij]
        self.node_dim = self.cfg.lidar_num_rays + 4 + 2
        self.edge_dim = 4

        self.target_assigner = target_assigner or build_assigner(self.cfg.assigner_name)
        self.assignment_info: Dict = {}

        self.positions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.headings = np.zeros((self.num_agents,), dtype=np.float32)
        self.linear_velocities = np.zeros((self.num_agents,), dtype=np.float32)
        self.angular_velocities = np.zeros((self.num_agents,), dtype=np.float32)
        self.previous_linear_velocities = np.zeros((self.num_agents,), dtype=np.float32)
        self.previous_angular_velocities = np.zeros((self.num_agents,), dtype=np.float32)
        self.linear_accelerations = np.zeros((self.num_agents,), dtype=np.float32)
        self.angular_accelerations = np.zeros((self.num_agents,), dtype=np.float32)

        self.target_positions = np.zeros((self.num_targets, 2), dtype=np.float32)
        self.target_velocities = np.zeros((self.num_targets, 2), dtype=np.float32)
        self.assignments = np.arange(self.num_agents, dtype=np.int64)
        self.assignment_cost_matrix = np.zeros((self.num_agents, self.num_targets), dtype=np.float32)

        self.obstacle_centers = np.zeros((self.cfg.num_obstacles, 2), dtype=np.float32)
        self.obstacle_radii = np.zeros((self.cfg.num_obstacles,), dtype=np.float32)

        self.step_count = 0
        self._steps_since_reassign = 0
        self.trajectory_lengths = np.zeros((self.num_agents,), dtype=np.float32)
        self.previous_target_distances = np.zeros((self.num_agents,), dtype=np.float32)

        # Paper-style flag: whether an agent has already received its one-time
        # arrival bonus. It is NOT used to freeze the UAV or judge success.
        self.arrived = np.zeros((self.num_agents,), dtype=bool)

        # Diagnostic only: targets currently reached by their assigned agents.
        # It is NOT used as the success condition.
        self.target_arrived = np.zeros((self.num_targets,), dtype=bool)

        self.done = False
        self.termination_reason = ""
        self.last_reward_terms: Dict[str, np.ndarray] = {}

    def set_target_assigner(self, target_assigner: BaseTargetAssigner) -> None:
        """Replace the online target assignment module at runtime."""
        self.target_assigner = target_assigner
        self._update_assignments()
        self.previous_target_distances = self._compute_assigned_target_distances()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment and return observations with shape [num_agents, obs_dim]."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.trajectory_lengths[:] = 0.0
        self.arrived[:] = False
        self.target_arrived[:] = False
        self.done = False
        self.termination_reason = ""
        self.last_reward_terms = {}

        if self.cfg.layout_mode == "cross":
            self._reset_layout_cross()
        else:
            self._reset_uavs()
            self._reset_targets()
        self._generate_obstacles()
        self._steps_since_reassign = self.cfg.reassign_interval  # force reassign on reset
        self._update_assignments()

        self.previous_target_distances = self._compute_assigned_target_distances()
        return self._get_obs()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Advance the simulation by one time step."""
        if self.done:
            return (
                self._get_obs(),
                np.zeros((self.num_agents,), dtype=np.float32),
                np.ones((self.num_agents,), dtype=bool),
                self._get_info(),
            )

        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_agents, self.action_dim):
            raise ValueError(
                f"Expected actions with shape {(self.num_agents, self.action_dim)}, "
                f"but got {actions.shape}."
            )

        # The current action was produced from the current assignment phi_t.
        # Keep both the assignment and the target snapshot fixed for reward / arrival
        # computation. This prevents artificial progress caused by either online
        # reassignment or target swapping / movement during the environment transition.
        assignments_for_reward = self.assignments.copy()
        target_positions_for_reward = self.target_positions.copy()
        previous_distances = self._compute_distances_for_assignments_and_targets(
            assignments_for_reward,
            target_positions_for_reward,
        )

        prev_positions = self.positions.copy()

        self._apply_actions(actions)

        displacement = np.linalg.norm(self.positions - prev_positions, axis=1)
        self.trajectory_lengths += displacement.astype(np.float32)
        self.step_count += 1

        # Reward and success are computed using the same assignment and target positions
        # that generated the action. Dynamic target updates only affect the next state.
        current_distances = self._compute_distances_for_assignments_and_targets(
            assignments_for_reward,
            target_positions_for_reward,
        )
        current_agent_arrived = current_distances <= self.cfg.arrival_threshold
        newly_arrived = current_agent_arrived & (~self.arrived)
        newly_arrived_target_ids = assignments_for_reward[newly_arrived]

        # Diagnostic target occupancy at the current step; not a persistent completion flag.
        self.target_arrived[:] = False
        if np.any(current_agent_arrived):
            self.target_arrived[assignments_for_reward[current_agent_arrived]] = True

        all_arrived = bool(np.all(current_agent_arrived))

        # Pre-compute collision distance matrices once; reuse for check + penalty.
        half = self.cfg.world_size / 2.0
        boundary_violation = bool(np.any(np.abs(self.positions) > half))

        # Agent-obstacle distances.
        if self.obstacle_centers.shape[0] > 0:
            ao_diff = self.positions[:, None, :] - self.obstacle_centers[None, :, :]
            ao_dists = np.linalg.norm(ao_diff, axis=-1)
            ao_thresh = self.cfg.uav_radius + self.obstacle_radii + self.cfg.obstacle_safety_margin
            ao_mask = ao_dists <= ao_thresh[None, :]
            obstacle_collision = bool(np.any(ao_mask))
        else:
            obstacle_collision = False

        # Inter-agent distances.
        ia_diff = self.positions[:, None, :] - self.positions[None, :, :]
        ia_dists = np.linalg.norm(ia_diff, axis=-1)
        ia_mask = (ia_dists <= self.cfg.inter_agent_min_distance) & ~np.eye(self.num_agents, dtype=bool)
        inter_agent_collision = bool(np.any(ia_mask))

        timeout = self.step_count >= self.cfg.max_steps

        failure = bool(boundary_violation or obstacle_collision or inter_agent_collision)
        self.done = bool(all_arrived or failure or timeout)
        self.termination_reason = self._build_termination_reason(
            all_arrived=all_arrived,
            boundary_violation=boundary_violation,
            obstacle_collision=obstacle_collision,
            inter_agent_collision=inter_agent_collision,
            timeout=timeout,
        )

        rewards = self._compute_rewards(
            previous_distances=previous_distances,
            current_distances=current_distances,
            current_agent_arrived=current_agent_arrived,
            all_arrived=all_arrived,
        )

        # Collision penalty from pre-computed masks.
        collision_penalty = np.zeros((self.num_agents,), dtype=np.float32)
        penalty = self.cfg.collision_penalty
        if inter_agent_collision:
            collision_penalty += ia_mask.sum(axis=1).astype(np.float32) * penalty
        if obstacle_collision:
            collision_penalty += ao_mask.sum(axis=1).astype(np.float32) * penalty
        if boundary_violation:
            collision_penalty[np.any(np.abs(self.positions) > half, axis=1)] += penalty

        rewards = rewards + collision_penalty

        # Mark agents that have already collected the one-time arrival bonus.
        self.arrived |= newly_arrived

        # Dynamic targets are part of the environment transition to s_{t+1}.
        # They are updated only after reward / termination are computed, so target
        # motion or swap cannot create fake progress reward.
        self._maybe_update_dynamic_targets()

        # Now solve assignment for the next observation o_{t+1}.
        self._update_assignments()
        self.previous_target_distances = self._compute_assigned_target_distances()

        dones = np.full((self.num_agents,), self.done, dtype=bool)
        info = self._get_info(
            all_arrived=all_arrived,
            boundary_violation=boundary_violation,
            obstacle_collision=obstacle_collision,
            inter_agent_collision=inter_agent_collision,
            timeout=timeout,
            failure=failure,
            current_agent_arrived=current_agent_arrived.copy(),
            newly_arrived=newly_arrived.copy(),
            newly_arrived_target_ids=newly_arrived_target_ids.copy(),
            current_target_distances=current_distances.copy(),
            assignments_used_for_reward=assignments_for_reward.copy(),
            target_positions_used_for_reward=target_positions_for_reward.copy(),
        )

        return self._get_obs(), rewards, dones, info

    def _apply_actions(self, actions: np.ndarray) -> None:
        """Apply remapped velocity commands using unicycle-style 2D kinematics."""
        self.previous_linear_velocities = self.linear_velocities.copy()
        self.previous_angular_velocities = self.angular_velocities.copy()

        v_raw = actions[:, 0]
        v_cmd = np.where(
            v_raw >= 0,
            v_raw * self.cfg.max_linear_velocity,
            v_raw * abs(self.cfg.min_linear_velocity),
        )
        omega_cmd = np.clip(actions[:, 1], -self.cfg.max_angular_velocity, self.cfg.max_angular_velocity)

        if self.cfg.freeze_arrived_uavs:
            v_cmd[self.arrived] = 0.0
            omega_cmd[self.arrived] = 0.0

        self.linear_velocities = v_cmd.astype(np.float32)
        self.angular_velocities = omega_cmd.astype(np.float32)

        self.linear_accelerations = (
            (self.linear_velocities - self.previous_linear_velocities) / self.cfg.dt
        ).astype(np.float32)
        self.angular_accelerations = (
            (self.angular_velocities - self.previous_angular_velocities) / self.cfg.dt
        ).astype(np.float32)

        self.headings = self._wrap_angle(self.headings + self.angular_velocities * self.cfg.dt)
        self.positions[:, 0] += self.linear_velocities * np.cos(self.headings) * self.cfg.dt
        self.positions[:, 1] += self.linear_velocities * np.sin(self.headings) * self.cfg.dt

    def _reset_uavs(self) -> None:
        """Place UAVs vertically along the left side of the map."""
        center = (self.num_agents - 1) / 2.0
        for i in range(self.num_agents):
            self.positions[i, 0] = self.cfg.start_x
            self.positions[i, 1] = (i - center) * self.cfg.start_y_gap
            self.headings[i] = 0.0

        self.linear_velocities[:] = 0.0
        self.angular_velocities[:] = 0.0
        self.previous_linear_velocities[:] = 0.0
        self.previous_angular_velocities[:] = 0.0
        self.linear_accelerations[:] = 0.0
        self.angular_accelerations[:] = 0.0

    def _reset_targets(self) -> None:
        """Place targets vertically along the right side of the map and initialize motion."""
        center = (self.num_targets - 1) / 2.0
        for j in range(self.num_targets):
            self.target_positions[j, 0] = self.cfg.target_x
            self.target_positions[j, 1] = (j - center) * self.cfg.target_y_gap

        self.target_velocities[:] = 0.0
        mode = self.cfg.target_motion_mode.lower()
        if self.cfg.dynamic_targets and "linear" in mode:
            angles = self.rng.uniform(-np.pi, np.pi, size=(self.num_targets,))
            self.target_velocities[:, 0] = self.cfg.target_speed * np.cos(angles)
            self.target_velocities[:, 1] = self.cfg.target_speed * np.sin(angles)

        if self.cfg.dynamic_targets and "racetrack" in mode:
            if not hasattr(self, "_racetrack_s"):
                self._racetrack_s = np.zeros(self.num_targets, dtype=np.float64)
            self._init_racetrack_positions()

    def _reset_layout_cross(self) -> None:
        """Cross layout: interleave agents and targets on both sides.

        Left side  (x=start_x):  A0 T1 A2 T3 A4 ...
        Right side (x=target_x): T0 A1 T2 A3 T4 ...
        """
        gap = self.cfg.start_y_gap
        center = (self.num_agents - 1) / 2.0

        for slot in range(self.num_agents):
            y = (slot - center) * gap
            if slot % 2 == 0:
                # Even slot: agent left, target right
                self.positions[slot, 0] = self.cfg.start_x
                self.positions[slot, 1] = y
                self.target_positions[slot, 0] = self.cfg.target_x
                self.target_positions[slot, 1] = y
            else:
                # Odd slot: target left, agent right
                self.positions[slot, 0] = self.cfg.target_x
                self.positions[slot, 1] = y
                self.target_positions[slot, 0] = self.cfg.start_x
                self.target_positions[slot, 1] = y

        self.headings[:] = 0.0
        self.linear_velocities[:] = 0.0
        self.angular_velocities[:] = 0.0
        self.previous_linear_velocities[:] = 0.0
        self.previous_angular_velocities[:] = 0.0
        self.linear_accelerations[:] = 0.0
        self.angular_accelerations[:] = 0.0
        self.target_velocities[:] = 0.0

    def _maybe_update_dynamic_targets(self) -> None:
        """Update target positions for dynamic-target experiments."""
        if not self.cfg.dynamic_targets:
            return

        mode = self.cfg.target_motion_mode.lower()
        if mode == "none":
            return

        if "linear" in mode:
            self._move_targets_linearly_with_bounce()

        if "racetrack" in mode:
            self._move_targets_racetrack()

        if "swap" in mode:
            if self.step_count < self.cfg.target_swap_start_step:
                return
            if self.cfg.target_swap_interval <= 0:
                return
            if self.step_count % self.cfg.target_swap_interval != 0:
                return
            self._swap_target_positions()

    def _move_targets_linearly_with_bounce(self) -> None:
        """Move targets inside a bounded goal area and bounce at boundaries."""
        self.target_positions += self.target_velocities * self.cfg.dt

        x_min, x_max = self.cfg.target_area_x_min, self.cfg.target_area_x_max
        y_min, y_max = self.cfg.target_area_y_min, self.cfg.target_area_y_max

        low_x = self.target_positions[:, 0] < x_min
        high_x = self.target_positions[:, 0] > x_max
        if np.any(low_x | high_x):
            self.target_velocities[low_x | high_x, 0] *= -1.0
            self.target_positions[:, 0] = np.clip(self.target_positions[:, 0], x_min, x_max)

        low_y = self.target_positions[:, 1] < y_min
        high_y = self.target_positions[:, 1] > y_max
        if np.any(low_y | high_y):
            self.target_velocities[low_y | high_y, 1] *= -1.0
            self.target_positions[:, 1] = np.clip(self.target_positions[:, 1], y_min, y_max)

    def _swap_target_positions(self) -> None:
        """Periodically permute target positions to emulate target reassignment pressure."""
        if self.num_targets <= 1:
            return

        permutation = self.rng.permutation(self.num_targets)
        # Avoid an all-fixed permutation when possible.
        if np.all(permutation == np.arange(self.num_targets)):
            permutation = np.roll(permutation, 1)

        self.target_positions = self.target_positions[permutation].copy()
        self.target_velocities = self.target_velocities[permutation].copy()

    def _move_targets_racetrack(self) -> None:
        """Move targets along a vertical stadium-shaped track.

        Track geometry (clockwise):
          Segment 0 — left vertical ↑  (front, slow): x = target_x
          Segment 1 — top arc, left→right through top (slow)
          Segment 2 — right vertical ↓ (back, fast): x = target_x + 2*R
          Segment 3 — bottom arc, right→left through bottom (slow)

        Arc-length is tracked per-target in self._racetrack_s (initialised
        lazily on first call). Different speeds are used for front vs back
        sides, so most targets stay on the front (visible) side.
        """
        if not hasattr(self, "_racetrack_s"):
            self._racetrack_s = np.zeros(self.num_targets, dtype=np.float64)
            self._init_racetrack_positions()
            return  # _init_racetrack_positions already set positions and velocities

        t_gap = self.cfg.target_y_gap / self.cfg.racetrack_front_speed
        if self._racetrack_elapsed >= t_gap:
            return

        R = self.cfg.racetrack_turn_radius
        L = self.cfg.racetrack_straight_half_length
        straight_len = 2.0 * L
        arc_len = np.pi * R
        total_len = 2.0 * straight_len + 2.0 * arc_len  # 4L + 2πR

        seg_boundaries = np.array([
            0.0,                     # seg 0 start (left vertical ↑)
            straight_len,            # seg 1 start (top arc)
            straight_len + arc_len,  # seg 2 start (right vertical ↓)
            straight_len + arc_len + straight_len,  # seg 3 start (bottom arc)
            total_len,               # end
        ])

        for i in range(self.num_targets):
            s = self._racetrack_s[i]

            # Determine segment and set segment speed.
            if s < seg_boundaries[1]:
                # Segment 0: left vertical ↑ (front, slow)
                speed = self.cfg.racetrack_front_speed
            elif s < seg_boundaries[2]:
                # Segment 1: top arc (slow)
                speed = self.cfg.racetrack_back_speed
            elif s < seg_boundaries[3]:
                # Segment 2: right vertical ↓ (back, fast)
                speed = self.cfg.racetrack_back_speed
            else:
                # Segment 3: bottom arc (slow)
                speed = self.cfg.racetrack_back_speed

            s += speed * self.cfg.dt
            if s >= total_len:
                s -= total_len

            self._racetrack_s[i] = s
            (x, y), (tx, ty) = self._racetrack_s_to_position(s)
            self.target_positions[i, 0] = x
            self.target_positions[i, 1] = y
            self.target_velocities[i, 0] = speed * tx
            self.target_velocities[i, 1] = speed * ty

        self._racetrack_elapsed += self.cfg.dt

    def _init_racetrack_positions(self) -> None:
        """Place targets on the left vertical, matching the standard layout.

        Uses target_y_gap (same as static / linear modes) and centres the
        formation vertically so that target spacing is consistent across
        motion modes. If the formation exceeds the straight segment, it
        is clipped to [-L, L].
        """
        L = self.cfg.racetrack_straight_half_length
        center = (self.num_targets - 1) / 2.0
        for i in range(self.num_targets):
            y = (i - center) * self.cfg.target_y_gap
            y = float(np.clip(y, -L, L))
            self.target_positions[i, 0] = self.cfg.target_x
            self.target_positions[i, 1] = y
            # Left vertical: arc-length s goes from 0 (y=-L) to 2L (y=+L).
            self._racetrack_s[i] = y + L
            self.target_velocities[i, 0] = 0.0
            self.target_velocities[i, 1] = self.cfg.racetrack_front_speed
        self._racetrack_elapsed = 0.0

    def _racetrack_s_to_position(self, s: float):
        """Map arc-length s to (x, y) and unit tangent (tx, ty) on the stadium.

        Track layout (clockwise):
          Segment 0: left vertical ↑,  s ∈ [0, 2L)
          Segment 1: top arc →,       s ∈ [2L, 2L+πR)
          Segment 2: right vertical ↓, s ∈ [2L+πR, 4L+πR)
          Segment 3: bottom arc ←,    s ∈ [4L+πR, 4L+2πR)
        """
        R = self.cfg.racetrack_turn_radius
        L = self.cfg.racetrack_straight_half_length
        straight_len = 2.0 * L
        arc_len = np.pi * R
        total_len = 2.0 * straight_len + 2.0 * arc_len

        # Wrap s into [0, total_len).
        s = s % total_len

        cx = self.cfg.target_x + R  # left vertical at x = target_x

        if s < straight_len:
            # Segment 0: left vertical ↑
            s_local = s
            x = cx - R
            y = -L + s_local
            tx, ty = 0.0, 1.0
        elif s < straight_len + arc_len:
            # Segment 1: top arc (left → right through top)
            s_local = s - straight_len
            alpha = np.pi - s_local / R  # π → 0 (clockwise through top)
            x = cx + R * np.cos(alpha)
            y = L + R * np.sin(alpha)
            tx, ty = float(np.sin(alpha)), float(-np.cos(alpha))
        elif s < 2.0 * straight_len + arc_len:
            # Segment 2: right vertical ↓
            s_local = s - straight_len - arc_len
            x = cx + R
            y = L - s_local
            tx, ty = 0.0, -1.0
        else:
            # Segment 3: bottom arc (right → left through bottom)
            s_local = s - 2.0 * straight_len - arc_len
            alpha = -s_local / R  # 0 → -π (clockwise through bottom)
            x = cx + R * np.cos(alpha)
            y = -L + R * np.sin(alpha)
            tx, ty = float(np.sin(alpha)), float(-np.cos(alpha))

        return (x, y), (tx, ty)

    def _update_assignments(self) -> None:
        """Update agent-target assignment through the pluggable assignment module."""
        self._steps_since_reassign += 1
        if self._steps_since_reassign < self.cfg.reassign_interval:
            return
        self._steps_since_reassign = 0

        dists = np.linalg.norm(
            self.positions[:, None, :] - self.positions[None, :, :], axis=-1
        )
        comm_adj = (dists <= self.cfg.communication_range) & ~np.eye(self.num_agents, dtype=bool)

        assignments, cost_matrix, assign_info = self.target_assigner.assign(
            agent_positions=self.positions.copy(),
            target_positions=self.target_positions.copy(),
            step_count=self.step_count,
            arrived=self.arrived.copy(),
            communication_graph=comm_adj,
        )

        assignments = np.asarray(assignments, dtype=np.int64)
        cost_matrix = np.asarray(cost_matrix, dtype=np.float32)

        if assignments.shape != (self.num_agents,):
            raise ValueError(
                f"Assigner returned assignments with shape {assignments.shape}, "
                f"expected {(self.num_agents,)}."
            )
        if np.any(assignments < 0) or np.any(assignments >= self.num_targets):
            raise ValueError(f"Invalid assignment indices: {assignments}.")

        self.assignments = assignments
        self.assignment_cost_matrix = cost_matrix
        self.assignment_info = dict(assign_info)

    def _generate_obstacles(self) -> None:
        """Generate non-overlapping circular obstacles inside the configured obstacle area."""
        if self.cfg.num_obstacles == 0:
            self.obstacle_centers = np.zeros((0, 2), dtype=np.float32)
            self.obstacle_radii = np.zeros((0,), dtype=np.float32)
            return

        centers = []
        radii = []
        max_attempts = 10_000
        attempts = 0

        while len(centers) < self.cfg.num_obstacles and attempts < max_attempts:
            attempts += 1
            radius = self.rng.uniform(self.cfg.obstacle_radius_min, self.cfg.obstacle_radius_max)
            center = np.array(
                [
                    self.rng.uniform(self.cfg.obstacle_area_x_min, self.cfg.obstacle_area_x_max),
                    self.rng.uniform(self.cfg.obstacle_area_y_min, self.cfg.obstacle_area_y_max),
                ],
                dtype=np.float32,
            )

            if self._is_valid_obstacle(center, radius, centers, radii):
                centers.append(center)
                radii.append(radius)

        if len(centers) < self.cfg.num_obstacles:
            raise RuntimeError(
                f"Only generated {len(centers)} obstacles out of {self.cfg.num_obstacles}. "
                "Try reducing num_obstacles or min_obstacle_spacing."
            )

        self.obstacle_centers = np.asarray(centers, dtype=np.float32)
        self.obstacle_radii = np.asarray(radii, dtype=np.float32)

    def _is_valid_obstacle(
        self,
        center: np.ndarray,
        radius: float,
        existing_centers: list[np.ndarray],
        existing_radii: list[float],
    ) -> bool:
        """Check whether a newly sampled obstacle is valid."""
        for pos in self.positions:
            if np.linalg.norm(center - pos) < radius + self.cfg.uav_radius + self.cfg.min_obstacle_spacing:
                return False

        for target in self.target_positions:
            if np.linalg.norm(center - target) < radius + self.cfg.arrival_threshold + self.cfg.min_obstacle_spacing:
                return False

        for other_center, other_radius in zip(existing_centers, existing_radii):
            min_dist = radius + other_radius + self.cfg.min_obstacle_spacing
            if np.linalg.norm(center - other_center) < min_dist:
                return False

        return True

    # ------------------------------------------------------------------
    # Observation construction: o_i = [z_i, u_i, g_i, q_i]
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build paper-style observations with shape [num_agents, obs_dim]."""
        lidar_obs = self._compute_lidar_observations()
        ego_obs = self._compute_ego_motion_observations()
        target_obs = self._compute_target_observations()
        topology_obs = self._compute_topology_observations()

        obs = np.concatenate([lidar_obs, ego_obs, target_obs, topology_obs], axis=1)
        if obs.shape != (self.num_agents, self.obs_dim):
            raise RuntimeError(
                f"Observation shape mismatch: got {obs.shape}, "
                f"expected {(self.num_agents, self.obs_dim)}."
            )
        return obs.astype(np.float32)

    def get_graph_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return graph observation for GNN-based MAPPO.

        Returns:
            node_features: [num_agents, node_dim] — LiDAR + ego + target.
            edge_index: [2, num_edges] — directed fully connected, no self-loops.
            edge_attr: [num_edges, edge_dim] — [dx, dy, dist, bearing] normalized.
        """
        lidar_obs = self._compute_lidar_observations()
        ego_obs = self._compute_ego_motion_observations()
        target_obs = self._compute_target_observations()

        node_features = np.concatenate([lidar_obs, ego_obs, target_obs], axis=1).astype(np.float32)
        if node_features.shape != (self.num_agents, self.node_dim):
            raise RuntimeError(
                f"Graph node feature shape mismatch: got {node_features.shape}, "
                f"expected {(self.num_agents, self.node_dim)}."
            )

        edge_index, edge_attr = self._build_agent_graph()
        return node_features, edge_index, edge_attr

    def _build_agent_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build a directed fully connected graph over UAV agents."""
        src_list, dst_list, edge_attrs = [], [], []

        world_scale = max(float(self.cfg.world_size), 1e-6)
        comm_scale = max(float(self.cfg.communication_range), 1e-6)

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    continue

                rel = self.positions[j] - self.positions[i]
                dx = float(rel[0] / world_scale)
                dy = float(rel[1] / world_scale)
                dist = float(np.linalg.norm(rel))
                dist_norm = dist / comm_scale

                bearing = np.arctan2(rel[1], rel[0]) - self.headings[i]
                bearing_norm = float(self._wrap_angle(bearing) / np.pi)

                src_list.append(i)
                dst_list.append(j)
                edge_attrs.append([dx, dy, dist_norm, bearing_norm])

        edge_index = np.asarray([src_list, dst_list], dtype=np.int64)
        edge_attr = np.asarray(edge_attrs, dtype=np.float32)
        return edge_index, edge_attr

    def _compute_lidar_observations(self) -> np.ndarray:
        """Vectorized 2D LiDAR via batch ray-circle intersection."""
        num_rays = self.cfg.lidar_num_rays
        max_range = self.cfg.lidar_range
        half_fov = self.cfg.lidar_fov / 2.0
        relative_angles = np.linspace(-half_fov, half_fov, num_rays, endpoint=False, dtype=np.float32)

        if self.obstacle_centers.shape[0] == 0:
            return np.ones((self.num_agents, num_rays), dtype=np.float32)

        # Ray directions for all agents: (N, R, 2)
        ray_angles = self.headings[:, None] + relative_angles[None, :]
        ray_dirs = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=-1)

        # Broadcast shapes: (N, 1, 1, 2) × (1, R, 1, 2) → (N, R, O, 2)
        origins = self.positions[:, None, None, :]
        dirs = ray_dirs[:, :, None, :]
        centers = self.obstacle_centers[None, None, :, :]
        radii = self.obstacle_radii[None, None, :] + self.cfg.uav_radius + self.cfg.obstacle_safety_margin

        # Ray-circle intersection: all (N, R, O) at once.
        oc = origins - centers
        b = 2.0 * np.sum(dirs * oc, axis=-1)
        c = np.sum(oc ** 2, axis=-1) - radii ** 2
        disc = b ** 2 - 4.0 * c

        lidar = np.full((self.num_agents, num_rays), max_range, dtype=np.float32)
        valid = disc >= 0
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0

        t1_ok = valid & (t1 >= 0) & (t1 <= max_range)
        t2_ok = valid & (t2 >= 0) & (t2 <= max_range)

        both = np.minimum(
            np.where(t1_ok, t1, max_range * 2),
            np.where(t2_ok, t2, max_range * 2),
        )
        np.min(both, axis=-1, out=lidar)

        return (lidar / max_range).clip(0.0, 1.0).astype(np.float32)

    def _compute_ego_motion_observations(self) -> np.ndarray:
        """u_i = [v, omega, a_v, a_omega], normalized."""
        max_linear_acc = max(self.cfg.max_linear_velocity / self.cfg.dt, 1e-6)
        max_angular_acc = max(self.cfg.max_angular_velocity / self.cfg.dt, 1e-6)

        v_scale = np.where(
            self.linear_velocities >= 0,
            self.cfg.max_linear_velocity,
            abs(self.cfg.min_linear_velocity),
        )
        v_norm = self.linear_velocities / np.maximum(v_scale, 1e-6)
        omega_norm = self.angular_velocities / self.cfg.max_angular_velocity
        a_v_norm = np.clip(self.linear_accelerations / max_linear_acc, -1.0, 1.0)
        a_omega_norm = np.clip(self.angular_accelerations / max_angular_acc, -1.0, 1.0)

        return np.stack([v_norm, omega_norm, a_v_norm, a_omega_norm], axis=1).astype(np.float32)

    def _compute_target_observations(self) -> np.ndarray:
        """g_i = [distance_to_assigned_target, relative_bearing], normalized."""
        max_distance = np.sqrt(2.0) * self.cfg.world_size
        target_vectors = self._compute_assigned_target_vectors()
        target_distances = np.linalg.norm(target_vectors, axis=1)
        distance_norm = target_distances / max_distance

        target_angles = np.arctan2(target_vectors[:, 1], target_vectors[:, 0])
        relative_bearing = self._wrap_angle(target_angles - self.headings)
        bearing_norm = relative_bearing / np.pi

        return np.stack([distance_norm, bearing_norm], axis=1).astype(np.float32)

    def _compute_topology_observations(self) -> np.ndarray:
        """q_i = normalized relative positions of all teammates (vectorized)."""
        scale = self.cfg.world_size
        rel = self.positions[None, :, :] - self.positions[:, None, :]  # (N, N, 2)
        mask = ~np.eye(self.num_agents, dtype=bool)
        rel = rel[mask]  # (N*(N-1), 2)

        if self.cfg.use_communication_range_mask:
            dist = np.linalg.norm(rel, axis=-1)
            rel[dist > self.cfg.communication_range] = 0.0

        topology = (rel / scale).reshape(self.num_agents, 2 * (self.num_agents - 1))
        return topology.astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_rewards(
        self,
        previous_distances: np.ndarray,
        current_distances: np.ndarray,
        current_agent_arrived: np.ndarray,
        all_arrived: bool,
    ) -> np.ndarray:
        """Simplified reward: progress toward target + task completion + time pressure."""
        progress = self.cfg.progress_scale * (previous_distances - current_distances)
        progress = np.clip(progress, -1.0, 1.0)
        progress[current_agent_arrived] = 1.0

        arrival = np.zeros((self.num_agents,), dtype=np.float32)
        if all_arrived:
            arrival += self.cfg.all_arrived_bonus

        step = np.full((self.num_agents,), self.cfg.step_penalty, dtype=np.float32)

        rewards = progress + arrival + step

        self.last_reward_terms = {
            "progress": progress.astype(np.float32),
            "arrival": arrival.astype(np.float32),
            "step": step.astype(np.float32),
            "total": rewards.astype(np.float32),
        }
        return rewards.astype(np.float32)

    # ------------------------------------------------------------------
    # Geometry, termination, diagnostics
    # ------------------------------------------------------------------
    def _compute_assigned_target_distances(self) -> np.ndarray:
        return self._compute_distances_for_assignments(self.assignments)

    def _compute_distances_for_assignments(self, assignments: np.ndarray) -> np.ndarray:
        assigned_targets = self.target_positions[assignments]
        return np.linalg.norm(self.positions - assigned_targets, axis=1).astype(np.float32)

    def _compute_distances_for_assignments_and_targets(
        self,
        assignments: np.ndarray,
        target_positions: np.ndarray,
    ) -> np.ndarray:
        assigned_targets = target_positions[assignments]
        return np.linalg.norm(self.positions - assigned_targets, axis=1).astype(np.float32)

    def _compute_assigned_target_vectors(self) -> np.ndarray:
        assigned_targets = self.target_positions[self.assignments]
        return (assigned_targets - self.positions).astype(np.float32)

    def _build_termination_reason(
        self,
        all_arrived: bool,
        boundary_violation: bool,
        obstacle_collision: bool,
        inter_agent_collision: bool,
        timeout: bool,
    ) -> str:
        reasons = []
        if all_arrived:
            reasons.append("success_all_agents_currently_arrived")
        if boundary_violation:
            reasons.append("boundary_violation")
        if obstacle_collision:
            reasons.append("obstacle_collision")
        if inter_agent_collision:
            reasons.append("inter_agent_collision")
        if timeout:
            reasons.append("timeout")
        return "+".join(reasons) if reasons else ""

    def get_global_state(self) -> np.ndarray:
        """
        Return a centralized critic state for CTDE training.

        Actor execution still uses only local observations o_i = [z_i, u_i, g_i, q_i].
        The critic can use richer global information during training because it is
        not needed at decentralized execution time.

        The returned state has a fixed dimension as long as num_agents,
        num_targets, lidar_num_rays, and num_obstacles are fixed.
        """
        half_world = self.cfg.world_size / 2.0
        max_linear_acc = max(self.cfg.max_linear_velocity / self.cfg.dt, 1e-6)
        max_angular_acc = max(self.cfg.max_angular_velocity / self.cfg.dt, 1e-6)

        state_parts = []

        # 1. Joint local observations: what all actors currently observe.
        # This gives the critic access to LiDAR risk, target bearing/distance,
        # and local swarm topology without changing the actor input.
        if self.cfg.critic_include_joint_obs:
            joint_obs = self._get_obs().reshape(-1)
            state_parts.append(joint_obs)

        # 2. Global UAV kinematics.
        uav_state = np.concatenate(
            [
                self.positions.reshape(-1) / half_world,
                np.cos(self.headings),
                np.sin(self.headings),
                np.where(
                    self.linear_velocities >= 0,
                    self.linear_velocities / self.cfg.max_linear_velocity,
                    self.linear_velocities / abs(self.cfg.min_linear_velocity),
                ),
                self.angular_velocities / self.cfg.max_angular_velocity,
                np.clip(self.linear_accelerations / max_linear_acc, -1.0, 1.0),
                np.clip(self.angular_accelerations / max_angular_acc, -1.0, 1.0),
            ]
        )
        state_parts.append(uav_state)

        # 3. Global target state.
        target_state = np.concatenate(
            [self.target_positions.reshape(-1) / half_world]
        )
        if self.cfg.dynamic_targets:
            target_speed_norm = max(self.cfg.target_speed, 1e-6)
            target_state = np.concatenate(
                [
                    target_state,
                    np.clip(self.target_velocities.reshape(-1) / target_speed_norm, -1.0, 1.0),
                ]
            )
        state_parts.append(target_state)

        # 4. Assignment identity.
        assignment_state = self.assignments.astype(np.float32) / max(1, self.num_targets - 1)
        state_parts.append(assignment_state)

        # 5. Assignment cost matrix.
        # This helps the critic estimate whether the current matching is easy or costly.
        if self.cfg.critic_include_cost_matrix:
            max_cost = 2.0 * (self.cfg.world_size ** 2)
            cost_state = np.clip(
                self.assignment_cost_matrix.reshape(-1) / max(max_cost, 1e-6),
                0.0,
                1.0,
            )
            state_parts.append(cost_state)

        # 6. Obstacle field summary statistics.
        if self.cfg.critic_include_obstacles:
            state_parts.append(self._get_obstacle_summary_state())

        # 7. Arrival / task flags.
        # arrived means the one-time arrival bonus has already been collected.
        # target_arrived is diagnostic for targets currently reached by assigned UAVs.
        state_parts.append(self.arrived.astype(np.float32))
        state_parts.append(self.target_arrived.astype(np.float32))

        # 8. Normalized time.
        state_parts.append(
            np.array([self.step_count / max(1, self.cfg.max_steps)], dtype=np.float32)
        )

        return np.concatenate(state_parts).astype(np.float32)

    def _get_obstacle_summary_state(self) -> np.ndarray:
        """
        Return compact obstacle field summary for the centralized critic.

        Per-agent features: (normalized_distance, dir_x, dir_y) to the nearest
        obstacle, giving the critic both range and direction for collision risk.
        """
        half_world = self.cfg.world_size / 2.0

        if self.obstacle_centers.shape[0] == 0:
            # no obstacles: far distance, zero direction
            empty = np.zeros((self.num_agents * 3,), dtype=np.float32)
            empty[0::3] = 1.0  # max normalized distance
            return empty

        centers = self.obstacle_centers

        diff = self.positions[:, None, :] - centers[None, :, :]           # [A, O, 2]
        dists = np.linalg.norm(diff, axis=-1)                              # [A, O]
        nearest_idx = np.argmin(dists, axis=1)                             # [A]

        nearest_dist = dists[np.arange(self.num_agents), nearest_idx] / self.cfg.lidar_range
        nearest_vec = centers[nearest_idx] - self.positions                # [A, 2]
        nearest_dir = nearest_vec / (np.linalg.norm(nearest_vec, axis=1, keepdims=True) + 1e-8)

        per_agent = np.empty((self.num_agents * 3,), dtype=np.float32)
        per_agent[0::3] = nearest_dist
        per_agent[1::3] = nearest_dir[:, 0]
        per_agent[2::3] = nearest_dir[:, 1]

        return per_agent

    def get_global_state_dim(self) -> int:
        """Return the current centralized critic state dimension."""
        return int(self.get_global_state().shape[0])

    def _get_info(self, **extra_flags) -> Dict:
        info = {
            "step_count": self.step_count,
            "done": self.done,
            "termination_reason": self.termination_reason,
            "positions": self.positions.copy(),
            "headings": self.headings.copy(),
            "linear_velocities": self.linear_velocities.copy(),
            "angular_velocities": self.angular_velocities.copy(),
            "linear_accelerations": self.linear_accelerations.copy(),
            "angular_accelerations": self.angular_accelerations.copy(),
            "trajectory_lengths": self.trajectory_lengths.copy(),
            "target_positions": self.target_positions.copy(),
            "target_velocities": self.target_velocities.copy(),
            "assignments": self.assignments.copy(),
            "assignment_cost_matrix": self.assignment_cost_matrix.copy(),
            "assignment_info": dict(self.assignment_info),
            "arrived_bonus_collected": self.arrived.copy(),
            "target_arrived_current": self.target_arrived.copy(),
            "obstacle_centers": self.obstacle_centers.copy(),
            "obstacle_radii": self.obstacle_radii.copy(),
            "obs_dim": self.obs_dim,
            "reward_terms": {k: v.copy() for k, v in self.last_reward_terms.items()},
        }
        info.update(extra_flags)
        return info

    @staticmethod
    def _wrap_angle(angle: np.ndarray) -> np.ndarray:
        return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)

    def render_state(self) -> None:
        print(f"Step: {self.step_count}")
        print(f"Done: {self.done}, reason: {self.termination_reason}")
        print(f"Assignments: {self.assignments.tolist()}")
        print(f"Arrival bonus collected: {self.arrived.tolist()}")
        print(f"Targets currently reached: {self.target_arrived.tolist()}")
        print(f"Assignment info: {self.assignment_info}")
        for i in range(self.num_agents):
            x, y = self.positions[i]
            target_id = self.assignments[i]
            tx, ty = self.target_positions[target_id]
            dist = np.linalg.norm(self.positions[i] - self.target_positions[target_id])
            print(
                f"UAV {i}: x={x:.2f}, y={y:.2f}, "
                f"heading={self.headings[i]:.2f}, "
                f"target={target_id}({tx:.2f},{ty:.2f}), "
                f"dist={dist:.2f}, bonus_collected={self.arrived[i]}"
            )

    def render_matplotlib(self, ax=None, show_lidar: bool = False):
        """Optional quick visualization using matplotlib."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        half_size = self.cfg.world_size / 2.0
        ax.set_xlim(-half_size, half_size)
        ax.set_ylim(-half_size, half_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"2D Multi-UAV Env | step={self.step_count}")

        for center, radius in zip(self.obstacle_centers, self.obstacle_radii):
            circle = plt.Circle(center, radius, fill=True, alpha=0.5)
            ax.add_patch(circle)

        ax.scatter(self.target_positions[:, 0], self.target_positions[:, 1], marker="*", s=160, label="Targets")
        ax.scatter(self.positions[:, 0], self.positions[:, 1], marker="o", label="UAVs")

        dx = np.cos(self.headings) * 0.5
        dy = np.sin(self.headings) * 0.5
        ax.quiver(self.positions[:, 0], self.positions[:, 1], dx, dy, angles="xy", scale_units="xy", scale=1)

        for i in range(self.num_agents):
            target = self.target_positions[self.assignments[i]]
            ax.plot([self.positions[i, 0], target[0]], [self.positions[i, 1], target[1]], linestyle="--", linewidth=1)
            ax.text(self.positions[i, 0], self.positions[i, 1], f"U{i}")
            ax.text(target[0], target[1], f"T{self.assignments[i]}")

        if show_lidar:
            self._draw_lidar(ax)

        ax.legend(loc="upper left")
        ax.grid(True)
        return ax

    def _draw_lidar(self, ax) -> None:
        """Draw LiDAR rays for debugging."""
        num_rays = self.cfg.lidar_num_rays
        half_fov = self.cfg.lidar_fov / 2.0
        relative_angles = np.linspace(-half_fov, half_fov, num_rays, endpoint=False, dtype=np.float32)
        lidar = self._compute_lidar_observations() * self.cfg.lidar_range

        for i in range(self.num_agents):
            origin = self.positions[i]
            ray_angles = self.headings[i] + relative_angles
            for r_id, angle in enumerate(ray_angles):
                end = origin + lidar[i, r_id] * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                ax.plot([origin[0], end[0]], [origin[1], end[1]], linewidth=0.3, alpha=0.3)


if __name__ == "__main__":
    cfg = UAVEnvConfig(
        num_agents=3,
        num_obstacles=10,
        assigner_name="hungarian",
        lidar_num_rays=35,
        dynamic_targets=True,
        target_motion_mode="linear_swap",
        target_swap_start_step=5,
        target_swap_interval=5,
        seed=42,
    )
    env = MultiUAV2DEnv(cfg)

    obs = env.reset()
    print("Initial obs shape:", obs.shape)
    print("Expected obs dim:", env.obs_dim)
    print("Global state shape:", env.get_global_state().shape)
    print("Global state dim:", env.get_global_state_dim())
    env.render_state()

    total_reward = np.zeros(cfg.num_agents, dtype=np.float32)
    for _ in range(5):
        actions = np.array(
            [
                [1.0, 0.1],
                [1.0, 0.0],
                [1.0, -0.1],
            ],
            dtype=np.float32,
        )
        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards
        env.render_state()
        print("Obs shape:", obs.shape)
        print("Rewards:", rewards, "Dones:", dones, "Reason:", info["termination_reason"])
        print("Assignments used for reward:", info.get("assignments_used_for_reward"))
        print("Next assignments:", info["assignments"])
        print("Reward terms:")
        for key, value in info["reward_terms"].items():
            print(f"  {key}: {value}")
        if dones.all():
            break

    print("\nTotal reward over test rollout:", total_reward)
