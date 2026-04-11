# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Vention linear actuator control."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mj_manipulator import Arm, GraspManager, Trajectory
from mj_manipulator.trajectory import create_linear_trajectory

from geodude.config import VentionBaseConfig

if TYPE_CHECKING:
    pass


class VentionBase:
    """Controls a Vention linear actuator for arm height adjustment.

    The Vention base moves the UR5e arm up/down on a linear rail.
    Collision checking considers the arm in its current configuration
    and verifies the path is safe before moving.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    config: VentionBaseConfig

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: VentionBaseConfig,
        arm: Arm,
    ):
        self.model: mujoco.MjModel = model
        self.data: mujoco.MjData = data
        self.config: VentionBaseConfig = config
        self._arm = arm

        # Get joint ID and qpos index
        self._joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            config.joint_name,
        )
        if self._joint_id == -1:
            raise ValueError(f"Joint '{config.joint_name}' not found in model")
        self._qpos_idx = model.jnt_qposadr[self._joint_id]

        # Get actuator ID
        self._actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            config.actuator_name,
        )
        if self._actuator_id == -1:
            raise ValueError(f"Actuator '{config.actuator_name}' not found")

        # Build set of body IDs for this arm (for collision checking)
        self._arm_body_ids: set[int] = set()
        self._build_arm_body_ids()

    def _build_arm_body_ids(self) -> None:
        """Build set of body IDs that belong to this arm including gripper."""
        for joint_name in self._arm.config.joint_names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id != -1:
                body_id = self.model.jnt_bodyid[joint_id]
                self._arm_body_ids.add(body_id)
                self._add_child_bodies(body_id)

    def _add_child_bodies(self, parent_id: int) -> None:
        """Recursively add child bodies to arm body set."""
        for i in range(self.model.nbody):
            if self.model.body_parentid[i] == parent_id and i not in self._arm_body_ids:
                self._arm_body_ids.add(i)
                self._add_child_bodies(i)

    # -- Entity protocol (for SimContext registration) -----------------------

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def joint_qpos_indices(self) -> list[int]:
        return [self._qpos_idx]

    @property
    def joint_qvel_indices(self) -> list[int]:
        return [self.model.jnt_dofadr[self._joint_id]]

    @property
    def actuator_ids(self) -> list[int]:
        return [self._actuator_id]

    @property
    def arm_body_ids(self) -> set[int]:
        """Body IDs belonging to this base's arm (including gripper).

        Useful for grasp-aware contact filtering — callers like
        :class:`geodude.bt.nodes.LiftBase` need to distinguish "the held
        object is touching the source surface" from "the held object is
        touching the gripper" (which is the grasp itself, not collision).
        Returning a copy keeps the internal set immutable to outside
        callers.
        """
        return set(self._arm_body_ids)

    @property
    def grasp_manager(self) -> GraspManager | None:
        """Grasp manager from the associated arm (for attached object tracking)."""
        return self._arm.grasp_manager

    # -- Height access ------------------------------------------------------

    def get_height(self) -> float:
        """Current height in meters."""
        return float(self.data.qpos[self._qpos_idx])

    @property
    def height_range(self) -> tuple[float, float]:
        return self.config.height_range

    def set_height(self, height: float) -> None:
        """Set height directly without collision checking or animation."""
        min_h, max_h = self.config.height_range
        tolerance = 1e-3
        if not (min_h - tolerance) <= height <= (max_h + tolerance):
            raise ValueError(f"Height {height} outside valid range [{min_h}, {max_h}]")
        height = max(min_h, min(max_h, height))
        self.data.qpos[self._qpos_idx] = height
        self.data.ctrl[self._actuator_id] = height
        mujoco.mj_forward(self.model, self.data)

    def plan_to(
        self,
        height: float,
        arm: Arm | None = None,
        *,
        check_collisions: bool = True,
        partial_ok: bool = False,
    ) -> Trajectory | None:
        """Plan base motion to target height.

        Args:
            height: Target height in meters.
            arm: Arm to check collisions against (defaults to the arm
                this base was initialized with).
            check_collisions: If True, verify path is collision-free.
            partial_ok: If True and a collision blocks the requested
                path partway through, return a trajectory to the
                longest collision-free prefix instead of None. Has no
                effect when ``check_collisions=False``. Mirrors
                ``mj_manipulator.cartesian_path.plan_cartesian_path``'s
                ``partial_ok`` semantics — useful for "lift as far as
                you can" callers like
                :class:`geodude.bt.nodes.LiftBase`. If even the first
                step toward ``height`` is in collision, still returns
                ``None``.

        Returns:
            Trajectory to the requested height, OR (under
            ``partial_ok``) a trajectory to the longest collision-free
            prefix. Returns ``None`` only when no motion at all is
            feasible.
        """
        min_h, max_h = self.config.height_range
        if not min_h <= height <= max_h:
            raise ValueError(f"Height {height} outside valid range [{min_h}, {max_h}]")

        current_height = self.get_height()

        if check_collisions:
            max_clear = self._max_collision_free_height(current_height, height)
            if max_clear is None:
                # Even the first step is in collision; nothing reachable.
                return None
            if not partial_ok and max_clear != height:
                # Strict mode: full path must be clear.
                return None
            # In partial_ok mode (or full-clear strict mode), retime to max_clear.
            target = max_clear
        else:
            target = height

        return create_linear_trajectory(
            start=current_height,
            end=target,
            vel_limit=self.config.kinematic_limits.velocity,
            acc_limit=self.config.kinematic_limits.acceleration,
            entity=self.config.name,
            joint_names=[self.config.joint_name],
        )

    def move_to(
        self,
        height: float,
        check_collisions: bool = True,
        viewer=None,
    ) -> bool:
        """Move to target height with trapezoidal velocity profile.

        .. deprecated::
            Use ``base.plan_to(height)`` and ``ctx.execute(traj)`` instead.
            This method bypasses SimContext and freezes physics during motion.

        Returns:
            True if movement succeeded, False if blocked by collision.
        """
        warnings.warn(
            "VentionBase.move_to() bypasses SimContext and freezes physics. "
            "Use base.plan_to(height) + ctx.execute(traj) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        traj = self.plan_to(height, check_collisions=check_collisions)
        if traj is None:
            return False

        import time as _time

        # Step through trajectory at real-time pace
        gm = self._arm.grasp_manager
        dt = traj.timestamps[1] - traj.timestamps[0] if traj.num_waypoints > 1 else 0.008
        for i in range(traj.num_waypoints):
            h = float(traj.positions[i, 0])
            self.data.qpos[self._qpos_idx] = h
            self.data.ctrl[self._actuator_id] = h
            mujoco.mj_forward(self.model, self.data)
            # Move grasped objects with the arm
            if gm is not None:
                gm.update_attached_poses()
                mujoco.mj_forward(self.model, self.data)
            if viewer is not None:
                viewer.sync()
                _time.sleep(dt)

        # Ensure final position is exact
        self.data.qpos[self._qpos_idx] = height
        self.data.ctrl[self._actuator_id] = height
        mujoco.mj_forward(self.model, self.data)
        if gm is not None:
            gm.update_attached_poses()
            mujoco.mj_forward(self.model, self.data)

        return True

    def _is_path_collision_free(self, start: float, end: float) -> bool:
        """Check if a linear path is fully collision-free.

        Returns True iff every sampled point along the path is
        collision-free. Implemented in terms of
        :meth:`_max_collision_free_height` so the two checks share
        their walker.
        """
        max_clear = self._max_collision_free_height(start, end)
        return max_clear == end

    def _max_collision_free_height(self, start: float, end: float) -> float | None:
        """Return the highest reachable height along ``start → end``.

        Walks the linear path in collision-check-resolution increments.
        At each sample, sets the base qpos and forward-kinematics the
        arm at its current joint configuration, then checks for
        environment collisions via :meth:`_has_arm_collision` (which is
        grasp-aware — held objects are allowed to touch the gripper).

        Returns:
            The height of the **last sample that was collision-free**
            along the path. If the path is fully clear, returns
            ``end``. If not even the first sample is clear (i.e. we're
            already in collision at ``start``), returns ``None``.

        Note:
            The returned value is at most one ``collision_check_resolution``
            short of the actual collision boundary, because we report
            the last clean sample rather than interpolating into the
            blocked segment. This is a deliberately conservative choice
            and matches how
            ``mj_manipulator.cartesian_path.plan_cartesian_path``'s
            ``partial_ok`` mode handles its own boundaries.
        """
        resolution = self.config.collision_check_resolution
        distance = abs(end - start)

        if distance < 1e-6:
            return end

        n_steps = max(2, int(np.ceil(distance / resolution)) + 1)
        heights = np.linspace(start, end, n_steps)

        arm_q = self._arm.get_joint_positions().copy()
        original_height = self.get_height()

        last_clear: float | None = None

        try:
            for h in heights:
                self.data.qpos[self._qpos_idx] = h
                for i, idx in enumerate(self._arm.joint_qpos_indices):
                    self.data.qpos[idx] = arm_q[i]
                mujoco.mj_forward(self.model, self.data)

                if self._has_arm_collision():
                    # First sample blocked → nothing reachable.
                    if last_clear is None:
                        return None
                    # Otherwise return the height of the last clean sample.
                    return float(last_clear)
                last_clear = h
            return float(last_clear)
        finally:
            self.data.qpos[self._qpos_idx] = original_height
            for i, idx in enumerate(self._arm.joint_qpos_indices):
                self.data.qpos[idx] = arm_q[i]
            mujoco.mj_forward(self.model, self.data)

    def _has_arm_collision(self) -> bool:
        """Check if arm is in collision with environment.

        Grasp-aware: contacts between the arm and grasped objects are
        allowed (the arm is holding them, not colliding).
        """
        # Get grasped body names from grasp manager
        grasped_bodies = set()
        gm = self._arm.grasp_manager
        if gm is not None:
            for obj_name in gm.grasped:
                body_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    obj_name,
                )
                if body_id >= 0:
                    grasped_bodies.add(body_id)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]

            body1_is_arm = body1 in self._arm_body_ids
            body2_is_arm = body2 in self._arm_body_ids

            # Skip arm-to-arm contacts (self-collision handled by MuJoCo excludes)
            if body1_is_arm and body2_is_arm:
                continue

            # Arm touching non-arm
            if body1_is_arm != body2_is_arm:
                other = body2 if body1_is_arm else body1
                # Allow contacts with grasped objects
                if other in grasped_bodies:
                    continue
                return True

        return False
