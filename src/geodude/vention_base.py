"""Vention linear actuator control."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator import Arm, KinematicExecutor, Trajectory
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

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: VentionBaseConfig,
        arm: Arm,
    ):
        self.model = model
        self.data = data
        self.config = config
        self._arm = arm

        # Get joint ID and qpos index
        self._joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, config.joint_name,
        )
        if self._joint_id == -1:
            raise ValueError(f"Joint '{config.joint_name}' not found in model")
        self._qpos_idx = model.jnt_qposadr[self._joint_id]

        # Get actuator ID
        self._actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, config.actuator_name,
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
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name,
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

    @property
    def name(self) -> str:
        return self.config.name

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
            raise ValueError(
                f"Height {height} outside valid range [{min_h}, {max_h}]"
            )
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
    ) -> Trajectory | None:
        """Plan base motion to target height.

        Args:
            height: Target height in meters.
            arm: Arm to check collisions against (defaults to the arm
                this base was initialized with).
            check_collisions: If True, verify path is collision-free.

        Returns:
            Trajectory if planning succeeded, None if collision detected.
        """
        min_h, max_h = self.config.height_range
        if not min_h <= height <= max_h:
            raise ValueError(
                f"Height {height} outside valid range [{min_h}, {max_h}]"
            )

        current_height = self.get_height()

        if check_collisions and not self._is_path_collision_free(current_height, height):
            return None

        return create_linear_trajectory(
            start=current_height,
            end=height,
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
        """Move to target height with collision checking (kinematic mode).

        For execution through SimContext, use plan_to() instead.

        Returns:
            True if movement succeeded, False if blocked by collision.
        """
        traj = self.plan_to(height, check_collisions=check_collisions)
        if traj is None:
            return False

        executor = KinematicExecutor(
            model=self.model,
            data=self.data,
            joint_qpos_indices=[self._qpos_idx],
            viewer=viewer,
        )
        success = executor.execute(traj)

        if success:
            self.data.ctrl[self._actuator_id] = height

        return success

    def _is_path_collision_free(self, start: float, end: float) -> bool:
        """Check if linear path is collision-free."""
        resolution = self.config.collision_check_resolution
        distance = abs(end - start)

        if distance < 1e-6:
            return True

        n_steps = max(2, int(np.ceil(distance / resolution)) + 1)
        heights = np.linspace(start, end, n_steps)

        arm_q = self._arm.get_joint_positions().copy()
        original_height = self.get_height()

        try:
            for h in heights:
                self.data.qpos[self._qpos_idx] = h
                for i, idx in enumerate(self._arm.joint_qpos_indices):
                    self.data.qpos[idx] = arm_q[i]
                mujoco.mj_forward(self.model, self.data)

                if self._has_arm_collision():
                    return False
            return True
        finally:
            self.data.qpos[self._qpos_idx] = original_height
            for i, idx in enumerate(self._arm.joint_qpos_indices):
                self.data.qpos[idx] = arm_q[i]
            mujoco.mj_forward(self.model, self.data)

    def _has_arm_collision(self) -> bool:
        """Check if arm is in collision with environment."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]

            body1_is_arm = body1 in self._arm_body_ids
            body2_is_arm = body2 in self._arm_body_ids

            if body1_is_arm != body2_is_arm:
                return True

        return False
