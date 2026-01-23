"""Vention linear actuator control."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from geodude.config import VentionBaseConfig

if TYPE_CHECKING:
    from geodude.arm import Arm


class VentionBase:
    """Controls a Vention linear actuator for arm height adjustment.

    The Vention base is a separate actuator that moves the UR5e arm up/down
    on a linear rail. It is controlled independently of the arm joints.

    Collision checking for base movement considers the arm in its current
    configuration and checks if moving the base would cause collisions.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: VentionBaseConfig,
        arm: "Arm",
    ):
        """Initialize the Vention base controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            config: Configuration for this base
            arm: The Arm attached to this base (for collision checking)
        """
        self.model = model
        self.data = data
        self.config = config
        self._arm = arm

        # Get joint ID and qpos index
        self._joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, config.joint_name
        )
        if self._joint_id == -1:
            raise ValueError(f"Joint '{config.joint_name}' not found in model")
        self._qpos_idx = model.jnt_qposadr[self._joint_id]

        # Get actuator ID
        self._actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, config.actuator_name
        )
        if self._actuator_id == -1:
            raise ValueError(f"Actuator '{config.actuator_name}' not found in model")

        # Build set of body IDs for this arm (for collision checking)
        self._arm_body_ids: set[int] = set()
        self._build_arm_body_ids()

    def _build_arm_body_ids(self) -> None:
        """Build set of body IDs that belong to this arm including gripper."""
        for joint_name in self._arm.config.joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id != -1:
                body_id = self.model.jnt_bodyid[joint_id]
                self._arm_body_ids.add(body_id)
                self._add_child_bodies(body_id)

    def _add_child_bodies(self, parent_id: int) -> None:
        """Recursively add child bodies (e.g., gripper) to arm body set."""
        for i in range(self.model.nbody):
            if self.model.body_parentid[i] == parent_id and i not in self._arm_body_ids:
                self._arm_body_ids.add(i)
                self._add_child_bodies(i)

    @property
    def name(self) -> str:
        """Base name ('left' or 'right')."""
        return self.config.name

    @property
    def height(self) -> float:
        """Current height in meters."""
        return float(self.data.qpos[self._qpos_idx])

    @property
    def height_range(self) -> tuple[float, float]:
        """Valid height range (min, max) in meters."""
        return self.config.height_range

    def set_height(self, height: float) -> None:
        """Set height directly without collision checking.

        Args:
            height: Target height in meters

        Raises:
            ValueError: If height is outside valid range
        """
        min_h, max_h = self.config.height_range
        tolerance = 1e-3  # Tolerance for floating point errors (1mm)
        if not (min_h - tolerance) <= height <= (max_h + tolerance):
            raise ValueError(
                f"Height {height} outside valid range [{min_h}, {max_h}]"
            )

        # Clamp to valid range to handle floating point errors
        height = max(min_h, min(max_h, height))

        self.data.qpos[self._qpos_idx] = height
        self.data.ctrl[self._actuator_id] = height
        mujoco.mj_forward(self.model, self.data)

    def move_to(
        self,
        height: float,
        check_collisions: bool = True,
    ) -> bool:
        """Move to target height with optional collision checking.

        Interpolates the path and checks for arm collisions at discrete steps.

        Args:
            height: Target height in meters
            check_collisions: If True, check for collisions along path

        Returns:
            True if movement succeeded, False if blocked by collision

        Raises:
            ValueError: If height is outside valid range
        """
        min_h, max_h = self.config.height_range
        if not min_h <= height <= max_h:
            raise ValueError(
                f"Height {height} outside valid range [{min_h}, {max_h}]"
            )

        if not check_collisions:
            self.set_height(height)
            return True

        # Check collision along path
        current_height = self.height
        if not self._is_path_collision_free(current_height, height):
            return False

        # Path is clear, move to target
        self.set_height(height)
        return True

    def _is_path_collision_free(self, start: float, end: float) -> bool:
        """Check if linear path is collision-free.

        Args:
            start: Starting height
            end: Ending height

        Returns:
            True if path is collision-free
        """
        resolution = self.config.collision_check_resolution
        distance = abs(end - start)

        if distance < 1e-6:
            # Already at target
            return True

        n_steps = max(2, int(np.ceil(distance / resolution)) + 1)
        heights = np.linspace(start, end, n_steps)

        # Save current arm configuration
        arm_q = self._arm.get_joint_positions().copy()
        original_height = self.height

        try:
            for h in heights:
                # Set base height
                self.data.qpos[self._qpos_idx] = h
                # Restore arm configuration (in case anything moved it)
                self._arm.set_joint_positions(arm_q)
                mujoco.mj_forward(self.model, self.data)

                if self._has_arm_collision():
                    return False

            return True
        finally:
            # Restore original state
            self.data.qpos[self._qpos_idx] = original_height
            self._arm.set_joint_positions(arm_q)
            mujoco.mj_forward(self.model, self.data)

    def _has_arm_collision(self) -> bool:
        """Check if arm is in collision with environment.

        Self-collision filtering is handled by MuJoCo via <exclude> tags,
        so we only check for arm-environment contacts.

        Returns:
            True if arm has a collision with environment
        """
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]

            body1_is_arm = body1 in self._arm_body_ids
            body2_is_arm = body2 in self._arm_body_ids

            # Only flag contacts between arm and non-arm (environment)
            if body1_is_arm != body2_is_arm:
                return True

        return False
