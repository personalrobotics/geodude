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

        # Build set of body IDs for THIS arm (for collision checking)
        # Uses same policy as arm planning: only ignore adjacent link contacts
        self._arm_body_ids: set[int] = set()
        self._gripper_body_ids: set[int] = set()
        self._adjacent_pairs: set[tuple[int, int]] = set()
        self._build_arm_body_ids()

    def _build_arm_body_ids(self) -> None:
        """Build set of body IDs that belong to this arm and adjacency info."""
        # First pass: collect arm link bodies
        arm_link_bodies = []
        for joint_name in self._arm.config.joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id != -1:
                body_id = self.model.jnt_bodyid[joint_id]
                self._arm_body_ids.add(body_id)
                arm_link_bodies.append(body_id)

        # Build adjacency from kinematic chain (parent-child relationships)
        for body_id in arm_link_bodies:
            parent_id = self.model.body_parentid[body_id]
            if parent_id in self._arm_body_ids:
                self._add_adjacent_pair(parent_id, body_id)

        # Add gripper bodies (children of last arm link)
        last_link = arm_link_bodies[-1] if arm_link_bodies else None
        if last_link is not None:
            self._add_gripper_bodies(last_link)

    def _add_adjacent_pair(self, body1: int, body2: int) -> None:
        """Add a pair of bodies as adjacent (allowed to contact)."""
        pair = (min(body1, body2), max(body1, body2))
        self._adjacent_pairs.add(pair)

    def _add_gripper_bodies(self, parent_id: int) -> None:
        """Recursively add gripper bodies and their adjacencies."""
        for i in range(self.model.nbody):
            if self.model.body_parentid[i] == parent_id:
                self._arm_body_ids.add(i)
                self._gripper_body_ids.add(i)
                # Gripper body is adjacent to its parent
                self._add_adjacent_pair(parent_id, i)
                # Recurse for gripper sub-bodies
                self._add_gripper_bodies(i)

    def _are_adjacent(self, body1: int, body2: int) -> bool:
        """Check if two bodies are adjacent in the kinematic chain."""
        pair = (min(body1, body2), max(body1, body2))
        return pair in self._adjacent_pairs

    def _both_gripper_bodies(self, body1: int, body2: int) -> bool:
        """Check if both bodies are gripper bodies (finger contacts allowed)."""
        return body1 in self._gripper_body_ids and body2 in self._gripper_body_ids

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
        if not min_h <= height <= max_h:
            raise ValueError(
                f"Height {height} outside valid range [{min_h}, {max_h}]"
            )

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
        """Check if arm is in collision.

        Uses the SAME policy as arm motion planning:
        - Ignore adjacent link contacts (parent-child in kinematic chain)
        - Ignore gripper-gripper contacts (finger self-contact)
        - Flag non-adjacent same-arm contacts (e.g., forearm hitting gripper)
        - Flag contacts between arm and vention frame
        - Flag contacts between arm and other arm
        - Flag contacts between arm and environment

        Returns:
            True if arm has a collision
        """
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            # Check if contact involves this arm
            body1_is_arm = body1 in self._arm_body_ids
            body2_is_arm = body2 in self._arm_body_ids

            if not body1_is_arm and not body2_is_arm:
                # Neither body is part of this arm - not our concern
                continue

            # Check same-arm contacts more carefully
            if body1_is_arm and body2_is_arm:
                # Allow: adjacent links (parent-child) or gripper-gripper
                if self._are_adjacent(body1, body2):
                    continue
                if self._both_gripper_bodies(body1, body2):
                    continue
                # Non-adjacent same-arm contact = real self-collision!
                return True

            # Arm is in contact with something else (vention, other arm, environment)
            # This is a collision
            return True

        return False
