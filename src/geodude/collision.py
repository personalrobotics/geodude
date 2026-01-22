"""Collision checking with grasp awareness."""

import mujoco
import numpy as np

from geodude.grasp_manager import GraspManager


class GraspAwareCollisionChecker:
    """Collision checker that respects grasp state.

    This wraps basic MuJoCo collision checking but uses the GraspManager
    to ensure collision groups are properly configured. When an object is
    grasped, its collision group is already updated by GraspManager, so
    MuJoCo's native collision filtering handles the rest.

    This class implements the CollisionChecker protocol expected by pycbirrt.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_names: list[str],
        grasp_manager: GraspManager,
    ):
        """Initialize the collision checker.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            joint_names: Names of joints to control (determines DOF mapping)
            grasp_manager: GraspManager instance for grasp state
        """
        self.model = model
        self.data = data
        self.grasp_manager = grasp_manager

        # Get joint indices for the controlled joints
        self.joint_indices = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"Joint '{name}' not found in model")
            # Get qpos index for this joint
            qpos_adr = model.jnt_qposadr[joint_id]
            self.joint_indices.append(qpos_adr)

        # Build a set of body IDs that belong to THIS arm
        # These are the bodies connected to the controlled joints
        self._arm_body_ids: set[int] = set()
        self._gripper_body_ids: set[int] = set()
        self._adjacent_pairs: set[tuple[int, int]] = set()

        # First pass: collect arm link bodies
        arm_link_bodies = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            body_id = model.jnt_bodyid[joint_id]
            self._arm_body_ids.add(body_id)
            arm_link_bodies.append(body_id)

        # Build adjacency from kinematic chain (parent-child relationships)
        for body_id in arm_link_bodies:
            parent_id = model.body_parentid[body_id]
            if parent_id in self._arm_body_ids:
                self._add_adjacent_pair(parent_id, body_id)

        # Add gripper bodies (children of last arm link)
        last_link = arm_link_bodies[-1] if arm_link_bodies else None
        if last_link is not None:
            self._add_gripper_bodies(last_link)

    def _add_adjacent_pair(self, body1: int, body2: int) -> None:
        """Add a pair of bodies as adjacent (allowed to contact)."""
        # Store in canonical order (smaller ID first)
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

    def is_valid(self, q: np.ndarray) -> bool:
        """Check if a configuration is collision-free.

        The GraspManager has already updated collision groups for any
        grasped objects, so we just need to set the configuration and
        check for contacts.

        Args:
            q: Joint configuration (only the controlled joints)

        Returns:
            True if collision-free, False otherwise
        """
        # Set joint positions
        for i, qpos_idx in enumerate(self.joint_indices):
            self.data.qpos[qpos_idx] = q[i]

        # Run forward kinematics to update collision geometry positions
        mujoco.mj_forward(self.model, self.data)

        # Check for collisions
        # ncon > 0 means there are contacts, but we need to filter out
        # expected contacts (like gripper holding object)
        return self._count_invalid_contacts() == 0

    def is_valid_batch(self, qs: np.ndarray) -> np.ndarray:
        """Check multiple configurations for collisions.

        Args:
            qs: Array of configurations, shape (n_configs, n_joints)

        Returns:
            Boolean array, True for collision-free configurations
        """
        results = np.zeros(len(qs), dtype=bool)
        for i, q in enumerate(qs):
            results[i] = self.is_valid(q)
        return results

    def _count_invalid_contacts(self) -> int:
        """Count contacts that indicate invalid collisions.

        We filter out contacts that are expected or don't involve the arm:
        - Contacts that don't involve the robot arm (e.g., object on table)
        - Contacts between adjacent links (successive joints in kinematic chain)
        - Contacts between gripper bodies (finger self-contact)
        - Contacts between gripper and grasped object (that's the grasp!)

        We DO flag as collisions:
        - Non-adjacent self-collision (e.g., forearm hitting gripper)
        - Arm hitting the vention frame
        - Arm hitting the other arm
        - Arm hitting environment objects

        Returns:
            Number of invalid (unexpected) contacts
        """
        invalid_count = 0

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            # Skip contacts that don't involve this arm
            body1_is_arm = body1 in self._arm_body_ids
            body2_is_arm = body2 in self._arm_body_ids
            if not body1_is_arm and not body2_is_arm:
                # Neither body is part of the arm - ignore (e.g., object on table)
                continue

            # Check same-arm contacts more carefully
            if body1_is_arm and body2_is_arm:
                # Both bodies are part of this arm - check if it's an allowed contact
                # Allow: adjacent links (parent-child) or gripper-gripper
                if self._are_adjacent(body1, body2):
                    continue
                if self._both_gripper_bodies(body1, body2):
                    continue
                # Non-adjacent same-arm contact = real self-collision!
                invalid_count += 1
                continue

            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)

            # Check if this is a gripper-grasped-object contact (expected)
            if self._is_expected_grasp_contact(body1_name, body2_name):
                continue

            # This is a collision: arm vs vention frame, arm vs other arm,
            # or arm vs environment object
            invalid_count += 1

        return invalid_count

    def _is_expected_grasp_contact(self, body1_name: str | None, body2_name: str | None) -> bool:
        """Check if contact between two bodies is an expected grasp contact."""
        if body1_name is None or body2_name is None:
            return False

        # Check if one is a grasped object
        grasped_obj = None
        other_body = None

        if self.grasp_manager.is_grasped(body1_name):
            grasped_obj = body1_name
            other_body = body2_name
        elif self.grasp_manager.is_grasped(body2_name):
            grasped_obj = body2_name
            other_body = body1_name

        if grasped_obj is None:
            return False

        # Contact with grasped object - check if it's with the gripper
        # that's holding it (expected) or something else (might be expected too,
        # like table contact)
        holder_arm = self.grasp_manager.get_holder(grasped_obj)

        # For now, allow all contacts with grasped objects
        # The collision group filtering should handle arm-object collisions
        # This catches gripper-object contacts which are expected
        return True


class SimpleCollisionChecker:
    """Simple collision checker without grasp awareness.

    For use when no objects are being manipulated.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_names: list[str],
    ):
        self.model = model
        self.data = data

        self.joint_indices = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"Joint '{name}' not found in model")
            qpos_adr = model.jnt_qposadr[joint_id]
            self.joint_indices.append(qpos_adr)

    def is_valid(self, q: np.ndarray) -> bool:
        for i, qpos_idx in enumerate(self.joint_indices):
            self.data.qpos[qpos_idx] = q[i]

        mujoco.mj_forward(self.model, self.data)
        return self.data.ncon == 0

    def is_valid_batch(self, qs: np.ndarray) -> np.ndarray:
        results = np.zeros(len(qs), dtype=bool)
        for i, q in enumerate(qs):
            results[i] = self.is_valid(q)
        return results
