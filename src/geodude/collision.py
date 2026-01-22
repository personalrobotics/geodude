"""Collision checking with grasp awareness."""

import mujoco
import numpy as np

from geodude.grasp_manager import GraspManager


class GraspAwareCollisionChecker:
    """Collision checker that respects grasp state.

    Checks for collisions between the robot arm and environment objects.
    Self-collision filtering is handled by MuJoCo via <exclude> tags in the model,
    so this checker only needs to detect arm-environment contacts.

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
            qpos_adr = model.jnt_qposadr[joint_id]
            self.joint_indices.append(qpos_adr)

        # Build set of body IDs that belong to this arm (including gripper)
        self._arm_body_ids: set[int] = set()
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            body_id = model.jnt_bodyid[joint_id]
            self._arm_body_ids.add(body_id)
            self._add_child_bodies(body_id)

    def _add_child_bodies(self, parent_id: int) -> None:
        """Recursively add child bodies (e.g., gripper) to arm body set."""
        for i in range(self.model.nbody):
            if self.model.body_parentid[i] == parent_id and i not in self._arm_body_ids:
                self._arm_body_ids.add(i)
                self._add_child_bodies(i)

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

        Self-collision filtering is handled by MuJoCo via <exclude> tags,
        so we only need to check for arm-environment collisions.

        Returns:
            Number of invalid (unexpected) contacts
        """
        invalid_count = 0

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]

            body1_is_arm = body1 in self._arm_body_ids
            body2_is_arm = body2 in self._arm_body_ids

            # Skip if neither body is arm (e.g., object on table)
            if not body1_is_arm and not body2_is_arm:
                continue

            # Skip same-arm contacts (MuJoCo handles self-collision via <exclude>)
            if body1_is_arm and body2_is_arm:
                continue

            # Arm contacting something external - check if expected grasp
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)

            if self._is_expected_grasp_contact(body1_name, body2_name):
                continue

            # Unexpected arm-environment collision
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
