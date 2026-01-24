"""Collision checking with grasp awareness."""

import mujoco
import numpy as np

from geodude.grasp_manager import GraspManager


class GraspAwareCollisionChecker:
    """Collision checker that respects grasp state.

    Checks for collisions between the robot arm and environment, AND non-adjacent
    self-collisions (e.g., forearm hitting gripper). Adjacent link collisions
    (shoulder-upper_arm, etc.) are filtered by MuJoCo's <exclude> tags.

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

        # Create temporary data for collision checking without visible state changes
        # This avoids flickering in the viewer during planning
        self._temp_data = mujoco.MjData(model)

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

        Uses temporary MjData to avoid visible flickering during planning.

        Args:
            q: Joint configuration (only the controlled joints)

        Returns:
            True if collision-free, False otherwise
        """
        # Copy current state to temp data (preserves grasped object states)
        self._temp_data.qpos[:] = self.data.qpos
        self._temp_data.qvel[:] = self.data.qvel

        # Set joint positions in temp data
        for i, qpos_idx in enumerate(self.joint_indices):
            self._temp_data.qpos[qpos_idx] = q[i]

        # Run forward kinematics to update collision geometry positions
        mujoco.mj_forward(self.model, self._temp_data)

        # Check for collisions using temp data
        return self._count_invalid_contacts(self._temp_data) == 0

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

    def _count_invalid_contacts(self, data: mujoco.MjData | None = None) -> int:
        """Count contacts that indicate invalid collisions.

        MuJoCo's <exclude> tags filter ADJACENT link contacts (shoulder-upper_arm, etc.)
        at the physics level - they won't appear in data.ncon at all.

        Non-adjacent self-collisions (forearm hitting gripper) WILL appear as contacts
        and should be counted as invalid collisions.

        Args:
            data: MjData to check contacts in (defaults to self.data for backward compatibility)

        Returns:
            Number of invalid (unexpected) contacts
        """
        if data is None:
            data = self.data

        invalid_count = 0

        for i in range(data.ncon):
            contact = data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]

            body1_is_arm = body1 in self._arm_body_ids
            body2_is_arm = body2 in self._arm_body_ids

            # Skip if neither body is arm (e.g., object on table)
            if not body1_is_arm and not body2_is_arm:
                continue

            # Self-collision (both bodies belong to this arm)
            # MuJoCo's <exclude> tags filter adjacent links, so any contact that
            # reaches here is a NON-ADJACENT self-collision (e.g., forearm-gripper)
            if body1_is_arm and body2_is_arm:
                invalid_count += 1
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
