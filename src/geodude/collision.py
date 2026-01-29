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

        Updates both arm joint positions AND attached object positions in temp_data.
        This ensures grasped objects are correctly positioned for collision checking.

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

        # Update attached object poses based on new gripper position
        # This ensures grasped objects move with the gripper during collision checking
        self.grasp_manager.update_attached_poses(self._temp_data)
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

    def _count_invalid_contacts(self, data: mujoco.MjData | None = None, debug: bool = False) -> int:
        """Count contacts that indicate invalid collisions.

        MuJoCo's <exclude> tags filter ADJACENT link contacts (shoulder-upper_arm, etc.)
        at the physics level - they won't appear in data.ncon at all.

        This method treats grasped objects as part of the robot:
        - Gripper-to-grasped-object contacts are ALLOWED (only the holding gripper)
        - Grasped-object-to-arm contacts are INVALID (object hitting elbow, etc.)
        - Grasped-object-to-environment contacts are INVALID (collision)
        - Non-adjacent arm self-collisions are INVALID

        Args:
            data: MjData to check contacts in (defaults to self.data for backward compatibility)
            debug: If True, print debug information about contacts

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

            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)

            body1_is_arm = body1 in self._arm_body_ids
            body2_is_arm = body2 in self._arm_body_ids
            body1_is_grasped = body1_name is not None and self.grasp_manager.is_grasped(body1_name)
            body2_is_grasped = body2_name is not None and self.grasp_manager.is_grasped(body2_name)

            # Treat grasped objects as part of the robot
            body1_is_robot = body1_is_arm or body1_is_grasped
            body2_is_robot = body2_is_arm or body2_is_grasped

            # Skip if neither body is robot (e.g., two objects on table)
            if not body1_is_robot and not body2_is_robot:
                continue

            # Both bodies are "robot" (arm or grasped object)
            if body1_is_robot and body2_is_robot:
                # Check if this is an expected gripper-object contact
                if self._is_gripper_object_contact(body1, body1_name, body1_is_arm, body1_is_grasped,
                                                    body2, body2_name, body2_is_arm, body2_is_grasped):
                    if debug:
                        print(f"  [OK] Gripper-object contact: {body1_name} <-> {body2_name}")
                    continue
                # Unexpected: arm self-collision or grasped object hitting non-gripper arm part
                if debug:
                    print(f"  [INVALID] Robot self-collision: {body1_name} <-> {body2_name}")
                invalid_count += 1
                continue

            # One is robot (arm or grasped object), one is environment
            if debug:
                robot_body = body1_name if body1_is_robot else body2_name
                env_body = body2_name if body1_is_robot else body1_name
                print(f"  [INVALID] Robot-environment: {robot_body} <-> {env_body}")
            invalid_count += 1

        return invalid_count

    def debug_contacts(self, q: np.ndarray) -> None:
        """Debug helper to print all contacts for a configuration."""
        # Copy current state to temp data
        self._temp_data.qpos[:] = self.data.qpos
        self._temp_data.qvel[:] = self.data.qvel

        # Set joint positions
        for i, qpos_idx in enumerate(self.joint_indices):
            self._temp_data.qpos[qpos_idx] = q[i]

        mujoco.mj_forward(self.model, self._temp_data)
        self.grasp_manager.update_attached_poses(self._temp_data)
        mujoco.mj_forward(self.model, self._temp_data)

        print(f"Total contacts: {self._temp_data.ncon}")
        print(f"Grasped objects: {list(self.grasp_manager.grasped.keys())}")
        print(f"Attached objects: {self.grasp_manager.get_attached_objects()}")
        self._count_invalid_contacts(self._temp_data, debug=True)

    def _is_gripper_object_contact(
        self,
        body1: int, body1_name: str | None, body1_is_arm: bool, body1_is_grasped: bool,
        body2: int, body2_name: str | None, body2_is_arm: bool, body2_is_grasped: bool,
    ) -> bool:
        """Check if contact is between a grasped object and the gripper holding it.

        Allows contacts between the grasped object and ANY part of the gripper
        (both fingers, finger pads, etc.), not just the specific attachment body.
        This is important because a grasped object naturally touches both fingers.
        """
        # Identify which is the grasped object and which is the arm body
        if body1_is_grasped and body2_is_arm:
            grasped_name = body1_name
            arm_body_id = body2
        elif body2_is_grasped and body1_is_arm:
            grasped_name = body2_name
            arm_body_id = body1
        else:
            # Both are arm or both are grasped - not a gripper-object contact
            return False

        # Get the gripper body that's holding this object
        gripper_body_name = self.grasp_manager.get_attachment_body(grasped_name)
        if gripper_body_name is None:
            return False

        # Find the gripper base body (common ancestor of all gripper parts)
        # The attachment body name is like "right_ur5e/gripper/right_follower"
        # We want to allow contacts with all parts under "right_ur5e/gripper/base"
        gripper_base_name = self._get_gripper_base_name(gripper_body_name)
        if gripper_base_name is None:
            # Fallback: use attachment body directly
            gripper_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_body_name)
        else:
            gripper_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_base_name)

        if gripper_base_id == -1:
            return False

        # Check if the arm body in contact is the gripper base or a descendant of it
        gripper_body_ids = self._get_body_and_descendants(gripper_base_id)
        return arm_body_id in gripper_body_ids

    def _get_gripper_base_name(self, attachment_body_name: str) -> str | None:
        """Find the gripper base body name from an attachment body name.

        Given "right_ur5e/gripper/right_follower", returns "right_ur5e/gripper/base"
        if it exists in the model. This allows contacts with all gripper parts.
        """
        # Extract the gripper prefix (e.g., "right_ur5e/gripper")
        parts = attachment_body_name.rsplit("/", 1)
        if len(parts) < 2:
            return None

        gripper_prefix = parts[0]  # "right_ur5e/gripper"
        gripper_base_name = f"{gripper_prefix}/base"

        # Check if this body exists
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_base_name)
        if body_id != -1:
            return gripper_base_name

        return None

    def _get_body_and_descendants(self, body_id: int) -> set[int]:
        """Get a body ID and all its descendant body IDs."""
        result = {body_id}
        for i in range(self.model.nbody):
            if self.model.body_parentid[i] == body_id:
                result.update(self._get_body_and_descendants(i))
        return result


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
