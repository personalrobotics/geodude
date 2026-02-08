"""Collision checking with grasp awareness.

Collision detection is handled entirely in software by filtering MuJoCo contacts.
No collision group changes are needed - we just check `is_grasped()` to determine
which contacts are expected (gripper-object) vs invalid (arm-object, object-environment).
"""

from __future__ import annotations

import logging
import mujoco
import numpy as np

from geodude.grasp_manager import GraspManager

logger = logging.getLogger(__name__)


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

        # Run forward kinematics again to generate contacts with updated object poses
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

    def is_arm_in_collision(
        self, q: np.ndarray | None = None, min_penetration: float = 0.005
    ) -> bool:
        """Check if arm links (not gripper or grasped objects) are colliding with environment.

        This is used during reactive Cartesian control (lift, etc.) where we want to:
        - ALLOW grasped-object-to-environment contacts (lifting out of surface)
        - ALLOW gripper-to-grasped-object contacts (holding the object)
        - DISALLOW arm-link-to-environment contacts (forearm hitting base, etc.)

        Args:
            q: Joint configuration to check. If None, uses current data state.
            min_penetration: Minimum penetration depth (in meters) to consider a collision.
                In physics mode, minor surface contacts due to controller drift should be
                ignored. Only contacts with penetration deeper than this threshold are
                reported. Default is 5mm.

        Returns:
            True if arm is in collision with environment, False otherwise
        """
        if q is not None:
            # Copy current state to temp data
            self._temp_data.qpos[:] = self.data.qpos
            self._temp_data.qvel[:] = self.data.qvel

            # Set joint positions
            for i, qpos_idx in enumerate(self.joint_indices):
                self._temp_data.qpos[qpos_idx] = q[i]

            mujoco.mj_forward(self.model, self._temp_data)
            self.grasp_manager.update_attached_poses(self._temp_data)
            mujoco.mj_forward(self.model, self._temp_data)
            data = self._temp_data
        else:
            data = self.data

        # Log grasp state for debugging
        grasped_objects = list(self.grasp_manager.grasped.keys())
        logger.debug(f"is_arm_in_collision: grasped_objects={grasped_objects}, ncon={data.ncon}")

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

            # Check if this is arm-link (not gripper) vs environment
            # Gripper bodies contain "gripper" in the name
            body1_is_gripper = body1_is_arm and body1_name is not None and "gripper" in body1_name
            body2_is_gripper = body2_is_arm and body2_name is not None and "gripper" in body2_name

            body1_is_arm_link = body1_is_arm and not body1_is_gripper
            body2_is_arm_link = body2_is_arm and not body2_is_gripper

            body1_is_env = not body1_is_arm and not body1_is_grasped
            body2_is_env = not body2_is_arm and not body2_is_grasped

            # Log all contacts for debugging
            logger.debug(
                f"  contact {i}: {body1_name} <-> {body2_name} | "
                f"arm={body1_is_arm},{body2_is_arm} gripper={body1_is_gripper},{body2_is_gripper} "
                f"grasped={body1_is_grasped},{body2_is_grasped} arm_link={body1_is_arm_link},{body2_is_arm_link} "
                f"env={body1_is_env},{body2_is_env}"
            )

            # Arm link vs environment = bad (if penetration exceeds threshold)
            if (body1_is_arm_link and body2_is_env) or (body2_is_arm_link and body1_is_env):
                # Check penetration depth - contact.dist is negative for penetration
                penetration = -contact.dist
                if penetration < min_penetration:
                    logger.debug(
                        f"  Ignoring minor contact (penetration={penetration*1000:.1f}mm < {min_penetration*1000:.0f}mm)"
                    )
                    continue

                arm_body = body1_name if body1_is_arm_link else body2_name
                env_body = body2_name if body1_is_arm_link else body1_name
                logger.debug(
                    f"Arm link collision: {arm_body} <-> {env_body} "
                    f"(penetration={penetration*1000:.1f}mm)"
                )
                return True

        return False

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
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
                contact_pos = contact.pos
                print(f"  [INVALID] Robot-environment: {robot_body} <-> {env_body}")
                print(f"            Geoms: {geom1_name} <-> {geom2_name}, contact_pos={contact_pos.round(3)}")
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


class CollisionChecker:
    """Unified collision checker with grasp-aware filtering.

    Takes grasp state as constructor arguments (not a live GraspManager),
    making it suitable for both single-threaded and parallel planning.
    Each instance uses its own MjData for thread safety.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_names: list[str],
        grasped_objects: frozenset[tuple[str, str]],
        attachments: dict[str, tuple[str, np.ndarray]],
    ):
        """Initialize the collision checker.

        Args:
            model: MuJoCo model (shared, read-only)
            data: MjData for this checker (should be a private copy)
            joint_names: Names of joints to control
            grasped_objects: Frozenset of (object_name, arm_name) tuples
            attachments: Dict of {object_name: (gripper_body_name, T_gripper_object)}
        """
        self.model = model
        self.data = data
        self._grasped_objects = grasped_objects
        self._attachments = attachments

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

    def _is_grasped(self, body_name: str) -> bool:
        """Check if a body is grasped."""
        return any(obj == body_name for obj, _ in self._grasped_objects)

    def _get_attachment_body(self, object_name: str) -> str | None:
        """Get the gripper body that an object is attached to."""
        if object_name in self._attachments:
            return self._attachments[object_name][0]
        return None

    def is_valid(self, q: np.ndarray) -> bool:
        """Check if a configuration is collision-free.

        Args:
            q: Joint configuration (only the controlled joints)

        Returns:
            True if collision-free, False otherwise
        """
        # Set joint positions
        for i, qpos_idx in enumerate(self.joint_indices):
            self.data.qpos[qpos_idx] = q[i]

        # Run forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Update attached object poses
        self._update_attached_poses()

        # Run forward kinematics again to generate contacts with updated object poses
        mujoco.mj_forward(self.model, self.data)

        # Check for collisions
        return self._count_invalid_contacts() == 0

    def is_valid_batch(self, qs: np.ndarray) -> np.ndarray:
        """Check multiple configurations for collisions."""
        results = np.zeros(len(qs), dtype=bool)
        for i, q in enumerate(qs):
            results[i] = self.is_valid(q)
        return results

    def _update_attached_poses(self) -> None:
        """Update poses of attached objects."""
        for obj_name, (gripper_body_name, T_gripper_object) in self._attachments.items():
            gripper_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_body_name
            )
            if gripper_body_id == -1:
                continue

            # Get current gripper pose
            pos = self.data.xpos[gripper_body_id].copy()
            mat = self.data.xmat[gripper_body_id].reshape(3, 3).copy()
            T_world_gripper = np.eye(4)
            T_world_gripper[:3, :3] = mat
            T_world_gripper[:3, 3] = pos

            # Compute new object pose
            T_world_object = T_world_gripper @ T_gripper_object

            # Set object pose
            self._set_body_pose(obj_name, T_world_object)

    def _set_body_pose(self, body_name: str, T: np.ndarray) -> None:
        """Set the pose of a freejoint body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return

        joint_id = self.model.body_jntadr[body_id]
        if joint_id == -1:
            return

        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            return

        qpos_adr = self.model.jnt_qposadr[joint_id]

        pos = T[:3, 3]
        mat = T[:3, :3]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())

        self.data.qpos[qpos_adr : qpos_adr + 3] = pos
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat

    def _count_invalid_contacts(self) -> int:
        """Count contacts that indicate invalid collisions."""
        invalid_count = 0

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]

            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)

            body1_is_arm = body1 in self._arm_body_ids
            body2_is_arm = body2 in self._arm_body_ids
            body1_is_grasped = body1_name is not None and self._is_grasped(body1_name)
            body2_is_grasped = body2_name is not None and self._is_grasped(body2_name)

            body1_is_robot = body1_is_arm or body1_is_grasped
            body2_is_robot = body2_is_arm or body2_is_grasped

            if not body1_is_robot and not body2_is_robot:
                continue

            if body1_is_robot and body2_is_robot:
                if self._is_gripper_object_contact(
                    body1, body1_name, body1_is_arm, body1_is_grasped,
                    body2, body2_name, body2_is_arm, body2_is_grasped,
                ):
                    continue
                invalid_count += 1
                continue

            invalid_count += 1

        return invalid_count

    def _is_gripper_object_contact(
        self,
        body1: int, body1_name: str | None, body1_is_arm: bool, body1_is_grasped: bool,
        body2: int, body2_name: str | None, body2_is_arm: bool, body2_is_grasped: bool,
    ) -> bool:
        """Check if contact is between a grasped object and the gripper holding it."""
        if body1_is_grasped and body2_is_arm:
            grasped_name = body1_name
            arm_body_id = body2
        elif body2_is_grasped and body1_is_arm:
            grasped_name = body2_name
            arm_body_id = body1
        else:
            return False

        gripper_body_name = self._get_attachment_body(grasped_name)
        if gripper_body_name is None:
            return False

        gripper_base_name = self._get_gripper_base_name(gripper_body_name)
        if gripper_base_name is None:
            gripper_base_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_body_name
            )
        else:
            gripper_base_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_base_name
            )

        if gripper_base_id == -1:
            return False

        gripper_body_ids = self._get_body_and_descendants(gripper_base_id)
        return arm_body_id in gripper_body_ids

    def _get_gripper_base_name(self, attachment_body_name: str) -> str | None:
        """Find the gripper base body name from an attachment body name."""
        parts = attachment_body_name.rsplit("/", 1)
        if len(parts) < 2:
            return None

        gripper_prefix = parts[0]
        gripper_base_name = f"{gripper_prefix}/base"

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
