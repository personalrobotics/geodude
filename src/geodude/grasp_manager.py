"""Grasp state management and collision group updates."""

import mujoco
import numpy as np

# Collision group bit definitions
# Bit 0 (value 1): Normal objects and robot arm
# Bit 1 (value 2): Grasped objects (only collides with gripper pads)
COLLISION_GROUP_NORMAL = 1
COLLISION_GROUP_GRASPED = 2
COLLISION_GROUP_GRIPPER_PADS = 3  # Both bits: collides with normal AND grasped


class GraspManager:
    """Manages grasp state and updates collision groups accordingly.

    When an object is grasped:
    - Its collision group changes from NORMAL (1) to GRASPED (2)
    - This means it no longer collides with the robot arm (group 1)
    - But still collides with gripper pads (group 3 = 1|2)
    - And still collides with environment/other objects

    This allows planning motions with a grasped object without false
    collisions between the object and the robot's arm.

    For kinematic execution (no physics), this class also tracks object
    attachments so grasped objects move with the gripper.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.grasped: dict[str, str] = {}  # object_name -> arm_name
        self._original_collision_groups: dict[str, list[tuple[int, int]]] = {}
        # Kinematic attachments: object_name -> (gripper_body_name, T_gripper_object)
        self._attachments: dict[str, tuple[str, np.ndarray]] = {}

    def mark_grasped(self, object_name: str, arm: str) -> None:
        """Mark an object as grasped by the specified arm.

        Updates the object's collision group so it doesn't collide with
        the robot arm during planning, but still collides with gripper
        pads and environment.

        Args:
            object_name: Name of the grasped object body in MuJoCo
            arm: Which arm is holding it ("left" or "right")
        """
        if object_name in self.grasped:
            return  # Already grasped

        self.grasped[object_name] = arm
        self._save_and_update_collision_group(object_name, grasped=True)

    def mark_released(self, object_name: str) -> None:
        """Mark an object as released.

        Restores the object's original collision group.

        Args:
            object_name: Name of the released object body in MuJoCo
        """
        if object_name not in self.grasped:
            return  # Not currently grasped

        del self.grasped[object_name]
        self._restore_collision_group(object_name)

    def get_grasped_by(self, arm: str) -> list[str]:
        """Get list of objects currently grasped by the specified arm."""
        return [obj for obj, holder in self.grasped.items() if holder == arm]

    def is_grasped(self, object_name: str) -> bool:
        """Check if an object is currently grasped."""
        return object_name in self.grasped

    def get_holder(self, object_name: str) -> str | None:
        """Get the arm holding an object, or None if not grasped."""
        return self.grasped.get(object_name)

    def attach_object(self, object_name: str, gripper_body_name: str) -> None:
        """Attach an object to a gripper for kinematic manipulation.

        Computes and stores the relative transform between gripper and object
        so the object can move with the gripper.

        Args:
            object_name: Name of the object body in MuJoCo
            gripper_body_name: Name of the gripper body to attach to
        """
        # Get current poses
        T_world_gripper = self._get_body_pose(gripper_body_name)
        T_world_object = self._get_body_pose(object_name)

        # Compute relative transform: T_gripper_object = inv(T_world_gripper) @ T_world_object
        T_gripper_object = np.linalg.inv(T_world_gripper) @ T_world_object

        self._attachments[object_name] = (gripper_body_name, T_gripper_object)

    def detach_object(self, object_name: str) -> None:
        """Detach an object from kinematic attachment.

        Args:
            object_name: Name of the object body
        """
        self._attachments.pop(object_name, None)

    def is_attached(self, object_name: str) -> bool:
        """Check if an object is kinematically attached."""
        return object_name in self._attachments

    def get_attached_objects(self) -> list[str]:
        """Get list of all kinematically attached objects."""
        return list(self._attachments.keys())

    def update_attached_poses(self) -> None:
        """Update poses of all kinematically attached objects.

        Call this after moving the gripper to update attached object positions.
        """
        for object_name, (gripper_body_name, T_gripper_object) in self._attachments.items():
            # Get current gripper pose
            T_world_gripper = self._get_body_pose(gripper_body_name)

            # Compute new object pose: T_world_object = T_world_gripper @ T_gripper_object
            T_world_object = T_world_gripper @ T_gripper_object

            # Set object pose
            self._set_body_pose(object_name, T_world_object)

    def _get_body_pose(self, body_name: str) -> np.ndarray:
        """Get the 4x4 pose matrix of a body.

        Args:
            body_name: Name of the body

        Returns:
            4x4 homogeneous transformation matrix
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in model")

        pos = self.data.xpos[body_id].copy()
        mat = self.data.xmat[body_id].reshape(3, 3).copy()

        T = np.eye(4)
        T[:3, :3] = mat
        T[:3, 3] = pos
        return T

    def _set_body_pose(self, body_name: str, T: np.ndarray) -> None:
        """Set the pose of a freejoint body.

        Args:
            body_name: Name of the body (must have a freejoint)
            T: 4x4 homogeneous transformation matrix
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in model")

        # Find the joint for this body
        joint_id = self.model.body_jntadr[body_id]
        if joint_id == -1:
            raise ValueError(f"Body '{body_name}' has no joint - cannot set pose")

        # Check if it's a freejoint
        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Body '{body_name}' joint is not a freejoint")

        # Get qpos address for this joint
        qpos_adr = self.model.jnt_qposadr[joint_id]

        # Extract position and convert rotation matrix to quaternion
        pos = T[:3, 3]
        mat = T[:3, :3]
        quat = self._mat_to_quat(mat)

        # Set qpos: [x, y, z, qw, qx, qy, qz]
        self.data.qpos[qpos_adr:qpos_adr + 3] = pos
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = quat

    def _mat_to_quat(self, mat: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        # Use MuJoCo's conversion
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return quat

    def _get_body_geom_ids(self, body_name: str) -> list[int]:
        """Get all geom IDs belonging to a body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in model")

        geom_ids = []
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == body_id:
                geom_ids.append(geom_id)
        return geom_ids

    def _save_and_update_collision_group(self, object_name: str, grasped: bool) -> None:
        """Save original collision groups and update to grasped state."""
        geom_ids = self._get_body_geom_ids(object_name)

        # Save original values
        self._original_collision_groups[object_name] = [
            (int(self.model.geom_contype[gid]), int(self.model.geom_conaffinity[gid]))
            for gid in geom_ids
        ]

        # Update to grasped collision group
        for geom_id in geom_ids:
            self.model.geom_contype[geom_id] = COLLISION_GROUP_GRASPED
            self.model.geom_conaffinity[geom_id] = COLLISION_GROUP_GRASPED

    def _restore_collision_group(self, object_name: str) -> None:
        """Restore original collision groups for an object."""
        if object_name not in self._original_collision_groups:
            return

        geom_ids = self._get_body_geom_ids(object_name)
        original = self._original_collision_groups.pop(object_name)

        for geom_id, (contype, conaffinity) in zip(geom_ids, original):
            self.model.geom_contype[geom_id] = contype
            self.model.geom_conaffinity[geom_id] = conaffinity


def detect_grasped_object(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_body_names: list[str],
    candidate_objects: list[str] | None = None,
) -> str | None:
    """Detect which object (if any) is being grasped by the gripper.

    Checks MuJoCo contacts to find objects in contact with gripper bodies.

    Args:
        model: MuJoCo model
        data: MuJoCo data (after mj_forward)
        gripper_body_names: Names of gripper bodies (pads, fingers)
        candidate_objects: Optional list of object names to consider.
                          If None, considers all bodies.

    Returns:
        Name of grasped object, or None if nothing is grasped
    """
    # Get gripper body IDs
    gripper_body_ids = set()
    for name in gripper_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            gripper_body_ids.add(body_id)

    if not gripper_body_ids:
        return None

    # Get candidate object body IDs
    candidate_body_ids: set[int] | None = None
    if candidate_objects is not None:
        candidate_body_ids = set()
        for name in candidate_objects:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                candidate_body_ids.add(body_id)

    # Check contacts
    contacted_objects: dict[int, int] = {}  # body_id -> contact_count

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]

        # Check if one is gripper and one is candidate object
        gripper_body = None
        other_body = None

        if body1 in gripper_body_ids:
            gripper_body = body1
            other_body = body2
        elif body2 in gripper_body_ids:
            gripper_body = body2
            other_body = body1

        if gripper_body is None:
            continue

        # Skip if other body is also part of gripper/robot
        if other_body in gripper_body_ids:
            continue

        # Skip if not in candidate list (when specified)
        if candidate_body_ids is not None and other_body not in candidate_body_ids:
            continue

        # Count contacts with this object
        contacted_objects[other_body] = contacted_objects.get(other_body, 0) + 1

    # Return object with most contacts (heuristic for "most grasped")
    if not contacted_objects:
        return None

    best_body_id = max(contacted_objects, key=lambda x: contacted_objects[x])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, best_body_id)
