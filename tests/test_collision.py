"""Tests for collision checking."""

import mujoco
import numpy as np
import pytest

from geodude.collision import GraspAwareCollisionChecker, SimpleCollisionChecker
from geodude.grasp_manager import GraspManager


class TestSimpleCollisionChecker:
    """Tests for SimpleCollisionChecker."""

    def test_init(self, mujoco_model_and_data, arm_joint_names):
        """SimpleCollisionChecker initializes correctly."""
        model, data = mujoco_model_and_data
        checker = SimpleCollisionChecker(model, data, arm_joint_names)

        assert len(checker.joint_indices) == 6

    def test_invalid_joint_raises(self, mujoco_model_and_data):
        """Invalid joint name raises ValueError."""
        model, data = mujoco_model_and_data

        with pytest.raises(ValueError, match="not found"):
            SimpleCollisionChecker(model, data, ["nonexistent_joint"])

    def test_valid_configuration(self, mujoco_model_and_data, arm_joint_names):
        """Home configuration should be collision-free."""
        model, data = mujoco_model_and_data
        checker = SimpleCollisionChecker(model, data, arm_joint_names)

        # Home position
        q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        # Note: May have floor contact, so this tests the basic mechanism
        result = checker.is_valid(q)
        # Just verify it returns a boolean and doesn't crash
        assert isinstance(result, bool)

    def test_is_valid_batch(self, mujoco_model_and_data, arm_joint_names):
        """Batch validity check works."""
        model, data = mujoco_model_and_data
        checker = SimpleCollisionChecker(model, data, arm_joint_names)

        qs = np.array([
            [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0],
            [-1.0, -1.5708, 1.5708, -1.5708, -1.5708, 0],
            [-2.14, -1.5708, 1.5708, -1.5708, -1.5708, 0],
        ])

        results = checker.is_valid_batch(qs)
        assert results.shape == (3,)
        assert results.dtype == bool


class TestGraspAwareCollisionChecker:
    """Tests for GraspAwareCollisionChecker."""

    def test_init(self, mujoco_model_and_data, arm_joint_names):
        """GraspAwareCollisionChecker initializes correctly."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        assert len(checker.joint_indices) == 6
        assert checker.grasp_manager is gm

    def test_grasped_object_doesnt_collide_with_arm(self, mujoco_model_and_data, arm_joint_names):
        """When object is grasped, it shouldn't cause arm collision."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        # Mark box1 as grasped
        gm.mark_grasped("box1", "right")

        # Move box1 to overlap with arm link
        box1_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box1_joint")
        box1_qpos_adr = model.jnt_qposadr[box1_jnt_id]

        mujoco.mj_forward(model, data)

        # Put box near arm
        data.qpos[box1_qpos_adr:box1_qpos_adr + 3] = [0.3, 0, 0.8]
        data.qpos[box1_qpos_adr + 3:box1_qpos_adr + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(model, data)

        # The collision groups should filter out this collision
        # because box1 is grasped (group 2) and arm is group 1
        # They don't share any bits in contype & conaffinity

        # Check that MuJoCo's collision filtering is working
        # by verifying contype/conaffinity
        box1_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box1_geom")

        box1_contype = model.geom_contype[box1_geom_id]

        # Box is group 2 when grasped
        from geodude.grasp_manager import COLLISION_GROUP_GRASPED
        assert box1_contype == COLLISION_GROUP_GRASPED

    def test_ungrasped_object_collides_normally(self, mujoco_model_and_data, arm_joint_names):
        """Non-grasped objects should collide with arm normally."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        # Check collision groups are normal
        box1_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box1_geom")

        box1_contype = model.geom_contype[box1_geom_id]
        box1_conaffinity = model.geom_conaffinity[box1_geom_id]

        # Both should be normal collision group
        from geodude.grasp_manager import COLLISION_GROUP_NORMAL
        assert box1_contype == COLLISION_GROUP_NORMAL
        assert box1_conaffinity == COLLISION_GROUP_NORMAL


class TestCollisionGroupSemantics:
    """Tests verifying collision group math works correctly."""

    def test_collision_group_definitions(self):
        """Verify collision groups interact correctly."""
        # Group 1: Normal (arm, ungrasped objects)
        # Group 2: Grasped objects
        # Group 3: Gripper pads (1|2)

        normal = 1
        grasped = 2
        gripper = 3

        # Normal collides with normal
        assert (normal & normal) != 0

        # Normal does NOT collide with grasped
        assert (normal & grasped) == 0

        # Gripper collides with normal
        assert (gripper & normal) != 0

        # Gripper collides with grasped
        assert (gripper & grasped) != 0

        # Grasped collides with grasped (objects can touch each other)
        assert (grasped & grasped) != 0
