"""Tests for Geodude robot class."""

from pathlib import Path

import numpy as np
import pytest

from geodude.config import GeodudConfig
from geodude.robot import Geodude


class TestGeodude:
    """Tests for Geodude class."""

    def test_init(self):
        """Geodude initializes correctly with default config."""
        robot = Geodude()
        assert robot.model is not None
        assert robot.data is not None
        assert robot.left_arm is not None
        assert robot.right_arm is not None
        assert robot.grasp_manager is not None

    def test_model_not_found_raises(self):
        """Missing model file raises FileNotFoundError."""
        config = GeodudConfig.default()
        config.model_path = Path("/nonexistent/path/to/model.xml")
        with pytest.raises(FileNotFoundError):
            Geodude(config)

    def test_left_arm_has_6_joints(self):
        """Left arm has 6 joints (UR5e)."""
        robot = Geodude()
        assert robot.left_arm.dof == 6

    def test_right_arm_has_6_joints(self):
        """Right arm has 6 joints (UR5e)."""
        robot = Geodude()
        assert robot.right_arm.dof == 6

    def test_arm_names(self):
        """Arms have correct names."""
        robot = Geodude()
        assert robot.left_arm.config.name == "left"
        assert robot.right_arm.config.name == "right"

    def test_named_poses(self):
        """named_poses returns configuration dict from keyframes."""
        robot = Geodude()
        assert "ready" in robot.named_poses

    def test_named_poses_have_left_right(self):
        """Named poses contain left and right arm configs."""
        robot = Geodude()
        ready = robot.named_poses["ready"]
        assert "left" in ready
        assert "right" in ready
        assert len(ready["left"]) == 6
        assert len(ready["right"]) == 6

    def test_forward(self):
        """forward runs FK without error."""
        robot = Geodude()
        robot.forward()

    def test_reset(self):
        """reset returns to initial state."""
        robot = Geodude()
        for i, idx in enumerate(robot.right_arm.joint_qpos_indices):
            robot.data.qpos[idx] = 0.5
        robot.reset()
        # After reset, time should be 0
        assert robot.data.time == 0

    def test_grippers_attached(self):
        """Both arms have grippers."""
        robot = Geodude()
        assert robot.left_arm.gripper is not None
        assert robot.right_arm.gripper is not None

    def test_gripper_actuator_ids(self):
        """Both grippers have valid actuator IDs."""
        robot = Geodude()
        assert robot.left_arm.gripper.actuator_id is not None
        assert robot.right_arm.gripper.actuator_id is not None

    def test_ik_solvers_attached(self):
        """Both arms have IK solvers."""
        robot = Geodude()
        assert robot.left_arm.ik_solver is not None
        assert robot.right_arm.ik_solver is not None

    def test_resolve_arms_both(self):
        """_resolve_arms(None) returns both arms."""
        robot = Geodude()
        arms = robot._resolve_arms(None)
        assert len(arms) == 2

    def test_resolve_arms_left(self):
        """_resolve_arms('left') returns left arm only."""
        robot = Geodude()
        arms = robot._resolve_arms("left")
        assert len(arms) == 1
        assert arms[0] is robot.left_arm

    def test_resolve_arms_invalid_raises(self):
        """_resolve_arms with invalid name raises."""
        robot = Geodude()
        with pytest.raises(ValueError):
            robot._resolve_arms("invalid_arm")

    def test_sim_context_creation(self):
        """sim() creates a SimContext that can be entered."""
        robot = Geodude()
        with robot.sim(headless=True) as ctx:
            assert ctx is not None
            assert ctx.is_running()

    def test_get_arm_spec(self):
        """get_arm_spec returns correct spec for each arm."""
        robot = Geodude()
        left_spec = robot.get_arm_spec(robot.left_arm)
        right_spec = robot.get_arm_spec(robot.right_arm)
        assert "left_ur5e" in left_spec.prefix
        assert "right_ur5e" in right_spec.prefix


class TestGeodueWithObjects:
    """Tests for Geodude with objects in the scene."""

    @pytest.fixture
    def robot_with_object(self, geodude_xml):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(geodude_xml)
            temp_path = Path(f.name)

        config = GeodudConfig.default()
        config.model_path = temp_path
        robot = Geodude(config)
        yield robot
        temp_path.unlink()

    def test_get_object_pose(self, robot_with_object):
        """get_object_pose returns 4x4 matrix."""
        pose = robot_with_object.get_object_pose("box1")
        assert pose.shape == (4, 4)
        assert np.allclose(pose[3, :], [0, 0, 0, 1])

    def test_get_object_pose_unknown_raises(self, robot_with_object):
        """get_object_pose with unknown object raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            robot_with_object.get_object_pose("nonexistent_object")


class TestEndEffectors:
    """Tests for end effector functionality."""

    def test_left_arm_ee_pose_valid(self):
        """Left arm EE pose is a valid transform."""
        robot = Geodude()
        pose = robot.left_arm.get_ee_pose()
        assert pose.shape == (4, 4)
        det = np.linalg.det(pose[:3, :3])
        assert np.isclose(det, 1.0, atol=1e-6)

    def test_right_arm_ee_pose_valid(self):
        """Right arm EE pose is a valid transform."""
        robot = Geodude()
        pose = robot.right_arm.get_ee_pose()
        assert pose.shape == (4, 4)
        det = np.linalg.det(pose[:3, :3])
        assert np.isclose(det, 1.0, atol=1e-6)


class TestJointLimits:
    """Tests for joint limits."""

    def test_joint_limits_reasonable(self):
        """Joint limits are reasonable for UR5e."""
        robot = Geodude()
        lower, upper = robot.right_arm.get_joint_limits()
        assert np.all(lower < 0)
        assert np.all(upper > 0)
        assert np.all(upper - lower > 3)
