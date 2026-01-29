"""Tests for Geodude robot class."""

import mujoco
import numpy as np
import pytest
from pathlib import Path

from geodude.config import GeodudConfig, ArmConfig
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

    def test_left_arm_property(self):
        """left_arm property returns Arm."""
        robot = Geodude()
        assert robot.left_arm.name == "left"

    def test_right_arm_property(self):
        """right_arm property returns Arm."""
        robot = Geodude()
        assert robot.right_arm.name == "right"

    def test_left_arm_has_6_joints(self):
        """Left arm has 6 joints (UR5e)."""
        robot = Geodude()
        assert robot.left_arm.dof == 6

    def test_right_arm_has_6_joints(self):
        """Right arm has 6 joints (UR5e)."""
        robot = Geodude()
        assert robot.right_arm.dof == 6

    def test_named_poses(self):
        """named_poses returns configuration dict from keyframes."""
        robot = Geodude()
        assert "ready" in robot.named_poses

    def test_go_to_ready(self):
        """go_to 'ready' moves both arms to ready pose (from keyframe)."""
        robot = Geodude()
        success = robot.go_to("ready")

        assert success
        # Arms have mirrored configurations (shoulder_pan differs in sign)
        expected_left = [-1.5708, -1.5708, 1.5708, -1.5708, 1.5708, 0]
        expected_right = [1.5708, -1.5708, 1.5708, -1.5708, 1.5708, 0]
        assert np.allclose(robot.left_arm.get_joint_positions(), expected_left, atol=0.01)
        assert np.allclose(robot.right_arm.get_joint_positions(), expected_right, atol=0.01)

    def test_go_to_unknown_pose_raises(self):
        """go_to with unknown pose raises ValueError."""
        robot = Geodude()

        with pytest.raises(ValueError, match="Unknown named pose"):
            robot.go_to("nonexistent")

    def test_step(self):
        """step advances simulation."""
        robot = Geodude()
        t0 = robot.get_time()

        robot.step(10)

        t1 = robot.get_time()
        assert t1 > t0

    def test_forward(self):
        """forward runs FK without stepping."""
        robot = Geodude()
        # Just verify it doesn't crash
        robot.forward()

    def test_get_time(self):
        """get_time returns simulation time."""
        robot = Geodude()
        assert robot.get_time() >= 0

    def test_reset(self):
        """reset returns to initial state."""
        robot = Geodude()

        # Modify state
        robot.step(100)
        robot.right_arm.set_joint_positions(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))

        # Reset
        robot.reset()

        # Should be back at initial time
        assert robot.get_time() == 0


class TestGeodueWithObjects:
    """Tests for Geodude with objects in the scene."""

    @pytest.fixture
    def robot_with_object(self, geodude_xml):
        """Create Geodude from XML with test objects."""
        import tempfile

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(geodude_xml)
            temp_path = Path(f.name)

        config = GeodudConfig.default()
        config.model_path = temp_path
        robot = Geodude(config)

        yield robot

        # Cleanup
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

    def test_set_object_pose(self, robot_with_object):
        """set_object_pose updates object position."""
        new_pose = np.eye(4)
        new_pose[:3, 3] = [0.1, 0.2, 0.3]

        robot_with_object.set_object_pose("box1", new_pose)
        result = robot_with_object.get_object_pose("box1")

        assert np.allclose(result[:3, 3], [0.1, 0.2, 0.3], atol=1e-6)

    def test_set_object_pose_unknown_raises(self, robot_with_object):
        """set_object_pose with unknown object raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            robot_with_object.set_object_pose("nonexistent", np.eye(4))


class TestGeodueEndEffectors:
    """Tests for end effector functionality."""

    def test_left_arm_ee_pose_valid_transform(self):
        """Left arm EE pose is a valid transform."""
        robot = Geodude()
        robot.go_to("ready")
        pose = robot.left_arm.get_ee_pose()

        assert pose.shape == (4, 4)
        # Check it's a valid rotation matrix (det=1)
        det = np.linalg.det(pose[:3, :3])
        assert np.isclose(det, 1.0, atol=1e-6)

    def test_right_arm_ee_pose_valid_transform(self):
        """Right arm EE pose is a valid transform."""
        robot = Geodude()
        robot.go_to("ready")
        pose = robot.right_arm.get_ee_pose()

        assert pose.shape == (4, 4)
        det = np.linalg.det(pose[:3, :3])
        assert np.isclose(det, 1.0, atol=1e-6)

    def test_right_gripper_actuator(self):
        """Right gripper has a working actuator."""
        robot = Geodude()
        assert robot.right_arm.gripper.actuator_id is not None

        # Can set gripper position
        robot.right_arm.gripper.set_position(0.5)
        assert robot.right_arm.gripper.get_position() == pytest.approx(0.5, abs=0.1)

    def test_left_arm_no_gripper_actuator(self):
        """Left arm has no gripper actuator (as per default geodude.xml)."""
        robot = Geodude()
        # Left arm gripper has no actuator in default geodude.xml
        assert robot.left_arm.gripper.actuator_id is None


class TestJointLimits:
    """Tests for joint limits."""

    def test_joint_limits_reasonable(self):
        """Joint limits are reasonable for UR5e."""
        robot = Geodude()
        lower, upper = robot.right_arm.get_joint_limits()

        # UR5e joints have ~2π range
        assert np.all(lower < 0)
        assert np.all(upper > 0)
        assert np.all(upper - lower > 3)  # At least ~180 degrees range
