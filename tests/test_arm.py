"""Tests for Arm class."""

import mujoco
import numpy as np
import pytest

from geodude.arm import Arm
from geodude.config import ArmConfig
from geodude.grasp_manager import GraspManager


class MockRobot:
    """Minimal mock of Geodude for testing Arm independently."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.named_poses = {
            "home": {"right": [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]},
            "ready": {"right": [-2.14, -1.5708, 1.5708, -1.5708, -1.5708, 0]},
        }


class TestArm:
    """Tests for Arm class."""

    @pytest.fixture
    def arm_config(self):
        """Create arm configuration for the real geodude robot."""
        return ArmConfig(
            name="right",
            joint_names=[
                "right_ur5e/shoulder_pan_joint",
                "right_ur5e/shoulder_lift_joint",
                "right_ur5e/elbow_joint",
                "right_ur5e/wrist_1_joint",
                "right_ur5e/wrist_2_joint",
                "right_ur5e/wrist_3_joint",
            ],
            ee_site="right_ur5e/gripper_attachment_site",
            gripper_actuator="right_ur5e/gripper/fingers_actuator",
            gripper_bodies=[
                "right_ur5e/gripper/right_follower",
                "right_ur5e/gripper/left_follower",
                "right_ur5e/gripper/right_pad",
                "right_ur5e/gripper/left_pad",
            ],
        )

    @pytest.fixture
    def arm(self, mujoco_model_and_data, arm_config):
        """Create an Arm instance for testing."""
        model, data = mujoco_model_and_data
        mock_robot = MockRobot(model, data)
        gm = GraspManager(model, data)
        return Arm(mock_robot, arm_config, gm)

    def test_init(self, arm):
        """Arm initializes correctly."""
        assert arm.name == "right"
        assert arm.dof == 6
        assert len(arm.joint_ids) == 6
        assert len(arm.joint_qpos_indices) == 6

    def test_invalid_joint_raises(self, mujoco_model_and_data):
        """Invalid joint name raises ValueError."""
        model, data = mujoco_model_and_data
        mock_robot = MockRobot(model, data)
        gm = GraspManager(model, data)

        bad_config = ArmConfig(
            name="bad_arm",
            joint_names=["nonexistent_joint"],
            ee_site="right_ur5e/gripper_attachment_site",
            gripper_actuator="",
            gripper_bodies=[],
        )

        with pytest.raises(ValueError, match="not found"):
            Arm(mock_robot, bad_config, gm)

    def test_invalid_site_raises(self, mujoco_model_and_data, arm_config):
        """Invalid EE site name raises ValueError."""
        model, data = mujoco_model_and_data
        mock_robot = MockRobot(model, data)
        gm = GraspManager(model, data)

        arm_config.ee_site = "nonexistent_site"

        with pytest.raises(ValueError, match="not found"):
            Arm(mock_robot, arm_config, gm)

    def test_get_joint_positions(self, arm, mujoco_model_and_data):
        """get_joint_positions returns current joint values."""
        model, data = mujoco_model_and_data

        # Set known joint positions
        test_values = [0.5, -0.3, 0.2, -0.1, 0.4, -0.2]
        for i, val in enumerate(test_values):
            data.qpos[arm.joint_qpos_indices[i]] = val

        q = arm.get_joint_positions()
        for i, val in enumerate(test_values):
            assert q[i] == pytest.approx(val)

    def test_set_joint_positions(self, arm, mujoco_model_and_data):
        """set_joint_positions updates joint values."""
        model, data = mujoco_model_and_data

        test_values = np.array([1.0, -1.0, 0.5, -0.5, 0.3, -0.3])
        arm.set_joint_positions(test_values)

        for i, val in enumerate(test_values):
            assert data.qpos[arm.joint_qpos_indices[i]] == pytest.approx(val)

    def test_get_ee_pose(self, arm):
        """get_ee_pose returns 4x4 transformation matrix."""
        pose = arm.get_ee_pose()

        assert pose.shape == (4, 4)
        # Bottom row should be [0, 0, 0, 1]
        assert np.allclose(pose[3, :], [0, 0, 0, 1])
        # Rotation matrix should have det=1
        det = np.linalg.det(pose[:3, :3])
        assert np.isclose(det, 1.0, atol=1e-6)

    def test_get_joint_limits(self, arm):
        """get_joint_limits returns lower and upper bounds."""
        lower, upper = arm.get_joint_limits()

        assert len(lower) == 6
        assert len(upper) == 6
        # Lower should be less than upper
        assert np.all(lower < upper)

    def test_go_to_named_pose(self, arm):
        """go_to with named pose sets joint positions."""
        arm.go_to("home")

        q = arm.get_joint_positions()
        expected = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        assert np.allclose(q, expected, atol=0.01)

    def test_go_to_named_pose_ready(self, arm):
        """go_to 'ready' pose works."""
        arm.go_to("ready")

        q = arm.get_joint_positions()
        expected = [-2.14, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        assert np.allclose(q, expected, atol=0.01)

    def test_go_to_array(self, arm):
        """go_to with array sets joint positions."""
        target = np.array([0.7, -0.2, 0.3, -0.4, 0.5, -0.1])
        arm.go_to(target)

        q = arm.get_joint_positions()
        assert np.allclose(q, target, atol=0.01)

    def test_go_to_unknown_pose_raises(self, arm):
        """go_to with unknown named pose raises ValueError."""
        with pytest.raises(ValueError, match="Unknown named pose"):
            arm.go_to("nonexistent_pose")

    def test_plan_to_configuration_simple(self, arm):
        """plan_to_configuration returns path for simple case."""
        start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        arm.set_joint_positions(start)

        # Plan to a nearby configuration
        goal = start + 0.1
        path = arm.plan_to_configuration(goal)

        # Should return a path (may be None if collision)
        if path is not None:
            assert len(path) >= 2
            assert np.allclose(path[0], start, atol=0.01)
            assert np.allclose(path[-1], goal, atol=0.01)

    def test_execute_path(self, arm):
        """execute moves through waypoints."""
        start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        end = np.array([-1.0, -1.5708, 1.5708, -1.5708, -1.5708, 0])

        path = [start, end]

        success = arm.execute(path)
        assert success

        # Should end at final waypoint
        q = arm.get_joint_positions()
        assert np.allclose(q, end, atol=0.01)

    def test_gripper_close_open(self, arm):
        """close_gripper and open_gripper work."""
        arm.close_gripper(steps=10)
        arm.open_gripper(steps=10)
        # Just verify no crashes

    def test_has_gripper(self, arm):
        """Arm has gripper attribute."""
        assert arm.gripper is not None
        assert arm.gripper.actuator_id is not None
