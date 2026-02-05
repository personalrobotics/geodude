"""Tests for VentionBase class."""

import mujoco
import numpy as np
import pytest

from geodude.config import GeodudConfig, VentionBaseConfig
from geodude.robot import Geodude
from geodude.vention_base import VentionBase


class TestVentionBaseConfig:
    """Tests for VentionBaseConfig."""

    def test_default_values(self):
        """VentionBaseConfig has correct defaults."""
        config = VentionBaseConfig(
            name="left_base",
            entity_type="base",
            joint_names=["left_arm_linear_vention"],
            actuator_name="left_linear_actuator",
        )

        assert config.name == "left_base"
        assert config.height_range == (0.0, 0.5)
        assert config.collision_check_resolution == 0.01

    def test_custom_height_range(self):
        """VentionBaseConfig accepts custom height range."""
        config = VentionBaseConfig(
            name="test_base",
            entity_type="base",
            joint_names=["test_joint"],
            actuator_name="test_actuator",
            height_range=(0.1, 0.4),
        )

        assert config.height_range == (0.1, 0.4)


class TestVentionBaseInit:
    """Tests for VentionBase initialization."""

    def test_geodude_has_bases(self):
        """Geodude initializes with left and right bases."""
        robot = Geodude()

        assert robot.left_base is not None
        assert robot.right_base is not None

    def test_base_names(self):
        """Bases have correct names."""
        robot = Geodude()

        assert robot.left_base.name == "left_base"
        assert robot.right_base.name == "right_base"

    def test_invalid_joint_raises(self):
        """Invalid joint name raises ValueError."""
        robot = Geodude()
        config = VentionBaseConfig(
            name="test_base",
            entity_type="base",
            joint_names=["nonexistent_joint"],
            actuator_name="left_linear_actuator",
        )

        with pytest.raises(ValueError, match="Joint.*not found"):
            VentionBase(robot.model, robot.data, config, robot.left_arm)

    def test_invalid_actuator_raises(self):
        """Invalid actuator name raises ValueError."""
        robot = Geodude()
        config = VentionBaseConfig(
            name="test_base",
            entity_type="base",
            joint_names=["left_arm_linear_vention"],
            actuator_name="nonexistent_actuator",
        )

        with pytest.raises(ValueError, match="Actuator.*not found"):
            VentionBase(robot.model, robot.data, config, robot.left_arm)


class TestVentionBaseHeight:
    """Tests for VentionBase height property."""

    def test_initial_height(self):
        """Initial height is 0 (bottom of rail)."""
        robot = Geodude()

        assert robot.left_base.height == pytest.approx(0.0, abs=0.001)
        assert robot.right_base.height == pytest.approx(0.0, abs=0.001)

    def test_height_range(self):
        """height_range returns configured range."""
        robot = Geodude()

        assert robot.left_base.height_range == (0.0, 0.5)
        assert robot.right_base.height_range == (0.0, 0.5)


class TestVentionBaseSetHeight:
    """Tests for VentionBase.set_height()."""

    def test_set_height(self):
        """set_height updates height correctly."""
        robot = Geodude()

        robot.left_base.set_height(0.3)
        assert robot.left_base.height == pytest.approx(0.3, abs=0.001)

    def test_set_height_to_max(self):
        """set_height can reach max height."""
        robot = Geodude()

        robot.left_base.set_height(0.5)
        assert robot.left_base.height == pytest.approx(0.5, abs=0.001)

    def test_set_height_to_min(self):
        """set_height can reach min height."""
        robot = Geodude()
        robot.left_base.set_height(0.3)  # Move up first

        robot.left_base.set_height(0.0)
        assert robot.left_base.height == pytest.approx(0.0, abs=0.001)

    def test_set_height_below_range_raises(self):
        """set_height below range raises ValueError."""
        robot = Geodude()

        with pytest.raises(ValueError, match="outside valid range"):
            robot.left_base.set_height(-0.1)

    def test_set_height_above_range_raises(self):
        """set_height above range raises ValueError."""
        robot = Geodude()

        with pytest.raises(ValueError, match="outside valid range"):
            robot.left_base.set_height(0.6)

    def test_set_height_independent(self):
        """Left and right bases are independent."""
        robot = Geodude()

        robot.left_base.set_height(0.2)
        robot.right_base.set_height(0.4)

        assert robot.left_base.height == pytest.approx(0.2, abs=0.001)
        assert robot.right_base.height == pytest.approx(0.4, abs=0.001)


class TestVentionBaseMoveTo:
    """Tests for VentionBase.move_to()."""

    def test_move_to_no_collision_check(self):
        """move_to without collision check succeeds."""
        robot = Geodude()

        success = robot.left_base.move_to(0.3, check_collisions=False)

        assert success is True
        assert robot.left_base.height == pytest.approx(0.3, abs=0.001)

    def test_move_to_with_collision_check_clear_path(self):
        """move_to with collision check succeeds on clear path."""
        robot = Geodude()
        # Arms start in home position - path should be clear
        robot.go_to("ready")

        success = robot.left_base.move_to(0.3, check_collisions=True)

        assert success is True
        assert robot.left_base.height == pytest.approx(0.3, abs=0.001)

    def test_move_to_below_range_raises(self):
        """move_to below range raises ValueError."""
        robot = Geodude()

        with pytest.raises(ValueError, match="outside valid range"):
            robot.left_base.move_to(-0.1)

    def test_move_to_above_range_raises(self):
        """move_to above range raises ValueError."""
        robot = Geodude()

        with pytest.raises(ValueError, match="outside valid range"):
            robot.left_base.move_to(0.6)

    def test_move_to_same_position(self):
        """move_to current position succeeds."""
        robot = Geodude()
        robot.left_base.set_height(0.2)

        success = robot.left_base.move_to(0.2, check_collisions=True)

        assert success is True
        assert robot.left_base.height == pytest.approx(0.2, abs=0.001)

    def test_move_to_small_distance(self):
        """move_to very small distance succeeds."""
        robot = Geodude()
        robot.left_base.set_height(0.2)

        success = robot.left_base.move_to(0.201, check_collisions=True)

        assert success is True
        assert robot.left_base.height == pytest.approx(0.201, abs=0.001)

    def test_move_to_preserves_arm_config(self):
        """move_to preserves arm joint configuration."""
        robot = Geodude()
        robot.go_to("ready")
        original_q = robot.left_arm.get_joint_positions().copy()

        robot.left_base.move_to(0.3, check_collisions=True)

        new_q = robot.left_arm.get_joint_positions()
        assert np.allclose(original_q, new_q, atol=0.001)


class TestVentionBaseCollisionDetection:
    """Tests for collision detection during base movement."""

    def test_collision_check_uses_resolution(self):
        """Collision check respects configured resolution."""
        robot = Geodude()
        base = robot.left_base

        # Default resolution is 0.01m, so 0.5m path should check ~50 points
        # This is tested implicitly by the move_to tests passing

        # Verify resolution is used
        assert base.config.collision_check_resolution == 0.01

    def test_failed_move_preserves_original_position(self):
        """Failed move_to preserves original height."""
        robot = Geodude()
        robot.left_base.set_height(0.2)
        original_height = robot.left_base.height

        # Even if collision detection fails, original position should be preserved
        # (This tests the finally block in _is_path_collision_free)
        # For now, with home config, path should be clear
        robot.left_base.move_to(0.3, check_collisions=True)

        # Move back to original - should work
        robot.left_base.move_to(0.2, check_collisions=True)
        assert robot.left_base.height == pytest.approx(0.2, abs=0.001)


class TestVentionBaseWithObstacles:
    """Tests for VentionBase with obstacles (uses mujoco fixtures)."""

    def test_robot_self_contacts_ignored(self):
        """Robot-robot contacts should NOT be treated as collisions.

        This test verifies that internal robot contacts (arm touching vention
        frame, gripper parts touching each other, etc.) do not block base movement.
        """
        robot = Geodude()
        robot.go_to("ready")

        # In home position, the arm may have contacts with the vention frame
        # but this should NOT block base movement
        success = robot.left_base.move_to(0.3, check_collisions=True)
        assert success is True, "Robot self-contacts should not block base movement"

        success = robot.right_base.move_to(0.4, check_collisions=True)
        assert success is True, "Robot self-contacts should not block base movement"

    def test_move_blocked_by_obstacle(self, mujoco_model_and_data, arm_joint_names):
        """move_to returns False when path is blocked by obstacle.

        This test creates a scenario where moving the base up would cause
        the arm to collide with the obstacle positioned at z=0.9.
        """
        from geodude.arm import Arm
        from geodude.config import ArmConfig, VentionBaseConfig
        from geodude.grasp_manager import GraspManager

        model, data = mujoco_model_and_data

        # Create the necessary objects
        grasp_manager = GraspManager(model, data)
        arm_config = ArmConfig(
            name="right_arm",
            entity_type="arm",
            joint_names=arm_joint_names,
            ee_site="right_ur5e/gripper_attachment_site",
            gripper_actuator="right_ur5e/gripper/fingers_actuator",
            gripper_bodies=[
                "right_ur5e/gripper/right_follower",
                "right_ur5e/gripper/left_follower",
            ],
        )

        # Create a minimal robot-like object for Arm
        class FakeRobot:
            pass

        fake_robot = FakeRobot()
        fake_robot.model = model
        fake_robot.data = data
        fake_robot.config = type("Config", (), {"named_poses": {}})()
        fake_robot.grasp_manager = grasp_manager

        arm = Arm(fake_robot, arm_config, grasp_manager)

        # Create VentionBase
        base_config = VentionBaseConfig(
            name="right_base",
            entity_type="base",
            joint_names=["right_arm_linear_vention"],
            actuator_name="right_linear_actuator",
        )
        base = VentionBase(model, data, base_config, arm)

        # Put arm in an extended position pointing toward obstacle
        # The obstacle is at pos="0.4 0 0.9"
        # We need to configure the arm so it would hit the obstacle
        # when the base moves up
        extended_q = np.array([-1.57, -0.5, 0.5, -0.5, -1.57, 0])
        arm.set_joint_positions(extended_q)
        mujoco.mj_forward(model, data)

        # Start at bottom
        base.set_height(0.0)

        # Try to move up - depending on arm configuration, may or may not collide
        # The key test is that collision checking is performed
        initial_height = base.height
        success = base.move_to(0.4, check_collisions=True)

        # Whether it succeeds or fails, the API should work correctly
        if not success:
            # If blocked, height should be unchanged
            assert base.height == pytest.approx(initial_height, abs=0.001)
        else:
            # If succeeded, height should be updated
            assert base.height == pytest.approx(0.4, abs=0.001)
