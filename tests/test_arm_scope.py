# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for the unified _ArmScope interface (robot.left / robot.right)."""

from geodude.robot import Geodude


class TestArmScopeDelegate:
    """robot.left delegates to Arm via __getattr__."""

    def test_get_ee_pose(self):
        robot = Geodude()
        pose = robot.left.get_ee_pose()
        assert pose.shape == (4, 4)

    def test_get_joint_positions(self):
        robot = Geodude()
        q = robot.left.get_joint_positions()
        assert len(q) == 6

    def test_dof(self):
        robot = Geodude()
        assert robot.left.dof == 6

    def test_config_name(self):
        robot = Geodude()
        assert robot.left.config.name == "left"
        assert robot.right.config.name == "right"

    def test_get_ft_wrench(self):
        robot = Geodude()
        wrench = robot.left.get_ft_wrench()
        assert len(wrench) == 6

    def test_gripper_accessible(self):
        robot = Geodude()
        assert robot.left.gripper is not None

    def test_ik_solver_accessible(self):
        robot = Geodude()
        assert robot.left.ik_solver is not None


class TestArmScopeDir:
    """Tab completion includes both ArmScope and Arm methods."""

    def test_dir_has_pickup(self):
        robot = Geodude()
        assert "pickup" in dir(robot.left)

    def test_dir_has_close(self):
        robot = Geodude()
        assert "close" in dir(robot.left)

    def test_dir_has_get_ee_pose(self):
        robot = Geodude()
        assert "get_ee_pose" in dir(robot.left)

    def test_dir_has_plan_to_pose(self):
        robot = Geodude()
        assert "plan_to_pose" in dir(robot.left)


class TestArmScopeCloseOpen:
    """robot.left.close() / open() require active context."""

    def test_close_without_context_raises(self):
        robot = Geodude()
        import pytest

        with pytest.raises(RuntimeError, match="No active execution context"):
            robot.left.close()

    def test_open_without_context_raises(self):
        robot = Geodude()
        import pytest

        with pytest.raises(RuntimeError, match="No active execution context"):
            robot.left.open()
