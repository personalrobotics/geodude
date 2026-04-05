# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for VentionBase class."""

import pytest

from geodude.config import VentionBaseConfig
from geodude.robot import Geodude
from geodude.vention_base import VentionBase


class TestVentionBaseConfig:
    """Tests for VentionBaseConfig."""

    def test_default_values(self):
        config = VentionBaseConfig(
            name="left_base",
            joint_name="left_arm_linear_vention",
            actuator_name="left_linear_actuator",
        )
        assert config.name == "left_base"
        assert config.height_range == (0.0, 0.5)
        assert config.collision_check_resolution == 0.01

    def test_custom_height_range(self):
        config = VentionBaseConfig(
            name="test_base",
            joint_name="test_joint",
            actuator_name="test_actuator",
            height_range=(0.1, 0.4),
        )
        assert config.height_range == (0.1, 0.4)


class TestVentionBaseInit:
    def test_geodude_has_bases(self):
        robot = Geodude()
        assert robot.left_base is not None
        assert robot.right_base is not None

    def test_base_names(self):
        robot = Geodude()
        assert robot.left_base.name == "left_base"
        assert robot.right_base.name == "right_base"

    def test_invalid_joint_raises(self):
        robot = Geodude()
        config = VentionBaseConfig(
            name="test_base",
            joint_name="nonexistent_joint",
            actuator_name="left_linear_actuator",
        )
        with pytest.raises(ValueError, match="Joint.*not found"):
            VentionBase(robot.model, robot.data, config, robot.left)

    def test_invalid_actuator_raises(self):
        robot = Geodude()
        config = VentionBaseConfig(
            name="test_base",
            joint_name="left_arm_linear_vention",
            actuator_name="nonexistent_actuator",
        )
        with pytest.raises(ValueError, match="Actuator.*not found"):
            VentionBase(robot.model, robot.data, config, robot.left)


class TestVentionBaseHeight:
    def test_initial_height(self):
        robot = Geodude()
        assert robot.left_base.get_height() == pytest.approx(0.0, abs=0.001)
        assert robot.right_base.get_height() == pytest.approx(0.0, abs=0.001)

    def test_height_range(self):
        robot = Geodude()
        assert robot.left_base.height_range == (0.0, 0.5)


class TestVentionBaseSetHeight:
    def test_set_height(self):
        robot = Geodude()
        robot.left_base.set_height(0.3)
        assert robot.left_base.get_height() == pytest.approx(0.3, abs=0.001)

    def test_set_height_to_max(self):
        robot = Geodude()
        robot.left_base.set_height(0.5)
        assert robot.left_base.get_height() == pytest.approx(0.5, abs=0.001)

    def test_set_height_below_range_raises(self):
        robot = Geodude()
        with pytest.raises(ValueError, match="outside valid range"):
            robot.left_base.set_height(-0.1)

    def test_set_height_above_range_raises(self):
        robot = Geodude()
        with pytest.raises(ValueError, match="outside valid range"):
            robot.left_base.set_height(0.6)

    def test_set_height_independent(self):
        robot = Geodude()
        robot.left_base.set_height(0.2)
        robot.right_base.set_height(0.4)
        assert robot.left_base.get_height() == pytest.approx(0.2, abs=0.001)
        assert robot.right_base.get_height() == pytest.approx(0.4, abs=0.001)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestVentionBaseMoveTo:
    def test_move_to_no_collision_check(self):
        robot = Geodude()
        success = robot.left_base.move_to(0.3, check_collisions=False)
        assert success is True
        assert robot.left_base.get_height() == pytest.approx(0.3, abs=0.001)

    def test_move_to_below_range_raises(self):
        robot = Geodude()
        with pytest.raises(ValueError, match="outside valid range"):
            robot.left_base.move_to(-0.1)

    def test_move_to_above_range_raises(self):
        robot = Geodude()
        with pytest.raises(ValueError, match="outside valid range"):
            robot.left_base.move_to(0.6)

    def test_move_to_same_position(self):
        robot = Geodude()
        robot.left_base.set_height(0.2)
        success = robot.left_base.move_to(0.2, check_collisions=True)
        assert success is True

    def test_robot_self_contacts_ignored(self):
        """Robot-robot contacts should NOT block base movement."""
        robot = Geodude()
        success = robot.left_base.move_to(0.3, check_collisions=True)
        assert success is True
