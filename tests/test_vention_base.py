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


class TestVentionBasePlanToPartialOk:
    """Tests for ``plan_to(partial_ok=True)`` — the longest-feasible-prefix mode.

    These tests use a contrived collision check (monkey-patching
    ``_has_arm_collision``) to simulate a virtual obstacle blocking
    the upper portion of the base's travel range. This avoids needing
    to construct a custom MuJoCo scene with a real obstacle, which
    would require duplicating geodude's model loading.
    """

    def test_partial_ok_full_path_clear_returns_full_trajectory(self):
        """If the full path is collision-free, partial_ok=True returns
        the same trajectory as the strict mode."""
        robot = Geodude()
        base = robot.left_base
        base.set_height(0.0)

        traj_strict = base.plan_to(0.3, check_collisions=True, partial_ok=False)
        traj_partial = base.plan_to(0.3, check_collisions=True, partial_ok=True)

        assert traj_strict is not None
        assert traj_partial is not None
        # Both should reach the same final height
        assert traj_strict.positions[-1, 0] == pytest.approx(0.3, abs=1e-3)
        assert traj_partial.positions[-1, 0] == pytest.approx(0.3, abs=1e-3)

    def test_partial_ok_blocked_midway_returns_prefix(self):
        """If a collision blocks the upper half of the path, partial_ok
        returns a trajectory ending at the last collision-free height,
        and strict mode returns None."""
        robot = Geodude()
        base = robot.left_base
        base.set_height(0.0)

        # Virtual obstacle at h > 0.15: monkey-patch the collision check
        original_has_collision = base._has_arm_collision

        def mock_has_collision():
            return base.get_height() > 0.15

        base._has_arm_collision = mock_has_collision
        try:
            # Strict mode: full path 0.0 → 0.3 must fail because the upper
            # half is blocked.
            traj_strict = base.plan_to(0.3, check_collisions=True, partial_ok=False)
            assert traj_strict is None

            # Partial mode: should return a trajectory ending somewhere
            # at-or-below 0.15 (the monkey-patched obstacle threshold).
            traj_partial = base.plan_to(0.3, check_collisions=True, partial_ok=True)
            assert traj_partial is not None
            final_h = float(traj_partial.positions[-1, 0])
            assert 0.0 < final_h <= 0.15 + 1e-6
        finally:
            base._has_arm_collision = original_has_collision

    def test_partial_ok_first_step_blocked_returns_none(self):
        """If the very first sample is in collision (we're already at an
        obstacle), partial_ok still returns None — there's nothing
        reachable."""
        robot = Geodude()
        base = robot.left_base
        base.set_height(0.0)

        original_has_collision = base._has_arm_collision

        def mock_has_collision():
            # Always in collision — even the start point fails.
            return True

        base._has_arm_collision = mock_has_collision
        try:
            traj = base.plan_to(0.3, check_collisions=True, partial_ok=True)
            assert traj is None
        finally:
            base._has_arm_collision = original_has_collision

    def test_partial_ok_no_collision_check_ignores_partial(self):
        """If check_collisions=False, partial_ok has no effect — the
        full requested path is returned regardless."""
        robot = Geodude()
        base = robot.left_base
        base.set_height(0.0)

        original_has_collision = base._has_arm_collision

        def always_blocked():
            return True

        base._has_arm_collision = always_blocked
        try:
            # Even though "everything is blocked", check_collisions=False
            # means the walker isn't even called and we get a full trajectory.
            traj = base.plan_to(0.3, check_collisions=False, partial_ok=True)
            assert traj is not None
            assert traj.positions[-1, 0] == pytest.approx(0.3, abs=1e-3)
        finally:
            base._has_arm_collision = original_has_collision
