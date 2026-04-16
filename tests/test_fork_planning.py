# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for fork-based planning in _plan_single.

Verifies that planning at different base heights doesn't mutate
live state (data.qpos, data.xpos).
"""

import mujoco
import numpy as np

from geodude.robot import Geodude


class TestForkPlanning:
    """Verify _plan_single plans in a fork without mutating live state."""

    def test_plan_single_does_not_mutate_base_qpos(self):
        """Base qpos in live data should be unchanged after _plan_single."""
        robot = Geodude()
        mujoco.mj_forward(robot.model, robot.data)

        arm = robot._right_arm
        base = robot._get_base_for_arm(arm)
        assert base is not None

        # Set base to a known height and snapshot live state
        base.set_height(0.25)
        original_qpos = robot.data.qpos.copy()

        # Plan at a different height
        robot._plan_single(
            arm,
            height=0.1,
            pose=arm.get_ee_pose(),
        )

        # Live qpos must be unchanged
        np.testing.assert_array_equal(
            robot.data.qpos,
            original_qpos,
            err_msg="Live qpos was mutated during planning",
        )

    def test_plan_single_does_not_mutate_xpos(self):
        """Body xpos in live data should be unchanged after _plan_single."""
        robot = Geodude()
        mujoco.mj_forward(robot.model, robot.data)

        arm = robot._right_arm
        base = robot._get_base_for_arm(arm)
        assert base is not None

        base.set_height(0.25)
        original_xpos = robot.data.xpos.copy()

        robot._plan_single(
            arm,
            height=0.4,
            pose=arm.get_ee_pose(),
        )

        np.testing.assert_array_equal(
            robot.data.xpos,
            original_xpos,
            err_msg="Live xpos was mutated during planning",
        )

    def test_plan_single_no_height_change_skips_fork_mutation(self):
        """When height matches current, no fork mutation needed."""
        robot = Geodude()
        mujoco.mj_forward(robot.model, robot.data)

        arm = robot._right_arm
        base = robot._get_base_for_arm(arm)
        base.set_height(0.25)
        original_qpos = robot.data.qpos.copy()

        # Plan at the same height — should skip base mutation in fork
        robot._plan_single(
            arm,
            height=0.25,
            pose=arm.get_ee_pose(),
        )

        np.testing.assert_array_equal(
            robot.data.qpos,
            original_qpos,
        )

    def test_plan_single_none_height(self):
        """Planning with height=None should not touch base at all."""
        robot = Geodude()
        mujoco.mj_forward(robot.model, robot.data)

        arm = robot._right_arm
        base = robot._get_base_for_arm(arm)
        base.set_height(0.25)
        original_qpos = robot.data.qpos.copy()

        robot._plan_single(
            arm,
            height=None,
            pose=arm.get_ee_pose(),
        )

        np.testing.assert_array_equal(
            robot.data.qpos,
            original_qpos,
        )
