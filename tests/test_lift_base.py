# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for ``geodude.bt.nodes.LiftBase``.

LiftBase is a thin orchestrator: precondition (gripper reports
holding) → lift → verify post-condition (gripper still reports
holding). Both halves go through
:class:`~mj_manipulator.grasp_verifier.GraspVerifier`, which the
geodude arm factory wires up with a wrist F/T signal per
personalrobotics/mj_manipulator#93. The tests below drive the
verifier directly (``gripper.grasp_verifier.mark_grasped(...)``) to
simulate a completed grasp without running the full physics close
sequence.

The drop-during-lift end-to-end path is exercised by the recycling
demo per geodude#173's acceptance criteria.
"""

from __future__ import annotations

import py_trees
import pytest
from py_trees.common import Access, Status

from geodude.bt.nodes import LiftBase
from geodude.robot import Geodude


@pytest.fixture(autouse=True)
def _clear_blackboard():
    """Each test starts with a clean blackboard so it doesn't leak state
    from earlier BT tests in the suite."""
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


def _make_node(robot: Geodude, ctx, side: str = "left") -> LiftBase:
    """Wire blackboard and return a ready-to-tick LiftBase node."""
    ns = f"/{side}"
    bb = py_trees.blackboard.Client(name=f"test_lift_base_{side}")
    bb.register_key(key=f"{ns}/robot", access=Access.WRITE)
    bb.register_key(key=f"{ns}/arm", access=Access.WRITE)
    bb.register_key(key="/context", access=Access.WRITE)
    bb.set(f"{ns}/robot", robot)
    bb.set(f"{ns}/arm", robot.arms[side])
    bb.set("/context", ctx)
    node = LiftBase(ns=ns)
    node.setup()
    node.initialise()
    return node


class TestLiftBaseIntegration:
    def test_no_held_object_returns_failure(self):
        """A freshly-loaded robot has no held object, so LiftBase should
        return FAILURE without touching the base. Verifies the
        precondition guard and that the orchestrator wires up the
        blackboard correctly.
        """
        robot = Geodude()
        with robot.sim(headless=True) as ctx:
            node = _make_node(robot, ctx)
            assert node.update() == Status.FAILURE

    def test_lift_happens_when_verifier_reports_held(self):
        """Regression for geodude#173: LiftBase used to skip the lift
        entirely when its contact-based precondition check saw zero
        source-surface contacts at start, returning SUCCESS without
        moving the base. After a real physics grasp, friction often
        pulls the held object 1–2 mm into the gripper, breaking the
        object↔table contact for that instant — exactly the case the
        precondition check missed.

        The fix (both the #173 mitigation and the #93 verifier-based
        rewrite) is to treat the lift as the action and the verifier's
        held-state as the post-condition. Always lift, then verify.

        We simulate a completed grasp by driving the verifier
        directly: ``mark_grasped`` records a baseline from the F/T
        signal. In kinematic sim the signal returns None, so the
        verifier degenerates to \"trust that mark_grasped was called\"
        — which is what we want for this test since the lift itself
        is the property under test, not the drop-detection path.
        The drop-detection path is exercised at integration time by
        the recycling demo.
        """
        robot = Geodude()
        with robot.sim(headless=True) as ctx:
            arm = robot.arms["left"]
            # vention_base is a real body that happens to be a
            # convenient stand-in: its only contacts are with the arm
            # (filtered by the verifier implicitly because we never
            # look at contacts) and it's guaranteed to exist.
            arm.gripper.grasp_verifier.mark_grasped("vention_base")
            assert arm.gripper.is_holding is True

            node = _make_node(robot, ctx)
            start_h = robot.left_base.get_height()
            result = node.update()
            end_h = robot.left_base.get_height()

            assert end_h > start_h, (
                "LiftBase did not move the base — the empty-baseline early-return regression from #173 is back"
            )
            assert result == Status.SUCCESS

    def test_object_dropped_during_lift_returns_failure(self):
        """Regression for the core geodude#173 bug class: if the
        gripper loses the object during the lift, LiftBase must
        return FAILURE so the BT recovery path fires.

        We simulate the drop by wrapping ``ctx.execute`` to call
        ``verifier.mark_released`` as a side effect — i.e. by the
        time execution returns, the verifier reports \"not held\".
        The post-check then re-queries ``gripper.is_holding``, sees
        False, and returns FAILURE.

        This is the contract LiftBase owes the BT: the post-check
        is authoritative. If you change LiftBase to skip the
        post-check or to return SUCCESS based on something the
        verifier disagrees with, this test breaks.
        """
        robot = Geodude()
        with robot.sim(headless=True) as ctx:
            arm = robot.arms["left"]
            arm.gripper.grasp_verifier.mark_grasped("vention_base")
            assert arm.gripper.is_holding is True

            # Wrap ctx.execute so that the verifier is cleared as a
            # side effect of the lift — this is what a real dropped
            # object looks like from LiftBase's perspective.
            original_execute = ctx.execute

            def execute_then_drop(*args, **kwargs):
                result = original_execute(*args, **kwargs)
                arm.gripper.grasp_verifier.mark_released()
                return result

            ctx.execute = execute_then_drop

            node = _make_node(robot, ctx)
            assert node.update() == Status.FAILURE
            assert arm.gripper.is_holding is False
