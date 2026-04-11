# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for ``geodude.bt.nodes.LiftBase``.

LiftBase is a thin orchestrator: precondition checks → lift → verify
post-condition. Two integration tests against a real :class:`Geodude` +
:class:`SimContext` cover the failure mode and the regression. The
contact-clearing happy path is exercised end-to-end by the recycling
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
        blackboard correctly."""
        robot = Geodude()
        with robot.sim(headless=True) as ctx:
            node = _make_node(robot, ctx)
            assert node.update() == Status.FAILURE

    def test_lift_happens_even_when_no_baseline_contacts_detected(self):
        """Regression for geodude#173: LiftBase used to skip the lift
        entirely when it observed zero source-surface contacts at start
        time, returning SUCCESS without moving the base. After a real
        physics grasp, friction often pulls the held object 1–2 mm into
        the gripper, breaking the object↔table contact for that
        instant — exactly the case the precondition check missed.

        We simulate that situation by marking a held object whose body
        has no source-surface contacts (the vention_base body itself,
        which only touches the arm — filtered by ``arm_body_ids``). The
        regression assertion is that the base height *increased*, i.e.
        LiftBase actually attempted the lift instead of bailing out
        early on the empty-baseline observation.
        """
        robot = Geodude()
        with robot.sim(headless=True) as ctx:
            arm = robot.arms["left"]
            # Mark a real model body as "grasped" so gripper.held_object
            # resolves to a valid body. vention_base touches only arm
            # bodies (filtered), so the post-check sees zero source
            # contacts and returns SUCCESS.
            arm.grasp_manager.mark_grasped("vention_base", arm.config.name)

            node = _make_node(robot, ctx)
            start_h = robot.left_base.get_height()
            result = node.update()
            end_h = robot.left_base.get_height()

            assert end_h > start_h, (
                "LiftBase did not move the base — the empty-baseline early-return regression from #173 is back"
            )
            assert result == Status.SUCCESS
