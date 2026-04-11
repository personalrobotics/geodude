# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for ``geodude.bt.nodes.LiftBase``.

Two layers of testing, matching the two layers of the production code:

1. **Unit tests on ``_classify_lift_outcome``** — pure-function tests
   that walk every branch of the lift's decision tree. No model, no
   physics, no fixtures, no monkey-patching. Each test is one line
   of input and one assertion.

2. **One integration smoke test on ``LiftBase.update``** — verifies
   the orchestrator wires up the blackboard, the robot, and the
   classifier correctly. Uses the only natural input where we don't
   need to fake contact state: a freshly-built robot has no held
   object, so ``LiftBase`` should return FAILURE.

Other integration paths (object touching surface, lift clears it,
lift can't clear it) are exercised end-to-end by the recycling demo.
The acceptance criterion in geodude#173 is "the recycling demo runs
cleanly with the fix" — these unit tests are the regression net so
the SUCCESS-on-failure paths can't come back without a test failure.
"""

from __future__ import annotations

import py_trees
import pytest
from py_trees.common import Access, Status

from geodude.bt.nodes import LiftBase, LiftFacts, _classify_lift_outcome
from geodude.robot import Geodude

# ---------------------------------------------------------------------------
# Pure-function unit tests for _classify_lift_outcome
#
# Each test corresponds to one branch of the decision tree. Asserting all
# branches catches every SUCCESS-on-failure regression directly.
# ---------------------------------------------------------------------------


class TestClassifyLiftOutcome:
    """Exhaustive tests for the lift decision function."""

    def test_no_base_returns_failure(self):
        """Config error: arm has no base configured. Should never SUCCESS."""
        facts = LiftFacts(has_base=False)
        assert _classify_lift_outcome(facts) == Status.FAILURE

    def test_no_held_object_returns_failure(self):
        """BT structure error: LiftBase ran without a Grasp before it.
        Conservative behavior is FAILURE so the bug doesn't hide."""
        facts = LiftFacts(has_base=True, has_held_object=False)
        assert _classify_lift_outcome(facts) == Status.FAILURE

    def test_already_clear_returns_success(self):
        """Held object isn't touching anything at start — nothing to lift
        away from. SUCCESS without moving."""
        facts = LiftFacts(
            has_base=True,
            has_held_object=True,
            n_baseline_contacts=0,
        )
        assert _classify_lift_outcome(facts) == Status.SUCCESS

    def test_no_headroom_returns_failure(self):
        """Base is at max travel and the held object is still touching
        the source surface. Lift can't help — return FAILURE so recovery
        fires."""
        facts = LiftFacts(
            has_base=True,
            has_held_object=True,
            n_baseline_contacts=1,
            base_has_headroom=False,
        )
        assert _classify_lift_outcome(facts) == Status.FAILURE

    def test_plan_failed_returns_failure(self):
        """Planner couldn't find any feasible motion (first step blocked).
        Same FAILURE because no motion can happen."""
        facts = LiftFacts(
            has_base=True,
            has_held_object=True,
            n_baseline_contacts=1,
            base_has_headroom=True,
            plan_succeeded=False,
        )
        assert _classify_lift_outcome(facts) == Status.FAILURE

    def test_lift_completed_but_object_still_touching_returns_failure(self):
        """The lift ran (planner succeeded, execution finished) but the
        held object is still touching the source surface. This is the
        verify-on-exit failure that the original bug was — the node
        used to silently SUCCESS here."""
        facts = LiftFacts(
            has_base=True,
            has_held_object=True,
            n_baseline_contacts=1,
            base_has_headroom=True,
            plan_succeeded=True,
            n_remaining_contacts=1,
        )
        assert _classify_lift_outcome(facts) == Status.FAILURE

    def test_lift_completed_and_cleared_returns_success(self):
        """Happy path: lift ran, baseline contacts gone after. SUCCESS."""
        facts = LiftFacts(
            has_base=True,
            has_held_object=True,
            n_baseline_contacts=1,
            base_has_headroom=True,
            plan_succeeded=True,
            n_remaining_contacts=0,
        )
        assert _classify_lift_outcome(facts) == Status.SUCCESS

    def test_default_facts_means_no_base_means_failure(self):
        """Sanity: a fully-default LiftFacts (everything unset) returns
        FAILURE because has_base defaults to False. This matters because
        update() uses defaults as a "wouldn't trigger failure" signal
        for unset fields, and the test confirms the defaults are
        chosen correctly."""
        assert _classify_lift_outcome(LiftFacts()) == Status.FAILURE


# ---------------------------------------------------------------------------
# Integration smoke test for LiftBase.update orchestration
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_blackboard():
    """Each test starts with a clean blackboard so it doesn't leak state
    from earlier BT tests in the suite."""
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


class TestLiftBaseIntegration:
    """One smoke test for the full update() path. Other paths are
    exercised by the recycling demo per the issue's acceptance criteria."""

    def test_no_held_object_smoke_test(self):
        """Fully-integrated path: real Geodude, real SimContext, real
        gripper with no held object. Should return FAILURE.

        This is the only LiftBase decision path reachable through
        natural (non-stubbed) robot state — a freshly-loaded robot
        always has gripper.held_object == None until something is
        grasped. Verifying this path means the orchestrator correctly
        reads from the blackboard, finds the base, queries the
        gripper, and returns the classifier's verdict.
        """
        robot = Geodude()
        with robot.sim(headless=True) as ctx:
            ns = "/left"

            bb = py_trees.blackboard.Client(name="test_lift_base_smoke")
            bb.register_key(key=f"{ns}/robot", access=Access.WRITE)
            bb.register_key(key=f"{ns}/arm", access=Access.WRITE)
            bb.register_key(key="/context", access=Access.WRITE)
            bb.set(f"{ns}/robot", robot)
            bb.set(f"{ns}/arm", robot.arms["left"])
            bb.set("/context", ctx)

            node = LiftBase(ns=ns)
            node.setup()
            node.initialise()

            # No held object → FAILURE
            assert node.update() == Status.FAILURE
