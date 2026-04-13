# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Geodude-specific behavior tree leaf nodes.

Only LiftBase remains here — it depends on VentionBase hardware.
GenerateGrasps and GeneratePlaceTSRs are now generic nodes in
mj_manipulator.bt.nodes, driven by the GraspSource protocol.
"""

from __future__ import annotations

import logging

import py_trees
from py_trees.common import Access, Status

logger = logging.getLogger(__name__)


class LiftBase(py_trees.behaviour.Behaviour):
    """Lift the Vention base, then verify the held object is still held.

    Geodude-specific node for post-grasp clearance on the Vention
    linear base. Replaces the arm-based ``SafeRetract`` used by
    fixed-base arms (Franka).

    The verifier's ``gripper.is_holding`` (backed by
    :class:`~mj_manipulator.grasp_verifier.GraspVerifier`) is the
    source of truth for both the precondition and the post-check.
    The verifier's decisive-negative check (gripper at mechanical
    stop = nothing held) works on both sim and real hardware.

    Reads: ``{ns}/robot``, ``{ns}/arm``, ``/context``

    Returns SUCCESS when: base lifted and gripper still reports holding.
    Returns FAILURE when:
        - No base configured for the arm.
        - Gripper does not report holding (grasp failed or never ran).
        - Base at max height (cannot lift further).
        - Base motion blocked by collision at the first step.
        - Object dropped during the lift (verifier went LOST).
    """

    LIFT_AMOUNT = 0.15  # meters — target lift, capped at base headroom

    def __init__(self, ns: str = "", name: str = "LiftBase"):
        super().__init__(name)
        self.ns = ns
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(key=f"{ns}/robot", access=Access.READ)
        self.bb.register_key(key=f"{ns}/arm", access=Access.READ)
        self.bb.register_key(key="/context", access=Access.READ)
        # See mj_manipulator/primitives.py _report_pickup_failure: this
        # key is read there to distinguish grasp-verification failure
        # from planning failure in top-level error messages.
        try:
            self.bb.register_key(key=f"{ns}/grasp_confirmed", access=Access.WRITE)
        except KeyError:
            pass

    def update(self) -> Status:
        robot = self.bb.get(f"{self.ns}/robot")
        arm = self.bb.get(f"{self.ns}/arm")
        ctx = self.bb.get("/context")

        # ----- Preconditions -----
        base = robot._get_base_for_arm(arm)
        if base is None:
            logger.warning("LiftBase: no base configured for arm %s", arm.config.name)
            return Status.FAILURE

        gripper = arm.gripper
        if gripper is None or not gripper.is_holding:
            # The verifier ran its first post-settling check during
            # the preceding pickup subtree ticks and
            # rejected the grasp, OR nothing was grasped to begin
            # with. Either way: the grasp didn't hold. Record the
            # verdict so _report_pickup_failure can classify this as
            # \"reached X but verifier rejected the grasp\" instead
            # of \"could not plan to X\".
            self.bb.set(f"{self.ns}/grasp_confirmed", False)
            logger.warning(
                "LiftBase: verifier on %s arm rejected the grasp (nothing to lift)",
                arm.config.name,
            )
            return Status.FAILURE

        # Precondition passed — verifier says we're holding. Record
        # the confirmation. Subsequent failures in this node are
        # dropped-during-lift, not grasp-close failures.
        self.bb.set(f"{self.ns}/grasp_confirmed", True)
        held_name = gripper.held_object

        # ----- Action: always attempt the lift -----
        current_h = base.get_height()
        max_h = base.height_range[1]
        target_h = min(current_h + self.LIFT_AMOUNT, max_h)
        start_h = current_h

        if target_h - current_h < 1e-4:
            logger.warning(
                "LiftBase: base already at max height %.3fm; cannot lift %s",
                current_h,
                held_name,
            )
            return Status.FAILURE
        else:
            traj = base.plan_to(target_h, check_collisions=True, partial_ok=True)
            if traj is None:
                logger.warning(
                    "LiftBase: cannot plan any base motion from %.3fm — first step blocked",
                    current_h,
                )
                return Status.FAILURE
            else:
                ctx.execute(traj)
                logger.info(
                    "LiftBase: lifted %s to %.3fm (target was %.3fm)",
                    held_name,
                    base.get_height(),
                    target_h,
                )

        # ----- Verify post-condition: verifier still thinks we hold it -----
        if not gripper.is_holding:
            end_h = base.get_height()
            lift_traveled = end_h - start_h
            if lift_traveled < 0.005:
                # Base barely moved — the abort-on-drop predicate
                # caught the verifier transition within a tick or two
                # of the trajectory starting. The grasp wasn't really
                # holding and there was nothing to transport.
                logger.warning(
                    "LiftBase: %s slipped immediately after grasp (base at %.3fm, did not lift)",
                    held_name,
                    end_h,
                )
            else:
                logger.warning(
                    "LiftBase: %s dropped during lift (base moved %.3fm to %.3fm of max %.3fm)",
                    held_name,
                    lift_traveled,
                    end_h,
                    max_h,
                )
            return Status.FAILURE

        return Status.SUCCESS
