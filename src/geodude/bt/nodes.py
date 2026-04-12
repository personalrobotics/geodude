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

    The point of this node is to ensure that, after a grasp, the held
    object remains in the gripper after we lift it off its source
    surface. The action is always a base lift; the SUCCESS criterion
    is the invariant that the gripper still believes it has the
    object, as judged by
    :class:`~mj_manipulator.grasp_verifier.GraspVerifier` on its
    live sensor signals (wrist F/T for UR5e + Robotiq).

    Why the verifier rather than a contact query:

    - On real hardware there's no MuJoCo contact state to inspect.
      The verifier uses signals — F/T, gripper position, joint
      torques — that exist on both sim and real robots.
    - After a physics grasp, friction can pull the object 1–2 mm
      into the gripper, breaking any object↔table contact for that
      instant. A contact-based precondition check sees nothing and
      skips the lift; that was the visible bug from geodude#173.
    - The invariant we care about is *\"object is still held\"*, not
      *\"object stopped touching the table\"*. The verifier answers
      the former directly. A dropped-during-lift object, a slipped
      grasp, a gripper that opened too early — all of these failure
      modes reduce to the same SUCCESS-criterion check: is the
      verifier still happy?

    Implementation:

    1. **Precondition:** the arm's gripper reports ``is_holding`` —
       i.e. the verifier thinks we have an object. FAILURE otherwise
       (config error, BT structure bug, or grasp never succeeded).
    2. **Action:** plan and execute the lift with
       ``base.plan_to(target, partial_ok=True)``. If the base is
       already at max headroom or the first step is in collision,
       skip the motion and fall straight through to the post-check.
    3. **Verify post-condition:** re-query ``gripper.is_holding``.
       FAILURE if it flipped to False during the lift (object
       dropped); SUCCESS otherwise.

    Reads: ``{ns}/robot``, ``{ns}/arm``, ``/context``
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
            # the preceding Sync / plan_and_execute ticks and
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
                "LiftBase: base already at max height %.3fm; skipping lift, will verify held state",
                current_h,
            )
        else:
            traj = base.plan_to(target_h, check_collisions=True, partial_ok=True)
            if traj is None:
                logger.warning(
                    "LiftBase: cannot plan any base motion from %.3fm — first step blocked",
                    current_h,
                )
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
