# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Geodude-specific behavior tree leaf nodes.

Only LiftBase remains here — it depends on VentionBase hardware.
GenerateGrasps and GeneratePlaceTSRs are now generic nodes in
mj_manipulator.bt.nodes, driven by the GraspSource protocol.
"""

from __future__ import annotations

import logging

import mujoco
import py_trees
from mj_manipulator.contacts import iter_contacts
from py_trees.common import Access, Status

logger = logging.getLogger(__name__)


class LiftBase(py_trees.behaviour.Behaviour):
    """Lift the Vention base, then verify the held object is clear of any
    source surface.

    The point of this node is to ensure that, after a grasp, the held
    object is no longer touching whatever it was resting on at grasp
    time (table, plate, bin floor, etc.). The action is *always* a base
    lift; the SUCCESS criterion is the post-condition that the held
    object has no contact with anything other than the arm or gripper.

    Why "always lift" rather than "only lift if we see a contact":
    after the gripper closes via physics, friction often pulls the
    object 1–2 mm vertically into the gripper, breaking the
    object↔table contact for that instant. A precondition check that
    sees zero baseline contacts and skips the lift returns SUCCESS
    without ever moving the base — exactly the visible bug from
    geodude#173. The lift is cheap and visible; trust the post-check
    instead of trying to skip work based on a flaky observation.

    The lift is planned with ``partial_ok=True`` so a collision-blocked
    upper portion still gets us as much travel as is reachable. If even
    the first step is blocked, or the base is already at max height, we
    skip the motion and rely on the post-check anyway.

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
        if gripper is None or gripper.held_object is None:
            logger.warning(
                "LiftBase: no held object on arm %s — was Grasp run first?",
                arm.config.name,
            )
            return Status.FAILURE

        held_name = gripper.held_object
        model = base.model
        data = base.data
        held_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, held_name)
        if held_body_id < 0:
            logger.warning("LiftBase: held object '%s' not found in model", held_name)
            return Status.FAILURE

        # ----- Action: always attempt the lift -----
        current_h = base.get_height()
        max_h = base.height_range[1]
        target_h = min(current_h + self.LIFT_AMOUNT, max_h)

        if target_h - current_h < 1e-4:
            logger.warning(
                "LiftBase: base already at max height %.3fm; skipping lift, will verify clearance",
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

        # ----- Verify post-condition: no source-surface contact -----
        mujoco.mj_forward(model, data)
        remaining = self._compute_source_contacts(model, data, base, held_body_id)
        if remaining:
            still_touching = self._format_other_bodies(model, remaining, held_body_id)
            logger.warning(
                "LiftBase: %s still touching %s after lift (base at %.3fm of max %.3fm)",
                held_name,
                ", ".join(still_touching),
                base.get_height(),
                max_h,
            )
            return Status.FAILURE

        return Status.SUCCESS

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _compute_source_contacts(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        base,
        held_body_id: int,
    ) -> set[tuple[int, int]]:
        """Return the set of contact pairs between the held body and a
        body that is neither the arm nor the gripper.

        These are the "source surface" contacts the lift needs to
        clear. Pairs are returned in canonical order
        ``(min(b1, b2), max(b1, b2))`` so set membership is well defined.
        """
        arm_and_gripper_ids = base.arm_body_ids
        out: set[tuple[int, int]] = set()
        for b1, b2, _ in iter_contacts(model, data):
            if b1 == b2:
                continue
            if b1 == held_body_id and b2 not in arm_and_gripper_ids:
                out.add((min(b1, b2), max(b1, b2)))
            elif b2 == held_body_id and b1 not in arm_and_gripper_ids:
                out.add((min(b1, b2), max(b1, b2)))
        return out

    @staticmethod
    def _format_other_bodies(
        model: mujoco.MjModel,
        pairs: set[tuple[int, int]],
        held_body_id: int,
    ) -> list[str]:
        """Format the non-held body in each pair as a sorted list of names."""
        return sorted(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
            for pair in pairs
            for b in pair
            if b != held_body_id
        )
