# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Geodude-specific behavior tree leaf nodes.

Only LiftBase remains here — it depends on VentionBase hardware.
GenerateGrasps and GeneratePlaceTSRs are now generic nodes in
mj_manipulator.bt.nodes, driven by the GraspSource protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import py_trees
from mj_manipulator.contacts import iter_contacts
from py_trees.common import Access, Status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LiftBase decision logic — pure function, exhaustively tested
# ---------------------------------------------------------------------------


@dataclass
class LiftFacts:
    """Facts about a base lift that determine the BT status.

    Defaults are chosen so an unset field never triggers FAILURE in
    :func:`_classify_lift_outcome`. The orchestrator (``LiftBase.update``)
    sets only the fields that are relevant at each decision point —
    e.g., if no held object is present, only ``has_base`` and
    ``has_held_object`` need to be set; the later fields stay at their
    "wouldn't trigger failure" defaults.
    """

    has_base: bool = False  # base configured for the arm?
    has_held_object: bool = False  # gripper currently holding something?
    n_baseline_contacts: int = 0  # number of source-surface contacts at start
    base_has_headroom: bool = True  # can the base lift any further?
    plan_succeeded: bool = True  # did the lift planner return a trajectory?
    n_remaining_contacts: int = 0  # source-surface contacts after the lift


def _classify_lift_outcome(facts: LiftFacts) -> Status:
    """Map a fully-populated set of lift facts to a BT status.

    Pure function — same inputs always produce the same output, no
    side effects, no model objects, no IO. The decision tree:

    1. **No base configured for the arm** → FAILURE (config error)
    2. **No held object** → FAILURE (BT structure bug: ran without Grasp)
    3. **Held object isn't touching anything** → SUCCESS (already clear)
    4. **Base has no headroom** → FAILURE (can't lift any further but
       the object is still touching the source surface)
    5. **Planner couldn't plan any motion** → FAILURE (first step blocked)
    6. **Lift completed but baseline contacts persist** → FAILURE
       (lifted as far as we could and the object is still touching)
    7. **Otherwise** → SUCCESS

    The check order matters: each step assumes the previous ones
    passed, so by the time we ask "did the lift clear the contacts"
    we already know the planner ran and found a path.
    """
    if not facts.has_base:
        return Status.FAILURE
    if not facts.has_held_object:
        return Status.FAILURE
    if facts.n_baseline_contacts == 0:
        return Status.SUCCESS
    if not facts.base_has_headroom:
        return Status.FAILURE
    if not facts.plan_succeeded:
        return Status.FAILURE
    if facts.n_remaining_contacts > 0:
        return Status.FAILURE
    return Status.SUCCESS


# ---------------------------------------------------------------------------
# LiftBase node — orchestrates the facts, calls _classify_lift_outcome
# ---------------------------------------------------------------------------


class LiftBase(py_trees.behaviour.Behaviour):
    """Lift the Vention base until the held object clears its source surface.

    The point of this node is to ensure that, after a grasp, the held
    object is no longer touching whatever it was resting on at grasp
    time (table, plate, bin floor, etc.). The base lift is the means;
    the disappearance of the held-object↔source contact is the end.
    The node only returns SUCCESS when that invariant holds.

    Implementation mirrors :func:`mj_manipulator.safe_retract.safe_retract`
    in shape:

    1. **Capture baseline contacts** involving the held object body,
       excluding contacts where the other side is the arm or gripper
       (those are the grasp itself, not the source surface).
    2. **Plan with partial_ok**: try the full requested lift (15 cm or
       to range max). If a collision blocks the path partway, take the
       longest collision-free prefix instead of giving up.
    3. **Execute with abort_fn**: stop the trajectory the moment all
       baseline contacts have disappeared. There's no need to keep
       lifting once the object is clear.
    4. **Verify on exit**: if any baseline contact remains, return
       FAILURE so the BT recovery path can react. Returning SUCCESS
       silently while the object is still in collision was the bug
       that motivated this rewrite (geodude#173).

    The decision logic itself lives in :func:`_classify_lift_outcome`,
    a pure function that takes a populated :class:`LiftFacts` and
    returns a status. ``update()`` is the orchestrator: it computes
    the facts incrementally, short-circuiting on early termination,
    and calls the classifier with whatever facts it has gathered. The
    classifier's defaults are chosen so unset fields don't accidentally
    trigger failures.

    Reads: ``{ns}/robot``, ``{ns}/arm``, ``/context``
    """

    LIFT_AMOUNT = 0.15  # meters — maximum target lift, not the minimum

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

        facts = LiftFacts()

        # ----- Fact 1: base configured? -----
        base = robot._get_base_for_arm(arm)
        if base is None:
            logger.warning("LiftBase: no base configured for arm %s", arm.config.name)
            return _classify_lift_outcome(facts)
        facts.has_base = True

        # ----- Fact 2: held object? -----
        gripper = arm.gripper
        if gripper is None or gripper.held_object is None:
            logger.warning(
                "LiftBase: no held object on arm %s — was Grasp run first?",
                arm.config.name,
            )
            return _classify_lift_outcome(facts)
        facts.has_held_object = True

        held_name = gripper.held_object
        model = base.model
        data = base.data

        held_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, held_name)
        if held_body_id < 0:
            logger.warning("LiftBase: held object '%s' not found in model", held_name)
            # Treat unknown body as "no held object" — same FAILURE result.
            facts.has_held_object = False
            return _classify_lift_outcome(facts)

        # ----- Fact 3: baseline source-surface contacts -----
        mujoco.mj_forward(model, data)
        baseline = self._compute_source_contacts(model, data, base, held_body_id)
        facts.n_baseline_contacts = len(baseline)

        if not baseline:
            logger.info(
                "LiftBase: held object '%s' has no source-surface contacts; no lift needed",
                held_name,
            )
            return _classify_lift_outcome(facts)

        baseline_names = self._format_other_bodies(model, baseline, held_body_id)
        logger.info(
            "LiftBase: %s touching %s — lifting until clear",
            held_name,
            ", ".join(baseline_names),
        )

        # ----- Fact 4: base headroom -----
        current_h = base.get_height()
        max_h = base.height_range[1]
        target_h = min(current_h + self.LIFT_AMOUNT, max_h)
        if target_h - current_h < 1e-4:
            logger.warning(
                "LiftBase: base already at max height %.3fm; cannot lift %s away from %s",
                current_h,
                held_name,
                ", ".join(baseline_names),
            )
            facts.base_has_headroom = False
            return _classify_lift_outcome(facts)

        # ----- Fact 5: plan succeeded? -----
        base_traj = base.plan_to(target_h, check_collisions=True, partial_ok=True)
        if base_traj is None:
            logger.warning(
                "LiftBase: cannot plan any base motion from %.3fm — first step blocked by collision",
                current_h,
            )
            facts.plan_succeeded = False
            return _classify_lift_outcome(facts)

        # ----- Execute with abort-on-clear -----
        #
        # Stop the trajectory the instant all baseline source contacts
        # have disappeared. The motion runs through the standard
        # trajectory runner so the abort fires once per control cycle.
        def _abort() -> bool:
            current = self._compute_source_contacts(model, data, base, held_body_id)
            return not (current & baseline)

        ctx.execute(base_traj, abort_fn=_abort)

        # ----- Fact 6: remaining baseline contacts after the lift -----
        mujoco.mj_forward(model, data)
        remaining = self._compute_source_contacts(model, data, base, held_body_id) & baseline
        facts.n_remaining_contacts = len(remaining)

        if remaining:
            still_touching = self._format_other_bodies(model, remaining, held_body_id)
            logger.warning(
                "LiftBase: lift completed but %s still touching %s (base now at %.3fm of max %.3fm)",
                held_name,
                ", ".join(still_touching),
                base.get_height(),
                max_h,
            )
        else:
            logger.info(
                "LiftBase: %s cleared from source surface (base now at %.3fm)",
                held_name,
                base.get_height(),
            )

        return _classify_lift_outcome(facts)

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
