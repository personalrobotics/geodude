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


class LiftBase(py_trees.behaviour.Behaviour):
    """Lift the Vention base to clear worktop clutter after grasping.

    Plans a collision-free base trajectory 15cm upward (clamped to range)
    and executes it through the context.

    Reads: ``{ns}/robot``, ``{ns}/arm``, ``/context``
    """

    LIFT_AMOUNT = 0.15  # meters

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

        base = robot._get_base_for_arm(arm)
        if base is None:
            return Status.SUCCESS  # no base, nothing to do

        current = base.get_height()
        target_h = min(current + self.LIFT_AMOUNT, base.height_range[1])
        if target_h - current < 0.01:
            return Status.SUCCESS  # already at max

        base_traj = base.plan_to(target_h, check_collisions=True)
        if base_traj is None:
            logging.getLogger(__name__).info(
                "Base lift to %.2fm blocked by collision",
                target_h,
            )
            return Status.SUCCESS  # non-critical, don't fail pickup

        ctx.execute(base_traj)
        ctx.sync()
        return Status.SUCCESS
