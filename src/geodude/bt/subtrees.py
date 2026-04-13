# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Geodude-specific behavior tree subtrees.

Extends mj_manipulator.bt with Vention base lifting and bimanual
arm selection. These are action sequences — recovery on failure is
handled by the primitives layer.
"""

from __future__ import annotations

import py_trees
from mj_manipulator.bt import pickup, place

from geodude.bt.nodes import LiftBase


def geodude_pickup(ns: str) -> py_trees.composites.Sequence:
    """Pick up an object, then lift the base to clear the table.

    Uses the generic ``pickup(with_lift=False)`` (which finds grasps,
    plans, moves, and closes the gripper) then appends a Vention base
    lift instead of the arm-based retraction that fixed-base arms use.

    Reads: ``{ns}/object_name``, ``{ns}/robot``
    (plus all blackboard keys needed by pickup)
    """
    return py_trees.composites.Sequence(
        name="Geodude pickup",
        memory=True,
        children=[
            pickup(ns, with_lift=False),
            LiftBase(ns=ns, name="Lift base to clear table"),
        ],
    )


def geodude_place(ns: str) -> py_trees.composites.Sequence:
    """Find placement poses, plan, move, and release.

    Uses the generic ``place()`` directly — no geodude-specific
    modifications needed for placing.

    Reads: ``{ns}/destination``, ``{ns}/robot``
    (plus all blackboard keys needed by place)
    """
    return place(ns)


def geodude_pickup_bimanual() -> py_trees.composites.Selector:
    """Try pickup with right arm, then left arm.

    Each arm gets its own namespace so they don't interfere.
    The Selector tries right first, falls back to left.
    """
    return py_trees.composites.Selector(
        name="Try both arms",
        memory=True,
        children=[
            geodude_pickup("/right"),
            geodude_pickup("/left"),
        ],
    )
