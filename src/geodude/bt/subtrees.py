# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Geodude-specific behavior tree subtrees.

Extends mj_manipulator.bt subtrees with bimanual arm selection
and automatic TSR generation.
"""

from __future__ import annotations

import py_trees
from mj_manipulator.bt import pickup_with_recovery, place_with_recovery

from geodude.bt.nodes import GenerateGrasps, GeneratePlaceTSRs, LiftBase


def geodude_pickup(ns: str) -> py_trees.composites.Sequence:
    """Generate grasp TSRs then pickup with recovery.

    Reads: ``{ns}/object_name``, ``{ns}/robot``
    (plus all blackboard keys needed by pickup_with_recovery)
    """
    return py_trees.composites.Sequence(
        name="geodude_pickup",
        memory=True,
        children=[
            GenerateGrasps(ns=ns),
            pickup_with_recovery(ns),
            LiftBase(ns=ns),
        ],
    )


def geodude_place(ns: str) -> py_trees.composites.Sequence:
    """Generate placement TSRs then place with recovery.

    Reads: ``{ns}/destination``, ``{ns}/robot``
    (plus all blackboard keys needed by place_with_recovery)
    """
    return py_trees.composites.Sequence(
        name="geodude_place",
        memory=True,
        children=[
            GeneratePlaceTSRs(ns=ns),
            place_with_recovery(ns),
        ],
    )


def geodude_pickup_bimanual() -> py_trees.composites.Selector:
    """Try pickup with right arm, then left arm.

    Each arm gets its own namespace so they don't interfere.
    The Selector tries right first, falls back to left.
    """
    return py_trees.composites.Selector(
        name="bimanual_pickup",
        memory=True,
        children=[
            geodude_pickup("/right"),
            geodude_pickup("/left"),
        ],
    )
