# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Geodude-specific behavior tree subtrees.

Extends mj_manipulator.bt subtrees with bimanual arm selection
and automatic TSR generation.
"""

from __future__ import annotations

import py_trees
from mj_manipulator.bt import pickup_with_recovery, place_with_recovery
from mj_manipulator.bt.nodes import GenerateGrasps, GeneratePlaceTSRs

from geodude.bt.nodes import LiftBase


def geodude_pickup(ns: str) -> py_trees.composites.Sequence:
    """Generate grasp TSRs then pickup with recovery.

    Geodude's UR5e is mounted on a Vention linear base with generous vertical
    clearance, so post-grasp retraction is done by :class:`LiftBase` (base up)
    rather than by a cartesian arm lift. We pass ``with_lift=False`` to skip
    the default ``SafeRetract`` in the generic pickup — doing both would be
    redundant and would drag the arm through extra cartesian motion with no
    benefit.

    Reads: ``{ns}/object_name``, ``{ns}/robot``
    (plus all blackboard keys needed by pickup_with_recovery)
    """
    return py_trees.composites.Sequence(
        name="geodude_pickup",
        memory=True,
        children=[
            GenerateGrasps(ns=ns),
            pickup_with_recovery(ns, with_lift=False),
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
