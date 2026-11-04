"""Grasp regions from prl_assets metadata, as ssgrasping inputs (#260).

An asset that declares ``grasp_regions`` is grasped from them alone: each named region,
parsed and validated by prl_assets' own reader (``prl_assets.grasp_regions.parse``),
becomes an ssgrasping ``GraspRegion`` placed in the asset frame by its
``T_asset_region``. prl_assets' asset frame is the object's MuJoCo body frame, which is
the pose the runtime registers for the object, so the regions compose with it directly.
An asset that does not declare them is grasped by the legacy path, from its
``geometric_properties``, until prl_assets#40 gives it regions; one asset is never
grasped by both. A ``grasp_regions`` section that is malformed, or declares no region, is
an error, never a reason to fall back.

The hand is Geodude's Robotiq 2F-140, through ssgrasping's sstsr adapter; each arm's
profile names that arm's manipulator, hand, gripper, and opening channel.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from prl_assets.grasp_regions import parse
from ssgrasping import GraspingProfile, GraspRegion, Primitive
from ssgrasping.sstsr import SstsrTemplates, profile
from tsr.hands.parallel_jaw import Robotiq2F140

from geodude.regions import pose

HAND = Robotiq2F140()
TEMPLATES = SstsrTemplates(HAND)
"""The 2F-140's grasps of each primitive sstsr's registry verifies for it."""


def profile_for(manipulator: str, hand: str, gripper: str, opening: str) -> GraspingProfile:
    """The 2F-140 on the arm with these names."""
    return profile(
        HAND, manipulator=manipulator, end_effector=hand, gripper=gripper, opening_channel=opening
    )


def _primitive(kind: str, params: Mapping[str, Any]) -> Primitive:
    if kind == "cylinder":
        return Primitive("cylinder", (params["radius"], params["height"]))
    if kind == "box":
        return Primitive("box", tuple(params["size"]))
    if kind == "sphere":
        return Primitive("sphere", (params["radius"],))
    raise ValueError(f"ssgrasping has no {kind} primitive")


def regions_of(meta: Mapping[str, Any]) -> tuple[GraspRegion, ...] | None:
    """The asset's grasp regions, or None if it declares none (the legacy path).

    Raises if its ``grasp_regions`` is malformed (prl_assets' validation), declares no
    region, or uses a primitive ssgrasping cannot grasp (a torus)."""
    if "grasp_regions" not in meta:
        return None
    declared = parse(dict(meta))
    if not declared:
        raise ValueError(f"{meta.get('name')!r} declares grasp_regions with no region")
    return tuple(
        GraspRegion(
            r.name,
            _primitive(r.primitive, r.params),
            pose(np.asarray(r.T_asset_region, dtype=float)),
            tuple(r.strategies),
        )
        for r in declared
    )
