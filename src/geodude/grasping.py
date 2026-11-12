"""Grasp regions from prl_assets metadata, as ssgrasping inputs (#260).

An asset that declares ``grasp_regions`` is grasped from them alone: each named region,
parsed and validated by prl_assets' own reader (``prl_assets.grasp_regions.parse``),
becomes an ssgrasping ``GraspRegion`` placed in the asset frame by its
``T_asset_region``. prl_assets' asset frame is the object's MuJoCo body frame, which is
the pose the runtime registers for the object, so the regions compose with it directly.
An asset that does not declare them is grasped by the legacy path, from its
``geometric_properties``; one asset is never grasped by both. Since prl_assets#40 the
recycling kinds declare regions equal to their legacy primitive, so they take this path
and make the legacy path's candidates. A ``grasp_regions`` section that is malformed,
or declares no region, is an error, never a reason to fall back.

The hand is Geodude's Robotiq 2F-140, through ssgrasping's sstsr adapter, with the grasp
policy the legacy path applies (``templates``); each arm's profile names that arm's
manipulator, hand, gripper, and opening channel. A region ssgrasping has no primitive for
(a torus) is refused on its own, and the asset is grasped by its other regions.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from prl_assets.grasp_regions import parse
from ssgrasping import GraspingProfile, GraspRegion, Primitive
from ssgrasping.sstsr import SstsrTemplates, profile
from tsr.hands.parallel_jaw import Robotiq2F140

from geodude.regions import MIN_SIDE_DEPTH_M, pose
from geodude.scene import meta

HAND = Robotiq2F140()


class _GeodudeHand:
    """The 2F-140 with Geodude's grasp policy, the one the legacy path applies
    (``regions.pregrasps``): cylinder side grasps no deeper than ``MIN_SIDE_DEPTH_M``
    (the deep ones fold the fingers, sstsr#189, geodude_assets#36), and box grasps whose
    fingertips keep ``clearance`` off the box's edges and far face beyond sstsr's default
    margin (#237). A box family no grasp of which fits the hand yields none."""

    def __init__(self, clearance: float) -> None:
        self._hand = Robotiq2F140()
        self._clearance = clearance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hand, name)

    def grasp_cylinder_side(self, cylinder_radius: float, cylinder_height: float) -> list[Any]:
        return [
            t
            for t in self._hand.grasp_cylinder_side(cylinder_radius, cylinder_height)
            if np.linalg.norm(t.Tw_e[:3, 3]) >= MIN_SIDE_DEPTH_M - 1e-4
        ]

    def _box(self, family: str, axis: int, size: tuple[float, float, float]) -> list[Any]:
        extent = size[axis]  # the box's extent along the approach
        hand = self._hand
        margin = max(self._clearance, hand.clearance_fraction * min(hand.finger_length, extent))
        try:
            return list(getattr(hand, family)(*size, clearance=margin))
        except ValueError:  # no grasp of this family fits the gripper
            return []

    def grasp_box_face_x(self, box_x: float, box_y: float, box_z: float) -> list[Any]:
        return self._box("grasp_box_face_x", 0, (box_x, box_y, box_z))

    def grasp_box_face_y(self, box_x: float, box_y: float, box_z: float) -> list[Any]:
        return self._box("grasp_box_face_y", 1, (box_x, box_y, box_z))

    def grasp_box_top(self, box_x: float, box_y: float, box_z: float) -> list[Any]:
        return self._box("grasp_box_top", 2, (box_x, box_y, box_z))

    def grasp_box_bottom(self, box_x: float, box_y: float, box_z: float) -> list[Any]:
        return self._box("grasp_box_bottom", 2, (box_x, box_y, box_z))


@functools.cache
def templates(clearance: float) -> SstsrTemplates:
    """The 2F-140's grasps, by Geodude's policy, of each primitive sstsr's registry
    verifies for it."""
    return SstsrTemplates(_GeodudeHand(clearance), registry_key="robotiq_2f140")


def profile_for(manipulator: str, hand: str, gripper: str, opening: str) -> GraspingProfile:
    """The 2F-140 on the arm with these names."""
    return profile(
        HAND, manipulator=manipulator, end_effector=hand, gripper=gripper, opening_channel=opening
    )


def _primitive(kind: str, params: Mapping[str, Any]) -> Primitive | None:
    if kind == "cylinder":
        return Primitive("cylinder", (params["radius"], params["height"]))
    if kind == "box":
        return Primitive("box", tuple(params["size"]))
    if kind == "sphere":
        return Primitive("sphere", (params["radius"],))
    return None  # a torus: ssgrasping has no primitive for it


@dataclass(frozen=True)
class Declared:
    """An asset's grasp regions as ssgrasping takes them, and the names of those it
    declares that ssgrasping cannot grasp (a torus), refused one by one."""

    regions: tuple[GraspRegion, ...]
    refused: tuple[str, ...]


def regions_of(meta: Mapping[str, Any]) -> Declared | None:
    """The asset's grasp regions, or None if it declares none (the legacy path).

    Raises if its ``grasp_regions`` is malformed (prl_assets' validation), or declares no
    region ssgrasping can grasp."""
    if "grasp_regions" not in meta:
        return None
    usable, refused = [], []
    for r in parse(dict(meta)):
        primitive = _primitive(r.primitive, r.params)
        if primitive is None:
            refused.append(r.name)
            continue
        usable.append(
            GraspRegion(
                r.name,
                primitive,
                pose(np.asarray(r.T_asset_region, dtype=float)),
                tuple(r.strategies),
            )
        )
    if not usable:
        raise ValueError(f"{meta.get('name')!r} declares no grasp region ssgrasping can grasp")
    return Declared(tuple(usable), tuple(refused))


@functools.cache
def regions_for(kind: str) -> Declared | None:
    """``regions_of`` the prl_assets object ``kind``, parsed once."""
    return regions_of(meta(kind))
