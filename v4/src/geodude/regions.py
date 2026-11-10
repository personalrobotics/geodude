"""Task space regions for Geodude's objects, from their prl_assets geometry and sstsr.

Grasp regions come from sstsr's templates for Geodude's Robotiq 2F-140, and placement
regions from sstsr's ``StablePlacer``; both are instantiated at an object's or a
surface's pose and converted to ssrobot's ``Tsr``. A container's drop regions are for
the object, from sstsr's ``OpeningPlacer`` on the opening its prl_assets ``placement``
declares (measured from its collision geometry, prl_assets#49), and its interior is the
legacy demo's.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np
from prl_assets.placement import parse as parse_placement
from ssrobot import Pose, Tsr
from tsr.hands.parallel_jaw import Robotiq2F140
from tsr.placement import OpeningPlacer, Shape
from tsr.placement.stable_placer import StablePlacer

STANDING_TOLERANCE_RAD = math.radians(2.0)
"""How far a rest ``place`` finds may tilt from the one an object's model file stands it
in and still be that rest: a model file rests its object on z = 0 to prl_assets' 1 mm
tolerance, not on an exactly level face."""
STANDOFF_M = 0.08
"""How far back along the approach axis a pregrasp is from its grasp."""
MIN_SIDE_DEPTH_M = 0.099
"""The least palm-to-axis distance of a cylinder side grasp offered: sstsr's shallow
(114 mm) and mid (99 mm) grasps, which hold the can within 15 mm of the fingertips, where
the modelled 2F-140's fingers stay parallel; its deep (84 mm) grasps fold them, and the can
slips in the carry (sstsr#189, geodude_assets#36). Over 100 seeds of four cans, racing,
with no clearance: 344 cans recycled against 313 with every depth."""


def matrix(pose: Pose) -> np.ndarray:
    """``pose`` as a 4x4 homogeneous transform."""
    w, x, y, z = pose.quat_wxyz
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    out[:3, 3] = pose.position
    return out


def pose(transform: np.ndarray) -> Pose:
    """A 4x4 homogeneous transform as a ``Pose``. The quaternion is MuJoCo's
    (``mju_mat2Quat``), which recovers every rotation, including the half turns whose
    signs cannot be read from the matrix's antisymmetric part (the roll-180° cylinder side
    grasps, for one)."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(transform[:3, :3], dtype=float).flatten())
    w, x, y, z = (float(v) for v in quat / np.linalg.norm(quat))
    px, py, pz = (float(v) for v in transform[:3, 3])
    return Pose(position=(px, py, pz), quat_wxyz=(w, x, y, z))


def tsr(region: Any) -> Tsr:
    """An instantiated sstsr region as ssrobot's ``Tsr``."""
    rows = [(float(low), float(high)) for low, high in region.Bw]
    return Tsr(
        origin=pose(region.T0_w),
        offset=pose(region.Tw_e),
        position_bounds=(rows[0], rows[1], rows[2]),
        rotation_bounds=(rows[3], rows[4], rows[5]),
    )


def height(geometry: Mapping[str, Any]) -> float:
    """An upright object's height."""
    return float(geometry["height"] if geometry["type"] == "cylinder" else geometry["size"][2])


def pregrasps(geometry: Mapping[str, Any], centre: Pose, clearance: float) -> list[Tsr]:
    """sstsr's grasps of an object with ``geometry`` (prl_assets ``geometric_properties``)
    standing at ``centre``, the families the legacy demo uses: a cylinder from the side,
    no deeper than ``MIN_SIDE_DEPTH_M``; a box on its x and y faces, from the top, and
    from the bottom. Each is backed off by ``STANDOFF_M`` along its approach. sstsr frames
    a grasped object at the middle of its base.

    A box's grasps keep their fingertips at least ``clearance`` off its edges and its far
    face, as well as sstsr's default margin, a tenth of the fingers' reach into the box:
    closed from the top on a 28 mm gelatin box, that default left the fingertips 2.8 mm
    above its base, and the four-bar fingers, swinging forward as they close, came to
    rest on the worktop, so the lift started in contact with it (#237)."""
    hand = Robotiq2F140()
    if geometry["type"] == "cylinder":
        templates = [
            t
            for t in hand.grasp_cylinder_side(geometry["radius"], geometry["height"])
            if np.linalg.norm(t.Tw_e[:3, 3]) >= MIN_SIDE_DEPTH_M - 1e-4
        ]
    elif geometry["type"] == "box":
        templates = []
        for family, axis in (
            (hand.grasp_box_face_x, 0),
            (hand.grasp_box_face_y, 1),
            (hand.grasp_box_top, 2),
            (hand.grasp_box_bottom, 2),
        ):
            extent = float(geometry["size"][axis])  # the box's extent along the approach
            margin = max(clearance, hand.clearance_fraction * min(hand.finger_length, extent))
            try:
                templates += family(*geometry["size"], clearance=margin)
            except ValueError:  # no grasp of this family fits the gripper
                continue
    else:
        raise ValueError(f"no grasps for a {geometry['type']}")
    base = matrix(centre)
    base[:3, 3] -= base[:3, 2] * height(geometry) / 2
    back = np.eye(4)
    back[2, 3] = -STANDOFF_M
    return [tsr(dataclasses.replace(t, Tw_e=t.Tw_e @ back).instantiate(base)) for t in templates]


@dataclasses.dataclass(frozen=True)
class Opening:
    """A container's rectangular top opening, standing upright at a base position: its
    centre on the rim plane, in the world, and its interior's half extents and depth."""

    rim: tuple[float, float, float]
    half: tuple[float, float]
    depth: float


def opening(container: Mapping[str, Any], base: tuple[float, float, float]) -> Opening:
    """The top opening ``container`` (prl_assets ``meta.yaml``) declares, with the
    container standing unrotated with its frame at ``base``. Only a level rectangle,
    axis-aligned in the container's frame, is a drop target: that is what the zones
    below bound."""
    declared = parse_placement(dict(container))
    found = [o for o in (declared.openings if declared else ()) if o.name == "top"]
    if not found:
        raise ValueError(f"{container.get('name')!r} declares no top opening")
    top = found[0]
    frame = top.frame()
    if top.shape != "rectangle" or not np.allclose(frame[:3, :3], np.eye(3), atol=1e-9):
        name = container.get("name")
        raise ValueError(f"cannot drop into {name!r}: its opening is not a level rectangle")
    sx, sy = top.size
    x, y, z = (float(b + t) for b, t in zip(base, frame[:3, 3], strict=True))
    return Opening(rim=(x, y, z), half=(float(sx) / 2, float(sy) / 2), depth=float(top.depth))


RELEASE_LOWEST_M = (0.02, 0.10)
"""How far above a bin's rim a dropped object's lowest point is released: above it, so no
part of the hand needs to reach below the rim, and low enough that it falls straight."""
RELEASE_DRIFT_M = 0.016
"""How far sideways a released object strays as the pads let go of one side first, before
it falls past the rim: up to 16 mm over 83 free releases (``scripts/calibrate_release.py``),
measured with the hand pointing down, as the legacy zone released. A corner that strays
over the rim strikes it, so a release keeps this inside the opening too (#237)."""
MIN_REST_MARGIN_DEG = 3.0
"""The least tipping angle of a resting orientation a drop releases in: a scanned hull has
many barely stable facets (a lying can's), and released on one, the object rolls."""


def bin_drops(
    container: Mapping[str, Any],
    base: tuple[float, float, float],
    shape: Shape,
    clearance: float,
) -> list[Tsr]:
    """Where to release the object ``shape`` describes into ``container`` (prl_assets
    ``meta.yaml``) standing at ``base``, by sstsr's ``OpeningPlacer`` on its declared top
    opening: regions for the object's frame, one per resting orientation that fits, any
    yaw, its lowest point ``RELEASE_LOWEST_M`` above the rim, and the whole object
    ``clearance`` plus ``RELEASE_DRIFT_M`` inside the opening. Empty when no
    orientation fits. They are for the object, not the hand: ssgrasping's ``drop``
    composes them with how it is held."""
    top = opening(container, base)
    placer = OpeningPlacer(size=(2 * top.half[0], 2 * top.half[1]), depth=top.depth)
    templates = placer.place(
        shape,
        lowest=RELEASE_LOWEST_M,
        clearance=clearance + RELEASE_DRIFT_M,
        min_margin_deg=MIN_REST_MARGIN_DEG,
    )
    frame = np.eye(4)
    frame[:3, 3] = top.rim
    return [tsr(t.instantiate(frame)) for t in templates]


def _span(shape: Shape, rotation: np.ndarray, origin: np.ndarray, axis: int) -> tuple[float, float]:
    """The least and greatest world coordinate along ``axis`` of the object ``shape``
    describes, at ``rotation`` with its origin at ``origin``: exact, from its support."""
    n = rotation[axis]  # the world axis, in the object's frame
    return float(origin[axis]) - shape.support(-n), float(origin[axis]) + shape.support(n)


def lowest(shape: Shape, at: np.ndarray) -> float:
    """The world height of the lowest point of the object ``shape`` describes, at ``at``."""
    return _span(shape, at[:3, :3], at[:3, 3], 2)[0]


def tilt(at: np.ndarray, rest: np.ndarray) -> float:
    """How far, in radians, an object at ``at`` is tilted from standing as ``rest`` does:
    the angle between world up, in the object's frame, at each."""
    now, standing = at[:3, :3].T[:, 2], rest[:3, :3].T[:, 2]
    return math.acos(float(np.clip(now @ standing, -1.0, 1.0)))


def interior(container: Mapping[str, Any], base: tuple[float, float, float]) -> Tsr:
    """Where an object's frame may rest inside an upright open box (prl_assets
    ``meta.yaml``) standing at ``base``: within its ``opening``, between its floor and
    its rim, any orientation."""
    top = opening(container, base)
    (hx, hy), (x, y, rim) = top.half, top.rim
    return Tsr(
        origin=Pose(position=(x, y, rim), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        position_bounds=((-hx, hx), (-hy, hy), (-top.depth, 0.0)),
        rotation_bounds=((-math.pi, math.pi),) * 3,
    )


def standing(placer: StablePlacer, shape: Shape, rest: np.ndarray) -> Any:
    """sstsr's template (``StablePlacer.place``) for the object ``shape`` describes
    resting as ``rest``, its model file's pose on a level surface, stands it: the rest
    whose up direction, in the object's frame, is nearest ``rest``'s, within
    ``STANDING_TOLERANCE_RAD``, with roll and pitch held at zero (sstsr frees them for a
    lone centred sphere, which rests the same whichever way it turns). sstsr frames the
    object at its own origin."""
    templates = placer.place(shape)
    if not templates:
        raise ValueError("sstsr finds no stable rest that fits on the surface")
    up = rest[:3, :3].T[:, 2]
    best, angle = None, math.inf
    for template in templates:
        frame = template.T_ref_tsr[:3, :3] @ template.Tw_e[:3, :3]
        between = math.acos(float(np.clip(frame.T[:, 2] @ up, -1.0, 1.0)))
        if between < angle:
            best, angle = template, between
    if best is None or angle > STANDING_TOLERANCE_RAD:
        raise ValueError(
            f"no rest that fits on the surface stands the object as its model file does:"
            f" the nearest is {math.degrees(angle):.1f} deg from it"
        )
    bounds = best.Bw.copy()
    bounds[3] = bounds[4] = (0.0, 0.0)
    return dataclasses.replace(best, Bw=bounds)


def upright(shape: Shape, rest: np.ndarray, spot: tuple[float, float, float], slack: float) -> Tsr:
    """sstsr's region for the object ``shape`` describes standing as ``rest`` does
    (``standing``) on a level surface, its origin within ``slack`` metres of ``spot`` in
    x and y, any yaw."""
    template = standing(StablePlacer(table_x=1.0, table_y=1.0), shape, rest)
    bounds = template.Bw.copy()
    bounds[0] = (-slack, slack)
    bounds[1] = (-slack, slack)
    surface = np.eye(4)
    surface[:3, 3] = spot
    return tsr(dataclasses.replace(template, Bw=bounds).instantiate(surface))
