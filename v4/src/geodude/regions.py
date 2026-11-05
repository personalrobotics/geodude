"""Task space regions for Geodude's objects, from their prl_assets geometry and sstsr.

Grasp regions come from sstsr's templates for Geodude's Robotiq 2F-140, and placement
regions from sstsr's ``StablePlacer``; both are instantiated at an object's or a
surface's pose and converted to ssrobot's ``Tsr``. A container's drop zone and interior
are the legacy demo's, from the opening its prl_assets ``placement`` declares (measured
from its collision geometry, prl_assets#49); they are this application's, not sstsr's or
ssgrasping's.
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
from tsr.placement import Shape
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
OPEN_HAND_HALF_M = (0.0375, 0.105)
"""Half the fully open 2F-140's extent across its grasp site's x and y (y the fingers'
axis): the bounding boxes of geodude_assets' gripper collision geoms, open as the robot
starts, in the grasp site's frame. Its reach along the approach is sstsr's
``Robotiq2F140.finger_length``, measured from the same model."""


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


def drop_zone(
    container: Mapping[str, Any],
    base: tuple[float, float, float],
    held_height: float,
) -> Tsr:
    """The legacy demo's drop zone over an upright open box (prl_assets ``meta.yaml``)
    standing at ``base``, from mj_manipulator's ``_generate_container_drop_tsrs``: the end
    effector pointing straight down, anywhere over the ``opening`` less the container's
    ``drop_zone_margin``, ``max(0.25, held_height + 0.10)`` above the rim (2 cm higher to
    5 cm lower), at any yaw."""
    top = opening(container, base)
    margin = float(container.get("policy", {}).get("placement", {}).get("drop_zone_margin", 0.05))
    hx, hy = (h - margin for h in top.half)
    x, y, z = top.rim
    return Tsr(
        origin=Pose(
            position=(x, y, z + max(0.25, held_height + 0.10)),
            quat_wxyz=(0.0, 1.0, 0.0, 0.0),  # x along the world's, z pointing down
        ),
        offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        position_bounds=((-hx, hx), (-hy, hy), (-0.02, 0.05)),
        rotation_bounds=((0.0, 0.0), (0.0, 0.0), (-math.pi, math.pi)),
    )


def _span(shape: Shape, rotation: np.ndarray, origin: np.ndarray, axis: int) -> tuple[float, float]:
    """The least and greatest world coordinate along ``axis`` of the object ``shape``
    describes, at ``rotation`` with its origin at ``origin``: exact, from its support."""
    n = rotation[axis]  # the world axis, in the object's frame
    return float(origin[axis]) - shape.support(-n), float(origin[axis]) + shape.support(n)


def held_drop_zones(
    container: Mapping[str, Any],
    base: tuple[float, float, float],
    held: Shape,
    grasp: Pose,
    clearance: float,
) -> list[Tsr]:
    """Where to release the object ``held`` describes (its collision geometry, in its own
    frame), held at ``grasp`` (its pose in the end effector's frame), into an upright
    open box standing at ``base``, so that no part of it can come to rest on the rim:
    the end effector pointing straight down, low enough that the object's lowest point
    is ``clearance`` below the rim, at each yaw a multiple of a quarter turn, and only
    where the held object's footprint lies inside the box's ``opening`` by
    ``clearance``. Where the open fingers then reach below the
    rim too, the open hand's footprint (``OPEN_HAND_HALF_M``) must also lie inside them.
    The footprint and lowest point are the collision geometry's, wherever its frame is.

    Released above the rim, an object strays sideways as the pads let go of one side
    first, by up to 16 mm before it falls past the rim (``scripts/calibrate_release.py``,
    83 free releases): a corner that strays over the rim strikes it. Seed 0's pop-tarts
    case, released 4.7 mm inside the walls and 3.6 cm above the rim, swept 1 cm onto it
    and wedged there (#237). A margin of that sweep leaves no zone for a horizontally held
    0.21 m bottle or box in the 0.238 m opening. Released below the rim, a straying
    corner meets the inner wall's face instead, and slides down it.

    The legacy zone lets the hand anywhere within ``drop_zone_margin`` of the walls at
    any yaw, whatever it holds. Over #201's development corpus, the 11 objects released
    from it that did not land, all tall boxes and bottles, came to rest on or above the
    rim (0.805 to 0.892 m above the bin's base, against an 0.80 m rim), 10 of them more
    than 5 cm off-centre in x or y. A yaw at which the footprint does not fit has no zone."""
    top = opening(container, base)
    in_hand = matrix(grasp)  # the object's pose in the hand's frame
    down = np.diag([1.0, -1.0, -1.0])  # the hand's z pointing down, its x along the world's
    x, y, rim = top.rim
    # The object's lowest point, from the hand.
    bottom = _span(held, down @ in_hand[:3, :3], down @ in_hand[:3, 3], 2)[0]
    hand_z = rim - clearance - bottom
    reaches = hand_z - Robotiq2F140().finger_length < rim + clearance
    hx, hy = OPEN_HAND_HALF_M
    hand = np.array([[sx * hx, sy * hy, 0.0] for sx in (-1, 1) for sy in (-1, 1)]).T
    zones = []
    for quarter in range(4):
        yaw = quarter * math.pi / 2
        c, s = math.cos(yaw), math.sin(yaw)
        turned = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ down
        fingers = turned @ hand
        bounds = []
        for axis in range(2):
            # The object's extent from the hand, in the world, and the open fingers'.
            least, most = _span(held, turned @ in_hand[:3, :3], turned @ in_hand[:3, 3], axis)
            if reaches:  # the open fingers must clear the walls as well
                least = min(least, float(fingers[axis].min()))
                most = max(most, float(fingers[axis].max()))
            limit = top.half[axis] - clearance
            bounds.append((-limit - least, limit - most))
        if any(low > high for low, high in bounds):
            continue
        offset = np.eye(4)
        offset[:3, :3] = turned
        zones.append(
            Tsr(  # bounds in the world's axes: the origin is unturned, the hand's turn offset
                origin=Pose(position=(x, y, hand_z), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
                offset=pose(offset),
                position_bounds=(
                    (bounds[0][0], bounds[0][1]),
                    (bounds[1][0], bounds[1][1]),
                    (0.0, 0.0),
                ),
                rotation_bounds=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            )
        )
    return zones


def standing_height(shape: Shape, rest: np.ndarray) -> float:
    """How tall the object ``shape`` describes stands as ``rest`` stands it."""
    up = rest[:3, :3].T[:, 2]  # world up, in the object's frame
    return float(shape.support(up) + shape.support(-up))


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
