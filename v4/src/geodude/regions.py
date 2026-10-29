"""Task space regions for Geodude's objects, from their prl_assets geometry and sstsr.

Grasp regions come from sstsr's templates for Geodude's Robotiq 2F-140, and placement
regions from sstsr's ``StablePlacer``; both are instantiated at an object's or a
surface's pose and converted to ssrobot's ``Tsr``. A container's drop zone and interior
are the legacy demo's, from its prl_assets geometry; they are this application's, not
sstsr's or ssgrasping's.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np
from ssrobot import Pose, Tsr
from tsr.hands.parallel_jaw import Robotiq2F140
from tsr.placement.stable_placer import StablePlacer

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


def pregrasps(geometry: Mapping[str, Any], centre: Pose) -> list[Tsr]:
    """sstsr's grasps of an object with ``geometry`` (prl_assets ``geometric_properties``)
    standing at ``centre``, the families the legacy demo uses: a cylinder from the side,
    no deeper than ``MIN_SIDE_DEPTH_M``; a box on its x and y faces, from the top, and
    from the bottom. Each is backed off by ``STANDOFF_M`` along its approach. sstsr frames
    a grasped object at the middle of its base."""
    hand = Robotiq2F140()
    if geometry["type"] == "cylinder":
        templates = [
            t
            for t in hand.grasp_cylinder_side(geometry["radius"], geometry["height"])
            if np.linalg.norm(t.Tw_e[:3, 3]) >= MIN_SIDE_DEPTH_M - 1e-4
        ]
    elif geometry["type"] == "box":
        templates = []
        for family in (
            hand.grasp_box_face_x,
            hand.grasp_box_face_y,
            hand.grasp_box_top,
            hand.grasp_box_bottom,
        ):
            try:
                templates += family(*geometry["size"])
            except ValueError:  # no grasp of this family fits the gripper
                continue
    else:
        raise ValueError(f"no grasps for a {geometry['type']}")
    base = matrix(centre)
    base[:3, 3] -= base[:3, 2] * height(geometry) / 2
    back = np.eye(4)
    back[2, 3] = -STANDOFF_M
    return [tsr(dataclasses.replace(t, Tw_e=t.Tw_e @ back).instantiate(base)) for t in templates]


def drop_zone(
    container: Mapping[str, Any],
    base: tuple[float, float, float],
    held_height: float,
) -> Tsr:
    """The legacy demo's drop zone over an upright open box (prl_assets ``meta.yaml``)
    standing at ``base``, from mj_manipulator's ``_generate_container_drop_tsrs``: the end
    effector pointing straight down, anywhere over the interior less the container's
    ``drop_zone_margin``, ``max(0.25, held_height + 0.10)`` above the rim (2 cm higher to
    5 cm lower), at any yaw."""
    geometry = container["geometric_properties"]
    if geometry["type"] != "open_box":
        raise ValueError(f"cannot drop into a {geometry['type']}")
    outer = geometry["outer_size"]
    wall = float(geometry.get("wall_thickness", 0.003))
    margin = float(container.get("policy", {}).get("placement", {}).get("drop_zone_margin", 0.05))
    hx, hy = (float(outer[i]) / 2 - wall - margin for i in range(2))
    x, y, z = base
    return Tsr(
        origin=Pose(
            position=(x, y, z + float(outer[2]) + max(0.25, held_height + 0.10)),
            quat_wxyz=(0.0, 1.0, 0.0, 0.0),  # x along the world's, z pointing down
        ),
        offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        position_bounds=((-hx, hx), (-hy, hy), (-0.02, 0.05)),
        rotation_bounds=((0.0, 0.0), (0.0, 0.0), (-math.pi, math.pi)),
    )


def held_drop_zones(
    container: Mapping[str, Any],
    base: tuple[float, float, float],
    held: Mapping[str, Any],
    grasp: Pose,
    clearance: float,
) -> list[Tsr]:
    """Where to release an object with ``held`` geometry, held at ``grasp`` (its pose in
    the end effector's frame), into an upright open box standing at ``base``, so that no
    part of it can come to rest on the rim: the end effector pointing straight down, low
    enough that the object's lowest point is ``clearance`` below the rim, at each yaw a
    multiple of a quarter turn, and only where the held object's footprint lies inside
    the box's inner walls by ``clearance``. Where the open fingers then reach below the
    rim too, the open hand's footprint (``OPEN_HAND_HALF_M``) must also lie inside them.
    A cylinder's footprint is its bounding box's.

    Released above the rim, an object strays sideways as the pads let go of one side
    first, by up to 12 mm before it falls past the rim (``scripts/calibrate_release.py``,
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
    geometry = container["geometric_properties"]
    if geometry["type"] != "open_box":
        raise ValueError(f"cannot drop into a {geometry['type']}")
    outer, inner = geometry["outer_size"], geometry["inner_size"]
    if held["type"] == "cylinder":
        half = np.array([held["radius"], held["radius"], held["height"] / 2])
    else:
        half = np.array(held["size"], dtype=float) / 2
    corners = (np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3) * half).T
    in_hand = matrix(grasp)
    in_hand = in_hand[:3, :3] @ corners + in_hand[:3, 3:4]  # the object's corners, hand frame
    down = np.diag([1.0, -1.0, -1.0])  # the hand's z pointing down, its x along the world's
    x, y, z = base
    rim = z + float(outer[2])
    lowest = float((down @ in_hand)[2].min())  # the object's lowest point, from the hand
    hand_z = rim - clearance - lowest
    reaches = hand_z - Robotiq2F140().finger_length < rim + clearance
    hx, hy = OPEN_HAND_HALF_M
    hand = np.array([[sx * hx, sy * hy, 0.0] for sx in (-1, 1) for sy in (-1, 1)]).T
    zones = []
    for quarter in range(4):
        yaw = quarter * math.pi / 2
        c, s = math.cos(yaw), math.sin(yaw)
        turned = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ down
        footprint = turned @ in_hand  # each corner's offset from the hand, in the world
        if reaches:  # the open fingers must clear the walls as well
            footprint = np.hstack([footprint, turned @ hand])
        bounds = []
        for axis in range(2):
            limit = float(inner[axis]) / 2 - clearance
            low = -limit - float(footprint[axis].min())
            high = limit - float(footprint[axis].max())
            bounds.append((low, high))
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


def interior(container: Mapping[str, Any], base: tuple[float, float, float]) -> Tsr:
    """Where an object's frame may rest inside an upright open box (prl_assets
    ``meta.yaml``) standing at ``base``: over its floor, below its rim, any orientation."""
    geometry = container["geometric_properties"]
    outer, inner = geometry["outer_size"], geometry["inner_size"]
    hx, hy = float(inner[0]) / 2, float(inner[1]) / 2
    return Tsr(
        origin=Pose(position=base, quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        position_bounds=(
            (-hx, hx),
            (-hy, hy),
            (float(outer[2]) - float(inner[2]), float(outer[2])),
        ),
        rotation_bounds=((-math.pi, math.pi),) * 3,
    )


def upright(geometry: Mapping[str, Any], spot: tuple[float, float, float], slack: float) -> Tsr:
    """sstsr's region for an object with ``geometry`` resting upright, its -z face down,
    its centre within ``slack`` metres of ``spot`` in x and y on a level surface, any yaw.
    sstsr's surface extent bounds the object's footprint, and it frames a placed object
    at its centre."""
    if geometry["type"] == "cylinder":
        footprint = float(geometry["radius"])
    else:
        footprint = math.hypot(*geometry["size"][:2]) / 2
    placer = StablePlacer(table_x=footprint + slack, table_y=footprint + slack)
    if geometry["type"] == "cylinder":
        templates = placer.place_cylinder(geometry["radius"], geometry["height"])
    else:
        templates = placer.place_box(*geometry["size"])
    template = next(t for t in templates if t.variant == "-z")
    surface = np.eye(4)
    surface[:3, 3] = spot
    return tsr(template.instantiate(surface))
