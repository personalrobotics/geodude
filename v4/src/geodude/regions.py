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
from collections.abc import Callable, Mapping
from typing import Any

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
PALM_CLEARANCE_M = 0.01
"""How close the palm (the hand's TSR frame, sstsr's ``grasp_site``) may come to an
object's surface before the fingers close."""


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
    """A 4x4 homogeneous transform as a ``Pose``."""
    r = transform[:3, :3]
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    px, py, pz = (float(v) for v in transform[:3, 3])
    return Pose(position=(px, py, pz), quat_wxyz=(w / n, x / n, y / n, z / n))


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


def deepest(geometry: Mapping[str, Any], centre: Pose) -> Callable[[Pose], float] | None:
    """For ssgrasping's ``GraspSpec.deepest`` (not used yet: in simulation the 2F-140 grips
    a can enveloped this deep with less than half the force of a fingertip grasp, and
    drops it more often; personalrobotics/geodude#223): how far the hand at a pregrasp pose may
    approach an object with ``geometry`` standing at ``centre``, so the palm stops
    ``PALM_CLEARANCE_M`` from its surface and the fingers envelop it. A cylinder: the
    distance along the approach axis to the cylinder's axis, less its radius and the
    clearance. None for other shapes: their grasps approach by ``STANDOFF_M``."""
    if geometry["type"] != "cylinder":
        return None
    radius = float(geometry["radius"])
    frame = matrix(centre)
    c, u = frame[:3, 3], frame[:3, 2]

    def reach(pose: Pose) -> float:
        hand = matrix(pose)
        p, a = hand[:3, 3], hand[:3, 2]
        w, a_off = c - p, a - (a @ u) * u  # both without their part along the axis
        w = w - (w @ u) * u
        return float((w @ a_off) / (a_off @ a_off)) - radius - PALM_CLEARANCE_M

    return reach


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
