"""Task space regions for Geodude's objects, from their prl_assets geometry and sstsr.

Grasp regions come from sstsr's templates for Geodude's Robotiq 2F-140, and placement
regions from sstsr's ``StablePlacer``; both are instantiated at an object's or a
surface's pose and converted to ssrobot's ``Tsr``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from ssrobot import Pose, Tsr
from tsr.hands.parallel_jaw import Robotiq2F140
from tsr.placement.stable_placer import StablePlacer

STANDOFF_M = 0.08
"""How far back along the approach axis a pregrasp is from its grasp."""


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
    at every depth; a box on its x and y faces, from the top, and from the bottom. Each is
    backed off by ``STANDOFF_M`` along its approach. sstsr frames a grasped object at the
    middle of its base."""
    hand = Robotiq2F140()
    if geometry["type"] == "cylinder":
        templates = hand.grasp_cylinder_side(geometry["radius"], geometry["height"])
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
