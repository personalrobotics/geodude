"""A drop zone keeps a held object's collision geometry inside the bin's opening, wherever
the object's frame is (#260).

``regions.held_drop_zones`` bounds the held object by its collision shape's support. The
end-to-end rounds hold only the recycling kinds, whose frames are centred on their
primitives, so they cannot tell that bound from one about the frame; a YCB-V object keeps
BOP's frame. The mug's collision geometry is 23 convex hulls, so their vertices are its
exact extent, independent of the support function. For held poses drawn at random, every
zone must put the hand where, at either end of each of its bounds, the vertices come
exactly ``CLEARANCE_M`` inside the opening (none outside, the zone no smaller than it
must be), with the lowest vertex ``CLEARANCE_M`` below the rim. The mug hangs far enough
below the hand that the open fingers stay above the rim, so the mug alone sets the
bounds. The legacy primitive, a box about the mug's frame, encloses the mug with room to
spare, so its zones leave the mug millimetres short of the walls: the check tells a
bound about the frame from the exact one.

The evidence is ``held_drop_zones.json``: per held pose, the zone count and, for the
shape and the legacy box, the least and greatest margin the vertices leave at the
bounds and the lowest vertex's error. Reproduce with
``uv run pytest tests/test_held_drop_zones.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ssrobot import Pose, Tsr
from tsr.placement import Box, Hull, Shape

from geodude import recycling, regions
from geodude.scene import meta, physical

POSES = 20
TOLERANCE_M = 1e-9


def _margins(zones: list[Tsr], points: np.ndarray, in_hand: np.ndarray) -> dict[str, float]:
    """Over ``zones``, with ``points`` (object frame) held at ``in_hand``: the least and
    greatest margin inside the opening less the clearance, with the hand at either end
    of each bound, and the most the lowest point misses the clearance below the rim."""
    top = regions.opening(meta("recycle_bin"), recycling.BINS["recycle_bin"][0])
    margins, depth = [], 0.0
    for zone in zones:
        turned = regions.matrix(zone.offset)[:3, :3]
        local = (turned @ (in_hand[:3, :3] @ points.T + in_hand[:3, 3:4])).T  # from the hand
        lowest = zone.origin.position[2] + local[:, 2].min()
        depth = max(depth, abs(lowest - (top.rim[2] - recycling.CLEARANCE_M)))
        for axis in range(2):
            limit = top.half[axis] - recycling.CLEARANCE_M
            low, high = zone.position_bounds[axis]
            centre = zone.origin.position[axis] - top.rim[axis]
            margins.append(limit + centre + low + local[:, axis].min())
            margins.append(limit - (centre + high + local[:, axis].max()))
    return {"least_m": min(margins), "most_m": max(margins), "depth_m": depth}


def test_a_drop_zone_bounds_the_held_collision_geometry(artifacts: Path) -> None:
    shape, _ = physical("mug")
    assert all(isinstance(p, Hull) for p in shape.parts)
    vertices = np.vstack([p.vertices for p in shape.parts])  # exact: each part is a hull
    legacy = Shape([Box(meta("mug")["geometric_properties"]["size"])])  # about the frame
    bin_, base = meta("recycle_bin"), recycling.BINS["recycle_bin"][0]
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for _ in range(POSES):
        q = rng.normal(size=4)
        in_hand = regions.matrix(
            Pose(position=(0.0, 0.0, 0.0), quat_wxyz=tuple(q / np.linalg.norm(q)))
        )
        # Far enough below the hand that its lowest point is below the open fingers'.
        in_hand[:3, 3] = (*rng.uniform(-0.02, 0.02, size=2), rng.uniform(0.22, 0.26))
        grasp = regions.pose(in_hand)
        zones = regions.held_drop_zones(bin_, base, shape, grasp, recycling.CLEARANCE_M)
        boxed = regions.held_drop_zones(bin_, base, legacy, grasp, recycling.CLEARANCE_M)
        rows.append(
            {
                "in_hand": in_hand.tolist(),
                "zones": len(zones),
                "shape": _margins(zones, vertices, in_hand) if zones else None,
                "legacy_box": _margins(boxed, vertices, in_hand) if boxed else None,
            }
        )
    (artifacts / "held_drop_zones.json").write_text(json.dumps(rows, indent=2) + "\n")
    held = [r["shape"] for r in rows if r["shape"]]
    assert len(held) == POSES, "a held pose has no zone"
    for r in held:
        assert r["least_m"] >= -TOLERANCE_M, r  # no vertex outside, at any bound
        assert r["most_m"] <= TOLERANCE_M, r  # and the zone no smaller than it must be
        assert r["depth_m"] <= TOLERANCE_M, r  # the lowest vertex the clearance below the rim
    loosest = max(r["legacy_box"]["most_m"] for r in rows if r["legacy_box"])
    assert loosest > 1e-3, f"the legacy box bounds the mug within {loosest} m"
