"""Geodude grasps two mugs by the grasp regions prl_assets declares for them (#260).

prl_assets' mug (``objects/ycbv/mug``) declares a ``body`` cylinder and a ``handle`` box,
each offset from the mug's frame and the handle turned about z. Two mugs are staged at
``SEED`` on the worktop, beside the bins, from the installed packages. An asset that
declares grasp regions is staged from its physical data alone: standing as its model
file rests it, its footprint bounding its geoms about its frame. Each mug is then picked in its own
run of that scene, by the arm on its side, and lifted. Recorded in ``grasp_regions.json``
per mug:
- its registered pose, each region's world frame as ssgrasping placed it, and the
  regions' labels and hand. Each region's origin is checked to equal the
  ``T_asset_region`` prl_assets declares (1e-9): Geodude's mapping;
- the region the grasp's planned pregrasp ended in (``pregrasp_region``), and the grasp's
  stages and outcome, and how far the mug rose with the hand (privileged, evaluation);
- its staging footprint and resting height, from its compiled model.
The mug's legacy ``geometric_properties`` (a box, ``open: true``) decides nothing here:
it is not grasped as a box, not staged from it, and not a bin (the bins are ``BINS``).

Reproduce with ``uv run pytest tests/test_grasp_regions.py``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ssgrasping import GraspObject, grasp_regions
from ssrobot import RobotContext, load_installed_package
from tsr.placement.stable_placer import StablePlacer

from geodude import grasping, recycling, regions
from geodude.scene import _physical, meta, stage_scene

SEED = 0
LIFT_M = 0.06


def _frame(pose: Any) -> list[list[float]]:
    return [list(pose.position), list(pose.quat_wxyz)]


def _pick(scene: Path, package: Any, name: str, everyone: list[str]) -> dict[str, Any]:
    with RobotContext(
        package.description, recycling.runtime(package, scene), object_poses=recycling.OBJECT_POSES
    ) as ctx:
        recycling.register(ctx, everyone)
        pose = recycling.registered_pose(ctx, name)
        arm = recycling.arm_for(pose.position)
        spec = recycling.grasp_spec(ctx, name, arm)
        declared = grasping.regions_for("mug")
        assert declared is not None
        # For the artifact: where ssgrasping placed each region for this registered pose.
        candidates = grasp_regions(
            GraspObject(name, pose, declared.regions),
            grasping.profile_for(arm.manipulator, arm.hand, arm.gripper, arm.opening),
            grasping.templates(recycling.CLEARANCE_M),
            approach_m=regions.STANDOFF_M,
        )
        before = recycling.pose_channel(ctx, name)[0]
        result = recycling.pick(ctx, name, arm=arm, planners=recycling.planners(package, arm))
        after = recycling.pose_channel(ctx, name)[0]
    return {
        "pose": pose,
        "frames": {r.region.name: r.frame for r in candidates.regions},
        "arm": arm.side,
        "labels": list(spec.pregrasp_labels or ()),
        "state": result.state.value,
        "stages": [(s.name, s.succeeded) for s in result.stages],
        "pregrasp_region": result.values.get("pregrasp_region"),
        "rose_m": after.position[2] - before.position[2],
    }


def test_geodude_grasps_mugs_by_their_declared_regions(artifacts: Path) -> None:
    package = load_installed_package("geodude_assets")
    mug = meta("mug")
    declared = grasping.regions_for("mug")
    assert declared is not None and declared.refused == ()
    assert [r.name for r in declared.regions] == ["body", "handle"]
    # Geodude's mapping: each region's origin is the T_asset_region prl_assets declares.
    declared_rows = {r["name"]: r["T_asset_region"] for r in mug["grasp_regions"]["regions"]}
    for region in declared.regions:
        assert np.allclose(regions.matrix(region.origin), declared_rows[region.name], atol=1e-9)
    _, footprint_m, rest = _physical("mug", StablePlacer(0.5, 0.5))
    with tempfile.TemporaryDirectory() as work:
        scene = stage_scene(
            Path(work) / "scene", package, objects={"mug": 2}, fixtures=recycling.BINS, seed=SEED
        )
        names = ["mug_0", "mug_1"]
        picks = {name: _pick(scene, package, name, names) for name in names}

    report: dict[str, Any] = {"seed": SEED, "hand": type(grasping.HAND).__name__, "mugs": {}}
    poses = [picks[n]["pose"] for n in names]
    assert poses[0] != poses[1]  # two instances, two poses
    for name, pick in picks.items():
        frames = {region: _frame(frame) for region, frame in pick["frames"].items()}
        assert pick["labels"] and all(
            lab.split("/")[0] in {"body", "handle"} for lab in pick["labels"]
        ), pick["labels"]
        assert pick["state"] == "succeeded", (name, pick["stages"])
        reached = pick["pregrasp_region"]
        assert reached and all(lab.split("/")[0] in {"body", "handle"} for lab in reached)
        assert pick["rose_m"] > LIFT_M / 2, (name, pick["rose_m"])  # it rose with the hand
        report["mugs"][name] = {
            "registered_pose": _frame(pick["pose"]),
            "region_frames": frames,
            "arm": pick["arm"],
            "labels": pick["labels"],
            "pregrasp_region": reached,
            "state": pick["state"],
            "stages": pick["stages"],
            "rose_m": pick["rose_m"],
            "staging": {
                "footprint_radius_m": footprint_m,
                "rest_height_m": float(rest[2, 3]),
                "source": "the compiled model: its rest pose at qpos0 and its geoms' boxes",
            },
        }
    (artifacts / "grasp_regions.json").write_text(json.dumps(report, indent=2) + "\n")
