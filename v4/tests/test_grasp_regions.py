"""Geodude grasps two mugs by the grasp regions prl_assets declares for them (#260).

prl_assets' mug (``objects/ycbv/mug``) declares a ``body`` cylinder and a ``handle`` box,
each offset from the mug's frame and the handle turned about z. Two mugs are staged at
``SEED`` on the worktop, beside the bins, from the installed packages. An asset that
declares grasp regions is staged from its physical data alone: the footprint of its
``dimensions``, standing as its model file rests it. Each mug is then picked in its own
run of that scene, by the arm on its side, and lifted. Recorded in ``grasp_regions.json``
per mug:
- its registered pose, each region's world frame (its pose composed with the region's
  ``T_asset_region``, checked to 1e-9), and the regions' labels and hand;
- the region the grasp's planned pregrasp ended in (``pregrasp_region``), and the grasp's
  stages and outcome, and how far the mug rose with the hand (privileged, evaluation);
- its staging footprint and resting height, from ``dimensions`` and the model file.
The mug's legacy ``geometric_properties`` (a box, ``open: true``) decides nothing here:
it is not grasped as a box, not staged from it, and not a bin (the bins are ``BINS``).

Reproduce with ``uv run pytest tests/test_grasp_regions.py``.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ssgrasping import GraspObject, grasp_regions
from ssrobot import RobotContext, load_installed_package

from geodude import grasping, recycling, regions
from geodude.scene import meta, stage_scene

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
        declared = grasping.regions_of(meta("mug"))
        assert declared is not None
        candidates = grasp_regions(
            GraspObject(name, pose, declared),
            grasping.profile_for(arm.manipulator, arm.hand, arm.gripper, arm.opening),
            grasping.TEMPLATES,
            approach_m=regions.STANDOFF_M,
        )
        before = recycling.pose_channel(ctx, name)[0]
        result = recycling.pick(ctx, name, arm=arm, planners=recycling.planners(package, arm))
        after = recycling.pose_channel(ctx, name)[0]
    return {
        "pose": pose,
        "frames": {r.region.name: r.frame for r in candidates.regions},
        "spec_matches": tuple(spec.pregrasps) == candidates.tsrs,
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
    declared = grasping.regions_of(mug)
    assert declared is not None and [r.name for r in declared] == ["body", "handle"]
    with tempfile.TemporaryDirectory() as work:
        scene = stage_scene(
            Path(work) / "scene", package, objects={"mug": 2}, fixtures=recycling.BINS, seed=SEED
        )
        names = ["mug_0", "mug_1"]
        picks = {name: _pick(scene, package, name, names) for name in names}

    report: dict[str, Any] = {"seed": SEED, "hand": type(grasping.HAND).__name__, "mugs": {}}
    poses = [picks[n]["pose"] for n in names]
    assert poses[0] != poses[1]  # two instances, two poses
    x, y, _ = mug["dimensions"]
    for name, pick in picks.items():
        # Each region's world frame is the mug's registered pose composed with the
        # region's T_asset_region, as prl_assets declares it.
        frames = {}
        declared_rows = {r["name"]: r["T_asset_region"] for r in mug["grasp_regions"]["regions"]}
        for region in declared:
            # The adapter's origin is prl_assets' T_asset_region, and ssgrasping places the
            # region at the registered pose composed with it.
            assert np.allclose(regions.matrix(region.origin), declared_rows[region.name], atol=1e-9)
            expected = regions.matrix(pick["pose"]) @ np.asarray(declared_rows[region.name])
            world = pick["frames"][region.name]
            assert np.allclose(regions.matrix(world), expected, atol=1e-9), (name, region.name)
            frames[region.name] = _frame(world)
        assert pick["spec_matches"], name  # the grasp ran on exactly these regions
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
                "footprint_radius_m": math.hypot(x, y) / 2,
                "source": "dimensions and the model file's rest pose",
            },
        }
    (artifacts / "grasp_regions.json").write_text(json.dumps(report, indent=2) + "\n")
