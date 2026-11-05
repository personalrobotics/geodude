"""Geodude grasps two mugs by the grasp regions prl_assets declares for them (#260).

prl_assets' mug (``objects/ycbv/mug``) declares a ``body`` cylinder and a ``handle`` box,
each offset from the mug's frame and the handle turned about z. Two mugs are staged at
``SEED`` on the worktop, beside the bins, from the installed packages. An asset that
declares grasp regions is staged from its physical data alone: standing as its model
file rests it, in the rest sstsr's ``place`` finds from its collision geometry, its
footprint bounding its geoms about its frame. Each mug is then picked in its own
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

Two more tests hold the recycling kinds, which declare their legacy primitive as one
region (prl_assets#40), to the legacy path:
- ``parity.json``: at a pose about a skew axis, each kind's region candidates equal the
  legacy path's one for one (origin, offset, bounds; 1e-9), with the pose, the declared
  regions, and every candidate recorded;
- ``staging.json``: each kind's footprint and rest height from its compiled model, beside
  its legacy circle (within 2 mm) and half height (within 0.1 mm), and the height sstsr
  rests it at from its collision geometry (within 0.1 mm of its model file's).

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
from ssrobot import Pose, RobotContext, load_installed_package
from tsr.placement.stable_placer import StablePlacer

from geodude import grasping, recycling, regions
from geodude.scene import _standing, meta, physical, stage_scene

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
    _, footprint_m, rest = _standing("mug", StablePlacer(0.5, 0.5))
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


def test_the_regions_path_makes_the_legacy_paths_candidates(artifacts: Path) -> None:
    """For every recycling kind, the regions prl_assets declares (its legacy primitive at
    the object frame, with the legacy families: prl_assets#40) give exactly the
    candidates the legacy path gives, one for one: origin, offset, and bounds, at a pose
    whose axes are not the world's. Switching a kind to the regions path changes nothing
    its grasp sees."""
    axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    w, (x, y, z) = math.cos(0.45), math.sin(0.45) * axis
    at = Pose(position=(0.42, -0.31, 0.87), quat_wxyz=(w, float(x), float(y), float(z)))
    arm = recycling.Arm("right")
    report = {}
    for kind in recycling.MIX_KINDS:
        declared = grasping.regions_for(kind)
        assert declared is not None and declared.refused == (), kind
        legacy = regions.pregrasps(meta(kind)["geometric_properties"], at, recycling.CLEARANCE_M)
        ours = grasp_regions(
            GraspObject(kind, at, declared.regions),
            grasping.profile_for(arm.manipulator, arm.hand, arm.gripper, arm.opening),
            grasping.templates(recycling.CLEARANCE_M),
            approach_m=regions.STANDOFF_M,
        ).tsrs
        assert len(ours) == len(legacy) > 0, (kind, len(ours), len(legacy))
        worst = 0.0
        for a, b in zip(legacy, ours, strict=True):
            worst = max(
                worst,
                float(np.abs(regions.matrix(a.origin) - regions.matrix(b.origin)).max()),
                float(np.abs(regions.matrix(a.offset) - regions.matrix(b.offset)).max()),
                float(np.abs(np.subtract(a.position_bounds, b.position_bounds)).max()),
                float(np.abs(np.subtract(a.rotation_bounds, b.rotation_bounds)).max()),
            )
        assert worst < 1e-9, (kind, worst)
        report[kind] = {
            "regions": [
                {
                    "name": r.name,
                    "primitive": r.primitive.kind,
                    "dimensions_m": list(r.primitive.dimensions),
                    "origin": _frame(r.origin),
                    "strategies": list(r.strategies or ()),
                }
                for r in declared.regions
            ],
            "candidates": [{"origin": _frame(c.origin), "offset": _frame(c.offset)} for c in ours],
            "max_difference": worst,
        }
    (artifacts / "parity.json").write_text(
        json.dumps({"pose": _frame(at), "kinds": report}, indent=2) + "\n"
    )


def test_physical_staging_stands_the_recycling_kinds_as_the_legacy_staging_did(
    artifacts: Path,
) -> None:
    """Declaring regions also stages a kind from its compiled model (``scene.physical``)
    instead of its legacy primitive. For the recycling kinds, whose frames are centred
    and whose legacy primitive is their shape, the two agree: each footprint within 2 mm
    of the legacy circle, and each rests at half its legacy height (0.1 mm), so a seeded
    scene stands its objects where it did."""
    report = {}
    for kind in recycling.MIX_KINDS:
        geometry = meta(kind)["geometric_properties"]
        shape, rest = physical(kind)
        radius = shape.reach(rest[:3, :3])
        template = regions.standing(StablePlacer(0.5, 0.5), shape, rest)
        if geometry["type"] == "cylinder":  # the legacy circle covering its primitive
            legacy_radius = float(geometry["radius"])
        else:
            legacy_radius = math.hypot(*geometry["size"][:2]) / 2
        legacy_rest = regions.height(geometry) / 2
        # Within 2 mm: a footprint bounds the colliders, and the glass's, a ring of boxes
        # around its rim, reach 49.0 mm against its 48 mm round legacy circle.
        assert abs(radius - legacy_radius) <= 2e-3, (kind, radius, legacy_radius)
        # Within 0.1 mm: model files write the resting height rounded (the gelatin box's,
        # rebuilt to its scan, rests at 15.03 mm against half its 30.1 mm legacy height).
        assert abs(float(rest[2, 3]) - legacy_rest) <= 1e-4, (kind, rest[2, 3], legacy_rest)
        assert np.allclose(rest[:3, :3], np.eye(3), atol=1e-9), kind  # upright as modelled
        # sstsr, from the collision geometry alone, rests it where its model file does.
        sstsr_rest = float(template.T_ref_tsr[2, 3])
        assert abs(sstsr_rest - float(rest[2, 3])) <= 1e-4, (kind, sstsr_rest, rest[2, 3])
        report[kind] = {
            "footprint_m": radius,
            "legacy_footprint_m": legacy_radius,
            "rest_height_m": float(rest[2, 3]),
            "legacy_half_height_m": legacy_rest,
            "sstsr_rest_height_m": sstsr_rest,
            "sstsr_rest": template.variant,
        }
    (artifacts / "staging.json").write_text(json.dumps(report, indent=2) + "\n")
