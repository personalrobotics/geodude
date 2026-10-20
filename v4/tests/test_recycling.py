"""Geodude 4.0, step 1 of the recycling demo on ssrobot (#199, #201).

A seeded recycling scene is staged from prl_assets: one can on the worktop and the
legacy demo's two bins. Geodude's right arm picks the can with ssgrasping's ``grasp``,
from sstsr's side grasps of the can's prl_assets geometry at its privileged pose
channel, and places it upright on the worktop with ``place``. The run is recorded and
written as a run bundle. The evidence is ``recycling.json``:
- the scene stages to the same bytes twice, and its manifest names the prl_assets
  source and every staged file's hash;
- every grasp and placement stage succeeds, nothing is left attached, and the
  attachment monitor reports no slip;
- the run gives the same trace twice;
- its bundle verifies, and its world materializes without a runtime.

Reproduce with ``uv run pytest tests/test_recycling.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ssrobot import JsonlTrace, RobotContext, load_installed_package, open_bundle, read_trace
from ssrobot.mujoco import materialize_world

from geodude import recycling
from geodude.scene import stage_scene

SEED = 0


def _files(root: Path) -> dict[str, str]:
    return {
        f.relative_to(root).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(root.rglob("*"))
        if f.is_file()
    }


def _run(work: Path) -> dict[str, Any]:
    package = load_installed_package("geodude_assets")
    scene = stage_scene(
        work / "scene", package, objects={"can": 1}, fixtures=recycling.BINS, seed=SEED
    )
    trace = work / "trace.jsonl"
    with (
        JsonlTrace(trace) as sink,
        RobotContext(package.description, recycling.runtime(package, scene), sinks=[sink]) as ctx,
    ):
        ctx.snapshot()
        picked, placed = recycling.pick_and_place(
            ctx, package, "can_0", (0.20, -0.50), arm=recycling.Arm("right")
        )
        ctx.snapshot()
        attached = [a.object for a in ctx.scene.attachments]
    violations = [r.time_ns for r in read_trace(trace) if r.kind.value == "violation"]
    return {
        "scene": _files(work / "scene"),
        "manifest": json.loads((work / "scene" / "manifest.json").read_text()),
        "grasp": [(s.name, s.succeeded) for s in picked.stages],
        "place": None if placed is None else [(s.name, s.succeeded) for s in placed.stages],
        "placed_pose": None if placed is None else placed.values.get("placed_pose"),
        "attached_at_end": attached,
        "violations": violations,
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    }


def test_geodude_recycles_one_can_on_ssrobot(artifacts: Path, tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first"), _run(tmp_path / "second")

    out = tmp_path / "recorded"
    assert recycling.main(["--seed", str(SEED), "--record", str(out)]) == 0
    bundle = open_bundle(out / "bundle")
    world = materialize_world(bundle)

    report = {
        **first,
        "repeat_identical": first == second,
        "bundle": {
            "files": [f.path for f in bundle.manifest.files],
            "snapshots": sum(1 for r in bundle.records if r.kind.value == "snapshot"),
            "world_matches": world.world == bundle.manifest.world,
        },
    }
    text = json.dumps(report, indent=2) + "\n"
    (artifacts / "recycling.json").write_text(text)

    assert report["repeat_identical"], "two runs differ"
    manifest = report["manifest"]
    assert manifest["seed"] == SEED and manifest["prl_assets"]["source"] is not None
    assert set(manifest["files"]) == set(report["scene"]) - {"manifest.json"}
    assert [o["name"] for o in manifest["objects"]] == ["can_0"]
    assert [f["name"] for f in manifest["fixtures"]] == ["recycle_bin_0", "recycle_bin_1"]
    assert all(ok for _, ok in report["grasp"]), report["grasp"]
    assert report["place"] is not None and all(ok for _, ok in report["place"]), report["place"]
    assert report["attached_at_end"] == [] and report["violations"] == []
    assert report["bundle"]["snapshots"] >= 2 and report["bundle"]["world_matches"]
