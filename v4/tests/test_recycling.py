"""Geodude 4.0, step 1 of the recycling demo on ssrobot (#199, #201).

A seeded recycling scene is staged from prl_assets: one can on the worktop and the
legacy demo's two bins. Geodude's right arm picks the can with ssgrasping's ``grasp``,
from sstsr's side grasps of the can's prl_assets geometry at its privileged pose
channel, and places it upright on the worktop with ``place``. ``geodude-recycling
--record`` reports the run in ``recycling.json``; this test adds to that report and
keeps it as its evidence:
- the sources it ran from and the seed, and the scene manifest naming the prl_assets
  source and every staged file's hash, with the scene staging to the same bytes twice;
- every grasp and placement stage, with the release confirmed, and nothing attached;
- where the can came to rest, from its pose channel after the run, with its tilt from
  upright, its base's height above the worktop, and its offset from the target, each
  within its tolerance;
- the same outcome and trace from two runs;
- a bundle that verifies, with a world that materializes without a runtime.

Reproduce with ``uv run pytest tests/test_recycling.py``. CI also runs the same command
from the built wheel in a clean environment (``scripts/check_installed.py``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ssrobot import JsonlTrace, RobotContext, load_installed_package, open_bundle
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
        result = recycling.demo(ctx, package)
    return {
        "scene": _files(work / "scene"),
        "outcome": result,
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    }


def test_geodude_recycles_one_can_on_ssrobot(artifacts: Path, tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first"), _run(tmp_path / "second")

    out = tmp_path / "recorded"
    assert recycling.main(["--seed", str(SEED), "--record", str(out)]) == 0
    report = json.loads((out / "recycling.json").read_text())
    bundle = open_bundle(out / "bundle")
    world = materialize_world(bundle)

    report["staged_files"] = first["scene"]
    report["repeat_identical"] = first == second
    report["recorded_matches_repeat"] = {k: report[k] for k in first["outcome"]} == first["outcome"]
    report["bundle"] = {
        "files": [f.path for f in bundle.manifest.files],
        "snapshots": sum(1 for r in bundle.records if r.kind.value == "snapshot"),
        "world_matches": world.world == bundle.manifest.world,
    }
    (artifacts / "recycling.json").write_text(json.dumps(report, indent=2) + "\n")

    assert report["seed"] == SEED
    assert report["sources"]["geodude"]["version"].startswith("4.")
    scene = report["scene"]
    assert scene["seed"] == SEED and scene["prl_assets"]["source"] is not None
    assert set(scene["files"]) == set(report["staged_files"]) - {"manifest.json"}
    assert [o["name"] for o in scene["objects"]] == ["can_0"]
    assert [f["name"] for f in scene["fixtures"]] == ["recycle_bin_0", "recycle_bin_1"]
    assert report["checks"] == dict.fromkeys(report["checks"], True), report["checks"]
    assert report["succeeded"] and report["attached"] == []
    assert report["repeat_identical"] and report["recorded_matches_repeat"]
    assert report["bundle"]["snapshots"] >= 2 and report["bundle"]["world_matches"]
