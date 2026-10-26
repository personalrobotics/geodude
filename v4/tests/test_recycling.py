"""Geodude 4.0, steps 1 to 3 of the recycling demo on ssrobot (#199, #201).

A seeded recycling scene is staged from prl_assets: one can on the worktop and the
legacy demo's two bins. The arm on the can's side picks it with ssgrasping's ``grasp``,
from sstsr's side grasps of the can's prl_assets geometry at its privileged pose
channel, and drops it with ``drop`` into the bin on its side, from the legacy demo's
drop zone. Seed 0 puts the can on the right arm's side; seed 3, under ``left``, on the
left arm's, where the right arm's grasp cannot be placed on the worktop.
``geodude-recycling --record`` reports the run in ``recycling.json``; this test adds to
that report and keeps it as its evidence:
- the sources it ran from and the seed, and the scene manifest naming the prl_assets
  source and every staged file's hash, with the scene staging to the same bytes twice;
- every grasp and drop stage, with the release confirmed, and nothing attached;
- where the can came to rest, from its pose channel after the run: inside the bin's
  prl_assets interior;
- the same outcome and trace from two runs;
- a bundle that verifies, with a world that materializes without a runtime;
- the left arm, for seed 3, chosen by the can's side and dropping into the left bin;
- step 1, under ``worktop``: the same pick, then ``place`` upright on the worktop, with
  the can's tilt from upright, its base's height above the worktop, and its offset from
  the target each within its tolerance.

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
LEFT_SEED = 3


def _files(root: Path) -> dict[str, str]:
    return {
        f.relative_to(root).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(root.rglob("*"))
        if f.is_file()
    }


def _run(work: Path, to: str = "bin", seed: int = SEED) -> dict[str, Any]:
    package = load_installed_package("geodude_assets")
    scene = stage_scene(
        work / "scene", package, objects={"can": 1}, fixtures=recycling.BINS, seed=seed
    )
    trace = work / "trace.jsonl"
    with (
        JsonlTrace(trace) as sink,
        RobotContext(package.description, recycling.runtime(package, scene), sinks=[sink]) as ctx,
    ):
        result = recycling.demo(ctx, package, to)
    return {
        "scene": _files(work / "scene"),
        "outcome": result,
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    }


def test_geodude_recycles_one_can_on_ssrobot(artifacts: Path, tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first"), _run(tmp_path / "second")
    worktop = _run(tmp_path / "worktop", "worktop")["outcome"]
    left = _run(tmp_path / "left", seed=LEFT_SEED)["outcome"]

    out = tmp_path / "recorded"
    assert recycling.main(["--seed", str(SEED), "--record", str(out)]) == 0
    report = json.loads((out / "recycling.json").read_text())
    bundle = open_bundle(out / "bundle")
    world = materialize_world(bundle)

    report["staged_files"] = first["scene"]
    report["repeat_identical"] = first == second
    report["recorded_matches_repeat"] = {k: report[k] for k in first["outcome"]} == first["outcome"]
    report["left"] = left
    report["worktop"] = worktop
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
    assert report["to"] == "bin" and report["bin"]["name"] == "recycle_bin_0"
    assert report["arm"] == "right"
    assert report["checks"] == dict.fromkeys(report["checks"], True), report["checks"]
    assert report["succeeded"] and report["attached"] == []
    assert [n for n, _ in report["drop"]] == [
        "preflight", "transfer", "release", "dropped",
    ]  # fmt: skip
    assert left["arm"] == "left"
    assert left["bin"]["name"] == "recycle_bin_1"
    assert left["checks"] == dict.fromkeys(left["checks"], True), left["checks"]
    assert left["succeeded"] and left["attached"] == []
    assert worktop["checks"] == dict.fromkeys(worktop["checks"], True), worktop["checks"]
    assert worktop["succeeded"] and worktop["attached"] == []
    assert report["repeat_identical"] and report["recorded_matches_repeat"]
    assert report["bundle"]["snapshots"] >= 2 and report["bundle"]["world_matches"]


def test_the_demo_refuses_impossible_can_counts_before_starting() -> None:
    """``--cans`` below 1 would stage nothing and report success; ``--to worktop`` places
    one can only. Both are refused at the command line, before any package, scene, or
    runtime is set up."""
    import pytest

    for argv in (["--cans", "0"], ["--cans", "-2"], ["--cans", "2", "--to", "worktop"]):
        with pytest.raises(SystemExit) as refused:
            recycling.main(argv)
        assert refused.value.code == 2, argv
