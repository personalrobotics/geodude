"""Geodude 4.0, steps 1 to 3 of the recycling demo on ssrobot (#199, #201).

A seeded recycling scene is staged from prl_assets: one can on the worktop and the
legacy demo's two bins. The arm on the can's side picks it with ssgrasping's ``grasp``,
from sstsr's side grasps of the can's prl_assets geometry at its privileged pose
channel, and drops it with ``drop`` into the bin on its side, from the legacy demo's
drop zone. Seed 0 puts the can on the right arm's side; seed 3, under ``left``, on the
left arm's, where the right arm's grasp cannot be placed on the worktop.
``geodude-recycling --record`` reports the run in ``recycling.json``, a
``geodude.outcomes.RecyclingRun`` this test decodes strictly; it adds to that report and
keeps it as its evidence:
- the sources it ran from and the seed, and the scene manifest naming the prl_assets
  source and every staged file's hash, with the scene staging to the same bytes twice;
- moved elsewhere with its bundle, an outcome that still verifies without any workspace
  path: the bundle opens, its ``bundle.json`` has the digest the outcome cites, and every
  staged file the outcome names is in the bundle with that hash;
- every grasp and drop stage, with the release confirmed, and nothing attached;
- the can's ``landing``, scoring the drop's ``landed`` after it ends, and where the can
  came to rest, from its pose channel: inside the bin's prl_assets interior;
- the same outcome from ``demo`` and from the command;
- a bundle that verifies, with a world that materializes without a runtime;
- the left arm, for seed 3, chosen by the can's side and dropping into the left bin;
- step 1, under ``worktop``: the same pick, then ``place`` upright on the worktop, with
  the can's tilt from upright, its base's height above the worktop, and its offset from
  the target each within its tolerance.

A second test ends a round while a landing is still observed: the landing is cancelled
and its attempt recorded whole, a complete attempt that decodes
(``landing_outlasted.json``).

Reproduce with ``uv run pytest tests/test_recycling.py``. CI also runs
``geodude-recycling --seed 0 --record`` from the built wheel in a clean environment and
checks its report (``scripts/check_installed.py``); it does not run this test from the
wheel.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from ssrobot import JsonlTrace, RobotContext, dumps, load_installed_package, loads, open_bundle
from ssrobot.mujoco import materialize_world

from geodude import recycling
from geodude.outcomes import Attempt, Destination, RecyclingRun
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
        RobotContext(
            package.description,
            recycling.runtime(package, scene),
            sinks=[sink],
            object_poses=recycling.OBJECT_POSES,
        ) as ctx,
    ):
        result = recycling.demo(ctx, package, to)
    return {
        "scene": _files(work / "scene"),
        "outcome": result,
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    }


def test_geodude_recycles_one_can_on_ssrobot(artifacts: Path, tmp_path: Path) -> None:
    first = _run(tmp_path / "first")
    worktop = _run(tmp_path / "worktop", "worktop")["outcome"]
    left = _run(tmp_path / "left", seed=LEFT_SEED)["outcome"]

    out = tmp_path / "recorded"
    assert recycling.main(["--seed", str(SEED), "--record", str(out)]) == 0
    moved = tmp_path / "elsewhere" / "moved"
    shutil.copytree(out, moved)
    shutil.rmtree(out)
    run = loads((moved / "recycling.json").read_bytes(), RecyclingRun)
    (attempt,) = run.attempts
    bundle = open_bundle(moved / "bundle")
    world = materialize_world(bundle)
    bundled = {f.path: f.sha256 for f in bundle.manifest.files}
    cited = hashlib.sha256((moved / "bundle" / "bundle.json").read_bytes()).hexdigest()

    report: dict[str, Any] = json.loads((moved / "recycling.json").read_text())
    report["staged_files"] = first["scene"]
    report["recorded_matches_repeat"] = attempt == first["outcome"]
    report["left"] = json.loads(dumps(left))
    report["worktop"] = json.loads(dumps(worktop))
    report["bundle"] = {
        "files": [f.path for f in bundle.manifest.files],
        "snapshots": sum(1 for r in bundle.records if r.kind.value == "snapshot"),
        "world_matches": world.world == bundle.manifest.world,
        "cited": run.bundle == cited,
        "staged_files_bundled": all(
            bundled.get(f"scene/{f.path}") == f.sha256 for f in run.scene.files
        ),
    }
    (artifacts / "recycling.json").write_text(json.dumps(report, indent=2) + "\n")

    assert run.seed == SEED
    assert run.scene.seed == SEED and run.scene.prl_assets.commit is not None
    assert {f.path for f in run.scene.files} == set(report["staged_files"]) - {"manifest.json"}
    assert attempt.destination is Destination.BIN and attempt.bin is not None
    assert attempt.bin.name == recycling.Arm("right").bin[0]
    assert attempt.arm == "right"
    assert all(attempt.checks.values()), attempt.checks
    assert attempt.succeeded and attempt.attached == ()
    assert run.cleared and run.objects[0].bin == attempt.bin.name
    assert left.arm == "left" and left.bin is not None
    assert left.bin.name == recycling.Arm("left").bin[0]
    assert all(left.checks.values()), left.checks
    assert left.succeeded and left.attached == ()
    assert worktop.destination is Destination.WORKTOP
    assert all(worktop.checks.values()), worktop.checks
    assert worktop.succeeded and worktop.attached == ()
    assert report["recorded_matches_repeat"]
    assert report["bundle"]["snapshots"] >= 2 and report["bundle"]["world_matches"]
    assert report["bundle"]["cited"] and report["bundle"]["staged_files_bundled"]


def test_a_landing_the_round_outlasts_is_recorded_ended(
    artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A round can end while a landing is still observed: here every landing waits far
    longer than the round's ticks. The landing is cancelled and the attempt recorded
    whole, its ``landed`` check failed: a complete attempt, which decodes as itself."""
    spec = recycling.landing_spec
    monkeypatch.setattr(
        recycling,
        "landing_spec",
        lambda obj, arm: dataclasses.replace(spec(obj, arm), land_ns=2**62),
    )
    package = load_installed_package("geodude_assets")
    scene = stage_scene(
        tmp_path / "scene", package, objects={"can": 1}, fixtures=recycling.BINS, seed=SEED
    )
    with RobotContext(
        package.description,
        recycling.runtime(package, scene),
        object_poses=recycling.OBJECT_POSES,
    ) as ctx:
        (attempt,) = recycling.recycle(ctx, package, ["can_0"])
    text = dumps(attempt)
    (artifacts / "landing_outlasted.json").write_text(json.dumps(json.loads(text), indent=2) + "\n")

    assert loads(text, Attempt) == attempt
    assert attempt.checks["released"] and attempt.checks["detached"]
    assert attempt.checks["landed"] is False and attempt.succeeded is False
    assert attempt.landing is not None and attempt.failure is not None


def test_the_demo_refuses_impossible_can_counts_before_starting() -> None:
    """``--cans`` below 1 would stage nothing and report success; ``--to worktop`` places
    one can only. Both are refused at the command line, before any package, scene, or
    runtime is set up."""
    for argv in (["--cans", "0"], ["--cans", "-2"], ["--cans", "2", "--to", "worktop"]):
        with pytest.raises(SystemExit) as refused:
            recycling.main(argv)
        assert refused.value.code == 2, argv
