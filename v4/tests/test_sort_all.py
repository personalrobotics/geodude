"""Geodude 4.0 clears the table: the legacy demo's ``sort_all`` over five seeded objects
(#201, step 5).

``geodude-recycling --mix 5 --seed 0 --record out`` stages five objects drawn from the
legacy demo's nine types with seed 0, the demo's default. It sorts them all into the
bins (``recycling.sort_all``), and records the run as a bundle and ``recycling.json``.
The test requires:
- every object comes to rest in a bin, with nothing left attached;
- the bundle opens and verifies (``open_bundle``);
- the corpus evaluator (``scripts/evaluate_recycling.py``) judges the recorded round as
  it judges the development corpus's seed 0, the same round: every object's verdict
  (``verdict``, ``classify``) is a ``recycled`` one, and its summary counts the round
  cleared.

The evidence is ``sort_all.json``: the outcome the seed decides (per attempt: the
object, arm, every stage, and the checks; per object: its bin and final pose), with the
evaluator's verdicts and summary; and the run's ``recycling.json``. The race reports are
left out of the outcome: which lower-priority candidate was stopped depends on how fast
the preferred one planned, and the selection does not. Determinism is
``test_planning_latency``'s gate.

Reproduce with ``uv run pytest tests/test_sort_all.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from ssrobot import open_bundle

from geodude import recycling
from tests.conftest import ROOT

SEED = 0
OBJECTS = 5


def _outcome(report: dict[str, Any]) -> dict[str, Any]:
    """The part of a ``sort_all`` report that the seed alone decides."""
    return {
        "attempts": [
            {
                key: attempt.get(key)
                for key in ("object", "attempt", "arm", "grasp", "drop", "landing", "checks")
            }
            for attempt in report["attempts"]
        ],
        "objects": {
            name: {"bin": o["bin"], "attached": o["attached"], "final_pose": o["final_pose"]}
            for name, o in report["objects"].items()
        },
    }


def _evaluator() -> ModuleType:
    """``scripts/evaluate_recycling.py``, imported."""
    spec = importlib.util.spec_from_file_location(
        "evaluate_recycling", ROOT / "scripts" / "evaluate_recycling.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evaluated(report: dict[str, Any], records: Any) -> dict[str, Any]:
    """The recorded round, as the evaluator's ``run_round`` reports one, judged by it."""
    evaluator = _evaluator()
    idle = evaluator.Idle()
    for record in records:
        idle(record)
    round_ = {
        "seed": SEED,
        "staged": report["scene"]["objects"],
        "outcomes": report["attempts"],
        "final": report["objects"],
        "timing": evaluator._timing(report["attempts"], idle.intervals, records[-1].time_ns),
    }
    round_["objects"] = evaluator._objects(round_)
    evaluation = evaluator._valid({"rounds": [round_]})
    return {
        "valid": evaluation["valid"],
        "verdicts": {o["object"]: o["verdict"] for o in round_["objects"]},
        "summary": evaluation.get("summary"),
    }


def test_geodude_sorts_five_seeded_objects_into_the_bins(artifacts: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    code = recycling.main(["--seed", str(SEED), "--mix", str(OBJECTS), "--record", str(out)])
    report = json.loads((out / "recycling.json").read_text())
    evaluated = _evaluated(report, open_bundle(out / "bundle").records)
    outcome = _outcome(report)
    (artifacts / "sort_all.json").write_text(
        json.dumps({**outcome, "evaluated": evaluated}, indent=2) + "\n"
    )
    (artifacts / "recycling.json").write_text(json.dumps(report, indent=2) + "\n")

    assert code == 0 and report["cleared"], {o: f["bin"] for o, f in report["objects"].items()}
    assert len(report["objects"]) == OBJECTS
    assert not any(o["attached"] for o in report["objects"].values())
    assert evaluated["valid"], evaluated
    assert all(v.startswith("recycled") for v in evaluated["verdicts"].values()), evaluated
    summary = evaluated["summary"]
    assert (summary["objects_staged"], summary["objects_recycled"]) == (OBJECTS, OBJECTS)
    assert (summary["rounds"], summary["rounds_cleared"]) == (1, 1)
