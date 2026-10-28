"""Geodude 4.0 clears the table: the legacy demo's ``sort_all`` over five seeded objects
(#201, step 5).

``geodude-recycling --mix 5 --seed 0 --record out`` stages five objects drawn from the
legacy demo's nine types with seed 0, the demo's default. It sorts them all into the
bins (``recycling.sort_all``), and records the run as a bundle and ``recycling.json``.
The test runs it twice and requires:
- every object comes to rest in a bin, with nothing left attached;
- both bundles open and verify (``open_bundle``), and name the same world, scene, and
  model;
- the run's deterministic outcome is byte-identical across the two runs. That is, per
  attempt: the object, arm, every stage, and the checks; per object: its bin and final
  pose.

The race reports are left out of that comparison: which lower-priority candidate was
stopped depends on how fast the preferred one planned, and the selection, which is
compared, does not.

The evidence is ``sort_all.json``, the deterministic outcome, and the first run's
``recycling.json``.

Reproduce with ``uv run pytest tests/test_sort_all.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ssrobot import open_bundle

from geodude import recycling

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


def test_geodude_sorts_five_seeded_objects_into_the_bins(artifacts: Path, tmp_path: Path) -> None:
    runs = []
    for i in range(2):
        out = tmp_path / f"run{i}"
        code = recycling.main(["--seed", str(SEED), "--mix", str(OBJECTS), "--record", str(out)])
        report = json.loads((out / "recycling.json").read_text())
        manifest = open_bundle(out / "bundle").manifest
        runs.append((code, report, manifest))
    outcome = _outcome(runs[0][1])
    (artifacts / "sort_all.json").write_text(json.dumps(outcome, indent=2) + "\n")
    (artifacts / "recycling.json").write_text(json.dumps(runs[0][1], indent=2) + "\n")

    code, report, manifest = runs[0]
    assert code == 0 and report["cleared"], {o: f["bin"] for o, f in report["objects"].items()}
    assert len(report["objects"]) == OBJECTS
    assert not any(o["attached"] for o in report["objects"].values())
    _, again, manifest_again = runs[1]
    for field in ("world", "scene_identity", "model_signature"):
        assert getattr(manifest, field) == getattr(manifest_again, field), field
    first = json.dumps(outcome, sort_keys=True)
    assert first == json.dumps(_outcome(again), sort_keys=True)
