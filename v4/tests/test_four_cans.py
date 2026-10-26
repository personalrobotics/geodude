"""Geodude 4.0 recycles four cans, racing its arms for each and planning each can while
the previous one is carried to its bin (personalrobotics/geodude#223).

``geodude-recycling --seed 4 --cans 4 --record`` stages four cans and recycles them all
with ``recycling.recycle``. For each can, both arms race (ssgrasping's ``Candidates``),
each through the drop into its own bin; the arm on the can's side is preferred. The race
for every can after the first is planned on the prediction of the previous drop's
transfer, which expects that can released onto its bin's floor, and the selected
pregrasp continues from where the transfer ends, after the release. Drops do not retreat.
The evidence is ``recycling.json``, kept by this test with:
- every can's outcome, its arm, and its race: each candidate's outcome and plans, and
  the selection, which is the preferred arm's when its prefix plans;
- each skill's lookahead log: what was planned ahead, the prediction's lineage, the
  continuation decision, and where each skill stopped looking ahead;
- from the run bundle's trace, for each can after the first: the ``predicted`` record
  of the transfer, made while the transfer executed, and the ``continued`` decision after
  it ended.
Which lower-priority candidate was stopped (``canceled_by_priority``) or finished depends
on how fast the preferred one planned; the selection, and so the run, does not.

Reproduce with ``uv run pytest tests/test_four_cans.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ssrobot import ExecutionState, TraceKind, open_bundle

from geodude import recycling

SEED = 4
CANS = 4


def _timeline(records: Any, transfer: str) -> dict[str, Any]:
    """When ``transfer`` was submitted and ended, when it was predicted, and the
    continuation decisions on that prediction, in runtime time."""
    submitted = next(
        r.time_ns
        for r in records
        if r.kind is TraceKind.SUBMITTED and r.payload.execution == transfer
    )
    ended = next(
        r.time_ns
        for r in records
        if r.kind is TraceKind.STATUS
        and r.payload.execution == transfer
        and r.payload.state is ExecutionState.SUCCEEDED
    )
    predicted = [
        r.time_ns for r in records if r.kind is TraceKind.PREDICTED and r.payload.source == transfer
    ]
    decisions = [
        {"time_ns": r.time_ns, "accepted": r.payload.accepted, "refusal": r.payload.refusal}
        for r in records
        if r.kind is TraceKind.CONTINUED and r.payload.source == transfer
    ]
    return {
        "transfer": transfer,
        "submitted_ns": submitted,
        "ended_ns": ended,
        "predicted_ns": predicted,
        "decisions": decisions,
    }


def test_geodude_recycles_four_cans_racing_its_arms(artifacts: Path, tmp_path: Path) -> None:
    out = tmp_path / "recorded"
    code = recycling.main(["--seed", str(SEED), "--cans", str(CANS), "--record", str(out)])
    report = json.loads((out / "recycling.json").read_text())
    records = open_bundle(out / "bundle").records
    cans = report["cans"]
    timelines = [
        _timeline(records, can["race"]["basis"]["during"])
        for can in cans
        if can["race"]["basis"]["on"] == "prediction"
    ]
    (artifacts / "recycling.json").write_text(
        json.dumps({**report, "timelines": timelines}, indent=2) + "\n"
    )

    assert code == 0 and report["succeeded"], [(c["object"], c["checks"]) for c in cans]
    assert len(cans) == CANS
    for can in cans:
        race = can["race"]
        assert {c["label"] for c in race["candidates"]} == {"right", "left"}
        assert race["selected"] == can["arm"]
        preferred = next(c for c in race["candidates"] if c["priority"] == recycling.PREFERRED)
        if preferred["outcome"] == "succeeded":
            assert can["arm"] == preferred["label"]
        approach = next(e for e in can["lookahead"]["grasp"] if e["stage"] == "approach")
        assert approach["prediction"]["lineage"] == [approach["prediction"]["base"]]
        # The race's approach check is kept, not planned again, even from a pregrasp that
        # ran as a continuation, rebased onto the actual arrival.
        assert approach.get("kept") == "the feasibility check, planned before", approach
        assert [e["stage"] for e in can["lookahead"]["drop"]] == ["release"]

    # The first can is raced on a snapshot; every later one during the transfer before it.
    assert cans[0]["race"]["basis"]["on"] == "snapshot"
    assert len(timelines) == CANS - 1
    for can, line in zip(cans[1:], timelines, strict=True):
        assert len(line["predicted_ns"]) == 1
        assert line["submitted_ns"] <= line["predicted_ns"][0] < line["ended_ns"]
        assert line["decisions"]
        assert all(d["time_ns"] >= line["ended_ns"] for d in line["decisions"])
        pregrasp = next(e for e in can["lookahead"]["grasp"] if e["stage"] == "pregrasp")
        assert pregrasp["during"] == line["transfer"]
        assert (pregrasp["decision"] == "continued") == line["decisions"][-1]["accepted"]
