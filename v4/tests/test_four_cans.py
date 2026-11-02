"""Geodude 4.0 recycles four cans, racing its arms for each and planning each can while
the previous one is carried to its bin (personalrobotics/geodude#223).

``geodude-recycling --seed 4 --cans 4 --record`` (the session's ``four_cans`` round,
which ``test_planning_latency`` also reads) stages four cans and recycles them all
with ``recycling.recycle``. For each can, both arms race (ssgrasping's ``Candidates``),
each through the drop into its own bin; the arm on the can's side is preferred. The race
for every can after the first is planned on the prediction of the previous drop's
transfer, which expects that can released onto its bin's floor, and the selected
pregrasp continues from where the transfer ends, after the release. Drops do not retreat.
The evidence is ``recycling.json``, a ``geodude.outcomes.RecyclingRun`` decoded strictly,
kept by this test with:
- every can's attempt, its arm, and its race: each candidate's outcome and plans, and
  the selection, which is the preferred arm's when its prefix plans;
- each skill's lookahead log: what was planned ahead, the prediction's lineage, the
  continuation decision, and where each skill stopped looking ahead;
- from the run bundle's trace, for each can after the first: the ``predicted`` record
  of the transfer, made while the transfer executed, and the ``continued`` decision after
  it ended.
Which lower-priority candidate was stopped or finished depends on how fast the preferred
one planned, so the outcome records it ``outranked``, without its plans; the selection,
and so the run, does not depend on it.

Reproduce with ``uv run pytest tests/test_four_cans.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ssrobot import ExecutionState, TraceKind, loads, open_bundle

from geodude import recycling
from geodude.outcomes import CandidateOutcome, RaceBasis, RecyclingRun

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


def test_geodude_recycles_four_cans_racing_its_arms(
    artifacts: Path, four_cans: tuple[int, Path]
) -> None:
    code, out = four_cans
    run = loads((out / "recycling.json").read_bytes(), RecyclingRun)
    records = open_bundle(out / "bundle").records
    cans = run.attempts
    races = [can.race for can in cans if can.race is not None]
    timelines = [
        _timeline(records, race.during)
        for race in races
        if race.on is RaceBasis.PREDICTION and race.during is not None
    ]
    report = json.loads((out / "recycling.json").read_text())
    (artifacts / "recycling.json").write_text(
        json.dumps({**report, "timelines": timelines}, indent=2) + "\n"
    )

    assert code == 0 and all(c.succeeded for c in cans), [(c.object, c.checks) for c in cans]
    assert len(cans) == CANS and len(races) == CANS
    for can, race in zip(cans, races, strict=True):
        assert {c.label for c in race.candidates} == {"right", "left"}
        assert race.selected == can.arm
        preferred = next(c for c in race.candidates if c.priority == recycling.PREFERRED)
        if preferred.outcome is CandidateOutcome.SUCCEEDED:
            assert can.arm == preferred.label
        assert can.lookahead is not None and can.drop is not None
        approach = next(e for e in can.lookahead.grasp if e.stage == "approach")
        assert approach.prediction is not None
        assert approach.prediction.lineage == (approach.prediction.base,)
        # The race's approach check is kept, not planned again, even from a pregrasp that
        # ran as a continuation, rebased onto the actual arrival.
        assert approach.kept is not None, approach
        assert "retreat" not in [s.name for s in can.drop]

    # The first can is raced on a snapshot; every later one during the transfer before it.
    assert races[0].on is RaceBasis.SNAPSHOT
    assert len(timelines) == CANS - 1
    for can, line in zip(cans[1:], timelines, strict=True):
        assert len(line["predicted_ns"]) == 1
        assert line["submitted_ns"] <= line["predicted_ns"][0] < line["ended_ns"]
        assert line["decisions"]
        assert all(d["time_ns"] >= line["ended_ns"] for d in line["decisions"])
        assert can.lookahead is not None
        pregrasp = next(e for e in can.lookahead.grasp if e.stage == "pregrasp")
        assert pregrasp.during == line["transfer"]
        assert (pregrasp.decision == "continued") == line["decisions"][-1]["accepted"]
