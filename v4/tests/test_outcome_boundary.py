"""A recycling outcome is refused, never coerced, when it is not a complete, consistent
``RecyclingRun`` (personalrobotics/geodude#246).

The four-can round's ``recycling.json`` (the session's ``four_cans`` recording) decodes.
Copies of it, each tampered with in one way, must each be refused with the diagnostic
code of the rule it breaks:
- ``missing``: an attempt without ``succeeded``;
- ``extra``: an attempt with a field the format does not have;
- ``wrong_type``: an attempt numbered by a string;
- ``inconsistent``: an attempt whose checks all passed, recorded as failed;
- ``partial_landing``: a dropped can, released and evaluated, without its landing: the
  entry an attempt was while its landing was still observed;
- ``other_version``: the outcome claimed in another version of the format;
- ``old_version``: the outcome claimed in version 1, which has neither ``no_grasp`` nor
  ``place back on the worktop``;
- ``no_grasp_privileged``, ``no_grasp_held``, ``no_grasp_attached``, ``no_grasp_raced``:
  an object the hand has no grasp of, refused on privileged evidence, held, attached, or
  raced;
- ``place_back_unrelated``: an object placed back after a failure other than
  ``no_drop_zone``;
- ``failure_missing``: an attempt by a selected arm whose drop failed, with no failure
  and every evaluated check passed;
- ``failure_missing_evaluated``: an attempt by a selected arm whose grasp failed and
  whose object is not in the bin, with no failure: the evaluation does not explain the
  skill's failed check;
- ``unstaged``: dispositions that leave out an object the scene staged;
- ``other_seed``: a run whose seed is not its scene's;
- ``unselected``: a race whose candidate succeeded, with none selected;
- ``selected_failed``: a race that selected a candidate that did not succeed;
- ``outranked_cost``: an outranked candidate with a cost, even a zero one.

Three copies have the shape of a real outcome and must decode: ``bounced_out``, an
attempt whose every skill succeeded but whose object came to rest outside the bin
(``in_bin`` false), not succeeded and with no failure: the evaluation explains it;
``no_grasp``, the first attempt replaced by one refused at ``grasp.regions`` as
``recycling`` records it; and ``placed_back``, the first attempt's drop replaced by
``no_drop_zone`` and a place-back that did not release, as when the placement fails.

The corpus evaluator refuses to reclassify an evaluation of format 7, or of outcomes of
version 1.

The evidence is ``tampered/``, kept by this test: every tampered copy, and
``diagnostics.json``, each copy's refusal (code, path, message), or null where it
decodes.

Reproduce with ``uv run pytest tests/test_outcome_boundary.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from ssrobot import ValidationError, loads

from geodude.outcomes import NO_DROP_ZONE, NO_GRASP, RecoveryAction, RecyclingRun
from tests.test_sort_all import _evaluator


def _missing(run: dict[str, Any]) -> None:
    del run["attempts"][0]["succeeded"]


def _extra(run: dict[str, Any]) -> None:
    run["attempts"][0]["note"] = "added by hand"


def _wrong_type(run: dict[str, Any]) -> None:
    run["attempts"][0]["number"] = "1"


def _inconsistent(run: dict[str, Any]) -> None:
    run["attempts"][0]["succeeded"] = False


def _partial_landing(run: dict[str, Any]) -> None:
    attempt = run["attempts"][0]
    attempt["landing"] = None
    attempt["times"]["landed"] = None


def _other_version(run: dict[str, Any]) -> None:
    run["version"] = RecyclingRun.VERSION + 1


def _old_version(run: dict[str, Any]) -> None:
    run["version"] = 1


def _no_grasp(run: dict[str, Any]) -> None:
    attempt = run["attempts"][0]
    for key in ("arm", "race", "lookahead", "grasp", "drop", "landing", "place", "times"):
        attempt[key] = None
    attempt.update(
        destination="bin",
        failure={
            "skill": "grasp",
            "stage": "regions",
            "reason": NO_GRASP,
            "cause": None,
            "detail": "no pregrasp region within the hand's limits",
            "evidence": "planned",
            "held": False,
            "attached": False,
        },
        recovery=None,
        transitions=[],
        attached=None,
        final_pose=None,
        bin=None,
        worktop=None,
        succeeded=False,
    )


def _forged_no_grasp(**change: Any) -> Callable[[dict[str, Any]], None]:
    def forge(run: dict[str, Any]) -> None:
        raced = run["attempts"][0]["race"]
        _no_grasp(run)
        failure = run["attempts"][0]["failure"]
        for key, value in change.items():
            if key == "race":
                run["attempts"][0]["race"] = raced
            else:
                failure[key] = value

    return forge


def _placed_back(run: dict[str, Any]) -> None:
    attempt = run["attempts"][0]
    attempt.update(drop=None, landing=None, succeeded=False)
    attempt["times"].update(released=None, landed=None)
    checks = attempt["bin"]["checks"]
    checks.update(dropped=False, released=False, landed=False, in_bin=False)
    attempt["failure"] = {
        "skill": "drop",
        "stage": "zone",
        "reason": NO_DROP_ZONE,
        "cause": None,
        "detail": "no drop zone fits the object as it is held",
        "evidence": "planned",
        "held": True,
        "attached": True,
    }
    attempt["recovery"] = {
        "action": RecoveryAction.PLACE_BACK.value,
        "released": False,
        "detail": "no drop zone fits it as held; placed back: failed",
    }


def _place_back_unrelated(run: dict[str, Any]) -> None:
    _placed_back(run)
    run["attempts"][0]["failure"]["reason"] = "transition_refuted"


def _failure_missing(run: dict[str, Any]) -> None:
    attempt = run["attempts"][0]
    attempt["bin"]["checks"]["dropped"] = False
    attempt["succeeded"] = False


def _failure_missing_evaluated(run: dict[str, Any]) -> None:
    attempt = run["attempts"][0]
    attempt["bin"]["checks"]["grasped"] = False
    attempt["bin"]["checks"]["in_bin"] = False
    attempt["succeeded"] = False


def _bounced_out(run: dict[str, Any]) -> None:
    attempt = run["attempts"][0]
    attempt["bin"]["checks"]["in_bin"] = False
    attempt["succeeded"] = False


def _unstaged(run: dict[str, Any]) -> None:
    del run["objects"][-1]


def _other_seed(run: dict[str, Any]) -> None:
    run["seed"] += 1


def _unselected(run: dict[str, Any]) -> None:
    run["attempts"][0]["race"]["selected"] = None


def _selected_failed(run: dict[str, Any]) -> None:
    run["attempts"][0]["race"]["selected"] = "nobody"


def _outranked_cost(run: dict[str, Any]) -> None:
    outranked = next(
        c for a in run["attempts"] for c in a["race"]["candidates"] if c["outcome"] == "outranked"
    )
    outranked["cost_s"] = 0.0


TAMPERED: dict[str, tuple[Callable[[dict[str, Any]], None], str | None]] = {
    "missing": (_missing, "missing_field"),
    "extra": (_extra, "unknown_field"),
    "wrong_type": (_wrong_type, "wrong_type"),
    "inconsistent": (_inconsistent, "inconsistent_success"),
    "partial_landing": (_partial_landing, "incomplete_landing"),
    "other_version": (_other_version, "version_mismatch"),
    "old_version": (_old_version, "version_mismatch"),
    "no_grasp": (_no_grasp, None),
    "no_grasp_privileged": (_forged_no_grasp(evidence="privileged"), "inconsistent_failure"),
    "no_grasp_held": (_forged_no_grasp(held=True), "inconsistent_failure"),
    "no_grasp_attached": (_forged_no_grasp(attached=True), "inconsistent_failure"),
    "no_grasp_raced": (_forged_no_grasp(race=True), "inconsistent_failure"),
    "placed_back": (_placed_back, None),
    "place_back_unrelated": (_place_back_unrelated, "inconsistent_recovery"),
    "failure_missing": (_failure_missing, "missing_failure"),
    "failure_missing_evaluated": (_failure_missing_evaluated, "missing_failure"),
    "unstaged": (_unstaged, "staged_mismatch"),
    "other_seed": (_other_seed, "seed_mismatch"),
    "unselected": (_unselected, "unselected_success"),
    "selected_failed": (_selected_failed, "selected_not_succeeded"),
    "outranked_cost": (_outranked_cost, "outranked_with_plans"),
    "bounced_out": (_bounced_out, None),
}
"""Each tampering, and the code it must be refused with: None where it must decode."""


def test_tampered_outcomes_are_refused_each_with_its_own_code(
    artifacts: Path, four_cans: tuple[int, Path]
) -> None:
    _, out = four_cans
    original = (out / "recycling.json").read_bytes()
    run = loads(original, RecyclingRun)
    landed = run.attempts[0]
    assert landed.landing is not None and landed.checks["released"] and landed.succeeded

    copies = artifacts / "tampered"
    copies.mkdir(exist_ok=True)
    diagnostics: dict[str, Any] = {}
    for name, (tamper, _) in TAMPERED.items():
        data = json.loads(original)
        tamper(data)
        text = json.dumps(data, indent=1) + "\n"
        (copies / f"{name}.json").write_text(text)
        try:
            loads(text, RecyclingRun)
        except ValidationError as refused:
            diagnostics[name] = {
                "code": refused.code,
                "path": refused.path,
                "message": refused.message,
            }
        else:
            diagnostics[name] = None
    (copies / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    codes = {name: (d or {}).get("code") for name, d in diagnostics.items()}
    assert codes == {name: code for name, (_, code) in TAMPERED.items()}, diagnostics


def test_the_evaluator_refuses_an_older_evaluation(tmp_path: Path) -> None:
    evaluator = _evaluator()
    stale = {
        "is format 7": {"format": 7, "outcome_version": RecyclingRun.VERSION},
        "of version 1": {"format": evaluator.FORMAT, "outcome_version": 1},
    }
    for refusal, header in stale.items():
        source = tmp_path / "evaluation.json"
        source.write_text(json.dumps({**header, "rounds": []}))
        with pytest.raises(SystemExit, match=refusal):
            evaluator.reclassify(source)
