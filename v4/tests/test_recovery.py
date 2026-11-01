"""Geodude chooses where to release a still-held object by what failed, never by prose
(personalrobotics/ssgrasping#34).

``recover`` opens the fingers only where the object has known support: in place after a
lift that did not plan, or over the bin after the attachment monitor saw it leave the
carry. ``safe_release`` decides that from the failed stage, its reason, and whether the
object is still declared attached, as ``bin_outcome`` records them (``_failure``) from
ssgrasping's own results. Each case here is a failure ssgrasping's end-to-end evidence
produces (its ``obstructed`` lift and ``slipped`` release): the decision holds whatever
the stage's detail says, and is withheld when the reason or the attachment differs.

Reproduce with ``uv run pytest tests/test_recovery.py``.
"""

from __future__ import annotations

from ssgrasping import Evidence, Reason, SkillResult, SkillState, Stage

from geodude.recycling import RELEASE_IN_PLACE, RELEASE_OVER_BIN, _failure, safe_release

DETAILS = (
    None,  # as ssgrasping words it today
    "",
    "reworded: the planner said something else entirely",
)


def _failed(
    skill: str,
    stage: str,
    reason: Reason,
    cause: str | None,
    detail: str,
    evidence: Evidence,
    attached: bool = True,
) -> SkillResult:
    ended = Stage(
        name=stage,
        succeeded=False,
        detail=detail,
        time_ns=0,
        revision=0,
        held=True,
        attached=attached,
        evidence=evidence,
        reason=reason,
        cause=cause,
    )
    return SkillResult(skill=skill, state=SkillState.FAILED, stages=(ended,), attachment=None)


CASES = (
    (
        ("grasp", "lift", Reason.NOT_PLANNED, "infeasible", Evidence.PLANNED),
        "planning infeasible: All 1 start configuration(s) in collision: Start[0]: in collision",
        RELEASE_IN_PLACE,
    ),
    (
        ("drop", "release", Reason.ATTACHMENT_LOST, None, Evidence.PRIVILEGED),
        "attachment monitor: the object left its declared transform during the transfer",
        RELEASE_OVER_BIN,
    ),
)


def test_release_is_chosen_by_stage_reason_and_attachment_not_detail() -> None:
    for (skill, stage, reason, cause, evidence), today, action in CASES:
        for detail in DETAILS:
            result = _failed(
                skill, stage, reason, cause, today if detail is None else detail, evidence
            )
            assert safe_release(_failure(result)) == action, (skill, stage, detail)
        # The same stage failing otherwise, or with nothing still attached, has no known
        # support to release onto.
        other = Reason.COMMAND_FAILED if reason is Reason.NOT_PLANNED else Reason.REFUSED
        assert safe_release(_failure(_failed(skill, stage, other, None, today, evidence))) is None
        detached = _failed(skill, stage, reason, cause, today, evidence, attached=False)
        assert safe_release(_failure(detached)) is None
    assert safe_release(None) is None
