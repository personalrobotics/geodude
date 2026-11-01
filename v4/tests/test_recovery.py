"""Geodude chooses how to let go of a still-attached object by what failed, never by
prose (personalrobotics/ssgrasping#34).

``recover`` opens the fingers only where that is known to be safe: in place after a lift
that did not plan, where the object rests on its support, or after a carry whose
gripper opening showed the object leave the hand (``transition_refuted``, cause
``retained``, personalrobotics/ssgrasping#28), where nothing is left in it.
``safe_release`` decides that from the failed stage, its reason and cause, and whether
the object is still declared attached, as ``bin_outcome`` records them (``_failure``)
from ssgrasping's own results. Each case here is a failure ssgrasping's end-to-end
evidence produces (its ``obstructed`` lift and ``pulled`` transfer, and a lift the
object leaves the same way): the decision holds whatever the stage's detail says, and is
withheld when the reason or the attachment differs.

Reproduce with ``uv run pytest tests/test_recovery.py``.
"""

from __future__ import annotations

from ssgrasping import Evidence, Reason, SkillResult, SkillState, Stage

from geodude.recycling import RELEASE_IN_PLACE, RELEASE_LOST, _failure, safe_release

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
        ("drop", "transfer", Reason.TRANSITION_REFUTED, "retained", Evidence.PORTABLE),
        "the opening fell below 0.224: the object left the hand",
        RELEASE_LOST,
    ),
    (
        ("grasp", "lift", Reason.TRANSITION_REFUTED, "retained", Evidence.PORTABLE),
        "the opening fell below 0.224: the object left the hand",
        RELEASE_LOST,
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
