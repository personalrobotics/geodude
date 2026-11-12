"""Geodude chooses how to let go of a still-attached object by what failed, never by
prose (personalrobotics/ssgrasping#34).

``recover`` opens the fingers only where that is known to be safe: in place after a lift
that did not plan, where the object rests on its support, or after a carry whose gripper
opening showed the object leave the hand (``transition_refuted``, cause ``retained``,
personalrobotics/ssgrasping#28), where nothing is left in it. ``safe_release`` decides
that from ssgrasping's failed ``Stage`` (``failed_stage``): its name, reason and cause,
whether the skill then believed the object held, and whether it is still declared
attached. Each case here is a failure ssgrasping's
end-to-end evidence produces (its ``obstructed`` lift and ``pulled`` transfer, and a
lift the object leaves the same way): the decision holds whatever the stage's detail
says, and is withheld when the reason or the attachment differs, or, for a lost object,
unless the skill reported it not held.

Reproduce with ``uv run pytest tests/test_recovery.py``.
"""

from __future__ import annotations

from ssgrasping import Evidence, Reason, SkillResult, SkillState, Stage

from geodude.recycling import RELEASE_IN_PLACE, RELEASE_LOST, failed_stage, safe_release

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
    held: bool | None,
    attached: bool = True,
) -> SkillResult:
    ended = Stage(
        name=stage,
        succeeded=False,
        detail=detail,
        time_ns=0,
        revision=0,
        held=held,
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
        True,
        RELEASE_IN_PLACE,
    ),
    (
        ("drop", "transfer", Reason.TRANSITION_REFUTED, "retained", Evidence.PORTABLE),
        "the opening fell below 0.224: the object left the hand",
        False,
        RELEASE_LOST,
    ),
    (
        ("grasp", "lift", Reason.TRANSITION_REFUTED, "retained", Evidence.PORTABLE),
        "the opening fell below 0.224: the object left the hand",
        False,
        RELEASE_LOST,
    ),
)


def test_release_is_chosen_by_stage_reason_and_attachment_not_detail() -> None:
    for (skill, stage, reason, cause, evidence), today, held, action in CASES:
        for detail in DETAILS:
            result = _failed(
                skill, stage, reason, cause, today if detail is None else detail, evidence, held
            )
            assert safe_release(failed_stage(result)) == action, (skill, stage, detail)
        if action == RELEASE_LOST:
            # Detached only if the skill reported the object not held.
            for believed in (None, True):
                kept = _failed(skill, stage, reason, cause, today, evidence, believed)
                assert safe_release(failed_stage(kept)) is None, (skill, stage, believed)
        # The same stage failing otherwise, or with nothing still attached, has no known
        # support to release onto.
        other = Reason.COMMAND_FAILED if reason is Reason.NOT_PLANNED else Reason.REFUSED
        otherwise = _failed(skill, stage, other, None, today, evidence, held)
        assert safe_release(failed_stage(otherwise)) is None
        detached = _failed(skill, stage, reason, cause, today, evidence, held, attached=False)
        assert safe_release(failed_stage(detached)) is None
    assert safe_release(None) is None
