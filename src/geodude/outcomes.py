"""The recycling demo's durable outcome: typed, versioned records (personalrobotics/geodude#246).

A run's outcome is a ``RecyclingRun``: its seed, what it ran from, the scene it staged,
the run bundle it cites by digest, every attempt, and where each object came to rest.
The records are built on ssrobot's strict wire base (``Record``, ``Value``, ``dumps``,
``loads``): decoding refuses a missing, extra, or wrongly typed field, a wrong schema or
version, a duplicate key, or a non-finite number, and never coerces. Each record's
``_validate`` refuses what is internally inconsistent, each with its own code:

- ``inconsistent_arm``: an attempt with no arm selected has skill evidence, or one with an
  arm has none;
- ``inconsistent_destination``: an attempt's result is not that of its destination;
- ``inconsistent_success``: ``succeeded`` disagrees with the checks;
- ``inconsistent_failure``: a failure without an arm, or beside a success, except
  ``no_grasp`` at grasp.regions and ``no_drop_zone`` at drop.zone, which an attempt with
  neither arm nor race carries, on planned evidence, neither held nor attached, with no
  lookahead, times or recovery: an object refused before it is raced;
- ``missing_failure``: a selected arm's attempt with a failed skill check and no failure
  (an object that bounced out of its bin after every skill succeeded is explained by
  ``in_bin`` alone, and needs none);
- ``inconsistent_recovery``: a recovery without a failure, a release without an action,
  or a place-back other than of a selected arm's object bound for a bin whose drop's
  transfer did not plan (released or not: the placement itself can fail);
- ``incomplete_landing``: a released, evaluated drop without its landing;
- ``unknown_reason``: a failure's reason is neither ssgrasping's nor ``no_drop_zone`` or
  ``no_grasp``;
- ``noncontiguous_attempts``: an object's attempts are not numbered 1, 2, ...;
- ``inconsistent_disposition``: an object's disposition disagrees with its attempts;
- ``inconsistent_evaluation``: evaluation fields present other than exactly when
  ``evaluated`` is set, or ``cleared`` not their verdict;
- ``staged_mismatch``: the dispositions are not exactly the scene's staged objects;
- ``seed_mismatch``: the run's seed is not its scene's;
- ``unselected_success``: a race with a successful candidate selected none;
- ``selected_not_succeeded``: a race selected a candidate that did not succeed;
- ``outranked_with_plans``: an outranked candidate carries plans or a cost.

An attempt still in progress, such as a drop whose landing is still observed, is never
one of these records: ``recycling.recycle`` keeps it internal until it is complete.

Nothing here depends on how long anything took in wall-clock time, or where a file was:
two runs of the same seed on the same OpenBLAS kernel encode to the same bytes. A race
records only what its selection claims (``Race``). ``RecyclingRun.VERSION`` versions this
format, and only it. 2: the reason ``no_grasp`` and the recovery ``place back on the
worktop``, which a version 1 decoder cannot read. 3: ssgrasping's reason
``gripper_obstructed``, a gripper command refused before it ran (ssgrasping#54), which a
version 2 decoder cannot read. 4: drops to regions for the object (#260): ``no_drop_zone``
refuses an object before it is raced, armless, when no resting orientation of it fits
the bin's opening, and a place-back follows a drop whose transfer did not plan, which a
version 3 decoder cannot read. 5: ssgrasping's reason ``transition_unobserved``, a carry
refused because its hand's opening cannot show that it still holds the object and it was
not told to assume so (ssgrasping#59), which a version 4 decoder cannot read. Geodude's
Robotiq hands' openings do establish a grasp, so it never records one.
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass, fields
from typing import ClassVar

from ssgrasping import EstimateReason, Evidence, Reason, Transition, TransitionOutcome
from ssrobot import (
    ObjectStatus,
    PlanStatus,
    Pose,
    Record,
    RestOutcome,
    StageOutcome,
    ValidationError,
    Value,
)

NO_DROP_ZONE = "no_drop_zone"
"""The reason of a drop that found no zone for the object as it is held: Geodude's,
beside ssgrasping's ``Reason``s."""
NO_GRASP = "no_grasp"
"""The reason of an object the hand has no grasp of at all: none of its grasp regions
gives a pregrasp region within the hand's limits, wherever it stands. It is never raced:
its attempt has no arm and no race. Geodude's, beside ssgrasping's ``Reason``s."""
REASONS = frozenset({*(r.value for r in Reason), NO_DROP_ZONE, NO_GRASP})


class Destination(enum.StrEnum):
    """Where an attempt takes its object."""

    BIN = "bin"
    WORKTOP = "worktop"


class RecoveryAction(enum.StrEnum):
    """How an object still attached after a failed attempt was let go."""

    RELEASE_IN_PLACE = "release in place"
    RELEASE_LOST = "detach the lost object"
    PLACE_BACK = "place back on the worktop"
    """Held firmly with no transfer to a release that plans: put back where it was picked
    from."""


class RaceBasis(enum.StrEnum):
    """What a race planned on."""

    SNAPSHOT = "snapshot"
    PREDICTION = "prediction"


class CandidateOutcome(enum.StrEnum):
    """How a candidate of a race ended. ``outranked``: a strictly higher priority was
    chosen, so whether this one was stopped, and where, or finished first depended on
    planning speed; its plans are not recorded."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    OUTRANKED = "outranked"


@dataclass(frozen=True, slots=True, kw_only=True)
class Source(Value):
    """A distribution the run ran from: its commit when installed from a VCS URL, None
    otherwise (an editable, local, wheel, or index install)."""

    name: str
    version: str
    commit: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Placed(Value):
    """An object the scene staged on the worktop."""

    name: str
    kind: str
    position: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True, kw_only=True)
class Fixture(Value):
    """A fixture the scene stood beside the table, its base at ``base``."""

    name: str
    kind: str
    base: tuple[float, float, float]


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneFile(Value):
    """A staged file, by its path relative to the scene and the SHA-256 of its bytes:
    the bundle's ``scene/<path>``."""

    path: str
    sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Rejected(Value):
    """A staging attempt refused for putting objects against the robot."""

    attempt: int
    near_robot: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Scene(Value):
    """The staged scene's manifest (``geodude.scene``): the prl_assets source, the seed,
    every placement and fixture, every staged file's hash, and the staging attempts."""

    prl_assets: Source
    seed: int
    objects: tuple[Placed, ...]
    fixtures: tuple[Fixture, ...]
    files: tuple[SceneFile, ...]
    attempt: int
    rejected: tuple[Rejected, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Prediction(Value):
    """The prediction a lookahead planned on, as ssgrasping logs it."""

    base: str
    lineage: tuple[str, ...]
    moved: tuple[str, ...]
    held: tuple[str, ...]
    unsettled: tuple[str, ...]
    releases: tuple[str, ...]
    removals: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class RestEvidence(Value):
    """What the rest watch before a continuation established (ssgrasping's
    ``rest_evidence``)."""

    outcome: RestOutcome
    detail: str
    channel: str
    privileged: bool
    criteria_source: str
    start_ns: int
    end_ns: int
    window: tuple[int, int] | None
    terminal: tuple[float, ...] | None
    samples: int
    longest: tuple[int, int] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class LookaheadEntry(Value):
    """One entry of a skill's lookahead log: a stage it did not plan ahead, and why
    (``boundary``), or one it planned on a prediction made ``during`` an execution, and
    what became of that plan."""

    stage: str
    boundary: str | None = None
    during: str | None = None
    prediction: Prediction | None = None
    kept: str | None = None
    raced: tuple[str, ...] | None = None
    decision: str | None = None
    refusal: str | None = None
    detail: str | None = None
    max_start_gap: float | None = None
    rest: RestEvidence | None = None
    rebase: str | None = None
    waited_for_rest_ns: int | None = None

    def _validate(self) -> None:
        if (self.boundary is None) == (self.prediction is None):
            raise ValidationError(
                "wrong_value", "a boundary, or a plan on a prediction", path="boundary"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class Lookahead(Value):
    """Each skill's lookahead log; ``drop`` is None when nothing was dropped."""

    grasp: tuple[LookaheadEntry, ...]
    drop: tuple[LookaheadEntry, ...] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class RacePlan(Value):
    """One try of one stage of one seed of a candidate's prefix: how it ended, the
    planner's status if it returned, and its validated duration if valid."""

    seed: int
    stage: str
    stage_seed: int | None
    status: StageOutcome
    plan_status: PlanStatus | None
    duration_s: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Candidate(Value):
    """One arm in a race: its outcome, why, the validated duration of its chosen
    prefix, and every plan it tried, none when it was ``outranked``."""

    label: str
    priority: int
    outcome: CandidateOutcome
    detail: str
    cost_s: float | None
    plans: tuple[RacePlan, ...]

    def _validate(self) -> None:
        if self.outcome is CandidateOutcome.OUTRANKED and (self.plans or self.cost_s is not None):
            raise ValidationError(
                "outranked_with_plans", "an outranked candidate has no plans or cost", path="plans"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class Race(Value):
    """The arms' race for an object (ssgrasping's ``Candidates``), as far as its
    selection is deterministic: what it planned on (a snapshot, or a prediction made
    during an execution), the rule, every candidate, and the selected one's label."""

    on: RaceBasis
    snapshot: str | None
    stage: str | None
    during: str | None
    prediction: Prediction | None
    rule: str
    candidates: tuple[Candidate, ...]
    selected: str | None

    def _validate(self) -> None:
        predicted = (self.stage, self.during, self.prediction)
        if self.on is RaceBasis.SNAPSHOT:
            wrong = self.snapshot is None or predicted != (None, None, None)
        else:
            wrong = self.snapshot is not None or None in predicted
        if wrong:
            raise ValidationError("wrong_value", "the basis is its kind's", path="on")
        winners = [c.label for c in self.candidates if c.outcome is CandidateOutcome.SUCCEEDED]
        if self.selected is not None and self.selected not in winners:
            raise ValidationError(
                "selected_not_succeeded", "the selected candidate succeeded", path="selected"
            )
        if self.selected is None and winners:
            raise ValidationError(
                "unselected_success", "a race with a success selects one", path="selected"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillStage(Value):
    """A skill stage: its name and whether it succeeded."""

    name: str
    succeeded: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class Times(Value):
    """When, on the runtime clock, an attempt's grasp recorded its first stage, its drop
    released, and its landing was judged."""

    grasp: int | None
    released: int | None
    landed: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Failure(Value):
    """The stage that decided an attempt that failed: its skill, name, reason and cause,
    detail (prose, never acted on), evidence kind, and whether the object was then held
    and declared attached."""

    skill: str
    stage: str
    reason: str | None
    cause: str | None
    detail: str
    evidence: Evidence
    held: bool | None
    attached: bool

    def _validate(self) -> None:
        if self.reason is not None and self.reason not in REASONS:
            raise ValidationError("unknown_reason", f"no reason {self.reason!r}", path="reason")


@dataclass(frozen=True, slots=True, kw_only=True)
class Recovery(Value):
    """What was done after a failed attempt that left its object attached: the action,
    None when nothing was safe, whether the object was released, and a detail."""

    action: RecoveryAction | None
    released: bool
    detail: str

    def _validate(self) -> None:
        if self.action is None and self.released:
            raise ValidationError(
                "inconsistent_recovery", "released with no action", path="released"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TransitionRecord(Value):
    """A transition a skill estimated: where, the estimate, whether the demo assumed it,
    and, where the simulator had the object's privileged state, its reference
    (``reference``): privileged, not ground truth, for evaluation only."""

    skill: str
    stage: str
    transition: Transition
    outcome: TransitionOutcome
    reason: EstimateReason
    assumed: bool
    reference: TransitionOutcome | None
    reference_source: str | None

    def _validate(self) -> None:
        if (self.reference is None) != (self.reference_source is None):
            raise ValidationError(
                "wrong_value", "a reference exactly with its source", path="reference_source"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class BinChecks(Value):
    """A drop's checks. ``landed`` and ``in_bin`` need evaluation: None without it."""

    EVALUATED: ClassVar[tuple[str, ...]] = ("landed", "in_bin")
    """The checks an evaluation makes, from privileged state; the others are the skills'."""

    grasped: bool
    dropped: bool
    released: bool
    landed: bool | None
    detached: bool
    in_bin: bool | None


@dataclass(frozen=True, slots=True, kw_only=True)
class InBin(Value):
    """A drop's result: the bin, its base, where the object rests relative to it (None
    without evaluation), and its checks."""

    name: str
    base: tuple[float, float, float]
    position_in_bin_m: tuple[float, float, float] | None
    checks: BinChecks


@dataclass(frozen=True, slots=True, kw_only=True)
class WorktopChecks(Value):
    """A placement's checks."""

    EVALUATED: ClassVar[tuple[str, ...]] = ("upright", "on_worktop", "at_target")
    """The checks an evaluation makes, from privileged state; the others are the skills'."""

    grasped: bool
    placed: bool
    released: bool
    nothing_attached: bool
    upright: bool
    on_worktop: bool
    at_target: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class Tolerances(Value):
    """How far from upright, above or below the worktop, and from the target a placed
    object may rest."""

    tilt_rad: float
    base_m: float
    offset_xy_m: float


@dataclass(frozen=True, slots=True, kw_only=True)
class OnWorktop(Value):
    """A placement's result: the target, how far the object's axis is from upright, its
    base's height above the worktop, its offset from the target, and its checks."""

    target: tuple[float, float]
    tilt_rad: float
    base_above_worktop_m: float
    offset_from_target_xy_m: tuple[float, float]
    tolerances: Tolerances
    checks: WorktopChecks


def passed(checks: BinChecks | WorktopChecks) -> bool | None:
    """Whether every check passed: None when one needs an evaluation that was not made."""
    values = [getattr(checks, f.name) for f in fields(checks)]
    return None if None in values else all(values)


@dataclass(frozen=True, slots=True, kw_only=True)
class Attempt(Value):
    """One complete attempt at an object: its number, the arm the race selected (None
    when no arm's prefix planned), the race, the scene revision of the registration it
    planned on, each skill's stages and lookahead, when it grasped, released and was
    judged, the failure and any recovery, every transition, what was still attached when
    it ended, where the object rests (with evaluation), its destination's result, and
    whether it succeeded (None without evaluation)."""

    object: str
    number: int
    arm: str | None
    race: Race | None
    registration: int | None
    lookahead: Lookahead | None
    destination: Destination
    grasp: tuple[SkillStage, ...] | None
    drop: tuple[SkillStage, ...] | None
    landing: tuple[SkillStage, ...] | None
    place: tuple[SkillStage, ...] | None
    times: Times | None
    failure: Failure | None
    recovery: Recovery | None
    transitions: tuple[TransitionRecord, ...]
    attached: tuple[str, ...] | None
    final_pose: Pose | None
    bin: InBin | None
    worktop: OnWorktop | None
    succeeded: bool | None

    @property
    def checks(self) -> dict[str, bool | None]:
        """Its destination's checks, by name; empty with no arm."""
        result = self.bin or self.worktop
        if result is None:
            return {}
        return {f.name: getattr(result.checks, f.name) for f in fields(result.checks)}

    def _validate(self) -> None:
        if self.number < 1:
            raise ValidationError("noncontiguous_attempts", "numbered from 1", path="number")
        evidence = (self.grasp, self.bin, self.worktop, self.attached)
        if self.arm is None:
            if evidence != (None,) * 4 or self.transitions or self.succeeded is not False:
                raise ValidationError(
                    "inconsistent_arm", "no arm, no skill ran: not succeeded", path="arm"
                )
        elif self.grasp is None or self.attached is None:
            raise ValidationError("inconsistent_arm", "an arm's grasp ran", path="grasp")
        if self.arm is not None:
            bin_ = self.destination is Destination.BIN
            if (self.bin is None) == bin_ or (self.worktop is None) != bin_:
                raise ValidationError(
                    "inconsistent_destination", "the destination's result only", path="bin"
                )
            dropped = self.drop is not None or self.landing is not None
            if (bin_ and self.place is not None) or (not bin_ and dropped):
                raise ValidationError(
                    "inconsistent_destination", "a drop goes to a bin", path="drop"
                )
            result = self.bin or self.worktop
            assert result is not None  # the destination's, above
            expected = passed(result.checks)
            evaluated = expected is not None
            if self.succeeded != expected:
                raise ValidationError(
                    "inconsistent_success", "succeeded is every check", path="succeeded"
                )
            if (self.final_pose is None) == evaluated:
                raise ValidationError(
                    "inconsistent_evaluation",
                    "a final pose exactly when evaluated",
                    path="final_pose",
                )
            if (
                self.bin is not None
                and self.bin.checks.released
                and evaluated
                and (self.landing is None)
            ):
                raise ValidationError(
                    "incomplete_landing",
                    "a released, evaluated drop has its landing",
                    path="landing",
                )
        refused = self.failure
        before = {NO_GRASP: ("grasp", "regions"), NO_DROP_ZONE: ("drop", "zone")}
        ungraspable = refused is not None and refused.reason in before
        if (
            refused is not None
            and ungraspable
            and (
                self.race is not None
                or self.arm is not None
                or (refused.skill, refused.stage) != before.get(refused.reason or "")
                or refused.evidence is not Evidence.PLANNED
                or refused.held is not False
                or refused.attached
                or self.recovery is not None
                or self.lookahead is not None
                or self.times is not None
            )
        ):
            raise ValidationError(
                "inconsistent_failure",
                "an object refused before it is raced (no_grasp at grasp.regions, "
                "no_drop_zone at drop.zone) is so on planned evidence, neither held nor "
                "attached: neither raced nor taken by an arm, with nothing planned, timed "
                "or recovered",
                path="failure",
            )
        if (
            self.failure is not None
            and not ungraspable
            and (self.arm is None or self.succeeded is True)
        ):
            raise ValidationError(
                "inconsistent_failure",
                "a failure only of a selected arm's attempt that did not succeed",
                path="failure",
            )
        result = self.bin or self.worktop
        if result is not None and self.failure is None:
            # A failed skill check is explained only by a failure; an evaluated check
            # alone (the object bounced out of its bin) needs none.
            failed = {k for k, ok in self.checks.items() if ok is False}
            unexplained = failed - set(result.checks.EVALUATED)
            if unexplained:
                raise ValidationError(
                    "missing_failure",
                    f"a selected arm's attempt that failed {sorted(unexplained)} names its failure",
                    path="failure",
                )
        if self.recovery is not None and self.failure is None:
            raise ValidationError(
                "inconsistent_recovery", "a recovery only after a failure", path="recovery"
            )
        if (
            self.recovery is not None
            and self.recovery.action is RecoveryAction.PLACE_BACK
            and (
                self.arm is None
                or self.destination is not Destination.BIN
                or self.failure is None
                or (self.failure.skill, self.failure.stage, self.failure.reason)
                != ("drop", "transfer", Reason.NOT_PLANNED.value)
            )
        ):
            raise ValidationError(
                "inconsistent_recovery",
                "a place-back only of a selected arm's object bound for a bin whose drop's "
                "transfer did not plan",
                path="recovery",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class Disposition(Value):
    """Where an object ended up: how many attempts it had, whether it is still
    attached, its state in the registered scene, and, with evaluation, the bin it rests
    in (None outside every bin) and its final pose."""

    object: str
    attempts: int
    attached: bool
    status: ObjectStatus
    bin: str | None
    final_pose: Pose | None


@dataclass(frozen=True, slots=True, kw_only=True)
class RecyclingRun(Record):
    """One run of the recycling demo: the seed; the OpenBLAS kernel it ran on; each of
    its sources; the staged scene; the SHA-256 of its run bundle's ``bundle.json``, which
    names every file of the bundle by hash, or None when no bundle was written; whether
    its outcomes were evaluated against privileged state; every attempt, in order; every
    staged object's disposition, by name; and, with evaluation, whether every object
    rests in a bin (``cleared``)."""

    SCHEMA = "geodude.RecyclingRun"
    VERSION = 5

    seed: int
    blas_kernel: str | None
    sources: tuple[Source, ...]
    scene: Scene
    bundle: str | None
    evaluated: bool
    attempts: tuple[Attempt, ...]
    objects: tuple[Disposition, ...]
    cleared: bool | None

    def _validate(self) -> None:
        if self.seed != self.scene.seed:
            raise ValidationError("seed_mismatch", "the run's seed is its scene's", path="seed")
        names = [d.object for d in self.objects]
        if set(names) != {o.name for o in self.scene.objects}:
            raise ValidationError(
                "staged_mismatch", "a disposition for each staged object only", path="objects"
            )
        if names != sorted(set(names)):
            raise ValidationError(
                "inconsistent_disposition", "one disposition per object, by name", path="objects"
            )
        seen: Counter[str] = Counter()
        last: dict[str, Attempt] = {}
        for i, a in enumerate(self.attempts):
            seen[a.object] += 1
            if a.number != seen[a.object]:
                raise ValidationError(
                    "noncontiguous_attempts",
                    f"{a.object}'s attempt {seen[a.object]} is numbered {a.number}",
                    path=f"attempts[{i}].number",
                )
            last[a.object] = a
            if a.arm is not None and (a.final_pose is not None) != self.evaluated:
                raise ValidationError(
                    "inconsistent_evaluation",
                    "an attempt is evaluated exactly when the run is",
                    path=f"attempts[{i}]",
                )
        for i, d in enumerate(self.objects):
            ended = last.get(d.object)
            if d.attempts != seen[d.object] or (
                d.attached and ended is not None and ended.succeeded
            ):
                raise ValidationError(
                    "inconsistent_disposition",
                    f"{d.object} does not agree with its attempts",
                    path=f"objects[{i}]",
                )
            if (d.final_pose is not None) != self.evaluated or (
                d.bin is not None and not self.evaluated
            ):
                raise ValidationError(
                    "inconsistent_evaluation",
                    "a final pose and bin exactly when evaluated",
                    path=f"objects[{i}]",
                )
        if set(seen) - set(names):
            raise ValidationError(
                "inconsistent_disposition",
                "every attempted object has a disposition",
                path="objects",
            )
        cleared = all(d.bin is not None for d in self.objects) if self.evaluated else None
        if self.cleared != cleared:
            raise ValidationError(
                "inconsistent_evaluation", "cleared is every object in a bin", path="cleared"
            )
