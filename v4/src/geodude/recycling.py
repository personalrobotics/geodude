"""The recycling demo on ssrobot and ssgrasping (personalrobotics/geodude#201).

Steps 1 to 3 of the port, in a seeded scene staged from prl_assets: one can, picked with
ssgrasping's ``grasp`` by the arm on its side of the worktop, and dropped with its
``drop`` into the bin on that arm's side (the default) or placed upright on the worktop
with its ``place`` (``--to worktop``). Control runs on what a physical robot has: joint,
gripper and wrist channels, and the registration of the objects on the table
(personalrobotics/geodude#248). Contexts keep registered object poses
(``ObjectPoses.REGISTERED``), and ``register`` is the only way an object's pose enters
control. Its source is the ground-truth stub the demo has always used, each object's
privileged pose channel (ssrobot#101), until a perception source exists. Privileged
state is otherwise read only for evaluation.

Run it, watch it, and record it::

    python -m geodude.recycling --seed 0 --viser --record out/

``--viser`` serves a live view at http://localhost:8080; ``--record`` writes the run as
a bundle (``out/bundle``) that mj_viser replays, and its outcome as ``out/recycling.json``:
the sources it ran from, the OpenBLAS kernel, the seed, every skill stage, and where the
can came to rest.
The installed command ``geodude-recycling`` is the same.

``--cans 4`` stages four cans and recycles them all (``recycle``): both arms race for
each, the arm on the can's side preferred, and each can is planned while the previous
can is carried to its bin (personalrobotics/geodude#223).
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import importlib.metadata
import json
import math
import random
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sscbirrt import CBiRRTConfig
from ssgrasping import (
    Candidate,
    Candidates,
    DropSpec,
    GraspSpec,
    LandingSpec,
    Lookahead,
    PlaceSpec,
    Reason,
    SkillExecution,
    SkillResult,
    Stage,
    Transition,
    TransitionEstimate,
    TransitionOutcome,
    drop,
    grasp,
    landing,
    place,
    run_together,
)
from ssrobot import (
    ClockMode,
    Execution,
    ExecutionState,
    ExecutionStatus,
    GripperCommand,
    GripperTermination,
    JsonlTrace,
    ObjectPoses,
    ObjectStatus,
    ObservationRequest,
    Planner,
    Pose,
    Registration,
    RobotContext,
    RobotPackage,
    SceneSnapshot,
    Timestamp,
    WrenchGuard,
    WrenchPart,
    load_installed_package,
    write_bundle,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT, SsikKinematics

from geodude import regions
from geodude.scene import meta, stage_scene, worktop

TARGET = (0.20, -0.50)
"""Where step 1 places the can on the worktop."""

BINS = {"recycle_bin": [(0.85, -0.35, 0.01), (-0.85, -0.35, 0.01)]}
"""The legacy demo's bins, either side of the robot, on the floor: ``recycle_bin_0`` on
the right arm's side, ``recycle_bin_1`` on the left's. Each arm reaches only its own
side's bin (#201)."""
BIN_OF = {"right": 0, "left": 1}
"""Which of ``BINS`` each arm drops into: the one on its side. The rule is data: racing
the arms (#223) evaluates each arm through its own bin."""
DESTINATIONS = ("bin", "worktop")
UP = (0.0, 0.0, 1.0)
"""The worktop's outward normal."""
CONTACT_N = 8.0
"""The guarded placement's force threshold, measured in personalrobotics/ssgrasping#10:
free motion peaks at 4.4-5.1 N, contact crosses 8 N within one tick."""
HOLD_TOLERANCE_M = 0.02
SWING_TOLERANCE_RAD = 0.4
"""How far a held can may move from its declared grasp, in position and rotation,
before the attachment monitor reports it no longer held. Measured over 40 carries
(ten four-can rounds): a can pinched at the fingertips settles up to 11 mm and, after a
knock, turns up to 0.87 rad; one the 2F-140 pulls deep into the hand as it closes, held
by its inner links, re-seats up to 15 mm and 0.33 rad and stays held. A deep grasp is
secure, so its re-seating must not count as losing the can; a can turned 0.5 rad or more
out of a pinch is still reported."""
PLACE_SLACK_M = 0.03
"""How far from ``TARGET``, in each of x and y, the placement region lets an object's
centre be."""
PLACE_TILT_RAD = 0.05
"""How far from upright a placed object may rest."""
PLACE_TOLERANCE_M = 0.01
"""How far outside its region a placed object may rest and still count: the transfer may
aim at a region's edge, and the arm tracks a plan only to the runtime's goal tolerance
(0.01 rad per joint), a few millimetres at the hand."""
CARRY_SCALING = 0.7
"""The fraction of the joints' velocity and acceleration limits a held can is carried at:
at full speed, sscbirrt 3.11's transfers swing the wrist fast enough to throw it from the
fingers (personalrobotics/ssgrasping#16)."""
CLEARANCE_M = 0.005
"""The clearance the grasp's pregrasp and the drop's transfer, Geodude's moves through
free space, keep from the worktop, the bins and the other cans
(``PlanningRequest.clearance_m``). Over 100 seeds of four cans, racing: 380 cans
recycled against 344 with none (personalrobotics/ssrobot#183)."""
LINE_TOLERANCE_M = 0.002
"""How far a straight approach or lift may stray from its line (``GraspSpec``'s
``line_tolerance_m``, 1 mm by default). After rest, the arm arrives at a pregrasp 0.6 mm
across its approach line at the median and 1.9 mm at p95, over 71 approaches in 20 seeds
(personalrobotics/ssrobot#206), so the approach planned ahead continues from inside its
line. The tradeoff: the wider line changes which grasps are made, so which cans fail
moves too. Over 20 seeds, 210 of 223 cans against 212 of 225 at 1 mm: some seeds gain
cans, others lose one to the in-hand slip (geodude_assets#36) or to a transfer refused
when compiled (ssrobot#207)."""
CHECK_ITERATIONS = 300
"""The iteration budget of the planner that checks a follow-on straight move."""
PREFERRED = 1
"""The race priority of the arm on an object's side, over the other arm's 0. Raced at
equal priority, the shorter prefix won, and it was often the other arm's: a long
cross-body carry, from which cans slipped out of the grasp (personalrobotics/geodude#223)."""
MIX_KINDS = (
    "can",
    "spam_can",
    "cracker_box",
    "gelatin_box",
    "sugar_box",
    "pocky_box",
    "fuze_bottle",
    "pop_tarts_case",
    "plastic_glass",
)
"""The legacy demo's graspable types (``src/geodude/demos/recycling.py``), from which its
mix of objects is drawn."""
MAX_TICKS = 200_000
SOURCES = (
    "geodude",
    "ssrobot",
    "ssgrasping",
    "ssmjviser",
    "sscbirrt",
    "ssik",
    "sstsr",
    "mujoco",
    "numpy",
    "prl-assets",
    "geodude-assets",
)
"""The distributions a run's outcome names."""
REFERENCE_BLAS_KERNEL = "Haswell"
"""The OpenBLAS kernel a claimed outcome is measured on (personalrobotics/geodude#241).

OpenBLAS picks its kernel for the CPU it runs on, and kernels round differently: a
quaternion's norm comes out one ulp apart on AVX-512 and AVX2 processors, which moves a
grasp region, its plans, and the outcome. A run is deterministic on one kernel, so a
claim names one. Haswell is the kernel every x86-64 host with AVX2 can run; select it
with ``OPENBLAS_CORETYPE=Haswell`` before numpy loads."""


@dataclass(frozen=True)
class Arm:
    """One of Geodude's arms, by the names its package declares."""

    side: str

    @property
    def manipulator(self) -> str:
        return f"{self.side}_arm"

    @property
    def hand(self) -> str:
        return f"{self.side}_hand"

    @property
    def gripper(self) -> str:
        return f"{self.side}_gripper"

    @property
    def opening(self) -> str:
        return f"{self.side}_gripper_opening"

    @property
    def wrench(self) -> str:
        return f"{self.side}_wrist_wrench"

    @property
    def bin(self) -> tuple[str, tuple[float, float, float]]:
        """The bin on this arm's side: its name and base."""
        index = BIN_OF[self.side]
        x, y, z = BINS["recycle_bin"][index]
        return f"recycle_bin_{index}", (x, y, z)


def arm_for(position: Sequence[float]) -> Arm:
    """The arm on the side of the worktop at ``position``: x >= 0 is the right arm's
    side, as its bin is. Retrying with the other arm is recovery policy (#209, #210)."""
    return Arm("right" if position[0] >= 0 else "left")


def planners(package: RobotPackage, arm: Arm) -> tuple[Planner, Planner]:
    """sscbirrt with ssik for ``arm``, at a resolution fine enough for straight moves,
    and the same with a small iteration budget for checking follow-on moves."""

    def make(**config: Any) -> Planner:
        return CBiRRT(
            package.description,
            kinematics=SsikKinematics.from_package(package, arm.manipulator),
            config=CBiRRTConfig(edge_resolution=0.01, **config),
        )

    return make(), make(max_iterations=CHECK_ITERATIONS)


def warm_planning(ctx: RobotContext, snapshot: SceneSnapshot, arms: Sequence[Arm]) -> float:
    """Build each of ``arms``' planning model and native collision scene now, from
    ``snapshot``, before any skill starts, and return the wall-clock seconds it took.
    ssrobot keeps one of each per group, whatever the clearance (ssrobot#182), so every
    later plan reuses them. Without this, the first plan for each arm builds them while
    the robot waits: 0.66 s for both arms on the measurement host (#224). It moves that
    work to bring-up; it does not remove it."""
    began = time.perf_counter()
    for arm in arms:
        ctx.planning_scene(snapshot, arm.manipulator)
    return time.perf_counter() - began


OBJECT_POSES = ObjectPoses.REGISTERED
"""How recycling's contexts keep object poses: registered, never read live."""
REGISTRATION = "pose_stub"
"""``pose_stub``'s source, as the trace records it: the privileged pose channel, standing
in for perception (ssrobot#101)."""

Read = Callable[[RobotContext, str], Registration]
"""A registration source: an object's registration, with its pose, when it was observed,
and the source that observed it, which the trace records."""


def true_pose(ctx: RobotContext, name: str) -> Pose:
    """An object's pose, from its privileged pose channel: for evaluation, never control."""
    return pose_channel(ctx, name)[0]


def pose_channel(ctx: RobotContext, name: str) -> tuple[Pose, Timestamp]:
    """An object's pose and when it was read, from its privileged pose channel: the
    registration stub until a perception source exists (ssrobot#101)."""
    observation = ctx.observe(ObservationRequest(channels=(f"{name}.pose",)))
    value = observation.readings[0].value
    assert isinstance(value, tuple)
    x, y, z, w, i, j, k = value
    return Pose(position=(x, y, z), quat_wxyz=(w, i, j, k)), observation.stamp


def pose_stub(ctx: RobotContext, name: str) -> Registration:
    """The demo's registration source until a perception source exists: the object's
    privileged pose channel (``REGISTRATION``)."""
    pose, stamp = pose_channel(ctx, name)
    return Registration(object=name, pose=pose, stamp=stamp, source=REGISTRATION)


def register(ctx: RobotContext, objects: Sequence[str], read: Read = pose_stub) -> None:
    """Register each of ``objects`` as ``read`` reports it, under the source it names: the
    only way an object's pose enters control."""
    for name in objects:
        registration = read(ctx, name)
        if registration.object != name:
            raise ValueError(f"asked to register {name!r}, read {registration.object!r}")
        ctx.register(name, registration.pose, stamp=registration.stamp, source=registration.source)


def on_table(ctx: RobotContext) -> list[str]:
    """The objects neither held nor removed as landed: those registration covers."""
    return [
        name
        for name in ctx.info.objects
        if ctx.scene.object_status(name) in (ObjectStatus.REGISTERED, ObjectStatus.UNKNOWN)
    ]


def registered_pose(ctx: RobotContext, name: str) -> Pose:
    """An object's pose as last registered: what control knows of it."""
    registration = ctx.scene.registration(name)
    if registration is None:
        raise ValueError(f"{name!r} is {ctx.scene.object_status(name).value}, not registered")
    return registration.pose


def assume_landed(ctx: RobotContext, obj: str) -> Callable[[TransitionEstimate], None]:
    """The demo's policy for a dropped object's unobserved ``landed``: assumed, so it is
    removed from the active scene (personalrobotics/geodude#248). Where it truly came to
    rest is evaluated, never assumed."""

    def remove(estimate: TransitionEstimate) -> None:
        ctx.remove(obj, source="assumed_landed")

    return remove


def mix(seed: int, count: int) -> dict[str, int]:
    """The legacy demo's mix: ``count`` objects drawn, with a generator seeded by
    ``seed``, from ``MIX_KINDS``, as ``stage_scene``'s counts by type."""
    drawn = random.Random(seed).choices(MIX_KINDS, k=count)
    return {kind: drawn.count(kind) for kind in sorted(set(drawn))}


def kind_of(name: str) -> str:
    """An instance's prl_assets type: ``can_0`` is a ``can``."""
    return name.rsplit("_", 1)[0]


def drive(ctx: RobotContext, execution: SkillExecution, tick: Callable[[], None]) -> SkillResult:
    """Advance ``execution`` to its end, stepping the manual clock and calling ``tick``
    after each step (a viewer refresh, a snapshot). While the skill waits only on a plan
    made on a worker, the clock does not step."""
    assert ctx.info.clock_mode is ClockMode.MANUAL
    if not run_together(ctx, [execution], max_ticks=MAX_TICKS, tick=tick):
        execution.cancel(f"not done after {MAX_TICKS} ticks")
    assert execution.result is not None
    return execution.result


def pick(
    ctx: RobotContext,
    obj: str,
    *,
    arm: Arm,
    planners: tuple[Planner, Planner],
    tick: Callable[[], None] = lambda: None,
) -> SkillResult:
    """Grasp ``obj`` with ``arm`` from every sstsr grasp of its prl_assets geometry, at
    its pose channel, and lift it."""
    planner, checker = planners
    spec = grasp_spec(ctx, obj, arm)
    return drive(ctx, grasp(ctx, planner, spec, check_planner=checker), tick)


def grasp_spec(ctx: RobotContext, obj: str, arm: Arm) -> GraspSpec:
    """``pick``'s grasp of ``obj`` by ``arm``."""
    geometry = meta(kind_of(obj))["geometric_properties"]
    return GraspSpec(
        manipulator=arm.manipulator,
        end_effector=arm.hand,
        gripper=arm.gripper,
        opening_channel=arm.opening,
        object=obj,
        support="vention_base",
        normal=UP,
        pregrasps=regions.pregrasps(geometry, registered_pose(ctx, obj), CLEARANCE_M),
        approach_m=regions.STANDOFF_M,
        lift_m=0.06,
        line_tolerance_m=LINE_TOLERANCE_M,
        clearance_m=CLEARANCE_M,
    )


def place_on_worktop(
    ctx: RobotContext,
    package: RobotPackage,
    obj: str,
    target: tuple[float, float],
    *,
    arm: Arm,
    planners: tuple[Planner, Planner],
    tick: Callable[[], None] = lambda: None,
) -> SkillResult:
    """Place the held ``obj`` upright on the worktop within ``PLACE_SLACK_M`` of
    ``target`` in x and y, under the measured contact guard."""
    geometry = meta(kind_of(obj))["geometric_properties"]
    planner, checker = planners
    surface = worktop(package).centre[2]
    spec = PlaceSpec(
        manipulator=arm.manipulator,
        end_effector=arm.hand,
        gripper=arm.gripper,
        opening_channel=arm.opening,
        object=obj,
        target="vention_base",
        normal=UP,
        regions=[regions.upright(geometry, (*target, surface), PLACE_SLACK_M)],
        preplace_m=0.04,
        overshoot_m=0.01,
        terminal_m=0.01,
        guard=WrenchGuard(
            channel=arm.wrench, part=WrenchPart.FORCE, threshold=CONTACT_N, tare=True
        ),
        retreat_m=0.08,
        region_tolerance=(PLACE_TOLERANCE_M, PLACE_TILT_RAD),
        carry_scaling=CARRY_SCALING,
    )
    # Released, the object rests on the worktop, where registration sees it again.
    placing = place(
        ctx, planner, spec, check_planner=checker, released=lambda _: register(ctx, [obj])
    )
    return drive(ctx, placing, tick)


def drop_in_bin(
    ctx: RobotContext,
    obj: str,
    *,
    arm: Arm,
    planners: tuple[Planner, Planner],
    tick: Callable[[], None] = lambda: None,
) -> tuple[SkillResult, SkillResult | None]:
    """Drop the held ``obj`` into the bin on ``arm``'s side, from wherever over it the
    object, as it is held, falls clear of the rim (``held_drop_spec``), then observe
    whether it came to rest inside (``landing_spec``). The drop's result, and the
    landing's when the release was confirmed."""
    planner, checker = planners
    spec = held_drop_spec(ctx, drop_spec(obj, arm), arm)
    if spec is None:
        raise ValueError(f"no drop zone over {arm.bin[0]} fits {obj!r} as it is held")
    dropping = drop(ctx, planner, spec, check_planner=checker, released=assume_landed(ctx, obj))
    dropped = drive(ctx, dropping, tick)
    if not _released(dropped):
        return dropped, None
    return dropped, drive(ctx, landing(ctx, landing_spec(obj, arm), _landed(dropped)), tick)


def drop_spec(obj: str, arm: Arm) -> DropSpec:
    """The drop of ``obj`` by ``arm`` into its bin from the legacy demo's drop zone: what
    the race checks before anything is held. The drop that runs is ``held_drop_spec``'s."""
    name, base = arm.bin
    container = meta(kind_of(name))
    height = regions.height(meta(kind_of(obj))["geometric_properties"])
    return DropSpec(
        manipulator=arm.manipulator,
        end_effector=arm.hand,
        gripper=arm.gripper,
        opening_channel=arm.opening,
        object=obj,
        container=name,
        normal=UP,
        regions=[regions.drop_zone(container, base, height)],
        retreat_m=0.0,  # the next can's motion starts from the drop: no retreat (#223)
        carry_scaling=CARRY_SCALING,
        clearance_m=CLEARANCE_M,
    )


NO_ZONE = {
    "skill": "drop",
    "stage": "zone",
    "reason": "no_drop_zone",
    "cause": None,
    "detail": "no drop zone fits the object as it is held",
    "evidence": "planned",
}
"""The failure recorded when ``held_drop_spec`` finds no zone: nothing was dropped. Its
held and attached state are the grasp's, as it ended."""


def held_drop_spec(ctx: RobotContext, raced: DropSpec, arm: Arm) -> DropSpec | None:
    """``raced``, the drop the race planned with the legacy zone (``drop_spec``), over the
    zones where the object, as it is now held, falls clear of the bin's rim
    (``regions.held_drop_zones``), a ``CLEARANCE_M`` inside its walls. None when no zone
    fits: the object, held so, cannot be dropped clear of the rim."""
    attachment = ctx.scene.attachment(raced.object)
    assert attachment is not None  # a grasp that succeeded left it attached
    name, base = arm.bin
    zones = regions.held_drop_zones(
        meta(kind_of(name)),
        base,
        meta(kind_of(raced.object))["geometric_properties"],
        attachment.transform,
        CLEARANCE_M,
    )
    return dataclasses.replace(raced, regions=zones) if zones else None


def landing_spec(obj: str, arm: Arm) -> LandingSpec:
    """Where ``obj``, dropped by ``arm``, must come to rest: inside that arm's bin."""
    name, base = arm.bin
    return LandingSpec(object=obj, inside=[regions.interior(meta(kind_of(name)), base)])


def sources() -> dict[str, Any]:
    """Each of ``SOURCES``: its version and, when it was installed from a URL, a path or
    a file, where from."""
    out: dict[str, Any] = {}
    for name in SOURCES:
        dist = importlib.metadata.distribution(name)
        direct = dist.read_text("direct_url.json")
        out[name] = {
            "version": dist.version,
            "source": None if direct is None else json.loads(direct),
        }
    return out


def blas_kernel() -> str | None:
    """The OpenBLAS kernel numpy's linear algebra runs on, or None when numpy is not
    built on the OpenBLAS it bundles (on macOS on Apple silicon it uses Accelerate)."""
    import numpy

    root = Path(numpy.__file__).parent
    for library in sorted(
        [*root.parent.glob("numpy.libs/*openblas*"), *root.glob(".dylibs/*openblas*")]
    ):
        loaded = ctypes.CDLL(str(library))
        for symbol in ("scipy_openblas_get_corename64_", "scipy_openblas_get_corename"):
            corename = getattr(loaded, symbol, None)
            if corename is not None:
                corename.restype = ctypes.c_char_p
                return str(corename().decode())
    return None


def _stages(result: SkillResult | None) -> list[list[Any]] | None:
    return None if result is None else [[s.name, s.succeeded] for s in result.stages]


def _times(
    picked: SkillResult, dropped: SkillResult | None, landed: SkillResult | None
) -> dict[str, int | None]:
    """When, on the runtime clock, an attempt's grasp recorded its first stage, its drop
    released, and its landing was judged."""
    released = (
        None
        if dropped is None
        else next((s.time_ns for s in dropped.stages if s.name == "release"), None)
    )
    return {
        "grasp": picked.stages[0].time_ns if picked.stages else None,
        "released": released,
        "landed": landed.stages[-1].time_ns if landed is not None and landed.stages else None,
    }


ASSUMED = (Transition.LANDED, Transition.PLACED)
"""The transitions the demo assumes true when no portable signal can tell (``unobserved``):
a confirmed release counts as landed, or placed (personalrobotics/geodude#248)."""


def _landed(dropped: SkillResult) -> TransitionEstimate:
    """The drop's ``landed`` estimate, made at its confirmed release."""
    return next(t for t in dropped.transitions if t.transition is Transition.LANDED)


def _transitions(*results: SkillResult | None) -> list[dict[str, Any]]:
    """Every transition the skills estimated, in order: where, the estimate, whether the
    demo assumed it (``ASSUMED``, when unobserved), and, where the simulator had the
    object's privileged state, its reference (``reference``): privileged, not ground
    truth, for evaluation only. A landing's evaluation scores the drop's ``landed``."""
    scored = {
        (e.estimate.stage, e.estimate.transition): e
        for result in results
        if result is not None
        for e in result.evaluations
    }
    out = []
    for result in results:
        if result is None:
            continue
        for t in result.transitions:
            judged = scored.get((t.stage, t.transition))
            out.append(
                {
                    "skill": result.skill,
                    "stage": t.stage,
                    "transition": t.transition.value,
                    "outcome": t.outcome.value,
                    "reason": t.reason.value,
                    "assumed": t.outcome is TransitionOutcome.UNOBSERVED
                    and t.transition in ASSUMED,
                    "reference": None if judged is None else judged.reference.value,
                    "reference_source": None if judged is None else judged.source,
                }
            )
    return out


def _failure(*results: SkillResult | None) -> dict[str, Any] | None:
    """The stage that decided the first of ``results`` that did not succeed: its skill,
    name, reason and cause, detail, evidence kind, and whether the object was then held
    and declared attached. None when every result present succeeded."""
    for result in results:
        if result is not None and not result.succeeded:
            stage = result.stage
            return {
                "skill": result.skill,
                "stage": stage.name,
                "reason": None if stage.reason is None else stage.reason.value,
                "cause": stage.cause,
                "detail": stage.detail,
                "evidence": stage.evidence.value,
                "held": stage.held,
                "attached": stage.attached,
            }
    return None


def _released(result: SkillResult | None) -> bool:
    return result is not None and any(s.name == "release" and s.succeeded for s in result.stages)


def _pose(pose: Pose) -> dict[str, list[float]]:
    return {"position": list(pose.position), "quat_wxyz": list(pose.quat_wxyz)}


def worktop_outcome(
    ctx: RobotContext,
    package: RobotPackage,
    obj: str,
    target: tuple[float, float],
    picked: SkillResult,
    placed: SkillResult | None,
) -> dict[str, Any]:
    """What a pick and worktop placement achieved, observed after it ends: every stage,
    whether the release was confirmed, where ``obj`` rests (from its pose channel), how
    far its axis is from upright, its base's height above the worktop, and what is still
    attached. ``succeeded`` requires every stage to succeed, nothing attached, and the
    object upright within ``PLACE_TILT_RAD``, its base on the worktop, and its centre
    within ``PLACE_SLACK_M`` plus ``PLACE_TOLERANCE_M`` of ``target`` in x and y. An
    evaluation: it reads the object's privileged pose."""
    final = true_pose(ctx, obj)
    _, x, y, _ = final.quat_wxyz
    tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))  # body z from world z
    offset = [final.position[i] - target[i] for i in range(2)]
    height = regions.height(meta(kind_of(obj))["geometric_properties"])
    base = final.position[2] - math.cos(tilt) * height / 2 - worktop(package).centre[2]
    attached = [a.object for a in ctx.scene.attachments]
    checks = {
        "grasped": picked.succeeded,
        "placed": placed is not None and placed.succeeded,
        "released": _released(placed),
        "nothing_attached": not attached,
        "upright": tilt <= PLACE_TILT_RAD,
        "on_worktop": abs(base) <= PLACE_TOLERANCE_M,
        "at_target": max(map(abs, offset)) <= PLACE_SLACK_M + PLACE_TOLERANCE_M,
    }
    return {
        "to": "worktop",
        "object": obj,
        "target": list(target),
        "grasp": _stages(picked),
        "place": _stages(placed),
        "transitions": _transitions(picked, placed),
        "final_pose": _pose(final),
        "tilt_rad": tilt,
        "base_above_worktop_m": base,
        "offset_from_target_xy_m": offset,
        "attached": attached,
        "tolerances": {
            "tilt_rad": PLACE_TILT_RAD,
            "base_m": PLACE_TOLERANCE_M,
            "offset_xy_m": PLACE_SLACK_M + PLACE_TOLERANCE_M,
        },
        "checks": checks,
        "succeeded": all(checks.values()),
    }


def _in_bin(pose: Pose, arm: Arm) -> bool:
    """Whether an object at ``pose`` has its centre within ``arm``'s bin's interior
    (``regions.interior``, as its landing checks): inside its inner walls, over its floor,
    below its rim."""
    name, base = arm.bin
    region = regions.interior(meta(kind_of(name)), base)
    local = [pose.position[i] - region.origin.position[i] for i in range(3)]
    return all(
        low <= v <= high for v, (low, high) in zip(local, region.position_bounds, strict=True)
    )


def bin_outcome(
    ctx: RobotContext,
    obj: str,
    arm: Arm,
    picked: SkillResult,
    dropped: SkillResult | None,
    landed: SkillResult | None,
    *,
    evaluate: bool = True,
) -> dict[str, Any]:
    """What a pick, drop, and landing achieved, observed when the landing ends, or the
    drop when it released nothing: every stage, whether the release was confirmed, and
    what is still attached, which may be the next object. With ``evaluate``, where
    ``obj`` rests, from its privileged pose, relative to the bin's base: ``succeeded``
    requires every stage to succeed, ``obj`` no longer attached, its landing scored
    ``confirmed`` (``landed``), and its centre within the bin's prl_assets interior:
    inside its inner walls, over its floor, below its rim. Without it, no privileged
    pose is read for the outcome, and what needs one is None."""
    name, base = arm.bin
    final = true_pose(ctx, obj) if evaluate else None
    local = None if final is None else [final.position[i] - base[i] for i in range(3)]
    attached = [a.object for a in ctx.scene.attachments]
    scored = next(
        (e.reference for e in (landed.evaluations if landed is not None else ())),
        None,
    )
    checks: dict[str, bool | None] = {
        "grasped": picked.succeeded,
        "dropped": dropped is not None and dropped.succeeded,
        "released": _released(dropped),
        "landed": None if not evaluate else scored is TransitionOutcome.CONFIRMED,
        "detached": obj not in attached,
        "in_bin": None if final is None else _in_bin(final, arm),
    }
    return {
        "to": "bin",
        "object": obj,
        "bin": {"name": name, "base": list(base)},
        "times_ns": _times(picked, dropped, landed),
        "grasp": _stages(picked),
        "drop": _stages(dropped),
        "landing": _stages(landed),
        "final_pose": None if final is None else _pose(final),
        "position_in_bin_m": local,
        "attached": attached,
        "checks": checks,
        "failure": _failure(picked, dropped, landed),
        "transitions": _transitions(picked, dropped, landed),
        "succeeded": all(checks.values()) if evaluate else None,
    }


def demo(
    ctx: RobotContext,
    package: RobotPackage,
    to: str = "bin",
    tick: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Pick ``can_0`` in a staged scene with the arm on its side, and drop it in that
    arm's bin (``to="bin"``) or place it at ``TARGET`` on the worktop
    (``to="worktop"``), with a snapshot before and after. Returns the destination's
    outcome, with the arm used."""
    if to not in DESTINATIONS:
        raise ValueError(f"no destination {to!r}; one of {DESTINATIONS}")
    obj = "can_0"
    register(ctx, on_table(ctx))
    start = ctx.snapshot()
    arm = arm_for(registered_pose(ctx, obj).position)
    warm_planning(ctx, start, [arm])
    chosen = planners(package, arm)
    picked = pick(ctx, obj, arm=arm, planners=chosen, tick=tick)
    after: SkillResult | None = None
    landed: SkillResult | None = None
    if picked.succeeded and to == "bin":
        after, landed = drop_in_bin(ctx, obj, arm=arm, planners=chosen, tick=tick)
    elif picked.succeeded:
        after = place_on_worktop(ctx, package, obj, TARGET, arm=arm, planners=chosen, tick=tick)
    ctx.snapshot()
    if to == "bin":
        result = bin_outcome(ctx, obj, arm, picked, after, landed)
    else:
        result = worktop_outcome(ctx, package, obj, TARGET, picked, after)
    return {"arm": arm.side, **result}


def candidates(
    ctx: RobotContext, obj: str, arms: dict[str, tuple[Planner, Planner]]
) -> list[Candidate]:
    """Each arm, as a candidate for ``obj``: its grasp, and its drop into its own bin
    (``BIN_OF``), with that arm's planners. The arm on ``obj``'s side (``arm_for``) is
    preferred (``PREFERRED``); the other is the fallback, taken when the preferred arm's
    prefix does not plan."""
    preferred = arm_for(registered_pose(ctx, obj).position).side
    return [
        Candidate(
            label=side,
            grasp=grasp_spec(ctx, obj, Arm(side)),
            planner=planner,
            checker=checker,
            drop=drop_spec(obj, Arm(side)),
            priority=PREFERRED if side == preferred else 0,
        )
        for side, (planner, checker) in arms.items()
    ]


RETRIES = 1
"""How many more times ``sort_all`` races an object whose attempt failed but left it
released, upright, back on the worktop: once, so that an evaluation can count successes
at the first attempt and after a retry apart (#201)."""
RELEASE_IN_PLACE = "release in place"
RELEASE_LOST = "detach the lost object"


def _wait(ctx: RobotContext, execution: Execution, tick: Callable[[], None]) -> ExecutionStatus:
    """Step until ``execution`` ends, calling ``tick`` after each step."""
    for _ in range(MAX_TICKS):
        if execution.done:
            break
        ctx.step()
        tick()
    return execution.status


def _release(
    ctx: RobotContext, obj: str, arm: Arm, spec: GraspSpec, tick: Callable[[], None]
) -> str | None:
    """Open ``arm``'s gripper, and detach ``obj`` only once the fingers reached their open
    target, nothing between them. None when released; otherwise why not, ``obj`` still
    attached."""
    command = GripperCommand(gripper=arm.gripper, opening=spec.open_opening)
    status = _wait(ctx, ctx.submit(command, timeout_ns=spec.gripper_timeout_ns), tick)
    result = status.gripper
    if (
        status.state is not ExecutionState.SUCCEEDED
        or result is None
        or result.termination is not GripperTermination.REACHED_TARGET
    ):
        ended = (
            "" if result is None else f", {result.termination.value} at {result.final_opening:.3f}"
        )
        return f"the gripper did not open: {status.state.value}{ended}"
    ctx.detach(obj)
    return None


def failed_stage(*results: SkillResult | None) -> Stage | None:
    """The stage that decided the first of ``results`` that did not succeed: None when
    every result present succeeded."""
    return next((r.stage for r in results if r is not None and not r.succeeded), None)


def safe_release(failed: Stage | None) -> str | None:
    """How an object still declared attached after its ``failed`` stage may be let go:
    ``RELEASE_IN_PLACE``, ``RELEASE_LOST``, or None, nowhere known (``recover``). By the
    stage's name, its reason and cause, whether the skill then believed the object held,
    and the object still declared attached; never by its detail, which is prose. A lost
    object is detached only if the skill reported it not held (``held`` exactly False).
    None, with no failed stage: every skill succeeded and the object is still held, as
    when no drop zone fits it."""
    if failed is None or not failed.attached:
        return None
    if (failed.name, failed.reason) == ("lift", Reason.NOT_PLANNED):  # only a grasp lifts
        return RELEASE_IN_PLACE
    lost = (failed.reason, failed.cause) == (Reason.TRANSITION_REFUTED, Transition.RETAINED)
    if lost and failed.held is False:
        return RELEASE_LOST
    return None


def recover(
    ctx: RobotContext,
    obj: str,
    arm: Arm,
    failed: Stage | None,
    spec: GraspSpec,
    planner: Planner,
    tick: Callable[[], None],
    read: Read = pose_stub,
) -> dict[str, Any]:
    """After an attempt that failed with ``obj`` still attached, release it where that is
    safe (``safe_release``), a narrow slice of personalrobotics/geodude#210's recovery:
    - **A lift that did not plan** (``grasp.lift``, ``not_planned``): the arm never moved
      after closing, so the object still rests where it was grasped, on its support.
      Open, detach once the fingers have opened, register it where it rests (``read``),
      and back the hand out along its approach by ``spec.approach_m``.
    - **A carry the object left** (``transition_refuted``, cause ``retained``): the
      gripper's opening showed the fingers close past it, and the carry stopped there.
      Nothing is in the hand, so opening it releases nothing: open, and detach once the
      fingers have opened, so the scene no longer carries the object.
    - **Anything else:** the object may be suspended, or its support unknown, so nothing
      is done, and it stays declared held.
    The fingers are never opened onto nothing known, and ``obj`` is detached only after
    they reached open. Returns what was done: its ``action`` (None when nothing),
    whether ``obj`` was ``released``, and a ``detail``."""
    action = safe_release(failed)
    if action == RELEASE_IN_PLACE:
        refused = _release(ctx, obj, arm, spec, tick)
        if refused is not None:
            return {"action": RELEASE_IN_PLACE, "released": False, "detail": refused}
        register(ctx, [obj], read)  # detached, it is unknown until registered where it rests
        hand = next(e.frame for e in ctx.description.end_effectors if e.name == arm.hand)
        snapshot = ctx.snapshot()
        view = ctx.manipulator(arm.manipulator)
        q = tuple(snapshot.position(j) for j in ctx.description.group(arm.manipulator).joints)
        approach = regions.matrix(
            ctx.planning_scene(snapshot, arm.manipulator).forward_kinematics(q, hand)
        )[:3, 2]
        back = view.plan_to_end_effector_offset(
            tuple(float(-v) for v in approach), spec.approach_m, planner=planner
        )
        if not back.succeeded:
            retreat = f"not planned: {back.status.value}"
        else:
            retreat = _wait(ctx, ctx.execute(back, planner=planner), tick).state.value
        return {
            "action": RELEASE_IN_PLACE,
            "released": True,
            "detail": f"released where grasped; backed out: {retreat}",
        }
    if action == RELEASE_LOST:
        refused = _release(ctx, obj, arm, spec, tick)
        return {
            "action": RELEASE_LOST,
            "released": refused is None,
            "detail": refused or "detached: the object had left the hand",
        }
    return {
        "action": None,
        "released": False,
        "detail": "no safe release "
        + ("with no drop zone" if failed is None else f"after {failed.name}")
        + ": still declared held",
    }


def _upright_on_worktop(ctx: RobotContext, obj: str, start: Any) -> bool:
    """Whether ``obj``, as last registered, rests upright, within ``PLACE_TILT_RAD``, at
    the height it was registered at when the round began, within ``PLACE_TOLERANCE_M``:
    back on the worktop, as it began."""
    now = registered_pose(ctx, obj)
    began = next(o.known_pose() for o in start.objects if o.name == obj)
    _, x, y, _ = now.quat_wxyz
    tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))
    return tilt <= PLACE_TILT_RAD and abs(now.position[2] - began.position[2]) <= PLACE_TOLERANCE_M


def recycle(
    ctx: RobotContext,
    package: RobotPackage,
    objects: Sequence[str],
    tick: Callable[[], None] = lambda: None,
    *,
    retries: int = 0,
    read: Read = pose_stub,
    evaluate: bool = True,
) -> list[dict[str, Any]]:
    """Pick each of ``objects`` in turn and drop it in a bin, racing both arms for each
    (``Candidates``): the arm on its side takes it if its whole planned prefix, through
    the drop into its own bin, plans; otherwise the other arm, if its prefix plans. The
    drop has no retreat. The next object's race is planned while this drop's transfer
    carries the object to its bin, on the transfer's prediction, which expects the object
    released there and removed from the scene as landed (``assume_landed``), and the
    gripper open as it started. The winner's pregrasp then continues from where the
    transfer ended, as soon as the release is confirmed.

    Every object on the table is registered, by ``read``, when the round begins and
    before every race planned from a snapshot, when no prediction is in flight; and an
    object recovered onto the table, before it is raced again. Nothing else reads an
    object's pose for control. With ``evaluate``, where each released object comes to
    rest is scored alongside the objects after it (ssgrasping's ``landing``), and its
    outcome is recorded when that ends; the round ends once every landing has. Without
    it, Geodude runs no landing and reads no privileged pose for its outcomes. The skills
    may still record privileged references for their transitions
    (``SkillResult.evaluations``) where the runtime declares those channels; none of it
    steers control.
    Within the grasp, the approach is planned during the pregrasp. Returns each attempt's
    ``bin_outcome``, with the object's attempt number, the arm and the race.

    An attempt that fails with the object still attached is recovered by ``recover``. If
    nothing is safe, the round stops there, the object declared held. An object released
    back onto the worktop, upright, is raced again from the current state, up to
    ``retries`` more times. So is an object the grasp observably missed (``missed``: the
    fingers closed on nothing), which may have moved since it was registered: whatever
    was planned ahead is discarded, every free object on the table is registered again,
    and the object is raced again from a fresh snapshot. Each attempt records the scene
    revision of the registration its race planned on (``registration``). A landing still
    observed when the round's ticks run out is cancelled, and its outcome recorded so: every
    attempt returned is complete."""
    arms = {side: planners(package, Arm(side)) for side in BIN_OF}
    register(ctx, on_table(ctx), read)
    registration = ctx.scene.revision  # the table as last registered
    start = ctx.snapshot()
    warm_planning(ctx, start, [Arm(side) for side in BIN_OF])
    opened = {  # every gripper starts open; a release opens it again
        side: {j: start.position(j) for j in ctx.description.gripper(Arm(side).gripper).joints}
        for side in BIN_OF
    }
    outcomes: list[dict[str, Any]] = []
    queue = list(objects)
    attempts: dict[str, int] = {}
    looking: Lookahead | None = None
    pending: Candidates | None = None
    released: list[str] = []  # dropped, and assumed landed
    # Each released object's landing, still observed, and how to record its outcome.
    landings: list[tuple[SkillExecution, Callable[[SkillResult], None]]] = []

    def stepped() -> None:
        """After each step: ``tick``, then advance every landing still observed, and
        record each object's outcome as its landing ends."""
        tick()
        for observed, record in list(landings):
            observed.advance()
            if observed.result is not None:
                landings.remove((observed, record))
                record(observed.result)

    i = -1
    while i + 1 < len(queue):
        i += 1
        obj = queue[i]
        attempts[obj] = attempts.get(obj, 0) + 1
        if pending is None and ctx.scene.revision != registration:
            # Nothing in flight, and the scene has changed: the table is registered again.
            register(ctx, on_table(ctx), read)
            registration = ctx.scene.revision
        if pending is None:
            pending = Candidates(ctx, candidates(ctx, obj, arms), unsettled=released)
        run_together(ctx, [pending], max_ticks=MAX_TICKS, tick=stepped)  # blocks; no steps
        selection, report = pending.selection, pending.report
        pending = None
        if selection is None:
            outcomes.append(
                {
                    "object": obj,
                    "attempt": attempts[obj],
                    "arm": None,
                    "race": report,
                    "registration": registration,
                    "succeeded": False,
                }
            )
            continue
        arm = Arm(selection.candidate.label)
        planner, checker = arms[arm.side]
        looking = Lookahead(unsettled=tuple(released))
        picked = drive(ctx, selection.grasp(ctx, lookahead=looking), stepped)
        dropped = None
        spec = None
        if picked.succeeded:
            raced = selection.candidate.drop
            assert raced is not None  # every candidate drops
            spec = held_drop_spec(ctx, raced, arm)
        if spec is not None:
            following = queue[i + 1] if i + 1 < len(queue) else None
            dropping = drop(
                ctx,
                planner,
                spec,
                check_planner=checker,
                lookahead=looking,
                released=assume_landed(ctx, obj),
            )

            def race_next(
                transfer: Execution,
                following: str | None = following,
                obj: str = obj,
                arm: Arm = arm,
            ) -> Candidates:
                assert following is not None
                return Candidates(
                    ctx,
                    candidates(ctx, following, arms),
                    after=transfer,
                    unsettled=released,
                    removals=(obj,),
                    joints=opened[arm.side],
                )

            pending = _drop_racing_next(ctx, dropping, race_next if following else None, stepped)
            assert dropping.result is not None
            dropped = dropping.result
            released.append(obj)
            if pending is not None and not _released(dropped):
                # Planned on a release that did not happen: none of it can be used. Stop it,
                # rather than let it finish, and race again.
                pending.close()
                pending = None
        logs = {
            skill: result.values.get("lookahead", [])
            for skill, result in (("grasp", picked), ("drop", dropped))
            if result is not None
        }
        entry: dict[str, Any] = {
            "object": obj,
            "attempt": attempts[obj],
            "arm": arm.side,
            "race": report,
            "registration": registration,
            "lookahead": logs,
        }
        outcomes.append(entry)
        if _released(dropped) and evaluate:

            def record(
                landed: SkillResult,
                entry: dict[str, Any] = entry,
                obj: str = obj,
                arm: Arm = arm,
                picked: SkillResult = picked,
                dropped: SkillResult | None = dropped,
            ) -> None:
                entry.update(bin_outcome(ctx, obj, arm, picked, dropped, landed))

            assert dropped is not None  # released
            observed = landing(ctx, landing_spec(obj, arm), _landed(dropped))
            observed.advance()
            landings.append((observed, record))
        else:
            entry.update(bin_outcome(ctx, obj, arm, picked, dropped, None, evaluate=evaluate))
            if picked.succeeded and spec is None:
                ended = picked.stage
                entry["failure"] = {**NO_ZONE, "held": ended.held, "attached": ended.attached}
        if not picked.succeeded and picked.stage.reason is Reason.MISSED:
            # Portable evidence that the object was not where it was registered: what was
            # planned ahead is void, and the table is registered again before a retry.
            if pending is not None:
                pending.close()
                pending = None
            register(ctx, on_table(ctx), read)
            registration = ctx.scene.revision
            if attempts[obj] <= retries:
                queue.append(obj)
            continue
        if ctx.scene.attachment(obj) is None:
            continue
        failed = failed_stage(picked, dropped)
        recovery = recover(ctx, obj, arm, failed, selection.candidate.grasp, planner, stepped, read)
        entry["recovery"] = recovery
        if not recovery["released"]:
            break  # still held, and nothing safe to do: the round stops, declared so
        if attempts[obj] <= retries:
            register(ctx, [obj], read)  # where the recovery left it, for the retry
            if _upright_on_worktop(ctx, obj, start):
                queue.append(obj)
    for _ in range(MAX_TICKS):  # the last landings, with nothing else to do
        if not landings:
            break
        ctx.step()
        stepped()
    for observed, recorded in landings:  # outlasted the round: ended, so recorded whole
        observed.cancel(f"not done after {MAX_TICKS} ticks")
        assert observed.result is not None
        recorded(observed.result)
    return outcomes


def sort_all(
    ctx: RobotContext,
    package: RobotPackage,
    tick: Callable[[], None] = lambda: None,
    *,
    read: Read = pose_stub,
    evaluate: bool = True,
) -> dict[str, Any]:
    """The legacy demo's ``sort_all``: recycle every object, racing the arms with
    lookahead (``recycle``), recovering a failed attempt where that is safe (``recover``),
    and racing an object released back onto the worktop once more (``RETRIES``), until
    none is left to try. Returns every attempt, and per object its attempts, whether it is
    still attached, and its state in the registered scene. With ``evaluate``, once every
    landing has ended, also which bin it rests in, if any, from its privileged pose: the
    table is ``cleared`` when every object does. Without it, ``bin`` and ``cleared`` are
    None."""
    objects = sorted(ctx.info.objects)
    attempts = recycle(ctx, package, objects, tick, retries=RETRIES, read=read, evaluate=evaluate)
    final: dict[str, dict[str, Any]] = {}
    for obj in objects:
        final[obj] = {
            "attempts": sum(1 for a in attempts if a["object"] == obj),
            "attached": ctx.scene.attachment(obj) is not None,
            "status": ctx.scene.object_status(obj).value,
        }
        if evaluate:
            pose = true_pose(ctx, obj)
            bins = [Arm(side).bin[0] for side in BIN_OF if _in_bin(pose, Arm(side))]
            final[obj].update(bin=bins[0] if bins else None, final_pose=_pose(pose))
    return {
        "attempts": attempts,
        "objects": final,
        "cleared": all(f["bin"] is not None for f in final.values()) if evaluate else None,
    }


def _drop_racing_next(
    ctx: RobotContext,
    dropping: SkillExecution,
    race_next: Callable[[Execution], Candidates] | None,
    tick: Callable[[], None],
) -> Candidates | None:
    """Drive ``dropping`` to its end; once its transfer is executing, start ``race_next``
    on it. Returns the race, possibly still planning. As ``run_together`` does, while
    every unfinished one waits only on a plan made on a worker and not yet ready, the
    clock does not step: the run's timeline does not depend on how long planning
    takes."""
    pending: Candidates | None = None
    dropping.advance()
    for _ in range(MAX_TICKS):
        if dropping.done:
            break
        active = dropping.active
        if pending is None and race_next is not None and dropping.stage == "transfer" and active:
            pending = race_next(active)
            pending.advance()
        planning = [x.planning for x in (dropping, pending) if x is not None and not x.done]
        if all(planning):
            wait([f for futures in planning for f in futures], return_when=FIRST_COMPLETED)
        else:
            ctx.step()
            tick()
        dropping.advance()
        if pending is not None:
            pending.advance()
    else:
        dropping.cancel(f"not done after {MAX_TICKS} ticks")
    return pending


def runtime(package: RobotPackage, scene: Path) -> MujocoRuntime:
    """Geodude in the staged scene, as the recycling demo runs it."""
    return MujocoRuntime(
        package,
        scene=scene,
        attach_tolerance_m=HOLD_TOLERANCE_M,
        attach_tolerance_rad=SWING_TOLERANCE_RAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Geodude's recycling demo, steps 1-3, on ssrobot.")
    parser.add_argument("--seed", type=int, default=0, help="the scene's seed")
    parser.add_argument(
        "--to", choices=DESTINATIONS, default="bin", help="drop the can in its bin, or place it"
    )
    parser.add_argument(
        "--cans",
        type=int,
        default=1,
        help="how many cans to stage; more than one recycles them all into the bins, racing "
        "the arms and planning each can while the previous one is carried to its bin",
    )
    parser.add_argument(
        "--mix",
        type=int,
        help="stage this many objects of the legacy demo's mix (MIX_KINDS) instead, and sort "
        "them all into the bins (sort_all)",
    )
    parser.add_argument("--viser", action="store_true", help="serve a live view on port 8080")
    parser.add_argument("--record", type=Path, help="write the run as a bundle in this directory")
    args = parser.parse_args(argv)
    if args.cans < 1:
        parser.error("--cans must be at least 1")
    if args.mix is not None and (args.mix < 1 or args.cans != 1 or args.to != "bin"):
        parser.error("--mix takes at least 1 object, into the bins, without --cans")
    if args.cans > 1 and args.to != "bin":
        parser.error("more than one can is recycled into the bins only")
    package = load_installed_package("geodude_assets")
    work = Path(tempfile.mkdtemp(prefix="geodude-"))
    try:
        scene = stage_scene(
            work / "scene",
            package,
            objects={"can": args.cans} if args.mix is None else mix(args.seed, args.mix),
            fixtures=BINS,
            seed=args.seed,
        )
        trace = work / "trace.jsonl"
        sinks: list[Any] = [JsonlTrace(trace)] if args.record else []
        channels = None
        if args.viser:
            from mj_viser.integrations.ssrobot import TraceChannels

            channels = TraceChannels(package.description)  # feeds the viewer's channel panels
            sinks.append(channels)
        with RobotContext(
            package.description, runtime(package, scene), sinks=sinks, object_poses=OBJECT_POSES
        ) as ctx:
            view: Any = None
            if channels is not None:
                from mj_viser import ContactOverlay, LiveView
                from mj_viser.integrations.ssrobot import render_source

                # mj_viser's adapter satisfies its protocol at run time, not for mypy
                # (personalrobotics/mj_viser#62).
                view = LiveView(
                    render_source(ctx),  # type: ignore[arg-type]
                    channels=channels.board,
                    label="Geodude 4.0: recycling",
                )
                view.viewer.add_panel(ContactOverlay())
                print(view.open())

            # Wall time minus simulated time, kept as a real robot keeps it.
            anchor = [time.monotonic() - ctx.now.time_ns / 1e9]

            def tick() -> None:
                if view is not None:
                    # A live view runs as a real robot does: never ahead of real time, and
                    # once held still (the context planning on its own thread, or this
                    # view drawing) it carries on from there instead of catching up, so
                    # every pause shows as long as it lasted.
                    ahead = ctx.now.time_ns / 1e9 - (time.monotonic() - anchor[0])
                    if ahead > 0:
                        time.sleep(ahead)
                    elif ahead < 0:
                        anchor[0] = time.monotonic() - ctx.now.time_ns / 1e9
                    view.refresh()

            if args.mix is not None:
                result = sort_all(ctx, package, tick)
                result["succeeded"] = result["cleared"]
                print(json.dumps({o: f["bin"] for o, f in result["objects"].items()}))
            elif args.cans == 1:
                result = demo(ctx, package, args.to, tick)
                print(json.dumps(result["checks"]))
            else:
                cans = recycle(ctx, package, sorted(ctx.info.objects), tick)
                result = {"cans": cans, "succeeded": all(c["succeeded"] for c in cans)}
                print(json.dumps({c["object"]: c["succeeded"] for c in cans}))
            if view is not None:
                if sys.stdin.isatty():
                    input("Press Enter to close the viewer. ")
                view.close()
        for sink in sinks:
            if isinstance(sink, JsonlTrace):
                sink.close()
        if args.record:
            args.record.mkdir(parents=True, exist_ok=True)
            write_bundle(
                args.record / "bundle", trace=trace, package=package, model="mjcf", scene=scene
            )
            report = {
                "seed": args.seed,
                "sources": sources(),
                "blas_kernel": blas_kernel(),
                "scene": json.loads((scene.parent / "manifest.json").read_text()),
                **result,
            }
            (args.record / "recycling.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"recorded {args.record / 'bundle'} and {args.record / 'recycling.json'}")
        ok = bool(result["succeeded"])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
