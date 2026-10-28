"""The recycling demo on ssrobot and ssgrasping (personalrobotics/geodude#201).

Steps 1 to 3 of the port, in a seeded scene staged from prl_assets: one can, picked with
ssgrasping's ``grasp`` by the arm on its side of the worktop, and dropped with its
``drop`` into the bin on that arm's side (the default) or placed upright on the worktop
with its ``place`` (``--to worktop``). Perception is the ground-truth stub the demo has
always used: each object's privileged pose channel (ssrobot#101).

Run it, watch it, and record it::

    python -m geodude.recycling --seed 0 --viser --record out/

``--viser`` serves a live view at http://localhost:8080; ``--record`` writes the run as
a bundle (``out/bundle``) that mj_viser replays, and its outcome as ``out/recycling.json``:
the sources it ran from, the seed, every skill stage, and where the can came to rest.
The installed command ``geodude-recycling`` is the same.

``--cans 4`` stages four cans and recycles them all (``recycle``): both arms race for
each, the arm on the can's side preferred, and each can is planned while the previous
can is carried to its bin (personalrobotics/geodude#223).
"""

from __future__ import annotations

import argparse
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
    SkillExecution,
    SkillResult,
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
    ObservationRequest,
    Planner,
    Pose,
    RobotContext,
    RobotPackage,
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
    "prl-assets",
    "geodude-assets",
)
"""The distributions a run's outcome names."""


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


def object_pose(ctx: RobotContext, name: str) -> Pose:
    """An object's pose, from its privileged pose channel: the ground-truth stub."""
    value = ctx.observe(ObservationRequest(channels=(f"{name}.pose",))).readings[0].value
    assert isinstance(value, tuple)
    x, y, z, w, i, j, k = value
    return Pose(position=(x, y, z), quat_wxyz=(w, i, j, k))


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
        pregrasps=regions.pregrasps(geometry, object_pose(ctx, obj)),
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
    return drive(ctx, place(ctx, planner, spec, check_planner=checker), tick)


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
    dropped = drive(ctx, drop(ctx, planner, spec, check_planner=checker), tick)
    if not _released(dropped):
        return dropped, None
    return dropped, drive(ctx, landing(ctx, landing_spec(obj, arm)), tick)


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
    "detail": "no drop zone fits the object as it is held",
    "evidence": "planned",
}
"""The failure recorded when ``held_drop_spec`` finds no zone: nothing was dropped."""


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


def _stages(result: SkillResult | None) -> list[list[Any]] | None:
    return None if result is None else [[s.name, s.succeeded] for s in result.stages]


def _failure(*results: SkillResult | None) -> dict[str, Any] | None:
    """The stage that decided the first of ``results`` that did not succeed: its skill,
    name, detail, and evidence kind. None when every result present succeeded."""
    for result in results:
        if result is not None and not result.succeeded:
            stage = result.stage
            return {
                "skill": result.skill,
                "stage": stage.name,
                "detail": stage.detail,
                "evidence": stage.evidence.value,
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
    within ``PLACE_SLACK_M`` plus ``PLACE_TOLERANCE_M`` of ``target`` in x and y."""
    final = object_pose(ctx, obj)
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
    """Whether an object at ``pose`` has its centre within the prl_assets interior of
    ``arm``'s bin: inside its inner walls, over its floor, below its rim."""
    name, base = arm.bin
    geometry = meta(kind_of(name))["geometric_properties"]
    outer, inner = geometry["outer_size"], geometry["inner_size"]
    local = [pose.position[i] - base[i] for i in range(3)]
    return bool(
        abs(local[0]) <= inner[0] / 2
        and abs(local[1]) <= inner[1] / 2
        and outer[2] - inner[2] <= local[2] <= outer[2]
    )


def bin_outcome(
    ctx: RobotContext,
    obj: str,
    arm: Arm,
    picked: SkillResult,
    dropped: SkillResult | None,
    landed: SkillResult | None,
) -> dict[str, Any]:
    """What a pick, drop, and landing achieved, observed when the landing ends, or the
    drop when it released nothing: every stage, whether the release was confirmed, where
    ``obj`` rests (from its pose channel) relative to the bin's base, and what is still
    attached, which may be the next object. ``succeeded`` requires every stage to
    succeed, ``obj`` no longer attached, and its centre within the bin's prl_assets
    interior: inside its inner walls, over its floor, below its rim."""
    name, base = arm.bin
    final = object_pose(ctx, obj)
    local = [final.position[i] - base[i] for i in range(3)]
    attached = [a.object for a in ctx.scene.attachments]
    checks = {
        "grasped": picked.succeeded,
        "dropped": dropped is not None and dropped.succeeded,
        "released": _released(dropped),
        "landed": landed is not None and landed.succeeded,
        "detached": obj not in attached,
        "in_bin": _in_bin(final, arm),
    }
    return {
        "to": "bin",
        "object": obj,
        "bin": {"name": name, "base": list(base)},
        "grasp": _stages(picked),
        "drop": _stages(dropped),
        "landing": _stages(landed),
        "final_pose": _pose(final),
        "position_in_bin_m": local,
        "attached": attached,
        "checks": checks,
        "failure": _failure(picked, dropped, landed),
        "succeeded": all(checks.values()),
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
    ctx.snapshot()
    arm = arm_for(object_pose(ctx, obj).position)
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
    ctx: RobotContext, obj: str, arms: dict[str, tuple[Planner, Planner]], *, race: bool = True
) -> list[Candidate]:
    """Each arm, as a candidate for ``obj``: its grasp, and its drop into its own bin
    (``BIN_OF``), with that arm's planners. The arm on ``obj``'s side (``arm_for``) is
    preferred (``PREFERRED``); the other is the fallback, taken when the preferred arm's
    prefix does not plan. Without ``race``, only the preferred arm."""
    preferred = arm_for(object_pose(ctx, obj).position).side
    if not race:
        arms = {preferred: arms[preferred]}
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


def expected_rest(obj: str, arm: Arm) -> Pose:
    """Where ``obj``, dropped by ``arm``, is expected to rest: upright on the floor of
    that arm's bin, at its centre. For planning the next can only; never checked."""
    name, (x, y, z) = arm.bin
    geometry = meta(kind_of(name))["geometric_properties"]
    floor = geometry["outer_size"][2] - geometry["inner_size"][2]
    height = regions.height(meta(kind_of(obj))["geometric_properties"])
    return Pose(position=(x, y, z + floor + height / 2), quat_wxyz=(1.0, 0.0, 0.0, 0.0))


RETRIES = 1
"""How many more times ``sort_all`` races an object whose attempt failed but left it
released, upright, back on the worktop: once, so that an evaluation can count successes
at the first attempt and after a retry apart (#201)."""
RELEASE_IN_PLACE = "release in place"
RELEASE_OVER_BIN = "release over the bin"


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


def recover(
    ctx: RobotContext,
    obj: str,
    arm: Arm,
    failure: dict[str, Any] | None,
    spec: GraspSpec,
    planner: Planner,
    tick: Callable[[], None],
) -> dict[str, Any]:
    """After an attempt that failed with ``obj`` still attached, release it where that is
    safe, by the failed stage (``failure``, from ``bin_outcome``), a narrow slice of
    personalrobotics/geodude#210's recovery:
    - **A lift that did not plan** (``grasp.lift``, planning): the arm never moved after
      closing, so the object still rests where it was grasped, on its support. Open,
      detach once the fingers have opened, and back the hand out along its approach by
      ``spec.approach_m``.
    - **A carry the attachment monitor saw the object leave** (``drop.release``): the
      transfer ended with the hand over the bin, in a zone the object, as held, fits
      (``held_drop_spec``). Open there, and detach once the fingers have opened; where
      the object comes to rest is then observed (``landing``).
    - **Anything else:** the object may be suspended, or its support unknown, so nothing
      is done, and it stays declared held.
    The fingers are never opened onto nothing known, and ``obj`` is detached only after
    they reached open. Returns what was done: its ``action`` (None when nothing),
    whether ``obj`` was ``released``, and a ``detail``."""
    skill, stage, detail = (
        ("", "", "") if failure is None else (failure["skill"], failure["stage"], failure["detail"])
    )
    if skill == "grasp" and stage == "lift" and detail.startswith("planning "):
        refused = _release(ctx, obj, arm, spec, tick)
        if refused is not None:
            return {"action": RELEASE_IN_PLACE, "released": False, "detail": refused}
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
    if skill == "drop" and stage == "release" and detail.startswith("attachment monitor"):
        refused = _release(ctx, obj, arm, spec, tick)
        return {
            "action": RELEASE_OVER_BIN,
            "released": refused is None,
            "detail": refused or "released over the bin",
        }
    return {
        "action": None,
        "released": False,
        "detail": f"no safe release after {skill}.{stage}: still declared held",
    }


def _upright_on_worktop(ctx: RobotContext, obj: str, start: Any) -> bool:
    """Whether ``obj`` rests upright, within ``PLACE_TILT_RAD``, at the height it was
    staged at, within ``PLACE_TOLERANCE_M``: back on the worktop, as it began."""
    now = object_pose(ctx, obj)
    began = next(o.pose for o in start.objects if o.name == obj)
    _, x, y, _ = now.quat_wxyz
    tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))
    return tilt <= PLACE_TILT_RAD and abs(now.position[2] - began.position[2]) <= PLACE_TOLERANCE_M


def recycle(
    ctx: RobotContext,
    package: RobotPackage,
    objects: Sequence[str],
    tick: Callable[[], None] = lambda: None,
    *,
    race: bool = True,
    lookahead: bool = True,
    retries: int = 0,
) -> list[dict[str, Any]]:
    """Pick each of ``objects`` in turn and drop it in a bin, racing both arms for each
    (``Candidates``): the arm on its side takes it if its whole planned prefix, through
    the drop into its own bin, plans; otherwise the other arm, if its prefix plans. The
    drop has no retreat. The next object's race is planned while this drop's transfer
    carries the object to its bin, on the transfer's prediction, which expects the object
    released there to rest on the bin's floor (``expected_rest``), the gripper open as it
    started, and every object dropped before it unsettled, since it may still be rolling
    in its bin. The winner's pregrasp then continues from where the transfer ended, as
    soon as the release is confirmed. Where each released object comes to rest is
    observed alongside the objects after it (ssgrasping's ``landing``), and its outcome
    is recorded when that ends; the round ends once every landing has.
    Within the grasp, the approach is planned during the pregrasp. Returns each attempt's
    ``bin_outcome``, with the object's attempt number, the arm and the race.

    An attempt that fails with the object still attached is recovered by ``recover``. If
    nothing is safe, the round stops there, the object declared held. An object released
    back onto the worktop, upright, is raced again from the current state, up to
    ``retries`` more times.

    Without ``race``, the only candidate is ``arm_for``'s arm; without ``lookahead``,
    nothing is planned during motion, and each race is planned on a snapshot when its
    object's turn comes. Both are for comparison."""
    arms = {side: planners(package, Arm(side)) for side in BIN_OF}
    start = ctx.snapshot()
    opened = {  # every gripper starts open; a release opens it again
        side: {j: start.position(j) for j in ctx.description.gripper(Arm(side).gripper).joints}
        for side in BIN_OF
    }
    outcomes: list[dict[str, Any]] = []
    queue = list(objects)
    attempts: dict[str, int] = {}
    looking: Lookahead | None = None
    pending: Candidates | None = None
    released: list[str] = []  # dropped, and possibly still settling in a bin
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
        if pending is None:
            pending = Candidates(ctx, candidates(ctx, obj, arms, race=race), unsettled=released)
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
                    "succeeded": False,
                }
            )
            continue
        arm = Arm(selection.candidate.label)
        planner, checker = arms[arm.side]
        looking = Lookahead(unsettled=tuple(released)) if lookahead else None
        picked = drive(ctx, selection.grasp(ctx, lookahead=looking), stepped)
        dropped = None
        spec = None
        if picked.succeeded:
            raced = selection.candidate.drop
            assert raced is not None  # every candidate drops
            spec = held_drop_spec(ctx, raced, arm)
        if spec is not None:
            following = queue[i + 1] if lookahead and i + 1 < len(queue) else None
            dropping = drop(ctx, planner, spec, check_planner=checker, lookahead=looking)

            def race_next(
                transfer: Execution,
                following: str | None = following,
                obj: str = obj,
                arm: Arm = arm,
            ) -> Candidates:
                assert following is not None
                return Candidates(
                    ctx,
                    candidates(ctx, following, arms, race=race),
                    after=transfer,
                    unsettled=released,
                    releases={obj: expected_rest(obj, arm)},
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
            "lookahead": logs,
        }
        outcomes.append(entry)
        if _released(dropped):

            def record(
                landed: SkillResult,
                entry: dict[str, Any] = entry,
                obj: str = obj,
                arm: Arm = arm,
                picked: SkillResult = picked,
                dropped: SkillResult | None = dropped,
            ) -> None:
                entry.update(bin_outcome(ctx, obj, arm, picked, dropped, landed))

            observed = landing(ctx, landing_spec(obj, arm))
            observed.advance()
            landings.append((observed, record))
        else:
            entry.update(bin_outcome(ctx, obj, arm, picked, dropped, None))
            if picked.succeeded and spec is None:
                entry["failure"] = dict(NO_ZONE)
        if ctx.scene.attachment(obj) is None:
            continue
        recovery = recover(
            ctx, obj, arm, entry["failure"], selection.candidate.grasp, planner, stepped
        )
        entry["recovery"] = recovery
        if not recovery["released"]:
            break  # still held, and nothing safe to do: the round stops, declared so
        if recovery["action"] == RELEASE_OVER_BIN:

            def recovered(
                landed: SkillResult, entry: dict[str, Any] = entry, obj: str = obj, arm: Arm = arm
            ) -> None:
                entry["recovery"]["landing"] = _stages(landed)
                entry["recovery"]["in_bin"] = landed.succeeded

            observed = landing(ctx, landing_spec(obj, arm))
            observed.advance()
            landings.append((observed, recovered))
        elif attempts[obj] <= retries and _upright_on_worktop(ctx, obj, start):
            queue.append(obj)
    for _ in range(MAX_TICKS):  # the last landings, with nothing else to do
        if not landings:
            break
        ctx.step()
        stepped()
    return outcomes


def sort_all(
    ctx: RobotContext, package: RobotPackage, tick: Callable[[], None] = lambda: None
) -> dict[str, Any]:
    """The legacy demo's ``sort_all``: recycle every object, racing the arms with
    lookahead (``recycle``), recovering a failed attempt where that is safe (``recover``),
    and racing an object released back onto the worktop once more (``RETRIES``), until
    none is left to try. Returns every attempt, and per object, once every landing has
    ended, which bin it rests in, if any. The table is ``cleared`` when every object
    does."""
    objects = sorted(ctx.info.objects)
    attempts = recycle(ctx, package, objects, tick, retries=RETRIES)
    final = {}
    for obj in objects:
        pose = object_pose(ctx, obj)
        bins = [Arm(side).bin[0] for side in BIN_OF if _in_bin(pose, Arm(side))]
        final[obj] = {
            "bin": bins[0] if bins else None,
            "attempts": sum(1 for a in attempts if a["object"] == obj),
            "attached": ctx.scene.attachment(obj) is not None,
            "final_pose": _pose(pose),
        }
    return {
        "attempts": attempts,
        "objects": final,
        "cleared": all(f["bin"] is not None for f in final.values()),
    }


def _drop_racing_next(
    ctx: RobotContext,
    dropping: SkillExecution,
    race_next: Callable[[Execution], Candidates] | None,
    tick: Callable[[], None],
) -> Candidates | None:
    """Drive ``dropping`` to its end; once its transfer is executing, start ``race_next``
    on it. Returns the race, possibly still planning."""
    pending: Candidates | None = None
    dropping.advance()
    for _ in range(MAX_TICKS):
        if dropping.done:
            break
        active = dropping.active
        if pending is None and race_next is not None and dropping.stage == "transfer" and active:
            pending = race_next(active)
            pending.advance()
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
        with RobotContext(package.description, runtime(package, scene), sinks=sinks) as ctx:
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
