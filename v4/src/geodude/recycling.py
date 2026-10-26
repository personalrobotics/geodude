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
import importlib.metadata
import json
import math
import shutil
import sys
import tempfile
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
    Lookahead,
    PlaceSpec,
    SkillExecution,
    SkillResult,
    drop,
    grasp,
    place,
    run_together,
)
from ssrobot import (
    ClockMode,
    Execution,
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
CHECK_ITERATIONS = 300
"""The iteration budget of the planner that checks a follow-on straight move."""
PREFERRED = 1
"""The race priority of the arm on an object's side, over the other arm's 0. Raced at
equal priority, the shorter prefix won, and it was often the other arm's: a long
cross-body carry, from which cans slipped out of the grasp (personalrobotics/geodude#223)."""
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
        held_opening=(0.2, 0.95),
        lift_m=0.06,
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
) -> SkillResult:
    """Drop the held ``obj`` into the bin on ``arm``'s side, from the legacy demo's drop
    zone over it, and check it came to rest inside."""
    planner, checker = planners
    spec = drop_spec(obj, arm)
    return drive(ctx, drop(ctx, planner, spec, check_planner=checker), tick)


def drop_spec(obj: str, arm: Arm) -> DropSpec:
    """``drop_in_bin``'s drop of ``obj`` by ``arm`` into its bin."""
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
        inside=[regions.interior(container, base)],
        carry_scaling=CARRY_SCALING,
    )


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


def bin_outcome(
    ctx: RobotContext, obj: str, arm: Arm, picked: SkillResult, dropped: SkillResult | None
) -> dict[str, Any]:
    """What a pick and drop achieved, observed after it ends: every stage, whether the
    release was confirmed, where ``obj`` rests (from its pose channel) relative to the
    bin's base, and what is still attached. ``succeeded`` requires every stage to
    succeed, nothing attached, and the object's centre within the bin's prl_assets
    interior: inside its inner walls, over its floor, below its rim."""
    name, base = arm.bin
    geometry = meta(kind_of(name))["geometric_properties"]
    outer, inner = geometry["outer_size"], geometry["inner_size"]
    final = object_pose(ctx, obj)
    local = [final.position[i] - base[i] for i in range(3)]
    attached = [a.object for a in ctx.scene.attachments]
    checks = {
        "grasped": picked.succeeded,
        "dropped": dropped is not None and dropped.succeeded,
        "released": _released(dropped),
        "nothing_attached": not attached,
        "in_bin": abs(local[0]) <= inner[0] / 2
        and abs(local[1]) <= inner[1] / 2
        and outer[2] - inner[2] <= local[2] <= outer[2],
    }
    return {
        "to": "bin",
        "object": obj,
        "bin": {"name": name, "base": list(base)},
        "grasp": _stages(picked),
        "drop": _stages(dropped),
        "final_pose": _pose(final),
        "position_in_bin_m": local,
        "attached": attached,
        "checks": checks,
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
    if picked.succeeded and to == "bin":
        after = drop_in_bin(ctx, obj, arm=arm, planners=chosen, tick=tick)
    elif picked.succeeded:
        after = place_on_worktop(ctx, package, obj, TARGET, arm=arm, planners=chosen, tick=tick)
    ctx.snapshot()
    if to == "bin":
        result = bin_outcome(ctx, obj, arm, picked, after)
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


def landing(obj: str, arm: Arm) -> Pose:
    """Where ``obj``, dropped by ``arm``, is expected to rest: upright on the floor of
    that arm's bin, at its centre. For planning the next can only; never checked."""
    name, (x, y, z) = arm.bin
    geometry = meta(kind_of(name))["geometric_properties"]
    floor = geometry["outer_size"][2] - geometry["inner_size"][2]
    height = regions.height(meta(kind_of(obj))["geometric_properties"])
    return Pose(position=(x, y, z + floor + height / 2), quat_wxyz=(1.0, 0.0, 0.0, 0.0))


def recycle(
    ctx: RobotContext,
    package: RobotPackage,
    objects: Sequence[str],
    tick: Callable[[], None] = lambda: None,
    *,
    race: bool = True,
    lookahead: bool = True,
) -> list[dict[str, Any]]:
    """Pick each of ``objects`` in turn and drop it in a bin, racing both arms for each
    (``Candidates``): the arm on its side takes it if its whole planned prefix, through
    the drop into its own bin, plans; otherwise the other arm, if its prefix plans. The
    drop has no retreat. The next object's race is planned while this drop's transfer
    carries the object to its bin, on the transfer's prediction, which expects the object
    released there to rest on the bin's floor (``landing``), the gripper open as it
    started, and every object dropped before it unsettled, since it may still be rolling
    in its bin. The winner's pregrasp then continues from where the transfer ended, once
    the release is confirmed and the object has landed.
    Within the grasp, the approach is planned during the pregrasp. Returns each object's
    ``bin_outcome``, with the arm and the race.

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
    looking: Lookahead | None = None
    pending: Candidates | None = None
    released: list[str] = []  # dropped, and possibly still settling in a bin
    for i, obj in enumerate(objects):
        if pending is None:
            pending = Candidates(ctx, candidates(ctx, obj, arms, race=race), unsettled=released)
        run_together(ctx, [pending], max_ticks=MAX_TICKS, tick=tick)  # blocks; no steps
        selection, report = pending.selection, pending.report
        pending = None
        if selection is None:
            outcomes.append({"object": obj, "arm": None, "race": report, "succeeded": False})
            continue
        arm = Arm(selection.candidate.label)
        planner, checker = arms[arm.side]
        looking = Lookahead(unsettled=tuple(released)) if lookahead else None
        picked = drive(ctx, selection.grasp(ctx, lookahead=looking), tick)
        dropped = None
        if picked.succeeded:
            following = objects[i + 1] if lookahead and i + 1 < len(objects) else None
            spec = selection.candidate.drop
            assert spec is not None  # every candidate drops
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
                    releases={obj: landing(obj, arm)},
                    joints=opened[arm.side],
                )

            pending = _drop_racing_next(ctx, dropping, race_next if following else None, tick)
            assert dropping.result is not None
            dropped = dropping.result
            released.append(obj)
            if pending is not None and not _released(dropped):
                run_together(ctx, [pending], max_ticks=MAX_TICKS)  # its planners are free again
                pending = None  # planned on a release that did not happen: race again
        outcome = bin_outcome(ctx, obj, arm, picked, dropped)
        logs = {
            skill: result.values.get("lookahead", [])
            for skill, result in (("grasp", picked), ("drop", dropped))
            if result is not None
        }
        outcomes.append({"arm": arm.side, "race": report, "lookahead": logs, **outcome})
        if ctx.scene.attachment(obj) is not None:
            break  # still held: recovery is #209/#210's
    return outcomes


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
    parser.add_argument("--viser", action="store_true", help="serve a live view on port 8080")
    parser.add_argument("--record", type=Path, help="write the run as a bundle in this directory")
    args = parser.parse_args(argv)
    package = load_installed_package("geodude_assets")
    work = Path(tempfile.mkdtemp(prefix="geodude-"))
    try:
        scene = stage_scene(
            work / "scene", package, objects={"can": args.cans}, fixtures=BINS, seed=args.seed
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

            def tick() -> None:
                if view is not None:
                    view.refresh()

            if args.cans == 1:
                result = demo(ctx, package, args.to, tick)
                print(json.dumps(result["checks"]))
            else:
                if args.to != "bin":
                    parser.error("more than one can is recycled into the bins only")
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
