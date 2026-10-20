"""The recycling demo on ssrobot and ssgrasping (personalrobotics/geodude#201).

Steps 1 and 2 of the port, in a seeded scene staged from prl_assets: one can, picked by
the right arm with ssgrasping's ``grasp``, and dropped with its ``drop`` into the bin on
that arm's side (step 2, the default) or placed upright on the worktop with its
``place`` (step 1, ``--to worktop``). Perception is the ground-truth stub the demo has
always used: each object's privileged pose channel (ssrobot#101).

Run it, watch it, and record it::

    python -m geodude.recycling --seed 0 --viser --record out/

``--viser`` serves a live view at http://localhost:8080; ``--record`` writes the run as
a bundle (``out/bundle``) that mj_viser replays, and its outcome as ``out/recycling.json``:
the sources it ran from, the seed, every skill stage, and where the can came to rest.
The installed command ``geodude-recycling`` is the same.
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
    DropSpec,
    GraspSpec,
    PlaceSpec,
    SkillExecution,
    SkillResult,
    drop,
    grasp,
    place,
)
from ssrobot import (
    ClockMode,
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
DESTINATIONS = ("bin", "worktop")
UP = (0.0, 0.0, 1.0)
"""The worktop's outward normal."""
CONTACT_N = 8.0
"""The guarded placement's force threshold, measured in personalrobotics/ssgrasping#10:
free motion peaks at 4.4-5.1 N, contact crosses 8 N within one tick."""
SWING_TOLERANCE_RAD = 0.2
"""The attachment monitor's rotation tolerance: a cylinder pinched between two pads
swings about the finger axis as the arm accelerates and settles back."""
PLACE_SLACK_M = 0.03
"""How far from ``TARGET``, in each of x and y, the placement region lets an object's
centre be."""
PLACE_TILT_RAD = 0.05
"""How far from upright a placed object may rest."""
PLACE_TOLERANCE_M = 0.01
"""How far outside its region a placed object may rest and still count: the transfer may
aim at a region's edge, and the arm tracks a plan only to the runtime's goal tolerance
(0.01 rad per joint), a few millimetres at the hand."""
CHECK_ITERATIONS = 300
"""The iteration budget of the planner that checks a follow-on straight move."""
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
        index = {"right": 0, "left": 1}[self.side]
        x, y, z = BINS["recycle_bin"][index]
        return f"recycle_bin_{index}", (x, y, z)


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
    after each step (a viewer refresh, a snapshot)."""
    assert ctx.info.clock_mode is ClockMode.MANUAL
    execution.advance()
    for _ in range(MAX_TICKS):
        if execution.done:
            break
        ctx.step()
        tick()
        execution.advance()
    else:
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
    geometry = meta(kind_of(obj))["geometric_properties"]
    planner, checker = planners
    spec = GraspSpec(
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
    return drive(ctx, grasp(ctx, planner, spec, check_planner=checker), tick)


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
    name, base = arm.bin
    container = meta(kind_of(name))
    height = regions.height(meta(kind_of(obj))["geometric_properties"])
    planner, checker = planners
    spec = DropSpec(
        manipulator=arm.manipulator,
        end_effector=arm.hand,
        gripper=arm.gripper,
        opening_channel=arm.opening,
        object=obj,
        container=name,
        normal=UP,
        regions=[regions.drop_zone(container, base, height)],
        retreat_m=0.08,
        inside=[regions.interior(container, base)],
    )
    return drive(ctx, drop(ctx, planner, spec, check_planner=checker), tick)


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
    """The right arm picks ``can_0`` in a staged scene and drops it in its bin
    (``to="bin"``) or places it at ``TARGET`` on the worktop (``to="worktop"``), with a
    snapshot before and after. Returns the destination's outcome."""
    if to not in DESTINATIONS:
        raise ValueError(f"no destination {to!r}; one of {DESTINATIONS}")
    arm, obj = Arm("right"), "can_0"
    chosen = planners(package, arm)
    ctx.snapshot()
    picked = pick(ctx, obj, arm=arm, planners=chosen, tick=tick)
    after: SkillResult | None = None
    if picked.succeeded and to == "bin":
        after = drop_in_bin(ctx, obj, arm=arm, planners=chosen, tick=tick)
    elif picked.succeeded:
        after = place_on_worktop(ctx, package, obj, TARGET, arm=arm, planners=chosen, tick=tick)
    ctx.snapshot()
    if to == "bin":
        return bin_outcome(ctx, obj, arm, picked, after)
    return worktop_outcome(ctx, package, obj, TARGET, picked, after)


def runtime(package: RobotPackage, scene: Path) -> MujocoRuntime:
    """Geodude in the staged scene, as the recycling demo runs it."""
    return MujocoRuntime(package, scene=scene, attach_tolerance_rad=SWING_TOLERANCE_RAD)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Geodude's recycling demo, steps 1-2, on ssrobot.")
    parser.add_argument("--seed", type=int, default=0, help="the scene's seed")
    parser.add_argument(
        "--to", choices=DESTINATIONS, default="bin", help="drop the can in its bin, or place it"
    )
    parser.add_argument("--viser", action="store_true", help="serve a live view on port 8080")
    parser.add_argument("--record", type=Path, help="write the run as a bundle in this directory")
    args = parser.parse_args(argv)
    package = load_installed_package("geodude_assets")
    work = Path(tempfile.mkdtemp(prefix="geodude-"))
    try:
        scene = stage_scene(
            work / "scene", package, objects={"can": 1}, fixtures=BINS, seed=args.seed
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

            result = demo(ctx, package, args.to, tick)
            print(json.dumps(result["checks"]))
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
