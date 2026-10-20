"""The recycling demo on ssrobot and ssgrasping (personalrobotics/geodude#201).

Step 1 of the port: one object picked by one arm and placed upright on the worktop,
through ssgrasping's ``grasp`` and ``place``, in a seeded scene staged from prl_assets.
Perception is the ground-truth stub the demo has always used: each object's privileged
pose channel (ssrobot#101).

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
from ssgrasping import GraspSpec, PlaceSpec, SkillExecution, SkillResult, grasp, place
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
"""The legacy demo's bins, either side of the robot, on the floor."""
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


def pick_and_place(
    ctx: RobotContext,
    package: RobotPackage,
    obj: str,
    target: tuple[float, float],
    *,
    arm: Arm,
    tick: Callable[[], None] = lambda: None,
) -> tuple[SkillResult, SkillResult | None]:
    """Grasp ``obj`` with ``arm`` from every sstsr grasp of its prl_assets geometry, and
    place it upright on the worktop within ``PLACE_SLACK_M`` of ``target`` in x and y."""
    geometry = meta(kind_of(obj))["geometric_properties"]
    planner, checker = planners(package, arm)
    surface = worktop(package).centre[2]
    picked = drive(
        ctx,
        grasp(
            ctx,
            planner,
            GraspSpec(
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
            ),
            check_planner=checker,
        ),
        tick,
    )
    if not picked.succeeded:
        return picked, None
    placed = drive(
        ctx,
        place(
            ctx,
            planner,
            PlaceSpec(
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
            ),
            check_planner=checker,
        ),
        tick,
    )
    return picked, placed


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


def outcome(
    ctx: RobotContext,
    package: RobotPackage,
    obj: str,
    target: tuple[float, float],
    picked: SkillResult,
    placed: SkillResult | None,
) -> dict[str, Any]:
    """What a pick-and-place achieved, observed after it ends: every stage, whether the
    release was confirmed, where ``obj`` rests (from its pose channel), how far its axis
    is from upright, its base's height above the worktop, and what is still attached.
    ``succeeded`` requires every stage to succeed, nothing attached, and the object
    upright within ``PLACE_TILT_RAD``, its base on the worktop, and its centre within
    ``PLACE_SLACK_M`` plus ``PLACE_TOLERANCE_M`` of ``target`` in x and y."""
    final = object_pose(ctx, obj)
    _, x, y, _ = final.quat_wxyz
    tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))  # body z from world z
    offset = [final.position[i] - target[i] for i in range(2)]
    height = regions.height(meta(kind_of(obj))["geometric_properties"])
    base = final.position[2] - math.cos(tilt) * height / 2 - worktop(package).centre[2]
    attached = [a.object for a in ctx.scene.attachments]
    released = placed is not None and any(
        s.name == "release" and s.succeeded for s in placed.stages
    )
    checks = {
        "grasped": picked.succeeded,
        "placed": placed is not None and placed.succeeded,
        "released": released,
        "nothing_attached": not attached,
        "upright": tilt <= PLACE_TILT_RAD,
        "on_worktop": abs(base) <= PLACE_TOLERANCE_M,
        "at_target": max(map(abs, offset)) <= PLACE_SLACK_M + PLACE_TOLERANCE_M,
    }
    return {
        "object": obj,
        "target": list(target),
        "grasp": [[s.name, s.succeeded] for s in picked.stages],
        "place": None if placed is None else [[s.name, s.succeeded] for s in placed.stages],
        "final_pose": {"position": list(final.position), "quat_wxyz": list(final.quat_wxyz)},
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


def demo(
    ctx: RobotContext, package: RobotPackage, tick: Callable[[], None] = lambda: None
) -> dict[str, Any]:
    """Step 1 in a staged scene: the right arm moves ``can_0`` to ``TARGET``, with a
    snapshot before and after. Returns its ``outcome``."""
    ctx.snapshot()
    picked, placed = pick_and_place(ctx, package, "can_0", TARGET, arm=Arm("right"), tick=tick)
    ctx.snapshot()
    return outcome(ctx, package, "can_0", TARGET, picked, placed)


def runtime(package: RobotPackage, scene: Path) -> MujocoRuntime:
    """Geodude in the staged scene, as the recycling demo runs it."""
    return MujocoRuntime(package, scene=scene, attach_tolerance_rad=SWING_TOLERANCE_RAD)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Geodude's recycling demo, step 1, on ssrobot.")
    parser.add_argument("--seed", type=int, default=0, help="the scene's seed")
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

            result = demo(ctx, package, tick)
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
