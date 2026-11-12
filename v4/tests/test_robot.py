"""Geodude's robot API drives every public call in a seeded recycling scene, each judged by
the physics (personalrobotics/geodude#209).

Two cans and a sugar box are staged at ``SEED``, all on the right arm's side. One
``Geodude`` drives, in turn:
- ``find_objects`` by anything, by type, and by instance;
- ``pickup("can")``: a type, resolved to whichever can and arm the race selects;
- ``place()``: into the holding arm's own bin, where the can then rests;
- ``robot.right.pickup`` of the other can (an arm chosen by side), then
  ``place("worktop")``: it stands upright where it was picked from;
- ``go_home``: each arm at the package's ``ready``, the one already there not moved
  (ssrobot#253);
- per arm, at rest: the end-effector pose against the hand's pose channel, the wrench
  zeroed by ``tare``, and ``close`` and ``open`` against the opening channel;
- an abort requested from the tick once a can is in the hand: the grasp is cancelled
  (``SkillExecution.cancel``), no recovery runs, and the result reports the can held,
  as ``holding`` does; ``open`` refuses while it is, and every call until
  ``clear_abort``.

The privileged pose of each object (``recycling.true_pose``), never the registered one,
is what the bin and the worktop are judged by. The evidence is ``robot.json``: every call,
what it returned, and where each object ended. Reproduce with
``uv run pytest tests/test_robot.py``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ssgrasping import Reason
from ssrobot import ObservationRequest, Pose, RobotContext, load_installed_package

from geodude import recycling, regions
from geodude.robot import HOME, Geodude, TaskResult
from geodude.scene import meta, physical, stage_scene

SEED = 0
OBJECTS = {"can": 2, "sugar_box": 1}
HOME_TOLERANCE_RAD = 0.01
"""How near ``ready`` each arm's joints come: the runtime's goal tolerance."""
FRAME_TOLERANCE_M = 1e-6


def _call(name: str, result: TaskResult) -> dict[str, Any]:
    return {
        "call": name,
        "succeeded": result.succeeded,
        "object": result.object,
        "arm": result.arm,
        "held": result.held,
        "attached": result.attached,
        "aborted": result.aborted,
        "detail": result.detail,
    }


def _reading(ctx: RobotContext, channel: str) -> tuple[float, ...]:
    value = ctx.observe(ObservationRequest(channels=(channel,))).reading(channel).value
    assert isinstance(value, tuple)
    return tuple(float(v) for v in value)


def _in_bin(pose: Pose, side: str) -> bool:
    name, base = recycling.Arm(side).bin
    region = regions.interior(meta(recycling.kind_of(name)), base)
    local = [pose.position[i] - region.origin.position[i] for i in range(3)]
    return all(lo <= v <= hi for v, (lo, hi) in zip(local, region.position_bounds, strict=True))


def test_robot_api_drives_every_call(artifacts: Path) -> None:
    package = load_installed_package("geodude_assets")
    calls: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as work:
        scene = stage_scene(
            Path(work) / "scene", package, objects=OBJECTS, fixtures=recycling.BINS, seed=SEED
        )
        runtime = recycling.runtime(package, scene)
        with RobotContext(package.description, runtime, object_poses=recycling.OBJECT_POSES) as ctx:
            armed: list[bool] = []

            def tick() -> None:  # the abort, once armed and a can is in a hand
                if armed and robot.holding() is not None:
                    robot.request_abort()

            robot = Geodude(ctx, package, tick=tick)
            everything = robot.find_objects()
            cans = robot.find_objects("can")
            assert everything == ["can_0", "can_1", "sugar_box_0"], everything
            assert cans == ["can_0", "can_1"], cans
            assert robot.find_objects("can_1") == ["can_1"]
            assert robot.holding() is None

            # A type, resolved: the race selects the can and the arm.
            picked = robot.pickup("can")
            calls.append(_call("pickup('can')", picked))
            assert picked and picked.held and picked.attached, picked
            assert picked.object in cans and picked.arm is not None, picked
            first, side = picked.object, picked.arm
            assert robot.holding() == (side, first)
            assert first not in robot.find_objects()

            placed = robot.place()
            calls.append(_call("place()", placed))
            assert placed and not placed.attached and placed.arm == side, placed
            assert robot.holding() is None
            assert first not in robot.find_objects()
            assert _in_bin(recycling.true_pose(ctx, first), side), recycling.true_pose(ctx, first)

            # An arm chosen by side, then the worktop.
            (second,) = robot.find_objects("can")
            origin = recycling.registered_pose(ctx, second).position
            picked = robot.right.pickup(second)
            calls.append(_call(f"right.pickup({second!r})", picked))
            assert picked and picked.arm == "right" and picked.object == second, picked
            placed = robot.place("worktop")
            calls.append(_call("place('worktop')", placed))
            assert placed and not placed.attached, placed
            at = recycling.true_pose(ctx, second)
            off = np.linalg.norm(np.subtract(at.position[:2], origin[:2]))
            assert off <= recycling.PLACE_SLACK_M + recycling.PLACE_TOLERANCE_M, off
            _, rest = physical("can")
            assert regions.tilt(regions.matrix(at), rest) <= recycling.PLACE_TILT_RAD, at

            home = robot.go_home()
            calls.append(_call("go_home()", home))
            assert home, home
            configuration = ctx.description.configuration(HOME)
            ready = dict(
                zip(
                    ctx.description.group(configuration.group).joints,
                    configuration.positions,
                    strict=True,
                )
            )
            for a in robot.arms:
                q = a.joint_positions()
                gap = max(abs(v - ready[j]) for v, j in zip(q, a.joints, strict=True))
                assert gap <= HOME_TOLERANCE_RAD, (a.side, gap)

            # Each arm's surface, against its channels.
            for a in robot.arms:
                pose = a.end_effector_pose()
                channel = _reading(ctx, f"{a.side}_hand_pose")
                assert np.allclose(pose.position, channel[:3], atol=FRAME_TOLERANCE_M), a.side
                assert abs(abs(np.dot(pose.quat_wxyz, channel[3:])) - 1.0) <= 1e-6, a.side
                raw = np.linalg.norm(a.wrench())  # untared: the hand's own weight
                a.tare()
                tared = np.linalg.norm(a.wrench())
                assert raw > 1.0 and tared <= 1e-9, (a.side, raw, tared)
                closed = a.close()
                calls.append(_call(f"{a.side}.close()", closed))
                assert closed and _reading(ctx, a.arm.opening)[0] < 0.05, closed
                opened = a.open()
                calls.append(_call(f"{a.side}.open()", opened))
                assert opened and _reading(ctx, a.arm.opening)[0] > 0.95, opened

            # Abort, once the can is in the hand: cancelled, no recovery, held.
            armed.append(True)
            aborted = robot.pickup("can")
            calls.append(_call("pickup('can'), aborted", aborted))
            assert not aborted and aborted.aborted, aborted
            assert aborted.held is True and aborted.attached and aborted.recovery is None, aborted
            (result,) = aborted.results
            assert result.stage.reason is Reason.CANCELLED, result.stage
            assert robot.holding() == (aborted.arm, second)
            assert aborted.arm is not None
            refused = robot.arm(aborted.arm).open()
            calls.append(_call(f"{aborted.arm}.open(), holding", refused))
            assert not refused and robot.holding() == (aborted.arm, second), refused
            again = robot.go_home()
            calls.append(_call("go_home(), aborted", again))
            assert not again and again.aborted, again
            robot.clear_abort()
            assert not robot.aborted

            ended = {
                name: {
                    "position": list(recycling.true_pose(ctx, name).position),
                    "quat_wxyz": list(recycling.true_pose(ctx, name).quat_wxyz),
                }
                for name in everything
            }
    report = {"seed": SEED, "objects": OBJECTS, "calls": calls, "ended": ended}
    (artifacts / "robot.json").write_text(json.dumps(report, indent=2) + "\n")
