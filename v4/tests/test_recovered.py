"""Geodude recovers from a failed lift and from a carry the object left, each by its rule,
leaving a declared state the physics agrees with (personalrobotics/geodude#210).

A lone can is staged at ``SEED`` and picked by the arm on its side, as ``sort_all`` picks
it. Each scenario makes one stage fail by a plausible cause, then calls ``recover``:
- ``shelf``: a fixed shelf ``SHELF_GAP_M`` above the can's top, less than the grasp's
  0.06 m lift. The hand reaches the can under it and closes, but the lift does not plan
  (``grasp.lift``, ``not_planned``), the can held and attached. ``recover`` releases it
  in place: the fingers open, the can is detached and registered where it rests, and the
  hand backs out along its approach. The can stands where it was picked from.
- ``pulled``: ssgrasping's transition study's pull (its ``test_drop``): ``PULL_N`` down on
  the can for ``PULL_NS``, ``PULL_AFTER_NS`` into its transfer to the bin, well past what
  the grip holds. The opening shows the fingers close past it, so the carry stops
  (``drop.transfer``, ``transition_refuted``, cause ``retained``), the can reported not
  held but still attached. ``recover`` detaches it: nothing is in the hand.

The privileged pose of the can (``recycling.true_pose``), never the declared one, is the
reference each scenario is judged by. The evidence is ``recovered.json``: per scenario,
the failed stage, the recovery, and where the can and the hand ended. Reproduce with
``uv run pytest tests/test_recovered.py``.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from ssgrasping import Reason, drop
from ssrobot import Pose, RobotContext, RobotPackage, load_installed_package

from geodude import outcomes, recycling, regions
from geodude.scene import physical, stage_scene

SEED = 0
CAN = "can_0"
SHELF_GAP_M = 0.05
"""Between the can's top and the shelf: less than the grasp's lift (checked)."""
PULL_N = 200.0
PULL_NS = 100_000_000
PULL_AFTER_NS = 500_000_000
STAND_TOLERANCE_M = 0.005
"""How far the can, released in place, may stand from where it was picked."""


def _top(kind: str) -> float:
    """How far above its frame an object standing as its model file stands it reaches, by
    its collision geometry."""
    shape, _ = physical(kind)
    return float(shape.support(np.array([0.0, 0.0, 1.0])))


def _staged(package: RobotPackage, work: Path, shelf: bool) -> Path:
    scene = stage_scene(
        work / "scene", package, objects={"can": 1}, fixtures=recycling.BINS, seed=SEED
    )
    if shelf:
        (placed,) = json.loads((scene.parent / "manifest.json").read_text())["objects"]
        x, y, z = placed["position"]
        at = z + _top("can") + SHELF_GAP_M + 0.01  # the board is 0.02 m thick
        board = (
            f'<body name="shelf" pos="{x} {y} {at}"><geom type="box" size="0.15 0.15 0.01"/></body>'
        )
        scene.write_text(scene.read_text().replace("</worldbody>", board + "</worldbody>", 1))
    return scene


def _hand(ctx: RobotContext, runtime: Any, arm: recycling.Arm) -> np.ndarray:
    """Where ``arm``'s hand frame (a site) is in the simulation: privileged, as the can's
    pose is."""
    frame = next(e.frame for e in ctx.description.end_effectors if e.name == arm.hand)
    return np.array(runtime._data.site(frame).xpos, dtype=float)


def _scenario(shelf: bool, fail: Callable[..., Any]) -> dict[str, Any]:
    package = load_installed_package("geodude_assets")
    with tempfile.TemporaryDirectory() as work:
        scene = _staged(package, Path(work), shelf)
        runtime = recycling.runtime(package, scene)
        with RobotContext(package.description, runtime, object_poses=recycling.OBJECT_POSES) as ctx:
            recycling.register(ctx, recycling.on_table(ctx))
            start = recycling.true_pose(ctx, CAN)
            arm = recycling.arm_for(recycling.registered_pose(ctx, CAN).position)
            planner, checker = recycling.planners(package, arm)
            spec = recycling.grasp_spec(ctx, CAN, arm)
            assert not shelf or SHELF_GAP_M < spec.lift_m, (SHELF_GAP_M, spec.lift_m)
            failed = fail(ctx, runtime, arm, planner, checker)
            failed_at = _hand(ctx, runtime, arm)
            recovery = recycling.recover(ctx, CAN, arm, failed, spec, planner, lambda: None)
            can = recycling.true_pose(ctx, CAN)
            return {
                "failed": None
                if failed is None
                else {
                    "stage": failed.name,
                    "reason": failed.reason.value,
                    "cause": failed.cause,
                    "held": failed.held,
                    "attached": failed.attached,
                },
                "recovery": {
                    "action": None if recovery.action is None else recovery.action.value,
                    "released": recovery.released,
                    "detail": recovery.detail,
                },
                "attached": ctx.scene.attachment(CAN) is not None,
                "start": {"position": list(start.position), "quat_wxyz": list(start.quat_wxyz)},
                "can": {"position": list(can.position), "quat_wxyz": list(can.quat_wxyz)},
                "hand_at_failure": [float(v) for v in failed_at],
                "hand": [float(v) for v in _hand(ctx, runtime, arm)],
                "approach_m": spec.approach_m,
            }


def _lift_under_shelf(ctx, runtime, arm, planner, checker):  # type: ignore[no-untyped-def]
    picked = recycling.pick(ctx, CAN, arm=arm, planners=(planner, checker))
    return recycling.failed_stage(picked)


def _pulled_in_transfer(ctx, runtime, arm, planner, checker):  # type: ignore[no-untyped-def]
    picked = recycling.pick(ctx, CAN, arm=arm, planners=(planner, checker))
    assert picked.succeeded, picked.stages
    spec = recycling.drop_spec(CAN, arm)
    assert spec is not None
    dropping = drop(ctx, planner, spec, check_planner=checker)
    body = runtime._model.body(CAN).id
    began: list[int] = []

    def pull() -> None:
        if not began and dropping.stage == "transfer" and dropping.active is not None:
            began.append(ctx.now.time_ns)
        since = None if not began else ctx.now.time_ns - began[0]
        pulling = since is not None and PULL_AFTER_NS <= since < PULL_AFTER_NS + PULL_NS
        runtime._data.xfrc_applied[body, 2] = -PULL_N if pulling else 0.0

    dropped = recycling.drive(ctx, dropping, pull)
    runtime._data.xfrc_applied[body, 2] = 0.0
    return recycling.failed_stage(picked, dropped)


def test_failed_stages_recover_by_their_rules(artifacts: Path) -> None:
    report = {
        "shelf": _scenario(True, _lift_under_shelf),
        "pulled": _scenario(False, _pulled_in_transfer),
    }
    (artifacts / "recovered.json").write_text(json.dumps(report, indent=2) + "\n")
    _, rest = physical("can")

    shelf = report["shelf"]
    assert shelf["failed"] is not None, shelf
    assert (shelf["failed"]["stage"], shelf["failed"]["reason"]) == ("lift", Reason.NOT_PLANNED)
    assert shelf["failed"]["held"] and shelf["failed"]["attached"], shelf
    assert shelf["recovery"]["action"] == outcomes.RecoveryAction.RELEASE_IN_PLACE, shelf
    assert shelf["recovery"]["released"] and not shelf["attached"], shelf
    assert shelf["recovery"]["detail"].endswith("backed out: succeeded"), shelf
    start, can = (np.array(shelf[k]["position"]) for k in ("start", "can"))
    assert np.linalg.norm(can - start) <= STAND_TOLERANCE_M, (start, can)
    pose = Pose(position=tuple(can), quat_wxyz=tuple(shelf["can"]["quat_wxyz"]))
    assert regions.tilt(regions.matrix(pose), rest) <= recycling.PLACE_TILT_RAD, shelf
    # Backed out along its approach, by its length, from where it closed.
    backed = np.linalg.norm(np.array(shelf["hand"]) - np.array(shelf["hand_at_failure"]))
    assert backed >= 0.9 * shelf["approach_m"], (backed, shelf["approach_m"])

    pulled = report["pulled"]
    assert pulled["failed"] is not None, pulled
    assert (pulled["failed"]["stage"], pulled["failed"]["reason"], pulled["failed"]["cause"]) == (
        "transfer",
        Reason.TRANSITION_REFUTED,
        "retained",
    ), pulled
    assert pulled["failed"]["held"] is False and pulled["failed"]["attached"], pulled
    assert pulled["recovery"]["action"] == outcomes.RecoveryAction.RELEASE_LOST, pulled
    # Detached after the fingers reached open, not with them left as they were.
    assert pulled["recovery"]["detail"] == "detached: the object had left the hand", pulled
    assert pulled["recovery"]["released"] and not pulled["attached"], pulled
    # The physics agrees nothing is in the hand: the can fell away from it, and rests
    # below where it was picked from, above the floor.
    can = np.array(pulled["can"]["position"])
    assert np.linalg.norm(np.array(pulled["hand"]) - can) > 0.1, pulled
    assert 0.0 < can[2] < np.array(pulled["start"]["position"])[2], pulled
