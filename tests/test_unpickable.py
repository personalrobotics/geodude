"""A round goes on past an object the hand cannot grasp, and drops one it holds from the
side upright (#260).

Two of the objects prl_assets measures grasp regions for (prl_assets#52) were defeated by
the 2F-140. The large marker, a 19.5 mm cylinder, has a region sstsr gives no grasp
within the hand's limits: it is recorded ``no_grasp`` and never raced, rather than
raising when its race is built. The bleach bottle is 251 mm tall against the bin's
0.238 m opening, and its body region is gripped on its faces. Held so, a hand pointing
down laid it across the opening, so no drop zone fitted it and it was put back. A drop is
now a placement released in the air: regions for the object, its resting orientations
over the opening (sstsr's ``OpeningPlacer``), composed with how it is held. Held from the
side, it is released standing, the hand level, and falls into the bin.

``sort_all`` stages a can beside the two at ``SEED`` and sorts them. The test requires:
- the can lands in a bin;
- the marker's one attempt is ``no_grasp``, with no arm and no race;
- the bleach bottle is recycled at its first attempt, nothing attached;
- the outcome encodes, decodes and validates (``RecyclingRun``), and the corpus
  evaluator's verdicts name each mechanism.

``test_a_held_object_with_no_release_is_put_back`` exercises the recovery a round now
reaches only when the race's prediction of the hold and the hold itself disagree: the
bleach bottle, picked, is dropped toward a bin moved out of reach, so its drop's
transfer does not plan, and ``recover`` puts it back where it was picked from, released,
standing as its model file stands it.

The evidence is ``unpickable.json``: the round's ``RecyclingRun`` and the evaluator's
verdicts. Reproduce with ``uv run pytest tests/test_unpickable.py``.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

from ssgrasping import Reason, drop
from ssrobot import Pose, RobotContext, dumps, load_installed_package, loads

from geodude import outcomes, recycling, regions
from geodude.scene import physical, stage_scene
from tests.test_sort_all import _evaluator

SEED = 0
KINDS = {"can": 1, "large_marker": 1, "bleach_cleanser": 1}


def test_a_round_goes_on_past_objects_it_cannot_grasp_or_drop(artifacts: Path) -> None:
    package = load_installed_package("geodude_assets")
    with tempfile.TemporaryDirectory() as work:
        scene = stage_scene(
            Path(work) / "scene", package, objects=KINDS, fixtures=recycling.BINS, seed=SEED
        )
        with RobotContext(
            package.description,
            recycling.runtime(package, scene),
            object_poses=recycling.OBJECT_POSES,
        ) as ctx:
            attempts, objects = recycling.sort_all(ctx, package)
        run = recycling.outcome(SEED, scene, None, attempts, objects)
    run = loads(dumps(run).encode(), outcomes.RecyclingRun)  # validates on decode
    evaluator = _evaluator()
    by_object = {d.object: d for d in run.objects}
    tried = {name: [a for a in run.attempts if a.object == name] for name in by_object}
    verdicts = {name: evaluator.verdict(tried[name], d) for name, d in by_object.items()}
    report = json.loads(dumps(run))
    (artifacts / "unpickable.json").write_text(
        json.dumps({"verdicts": verdicts, "run": report}, indent=2) + "\n"
    )

    assert by_object["can_0"].bin is not None, verdicts
    (marker,) = tried["large_marker_0"]
    assert marker.arm is None and marker.race is None and marker.failure is not None
    assert marker.failure.reason == outcomes.NO_GRASP, marker.failure
    assert verdicts["large_marker_0"] == "grasp.regions:no_grasp", verdicts
    (attempt,) = tried["bleach_cleanser_0"]
    assert attempt.succeeded and attempt.failure is None, attempt
    final = by_object["bleach_cleanser_0"]
    assert final.bin is not None and not final.attached, final
    assert verdicts["bleach_cleanser_0"] == "recycled", verdicts


def test_a_held_object_with_no_release_is_put_back(artifacts: Path) -> None:
    package = load_installed_package("geodude_assets")
    with tempfile.TemporaryDirectory() as work:
        scene = stage_scene(
            Path(work) / "scene", package, objects=KINDS, fixtures=recycling.BINS, seed=SEED
        )
        with RobotContext(
            package.description,
            recycling.runtime(package, scene),
            object_poses=recycling.OBJECT_POSES,
        ) as ctx:
            obj = "bleach_cleanser_0"
            recycling.register(ctx, recycling.on_table(ctx))
            origin = recycling.registered_pose(ctx, obj).position[:2]
            arm = recycling.arm_for(recycling.registered_pose(ctx, obj).position)
            planner, checker = recycling.planners(package, arm)
            spec = recycling.grasp_spec(ctx, obj, arm)
            picked = recycling.pick(ctx, obj, arm=arm, planners=(planner, checker))
            assert picked.succeeded, picked.stages
            drop_spec = recycling.drop_spec(obj, arm)
            assert drop_spec is not None
            away = [  # the bin's regions, two metres out of reach
                dataclasses.replace(
                    r,
                    origin=Pose(
                        position=(r.origin.position[0] + 2.0, *r.origin.position[1:]),
                        quat_wxyz=r.origin.quat_wxyz,
                    ),
                )
                for r in drop_spec.regions
            ]
            dropping = drop(
                ctx, planner, dataclasses.replace(drop_spec, regions=away), check_planner=checker
            )
            dropped = recycling.drive(ctx, dropping, lambda: None)
            failed = recycling.failed_stage(picked, dropped)
            recovery = recycling.recover(
                ctx,
                obj,
                arm,
                failed,
                spec,
                planner,
                lambda: None,
                package=package,
                checker=checker,
                origin=(float(origin[0]), float(origin[1])),
            )
            bleach = recycling.true_pose(ctx, obj)
            attached = ctx.scene.attachment(obj)
    report = {
        "failed": None if failed is None else [failed.name, failed.reason, failed.held],
        "recovery": [recovery.action, recovery.released, recovery.detail],
    }
    (artifacts / "put_back.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    assert failed is not None and (failed.name, failed.reason) == ("transfer", Reason.NOT_PLANNED)
    assert failed.held and failed.attached, failed
    assert recovery.action is outcomes.RecoveryAction.PLACE_BACK and recovery.released, recovery
    assert attached is None
    _, rest = physical("bleach_cleanser")
    assert regions.tilt(regions.matrix(bleach), rest) <= recycling.PLACE_TILT_RAD
