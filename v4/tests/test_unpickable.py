"""A round goes on past an object the hand cannot grasp or drop (#260).

Two of the objects prl_assets measures grasp regions for (prl_assets#52) defeat the
2F-140 in opposite ways. The large marker, a 19.5 mm cylinder, has a region sstsr gives
no grasp within the hand's limits: it is recorded ``no_grasp`` and never raced, rather
than raising when its race is built. The bleach bottle is 251 mm tall against the bin's
0.238 m opening, and its body region is gripped on its faces, so held it lies across the
opening and no drop zone fits it: it is put back on the worktop where it was picked from
(``PLACE_BACK``) and the round goes on, rather than stopping with it held.

``sort_all`` stages a can beside the two at ``SEED`` and sorts them. The test requires:
- the can lands in a bin;
- the marker's one attempt is ``no_grasp``, with no arm and no race;
- the bleach bottle's one attempt fails with ``no_drop_zone`` and is recovered by
  placing it back, released, and it is not raced again (held so again, it would not fit
  again); it ends on the worktop, standing as its model file stands it, nothing
  attached;
- the outcome encodes, decodes and validates (``RecyclingRun``), and the corpus
  evaluator's verdicts name each mechanism.

The evidence is ``unpickable.json``: the round's ``RecyclingRun`` and the evaluator's
verdicts. Reproduce with ``uv run pytest tests/test_unpickable.py``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ssrobot import RobotContext, dumps, load_installed_package, loads

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
            bleach = recycling.true_pose(ctx, "bleach_cleanser_0")
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
    (attempt,) = tried["bleach_cleanser_0"]  # placed back, and not raced again
    assert attempt.failure is not None and attempt.failure.reason == outcomes.NO_DROP_ZONE
    assert attempt.recovery is not None and attempt.recovery.released, attempt.recovery
    assert attempt.recovery.action is outcomes.RecoveryAction.PLACE_BACK
    final = by_object["bleach_cleanser_0"]
    assert final.bin is None and not final.attached, final
    _, rest = physical("bleach_cleanser")
    assert regions.tilt(regions.matrix(bleach), rest) <= recycling.PLACE_TILT_RAD
    assert verdicts["bleach_cleanser_0"] == "drop.zone", verdicts
