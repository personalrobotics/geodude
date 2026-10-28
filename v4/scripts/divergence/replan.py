"""Replan one recorded request from its saved snapshot, several times, with no abort.

Usage: replan.py SEED REQUEST_PREFIX RUN [RUN ...]
Prints every result the runs recorded for the request, then each replan's.
"""

import json
import sys
import tempfile
from pathlib import Path

from ssrobot import PlanResult, RobotContext, SceneSnapshot, load_installed_package
from ssrobot._wire import fingerprint, loads

from geodude import recycling
from geodude.scene import stage_scene

seed, prefix, runs = int(sys.argv[1]), sys.argv[2], [Path(p) for p in sys.argv[3:]]
recorded = []
for run in runs:
    for e in map(json.loads, (run / "plans.jsonl").read_text().splitlines()):
        if e["request"].startswith(prefix):
            recorded.append((run.name, e))
for name, e in recorded:
    print(f"recorded {name}: {e['planner']} {e['status']} traj={(e['trajectory'] or '')[:8]}")
name, e = recorded[0]
run = next(r for r in runs if r.name == name)
snapshot = loads((run / "snapshots" / f"{e['materialized']}.json").read_text(), SceneSnapshot)
request = loads((run / "results" / f"{e['result']}.json").read_text(), PlanResult).request
package = load_installed_package("geodude_assets")
arm = recycling.Arm("right" if e["planner"].startswith("right") else "left")
planner = recycling.planners(package, arm)[0 if e["planner"].endswith("planner") else 1]
with tempfile.TemporaryDirectory() as work:
    scene = stage_scene(
        Path(work) / "scene",
        package,
        objects=recycling.mix(seed, 5),
        fixtures=recycling.BINS,
        seed=seed,
    )
    with RobotContext(package.description, recycling.runtime(package, scene)) as ctx:
        pscene = ctx.planning_scene(snapshot, request.group, clearance_m=request.clearance_m)
        for i in range(5):
            r = planner.plan(request, pscene)
            t = None if r.trajectory is None else fingerprint(r.trajectory)[:8]
            print(f"replan {i}: {r.status.value} traj={t}")
