"""Cross-validate the diverging plans of two diagnostic rounds, without executing them.

Usage: crossval.py SEED RUN_A ORDER_A RUN_B ORDER_B
Loads each run's plan (by its order in plans.jsonl): its request, result and snapshot.
Prints how the two snapshots differ, then validates each run's start state, and each
successful trajectory, in the other run's snapshot, with the same planner and validator.
"""

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from ssrobot import PlanResult, RobotContext, SceneSnapshot, load_installed_package
from ssrobot._wire import loads

from geodude import recycling
from geodude.scene import stage_scene

seed = int(sys.argv[1])
runs = [(Path(sys.argv[2]), int(sys.argv[3])), (Path(sys.argv[4]), int(sys.argv[5]))]


def load(run: Path, order: int):
    entry = next(
        e
        for e in map(json.loads, (run / "plans.jsonl").read_text().splitlines())
        if e["order"] == order
    )
    snapshot = loads(
        (run / "snapshots" / f"{entry['materialized']}.json").read_text(), SceneSnapshot
    )
    result = loads((run / "results" / f"{entry['result']}.json").read_text(), PlanResult)
    return entry, snapshot, result


(ea, sa, ra), (eb, sb, rb) = (load(*r) for r in runs)
print(
    "A",
    ea["planner"],
    ea["status"],
    "t",
    sa.stamp.time_ns / 1e9,
    "| B",
    eb["planner"],
    eb["status"],
    "t",
    sb.stamp.time_ns / 1e9,
)
pa, pb = np.array(sa.positions), np.array(sb.positions)
diff = np.abs(pa - pb)
for i in np.argsort(-diff)[:8]:
    if diff[i] > 1e-6:
        print(f"  joint {sa.joints[i]:45} A {pa[i]: .6f} B {pb[i]: .6f} |d| {diff[i]:.6f}")
for oa, ob in zip(sa.objects, sb.objects, strict=False):
    d = np.linalg.norm(np.array(oa.pose.position) - np.array(ob.pose.position))
    if d > 1e-5:
        print(f"  object {oa.name:20} moved {d * 1000:.2f} mm between the snapshots")
print(
    "  attachments A",
    [(a.object, a.held) for a in sa.attachments],
    "B",
    [(a.object, a.held) for a in sb.attachments],
)

package = load_installed_package("geodude_assets")
side = ea["planner"].split(":")[0]
arm = recycling.Arm("right" if side.startswith("right") else "left")
planner = recycling.planners(package, arm)[0 if ea["planner"].endswith("planner") else 1]
with tempfile.TemporaryDirectory() as work:
    scene = stage_scene(
        Path(work) / "scene",
        package,
        objects=recycling.mix(seed, 5),
        fixtures=recycling.BINS,
        seed=seed,
    )
    with RobotContext(package.description, recycling.runtime(package, scene)) as ctx:
        for name, (_e, r), other in (("A", (ea, ra), sb), ("B", (eb, rb), sa)):
            req = replace(r.request, snapshot=other.fingerprint(), revision=other.revision)
            pscene = ctx.planning_scene(other, req.group, clearance_m=req.clearance_m)
            validator = planner.validator(req, pscene)
            start = validator.validate_state(r.request.start)
            witness = start.witness
            print(
                f"{name}'s start in the other snapshot: valid={start.valid}"
                + (
                    ""
                    if witness is None
                    else f" ({witness.failure}, {getattr(witness, 'detail', '')})"
                )
            )
            contacts = sorted({f"{c.first}/{c.second}" for c in pscene.contacts(r.request.start)})
            print(f"   contacts at that start: {contacts}")
            own = ctx.planning_scene(
                sa if name == "A" else sb, req.group, clearance_m=req.clearance_m
            )
            print(
                f"   contacts at that start in its own snapshot: "
                f"{sorted({f'{c.first}/{c.second}' for c in own.contacts(r.request.start)})}"
            )
            if r.trajectory is not None:
                v = validator.validate_trajectory(r.trajectory, path=r.path)
                print(
                    f"   its trajectory in the other snapshot: valid={v.valid}"
                    + ("" if v.witness is None else f" ({v.witness.failure})")
                )
