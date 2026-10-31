"""When an object dropped into Geodude's bin counts as at rest (personalrobotics/ssgrasping#35).

``landing`` waits until a released object is observed at rest, by criteria the
application declares (``recycling.LANDING_REST``), and fails after a timeout
(``recycling.LANDING_TIMEOUT_NS``). This script measures both on today's model, away from
everything but the release itself: no trial plans a pick or a carry, so a grasp or a plan
that fails never censors one.

The variation is declared, not measured: an object moving less than ``TRANSLATION_M`` along
every axis and ``ROTATION_RAD`` in orientation is not moving in any way the landing check
cares about. The trials validate it and measure the dwell.

**Holds.** For each of the nine legacy types (``recycling.MIX_KINDS``), one object is staged
on the worktop at a seed of this calibration's own (6000 and up: no corpus uses them) and
picked with the demo's skills, as it holds it before every drop: its pose in the hand, and
every joint of the gripper. A type tries up to ``HOLD_SEEDS`` seeds.

**Trials.** Each starts, in its own process, from a prescribed state: the arm at an IK
solution for a release pose over its bin, the gripper's joints as the hold left them, and
the object in the hand at the hold's pose (``MujocoRuntime(start=...)``). The release poses
are the centre, the middle of each edge and each corner of every drop zone the demo releases
from (``regions.held_drop_zones``), where walls and rims are struck. The gripper closes with
a real command, the attachment is declared, and then the demo's own release opens the
gripper and detaches (``recycling._release``). From the release, the object's pose channel
is read every control tick:
- until it has stayed within the variation for ``HOLD_NS``, or for ``CAP_NS`` at most;
- the dwell is one tick longer than the longest interval, judged from the release, that any
  trial stayed within the variation and then moved: a bounce's apex, a rock, a slowing roll.

**Verification.** Every trial is run again with the criteria and a real ``ObjectRest`` watch
started at the release, as ``landing`` starts one. It passes when the watch comes to rest and
the object stays within the variation of the window that established it for ``HOLD_NS`` after.

**Timeout.** ``recycling.LANDING_TIMEOUT_NS`` is twice the slowest observed rest after the
release, in whole seconds: an operational failure bound, never the observed maximum. The
artifact reports the slowest rest, the timeout, and the headroom between them.

    OPENBLAS_CORETYPE=Haswell uv run python scripts/calibrate_landing.py out/landing-rest.json

The artifact records every hold and trial, the maxima by type and overall, the criteria, the
commit and whether the tree was clean, the sources and the OpenBLAS kernel.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

TRANSLATION_M = 1e-3
"""Declared: how far along any axis an object at rest may move."""
ROTATION_RAD = 0.02
"""Declared: how far an object at rest may turn."""
HOLD_NS = 5_000_000_000
"""How long an object must stay within the variation to count as having come to rest."""
CAP_NS = 60_000_000_000
"""How long after the release a trial is followed at most."""
HOLD_SEEDS = range(6000, 6010)
JOBS = 16
"""Trials run at once, each in its own process."""


def _within(run: list[tuple[int, list[float]]]) -> bool:
    """Whether ``run``'s poses stay within the variation, as ssrobot's rest watch judges a
    pose channel: each axis's range, and the angle from the first orientation."""
    for i in range(3):
        values = [v[i] for _, v in run]
        if max(values) - min(values) > TRANSLATION_M:
            return False
    first = run[0][1][3:]
    for _, v in run[1:]:
        dot = abs(sum(a * b for a, b in zip(first, v[3:], strict=True)))
        if 2 * math.acos(min(1.0, dot)) > ROTATION_RAD:
            return False
    return True


def hold(kind: str) -> dict[str, Any]:
    """How the demo holds one ``kind`` after its grasp: the arm, the object's pose in the
    hand, and every gripper joint; or why no seed gave one."""
    from ssrobot import RobotContext, load_installed_package

    from geodude import recycling
    from geodude.scene import stage_scene

    package = load_installed_package("geodude_assets")
    tried = []
    for seed in HOLD_SEEDS:
        with tempfile.TemporaryDirectory() as work:
            scene = stage_scene(
                Path(work) / "scene",
                package,
                objects={kind: 1},
                fixtures=recycling.BINS,
                seed=seed,
            )
            with RobotContext(package.description, recycling.runtime(package, scene)) as ctx:
                (obj,) = sorted(o for o in ctx.info.objects if not o.startswith("recycle_bin"))
                arm = recycling.arm_for(recycling.object_pose(ctx, obj).position)
                picked = recycling.pick(
                    ctx, obj, arm=arm, planners=recycling.planners(package, arm)
                )
                if not picked.succeeded:
                    tried.append(
                        {"seed": seed, "stages": [[s.name, s.succeeded] for s in picked.stages]}
                    )
                    continue
                attachment = ctx.scene.attachment(obj)
                assert attachment is not None  # a grasp that succeeded left it attached
                snapshot = ctx.snapshot()
                joints = ctx.description.component_joints(arm.gripper)
                transform = attachment.transform
                return {
                    "kind": kind,
                    "seed": seed,
                    "arm": arm.side,
                    "transform": [list(transform.position), list(transform.quat_wxyz)],
                    "gripper": {j: snapshot.position(j) for j in joints},
                    "tried": tried,
                }
    return {"kind": kind, "seed": None, "tried": tried}


def releases(held: dict[str, Any]) -> list[dict[str, Any]]:
    """Every prescribed release of ``held``: each drop zone's centre, edge middles, and
    corners, as the hand's pose in the world."""
    import numpy as np
    from ssrobot import Pose

    from geodude import recycling, regions
    from geodude.scene import meta

    arm = recycling.Arm(held["arm"])
    name, base = arm.bin
    position, quat = held["transform"]
    zones = regions.held_drop_zones(
        meta(recycling.kind_of(name)),
        base,
        meta(held["kind"])["geometric_properties"],
        Pose(position=tuple(position), quat_wxyz=tuple(quat)),
        recycling.CLEARANCE_M,
    )
    out = []
    for z, zone in enumerate(zones):
        (x0, x1), (y0, y1), _ = zone.position_bounds
        for fx, fy in [
            (0.5, 0.5),
            (0, 0.5),
            (1, 0.5),
            (0.5, 0),
            (0.5, 1),
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]:
            shift = np.eye(4)
            shift[:2, 3] = (x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))
            hand = regions.matrix(zone.origin) @ shift @ regions.matrix(zone.offset)
            pose = regions.pose(hand)
            out.append(
                {
                    "zone": z,
                    "at": [fx, fy],
                    "hand": [list(pose.position), list(pose.quat_wxyz)],
                }
            )
    return out


def trial(held: dict[str, Any], release: dict[str, Any], dwell_ns: int | None) -> dict[str, Any]:
    """One release of ``held`` from ``release``, in this process. Without ``dwell_ns``,
    the object's pose from the release until it has rested for ``HOLD_NS``; with it, a
    rest watch by the criteria, and the pose for ``HOLD_NS`` after it ends."""
    import numpy as np
    from ssrobot import (
        GripperCommand,
        ObjectRest,
        ObservationRequest,
        Pose,
        RestCriteria,
        RestOutcome,
        RobotContext,
        load_installed_package,
    )
    from ssrobot.mujoco import MujocoRuntime
    from ssrobot.planning import SsikKinematics

    from geodude import recycling, regions
    from geodude.scene import Placement, _write

    package = load_installed_package("geodude_assets")
    description = package.description
    arm = recycling.Arm(held["arm"])
    kind = held["kind"]
    obj = f"{kind}_0"
    out: dict[str, Any] = {"kind": kind, "zone": release["zone"], "at": release["at"]}
    hand = Pose(position=tuple(release["hand"][0]), quat_wxyz=tuple(release["hand"][1]))
    position, quat = held["transform"]
    in_hand = Pose(position=tuple(position), quat_wxyz=tuple(quat))
    world = regions.pose(regions.matrix(hand) @ regions.matrix(in_hand))
    with tempfile.TemporaryDirectory() as work:
        bins = Path(work) / "bins"
        bins.mkdir()
        _write(bins, [], recycling.BINS, 0)
        with RobotContext(description, recycling.runtime(package, bins / "scene.xml")) as ctx:
            group = ctx.manipulator(arm.manipulator).group
            joints = description.group(group).joints
            snapshot = ctx.snapshot()
            ready = np.array([snapshot.position(j) for j in joints])
            scene = ctx.planning_scene(snapshot, group)
            frame = next(e.frame for e in description.end_effectors if e.name == arm.hand)
            solver = SsikKinematics.from_package(package, arm.manipulator).bind(scene, frame)
            free = [
                q
                for q in solver.solve(regions.matrix(hand), q_init=ready)
                if not scene.contacts(tuple(float(v) for v in q))
            ]
        if not free:
            return {**out, "outcome": "unstaged: no collision-free IK solution"}
        q = min(free, key=lambda q: float(np.abs(q - ready).max()))
        staged = Path(work) / "staged"
        staged.mkdir()
        placement = Placement(
            name=obj, kind=kind, position=world.position, quat_wxyz=world.quat_wxyz
        )
        _write(staged, [placement], recycling.BINS, 0)
        start = {**dict(zip(joints, (float(v) for v in q), strict=True)), **held["gripper"]}
        runtime = MujocoRuntime(
            package,
            scene=staged / "scene.xml",
            attach_tolerance_m=recycling.HOLD_TOLERANCE_M,
            attach_tolerance_rad=recycling.SWING_TOLERANCE_RAD,
            start=start,
        )
        with RobotContext(description, runtime) as ctx:
            spec = recycling.grasp_spec(ctx, obj, arm)
            close = GripperCommand(gripper=arm.gripper, opening=spec.close_opening)
            closing = ctx.submit(close, timeout_ns=spec.gripper_timeout_ns)
            recycling._wait(ctx, closing, lambda: None)
            ctx.attach(obj, arm.hand, transform=in_hand)
            before = recycling.object_pose(ctx, obj)
            out["slip_m"] = math.dist(before.position, world.position)
            why = recycling._release(ctx, obj, arm, spec, lambda: None)
            if why is not None:
                return {**out, "outcome": f"unreleased: {why}"}
            released = ctx.now.time_ns
            channel = ObservationRequest(channels=(f"{obj}.pose",))

            def read() -> tuple[int, list[float]]:
                reading = ctx.observe(channel).readings[0]
                value = reading.value
                assert isinstance(value, tuple)
                return reading.stamp.time_ns - released, list(value)

            out["period_ns"] = ctx.trajectory_contract.period_ns
            if dwell_ns is None:
                return {**out, "outcome": "followed", **_follow(ctx, read)}
            criteria = RestCriteria(
                max_gap_ns=ctx.trajectory_contract.period_ns,
                dwell_ns=dwell_ns,
                translation_m=TRANSLATION_M,
                rotation_rad=ROTATION_RAD,
                source="calibrate_landing.py: verification",
            )
            watch = ctx.watch_rest(
                ObjectRest(object=obj, criteria=criteria), deadline_ns=released + CAP_NS
            )
            while not watch.done:
                ctx.step()
            rest = watch.result
            assert rest is not None
            if rest.outcome is not RestOutcome.AT_REST:
                return {**out, "outcome": f"verified: {rest.outcome.value}", "passed": False}
            run = [(s.time_ns - released, list(s.value)) for s in rest.window]
            fired = rest.end.time_ns - released
            while ctx.now.time_ns - released < fired + HOLD_NS:
                ctx.step()
                run.append(read())
            return {
                **out,
                "outcome": "verified: at_rest",
                "rested_ns": fired,
                "passed": _within(run),
            }


def _follow(ctx: Any, read: Any) -> dict[str, Any]:
    """Read the pose each tick until it has stayed within the variation for ``HOLD_NS``,
    or ``CAP_NS`` passes: when it settled, and the longest interval it stayed within the
    variation and then moved, as the watch's judge would see it."""
    run: list[tuple[int, list[float]]] = [read()]
    longest, at = 0, 0  # the longest still interval that then moved, and when it broke
    while run[-1][0] < CAP_NS:
        ctx.step()
        sample = read()
        before = run[-1][0] - run[0][0]
        run.append(sample)
        broke = False
        while not _within(run):
            run.pop(0)
            broke = True
        if broke and before > longest:
            longest, at = before, sample[0]
        if run[-1][0] - run[0][0] >= HOLD_NS:
            return {"settled_ns": run[0][0], "false_still_ns": longest, "broke_at_ns": at}
    return {"settled_ns": None, "false_still_ns": longest, "broke_at_ns": at}


def _in_subprocess(*args: str) -> dict[str, Any]:
    done = subprocess.run([sys.executable, __file__, *args], capture_output=True, text=True)
    if done.returncode != 0:
        return {"args": list(args), "error": done.stderr[-4000:]}
    result: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    return result


def _provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    from geodude import recycling

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "sources": recycling.sources(),
        "blas_kernel": recycling.blas_kernel(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("out", nargs="?", type=Path, help="where to write the calibration")
    parser.add_argument("--jobs", type=int, default=JOBS)
    parser.add_argument("--hold", help=argparse.SUPPRESS)
    parser.add_argument("--trial", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.hold:
        print(json.dumps(hold(args.hold)))
        return 0
    if args.trial:
        held, release, dwell = json.loads(args.trial)
        print(json.dumps(trial(held, release, dwell)))
        return 0
    if args.out is None:
        parser.error("an output path is needed")
    if args.out.exists():
        parser.error(f"{args.out} exists: a calibration is never overwritten")
    from geodude import recycling

    provenance = _provenance()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        holds = list(pool.map(lambda k: _in_subprocess("--hold", k), recycling.MIX_KINDS))
        cases = [
            (held, release)
            for held in holds
            if held.get("seed") is not None
            for release in releases(held)
        ]

        def run(dwell: int | None) -> list[dict[str, Any]]:
            return list(
                pool.map(
                    lambda c: _in_subprocess("--trial", json.dumps([c[0], c[1], dwell])),
                    cases,
                )
            )

        followed = run(None)
        judged = [t for t in followed if t.get("outcome") == "followed"]
        settled = [t for t in judged if t["settled_ns"] is not None]
        (period,) = {t["period_ns"] for t in judged}  # the control tick, one for every trial
        dwell = max(t["false_still_ns"] for t in judged) + period
        verified = run(dwell)
    slowest = max(t["rested_ns"] for t in verified if "rested_ns" in t)
    timeout = 2 * math.ceil(slowest / 1e9) * 1_000_000_000
    by_kind = {}
    for kind in recycling.MIX_KINDS:
        rows = [t for t in verified if t.get("kind") == kind]
        found = [t for t in followed if t.get("kind") == kind and t.get("outcome") == "followed"]
        by_kind[kind] = {
            "trials": len(rows),
            "passed": sum(bool(t.get("passed")) for t in rows),
            "false_still_ns": max((t["false_still_ns"] for t in found), default=None),
            "rested_ns": max((t["rested_ns"] for t in rows if "rested_ns" in t), default=None),
        }
    report = {
        **provenance,
        "criteria": {
            "translation_m": TRANSLATION_M,
            "rotation_rad": ROTATION_RAD,
            "max_gap_ns": period,
            "dwell_ns": dwell,
        },
        "hold_ns": HOLD_NS,
        "cap_ns": CAP_NS,
        "timeout_ns": timeout,
        "slowest_rest_ns": slowest,
        "headroom_ns": timeout - slowest,
        "judged": len(judged),
        "never_settled": len(judged) - len(settled),
        "passed": sum(bool(t.get("passed")) for t in verified),
        "verified": len(verified),
        "all_passed": sum(bool(t.get("passed")) for t in verified) == len(judged) == len(cases),
        "by_kind": by_kind,
        "holds": holds,
        "followed": followed,
        "verified_trials": verified,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(report, indent=1) + "\n")
    partial.replace(args.out)
    keys = (
        "criteria",
        "timeout_ns",
        "slowest_rest_ns",
        "headroom_ns",
        "judged",
        "never_settled",
        "passed",
        "verified",
        "all_passed",
        "by_kind",
    )
    print(json.dumps({k: report[k] for k in keys}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
