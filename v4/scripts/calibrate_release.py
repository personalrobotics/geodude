"""How far an object released above a bin's rim by the 2F-140 strays sideways before it
falls past the rim (#237).

The release itself moves an object: the pads let go of one side first, so it tilts and
drifts as it slides out, and a corner that sweeps over the rim before the object's bottom
has fallen below it strikes the rim. This script measures that sweep, away from any bin.
It is why ``regions.held_drop_zones`` releases below the rim instead.

Each trial stages one object of a legacy type (``recycling.MIX_KINDS``) at a seed, with
the bins staged far out of reach, and runs the demo's own skills on it: ``pick``, then
``drop`` from the legacy zone over the bin's real base (``recycling.drop_spec``), which
releases ``max(0.25, height + 0.10)`` above the rim. Nothing is there to strike, so the
object falls freely. From the start of
the drop's ``release`` stage until the object's lowest corner falls below the rim's
height, every physics step is read, and the trial records:
- ``sweep_m``: the most any side of the footprint (the bounding box of the object's
  corners in x and y, as ``regions.held_drop_zones`` computes it) moves out beyond where
  it was when the release began;
- ``tilt_rad`` and ``drift_m``: the object's tilt from its orientation then, and its
  centre's sideways travel, when it passes the rim;
- ``drop_m``: how far its lowest corner fell to the rim.

Seeds 5000 and up are this calibration's alone: none of the recycling corpora uses them.
Trials run one process each, so that every trial starts from the same state::

    OPENBLAS_CORETYPE=Haswell uv run python scripts/calibrate_release.py out/release.json

The artifact records every trial, the maxima by type and overall, the commit, its
cleanliness, the sources and the OpenBLAS kernel.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SEEDS = range(5000, 5010)
FAR = {"recycle_bin": [(3.0, 3.0, 0.01), (-3.0, 3.0, 0.01)]}
"""Where the bins are staged in a trial: far out of reach, so the release meets no wall.
``drop`` still finds its container there, and its zones are over ``recycling.BINS``."""


def _corners(geometry: dict[str, Any]) -> Any:
    import numpy as np

    if geometry["type"] == "cylinder":
        half = np.array([geometry["radius"], geometry["radius"], geometry["height"] / 2])
    else:
        half = np.array(geometry["size"], dtype=float) / 2
    return np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3) * half


def trial(kind: str, seed: int) -> dict[str, Any]:
    """One trial, in this process."""
    import mujoco
    import numpy as np
    import ssgrasping._skill as skill
    from ssgrasping import drop
    from ssrobot import RobotContext, load_installed_package
    from ssrobot.mujoco import _runtime

    from geodude import recycling
    from geodude.scene import meta, stage_scene

    package = load_installed_package("geodude_assets")
    corners = _corners(meta(kind)["geometric_properties"])
    rim = (
        recycling.BINS["recycle_bin"][0][2]
        + meta("recycle_bin")["geometric_properties"]["outer_size"][2]
    )
    watching: dict[str, Any] = {}
    begin, step = skill.Run.begin, _runtime.MujocoRuntime.step

    def tracked_begin(self: Any, name: str) -> None:
        begin(self, name)
        if self.skill == "drop" and name == "release":
            watching["on"] = True

    def tracked_step(self: Any) -> None:
        step(self)
        if not watching.get("on") or watching.get("done"):
            return
        model, data = self._model, self._data
        if "body" not in watching:
            watching["body"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
        b = watching["body"]
        world = data.xmat[b].reshape(3, 3) @ corners.T + data.xpos[b][:, None]
        low, high = world[:2].min(axis=1), world[:2].max(axis=1)
        if "start" not in watching:
            watching["start"] = {
                "low": low,
                "high": high,
                "centre": data.xpos[b].copy(),
                "z": data.xmat[b].reshape(3, 3)[:, 2].copy(),
                "lowest": float(world[2].min()),
                "time": float(data.time),
            }
            watching["sweep"] = 0.0
        start = watching["start"]
        out = max(float((start["low"] - low).max()), float((high - start["high"]).max()))
        watching["sweep"] = max(watching["sweep"], out)
        touching = any(
            b in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
            and "gripper"
            in (
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(
                        model.geom_bodyid[c.geom2]
                        if model.geom_bodyid[c.geom1] == b
                        else model.geom_bodyid[c.geom1]
                    ),
                )
                or ""
            )
            for c in data.contact[: data.ncon]
        )
        if touching:
            watching["last_touch"] = float(data.time) - start["time"]
        if float(world[2].min()) < rim:
            z = data.xmat[b].reshape(3, 3)[:, 2]
            watching["done"] = {
                "sweep_m": round(watching["sweep"], 5),
                "tilt_rad": round(math.acos(min(1.0, float(z @ start["z"]))), 5),
                "drift_m": round(float(np.hypot(*(data.xpos[b][:2] - start["centre"][:2]))), 5),
                "drop_m": round(start["lowest"] - rim, 5),
                "fall_s": round(float(data.time) - start["time"], 4),
                "released_after_s": round(watching.get("last_touch", 0.0), 4),
            }

    skill.Run.begin = tracked_begin  # type: ignore[method-assign]
    _runtime.MujocoRuntime.step = tracked_step  # type: ignore[method-assign]

    with tempfile.TemporaryDirectory() as work:
        scene = stage_scene(
            Path(work) / "scene", package, objects={kind: 1}, fixtures=FAR, seed=seed
        )
        with RobotContext(package.description, recycling.runtime(package, scene)) as ctx:
            (obj,) = sorted(o for o in ctx.info.objects if not o.startswith("recycle_bin"))
            arm = recycling.arm_for(recycling.object_pose(ctx, obj).position)
            planners = recycling.planners(package, arm)
            out: dict[str, Any] = {"kind": kind, "seed": seed, "object": obj, "arm": arm.side}
            picked = recycling.pick(ctx, obj, arm=arm, planners=planners)
            if not picked.succeeded:
                return {**out, "outcome": "not picked"}
            spec = recycling.drop_spec(obj, arm)
            dropped = recycling.drive(
                ctx, drop(ctx, planners[0], spec, check_planner=planners[1]), lambda: None
            )
            for _ in range(2000):  # long enough for any release to fall past the rim
                if watching.get("done"):
                    break
                ctx.step()
    if not watching.get("done"):
        return {
            **out,
            "outcome": "not released",
            "drop": [[s.name, s.succeeded] for s in dropped.stages],
        }
    return {**out, "outcome": "released", **watching["done"]}


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
    parser.add_argument("--trial", nargs=2, metavar=("KIND", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.trial:
        kind, seed = args.trial
        print(json.dumps(trial(kind, int(seed))))
        return 0
    if args.out is None:
        parser.error("an output path is needed")
    if args.out.exists():
        parser.error(f"{args.out} exists: a calibration is never overwritten")
    from geodude import recycling

    trials = []
    for kind in recycling.MIX_KINDS:
        for seed in SEEDS:
            run = [sys.executable, __file__, "--trial", kind, str(seed)]
            done = subprocess.run(run, capture_output=True, text=True, check=True)
            trials.append(json.loads(done.stdout.strip().splitlines()[-1]))
    released = [t for t in trials if t["outcome"] == "released"]
    keys = ("sweep_m", "tilt_rad", "drift_m")
    report = {
        **_provenance(),
        "seeds": [SEEDS.start, SEEDS.stop - 1],
        "maxima": {k: max(t[k] for t in released) for k in keys},
        "by_kind": {
            kind: {
                "released": len(rows),
                "trials": sum(t["kind"] == kind for t in trials),
                **{k: max(t[k] for t in rows) for k in keys if rows},
            }
            for kind in recycling.MIX_KINDS
            for rows in [[t for t in released if t["kind"] == kind]]
        },
        "trials": trials,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(report, indent=1) + "\n")
    partial.replace(args.out)
    print(json.dumps({k: report[k] for k in ("maxima", "by_kind")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
