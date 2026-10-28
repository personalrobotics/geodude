"""Where the watched robot stands still in a recycling round, paced like the live view
(personalrobotics/ssgrasping#21).

Each sample is one fresh process that stages ``--cans`` cans at a seed and recycles them
(``recycling.recycle``, racing both arms and looking ahead, as ``geodude-recycling
--cans N`` does), paced as the live view paces it: never ahead of real time, and once
held still, by planning on the context's thread or by waiting for a plan, carrying on
from there instead of catching up. Each such pause is time the robot stood still
waiting on the computer, and is charged to the skill stage in progress, or to
``between skills``, where each object's race is planned.

``before`` plans no drop's transfer during its grasp's lift (``transfer_follow``
disabled), as before #21; ``after`` is the demo as it is. Samples run one at a time,
interleaved by seed and variant, so the host's state affects both alike; wall-clock
times belong to the host that measured them:

    uv run python scripts/time_recycling.py out/idle.json --seeds 0-19

The artifact records, per sample, the cans recycled, the simulated and wall-clock
duration, the total standing time and its split by stage, with the commit, its
cleanliness, the sources, and the host's load before and after.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

VARIANTS = ("before", "after")
BETWEEN = "between skills"


def sample(variant: str, seed: int, cans: int) -> dict[str, Any]:
    """One paced round, in this process."""
    import ssgrasping._skill as skill
    import ssgrasping.execution as execution
    from ssrobot import RobotContext, load_installed_package

    from geodude import recycling
    from geodude.scene import stage_scene

    where = [BETWEEN]
    begin, finish = skill.Run.begin, execution.SkillExecution._finish

    def tracked_begin(self: Any, name: str) -> None:
        where[0] = f"{self.skill}.{name}"
        begin(self, name)

    def tracked_finish(self: Any, state: Any) -> None:
        finish(self, state)
        where[0] = BETWEEN

    skill.Run.begin = tracked_begin  # type: ignore[method-assign]
    execution.SkillExecution._finish = tracked_finish  # type: ignore[method-assign]
    if variant == "before":
        recycling.transfer_follow = lambda *a, **k: None  # type: ignore[assignment]

    package = load_installed_package("geodude_assets")
    stalls: dict[str, float] = defaultdict(float)
    with tempfile.TemporaryDirectory() as work:
        scene = stage_scene(
            Path(work) / "scene", package, objects={"can": cans}, fixtures=recycling.BINS, seed=seed
        )
        with RobotContext(package.description, recycling.runtime(package, scene)) as ctx:
            anchor = [time.monotonic() - ctx.now.time_ns / 1e9]

            def tick() -> None:
                ahead = ctx.now.time_ns / 1e9 - (time.monotonic() - anchor[0])
                if ahead > 0:
                    time.sleep(ahead)
                elif ahead < 0:
                    stalls[where[0]] -= ahead
                    anchor[0] = time.monotonic() - ctx.now.time_ns / 1e9

            began_sim, began_wall = ctx.now.time_ns, time.perf_counter()
            outcomes = recycling.recycle(ctx, package, sorted(ctx.info.objects), tick)
            sim_s = (ctx.now.time_ns - began_sim) / 1e9
            wall_s = time.perf_counter() - began_wall
    return {
        "variant": variant,
        "seed": seed,
        "recycled": sum(bool(o["succeeded"]) for o in outcomes),
        "attempts": len(outcomes),
        "sim_s": round(sim_s, 3),
        "wall_s": round(wall_s, 3),
        "stall_s": round(sum(stalls.values()), 4),
        "stall_by_stage_s": {k: round(v, 4) for k, v in sorted(stalls.items())},
    }


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
        "load": list(os.getloadavg()),
        "cpus": os.cpu_count(),
    }


def _seeds(text: str) -> list[int]:
    low, _, high = text.partition("-")
    return list(range(int(low), int(high) + 1)) if high else [int(s) for s in text.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("out", nargs="?", type=Path, help="where to write the measurement")
    parser.add_argument("--seeds", default="0-19", help="a range (0-19) or a list (3,7)")
    parser.add_argument("--cans", type=int, default=4)
    parser.add_argument("--sample", nargs=2, metavar=("VARIANT", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sample:
        variant, seed = args.sample
        print(json.dumps(sample(variant, int(seed), args.cans)))
        return 0
    if args.out is None:
        parser.error("an output path is needed")
    if args.out.exists():
        parser.error(f"{args.out} exists: a measurement is never overwritten")
    provenance = _provenance()
    samples = []
    for seed in _seeds(args.seeds):
        for variant in VARIANTS:
            run = [
                sys.executable,
                __file__,
                "--sample",
                variant,
                str(seed),
                "--cans",
                str(args.cans),
            ]
            done = subprocess.run(run, capture_output=True, text=True, check=True)
            samples.append(json.loads(done.stdout.strip().splitlines()[-1]))
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [s for s in samples if s["variant"] == variant]
        per_can = sum(s["attempts"] for s in rows)
        stages: dict[str, float] = defaultdict(float)
        for s in rows:
            for stage, seconds in s["stall_by_stage_s"].items():
                stages[stage] += seconds
        summary[variant] = {
            "rounds": len(rows),
            "recycled": sum(s["recycled"] for s in rows),
            "attempts": per_can,
            "stall_per_can_s": round(sum(s["stall_s"] for s in rows) / per_can, 4),
            "stall_per_can_by_stage_s": {
                k: round(v / per_can, 4) for k, v in sorted(stages.items())
            },
            "sim_per_round_s": round(sum(s["sim_s"] for s in rows) / len(rows), 3),
        }
    report = {
        **provenance,
        "load_after": list(os.getloadavg()),
        "cans": args.cans,
        "summary": summary,
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(report, indent=1) + "\n")
    partial.replace(args.out)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
