"""Time the recycling demo's start, with and without building the planning scenes first
(#224).

Each sample is one fresh process that stages ``--cans`` cans at ``--seed`` and recycles
them (``recycling.recycle``), as ``geodude-recycling --cans N`` does, and records, in
wall-clock seconds from the context opening:
- ``warm_up_s``: how long ``recycling.warm_planning`` took (0 when it is skipped);
- ``first_command_s`` and ``first_motion_s``: when the first command was submitted, and
  the first planned arm motion executed;
- every ``NativeScene.from_model`` build of the demo's context: when, how long, whether
  it ran inside the warm-up, and whether after the first skill stage began.
  ``stage_scene``'s own clearance check builds its scenes earlier, in a context of its
  own, and is counted apart;
- the cans' outcomes, which must not depend on the variant.

``cold`` skips the warm-up, as before #224; ``warm`` is the demo as it is. Samples run
one at a time, alternating between the variants, so the host's state affects both alike:

    uv run python scripts/time_startup.py out/startup.json --samples 5

Planning runs off the simulated clock, so the warm-up changes no outcome and no
simulated time; it moves the scenes' build from the first plans, where a real robot
waits, to bring-up.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

VARIANTS = ("cold", "warm")


def sample(variant: str, seed: int, cans: int) -> dict[str, Any]:
    """One run of the demo, in this process."""
    import ssgrasping._skill as skill
    import ssrobot.mujoco._planning as planning
    from ssrobot import RobotContext, load_installed_package

    from geodude import recycling
    from geodude.scene import stage_scene

    clock: dict[str, Any] = {"opened": None, "skill": None, "command": None, "motion": None}
    builds: list[dict[str, float | bool]] = []
    warmed = [0.0]
    warming = [False]

    def since() -> float:
        return float(time.perf_counter() - clock["opened"])

    native = planning.native_mujoco  # type: ignore[attr-defined]  # sscbirrt's, as ssrobot uses it
    build = native.NativeScene.from_model
    staging = [0]  # builds by stage_scene's own clearance check, before the demo's context

    def timed_build(*args: Any, **kwargs: Any) -> Any:
        began = time.perf_counter()
        scene = build(*args, **kwargs)
        if clock["opened"] is None:
            staging[0] += 1
            return scene
        builds.append(
            {
                "at_s": round(began - clock["opened"], 4),
                "took_s": round(time.perf_counter() - began, 4),
                "in_warm_up": warming[0],
                "after_first_skill": clock["skill"] is not None,
            }
        )
        return scene

    native.NativeScene.from_model = timed_build
    begin = skill.Run.begin

    def first_stage(self: Any, name: str, *a: Any, **k: Any) -> Any:
        if clock["skill"] is None:
            clock["skill"] = since()
        return begin(self, name, *a, **k)

    skill.Run.begin = first_stage  # type: ignore[method-assign]
    warm = recycling.warm_planning

    def timed_warm(*a: Any, **k: Any) -> float:
        if variant == "cold":
            return 0.0
        warming[0] = True
        try:
            warmed[0] = warm(*a, **k)
        finally:
            warming[0] = False
        return warmed[0]

    recycling.warm_planning = timed_warm

    package = load_installed_package("geodude_assets")
    with tempfile.TemporaryDirectory() as work:
        scene = stage_scene(
            Path(work) / "scene", package, objects={"can": cans}, fixtures=recycling.BINS, seed=seed
        )
        clock["opened"] = time.perf_counter()
        with RobotContext(package.description, recycling.runtime(package, scene)) as ctx:
            submit, execute, cont = ctx.submit, ctx.execute, ctx.execute_continuation

            def first(key: str, call: Any) -> Any:
                def wrapped(*a: Any, **k: Any) -> Any:
                    if clock[key] is None:
                        clock[key] = since()
                    return call(*a, **k)

                return wrapped

            ctx.submit = first("command", submit)  # type: ignore[method-assign]
            ctx.execute = first("motion", execute)  # type: ignore[method-assign]
            ctx.execute_continuation = first("motion", cont)  # type: ignore[method-assign]
            outcomes = recycling.recycle(ctx, package, sorted(ctx.info.objects))
            total = since()
    return {
        "variant": variant,
        "warm_up_s": round(warmed[0], 4),
        "first_skill_s": round(clock["skill"], 4),
        "first_command_s": round(clock["command"], 4),
        "first_motion_s": round(clock["motion"], 4),
        "round_s": round(total, 3),
        "native_scene_builds": builds,
        "staging_builds": staging[0],
        "builds_after_first_skill": sum(1 for b in builds if b["after_first_skill"]),
        "builds_outside_warm_up": sum(1 for b in builds if not b["in_warm_up"]),
        "outcomes": {o["object"]: o["succeeded"] for o in outcomes},
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


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("out", nargs="?", type=Path, help="where to write the measurement")
    parser.add_argument("--samples", type=int, default=5, help="samples of each variant")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--cans", type=int, default=4)
    parser.add_argument(
        "--sample", choices=VARIANTS, help=argparse.SUPPRESS
    )  # one sample, to stdout
    args = parser.parse_args()
    if args.sample:
        print(json.dumps(sample(args.sample, args.seed, args.cans)))
        return 0
    if args.out is None:
        parser.error("an output path is needed")
    if args.out.exists():
        parser.error(f"{args.out} exists: a measurement is never overwritten")
    provenance = _provenance()
    samples = []
    for _ in range(args.samples):
        for variant in VARIANTS:
            run = [
                sys.executable,
                __file__,
                "--sample",
                variant,
                "--seed",
                str(args.seed),
                "--cans",
                str(args.cans),
            ]
            done = subprocess.run(run, capture_output=True, text=True, check=True)
            samples.append(json.loads(done.stdout.strip().splitlines()[-1]))
    summary = {
        variant: {
            key: _median([s[key] for s in samples if s["variant"] == variant])
            for key in ("warm_up_s", "first_command_s", "first_motion_s", "round_s")
        }
        | {
            key: max(s[key] for s in samples if s["variant"] == variant)
            for key in ("builds_after_first_skill", "builds_outside_warm_up")
        }
        for variant in VARIANTS
    }
    outcomes = {json.dumps(s["outcomes"], sort_keys=True) for s in samples}
    report = {
        **provenance,
        "load_after": list(os.getloadavg()),
        "seed": args.seed,
        "cans": args.cans,
        "samples_per_variant": args.samples,
        "outcomes_identical": len(outcomes) == 1,
        "summary": summary,
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(report, indent=1) + "\n")
    partial.replace(args.out)
    print(
        json.dumps(
            {"outcomes_identical": report["outcomes_identical"], "summary": summary}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
