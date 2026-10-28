"""Evaluate the recycling demo over a seeded corpus of mixed rounds (#201).

Each round stages ``recycling.mix(seed, OBJECTS)``, the legacy demo's five objects drawn
from its nine types, and recycles them with ``recycling.recycle``, racing both arms with
lookahead, as the demo runs. The evaluation is one JSON artifact:
- what it ran: the corpus, its seeds, the Geodude commit and whether the tree was
  dirty, every ``recycling.SOURCES`` distribution, and the host's load;
- per round: the staged objects and their poses, and per staged object its outcome, with
  the deciding stage's detail, and its mechanism (``classify``);
- the totals the bar is stated in: objects recycled out of every object staged,
  including any never attempted because its round stopped, and rounds cleared.

Planning has a wall-clock budget, so a round run beside others may time out where it
would not alone. Rounds run in parallel, at most ``--jobs`` at once; every round that
records a planning timeout is then run again alone, and its verdict is that run's. Each
round records which run its verdict came from.

The corpora are fixed (#201): ``development`` (seeds 0-39), for diagnosis and every fix,
and ``acceptance`` (seeds 1000-1039), run once, on the final pins::

    uv run python scripts/evaluate_recycling.py development out/development.json

``--seeds 3,7`` evaluates only those seeds of the corpus. ``--classify`` reclassifies an
existing evaluation, without running anything, after a change to ``classify``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

CORPORA = {"development": range(0, 40), "acceptance": range(1000, 1040)}
OBJECTS = 5
"""Objects per round: the legacy demo's ``spawn_count``."""
JOBS = 16
"""Rounds run at once by default: a quarter of the measurement host's 64 cores, since
each round's races plan on worker threads."""


def run_round(seed: int) -> dict[str, Any]:
    """One round of ``seed``, in this process."""
    from ssrobot import RobotContext, load_installed_package

    from geodude import recycling
    from geodude.scene import stage_scene

    package = load_installed_package("geodude_assets")
    work = Path(tempfile.mkdtemp(prefix="geodude-eval-"))
    try:
        scene = stage_scene(
            work / "scene",
            package,
            objects=recycling.mix(seed, OBJECTS),
            fixtures=recycling.BINS,
            seed=seed,
        )
        staged = json.loads((scene.parent / "manifest.json").read_text())["objects"]
        with RobotContext(package.description, recycling.runtime(package, scene)) as ctx:
            outcomes = recycling.recycle(ctx, package, sorted(ctx.info.objects))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"seed": seed, "staged": staged, "outcomes": outcomes}


def _timed_out(round_: dict[str, Any]) -> bool:
    """Whether any plan of the round ran out of its budget: a planning stage's detail,
    or a raced plan's status."""
    for outcome in round_["outcomes"]:
        failure = outcome.get("failure") or {}
        if "planning timeout" in failure.get("detail", ""):
            return True
        for candidate in outcome["race"]["candidates"]:
            if "timeout" in candidate["detail"].lower() or any(
                p["status"] == "timeout" for p in candidate["plans"]
            ):
                return True
    return False


def classify(outcome: dict[str, Any] | None) -> str:
    """What failed for one staged object, by its outcome: ``recycled``, or a mechanism.
    ``None`` is an object never attempted, because its round stopped first."""
    if outcome is None:
        return "not_attempted"
    if outcome["succeeded"]:
        return "recycled"
    if outcome["arm"] is None:
        refused = {
            str(p["stage"])
            for c in outcome["race"]["candidates"]
            for p in c["plans"]
            if p["status"] != "valid"
        }
        return "no_arm_plans:" + "+".join(sorted(refused))
    failure = outcome.get("failure")
    if failure is None:
        failed = sorted(k for k, ok in outcome["checks"].items() if not ok)
        return "checks:" + "+".join(failed)
    skill, stage, detail = str(failure["skill"]), str(failure["stage"]), str(failure["detail"])
    if stage == "acquired":
        return f"{skill}.acquired:" + detail.split(":", 1)[0]  # missed or indeterminate
    if detail.startswith("planning "):
        return f"{skill}.{stage}:" + detail.split(":", 1)[0].removeprefix("planning ")
    return f"{skill}.{stage}"


def _summary(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    kinds: dict[str, Counter[bool]] = {}
    mechanisms: Counter[str] = Counter()
    recycled = staged = cleared = 0
    for round_ in rounds:
        verdicts = [o["mechanism"] for o in round_["objects"]]
        staged += len(verdicts)
        recycled += verdicts.count("recycled")
        cleared += all(v == "recycled" for v in verdicts)
        for o in round_["objects"]:
            kinds.setdefault(o["kind"], Counter())[o["mechanism"] == "recycled"] += 1
            if o["mechanism"] != "recycled":
                mechanisms[o["mechanism"]] += 1
    return {
        "objects_staged": staged,
        "objects_recycled": recycled,
        "rounds": len(rounds),
        "rounds_cleared": cleared,
        "by_kind": {
            k: {"recycled": c[True], "staged": c[True] + c[False]} for k, c in sorted(kinds.items())
        },
        "mechanisms": dict(mechanisms.most_common()),
    }


def _objects(round_: dict[str, Any]) -> list[dict[str, Any]]:
    """Every staged object of a round, with its outcome and mechanism."""
    by_name = {o["object"]: o for o in round_["outcomes"]}
    out = []
    for placement in round_["staged"]:
        outcome = by_name.get(placement["name"])
        out.append(
            {
                "object": placement["name"],
                "kind": placement["kind"],
                "staged": placement,
                "mechanism": classify(outcome),
                "outcome": outcome,
            }
        )
    return out


def _provenance() -> dict[str, Any]:
    from geodude import recycling

    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "sources": recycling.sources(),
        "load": list(os.getloadavg()),
        "cpus": os.cpu_count(),
    }


def _round_in_subprocess(seed: int) -> dict[str, Any]:
    done = subprocess.run(
        [sys.executable, __file__, "--round", str(seed)], capture_output=True, text=True
    )
    if done.returncode != 0:
        return {"seed": seed, "error": done.stderr[-4000:]}
    result: dict[str, Any] = json.loads(done.stdout)
    return result


def evaluate(corpus: str, seeds: list[int], jobs: int) -> dict[str, Any]:
    provenance = _provenance()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rounds = {
            r["seed"]: {**r, "run": "parallel"} for r in pool.map(_round_in_subprocess, seeds)
        }
    for seed in seeds:  # alone, every round a planning budget may have decided
        if "error" not in rounds[seed] and _timed_out(rounds[seed]):
            rounds[seed] = {**_round_in_subprocess(seed), "run": "alone, after a timeout"}
    finished = [rounds[s] for s in seeds]
    for round_ in finished:
        if "error" not in round_:
            round_["objects"] = _objects(round_)
    return {
        "corpus": corpus,
        "seeds": seeds,
        "objects_per_round": OBJECTS,
        "jobs": jobs,
        **provenance,
        "load_after": list(os.getloadavg()),
        "errors": [r["seed"] for r in finished if "error" in r],
        "summary": _summary([r for r in finished if "error" not in r]),
        "rounds": finished,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("corpus", nargs="?", choices=sorted(CORPORA))
    parser.add_argument("out", nargs="?", type=Path, help="where to write the evaluation")
    parser.add_argument("--seeds", help="comma-separated seeds of the corpus, instead of all")
    parser.add_argument("--jobs", type=int, default=JOBS, help="rounds run at once")
    parser.add_argument("--round", type=int, help=argparse.SUPPRESS)  # one round, to stdout
    parser.add_argument("--classify", type=Path, help="reclassify this evaluation in place")
    args = parser.parse_args()
    if args.round is not None:
        print(json.dumps(run_round(args.round)))
        return 0
    if args.classify is not None:
        evaluation = json.loads(args.classify.read_text())
        for round_ in evaluation["rounds"]:
            if "error" not in round_:
                round_["objects"] = _objects(round_)
        evaluation["summary"] = _summary([r for r in evaluation["rounds"] if "error" not in r])
        args.classify.write_text(json.dumps(evaluation, indent=1) + "\n")
        print(json.dumps(evaluation["summary"], indent=1))
        return 0
    if args.corpus is None or args.out is None:
        parser.error("a corpus and an output path are needed")
    seeds = list(CORPORA[args.corpus])
    if args.seeds:
        chosen = [int(s) for s in args.seeds.split(",")]
        if not set(chosen) <= set(seeds):
            parser.error(f"--seeds must be seeds of the {args.corpus} corpus")
        seeds = chosen
    if args.out.exists():
        parser.error(f"{args.out} exists: an evaluation is never overwritten")
    evaluation = evaluate(args.corpus, seeds, args.jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(evaluation, indent=1) + "\n")
    partial.replace(args.out)
    print(json.dumps(evaluation["summary"], indent=1))
    return 1 if evaluation["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
