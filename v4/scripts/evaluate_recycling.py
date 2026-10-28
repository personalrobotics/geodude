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

An evaluation in which any round errored is invalid (``"valid": false``): it has no
``summary``, only a ``partial_summary`` of the rounds that finished, and the script exits
nonzero. ``FORMAT`` versions the artifact.

Planning has a wall-clock budget, so a round run beside others may time out where it
would not alone. Rounds run in parallel, at most ``--jobs`` at once; every round that
records a planning timeout is then run again alone, and its verdict is that run's. Each
round records which run its verdict came from.

The corpora are fixed (#201): ``development`` (seeds 0-39), for diagnosis and every fix,
and ``acceptance`` (seeds 1000-1039), run once, on the final pins::

    uv run python scripts/evaluate_recycling.py development out/development.json

``--seeds 3,7`` evaluates only those seeds of the development corpus. The acceptance
corpus runs whole, from a clean tree, or not at all.

``--classify EVALUATION OUT`` reclassifies an existing evaluation, without running
anything, after a change to ``classify``. It never touches ``EVALUATION``: ``OUT`` is a
derived artifact that records the original's SHA-256 and the classifying commit.
"""

from __future__ import annotations

import argparse
import hashlib
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

FORMAT = 1
"""The evaluation artifact's format: raised whenever a field's meaning changes."""
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
    if stage in ("obstructed", "acquired"):
        return f"{skill}.{stage}:" + detail.split(":", 1)[0]  # missed, or lost in the lift
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


def _valid(evaluation: dict[str, Any]) -> dict[str, Any]:
    """``evaluation`` with its verdict: valid, with a summary, only if every round ran."""
    completed = [r for r in evaluation["rounds"] if "error" not in r]
    errors = [r["seed"] for r in evaluation["rounds"] if "error" in r]
    out = {k: v for k, v in evaluation.items() if k not in ("summary", "partial_summary")}
    out.update(errors=errors, valid=not errors)
    if errors:
        out["partial_summary"] = _summary(completed)
    else:
        out["summary"] = _summary(completed)
    return out


def _write(out: Path, evaluation: dict[str, Any]) -> None:
    """Write ``evaluation`` to ``out`` whole or not at all."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    partial.write_text(json.dumps(evaluation, indent=1) + "\n")
    partial.replace(out)


def _report(evaluation: dict[str, Any]) -> int:
    if evaluation["valid"]:
        print(json.dumps(evaluation["summary"], indent=1))
        return 0
    print(f"INVALID: rounds {evaluation['errors']} errored", file=sys.stderr)
    return 1


def reclassify(source: Path) -> dict[str, Any]:
    """``source``'s evaluation, reclassified by this commit's ``classify``."""
    raw = source.read_bytes()
    evaluation = json.loads(raw)
    if evaluation.get("format") != FORMAT:
        raise SystemExit(f"{source} is format {evaluation.get('format')}, not {FORMAT}")
    for round_ in evaluation["rounds"]:
        if "error" not in round_:
            round_["objects"] = _objects(round_)
    provenance = _provenance()
    evaluation["derived"] = {
        "from_sha256": hashlib.sha256(raw).hexdigest(),
        "classifier_commit": provenance["commit"],
        "classifier_dirty": provenance["dirty"],
    }
    return _valid(evaluation)


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
    return _valid(
        {
            "format": FORMAT,
            "corpus": corpus,
            "seeds": seeds,
            "objects_per_round": OBJECTS,
            "jobs": jobs,
            **provenance,
            "load_after": list(os.getloadavg()),
            "rounds": finished,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("corpus", nargs="?", choices=sorted(CORPORA))
    parser.add_argument("out", nargs="?", type=Path, help="where to write the evaluation")
    parser.add_argument("--seeds", help="comma-separated seeds of the corpus, instead of all")
    parser.add_argument("--jobs", type=int, default=JOBS, help="rounds run at once")
    parser.add_argument("--round", type=int, help=argparse.SUPPRESS)  # one round, to stdout
    parser.add_argument(
        "--classify",
        nargs=2,
        type=Path,
        metavar=("EVALUATION", "OUT"),
        help="reclassify an evaluation into a new, derived artifact",
    )
    args = parser.parse_args()
    if args.round is not None:
        print(json.dumps(run_round(args.round)))
        return 0
    if args.classify is not None:
        source, out = args.classify
        if out.exists():
            parser.error(f"{out} exists: an evaluation is never overwritten")
        derived = reclassify(source)
        _write(out, derived)
        return _report(derived)
    if args.corpus is None or args.out is None:
        parser.error("a corpus and an output path are needed")
    seeds = list(CORPORA[args.corpus])
    if args.corpus == "acceptance":
        if args.seeds:
            parser.error("the acceptance corpus runs whole: --seeds is refused")
        if _provenance()["dirty"]:
            parser.error("the acceptance corpus runs from a clean tree")
    elif args.seeds:
        chosen = [int(s) for s in args.seeds.split(",")]
        if not set(chosen) <= set(seeds):
            parser.error(f"--seeds must be seeds of the {args.corpus} corpus")
        seeds = chosen
    if args.out.exists():
        parser.error(f"{args.out} exists: an evaluation is never overwritten")
    evaluation = evaluate(args.corpus, seeds, args.jobs)
    _write(args.out, evaluation)
    return _report(evaluation)


if __name__ == "__main__":
    sys.exit(main())
