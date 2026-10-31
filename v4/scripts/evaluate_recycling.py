"""Evaluate the recycling demo over a seeded corpus of mixed rounds (#201).

Each round stages ``recycling.mix(seed, OBJECTS)``, the legacy demo's five objects drawn
from its nine types, and sorts them into the bins with ``recycling.sort_all``, as the demo
runs: racing both arms with lookahead, recovering a failed attempt where that is safe,
and retrying an object released back onto the worktop once. The evaluation is one JSON
artifact:
- what it ran: the corpus, its seeds, the Geodude commit and whether the tree was
  dirty, every ``recycling.SOURCES`` distribution, and the host: its platform, Python,
  CPU model and count, OpenBLAS kernel (``recycling.blas_kernel``), and load;
- per round: the staged objects and their poses, and per staged object every attempt's
  outcome, with the deciding stage's detail and any recovery, where it finally rests,
  and its verdict (``verdict``);
- per round, on the runtime clock (``timing``): how long it ran, and how much of that
  no command was active; and the same per staged object, from its first grasp's first
  stage to its last attempt's landing judged, or else its release or grasp;
- the totals the bar is stated in: objects recycled, judged by where each finally rests,
  out of every object staged, including any never attempted because its round stopped;
  how many of them at the first attempt; and rounds cleared.

An evaluation in which any round errored is invalid (``"valid": false``): it has no
``summary``, only a ``partial_summary`` of the rounds that finished, and the script exits
nonzero. ``FORMAT`` versions the artifact.

Planning has a wall-clock budget, so a round run beside others may time out where it
would not alone. Rounds run in parallel, at most ``--jobs`` at once; every round that
records a planning timeout is then run again alone, and its verdict is that run's. Each
round records which run its verdict came from.

The corpora are fixed: ``development`` (seeds 0-39), for diagnosis and every fix, and
two frozen corpora, each run once, whole, from a clean tree, on the reference OpenBLAS
kernel (``recycling.REFERENCE_BLAS_KERNEL``, selected with ``OPENBLAS_CORETYPE``), or not
at all:
- ``acceptance`` (seeds 1000-1039) was #201's, and is spent: it is refused;
- ``cutover`` (seeds 2000-2039) was the Geodude 4.0 cutover's (#213), and is spent: it
  is refused. Its evaluation is ``evidence/cutover/``.

A spent corpus's seeds are refused one at a time too (``--round``). The development
corpus runs as::

    uv run python scripts/evaluate_recycling.py development out/development.json

``--seeds 3,7`` evaluates only those seeds of the development corpus.

``--classify EVALUATION OUT`` reclassifies an existing evaluation, without running
anything, after a change to ``classify``. It never touches ``EVALUATION``: ``OUT`` is a
derived artifact that records the original's SHA-256 and the classifying commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

FORMAT = 3
"""The evaluation artifact's format: raised whenever a field's meaning changes. 2: per
object, its attempts, where it finally rests, and its verdict, in place of one outcome
and its mechanism."""
CORPORA = {
    "development": range(0, 40),
    "acceptance": range(1000, 1040),
    "cutover": range(2000, 2040),
}
FROZEN = {"acceptance", "cutover"}
"""Corpora that run once, whole, from a clean tree."""
SPENT = {
    "acceptance": "it was run once, for #201, and stays that claim's evidence",
    "cutover": "it was run once, for #213, and stays that claim's evidence (evidence/cutover/)",
}
"""Frozen corpora already run: their outcomes are known, so they are never run again."""
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
        idle = Idle()
        runtime = recycling.runtime(package, scene)
        with RobotContext(package.description, runtime, sinks=[idle]) as ctx:
            result = recycling.sort_all(ctx, package)
            ended = ctx.now.time_ns
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {
        "seed": seed,
        "staged": staged,
        "outcomes": result["attempts"],
        "final": result["objects"],
        "timing": _timing(result["attempts"], idle.intervals, ended),
    }


class Idle:
    """A trace sink: the intervals of runtime time in which no command was active."""

    def __init__(self) -> None:
        self.active: set[str] = set()
        self.intervals: list[list[int]] = []
        self.time_ns = 0

    def __call__(self, record: Any) -> None:
        from ssrobot import TraceKind

        if record.kind is TraceKind.SUBMITTED:
            self.active.add(record.payload.execution)
        elif record.kind is TraceKind.STATUS and record.payload.state.terminal:
            self.active.discard(record.payload.execution)
        elif record.kind is TraceKind.STEPPED:
            if not self.active:
                if self.intervals and self.intervals[-1][1] == self.time_ns:
                    self.intervals[-1][1] = record.time_ns
                else:
                    self.intervals.append([self.time_ns, record.time_ns])
            self.time_ns = record.time_ns


def _idle_within(intervals: list[list[int]], start: int, end: int) -> int:
    return sum(max(0, min(b, end) - max(a, start)) for a, b in intervals)


def _timing(attempts: list[dict[str, Any]], idle: list[list[int]], ended: int) -> dict[str, Any]:
    """How long the round ran and how much of it no command was active, in total and per
    attempted object (see the module)."""
    objects: dict[str, Any] = {}
    for attempt in attempts:
        times = attempt.get("times_ns") or {}
        if times.get("grasp") is None:
            continue
        end = times["landed"] or times["released"] or times["grasp"]
        span = objects.setdefault(attempt["object"], {"start_ns": times["grasp"]})
        span["end_ns"] = end
    for span in objects.values():
        span["sim_ns"] = span["end_ns"] - span["start_ns"]
        span["idle_ns"] = _idle_within(idle, span["start_ns"], span["end_ns"])
    return {"sim_ns": ended, "idle_ns": _idle_within(idle, 0, ended), "objects": objects}


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


def verdict(attempts: list[dict[str, Any]], final: dict[str, Any]) -> str:
    """One staged object's verdict, by where it finally rests and its attempts:
    ``recycled`` at the first attempt, ``recycled_after_retry`` or
    ``recycled_after_recovery`` (released over its bin after a slip), or, resting outside
    every bin, the mechanism of its last attempt (``classify``)."""
    if final["bin"] is not None:
        if attempts and attempts[0]["succeeded"]:
            return "recycled"
        if any(a["succeeded"] for a in attempts):
            return "recycled_after_retry"
        return "recycled_after_recovery"
    if attempts and attempts[-1]["succeeded"]:
        return "left_its_bin"  # landed in it, and rests elsewhere by the end
    return classify(attempts[-1] if attempts else None)


def _summary(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    kinds: dict[str, Counter[bool]] = {}
    mechanisms: Counter[str] = Counter()
    recycled = first = staged = cleared = 0
    for round_ in rounds:
        verdicts = [o["verdict"] for o in round_["objects"]]
        staged += len(verdicts)
        recycled += sum(v.startswith("recycled") for v in verdicts)
        first += verdicts.count("recycled")
        cleared += all(v.startswith("recycled") for v in verdicts)
        for o in round_["objects"]:
            kinds.setdefault(o["kind"], Counter())[o["verdict"].startswith("recycled")] += 1
            if not o["verdict"].startswith("recycled"):
                mechanisms[o["verdict"]] += 1
    return {
        "objects_staged": staged,
        "objects_recycled": recycled,
        "recycled_at_first_attempt": first,
        "recycled_after_retry": sum(
            o["verdict"] == "recycled_after_retry" for r in rounds for o in r["objects"]
        ),
        "recycled_after_recovery": sum(
            o["verdict"] == "recycled_after_recovery" for r in rounds for o in r["objects"]
        ),
        "rounds": len(rounds),
        "rounds_cleared": cleared,
        "sim_ns": sum(r["timing"]["sim_ns"] for r in rounds),
        "idle_ns": sum(r["timing"]["idle_ns"] for r in rounds),
        "by_kind": {
            k: {"recycled": c[True], "staged": c[True] + c[False]} for k, c in sorted(kinds.items())
        },
        "mechanisms": dict(mechanisms.most_common()),
    }


def _objects(round_: dict[str, Any]) -> list[dict[str, Any]]:
    """Every staged object of a round, with its attempts, where it finally rests, and
    its verdict."""
    out = []
    for placement in round_["staged"]:
        name = placement["name"]
        attempts = [o for o in round_["outcomes"] if o["object"] == name]
        final = round_["final"][name]
        out.append(
            {
                "object": name,
                "kind": placement["kind"],
                "staged": placement,
                "verdict": verdict(attempts, final),
                "attempts": attempts,
                "final": final,
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
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": _cpu_model(),
            "blas_kernel": recycling.blas_kernel(),
        },
    }


def _cpu_model() -> str:
    """The CPU's model name, as the OS reports it, or ``unknown``."""
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return "unknown"


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
        for corpus, reason in SPENT.items():
            if args.round in CORPORA[corpus]:
                parser.error(f"seed {args.round} is of the spent {corpus} corpus: {reason}")
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
    if args.corpus in SPENT:
        parser.error(f"the {args.corpus} corpus is spent: {SPENT[args.corpus]}")
    if args.corpus in FROZEN:
        from geodude import recycling

        if args.seeds:
            parser.error(f"the {args.corpus} corpus runs whole: --seeds is refused")
        if _provenance()["dirty"]:
            parser.error(f"the {args.corpus} corpus runs from a clean tree")
        kernel = recycling.blas_kernel()
        if kernel != recycling.REFERENCE_BLAS_KERNEL:
            parser.error(
                f"the {args.corpus} corpus runs on the {recycling.REFERENCE_BLAS_KERNEL} "
                f"OpenBLAS kernel, not {kernel}: set "
                f"OPENBLAS_CORETYPE={recycling.REFERENCE_BLAS_KERNEL}"
            )
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
