"""Find where two variants' recycling rounds first diverge (#201), for diagnosis only.

``round SEED OUT`` runs one round of the development corpus, as
``evaluate_recycling.py`` does (``recycling.sort_all``), with every planner wrapped to
record each plan it makes. ``OUT`` gets:
- ``round.json``: the staged objects and ``sort_all``'s report: per attempt, the race,
  every stage, and the checks;
- ``plans.jsonl``: per plan, in the order made, the snapshot's fingerprint and
  revision, the request's fingerprint and seed, the group, the goal's fingerprint, the
  status, and the path's, trajectory's and validations' fingerprints, with each
  validator's identity;
- ``snapshots/`` and ``results/``: each planning snapshot and each result, by
  fingerprint, so a plan can be validated in another run's snapshot.

``compare A B`` finds the first attempt and stage at which two rounds' decisions differ,
and the plans made for it in each.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from ssrobot._wire import dumps, fingerprint

_MADE = [0]  # plans recorded so far in this process
SEEDS = range(0, 40)  # the development corpus only: the acceptance corpus is never run here
OBJECTS = 5


class Recording:
    """A planner that records every plan its ``inner`` makes."""

    def __init__(self, inner: Any, label: str, out: Path, lock: threading.Lock) -> None:
        self._inner, self._label, self._out, self._lock = inner, label, out, lock

    @property
    def name(self) -> str:
        return str(self._inner.name)

    def capabilities(self, request: Any) -> Any:
        return self._inner.capabilities(request)

    def validator(self, request: Any, scene: Any) -> Any:
        return self._inner.validator(request, scene)

    def plan(self, request: Any, scene: Any, *, abort: Any = None) -> Any:
        result = self._inner.plan(request, scene, abort=abort)
        snapshot = scene.snapshot
        entry = {
            "planner": self._label,
            "snapshot": request.snapshot,
            "materialized": fingerprint(snapshot),
            "revision": request.revision,
            "time_ns": snapshot.stamp.time_ns,
            "request": fingerprint(request),
            "seed": request.seed,
            "group": request.group,
            "goal": fingerprint(request.goal),
            "goal_kind": type(request.goal).__name__,
            "clearance_m": request.clearance_m,
            "status": result.status.value,
            "result": fingerprint(result),
            "path": None if result.path is None else fingerprint(result.path),
            "trajectory": None if result.trajectory is None else fingerprint(result.trajectory),
            "validation": [
                {"motion": v.motion, "valid": v.valid, "validator": fingerprint(v.validator)}
                for v in result.validation
            ],
        }
        with self._lock:
            entry["order"] = _MADE[0]
            _MADE[0] += 1
            for kind, obj, key in (
                ("snapshots", snapshot, entry["materialized"]),
                ("results", result, entry["result"]),
            ):
                path = self._out / kind / f"{key}.json"
                if not path.exists():
                    path.write_text(dumps(obj))
            with (self._out / "plans.jsonl").open("a") as plans:
                plans.write(json.dumps(entry) + "\n")
        return result


def run_round(seed: int, out: Path) -> None:
    from ssrobot import RobotContext, load_installed_package

    from geodude import recycling
    from geodude.scene import stage_scene

    out.mkdir(parents=True)
    (out / "snapshots").mkdir()
    (out / "results").mkdir()
    lock = threading.Lock()
    made = recycling.planners

    def recorded(package: Any, arm: Any) -> tuple[Any, Any]:
        planner, checker = made(package, arm)
        side = arm.manipulator
        return (
            Recording(planner, f"{side}:planner", out, lock),
            Recording(checker, f"{side}:checker", out, lock),
        )

    recycling.planners = recorded  # type: ignore[assignment]
    package = load_installed_package("geodude_assets")
    work = Path(tempfile.mkdtemp(prefix="geodude-diag-"))
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
            result = recycling.sort_all(ctx, package)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    report = {"seed": seed, "sources": recycling.sources(), "staged": staged, **result}
    (out / "round.json").write_text(json.dumps(report, indent=1, default=str) + "\n")


def _decisions(round_: dict[str, Any]) -> list[tuple[str, Any]]:
    """A round's decisions in order, without times: per attempt, the object, the arm
    chosen, and each stage's name and outcome."""
    out: list[tuple[str, Any]] = []
    for attempt in round_["attempts"]:
        key = f"{attempt['object']}#{attempt.get('attempt', 0)}"
        out.append((key, ("arm", attempt["arm"])))
        for part in ("grasp", "drop", "landing"):
            for stage in attempt.get(part) or []:
                out.append((key, (part, *stage)))
        out.append((key, ("succeeded", attempt["succeeded"])))
    return out


def compare(a: Path, b: Path) -> dict[str, Any]:
    ra, rb = (json.loads((p / "round.json").read_text()) for p in (a, b))
    da, db = _decisions(ra), _decisions(rb)
    first = next((i for i, (x, y) in enumerate(zip(da, db, strict=False)) if x != y), None)
    if first is None and len(da) == len(db):
        return {"identical": True, "decisions": len(da)}
    first = min(len(da), len(db)) if first is None else first
    return {
        "identical": False,
        "index": first,
        "before": da[max(0, first - 3) : first],
        "a": da[first : first + 3],
        "b": db[first : first + 3],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("round")
    one.add_argument("seed", type=int)
    one.add_argument("out", type=Path)
    two = sub.add_parser("compare")
    two.add_argument("a", type=Path)
    two.add_argument("b", type=Path)
    args = parser.parse_args()
    if args.command == "round":
        if args.seed not in SEEDS:
            parser.error("only development seeds (0-39) are diagnosed")
        if args.out.exists():
            parser.error(f"{args.out} exists: a run is never overwritten")
        run_round(args.seed, args.out)
        return 0
    print(json.dumps(compare(args.a, args.b), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
