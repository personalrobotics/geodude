"""Summarize grip_probe.py's records over a run (geodude_assets#36), for diagnosis only.

``grip_summary.py RUN_DIR [SEED ...]`` prints, per attempt in the named seeds (default:
those with a failure): the object, how it ended, the links holding it at ``attach`` with
their normal forces, the largest displacement per stage before release, the first slip
past ``SLIP_MM`` and the stage it began in, and how each finger's passive joints
(spring link, follower) and its driver moved from ``attach`` to that slip. ``--all``
tabulates every attempt's peak displacement by object type and outcome instead.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def stages(samples: list[dict]) -> list[tuple[str, float, float]]:
    out: list[list] = []
    for x in samples:
        if x["moved_mm"] is None:
            continue
        if not out or out[-1][0] != x["stage"]:
            out.append([x["stage"], 0.0, 0.0])
        out[-1][1] = max(out[-1][1], x["moved_mm"])
        out[-1][2] = max(out[-1][2], x["turned_rad"])
    return [(s, round(mm, 1), round(r, 3)) for s, mm, r in out]


def detail(path: Path) -> None:
    g = json.loads(path.read_text())
    failed = {f[0]: f[1:] for f in g["failures"]}
    print(f"== seed {g['seed']}: failures {g['failures']}")
    for a in g["attempts"]:
        att, slip = a["at_attach"], a["first_slip"]
        print(
            f"  {a['object']} ({a.get('hand')}): {'FAILED ' + str(failed.get(a['object'])) if a['object'] in failed else 'ok'}"
        )
        if att:
            print(
                f"    at attach t={att['t']}: contacts {att['contacts_n']} actuator {att['actuator_n']}"
            )
        print(f"    peak moved per stage: {stages(a['samples'])}")
        if slip and att:
            dj = {
                k: round(slip["joints"][k] - att["joints"][k], 3)
                for k in att["joints"]
                if abs(slip["joints"][k] - att["joints"][k]) > 0.002
            }
            print(
                f"    first slip t={slip['t']} in {slip['stage']}: moved {slip['moved_mm']} mm; "
                f"contacts {slip['contacts_n']}; joint change since attach {dj}"
            )
        lost = a["monitor_lost"]
        if lost:
            print(f"    monitor lost t={lost['t']} in {lost['stage']}, moved {lost['moved_mm']} mm")


def table(run: Path) -> None:
    rows: dict = defaultdict(list)
    for path in sorted(run.glob("seed-*.json")):
        g = json.loads(path.read_text())
        failed = {f[0] for f in g["failures"]}
        for a in g["attempts"]:
            kind = a["object"].rsplit("_", 1)[0]
            peak = max([x["moved_mm"] or 0 for x in a["samples"]] + [0])
            rows[(kind, a["object"] in failed)].append(peak)
    for (kind, failed), peaks in sorted(rows.items()):
        peaks.sort()
        print(
            f"  {kind:16} {'failed' if failed else 'ok':6} n={len(peaks):3} peak moved mm: "
            f"median {peaks[len(peaks) // 2]:.1f}, max {peaks[-1]:.1f}"
        )


if __name__ == "__main__":
    run = Path(sys.argv[1])
    if "--all" in sys.argv:
        table(run)
    else:
        seeds = [int(s) for s in sys.argv[2:]]
        paths = [run / f"seed-{s}.json" for s in seeds] or sorted(
            p for p in run.glob("seed-*.json") if json.loads(p.read_text())["failures"]
        )
        for p in paths:
            detail(p)
