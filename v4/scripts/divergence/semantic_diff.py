"""First semantic divergence between two diagnostic rounds, stage names normalized
(A's pre-lift ``acquired`` is B's ``obstructed``; B's post-lift ``acquired`` dropped)."""

import json
import sys
from pathlib import Path


def decisions(path: Path, variant: str) -> list:
    r = json.loads((path / "round.json").read_text())
    out = []
    for i, at in enumerate(r["attempts"]):
        key = f"{i}:{at['object']}#{at.get('attempt', 0)}"
        race = at.get("race") or {}
        out.append(
            (
                key,
                "race",
                race.get("selected"),
                tuple((c["label"], c["outcome"]) for c in race.get("candidates", [])),
            )
        )
        out.append((key, "arm", at["arm"]))
        for part in ("grasp", "drop"):
            seen_lift = False
            for name, ok in at.get(part) or []:
                if part == "grasp" and variant == "a" and name == "acquired":
                    name = "obstructed"
                if part == "grasp" and name == "lift":
                    seen_lift = True
                if part == "grasp" and variant != "a" and name == "acquired" and seen_lift:
                    continue
                out.append((key, part, name, ok))
        landing = at.get("landing")
        if landing:
            out.append((key, "landing", landing if isinstance(landing, list) else str(landing)))
        rec = at.get("recovery")
        if rec:
            out.append((key, "recovery", rec.get("action"), rec.get("released"), rec.get("in_bin")))
        out.append((key, "succeeded", at["succeeded"]))
    return out


a, b = Path(sys.argv[1]), Path(sys.argv[2])
da, db = decisions(a, "a"), decisions(b, "b")
first = next(
    (i for i, (x, y) in enumerate(zip(da, db, strict=False)) if x != y), min(len(da), len(db))
)
for x in da[max(0, first - 2) : first]:
    print("  same", x)
print("A:", da[first : first + 4])
print("B:", db[first : first + 4])
