"""How long planning takes never changes what Geodude does.

The recycling demo plans on workers while the robot moves: each can's race while the
previous one is carried, and a skill's next motion. A driver that steps the clock
while every skill it drives waits only on such a plan lets simulated time pass for as
long as planning takes in wall-clock time, so the run would depend on the host. This
test compares the four-can round of ``test_four_cans`` (the session's ``four_cans``
recording) with the same round run again with every planner call delayed by 0 to 3 s,
chosen per request from its fingerprint so that plans finish in another order, and the
first call always 3 s, so the longest delay is exercised. The longest delay outlasts
any unpaced skill stage of the round, so a skill waiting on a delayed plan would see the
clock step if its driver stepped it.

The evidence is ``latency.json``, kept by this test: for each run, the cans' outcomes,
the SHA-256 of its outcome file (``recycling.json``, a ``RecyclingRun``, which cites its
bundle by digest), the runtime time the trace ends, and every trace record kind's
count and digest of its records (runtime time and payload, in order). Both runs must
agree on all of it: their outcome files are byte-identical, no field excluded.

It is the suite's determinism gate: the delayed run repeats the round under a different
planning order, so a nondeterministic round fails it, and no other test repeats a round
only to compare it with itself.

Reproduce with ``uv run pytest tests/test_planning_latency.py``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from ssrobot import loads, open_bundle
from ssrobot._wire import fingerprint
from ssrobot.planning import _cbirrt

from geodude import recycling
from geodude.outcomes import RecyclingRun
from tests.conftest import FOUR_CANS

DELAYS_S = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
"""The delays a planner call may get; 3 s outlasts every stage of the unpaced round."""


def _round(code: int, out: Path) -> dict[str, Any]:
    outcome = (out / "recycling.json").read_bytes()
    run = loads(outcome, RecyclingRun)
    records = open_bundle(out / "bundle").records
    digests: dict[str, Any] = defaultdict(hashlib.sha256)
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        payload = "" if r.payload is None else fingerprint(r.payload)
        digests[r.kind.value].update(f"{r.time_ns}:{payload}\n".encode())
        counts[r.kind.value] += 1
    return {
        "code": code,
        "outcomes": [[a.object, a.arm, a.succeeded] for a in run.attempts],
        "outcome_sha256": hashlib.sha256(outcome).hexdigest(),
        "end_ns": records[-1].time_ns,
        "records": {k: [counts[k], digests[k].hexdigest()] for k in sorted(digests)},
    }


def test_planning_latency_changes_nothing(
    artifacts: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    four_cans: tuple[int, Path],
) -> None:
    prompt = _round(*four_cans)

    plan = _cbirrt.CBiRRT.plan
    delayed: list[float] = []
    lock = threading.Lock()

    def slow(self: Any, request: Any, scene: Any, **kwargs: Any) -> Any:
        # Spread by the request's fingerprint, so plans finish in another order; the
        # first call always gets the longest, so it is exercised whatever the
        # fingerprints are (a numerically equal request can hash differently).
        with lock:
            first = not delayed
            delay = (
                DELAYS_S[-1]
                if first
                else DELAYS_S[int(fingerprint(request)[:8], 16) % len(DELAYS_S)]
            )
            delayed.append(delay)
        time.sleep(delay)
        return plan(self, request, scene, **kwargs)

    monkeypatch.setattr(_cbirrt.CBiRRT, "plan", slow)
    out = tmp_path / "delayed"
    late = _round(recycling.main([*FOUR_CANS, "--record", str(out)]), out)
    (artifacts / "latency.json").write_text(
        json.dumps({"delays_s": DELAYS_S, "prompt": prompt, "delayed": late}, indent=2) + "\n"
    )

    assert max(delayed) == DELAYS_S[-1], "no call got the longest delay"
    assert late == prompt
    assert prompt["code"] == 0, prompt["outcomes"]
