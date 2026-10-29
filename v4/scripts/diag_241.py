"""Diagnostic for #241, not for merge: seed 0's sort_all round once, dumping every
planning request (canonical JSON, by fingerprint) and each call's result fingerprint,
with the host's CPU and the native libraries' runtime configuration."""

import contextlib
import io
import json
import platform
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from ssrobot._wire import dumps, fingerprint
from ssrobot.planning import _cbirrt

from geodude import recycling

out = Path(sys.argv[1])
(out / "requests").mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
calls = []
plan = _cbirrt.CBiRRT.plan


def dumped(self, request, scene, **kwargs):
    fp = fingerprint(request)
    with lock:
        (out / "requests" / f"{fp}.json").write_text(dumps(request))
    result = plan(self, request, scene, **kwargs)
    with lock:
        calls.append({"request": fp, "result": fingerprint(result), "thread": threading.current_thread().name})
    return result


_cbirrt.CBiRRT.plan = dumped
code = recycling.main(["--seed", "0", "--mix", "5", "--record", str(out / "run")])
(out / "calls.json").write_text(json.dumps(calls, indent=1))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    np.show_runtime()
    np.show_config()
lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
report = json.loads((out / "run" / "recycling.json").read_text())
(out / "host.json").write_text(json.dumps({
    "code": code,
    "cleared": report["cleared"],
    "bins": {o: f["bin"] for o, f in report["objects"].items()},
    "first_stage_request": report["attempts"][0]["race"]["chains"]["chains"][0]["attempts"][0]["stages"][0]["request"],
    "platform": platform.platform(),
    "lscpu": lscpu,
    "numpy": buf.getvalue(),
}, indent=1))
print(json.dumps({"cleared": report["cleared"], "code": code}))
