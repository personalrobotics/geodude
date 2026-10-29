"""Diagnostic for #241, not for merge: seed 0's round, logging regions.pose's
intermediates bit for bit (MuJoCo's quaternion, its norm, the result) and the first
race stage's request fingerprint."""

import json
import os
import sys
import threading
from pathlib import Path

import mujoco
import numpy as np
from ssrobot._wire import fingerprint
from ssrobot.planning import _cbirrt

from geodude import recycling, regions

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
poses = []
original = regions.pose


def logged(transform):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(transform[:3, :3], dtype=float).flatten())
    norm = np.linalg.norm(quat)
    result = original(transform)
    with lock:
        if len(poses) < 400:
            poses.append({
                "matrix": [float(v).hex() for v in np.asarray(transform, dtype=float).flatten()],
                "mat2quat": [float(v).hex() for v in quat],
                "norm": float(norm).hex(),
                "norm_python": (sum(float(v) * float(v) for v in quat) ** 0.5).hex(),
                "dot": float(quat.dot(quat)).hex(),
                "result": [float(v).hex() for v in result.quat_wxyz],
            })
    return result


regions.pose = logged
recycling.main(["--seed", "0", "--mix", "5", "--record", str(out / "run")])
report = json.loads((out / "run" / "recycling.json").read_text())
(out / "probe.json").write_text(json.dumps({
    "coretype": os.environ.get("OPENBLAS_CORETYPE", "default"),
    "cleared": report["cleared"],
    "first_stage_request": report["attempts"][0]["race"]["chains"]["chains"][0]["attempts"][0]["stages"][0]["request"],
    "poses": poses,
}))
