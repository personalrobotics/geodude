"""Geodude 4.0 recycles each of the legacy demo's nine object types (#201, step 4).

For each type in ``recycling.MIX_KINDS``, one instance is staged alone on the worktop at
``SEED`` and recycled with ``recycling.sort_all``: both arms race for it, the winner grasps
it and drops it into its own bin, and its landing is observed. A type passes when its
first attempt succeeds, every stage of it, and it comes to rest inside a bin.

The evidence is ``object_types.json``: per type, the arm, every stage of the grasp, drop,
and landing, and where the object rests in its bin. The run is deterministic: the same
seed gives the same file.

Reproduce with ``uv run pytest tests/test_object_types.py``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from ssrobot import RobotContext, load_installed_package

from geodude import recycling
from geodude.scene import stage_scene

SEED = 0


def _recycle_alone(kind: str) -> dict[str, Any]:
    package = load_installed_package("geodude_assets")
    with tempfile.TemporaryDirectory() as work:
        scene = stage_scene(
            Path(work) / "scene", package, objects={kind: 1}, fixtures=recycling.BINS, seed=SEED
        )
        with RobotContext(package.description, recycling.runtime(package, scene)) as ctx:
            result = recycling.sort_all(ctx, package)
    first = result["attempts"][0]
    return {
        "arm": first["arm"],
        "grasp": first.get("grasp"),
        "drop": first.get("drop"),
        "landing": first.get("landing"),
        "failure": first.get("failure"),
        "position_in_bin_m": first.get("position_in_bin_m"),
        "succeeded": first["succeeded"],
        "bin": result["objects"][f"{kind}_0"]["bin"],
    }


def test_geodude_recycles_every_legacy_object_type(artifacts: Path) -> None:
    types = {kind: _recycle_alone(kind) for kind in recycling.MIX_KINDS}
    (artifacts / "object_types.json").write_text(json.dumps(types, indent=2) + "\n")

    failed = {k: t["failure"] for k, t in types.items() if not (t["succeeded"] and t["bin"])}
    assert not failed, failed
