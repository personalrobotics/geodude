#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Validate the geodude_assets Robotiq 2F-140 against mj_manipulator's framework.

This is the geodude-side companion to
``mj_manipulator/scripts/validate_gripper.py``. The 2F-140 model lives
in ``geodude_assets``, a workspace-only package, so its registration
can't live in mj_manipulator without pulling in an optional dep. This
script registers the 2F-140 with mj_manipulator's ``GRIPPERS`` dict at
runtime and delegates to the same ``validate(...)`` machinery.

Usage::

    uv run python scripts/validate_2f140.py
    uv run python scripts/validate_2f140.py --object spam_can --samples 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _register_2f140_gripper() -> None:
    """Add the geodude_assets 2F-140 to mj_manipulator's GRIPPERS registry."""
    import geodude_assets

    # Import mj_manipulator's scripts as modules. They live next to the
    # mj_manipulator package source under its repo root, not under the
    # installed package. Walk up from mj_manipulator.__file__ to find
    # the repo root, then add scripts/ to sys.path.
    import mj_manipulator as _mjm

    mjm_scripts = Path(_mjm.__file__).resolve().parents[2] / "scripts"
    if not mjm_scripts.is_dir():
        raise RuntimeError(
            f"Could not find mj_manipulator/scripts/ at {mjm_scripts}. "
            "This script expects a workspace layout where mj_manipulator's "
            "scripts/ is a sibling of its src/."
        )
    sys.path.insert(0, str(mjm_scripts))

    # Import once loaded; module-level effects (registry population etc.)
    # happen at import time.
    import visualize_grasps as vg

    _2f140_xml = Path(geodude_assets.__file__).parent / "models" / "robotiq_2f140" / "2f140.xml"
    if not _2f140_xml.is_file():
        raise FileNotFoundError(f"2F-140 XML not found at {_2f140_xml}")

    vg.GRIPPERS["robotiq_2f140"] = vg.GripperSpec(
        # grasp_site is already baked into the geodude_assets 2f140.xml at
        # pos=[0, 0, 0.100] with the -90° about-z TSR-alignment rotation.
        xml_path_resolver=lambda: _2f140_xml,
        hand_type="robotiq_2f140",
        add_grasp_site=False,
        grasp_site_name="grasp_site",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", default="can", help="prl_assets object type (default: 'can').")
    parser.add_argument("--samples", type=int, default=50, help="Samples per TSR template (default: 50).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    _register_2f140_gripper()

    # Importable now that the scripts dir is on sys.path.
    import validate_gripper as vg_val

    ok = vg_val.validate("robotiq_2f140", args.object, args.samples, args.seed)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
