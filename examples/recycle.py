#!/usr/bin/env python3
"""Recycling demo — bimanual pick and place with Geodude.

Demonstrates the full student API::

    robot.pickup()                  # pick up anything
    robot.pickup("can")             # pick up any can
    robot.right.pickup("can")      # right arm picks any can
    robot.place("recycle_bin")      # place in any bin
    robot.go_home()

Usage:
    uv run mjpython examples/recycle.py
    uv run mjpython examples/recycle.py --physics
    uv run mjpython examples/recycle.py --headless --cycles 10
"""

from __future__ import annotations

import argparse
import logging

import mujoco
import numpy as np
from asset_manager import AssetManager
from prl_assets import OBJECTS_DIR
from tsr.placement import TablePlacer

from geodude import Geodude

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logging.getLogger("toppra").setLevel(logging.WARNING)

# Bin positions: floor-standing, one per side
RIGHT_BIN_POS = [0.85, -0.35, 0.01]
LEFT_BIN_POS = [-0.85, -0.35, 0.01]

# Object geometry from prl_assets
_ASSETS = AssetManager(str(OBJECTS_DIR))
_CAN_GP = _ASSETS.get("can")["geometric_properties"]
_SPAM_GP = _ASSETS.get("potted_meat_can")["geometric_properties"]


def _sample_placements(worktop_pos, objects, min_sep=0.10):
    """Sample non-overlapping placements on the worktop using TSR.

    Args:
        worktop_pos: [x, y, z] of worktop surface center.
        objects: list of (count, geometric_properties) tuples.
        min_sep: minimum XY separation between objects.

    Returns:
        List of (pos, half_height) tuples.
    """
    table_hx, table_hy = 0.15, 0.08
    placer = TablePlacer(table_hx, table_hy)

    table_surface = np.eye(4)
    table_surface[:3, 3] = worktop_pos

    placements = []
    for count, gp in objects:
        if gp["type"] == "cylinder":
            templates = placer.place_cylinder(gp["radius"], gp["height"])
            half_h = gp["height"] / 2
        elif gp["type"] == "box":
            templates = placer.place_box(gp["size"][0], gp["size"][1], gp["size"][2])
            half_h = gp["size"][2] / 2
        else:
            continue

        for _ in range(count):
            tsr = templates[0].instantiate(table_surface)
            for _attempt in range(50):
                pose = tsr.sample()
                pos = pose[:3, 3]
                if all(np.linalg.norm(pos[:2] - np.array(p[:2])) > min_sep
                       for p, _ in placements):
                    break
            placements.append((list(pos), half_h))

    return placements


def main():
    parser = argparse.ArgumentParser(description="Geodude recycling demo")
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--cans", type=int, default=3)
    args = parser.parse_args()

    mode = "Physics" if args.physics else "Kinematic"
    print(f"\n{'='*60}")
    print(f"  Geodude Recycling Demo — {mode} Mode")
    print(f"  {args.cans} cans, up to {args.cycles} cycles")
    print(f"{'='*60}\n")

    robot = Geodude(objects={"can": args.cans, "potted_meat_can": 1, "recycle_bin": 2})

    # Worktop position for spawning
    worktop_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    worktop_pos = robot.data.site_xpos[worktop_id].copy()

    # Place bins, raise bases to midpoint, set arms to ready
    robot.env.registry.activate("recycle_bin", pos=RIGHT_BIN_POS)
    robot.env.registry.activate("recycle_bin", pos=LEFT_BIN_POS)
    for base in [robot.left_base, robot.right_base]:
        if base is not None:
            base.set_height(0.25)  # midpoint of 0–0.5m range
    for side, arm in [("left", robot.left_arm), ("right", robot.right_arm)]:
        q = np.array(robot.named_poses["ready"][side])
        for i, idx in enumerate(arm.joint_qpos_indices):
            robot.data.qpos[idx] = q[i]

    # Spawn objects on worktop using TSR placement (non-overlapping)
    placements = _sample_placements(worktop_pos, [
        (args.cans, _CAN_GP),
        (1, _SPAM_GP),
    ])
    can_idx = 0
    spam_idx = 0
    for pos, _ in placements:
        if can_idx < args.cans:
            robot.env.registry.activate("can", pos=pos)
            can_idx += 1
        else:
            robot.env.registry.activate("potted_meat_can", pos=pos)
            spam_idx += 1
    mujoco.mj_forward(robot.model, robot.data)

    with robot.sim(physics=args.physics, headless=args.headless) as ctx:
        for cycle in range(1, args.cycles + 1):
            if not ctx.is_running():
                break

            print(f"\n--- Cycle {cycle} ---")

            # ====================================
            # THE STUDENT API
            # ====================================

            # Pick up anything on the table
            if not robot.pickup():
                print("  Nothing left to pick up")
                break

            # Which arm grabbed what?
            for side in ("left", "right"):
                held = list(robot.grasp_manager.get_grasped_by(side))
                if held:
                    holding_side = side
                    held_object = held[0]
                    break

            print(f"  {holding_side} arm picked up {held_object}")

            # Place in same-side bin
            bin_type = "recycle_bin_0" if holding_side == "right" else "recycle_bin_1"
            if not robot.place(bin_type):
                print("  Place FAILED")
                robot.go_home()
                continue

            # Hide object and go home
            robot.env.registry.hide(held_object)
            mujoco.mj_forward(robot.model, robot.data)
            ctx.sync()
            print(f"  Placed in {holding_side} bin")

            robot.go_home()

        print(f"\nCompleted {cycle} cycles.")


if __name__ == "__main__":
    main()
