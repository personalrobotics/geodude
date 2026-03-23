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
import random

import mujoco
import numpy as np

from geodude import Geodude

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logging.getLogger("toppra").setLevel(logging.WARNING)

# Bin positions: floor-standing, one per side
RIGHT_BIN_POS = [0.85, -0.35, 0.01]
LEFT_BIN_POS = [-0.85, -0.35, 0.01]

# Can half-height for spawning on worktop surface
CAN_HALF_HEIGHT = 0.123 / 2


def random_worktop_pos(worktop_pos):
    """Random position on the worktop surface for a can."""
    return [
        worktop_pos[0] + random.uniform(-0.15, 0.15),
        worktop_pos[1] + random.uniform(-0.08, 0.08),
        worktop_pos[2] + CAN_HALF_HEIGHT + 0.005,
    ]


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

    robot = Geodude(objects={"can": args.cans, "recycle_bin": 2})

    # Worktop position for spawning
    worktop_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    worktop_pos = robot.data.site_xpos[worktop_id].copy()

    # Place bins and set arms to ready
    robot.env.registry.activate("recycle_bin", pos=RIGHT_BIN_POS)
    robot.env.registry.activate("recycle_bin", pos=LEFT_BIN_POS)
    for side, arm in [("left", robot.left_arm), ("right", robot.right_arm)]:
        q = np.array(robot.named_poses["ready"][side])
        for i, idx in enumerate(arm.joint_qpos_indices):
            robot.data.qpos[idx] = q[i]

    # Spawn all cans
    for _ in range(args.cans):
        robot.env.registry.activate("can", pos=random_worktop_pos(worktop_pos))
    mujoco.mj_forward(robot.model, robot.data)

    with robot.sim(physics=args.physics, headless=args.headless) as ctx:
        for cycle in range(1, args.cycles + 1):
            if not ctx.is_running():
                break

            print(f"\n--- Cycle {cycle} ---")

            # ====================================
            # THE STUDENT API
            # ====================================

            # Pick up any can (planner picks the easiest one)
            if not robot.pickup("can"):
                print("  No more cans to pick up")
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
