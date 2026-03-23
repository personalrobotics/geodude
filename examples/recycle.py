#!/usr/bin/env python3
"""Recycling demo — bimanual pick and place with Geodude.

Demonstrates the clean student API::

    robot.pickup("can_0")
    robot.place("recycle_bin_0")
    robot.go_home()

TSR generation, planning, execution, and recovery are all automatic.

Usage:
    uv run mjpython examples/recycle.py
    uv run mjpython examples/recycle.py --physics
    uv run mjpython examples/recycle.py --headless --cycles 5
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

# Bin positions
RIGHT_BIN_POS = [0.85, -0.35, 0.01]
LEFT_BIN_POS = [-0.85, -0.35, 0.01]


def sample_can_position(worktop_pos):
    """Random position on the worktop surface."""
    return [
        worktop_pos[0] + random.uniform(-0.15, 0.15),
        worktop_pos[1] + random.uniform(-0.08, 0.08),
        worktop_pos[2] + 0.123 / 2 + 0.005,  # can half-height + small gap
    ]


def main():
    parser = argparse.ArgumentParser(description="Geodude recycling demo")
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cycles", type=int, default=5)
    args = parser.parse_args()

    mode = "Physics" if args.physics else "Kinematic"
    print(f"\n{'='*60}")
    print(f"  Geodude Recycling Demo — {mode} Mode")
    print(f"{'='*60}\n")

    robot = Geodude(objects={"can": 1, "recycle_bin": 2})

    # Get worktop position
    worktop_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    worktop_pos = robot.data.site_xpos[worktop_id].copy()

    # Place bins and set arms to ready
    robot.env.registry.activate("recycle_bin", pos=RIGHT_BIN_POS)
    robot.env.registry.activate("recycle_bin", pos=LEFT_BIN_POS)
    for side, arm in [("left", robot.left_arm), ("right", robot.right_arm)]:
        q = np.array(robot.named_poses["ready"][side])
        for i, idx in enumerate(arm.joint_qpos_indices):
            robot.data.qpos[idx] = q[i]

    # Spawn first can
    can_pos = sample_can_position(worktop_pos)
    robot.env.registry.activate("can", pos=can_pos)
    mujoco.mj_forward(robot.model, robot.data)

    with robot.sim(physics=args.physics, headless=args.headless) as ctx:
        for cycle in range(1, args.cycles + 1):
            if not ctx.is_running():
                break

            print(f"\n--- Cycle {cycle} ---")
            can_pose = robot.get_object_pose("can_0")
            print(f"  Can at {can_pose[:3, 3].round(3)}")

            # ============================
            # THE STUDENT API — 3 lines
            # ============================
            if not robot.pickup("can_0"):
                print("  Pickup FAILED")
                robot.go_home()
                continue

            # Determine which arm picked up, place in same-side bin
            for side in ("left", "right"):
                if robot.grasp_manager.get_grasped_by(side):
                    holding_side = side
                    break
            bin_name = "recycle_bin_0" if holding_side == "right" else "recycle_bin_1"

            if not robot.place(bin_name):
                print("  Place FAILED")
                robot.go_home()
                continue

            # Hide can and return home
            robot.env.registry.hide("can_0")
            mujoco.mj_forward(robot.model, robot.data)
            ctx.sync()
            print(f"  Dropped into {holding_side} bin")

            robot.go_home()

            # Spawn next can
            if cycle < args.cycles:
                can_pos = sample_can_position(worktop_pos)
                robot.env.registry.activate("can", pos=can_pos)
                mujoco.mj_forward(robot.model, robot.data)

        print(f"\nCompleted {cycle} cycles.")


if __name__ == "__main__":
    main()
