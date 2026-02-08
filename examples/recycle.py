#!/usr/bin/env python3
"""Recycling Demo - Pick up objects and place them in bins.

Demonstrates the high-level manipulation primitives API:
- robot.pickup() with automatic affordance discovery
- robot.place() with automatic destination planning
- robot.get_pickable_objects() to find what can be picked up
- Same code works in simulation and hardware

For the lower-level manual approach, see recycle_manual.py.

Usage:
    uv run mjpython examples/recycle.py            # Kinematic mode (default)
    uv run mjpython examples/recycle.py --physics  # Physics simulation
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from geodude import Geodude

# Configure logging to see pickup/place introspection
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
)
from tsr import TSR
from tsr.core.tsr_primitive import load_template_file

TSR_DIR = Path(__file__).parent.parent / "tsr_templates"

# Bin positions (floor-standing, 0.80m tall bins)
# x=±0.85 to clear table collision (table extends to x=±0.65, bins are 0.25m wide)
# Small z offset (0.01m) to avoid interpenetration with floor at startup
RIGHT_BIN_POS = [0.85, -0.35, 0.01]
LEFT_BIN_POS = [-0.85, -0.35, 0.01]


def sample_can_placement(robot):
    """Sample random can placement on worktop."""
    import mujoco
    import random

    templates = [
        load_template_file(str(TSR_DIR / "places" / "can_on_table_upright.yaml")),
        load_template_file(str(TSR_DIR / "places" / "can_on_table_flipped.yaml")),
    ]
    template = random.choice(templates)

    worktop_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    worktop_pos = robot.data.site_xpos[worktop_id].copy()

    Bw = template.Bw.copy()
    Bw[0, :] = [-0.3, 0.3]
    Bw[1, :] = [-0.15, 0.15]

    tsr = TSR(Bw=Bw)
    xyzrpy = tsr.sample_xyzrpy()
    pos = worktop_pos + xyzrpy[:3]

    rot = TSR.rpy_to_rot(xyzrpy[3:6])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())

    return pos, quat


def main():
    import mujoco
    import traceback

    parser = argparse.ArgumentParser(description="Recycling demo with manipulation primitives")
    parser.add_argument("--physics", action="store_true", help="Enable physics simulation")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--cycles", type=int, default=5, help="Number of cycles to run")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging")
    args = parser.parse_args()

    # Set logging level based on --debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("geodude").setLevel(logging.DEBUG)

    mode = "Physics" if args.physics else "Kinematic"
    print(f"Recycling Demo (Primitives API) - {mode} Mode", flush=True)
    print("=" * 50, flush=True)

    try:
        # Create robot with objects
        robot = Geodude(objects={"can": 1, "recycle_bin": 2})

        # Place can and bins
        can_pos, can_quat = sample_can_placement(robot)
        print(f"Can at {can_pos.round(3)}", flush=True)
        robot.env.registry.activate("can", pos=can_pos, quat=can_quat)
        robot.env.registry.activate("recycle_bin", pos=RIGHT_BIN_POS)
        robot.env.registry.activate("recycle_bin", pos=LEFT_BIN_POS)

        # Set initial arm positions
        # In kinematic mode: positions are set directly via go_to
        # In physics mode: the controller initializes qpos to ready positions
        # (so this is mainly for kinematic mode, physics mode handles it internally)
        robot.go_to("ready")
        mujoco.mj_forward(robot.model, robot.data)

        # viewer=False means headless mode (no viewer created)
        # viewer=None means auto-create viewer
        viewer = False if args.headless else None
        with robot.sim(physics=args.physics, viewer=viewer) as ctx:
            ctx.sync()

            cycle = 0
            max_cycles = args.cycles
            while ctx.is_running() and cycle < max_cycles:
                cycle += 1
                print(f"\n{'='*40}\nCycle {cycle}\n{'='*40}", flush=True)

                # Check what's pickable
                pickable = robot.get_pickable_objects()
                print(f"Pickable objects: {pickable}", flush=True)

                if not pickable:
                    print("No pickable objects, spawning new can...", flush=True)
                    can_pos, can_quat = sample_can_placement(robot)
                    robot.env.registry.activate("can", pos=can_pos, quat=can_quat)
                    ctx.sync()
                    continue

                # ============================================================
                # THE MAGIC: pickup + place with automatic affordance discovery
                # ============================================================

                # 1. Pick up any pickable object (auto-discovers grasp TSRs)
                target = pickable[0]
                print(f"\n1. Picking up {target}...", flush=True)
                if not robot.pickup(target):
                    print("   Pickup failed!", flush=True)
                    break

                print("   Pickup succeeded!", flush=True)

                # 2. Place in recycle bin (auto-discovers place TSRs)
                # Choose bin based on which arm picked up (same side for easier reach)
                from geodude.primitives import _find_arm_holding_object
                holding_arm = _find_arm_holding_object(robot)
                # recycle_bin_0 is at x=0.85 (right), recycle_bin_1 is at x=-0.85 (left)
                bin_name = "recycle_bin_0" if holding_arm.side == "right" else "recycle_bin_1"

                print(f"\n2. Placing in {bin_name}...", flush=True)
                if not robot.place(bin_name):
                    print("   Place failed!", flush=True)
                    break

                print("   Place succeeded!", flush=True)

                # 3. Hide object and return to ready
                robot.env.registry.hide(target)

                print("\n3. Returning to ready...", flush=True)
                ready_config = np.array(robot.named_poses["ready"][holding_arm.side])
                ready_result = holding_arm.plan_to(ready_config)
                if ready_result is not None:
                    ctx.execute(ready_result)
                ctx.sync()

                print(f"\nCycle {cycle} complete!", flush=True)

                if not ctx.is_running():
                    break

                # Spawn new can
                print("\nSpawning new can...", flush=True)
                can_pos, can_quat = sample_can_placement(robot)
                print(f"Can at {can_pos.round(3)}", flush=True)
                robot.env.registry.activate("can", pos=can_pos, quat=can_quat)
                ctx.sync()

            print(f"\nCompleted {cycle} cycles.", flush=True)

    except Exception as e:
        print(f"\n\nERROR: {e}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
