#!/usr/bin/env python3
"""Recycling Demo - Manual approach with explicit TSR loading.

This is the low-level version showing what happens under the hood.
For the simpler high-level API, see recycle.py.

Demonstrates:
- Manual TSR template loading and instantiation
- robot.plan_to_tsr() with base_heights for bimanual planning
- ctx.execute() for trajectory execution
- ctx.arm().grasp() / ctx.arm().release() for manipulation

Usage:
    uv run mjpython examples/recycle_manual.py
"""

from pathlib import Path

import numpy as np

from geodude import Geodude
from geodude.tsr_utils import compensate_tsr_for_gripper
from tsr import TSR
from tsr.core.tsr_primitive import load_template_file

TSR_DIR = Path(__file__).parent.parent / "tsr_templates"

# Bin positions
RIGHT_BIN_POS = [0.75, -0.35, 0.50]
LEFT_BIN_POS = [-0.75, -0.35, 0.50]

# Base heights to try
BASE_HEIGHTS = [0.2, 0.0, 0.4]


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


def load_grasp_tsr(object_pos):
    """Load side grasp TSR for can."""
    template = load_template_file(str(TSR_DIR / "grasps" / "can_side_grasp.yaml"))

    T0_w = np.eye(4)
    T0_w[:3, 3] = object_pos

    tsr = TSR(T0_w=T0_w, Tw_e=template.Tw_e, Bw=template.Bw)
    return compensate_tsr_for_gripper(tsr, template.subject)


def load_place_tsr(robot, bin_name):
    """Load drop TSR for recycle bin."""
    import mujoco

    template = load_template_file(str(TSR_DIR / "places" / "recycle_bin_drop.yaml"))

    bin_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, bin_name)
    T0_w = np.eye(4)
    T0_w[:3, 3] = robot.data.xpos[bin_id].copy()

    tsr = TSR(T0_w=T0_w, Tw_e=template.Tw_e, Bw=template.Bw)
    return compensate_tsr_for_gripper(tsr, template.subject)


def main():
    import mujoco

    print("Simplified Recycling Demo (Execution Context API)", flush=True)
    print("=" * 50, flush=True)

    # Create robot with objects
    robot = Geodude(objects={"can": 1, "recycle_bin": 2})

    # Place can and bins
    can_pos, can_quat = sample_can_placement(robot)
    print(f"Can at {can_pos.round(3)}", flush=True)
    robot.env.registry.activate("can", pos=can_pos, quat=can_quat)
    robot.env.registry.activate("recycle_bin", pos=RIGHT_BIN_POS)
    robot.env.registry.activate("recycle_bin", pos=LEFT_BIN_POS)
    mujoco.mj_forward(robot.model, robot.data)

    # Use the new execution context API (kinematic mode for reliable testing)
    with robot.sim(physics=False) as ctx:
        robot.go_to("ready")
        ctx.sync()

        cycle = 0
        while ctx.is_running():
            cycle += 1
            print(f"\n{'='*40}\nCycle {cycle}\n{'='*40}", flush=True)

            # Get can position
            can_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "can_0")
            can_pos = robot.data.xpos[can_id].copy()
            print(f"Can at {can_pos.round(3)}", flush=True)

            # 1. Plan grasp (both arms at multiple heights)
            print("\n1. Planning grasp...", flush=True)
            grasp_tsr = load_grasp_tsr(can_pos)

            result = robot.plan_to_tsr(
                grasp_tsr,
                base_heights=BASE_HEIGHTS,
            )

            if result is None:
                print("   Failed to plan grasp!", flush=True)
                break

            arm_name = result.arm.side
            print(f"   {arm_name.upper()} arm @ base height {result.base_height:.1f}m", flush=True)

            # 2. Execute grasp motion
            print("\n2. Executing grasp motion...", flush=True)
            ctx.execute(result)
            ctx.sync()

            # 3. Grasp the can (new simplified API)
            print("\n3. Grasping...", flush=True)
            ctx.arm(arm_name).grasp("can_0")

            # 4. Lift
            print("\n4. Lifting...", flush=True)
            lift_pose = result.arm.get_ee_pose().copy()
            lift_pose[2, 3] += 0.1  # Lift 10cm
            lift_result = result.arm.plan_to_pose(lift_pose, timeout=5.0)
            if lift_result is not None:
                ctx.execute(lift_result)
                robot.grasp_manager.update_attached_poses()
            ctx.sync()

            # 5. Plan to bin
            bin_name = "recycle_bin_1" if arm_name == "left" else "recycle_bin_0"
            print(f"\n5. Planning to {bin_name}...", flush=True)
            place_tsr = load_place_tsr(robot, bin_name)
            place_result = result.arm.plan_to_tsr(place_tsr, timeout=30.0)

            if place_result is None:
                print("   Failed to plan place! Trying with longer timeout...", flush=True)
                place_result = result.arm.plan_to_tsr(place_tsr, timeout=60.0)

            if place_result is None:
                print("   Still failed to plan place!", flush=True)
                break

            ctx.execute(place_result)

            # 6. Release (new simplified API)
            print("\n6. Releasing...", flush=True)
            ctx.arm(arm_name).release("can_0")

            # Hide can
            robot.env.registry.hide("can_0")
            ctx.sync()

            # 7. Return to ready
            print("\n7. Returning to ready...", flush=True)
            ready_config = np.array(robot.named_poses["ready"][arm_name])
            ready_result = result.arm.plan_to(ready_config)
            if ready_result is not None:
                ctx.execute(ready_result)

            print(f"\nDone! {arm_name.upper()} arm recycled can.", flush=True)

            if not ctx.is_running():
                break

            # Spawn new can
            print("\nSpawning new can...", flush=True)
            can_pos, can_quat = sample_can_placement(robot)
            print(f"Can at {can_pos.round(3)}", flush=True)
            robot.env.registry.activate("can", pos=can_pos, quat=can_quat)
            ctx.sync()

        print(f"\nCompleted {cycle} cycles.", flush=True)


if __name__ == "__main__":
    main()
