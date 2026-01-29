#!/usr/bin/env python3
"""Record gripper joint trajectory from physics simulation.

Runs the gripper through its full range using physics, recording all joint
positions at each step. Outputs a numpy array that can be used for kinematic
interpolation.
"""

import mujoco
import numpy as np

from geodude import Geodude


def get_gripper_joint_positions(model, data, prefix: str) -> dict[str, float]:
    """Get all gripper joint positions."""
    joint_types = ["driver", "coupler", "follower", "spring_link"]
    positions = {}
    for side in ["right", "left"]:
        for jtype in joint_types:
            joint_name = f"{prefix}/{side}_{jtype}_joint"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                qpos_adr = model.jnt_qposadr[joint_id]
                positions[f"{side}_{jtype}"] = data.qpos[qpos_adr]
    return positions


def main():
    # Create robot
    robot = Geodude()
    model = robot.model
    data = robot.data

    # Gripper prefix
    prefix = "right_ur5e/gripper"
    actuator_name = "right_ur5e/gripper/fingers_actuator"
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

    # Control range
    ctrl_open = 0.0
    ctrl_closed = 255.0

    # Record trajectory: open -> closed
    print("Recording gripper closing trajectory...")

    # Start fully open
    data.ctrl[actuator_id] = ctrl_open
    for _ in range(200):
        mujoco.mj_step(model, data)

    trajectory = []
    joint_names = None

    # Gradually close
    n_steps = 100
    for i in range(n_steps + 1):
        # Set control
        t = i / n_steps
        ctrl = ctrl_open + t * (ctrl_closed - ctrl_open)
        data.ctrl[actuator_id] = ctrl

        # Step physics to let gripper move
        for _ in range(20):
            mujoco.mj_step(model, data)

        # Record positions
        positions = get_gripper_joint_positions(model, data, prefix)
        if joint_names is None:
            joint_names = sorted(positions.keys())

        trajectory.append([positions[name] for name in joint_names])

        if i % 20 == 0:
            print(f"  {i}/{n_steps}: ctrl={ctrl:.1f}")

    trajectory = np.array(trajectory)

    print(f"\nRecorded {len(trajectory)} waypoints")
    print(f"Joint names: {joint_names}")
    print(f"\nTrajectory shape: {trajectory.shape}")
    print(f"\nOpen position (t=0):")
    for j, name in enumerate(joint_names):
        print(f"  {name}: {trajectory[0, j]:.6f}")
    print(f"\nClosed position (t=1):")
    for j, name in enumerate(joint_names):
        print(f"  {name}: {trajectory[-1, j]:.6f}")

    # Save as Python code for easy copy-paste
    print("\n" + "=" * 60)
    print("Copy this into gripper.py:")
    print("=" * 60)
    print(f"\n# Gripper joint trajectory (recorded from physics)")
    print(f"# Shape: ({len(trajectory)}, {len(joint_names)}) - {n_steps+1} waypoints, {len(joint_names)} joints")
    print(f"_GRIPPER_JOINT_NAMES = {joint_names}")
    print(f"_GRIPPER_TRAJECTORY = np.array([")
    for row in trajectory:
        print(f"    [{', '.join(f'{v:.6f}' for v in row)}],")
    print("])")


if __name__ == "__main__":
    main()
