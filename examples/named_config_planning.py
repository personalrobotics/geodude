"""Example: Move between named configurations using CBiRRT planning.

This script demonstrates how to plan collision-free paths between named
robot configurations (like 'home' and 'ready') using the CBiRRT planner,
and execute them using TOPP-RA time-optimal trajectory generation.

The robot will:
1. Start at the 'home' configuration
2. Plan a path to 'ready' using CBiRRT
3. Execute the path with TOPP-RA retiming (respects velocity/acceleration limits)
4. Plan back to 'home'

Usage:
    # Headless mode (default) - just runs planning
    uv run python examples/named_config_planning.py

    # With viewer (macOS requires mjpython)
    uv run mjpython examples/named_config_planning.py --viewer
"""

import argparse
import time

import mujoco
import numpy as np

from geodude import Geodude


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="Launch interactive viewer")
    args = parser.parse_args()

    # Initialize robot
    print("Initializing Geodude robot...")
    robot = Geodude()
    arm = robot.right_arm

    # Get named configurations
    q_home = np.array(robot.named_poses["home"]["right"])
    q_ready = np.array(robot.named_poses["ready"]["right"])

    print(f"Home config:  {q_home}")
    print(f"Ready config: {q_ready}")

    # Start at home position
    print("\nSetting initial position to 'home'...")
    arm.set_joint_positions(q_home)
    mujoco.mj_forward(robot.model, robot.data)

    # Plan from home to ready
    print("\n--- Planning: home -> ready ---")
    start_time = time.time()
    path_to_ready = arm.plan_to_configuration(q_ready, seed=42)
    plan_time = time.time() - start_time

    if path_to_ready is None:
        print("Planning failed!")
        return

    print(f"Planning succeeded in {plan_time:.3f}s")
    print(f"Path has {len(path_to_ready)} waypoints")

    # Plan from ready back to home
    print("\n--- Planning: ready -> home ---")
    arm.set_joint_positions(q_ready)
    mujoco.mj_forward(robot.model, robot.data)

    start_time = time.time()
    path_to_home = arm.plan_to_configuration(q_home, seed=42)
    plan_time = time.time() - start_time

    if path_to_home is None:
        print("Planning failed!")
        return

    print(f"Planning succeeded in {plan_time:.3f}s")
    print(f"Path has {len(path_to_home)} waypoints")

    # If viewer requested, show the motion
    if args.viewer:
        from mujoco import viewer as mj_viewer

        arm.set_joint_positions(q_home)
        mujoco.mj_forward(robot.model, robot.data)

        print("\nLaunching viewer...")
        with mj_viewer.launch_passive(robot.model, robot.data) as viewer:
            viewer.sync()
            time.sleep(1.0)

            # Execute path to ready using TOPP-RA retiming
            print("Executing: home -> ready (with TOPP-RA retiming)")
            arm.execute(path_to_ready)
            viewer.sync()
            time.sleep(0.5)

            # Execute path back home
            print("Executing: ready -> home (with TOPP-RA retiming)")
            arm.execute(path_to_home)
            viewer.sync()

            print("\nDone! Close the viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
    else:
        print("\nPlanning complete. Use --viewer flag to visualize.")


if __name__ == "__main__":
    main()
