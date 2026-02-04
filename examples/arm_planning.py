#!/usr/bin/env python3
"""Arm Planning Demo - Unified planning API with Geodude.

Demonstrates:
- Unified plan_to() API with execute parameter
- Planning at different Vention base heights with base_heights parameter
- Heights are tried in order (put most likely first for fastest results)
- PlanResult for compound base+arm trajectories

Usage:
    uv run mjpython examples/arm_planning.py
    uv run mjpython examples/arm_planning.py --base-physics  # Use physics executor for base
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from geodude import Geodude


def generate_random_goal(arm, rng: np.random.Generator, max_attempts: int = 50):
    """Generate a random collision-free joint configuration."""
    checker = arm._get_collision_checker()
    lower, upper = arm.get_joint_limits()

    for _ in range(max_attempts):
        q = rng.uniform(lower, upper)
        if checker.is_valid(q):
            return q
    return None


def main():
    parser = argparse.ArgumentParser(description="Geodude Arm Planning Demo")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random)")
    parser.add_argument("--base-physics", action="store_true",
                        help="Use physics executor for base (default: kinematic)")
    args = parser.parse_args()

    executor_type = "physics" if args.base_physics else "kinematic"

    print(f"Geodude Unified Planning API Demo (executor: {executor_type})", flush=True)
    print("=" * 50, flush=True)
    print("Running until viewer is closed...\n", flush=True)

    robot = Geodude()
    rng = np.random.default_rng(args.seed)

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        # Preferred camera view for paper-quality screenshots
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -26.5
        viewer.cam.distance = 2.96
        viewer.cam.lookat[:] = [0.188, 0.001, 1.141]

        iteration = 0
        while viewer.is_running():
            iteration += 1
            print(f"\nIteration {iteration}", flush=True)

            # ============================================================
            # RIGHT ARM: Simple plan_to() with execute=True (default)
            # ============================================================
            print("  Right arm: plan_to(goal)...", flush=True)
            right_success = False
            for attempt in range(10):
                goal = generate_random_goal(robot.right_arm, rng)
                if goal is None:
                    continue

                t0 = time.perf_counter()
                # NEW API: plan_to() automatically plans and executes
                trajectory = robot.right_arm.plan_to(
                    goal,
                    viewer=viewer,
                    executor_type=executor_type,
                )

                if trajectory is not None:
                    print(f"    Success in {time.perf_counter()-t0:.2f}s, "
                          f"duration={trajectory.duration:.2f}s", flush=True)
                    right_success = True
                    break

            if not right_success:
                print("    Could not find reachable goal", flush=True)

            # ============================================================
            # LEFT ARM: plan_to() with base_heights parameter
            # ============================================================
            if robot.left_base:
                print("  Left arm: plan_to(goal, base_heights=[...])...", flush=True)
                left_success = False
                for attempt in range(10):
                    goal = generate_random_goal(robot.left_arm, rng)
                    if goal is None:
                        continue

                    # Include current height plus other options
                    current_height = robot.left_base.height
                    heights = sorted(set([current_height, 0.0, 0.15, 0.3, 0.45]))

                    t0 = time.perf_counter()
                    # NEW API: base_heights triggers height search (in order)
                    # Returns PlanResult with both base and arm trajectories
                    result = robot.left_arm.plan_to(
                        goal,
                        base_heights=heights,
                        viewer=viewer,
                        executor_type=executor_type,
                    )

                    if result is not None:
                        print(f"    Success at height {result.base_height:.2f}m "
                              f"in {time.perf_counter()-t0:.2f}s", flush=True)
                        print(f"    Trajectories: {[t.entity for t in result.trajectories]}", flush=True)
                        left_success = True
                        break

                if not left_success:
                    print("    Could not find reachable goal", flush=True)

            # Brief pause before next iteration
            time.sleep(0.5)


if __name__ == "__main__":
    main()
