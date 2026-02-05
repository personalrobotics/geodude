#!/usr/bin/env python3
"""Arm Planning Demo - Unified planning API with Geodude.

Demonstrates:
- Unified plan_to() API returning Trajectory or PlanResult
- Execution via robot.sim() context and ctx.execute()
- Planning at different Vention base heights with base_heights parameter
- Heights are tried in order (put most likely first for fastest results)
- PlanResult for compound base+arm trajectories

Usage:
    uv run mjpython examples/arm_planning.py
"""

import time

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
    print("Geodude Unified Planning API Demo", flush=True)
    print("=" * 50, flush=True)
    print("Running until viewer is closed...\n", flush=True)

    robot = Geodude()
    rng = np.random.default_rng()

    # Use execution context for simulation
    with robot.sim(physics=False) as ctx:
        iteration = 0
        while ctx.is_running():
            iteration += 1
            print(f"\nIteration {iteration}", flush=True)

            # ============================================================
            # RIGHT ARM: Simple plan_to() returning Trajectory
            # ============================================================
            print("  Right arm: plan_to(goal)...", flush=True)
            right_success = False
            for attempt in range(10):
                goal = generate_random_goal(robot.right_arm, rng)
                if goal is None:
                    continue

                t0 = time.perf_counter()
                # plan_to() returns Trajectory (or None)
                trajectory = robot.right_arm.plan_to(goal)

                if trajectory is not None:
                    # Execute via context
                    ctx.execute(trajectory)
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
                    # base_heights returns PlanResult with base and arm trajectories
                    result = robot.left_arm.plan_to(
                        goal,
                        base_heights=heights,
                    )

                    if result is not None:
                        # Execute via context (handles base and arm)
                        ctx.execute(result)
                        print(f"    Success at height {result.base_height:.2f}m "
                              f"in {time.perf_counter()-t0:.2f}s", flush=True)
                        print(f"    Trajectories: {[t.entity for t in result.trajectories]}", flush=True)
                        left_success = True
                        break

                if not left_success:
                    print("    Could not find reachable goal", flush=True)

            # Brief pause before next iteration
            ctx.sync()
            time.sleep(0.5)


if __name__ == "__main__":
    main()
