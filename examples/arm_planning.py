#!/usr/bin/env python3
"""Arm Planning Demo - Simple parallel planning with Geodude.

Demonstrates:
- Thread-safe parallel planning with isolated planners
- Planning at different Vention base heights
- First-success-wins parallel execution

Usage:
    uv run mjpython examples/arm_planning.py
    uv run mjpython examples/arm_planning.py --iterations 10
"""

import argparse
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import mujoco
import mujoco.viewer
import numpy as np

from geodude import Geodude
from geodude.trajectory import Trajectory


def generate_random_goal(arm, rng: np.random.Generator, max_attempts: int = 50):
    """Generate a random collision-free joint configuration."""
    checker = arm._get_collision_checker()
    lower, upper = arm.get_joint_limits()

    for _ in range(max_attempts):
        q = rng.uniform(lower, upper)
        if checker.is_valid(q):
            return q
    return None


def plan_parallel(arm, goal, num_forks: int = 4):
    """Plan to a goal using parallel forks, return first success.

    Each fork gets an isolated planner via arm.create_planner().
    """
    from pycbirrt import CBiRRTConfig

    start = arm.get_joint_positions()
    config = CBiRRTConfig(
        timeout=10.0,
        max_iterations=5000,
        step_size=0.1,
        goal_bias=0.1,
        smoothing_iterations=100,
    )

    def plan_fork(fork_id):
        try:
            planner = arm.create_planner(config)
            path = planner.plan(start, goal=goal)
            return (fork_id, path)
        except Exception:
            return (fork_id, None)

    with ThreadPoolExecutor(max_workers=num_forks) as executor:
        futures = {executor.submit(plan_fork, i): i for i in range(num_forks)}

        while futures:
            done, _ = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
            for future in done:
                fork_id, path = future.result()
                if path is not None:
                    for f in futures:
                        f.cancel()
                    return fork_id, path
                del futures[future]

    return None, None


def plan_at_heights(arm, base, goal, heights):
    """Plan to a goal trying different base heights in parallel.

    Demonstrates the new base_height parameter in create_planner().
    """
    from pycbirrt import CBiRRTConfig

    start = arm.get_joint_positions()
    base_joint = base.config.joint_name
    config = CBiRRTConfig(
        timeout=10.0,
        max_iterations=5000,
        step_size=0.1,
        goal_bias=0.1,
        smoothing_iterations=100,
    )

    def plan_at_height(height):
        try:
            planner = arm.create_planner(
                config,
                base_joint_name=base_joint,
                base_height=height,
            )
            path = planner.plan(start, goal=goal)
            return (height, path)
        except Exception:
            # Start config may be in collision at this base height
            return (height, None)

    with ThreadPoolExecutor(max_workers=len(heights)) as executor:
        futures = {executor.submit(plan_at_height, h): h for h in heights}

        while futures:
            done, _ = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
            for future in done:
                height, path = future.result()
                if path is not None:
                    for f in futures:
                        f.cancel()
                    return height, path
                del futures[future]

    return None, None


def execute_path(arm, path, viewer):
    """Execute a planned path with time-optimal retiming."""
    trajectory = Trajectory.from_path(
        path,
        arm.config.kinematic_limits.velocity,
        arm.config.kinematic_limits.acceleration,
    )

    t = 0.0
    dt = 0.008
    while t <= trajectory.duration:
        pos, _, _ = trajectory.sample(t)
        arm.set_joint_positions(pos)
        viewer.sync()
        time.sleep(dt)
        t += dt


def move_base(base, target, viewer):
    """Move base with hardware-realistic motion profile."""
    base.move_to(target, viewer=viewer)


def main():
    parser = argparse.ArgumentParser(description="Geodude Arm Planning Demo")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Geodude Arm Planning Demo", flush=True)
    print("=" * 40, flush=True)

    robot = Geodude()
    rng = np.random.default_rng(args.seed)

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -25

        for i in range(args.iterations):
            print(f"\nIteration {i+1}/{args.iterations}", flush=True)

            # Plan right arm to random goal
            goal = generate_random_goal(robot.right_arm, rng)
            if goal is None:
                print("  Right: failed to find valid goal", flush=True)
                continue

            print(f"  Right: planning (4 forks)...", flush=True)
            t0 = time.perf_counter()
            fork_id, path = plan_parallel(robot.right_arm, goal)

            if path:
                print(f"  Right: fork {fork_id} won in {time.perf_counter()-t0:.2f}s, {len(path)} waypoints", flush=True)
                execute_path(robot.right_arm, path, viewer)
            else:
                print(f"  Right: all forks failed", flush=True)

            # Plan left arm with different base heights
            if robot.left_base:
                goal = generate_random_goal(robot.left_arm, rng)
                if goal is None:
                    print("  Left: failed to find valid goal", flush=True)
                    continue

                # Include current height (always valid for start config) plus other options
                current_height = robot.left_base.height
                heights = sorted(set([current_height, 0.0, 0.15, 0.3, 0.45]))
                print(f"  Left: planning at heights {heights}...", flush=True)
                t0 = time.perf_counter()
                height, path = plan_at_heights(robot.left_arm, robot.left_base, goal, heights)

                if path:
                    print(f"  Left: height {height:.2f}m won in {time.perf_counter()-t0:.2f}s", flush=True)
                    move_base(robot.left_base, height, viewer)
                    execute_path(robot.left_arm, path, viewer)
                else:
                    print(f"  Left: all heights failed", flush=True)

        print("\nDemo complete. Close viewer to exit.", flush=True)
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
