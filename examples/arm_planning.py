#!/usr/bin/env python3
"""Arm Planning Demo - Reference implementation for Geodude.

This demo showcases the recommended pattern for motion planning with Geodude:
- TRUE parallel planning with thread-safe isolated planners
- Multiple forks racing to find paths faster
- Switching between kinematic and physics-based execution
- Clean timing metrics for profiling

Usage:
    uv run mjpython examples/arm_planning.py
    uv run mjpython examples/arm_planning.py --executor kinematic --forks 8
    uv run mjpython examples/arm_planning.py --iterations 20 --seed 42
"""

import argparse
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from geodude.robot import Geodude
from geodude.trajectory import Trajectory
from geodude.executor import KinematicExecutor, PhysicsExecutor


# ══════════════════════════════════════════════════════════════════════════════
# Pretty Printing
# ══════════════════════════════════════════════════════════════════════════════

def print_header(executor: str, forks: int, seed: int, iterations: int) -> None:
    """Print the demo header."""
    print("╔══════════════════════════════════════════════════════════════╗", flush=True)
    print("║  Geodude Arm Planning Demo                                   ║", flush=True)
    print(f"║  Executor: {executor:<8} │ Forks: {forks:<2} │ Seed: {seed:<6} │ Iter: {iterations:<3} ║", flush=True)
    print("╚══════════════════════════════════════════════════════════════╝", flush=True)
    print(flush=True)


def print_init_info(robot: Geodude) -> None:
    """Print initialization info."""
    print("Initializing...", flush=True)
    print(f"  Model: {robot.config.model_path.name}", flush=True)
    print(f"  Arms: 6-DOF UR5e x 2", flush=True)
    if robot.left_base and robot.right_base:
        print(f"  Bases: Vention linear (0.5m travel)", flush=True)
    print(flush=True)


def print_separator(title: str = "") -> None:
    """Print a section separator."""
    if title:
        print(f"──────────────────────────────────────────────────────────────", flush=True)
        print(f" {title}", flush=True)
        print(f"──────────────────────────────────────────────────────────────", flush=True)
    else:
        print(f"──────────────────────────────────────────────────────────────", flush=True)


def print_planning_start(arm_name: str, num_forks: int) -> None:
    """Print planning start message."""
    print(f"→ {arm_name.capitalize()} arm: planning ({num_forks} forks)", flush=True)


def print_planning_result(fork_id: int, plan_time: float, retime_time: float,
                          num_waypoints: int, success: bool) -> None:
    """Print planning result."""
    if success:
        print(f"  ✓ Fork {fork_id} won │ plan: {plan_time:.2f}s │ retime: {retime_time:.2f}s │ {num_waypoints} waypoints", flush=True)
    else:
        print(f"  ✗ All forks failed", flush=True)


def print_execution_start(arm_name: str) -> None:
    """Print execution start message."""
    print(f"→ {arm_name.capitalize()} arm: executing", flush=True)


def print_execution_result(exec_time: float, max_error_deg: float,
                           traj_duration: float, success: bool,
                           failure_reason: str = "") -> None:
    """Print execution result."""
    if success:
        print(f"  ✓ {exec_time:.2f}s │ max error: {max_error_deg:.1f}° │ traj duration: {traj_duration:.2f}s", flush=True)
    else:
        print(f"  ✗ {exec_time:.2f}s │ {failure_reason}", flush=True)


def print_base_move(side: str, start: float, end: float, duration: float) -> None:
    """Print base movement."""
    print(f"  {side.capitalize()}: {start:.2f}m → {end:.2f}m ({duration:.2f}s) ✓", flush=True)


def print_summary(total_motions: int, successes: int, collisions: int,
                  tracking_failures: int) -> None:
    """Print final summary."""
    print(flush=True)
    print_separator("Summary")
    success_rate = (successes / total_motions * 100) if total_motions > 0 else 0
    print(f"  Arm motions: {successes}/{total_motions} successful ({success_rate:.0f}%)", flush=True)
    if collisions > 0:
        print(f"  Collisions: {collisions}", flush=True)
    if tracking_failures > 0:
        print(f"  Tracking failures: {tracking_failures}", flush=True)
    print(flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Planning
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlanResult:
    """Result from a planning attempt."""
    success: bool
    fork_id: int
    path: list | None
    goal: np.ndarray | None
    plan_time: float
    retime_time: float
    trajectory: Trajectory | None

    @property
    def num_waypoints(self) -> int:
        return self.trajectory.num_waypoints if self.trajectory else 0


def generate_random_collision_free_config(arm, rng: np.random.Generator,
                                          max_attempts: int = 100) -> np.ndarray | None:
    """Generate a random collision-free joint configuration."""
    checker = arm._get_collision_checker()

    for _ in range(max_attempts):
        # Sample random config within joint limits
        q = np.zeros(6)
        for i, name in enumerate(arm.config.joint_names):
            joint_id = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            low, high = arm.model.jnt_range[joint_id]
            q[i] = rng.uniform(low, high)

        if checker.is_valid(q):
            return q

    return None


def plan_single_fork(arm, goal: np.ndarray, _seed: int, fork_id: int) -> PlanResult:
    """Plan to a goal configuration in a single fork (thread-safe).

    Uses arm.create_planner() to get an isolated planner with its own MjData,
    enabling true parallel execution across threads.

    Note: seed parameter is kept for API compatibility but randomness comes from
    the planner's internal sampling which varies naturally across parallel forks.
    """
    from pycbirrt import CBiRRTConfig

    # Create isolated planner for this thread
    config = CBiRRTConfig(
        timeout=10.0,
        max_iterations=5000,
        step_size=0.1,
        goal_bias=0.1,
        ik_num_seeds=1,
        smoothing_iterations=100,
    )
    planner = arm.create_planner(config)
    start = arm.get_joint_positions()

    # Time the planning
    plan_start = time.perf_counter()
    path = planner.plan(start, goal=goal)
    plan_time = time.perf_counter() - plan_start

    if path is None:
        return PlanResult(
            success=False, fork_id=fork_id, path=None, goal=goal,
            plan_time=plan_time, retime_time=0, trajectory=None
        )

    # Time the retiming
    retime_start = time.perf_counter()
    try:
        trajectory = Trajectory.from_path(
            path,
            arm.config.kinematic_limits.velocity,
            arm.config.kinematic_limits.acceleration,
        )
        retime_time = time.perf_counter() - retime_start

        return PlanResult(
            success=True, fork_id=fork_id, path=path, goal=goal,
            plan_time=plan_time, retime_time=retime_time, trajectory=trajectory
        )
    except (ValueError, RuntimeError):
        retime_time = time.perf_counter() - retime_start
        return PlanResult(
            success=False, fork_id=fork_id, path=path, goal=goal,
            plan_time=plan_time, retime_time=retime_time, trajectory=None
        )


def parallel_plan_to_random(robot: Geodude, arm, num_forks: int,
                            rng: np.random.Generator, base_seed: int) -> PlanResult | None:
    """Plan to random configurations in TRUE parallel, return first successful result.

    This demonstrates thread-safe parallel planning with isolated planners:
    1. Generate K different random collision-free goals
    2. Launch K parallel planners using ThreadPoolExecutor
    3. Return the first successful plan, cancel remaining

    Each fork uses arm.create_planner() which creates an isolated CBiRRT planner
    with its own MjData copy, enabling true parallel execution.

    Args:
        robot: The Geodude robot instance
        arm: The arm to plan for (robot.left_arm or robot.right_arm)
        num_forks: Number of parallel planning attempts to different goals
        rng: Random number generator
        base_seed: Base seed for reproducibility

    Returns:
        PlanResult if any attempt succeeded, None if all failed
    """
    # Generate random goals
    goals = []
    for i in range(num_forks):
        fork_rng = np.random.default_rng(base_seed + i * 1000)
        goal = generate_random_collision_free_config(arm, fork_rng)
        if goal is not None:
            goals.append((i, goal))

    if not goals:
        return None

    # Plan to all goals in TRUE parallel - first success wins
    with ThreadPoolExecutor(max_workers=num_forks) as executor:
        # Submit all planning tasks
        futures = {
            executor.submit(plan_single_fork, arm, goal, base_seed + fork_id, fork_id): fork_id
            for fork_id, goal in goals
        }

        # Wait for first successful result
        while futures:
            done, _ = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    result = future.result()
                    if result.success:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        return result
                except Exception:
                    pass
                del futures[future]

    # All failed - return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Execution
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionResult:
    """Result from trajectory execution."""
    success: bool
    exec_time: float
    max_error_deg: float
    traj_duration: float
    failure_reason: str = ""


def execute_trajectory(robot: Geodude, arm, trajectory: Trajectory,
                       executor_type: str, viewer) -> ExecutionResult:
    """Execute a trajectory with the specified executor.

    Args:
        robot: The Geodude robot instance
        arm: The arm to execute on
        trajectory: The trajectory to execute
        executor_type: "kinematic" or "physics"
        viewer: MuJoCo viewer for visualization

    Returns:
        ExecutionResult with timing and error info
    """
    control_dt = 0.008  # 125 Hz

    # Create executor
    if executor_type == "kinematic":
        executor = KinematicExecutor(
            robot.model, robot.data,
            arm.joint_qpos_indices,
            control_dt=control_dt,
            viewer=viewer
        )
    else:
        executor = PhysicsExecutor(
            robot.model, robot.data,
            arm.joint_qpos_indices,
            arm.actuator_ids,
            control_dt=control_dt,
            lookahead_time=0.1,
            viewer=viewer
        )

    # Execute trajectory
    max_error = 0.0
    exec_start = time.perf_counter()

    for i in range(trajectory.num_waypoints):
        q_des = trajectory.positions[i]
        qd_des = trajectory.velocities[i]

        # Apply control - different API for each executor type
        if executor_type == "kinematic":
            # KinematicExecutor: directly set position (perfect tracking)
            executor.set_position(q_des)
            if viewer:
                viewer.sync()
            time.sleep(control_dt)
        else:
            # PhysicsExecutor: set target and step physics
            executor.set_target(q_des, qd_des)
            executor.step()

            # Check for collision (physics only)
            if robot.data.ncon > 0:
                # Check if arm is involved in collision
                for ci in range(robot.data.ncon):
                    contact = robot.data.contact[ci]
                    b1 = robot.model.geom_bodyid[contact.geom1]
                    b2 = robot.model.geom_bodyid[contact.geom2]
                    n1 = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
                    n2 = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""

                    if arm.name in n1 or arm.name in n2:
                        if "base" in n1.lower() or "base" in n2.lower():
                            exec_time = time.perf_counter() - exec_start
                            return ExecutionResult(
                                success=False, exec_time=exec_time,
                                max_error_deg=np.rad2deg(max_error),
                                traj_duration=trajectory.duration,
                                failure_reason=f"Collision: {n1} <-> {n2}"
                            )

            # Real-time pacing for physics
            time.sleep(control_dt)

        # Track error (for physics; kinematic is always 0)
        q_actual = np.array([robot.data.qpos[idx] for idx in arm.joint_qpos_indices])
        error = np.max(np.abs(q_des - q_actual))
        max_error = max(max_error, error)

        # Check tracking threshold (physics only - kinematic has perfect tracking)
        if executor_type == "physics" and error > arm.config.tracking_thresholds.max_error:
            exec_time = time.perf_counter() - exec_start
            return ExecutionResult(
                success=False, exec_time=exec_time,
                max_error_deg=np.rad2deg(max_error),
                traj_duration=trajectory.duration,
                failure_reason=f"Tracking error: {np.rad2deg(error):.1f}° > {np.rad2deg(arm.config.tracking_thresholds.max_error):.0f}°"
            )

    exec_time = time.perf_counter() - exec_start
    return ExecutionResult(
        success=True, exec_time=exec_time,
        max_error_deg=np.rad2deg(max_error),
        traj_duration=trajectory.duration
    )


def move_base(robot: Geodude, base, target_height: float, viewer) -> float:
    """Move a base to target height, return duration."""
    start_height = base.height
    start_time = time.perf_counter()

    # Animate the movement
    steps = 50
    for i in range(steps + 1):
        alpha = i / steps
        height = start_height + alpha * (target_height - start_height)
        base.set_height(height)
        mujoco.mj_forward(robot.model, robot.data)
        if viewer:
            viewer.sync()
        time.sleep(0.01)

    return time.perf_counter() - start_time


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Geodude Arm Planning Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run mjpython examples/arm_planning.py
  uv run mjpython examples/arm_planning.py --executor kinematic
  uv run mjpython examples/arm_planning.py --forks 8 --iterations 20
        """
    )
    parser.add_argument(
        "--executor", choices=["kinematic", "physics"], default="physics",
        help="Executor type (default: physics)"
    )
    parser.add_argument(
        "--forks", type=int, default=4,
        help="Number of parallel planning forks (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of planning iterations (default: 5)"
    )
    args = parser.parse_args()

    # Print header
    print_header(args.executor, args.forks, args.seed, args.iterations)

    # Initialize robot
    robot = Geodude()
    print_init_info(robot)

    # Launch viewer
    print("Launching viewer...", flush=True)
    viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

    # Set default camera view (side view from left)
    viewer.cam.lookat[:] = [0.085, -0.006, 1.398]
    viewer.cam.distance = 2.577
    viewer.cam.azimuth = -90
    viewer.cam.elevation = -26.5

    # Visual settings for cleaner appearance
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    print(flush=True)

    # Statistics
    total_motions = 0
    successes = 0
    collisions = 0
    tracking_failures = 0

    rng = np.random.default_rng(args.seed)

    try:
        for iteration in range(args.iterations):
            print_separator(f"Iteration {iteration + 1}/{args.iterations}")
            print(flush=True)

            # ── Right arm ──
            print_planning_start("right", args.forks)
            result = parallel_plan_to_random(
                robot, robot.right_arm, args.forks, rng,
                base_seed=args.seed + iteration * 10000
            )

            if result and result.success:
                print_planning_result(
                    result.fork_id, result.plan_time, result.retime_time,
                    result.num_waypoints, True
                )

                print_execution_start("right")
                exec_result = execute_trajectory(
                    robot, robot.right_arm, result.trajectory,
                    args.executor, viewer
                )
                print_execution_result(
                    exec_result.exec_time, exec_result.max_error_deg,
                    exec_result.traj_duration, exec_result.success,
                    exec_result.failure_reason
                )

                total_motions += 1
                if exec_result.success:
                    successes += 1
                elif "Collision" in exec_result.failure_reason:
                    collisions += 1
                else:
                    tracking_failures += 1
            else:
                print_planning_result(0, 0, 0, 0, False)

            print(flush=True)

            # ── Left arm ──
            print_planning_start("left", args.forks)
            result = parallel_plan_to_random(
                robot, robot.left_arm, args.forks, rng,
                base_seed=args.seed + iteration * 10000 + 5000
            )

            if result and result.success:
                print_planning_result(
                    result.fork_id, result.plan_time, result.retime_time,
                    result.num_waypoints, True
                )

                print_execution_start("left")
                exec_result = execute_trajectory(
                    robot, robot.left_arm, result.trajectory,
                    args.executor, viewer
                )
                print_execution_result(
                    exec_result.exec_time, exec_result.max_error_deg,
                    exec_result.traj_duration, exec_result.success,
                    exec_result.failure_reason
                )

                total_motions += 1
                if exec_result.success:
                    successes += 1
                elif "Collision" in exec_result.failure_reason:
                    collisions += 1
                else:
                    tracking_failures += 1
            else:
                print_planning_result(0, 0, 0, 0, False)

            print(flush=True)

            # ── Base movement ──
            if robot.right_base and robot.left_base:
                print("→ Bases: moving", flush=True)

                # Random heights
                right_target = rng.uniform(0.0, 0.4)
                left_target = rng.uniform(0.0, 0.4)

                right_start = robot.right_base.height
                left_start = robot.left_base.height

                right_duration = move_base(robot, robot.right_base, right_target, viewer)
                print_base_move("right", right_start, right_target, right_duration)

                left_duration = move_base(robot, robot.left_base, left_target, viewer)
                print_base_move("left", left_start, left_target, left_duration)

            print(flush=True)

        # Print summary
        print_summary(total_motions, successes, collisions, tracking_failures)

        # Keep viewer open
        print("Demo complete. Close viewer to exit.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
