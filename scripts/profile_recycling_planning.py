#!/usr/bin/env python3
"""Profile planning performance for the recycling demo.

Identifies where time is spent to find 10x improvement opportunities.

Usage:
    uv run mjpython scripts/profile_recycling_planning.py
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np

from geodude import Geodude
from geodude.tsr_utils import compensate_tsr_for_gripper
from pycbirrt import CBiRRTConfig
from tsr import TSR
from tsr.core.tsr_primitive import load_template_file

TSR_DIR = Path(__file__).parent.parent / "tsr_templates"
BASE_HEIGHTS = [0.0, 0.2, 0.4]

# Global timing accumulator
TIMINGS = defaultdict(list)


@contextmanager
def timed(name: str):
    """Context manager to time a code block."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    TIMINGS[name].append(elapsed)


def print_timings():
    """Print timing summary."""
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)

    # Sort by total time
    totals = [(name, sum(times), len(times), times) for name, times in TIMINGS.items()]
    totals.sort(key=lambda x: -x[1])

    for name, total, count, times in totals:
        avg = total / count
        min_t = min(times)
        max_t = max(times)
        print(f"{name:40s}  total={total:6.3f}s  count={count:2d}  avg={avg:.3f}s  min={min_t:.3f}s  max={max_t:.3f}s")


def load_grasp_tsr(object_pos):
    """Load side grasp TSR for can at given position."""
    template = load_template_file(str(TSR_DIR / "grasps" / "can_side_grasp.yaml"))
    T0_w = np.eye(4)
    T0_w[:3, 3] = object_pos
    tsr = TSR(T0_w=T0_w, Tw_e=template.Tw_e, Bw=template.Bw)
    return compensate_tsr_for_gripper(tsr, template.subject)


def profile_single_arm_planning(robot, arm, grasp_tsr, height, timeout=15.0):
    """Profile planning for a single arm at a single height."""
    base = robot.left_base if "left" in arm.config.name else robot.right_base
    arm_name = "left" if "left" in arm.config.name else "right"

    with timed(f"get_joint_positions ({arm_name})"):
        q_start = arm.get_joint_positions().copy()

    with timed(f"get_base_height ({arm_name})"):
        current_height = base.height

    # Profile planner creation
    defaults = arm.config.planning_defaults
    config = CBiRRTConfig(
        timeout=timeout,
        max_iterations=defaults.max_iterations,
        step_size=defaults.step_size,
        goal_bias=defaults.goal_bias,
        smoothing_iterations=defaults.smoothing_iterations,
    )

    with timed(f"create_planner ({arm_name} @ {height}m)"):
        planner = arm.create_planner(
            config=config,
            base_joint_name=base.config.joint_name,
            base_height=height,
        )

    # Profile actual planning
    with timed(f"planner.plan ({arm_name} @ {height}m)"):
        path = planner.plan(q_start, goal_tsrs=[grasp_tsr])

    return path


def profile_height_filtering(robot, arm, heights):
    """Profile base height filtering."""
    base = robot.left_base if "left" in arm.config.name else robot.right_base
    arm_name = "left" if "left" in arm.config.name else "right"

    q_start = arm.get_joint_positions().copy()

    with timed(f"filter_reachable_heights ({arm_name})"):
        current_height, reachable = base.filter_reachable_heights(heights, q_start)

    return reachable


def profile_parallel_planning(robot, grasp_tsr, timeout=15.0):
    """Profile the full parallel planning as done in recycle_objects.py."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from geodude.planning import PlanResult
    from geodude.trajectory import Trajectory, create_linear_trajectory

    arms = [robot.left_arm, robot.right_arm]
    tasks = [(arm, h) for arm in arms for h in BASE_HEIGHTS]

    def plan_at(arm, height):
        """Plan for one arm at one height (with internal timing)."""
        arm_name = "left" if "left" in arm.config.name else "right"
        base = robot.left_base if arm_name == "left" else robot.right_base

        t_start = time.perf_counter()
        q_start = arm.get_joint_positions().copy()
        t_get_q = time.perf_counter()

        current_height = base.height

        defaults = arm.config.planning_defaults
        config = CBiRRTConfig(
            timeout=timeout,
            max_iterations=defaults.max_iterations,
            step_size=defaults.step_size,
            goal_bias=defaults.goal_bias,
            smoothing_iterations=defaults.smoothing_iterations,
        )

        t_config = time.perf_counter()
        planner = arm.create_planner(
            config=config,
            base_joint_name=base.config.joint_name,
            base_height=height,
        )
        t_planner = time.perf_counter()

        path = planner.plan(q_start, goal_tsrs=[grasp_tsr])
        t_plan = time.perf_counter()

        # Return timing info along with result
        timings = {
            "get_q": t_get_q - t_start,
            "config": t_config - t_get_q,
            "create_planner": t_planner - t_config,
            "plan": t_plan - t_planner,
            "total": t_plan - t_start,
        }

        if path is None:
            return None, timings

        # Build result
        arm_trajectory = Trajectory.from_path(
            path,
            vel_limits=arm.config.kinematic_limits.velocity,
            acc_limits=arm.config.kinematic_limits.acceleration,
            entity=arm.config.name,
            joint_names=arm.config.joint_names,
        )
        base_trajectory = create_linear_trajectory(
            start=current_height,
            end=height,
            vel_limit=base.config.kinematic_limits.velocity,
            acc_limit=base.config.kinematic_limits.acceleration,
            entity=base.config.name,
            joint_names=base.config.joint_names,
        )
        result = PlanResult(
            arm=arm,
            arm_trajectory=arm_trajectory,
            base_trajectory=base_trajectory,
            base_height=height,
        )
        return result, timings

    print(f"\nRacing {len(tasks)} planners: {[(('L' if 'left' in a.config.name else 'R'), h) for a, h in tasks]}")

    all_timings = []
    winner = None
    winner_task = None

    with timed("parallel_planning_total"):
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(plan_at, arm, h): (arm, h) for arm, h in tasks}

            while futures:
                done, _ = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    arm, h = futures[future]
                    result, timings = future.result()
                    arm_name = "left" if "left" in arm.config.name else "right"
                    timings["arm"] = arm_name
                    timings["height"] = h
                    timings["success"] = result is not None
                    all_timings.append(timings)

                    if result is not None and winner is None:
                        winner = result
                        winner_task = (arm_name, h)
                        # Cancel remaining
                        for f in futures:
                            f.cancel()
                        futures.clear()
                        break
                    del futures[future]

    # Print detailed timings
    print("\nPer-planner breakdown:")
    print(f"{'Arm':<6} {'Height':<8} {'Success':<8} {'get_q':<8} {'config':<8} {'create':<8} {'plan':<8} {'total':<8}")
    print("-" * 70)
    for t in sorted(all_timings, key=lambda x: x["total"]):
        print(f"{t['arm']:<6} {t['height']:<8.1f} {str(t['success']):<8} "
              f"{t['get_q']*1000:>6.1f}ms {t['config']*1000:>6.1f}ms "
              f"{t['create_planner']*1000:>6.1f}ms {t['plan']*1000:>6.1f}ms {t['total']*1000:>6.1f}ms")

    if winner:
        print(f"\nWinner: {winner_task[0].upper()} @ {winner_task[1]}m")

    return winner


def profile_create_planner_breakdown(robot, arm, height):
    """Deep dive into create_planner to see what's slow."""
    base = robot.left_base if "left" in arm.config.name else robot.right_base
    arm_name = "left" if "left" in arm.config.name else "right"

    print(f"\n--- create_planner breakdown ({arm_name} @ {height}m) ---")

    defaults = arm.config.planning_defaults
    config = CBiRRTConfig(
        timeout=15.0,
        max_iterations=defaults.max_iterations,
        step_size=defaults.step_size,
        goal_bias=defaults.goal_bias,
        smoothing_iterations=defaults.smoothing_iterations,
    )

    # Time MjData copy
    t0 = time.perf_counter()
    data_copy = mujoco.MjData(arm.model)
    t_data_copy = time.perf_counter() - t0
    print(f"  MjData copy: {t_data_copy*1000:.2f}ms")

    # Time data copy (state)
    t0 = time.perf_counter()
    data_copy.qpos[:] = arm.data.qpos
    data_copy.qvel[:] = arm.data.qvel
    t_copy_data = time.perf_counter() - t0
    print(f"  Copy qpos/qvel: {t_copy_data*1000:.2f}ms")

    # Time setting base height
    t0 = time.perf_counter()
    base_joint_id = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_JOINT, base.config.joint_name)
    base_qpos_idx = arm.model.jnt_qposadr[base_joint_id]
    data_copy.qpos[base_qpos_idx] = height
    mujoco.mj_forward(arm.model, data_copy)
    t_set_height = time.perf_counter() - t0
    print(f"  Set base height + mj_forward: {t_set_height*1000:.2f}ms")

    # Now time the actual create_planner
    t0 = time.perf_counter()
    planner = arm.create_planner(
        config=config,
        base_joint_name=base.config.joint_name,
        base_height=height,
    )
    t_create = time.perf_counter() - t0
    print(f"  Full create_planner: {t_create*1000:.2f}ms")


def main():
    print("=" * 60)
    print("RECYCLING DEMO PLANNING PROFILER")
    print("=" * 60)

    # Create robot
    with timed("create_robot"):
        robot = Geodude(objects={"can": 1, "recycle_bin": 2})

    model, data = robot.model, robot.data

    # Test with multiple can positions - easy vs hard
    CAN_POSITIONS = [
        ("easy_center", np.array([0.0, -0.3, 0.816])),
        ("right_deep", np.array([-0.099, -0.548, 0.816])),  # From demo
        ("left_far", np.array([0.3, -0.4, 0.816])),
        ("edge_case", np.array([-0.4, -0.25, 0.816])),
    ]

    # Use the harder position from the actual demo
    can_pos = CAN_POSITIONS[1][1]  # right_deep
    print(f"\nCan position: {can_pos} (harder position from demo)")

    # Go to ready pose
    with timed("go_to_ready"):
        robot.go_to("ready")
    mujoco.mj_forward(model, data)

    # Load grasp TSR
    with timed("load_grasp_tsr"):
        grasp_tsr = load_grasp_tsr(can_pos)

    print("\n" + "=" * 60)
    print("TEST 1: Height filtering")
    print("=" * 60)

    for arm in [robot.left_arm, robot.right_arm]:
        reachable = profile_height_filtering(robot, arm, BASE_HEIGHTS)
        arm_name = "left" if "left" in arm.config.name else "right"
        print(f"  {arm_name}: reachable heights = {reachable}")

    print("\n" + "=" * 60)
    print("TEST 2: create_planner breakdown")
    print("=" * 60)

    profile_create_planner_breakdown(robot, robot.right_arm, 0.0)
    profile_create_planner_breakdown(robot, robot.right_arm, 0.2)

    print("\n" + "=" * 60)
    print("TEST 3: Sequential single-arm planning (for comparison)")
    print("=" * 60)

    for arm in [robot.right_arm]:
        arm_name = "left" if "left" in arm.config.name else "right"
        for height in BASE_HEIGHTS:
            print(f"\n{arm_name} @ {height}m:", flush=True)
            path = profile_single_arm_planning(robot, arm, grasp_tsr, height)
            if path:
                print(f"  Found path with {len(path)} waypoints")
            else:
                print(f"  No path found")

    print("\n" + "=" * 60)
    print("TEST 4: Parallel bimanual planning (as in demo)")
    print("=" * 60)

    # Reset to ready
    robot.go_to("ready")
    mujoco.mj_forward(model, data)

    # Run parallel planning multiple times to get averages
    for i in range(3):
        print(f"\n--- Run {i+1} ---")
        TIMINGS.clear()  # Clear per-run
        result = profile_parallel_planning(robot, grasp_tsr)

    print("\n" + "=" * 60)
    print("TEST 4b: Test with ALL can positions")
    print("=" * 60)

    for name, pos in CAN_POSITIONS:
        robot.go_to("ready")
        mujoco.mj_forward(model, data)

        test_tsr = load_grasp_tsr(pos)
        print(f"\n--- {name}: {pos} ---")
        t0 = time.perf_counter()
        result = profile_parallel_planning(robot, test_tsr)
        total = time.perf_counter() - t0
        print(f"TOTAL wall time: {total:.2f}s")

    print("\n" + "=" * 60)
    print("TEST 5: Planner reuse test")
    print("=" * 60)

    # Test if reusing planner is faster
    arm = robot.right_arm
    base = robot.right_base

    defaults = arm.config.planning_defaults
    config = CBiRRTConfig(
        timeout=15.0,
        max_iterations=defaults.max_iterations,
        step_size=defaults.step_size,
        goal_bias=defaults.goal_bias,
        smoothing_iterations=defaults.smoothing_iterations,
    )

    # Create planner once
    t0 = time.perf_counter()
    planner = arm.create_planner(
        config=config,
        base_joint_name=base.config.joint_name,
        base_height=0.0,
    )
    print(f"First create_planner: {(time.perf_counter()-t0)*1000:.2f}ms")

    # Plan multiple times with same planner
    q_start = arm.get_joint_positions().copy()
    for i in range(3):
        t0 = time.perf_counter()
        path = planner.plan(q_start, goal_tsrs=[grasp_tsr])
        print(f"Plan {i+1}: {(time.perf_counter()-t0)*1000:.2f}ms, path={len(path) if path else None} waypoints")

    # Summary
    print_timings()

    print("\n" + "=" * 60)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 60)
    print("""
Based on profiling, potential 10x improvements:

1. PLANNER REUSE: If create_planner is slow, cache planners per (arm, height).
   - MjData copy is ~Xms per call
   - With 6 parallel planners, that's 6x overhead

2. EARLY TERMINATION: First-success strategy already helps, but:
   - Could prioritize heights more likely to succeed
   - Could skip heights that historically fail

3. REDUCE HEIGHTS: If 3 heights × 2 arms = 6 planners is overkill:
   - Maybe 2 heights is enough
   - Or use adaptive height selection

4. IK CACHING: If same TSR is planned to multiple times:
   - Cache IK solutions
   - Reuse collision-free configurations

5. PARALLEL IK: Currently sequential in some paths
   - Could parallelize IK computation

6. SMARTER TIMEOUT: 15s timeout might be too long for "easy" cases
   - Adaptive timeout based on distance to goal
""")


if __name__ == "__main__":
    main()
