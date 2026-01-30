"""Parallel planning utilities.

Helper functions for running multiple CBiRRT planners in parallel.

Usage:
    # Plan both arms in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(
            lambda: robot.left_arm.create_planner().plan(start, goal=goal_l)
        )
        right_future = executor.submit(
            lambda: robot.right_arm.create_planner().plan(start, goal=goal_r)
        )
        left_path, right_path = left_future.result(), right_future.result()

    # Try multiple goals, return first success
    path = plan_first_success(robot.right_arm, [tsr1, tsr2, tsr3])

    # Try all goals, return best path
    path = plan_best_of_all(robot.right_arm, [tsr1, tsr2, tsr3])
"""

from __future__ import annotations

from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from geodude.arm import Arm


def plan_first_success(
    arm: Arm,
    goals: list,
    timeout: float = 30.0,
    max_workers: int | None = None,
) -> np.ndarray | None:
    """Plan to multiple goals in parallel, return first successful path.

    Launches parallel planners for each goal. Returns the first path found,
    cancelling remaining planners.

    Args:
        arm: Arm to plan for
        goals: List of goal TSRs or configurations (np.ndarray)
        timeout: Per-planner timeout in seconds
        max_workers: Max parallel planners (default: len(goals))

    Returns:
        First successful path as numpy array, or None if all fail
    """
    from pycbirrt import CBiRRTConfig

    if max_workers is None:
        max_workers = len(goals)

    start = arm.get_joint_positions()

    # Create config with timeout
    config = CBiRRTConfig(
        timeout=timeout,
        max_iterations=5000,
        step_size=0.1,
        goal_bias=0.1,
        smoothing_iterations=100,
    )

    def plan_goal(goal):
        planner = arm.create_planner(config)
        if isinstance(goal, np.ndarray):
            return planner.plan(start, goal=goal)
        else:
            # Assume it's a TSR
            return planner.plan(start, goal_tsrs=[goal])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(plan_goal, g): g for g in goals}

        while futures:
            done, _ = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    path = future.result()
                    if path is not None:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        return path
                except Exception:
                    pass
                del futures[future]

    return None


def plan_best_of_all(
    arm: Arm,
    goals: list,
    timeout: float = 30.0,
    max_workers: int | None = None,
    metric: Callable[[np.ndarray], float] | None = None,
) -> np.ndarray | None:
    """Plan to multiple goals in parallel, return best path.

    Launches parallel planners for all goals, waits for all to complete,
    then returns the best path according to the metric.

    Args:
        arm: Arm to plan for
        goals: List of goal TSRs or configurations (np.ndarray)
        timeout: Per-planner timeout in seconds
        max_workers: Max parallel planners (default: len(goals))
        metric: Path quality metric (lower is better). Default: path length

    Returns:
        Best path by metric, or None if all fail
    """
    from pycbirrt import CBiRRTConfig

    if metric is None:

        def default_metric(path: np.ndarray) -> float:
            return float(len(path))

        metric = default_metric

    if max_workers is None:
        max_workers = len(goals)

    start = arm.get_joint_positions()

    # Create config with timeout
    config = CBiRRTConfig(
        timeout=timeout,
        max_iterations=5000,
        step_size=0.1,
        goal_bias=0.1,
        smoothing_iterations=100,
    )

    def plan_goal(goal):
        planner = arm.create_planner(config)
        if isinstance(goal, np.ndarray):
            return planner.plan(start, goal=goal)
        else:
            # Assume it's a TSR
            return planner.plan(start, goal_tsrs=[goal])

    paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(plan_goal, g) for g in goals]
        wait(futures, return_when=ALL_COMPLETED)

        for future in futures:
            try:
                path = future.result()
                if path is not None:
                    paths.append(path)
            except Exception:
                pass

    if not paths:
        return None

    return min(paths, key=metric)
