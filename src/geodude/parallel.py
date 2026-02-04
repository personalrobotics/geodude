"""Planning utilities for multiple goals and base heights.

Helper functions for running CBiRRT planners with multiple goals.

Usage:
    # Try multiple goals, return first success
    path = plan_first_success(robot.right_arm, [tsr1, tsr2, tsr3])

    # Try all goals, return best path
    path = plan_best_of_all(robot.right_arm, [tsr1, tsr2, tsr3])

    # Plan with base height search
    result = plan_with_base_heights(robot.right_arm, [grasp_tsr], [0.0, 0.2, 0.4])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from geodude.arm import Arm
    from geodude.planning import PlanResult


def plan_first_success(
    arm: Arm,
    goals: list,
    timeout: float = 30.0,
) -> np.ndarray | None:
    """Plan to multiple goals sequentially, return first successful path.

    Tries each goal in order. Returns the first path found.

    Args:
        arm: Arm to plan for
        goals: List of goal TSRs or configurations (np.ndarray), tried in order
        timeout: Per-planner timeout in seconds

    Returns:
        First successful path as numpy array, or None if all fail
    """
    from pycbirrt import CBiRRTConfig

    start = arm.get_joint_positions()

    # Create config with timeout, using arm's planning defaults
    defaults = arm.config.planning_defaults
    config = CBiRRTConfig(
        timeout=timeout,
        max_iterations=defaults.max_iterations,
        step_size=defaults.step_size,
        goal_bias=defaults.goal_bias,
        smoothing_iterations=defaults.smoothing_iterations,
    )

    # Try each goal sequentially
    for goal in goals:
        try:
            planner = arm.create_planner(config)
            if isinstance(goal, np.ndarray):
                path = planner.plan(start, goal=goal)
            else:
                # Assume it's a TSR
                path = planner.plan(start, goal_tsrs=[goal])

            if path is not None:
                return path
        except Exception:
            continue

    return None


def plan_best_of_all(
    arm: Arm,
    goals: list,
    timeout: float = 30.0,
    metric: Callable[[np.ndarray], float] | None = None,
) -> np.ndarray | None:
    """Plan to multiple goals sequentially, return best path.

    Tries all goals, collects successful paths, then returns the best
    according to the metric.

    Args:
        arm: Arm to plan for
        goals: List of goal TSRs or configurations (np.ndarray)
        timeout: Per-planner timeout in seconds
        metric: Path quality metric (lower is better). Default: path length

    Returns:
        Best path by metric, or None if all fail
    """
    from pycbirrt import CBiRRTConfig

    if metric is None:

        def default_metric(path: np.ndarray) -> float:
            return float(len(path))

        metric = default_metric

    start = arm.get_joint_positions()

    # Create config with timeout, using arm's planning defaults
    defaults = arm.config.planning_defaults
    config = CBiRRTConfig(
        timeout=timeout,
        max_iterations=defaults.max_iterations,
        step_size=defaults.step_size,
        goal_bias=defaults.goal_bias,
        smoothing_iterations=defaults.smoothing_iterations,
    )

    # Try all goals sequentially
    paths = []
    for goal in goals:
        try:
            planner = arm.create_planner(config)
            if isinstance(goal, np.ndarray):
                path = planner.plan(start, goal=goal)
            else:
                # Assume it's a TSR
                path = planner.plan(start, goal_tsrs=[goal])

            if path is not None:
                paths.append(path)
        except Exception:
            continue

    if not paths:
        return None

    return min(paths, key=metric)


def plan_with_base_heights(
    arm: "Arm",
    goal_tsrs: list,
    base_heights: list[float],
    *,
    execute: bool = True,
    timeout: float = 30.0,
    seed: int | None = None,
    viewer=None,
    executor_type: str = "physics",
) -> "PlanResult | None":
    """Plan arm motion at different base heights.

    Searches through multiple base heights to find one where the arm can
    reach the goal TSRs. Heights are pre-filtered by collision checking
    before attempting arm planning.

    This is a convenience wrapper around arm.plan_to_tsr() with base_heights.

    Args:
        arm: Arm to plan for
        goal_tsrs: List of goal TSRs (union - any one)
        base_heights: List of base heights to search (tried in order)
        execute: If True (default), execute trajectories after planning
        timeout: Per-planner timeout in seconds
        seed: Random seed for reproducibility
        viewer: Optional MuJoCo viewer for execution
        executor_type: "physics" or "kinematic" for execution

    Returns:
        PlanResult with arm and base trajectories, or None if all fail

    Example:
        from geodude.parallel import plan_with_base_heights

        result = plan_with_base_heights(
            robot.right_arm,
            [grasp_tsr],
            base_heights=[0.0, 0.1, 0.2, 0.3, 0.4],
            execute=False,
        )
        if result:
            print(f"Found solution at height {result.base_height}")
            # Execute manually
            for traj in result.trajectories:
                print(f"Execute {traj.entity}: {traj.duration:.2f}s")
    """
    return arm.plan_to_tsr(
        goal_tsrs,
        execute=execute,
        base_heights=base_heights,
        timeout=timeout,
        seed=seed,
        viewer=viewer,
        executor_type=executor_type,
    )
