"""Thread-safe parallel planning utilities.

Provides isolated planning contexts for running multiple CBiRRT planners
in parallel - e.g., plan left arm to goal A while simultaneously planning
right arm to goal B.

Key classes:
- GraspStateSnapshot: Immutable snapshot of grasp state
- PlanningContext: Isolated context with own MjData for thread-safe planning

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import mujoco
import numpy as np

if TYPE_CHECKING:
    from geodude.arm import Arm
    from geodude.grasp_manager import GraspManager


@dataclass(frozen=True)
class GraspStateSnapshot:
    """Immutable snapshot of grasp state for thread-safe planning.

    Captures the current grasp state so it can be used by parallel planners
    without being affected by changes to the live GraspManager.

    Attributes:
        grasped_objects: Frozenset of (object_name, arm_name) tuples
        attachments: Tuple of (object_name, gripper_body_name, T_gripper_object) tuples
    """

    grasped_objects: frozenset[tuple[str, str]]  # (object_name, arm_name)
    attachments: tuple[tuple[str, str, tuple[tuple[float, ...], ...]], ...]

    @classmethod
    def from_grasp_manager(cls, gm: GraspManager) -> GraspStateSnapshot:
        """Create a snapshot from a live GraspManager.

        Args:
            gm: GraspManager to snapshot

        Returns:
            Immutable snapshot of the grasp state
        """
        # Snapshot grasped objects
        grasped = frozenset(gm.grasped.items())

        # Snapshot attachments (convert numpy arrays to tuples for immutability)
        attachments_list = []
        for obj_name, (gripper_body, transform) in gm._attachments.items():
            # Convert 4x4 numpy array to tuple of tuples
            transform_tuple = tuple(tuple(row) for row in transform)
            attachments_list.append((obj_name, gripper_body, transform_tuple))

        return cls(
            grasped_objects=grasped,
            attachments=tuple(attachments_list),
        )

    def is_grasped(self, object_name: str) -> bool:
        """Check if an object is grasped in this snapshot."""
        return any(obj == object_name for obj, _ in self.grasped_objects)

    def get_holder(self, object_name: str) -> str | None:
        """Get the arm holding an object, or None if not grasped."""
        for obj, arm in self.grasped_objects:
            if obj == object_name:
                return arm
        return None

    def get_attachment_body(self, object_name: str) -> str | None:
        """Get the gripper body that an object is attached to."""
        for obj, gripper_body, _ in self.attachments:
            if obj == object_name:
                return gripper_body
        return None

    def get_attachment_transform(self, object_name: str) -> np.ndarray | None:
        """Get the transform from gripper to attached object."""
        for obj, _, transform_tuple in self.attachments:
            if obj == object_name:
                return np.array(transform_tuple)
        return None


class PlanningContext:
    """Isolated context for thread-safe parallel planning.

    Each PlanningContext has its own MjData copy, allowing multiple planners
    to run in parallel without interfering with each other.

    The MjModel is shared (read-only during planning) - no collision group
    mutations happen during planning with snapshot-based collision checking.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        source_data: mujoco.MjData,
        grasp_snapshot: GraspStateSnapshot,
        joint_names: list[str],
    ):
        """Create an isolated planning context.

        Args:
            model: MuJoCo model (shared, read-only)
            source_data: MjData to copy initial state from
            grasp_snapshot: Immutable snapshot of grasp state
            joint_names: Names of joints to control
        """
        self.model = model
        self.data = mujoco.MjData(model)  # Private copy
        self.grasp_snapshot = grasp_snapshot
        self.joint_names = joint_names

        # Sync state from source
        self._sync_from(source_data)

        # Create collision checker for this context
        from geodude.collision import SnapshotCollisionChecker

        self.collision_checker = SnapshotCollisionChecker(
            model=self.model,
            data=self.data,
            joint_names=joint_names,
            grasp_snapshot=grasp_snapshot,
        )

    def _sync_from(self, source: mujoco.MjData) -> None:
        """Copy state from source MjData."""
        self.data.qpos[:] = source.qpos
        self.data.qvel[:] = source.qvel
        mujoco.mj_forward(self.model, self.data)


def fork_for_planning(arm: Arm) -> PlanningContext:
    """Create an isolated planning context for an arm.

    Safe to call from multiple threads for different arms. Each call creates
    a new PlanningContext with its own MjData copy.

    Args:
        arm: Arm to create planning context for

    Returns:
        Isolated PlanningContext for thread-safe planning
    """
    snapshot = GraspStateSnapshot.from_grasp_manager(arm._grasp_manager)
    return PlanningContext(
        model=arm._model,
        source_data=arm._data,
        grasp_snapshot=snapshot,
        joint_names=arm._config.joint_names,
    )


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
        ik_num_seeds=1,
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
        ik_num_seeds=1,
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
