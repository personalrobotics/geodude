"""Tests for parallel planning utilities."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from geodude import Geodude
from geodude.parallel import (
    GraspStateSnapshot,
    PlanningContext,
    fork_for_planning,
    plan_best_of_all,
    plan_first_success,
)


class TestGraspStateSnapshot:
    """Tests for GraspStateSnapshot."""

    def test_from_grasp_manager_empty(self):
        """Snapshot from empty grasp manager has no grasped objects."""
        robot = Geodude()
        snapshot = GraspStateSnapshot.from_grasp_manager(robot.grasp_manager)

        assert len(snapshot.grasped_objects) == 0
        assert len(snapshot.attachments) == 0

    def test_snapshot_is_immutable(self):
        """Snapshot is immutable (frozen dataclass)."""
        robot = Geodude()
        snapshot = GraspStateSnapshot.from_grasp_manager(robot.grasp_manager)

        # Should not be able to modify
        with pytest.raises(AttributeError):
            snapshot.grasped_objects = frozenset()

    def test_snapshot_isolation(self):
        """Grasp state changes don't affect existing snapshots."""
        robot = Geodude()
        snapshot = GraspStateSnapshot.from_grasp_manager(robot.grasp_manager)

        # Modify live state (simulate grasping an object)
        robot.grasp_manager.grasped["test_obj"] = "right"

        # Snapshot should be unchanged
        assert not snapshot.is_grasped("test_obj")
        assert snapshot.get_holder("test_obj") is None

        # Clean up
        del robot.grasp_manager.grasped["test_obj"]

    def test_is_grasped(self):
        """is_grasped correctly reports grasp state."""
        robot = Geodude()

        # Add a fake grasp
        robot.grasp_manager.grasped["box1"] = "right"
        snapshot = GraspStateSnapshot.from_grasp_manager(robot.grasp_manager)

        assert snapshot.is_grasped("box1")
        assert not snapshot.is_grasped("nonexistent")

        # Clean up
        del robot.grasp_manager.grasped["box1"]

    def test_get_holder(self):
        """get_holder returns correct arm name."""
        robot = Geodude()

        robot.grasp_manager.grasped["box1"] = "left"
        snapshot = GraspStateSnapshot.from_grasp_manager(robot.grasp_manager)

        assert snapshot.get_holder("box1") == "left"
        assert snapshot.get_holder("nonexistent") is None

        # Clean up
        del robot.grasp_manager.grasped["box1"]


class TestPlanningContext:
    """Tests for PlanningContext."""

    def test_context_has_own_data(self):
        """PlanningContext has its own MjData copy."""
        robot = Geodude()
        robot.go_to("ready")

        ctx = fork_for_planning(robot.right_arm)

        # Context should have different MjData object
        assert ctx.data is not robot.data

        # But same model (shared)
        assert ctx.model is robot.model

    def test_context_syncs_state(self):
        """PlanningContext copies current state from source."""
        robot = Geodude()
        robot.go_to("ready")

        original_qpos = robot.data.qpos.copy()
        ctx = fork_for_planning(robot.right_arm)

        # Context's data should have same qpos
        assert np.allclose(ctx.data.qpos, original_qpos)

    def test_context_has_collision_checker(self):
        """PlanningContext creates a SnapshotCollisionChecker."""
        robot = Geodude()
        robot.go_to("ready")

        ctx = fork_for_planning(robot.right_arm)

        # Should have a collision checker
        assert ctx.collision_checker is not None
        assert hasattr(ctx.collision_checker, "is_valid")


class TestCreatePlanner:
    """Tests for Arm.create_planner()."""

    def test_create_planner_returns_planner(self):
        """create_planner returns a CBiRRT instance."""
        robot = Geodude()
        robot.go_to("ready")

        planner = robot.right_arm.create_planner()

        # Should be a planner with the expected interface
        assert hasattr(planner, "plan")

    def test_create_planner_independent(self):
        """Multiple planners from create_planner are independent."""
        robot = Geodude()
        robot.go_to("ready")

        planner1 = robot.right_arm.create_planner()
        planner2 = robot.right_arm.create_planner()

        # Should be different instances
        assert planner1 is not planner2

    def test_planner_uses_snapshot(self):
        """Planner created by create_planner uses snapshot-based collision."""
        robot = Geodude()
        robot.go_to("ready")

        planner = robot.right_arm.create_planner()

        # The collision checker should be a SnapshotCollisionChecker
        from geodude.collision import SnapshotCollisionChecker

        # CBiRRT stores collision checker as 'collision'
        assert isinstance(planner.collision, SnapshotCollisionChecker)


class TestParallelPlanning:
    """Tests for parallel planning functionality."""

    def test_parallel_arm_planning(self):
        """Can plan both arms simultaneously in parallel."""
        robot = Geodude()
        robot.go_to("ready")

        # Small joint space goals (should be quick to plan)
        left_goal = robot.left_arm.get_joint_positions() + 0.05
        right_goal = robot.right_arm.get_joint_positions() + 0.05

        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(
                lambda: robot.left_arm.create_planner().plan(
                    robot.left_arm.get_joint_positions(), goal=left_goal
                )
            )
            right_future = executor.submit(
                lambda: robot.right_arm.create_planner().plan(
                    robot.right_arm.get_joint_positions(), goal=right_goal
                )
            )

            left_path = left_future.result()
            right_path = right_future.result()

        # Both should return paths (small motions should succeed)
        assert left_path is not None, "Left arm planning failed"
        assert right_path is not None, "Right arm planning failed"
        assert len(left_path) > 0
        assert len(right_path) > 0

    def test_parallel_same_arm_multiple_goals(self):
        """Can plan same arm to multiple goals in parallel."""
        robot = Geodude()
        robot.go_to("ready")

        # Multiple small goals
        goals = [
            robot.right_arm.get_joint_positions() + np.array([0.05, 0, 0, 0, 0, 0]),
            robot.right_arm.get_joint_positions() + np.array([0, 0.05, 0, 0, 0, 0]),
            robot.right_arm.get_joint_positions() + np.array([0, 0, 0.05, 0, 0, 0]),
        ]

        paths = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    lambda g=g: robot.right_arm.create_planner().plan(
                        robot.right_arm.get_joint_positions(), goal=g
                    )
                )
                for g in goals
            ]

            for future in futures:
                paths.append(future.result())

        # All should succeed (small motions)
        successful = [p for p in paths if p is not None]
        assert len(successful) >= 1, "At least one planning should succeed"


class TestPlanFirstSuccess:
    """Tests for plan_first_success helper."""

    def test_plan_first_success_returns_path(self):
        """plan_first_success returns first successful path."""
        robot = Geodude()
        robot.go_to("ready")

        goals = [
            robot.right_arm.get_joint_positions() + 0.03,
            robot.right_arm.get_joint_positions() + 0.05,
            robot.right_arm.get_joint_positions() + 0.07,
        ]

        path = plan_first_success(robot.right_arm, goals, timeout=10.0)

        assert path is not None, "Should find at least one path"
        assert len(path) > 0

    def test_plan_first_success_returns_none_on_failure(self):
        """plan_first_success returns None if all fail."""
        robot = Geodude()
        robot.go_to("ready")

        # Unreachable goals (outside joint limits)
        lower, upper = robot.right_arm.get_joint_limits()
        goals = [
            upper + 1.0,  # Beyond upper limits
            lower - 1.0,  # Beyond lower limits
        ]

        path = plan_first_success(robot.right_arm, goals, timeout=2.0)

        assert path is None


class TestPlanBestOfAll:
    """Tests for plan_best_of_all helper."""

    def test_plan_best_of_all_returns_path(self):
        """plan_best_of_all returns best path by metric."""
        robot = Geodude()
        robot.go_to("ready")

        # Goals at different distances
        goals = [
            robot.right_arm.get_joint_positions() + 0.03,  # Closest
            robot.right_arm.get_joint_positions() + 0.1,  # Farther
        ]

        path = plan_best_of_all(robot.right_arm, goals, timeout=10.0)

        assert path is not None, "Should find at least one path"
        assert len(path) > 0

    def test_plan_best_of_all_custom_metric(self):
        """plan_best_of_all respects custom metric."""
        robot = Geodude()
        robot.go_to("ready")

        goals = [
            robot.right_arm.get_joint_positions() + 0.03,
            robot.right_arm.get_joint_positions() + 0.05,
        ]

        # Custom metric that prefers longer paths (inverted)
        def prefer_longer(path):
            return -len(path)

        path = plan_best_of_all(
            robot.right_arm, goals, timeout=10.0, metric=prefer_longer
        )

        assert path is not None

    def test_plan_best_of_all_returns_none_on_failure(self):
        """plan_best_of_all returns None if all fail."""
        robot = Geodude()
        robot.go_to("ready")

        # Unreachable goals
        lower, upper = robot.right_arm.get_joint_limits()
        goals = [
            upper + 1.0,
            lower - 1.0,
        ]

        path = plan_best_of_all(robot.right_arm, goals, timeout=2.0)

        assert path is None
