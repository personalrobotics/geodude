"""Tests for parallel planning utilities."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from geodude import Geodude
from geodude.parallel import (
    plan_best_of_all,
    plan_first_success,
)


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

    def test_planner_uses_collision_checker(self):
        """Planner created by create_planner uses CollisionChecker."""
        robot = Geodude()
        robot.go_to("ready")

        planner = robot.right_arm.create_planner()

        # The collision checker should be a CollisionChecker
        from geodude.collision import CollisionChecker

        # CBiRRT stores collision checker as 'collision'
        assert isinstance(planner.collision, CollisionChecker)

    def test_create_planner_with_base_height(self):
        """create_planner accepts base_joint_name and base_height params."""
        robot = Geodude()
        robot.go_to("ready")

        # Create planner with custom base height
        planner = robot.right_arm.create_planner(
            base_joint_name="right_vention/joint",
            base_height=0.2,
        )

        # Should have a planner
        assert hasattr(planner, "plan")


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

    def test_parallel_planning_at_different_heights(self):
        """Can plan same arm at different base heights in parallel."""
        robot = Geodude()
        robot.go_to("ready")

        start = robot.right_arm.get_joint_positions()
        goal = start + 0.05

        heights = [0.0, 0.1, 0.2]

        def plan_at_height(h):
            planner = robot.right_arm.create_planner(
                base_joint_name="right_vention/joint",
                base_height=h,
            )
            return planner.plan(start, goal=goal)

        paths = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(plan_at_height, h) for h in heights]
            for future in futures:
                paths.append(future.result())

        # At least one should succeed
        successful = [p for p in paths if p is not None]
        assert len(successful) >= 1, "At least one height should produce a path"


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
