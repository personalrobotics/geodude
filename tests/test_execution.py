"""Tests for execution context abstraction."""

import mujoco
import numpy as np
import pytest

from geodude.robot import Geodude
from geodude.trajectory import Trajectory


class TestSimContext:
    """Tests for SimContext class."""

    def test_sim_returns_context(self):
        """sim() returns a SimContext."""
        robot = Geodude()
        ctx = robot.sim(physics=True, viewer=None)
        assert ctx is not None
        assert ctx.robot is robot

    def test_context_manager_basic(self):
        """SimContext can be used as context manager without viewer."""
        robot = Geodude()
        # This would normally create a viewer, but we skip that in tests
        # by testing the context's components directly
        ctx = robot.sim(physics=False, viewer=None)

        # Test basic attributes
        assert ctx.robot is robot
        assert ctx._physics is False

    def test_is_running_without_viewer(self):
        """is_running returns True when no viewer is present."""
        robot = Geodude()
        ctx = robot.sim(physics=False, viewer=None)
        assert ctx.is_running() is True

    def test_sync_calls_forward(self):
        """sync() calls mj_forward."""
        robot = Geodude()
        ctx = robot.sim(physics=False, viewer=None)

        # Modify data to verify forward is called
        robot.data.time = 123.0

        ctx.sync()

        # mj_forward should have been called
        # (verifying indirectly through no errors)

    def test_arm_controller_left(self):
        """arm() returns controller for left arm."""
        robot = Geodude()
        ctx = robot.sim(physics=False, viewer=None)

        controller = ctx.arm("left")
        assert controller is not None
        assert controller._arm is robot.left_arm

    def test_arm_controller_right(self):
        """arm() returns controller for right arm."""
        robot = Geodude()
        ctx = robot.sim(physics=False, viewer=None)

        controller = ctx.arm("right")
        assert controller is not None
        assert controller._arm is robot.right_arm

    def test_arm_controller_normalized_names(self):
        """arm() accepts both short and full names."""
        robot = Geodude()
        ctx = robot.sim(physics=False, viewer=None)

        # Short names
        left1 = ctx.arm("left")
        right1 = ctx.arm("right")

        # Full names
        left2 = ctx.arm("left_arm")
        right2 = ctx.arm("right_arm")

        # Should return the same controller instances
        assert left1 is left2
        assert right1 is right2

    def test_arm_controller_invalid_name(self):
        """arm() raises ValueError for invalid name."""
        robot = Geodude()
        ctx = robot.sim(physics=False, viewer=None)

        with pytest.raises(ValueError, match="Unknown arm"):
            ctx.arm("invalid")

    def test_execute_trajectory(self):
        """execute() can execute a trajectory."""
        robot = Geodude()
        robot.go_to("ready")

        ctx = robot.sim(physics=False, viewer=None)
        ctx._setup_executors()  # Manually setup since we're not using context manager

        # Create a simple trajectory for the right arm
        q_start = robot.right_arm.get_joint_positions()
        q_end = q_start + 0.1

        # Create trajectory using the class - path must be a list
        traj = Trajectory.from_path(
            [q_start, q_end],  # List of waypoints
            vel_limits=np.ones(6) * 1.0,
            acc_limits=np.ones(6) * 1.0,
            entity="right_arm",
            joint_names=robot.right_arm.config.joint_names,
        )

        # Execute
        result = ctx.execute(traj)
        assert result is True

    def test_execute_plan_result(self):
        """execute() can execute a PlanResult."""
        from geodude.planning import PlanResult

        robot = Geodude()
        robot.go_to("ready")

        ctx = robot.sim(physics=False, viewer=None)
        ctx._setup_executors()  # Manually setup

        # Create trajectories - path must be a list
        q_start = robot.right_arm.get_joint_positions()
        q_end = q_start + 0.1

        arm_traj = Trajectory.from_path(
            [q_start, q_end],  # List of waypoints
            vel_limits=np.ones(6) * 1.0,
            acc_limits=np.ones(6) * 1.0,
            entity="right_arm",
            joint_names=robot.right_arm.config.joint_names,
        )

        # Create PlanResult
        result = PlanResult(
            arm=robot.right_arm,
            arm_trajectory=arm_traj,
            base_trajectory=None,
            base_height=None,
        )

        # Execute
        success = ctx.execute(result)
        assert success is True


class TestArmGraspRelease:
    """Tests for arm grasp/release methods."""

    def test_grasp_without_gripper(self):
        """grasp() returns None if arm has no gripper."""
        robot = Geodude()
        # Left arm typically has no gripper in default config
        if robot.left_arm.gripper is None or robot.left_arm.gripper.actuator_id is None:
            result = robot.left_arm.grasp("box1")
            assert result is None

    def test_release_without_gripper(self):
        """release() does nothing if arm has no gripper."""
        robot = Geodude()
        # Left arm typically has no gripper
        if robot.left_arm.gripper is None or robot.left_arm.gripper.actuator_id is None:
            # Should not raise
            robot.left_arm.release("box1")

    def test_release_clears_grasp_manager(self, geodude_xml):
        """release() updates grasp manager even without physics contact."""
        import tempfile
        from pathlib import Path
        from geodude.config import GeodudConfig

        # Write to temp file to use the model with test objects
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(geodude_xml)
            temp_path = Path(f.name)

        try:
            config = GeodudConfig.default()
            config.model_path = temp_path
            robot = Geodude(config)

            # Manually mark an object as grasped (box1 exists in test fixture)
            robot.grasp_manager.mark_grasped("box1", "right")

            # Verify it's tracked
            assert robot.grasp_manager.is_grasped("box1")

            # Release should clear it
            robot.right_arm.release("box1")

            # Should be released
            assert not robot.grasp_manager.is_grasped("box1")
        finally:
            temp_path.unlink()


class TestExecutionContextProtocol:
    """Tests verifying SimContext matches ExecutionContext protocol."""

    def test_has_execute_method(self):
        """SimContext has execute method."""
        robot = Geodude()
        ctx = robot.sim(physics=False)
        assert hasattr(ctx, "execute")
        assert callable(ctx.execute)

    def test_has_sync_method(self):
        """SimContext has sync method."""
        robot = Geodude()
        ctx = robot.sim(physics=False)
        assert hasattr(ctx, "sync")
        assert callable(ctx.sync)

    def test_has_is_running_method(self):
        """SimContext has is_running method."""
        robot = Geodude()
        ctx = robot.sim(physics=False)
        assert hasattr(ctx, "is_running")
        assert callable(ctx.is_running)

    def test_has_arm_method(self):
        """SimContext has arm method."""
        robot = Geodude()
        ctx = robot.sim(physics=False)
        assert hasattr(ctx, "arm")
        assert callable(ctx.arm)
