"""Tests for the Cartesian velocity control module."""

import numpy as np
import pytest

from geodude.cartesian import (
    CartesianControlConfig,
    TwistStepResult,
    MoveUntilTouchResult,
    get_ee_jacobian,
    twist_to_joint_velocity,
    step_twist,
)


class TestCartesianControlConfig:
    """Tests for CartesianControlConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CartesianControlConfig()
        assert config.length_scale == 0.1
        assert config.damping == 1e-4
        assert config.joint_margin_deg == 5.0
        assert config.velocity_scale == 1.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CartesianControlConfig(
            length_scale=0.2,
            damping=1e-3,
            joint_margin_deg=10.0,
            velocity_scale=0.5,
        )
        assert config.length_scale == 0.2
        assert config.damping == 1e-3
        assert config.joint_margin_deg == 10.0
        assert config.velocity_scale == 0.5

    def test_invalid_length_scale(self):
        """Test that invalid length_scale raises ValueError."""
        with pytest.raises(ValueError, match="length_scale must be > 0"):
            CartesianControlConfig(length_scale=0)
        with pytest.raises(ValueError, match="length_scale must be > 0"):
            CartesianControlConfig(length_scale=-0.1)

    def test_invalid_damping(self):
        """Test that negative damping raises ValueError."""
        with pytest.raises(ValueError, match="damping must be >= 0"):
            CartesianControlConfig(damping=-1e-4)

    def test_invalid_joint_margin(self):
        """Test that negative joint_margin_deg raises ValueError."""
        with pytest.raises(ValueError, match="joint_margin_deg must be >= 0"):
            CartesianControlConfig(joint_margin_deg=-5.0)

    def test_invalid_velocity_scale(self):
        """Test that invalid velocity_scale raises ValueError."""
        with pytest.raises(ValueError, match="velocity_scale must be in"):
            CartesianControlConfig(velocity_scale=0)
        with pytest.raises(ValueError, match="velocity_scale must be in"):
            CartesianControlConfig(velocity_scale=1.5)
        with pytest.raises(ValueError, match="velocity_scale must be in"):
            CartesianControlConfig(velocity_scale=-0.1)


class TestTwistToJointVelocity:
    """Tests for the core twist_to_joint_velocity QP solver."""

    @pytest.fixture
    def typical_6dof_setup(self):
        """Create typical 6-DOF arm parameters."""
        # Random but well-conditioned Jacobian
        np.random.seed(42)
        J = np.random.randn(6, 6)
        # Make it reasonably well-conditioned
        U, s, Vt = np.linalg.svd(J)
        s = np.clip(s, 0.5, 2.0)  # Condition number ~4
        J = U @ np.diag(s) @ Vt

        q_current = np.zeros(6)
        q_min = np.full(6, -np.pi)
        q_max = np.full(6, np.pi)
        qd_max = np.full(6, 2.0)  # 2 rad/s max
        dt = 0.004  # 4ms timestep

        return {
            "J": J,
            "q_current": q_current,
            "q_min": q_min,
            "q_max": q_max,
            "qd_max": qd_max,
            "dt": dt,
        }

    def test_zero_twist_gives_zero_velocity(self, typical_6dof_setup):
        """Zero twist should produce (near) zero joint velocities."""
        params = typical_6dof_setup
        twist = np.zeros(6)

        result = twist_to_joint_velocity(
            J=params["J"],
            twist=twist,
            q_current=params["q_current"],
            q_min=params["q_min"],
            q_max=params["q_max"],
            qd_max=params["qd_max"],
            dt=params["dt"],
        )

        # Should be essentially zero (within numerical tolerance)
        assert np.allclose(result.joint_velocities, 0, atol=1e-8)
        assert result.achieved_fraction == 1.0  # No motion requested

    def test_achievable_twist_fully_achieved(self, typical_6dof_setup):
        """Small achievable twist should be fully achieved."""
        params = typical_6dof_setup
        twist = np.array([0.01, 0, 0, 0, 0, 0])  # 1cm/s in x

        result = twist_to_joint_velocity(
            J=params["J"],
            twist=twist,
            q_current=params["q_current"],
            q_min=params["q_min"],
            q_max=params["q_max"],
            qd_max=params["qd_max"],
            dt=params["dt"],
        )

        # Should achieve most of the twist
        assert result.achieved_fraction > 0.95
        # Twist error should be small
        assert result.twist_error < 0.001

    def test_velocity_limits_respected(self, typical_6dof_setup):
        """Joint velocities should respect velocity limits."""
        params = typical_6dof_setup
        # Large twist that would require high joint velocities
        twist = np.array([1.0, 1.0, 1.0, 0, 0, 0])  # 1m/s - very fast

        result = twist_to_joint_velocity(
            J=params["J"],
            twist=twist,
            q_current=params["q_current"],
            q_min=params["q_min"],
            q_max=params["q_max"],
            qd_max=params["qd_max"],
            dt=params["dt"],
        )

        # All joint velocities should be within limits
        assert np.all(np.abs(result.joint_velocities) <= params["qd_max"] + 1e-6)

    def test_position_limits_respected(self, typical_6dof_setup):
        """Joint velocities should respect position limits."""
        params = typical_6dof_setup
        # Start near upper limit but outside margin (10 degrees from limit, margin is 5)
        q_current = params["q_max"] - np.deg2rad(10)
        twist = np.array([0.1, 0, 0, 0, 0, 0])

        config = CartesianControlConfig(joint_margin_deg=5.0)

        result = twist_to_joint_velocity(
            J=params["J"],
            twist=twist,
            q_current=q_current,
            q_min=params["q_min"],
            q_max=params["q_max"],
            qd_max=params["qd_max"],
            dt=params["dt"],
            config=config,
        )

        # After one step, position should stay within limits (hard limits, not margin)
        q_new = q_current + result.joint_velocities * params["dt"]
        # Allow small numerical tolerance - should respect hard limits
        assert np.all(q_new <= params["q_max"] + 1e-6)
        assert np.all(q_new >= params["q_min"] - 1e-6)

    def test_velocity_scale_reduces_speed(self, typical_6dof_setup):
        """velocity_scale should reduce maximum joint velocities."""
        params = typical_6dof_setup
        # Use a large twist that will hit velocity limits
        twist = np.array([5.0, 5.0, 0, 0, 0, 0])

        # Full speed
        result_full = twist_to_joint_velocity(
            J=params["J"],
            twist=twist,
            q_current=params["q_current"],
            q_min=params["q_min"],
            q_max=params["q_max"],
            qd_max=params["qd_max"],
            dt=params["dt"],
            config=CartesianControlConfig(velocity_scale=1.0),
        )

        # Half speed
        result_half = twist_to_joint_velocity(
            J=params["J"],
            twist=twist,
            q_current=params["q_current"],
            q_min=params["q_min"],
            q_max=params["q_max"],
            qd_max=params["qd_max"],
            dt=params["dt"],
            config=CartesianControlConfig(velocity_scale=0.5),
        )

        # Half speed config means velocity limits are halved
        # So max joint velocity with half scale should be <= qd_max * 0.5
        max_vel_half = np.max(np.abs(result_half.joint_velocities))
        assert max_vel_half <= params["qd_max"][0] * 0.5 + 1e-6

    def test_limiting_factor_detection(self, typical_6dof_setup):
        """Test that limiting factors are correctly identified."""
        params = typical_6dof_setup

        # Very large twist should hit velocity limits
        twist = np.array([10.0, 10.0, 10.0, 0, 0, 0])
        result = twist_to_joint_velocity(
            J=params["J"],
            twist=twist,
            q_current=params["q_current"],
            q_min=params["q_min"],
            q_max=params["q_max"],
            qd_max=params["qd_max"],
            dt=params["dt"],
        )
        # Should hit some limit
        assert result.limiting_factor in ["velocity", "joint_limit", None]

    def test_result_types(self, typical_6dof_setup):
        """Test that result has correct types."""
        params = typical_6dof_setup
        twist = np.array([0.01, 0, 0, 0, 0, 0])

        result = twist_to_joint_velocity(
            J=params["J"],
            twist=twist,
            q_current=params["q_current"],
            q_min=params["q_min"],
            q_max=params["q_max"],
            qd_max=params["qd_max"],
            dt=params["dt"],
        )

        assert isinstance(result, TwistStepResult)
        assert isinstance(result.joint_velocities, np.ndarray)
        assert result.joint_velocities.shape == (6,)
        assert isinstance(result.twist_error, float)
        assert isinstance(result.achieved_fraction, float)
        assert 0.0 <= result.achieved_fraction <= 1.0


class TestGetEEJacobian:
    """Tests for Jacobian computation."""

    def test_jacobian_shape(self, mujoco_model_and_data, arm_joint_names):
        """Test that Jacobian has correct shape."""
        import mujoco

        model, data = mujoco_model_and_data

        # Get EE site ID
        ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_ur5e/gripper/tcp"
        )

        # Get joint velocity indices
        joint_vel_indices = []
        for name in arm_joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof_adr = model.jnt_dofadr[joint_id]
            joint_vel_indices.append(dof_adr)

        J = get_ee_jacobian(model, data, ee_site_id, joint_vel_indices)

        assert J.shape == (6, 6)

    def test_jacobian_changes_with_config(self):
        """Test that Jacobian changes when joint configuration changes."""
        import mujoco
        from geodude import Geodude

        robot = Geodude()
        robot.go_to("ready")
        mujoco.mj_forward(robot.model, robot.data)

        arm = robot.right_arm
        model = robot.model
        data = robot.data

        # Jacobian at ready config
        J1 = get_ee_jacobian(
            model, data, arm.ee_site_id, arm.joint_qpos_indices
        ).copy()

        # Change configuration
        for idx in arm.joint_qpos_indices:
            data.qpos[idx] += 0.5
        mujoco.mj_forward(model, data)

        # Jacobian at new config
        J2 = get_ee_jacobian(model, data, arm.ee_site_id, arm.joint_qpos_indices)

        # Both should be non-zero and different
        assert not np.allclose(J1, 0), "J1 should be non-zero"
        assert not np.allclose(J2, 0), "J2 should be non-zero"
        assert not np.allclose(J1, J2), "Jacobians should differ"


class TestStepTwist:
    """Tests for step_twist function with actual robot."""

    @pytest.fixture
    def robot(self):
        """Create a Geodude robot for testing."""
        from geodude import Geodude

        return Geodude()

    def test_step_twist_returns_valid_positions(self, robot):
        """Test that step_twist returns valid joint positions."""
        import mujoco

        robot.go_to("ready")
        mujoco.mj_forward(robot.model, robot.data)

        arm = robot.right_arm
        twist = np.array([0.01, 0, 0, 0, 0, 0])  # 1cm/s in x

        q_new, result = step_twist(arm, twist, frame="world", dt=0.004)

        assert q_new.shape == (6,)
        assert isinstance(result, TwistStepResult)
        assert result.achieved_fraction > 0.5  # Should achieve significant motion

    def test_step_twist_world_vs_hand_frame(self, robot):
        """Test that world and hand frames produce different results."""
        import mujoco

        robot.go_to("ready")
        mujoco.mj_forward(robot.model, robot.data)

        arm = robot.right_arm
        twist = np.array([0.01, 0, 0, 0, 0, 0])

        q_world, _ = step_twist(arm, twist, frame="world", dt=0.004)
        q_hand, _ = step_twist(arm, twist, frame="hand", dt=0.004)

        # Should be different (unless EE happens to be aligned with world)
        # In ready pose, they should differ
        assert not np.allclose(q_world, q_hand, atol=1e-6)

    def test_step_twist_integration_accuracy(self, robot):
        """Test that multiple steps integrate accurately."""
        import mujoco

        robot.go_to("ready")
        mujoco.mj_forward(robot.model, robot.data)

        arm = robot.right_arm
        ee_start = robot.data.site_xpos[arm.ee_site_id].copy()

        # Move down at 1cm/s for 10 steps (4ms each = 40ms total)
        twist = np.array([0, 0, -0.01, 0, 0, 0])
        dt = 0.004
        n_steps = 10

        for _ in range(n_steps):
            q_new, _ = step_twist(arm, twist, frame="world", dt=dt)
            for i, qpos_idx in enumerate(arm.joint_qpos_indices):
                robot.data.qpos[qpos_idx] = q_new[i]
            mujoco.mj_forward(robot.model, robot.data)

        ee_end = robot.data.site_xpos[arm.ee_site_id].copy()
        displacement = ee_end - ee_start

        # Expected: -0.01 * 0.004 * 10 = -0.0004m in z
        expected_z = -0.01 * dt * n_steps
        assert np.isclose(displacement[2], expected_z, atol=1e-5)
        # x and y should be negligible
        assert np.abs(displacement[0]) < 1e-5
        assert np.abs(displacement[1]) < 1e-5


class TestMoveUntilTouchResult:
    """Tests for MoveUntilTouchResult dataclass."""

    def test_success_contact(self):
        """Test successful contact result."""
        result = MoveUntilTouchResult(
            success=True,
            terminated_by="contact",
            distance_moved=0.03,
            final_force=np.array([0, 0, 5.0]),
            final_torque=np.array([0, 0, 0]),
        )
        assert result.success
        assert result.terminated_by == "contact"
        assert result.distance_moved == 0.03

    def test_max_distance_failure(self):
        """Test max_distance termination result."""
        result = MoveUntilTouchResult(
            success=False,
            terminated_by="max_distance",
            distance_moved=0.05,
        )
        assert not result.success
        assert result.terminated_by == "max_distance"

    def test_no_progress_failure(self):
        """Test no_progress termination result."""
        result = MoveUntilTouchResult(
            success=False,
            terminated_by="no_progress",
            distance_moved=0.01,
        )
        assert not result.success
        assert result.terminated_by == "no_progress"
