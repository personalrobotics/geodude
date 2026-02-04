"""Tests for unified planning API (Issues #30, #31, #32)."""

import numpy as np
import pytest

from geodude.config import ArmConfig, EntityConfig, VentionBaseConfig
from geodude.planning import PlanResult
from geodude.trajectory import Trajectory, create_linear_trajectory


class TestEntityConfig:
    """Tests for EntityConfig and inheritance."""

    def test_entity_config_creation(self):
        """EntityConfig can be created with required fields."""
        entity = EntityConfig(
            name="test_entity",
            entity_type="custom",
            joint_names=["j1", "j2"],
        )
        assert entity.name == "test_entity"
        assert entity.entity_type == "custom"
        assert entity.joint_names == ["j1", "j2"]

    def test_arm_config_inherits_entity_config(self):
        """ArmConfig inherits from EntityConfig."""
        arm = ArmConfig(
            name="test_arm",
            entity_type="arm",
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            ee_site="ee_site",
        )
        assert isinstance(arm, EntityConfig)
        assert arm.name == "test_arm"
        assert arm.entity_type == "arm"
        assert len(arm.joint_names) == 6

    def test_arm_config_sets_entity_type(self):
        """ArmConfig sets entity_type to 'arm' in __post_init__."""
        arm = ArmConfig(
            name="my_arm",
            entity_type="will_be_overwritten",
            joint_names=["j1"],
        )
        # __post_init__ should set entity_type to "arm"
        assert arm.entity_type == "arm"

    def test_vention_base_config_inherits_entity_config(self):
        """VentionBaseConfig inherits from EntityConfig."""
        base = VentionBaseConfig(
            name="test_base",
            entity_type="base",
            joint_names=["linear_joint"],
            actuator_name="linear_actuator",
        )
        assert isinstance(base, EntityConfig)
        assert base.name == "test_base"
        assert base.entity_type == "base"

    def test_vention_base_config_joint_name_property(self):
        """VentionBaseConfig.joint_name returns first joint_names entry."""
        base = VentionBaseConfig(
            name="test_base",
            entity_type="base",
            joint_names=["my_linear_joint"],
            actuator_name="my_actuator",
        )
        assert base.joint_name == "my_linear_joint"


class TestTrajectoryEntity:
    """Tests for Trajectory entity and joint_names fields."""

    def test_trajectory_with_entity_info(self):
        """Trajectory can store entity and joint_names."""
        traj = Trajectory(
            timestamps=np.array([0.0, 0.1, 0.2]),
            positions=np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
            velocities=np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 0.0]]),
            accelerations=np.array([[50.0, 50.0], [0.0, 0.0], [-50.0, -50.0]]),
            entity="left_arm",
            joint_names=["joint1", "joint2"],
        )
        assert traj.entity == "left_arm"
        assert traj.joint_names == ["joint1", "joint2"]
        assert traj.dof == 2

    def test_trajectory_entity_optional(self):
        """Trajectory entity fields are optional (backward compatible)."""
        traj = Trajectory(
            timestamps=np.array([0.0, 0.1]),
            positions=np.array([[0.0], [1.0]]),
            velocities=np.array([[0.0], [0.0]]),
            accelerations=np.array([[0.0], [0.0]]),
        )
        assert traj.entity is None
        assert traj.joint_names is None

    def test_trajectory_validates_joint_names_length(self):
        """Trajectory raises error if joint_names length != DOF."""
        with pytest.raises(ValueError, match="joint_names length"):
            Trajectory(
                timestamps=np.array([0.0]),
                positions=np.array([[0.0, 0.0]]),  # DOF=2
                velocities=np.array([[0.0, 0.0]]),
                accelerations=np.array([[0.0, 0.0]]),
                joint_names=["only_one"],  # Length=1, should fail
            )

    def test_create_linear_trajectory_with_entity(self):
        """create_linear_trajectory includes entity info."""
        traj = create_linear_trajectory(
            start=0.0,
            end=0.3,
            vel_limit=0.1,
            acc_limit=0.2,
            entity="left_base",
            joint_names=["linear_joint"],
        )
        assert traj.entity == "left_base"
        assert traj.joint_names == ["linear_joint"]
        assert traj.dof == 1

    def test_trajectory_from_path_with_entity(self):
        """Trajectory.from_path includes entity info."""
        path = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        traj = Trajectory.from_path(
            path,
            vel_limits=np.array([1.0, 1.0]),
            acc_limits=np.array([1.0, 1.0]),
            entity="right_arm",
            joint_names=["j1", "j2"],
        )
        assert traj.entity == "right_arm"
        assert traj.joint_names == ["j1", "j2"]


class TestPlanResult:
    """Tests for PlanResult dataclass."""

    def _make_trajectory(self, entity: str, joint_names: list[str], duration: float = 1.0):
        """Helper to create a simple trajectory."""
        n_points = max(2, int(duration / 0.1) + 1)
        return Trajectory(
            timestamps=np.linspace(0, duration, n_points),
            positions=np.zeros((n_points, len(joint_names))),
            velocities=np.zeros((n_points, len(joint_names))),
            accelerations=np.zeros((n_points, len(joint_names))),
            entity=entity,
            joint_names=joint_names,
        )

    def test_plan_result_arm_only(self):
        """PlanResult works with arm trajectory only."""
        arm_traj = self._make_trajectory("right_arm", ["j1", "j2"], duration=2.0)
        result = PlanResult(
            arm=None,  # Would be Arm instance in real usage
            arm_trajectory=arm_traj,
        )
        assert result.success
        assert len(result.trajectories) == 1
        assert result.trajectories[0].entity == "right_arm"
        assert result.total_duration == pytest.approx(2.0, rel=0.1)

    def test_plan_result_with_base(self):
        """PlanResult includes base trajectory in execution order."""
        arm_traj = self._make_trajectory("right_arm", ["j1", "j2"], duration=2.0)
        base_traj = self._make_trajectory("right_base", ["linear"], duration=0.5)

        result = PlanResult(
            arm=None,
            arm_trajectory=arm_traj,
            base_trajectory=base_traj,
            base_height=0.3,
        )

        assert result.success
        assert len(result.trajectories) == 2
        # Base should come first in execution order
        assert result.trajectories[0].entity == "right_base"
        assert result.trajectories[1].entity == "right_arm"
        assert result.base_height == 0.3
        assert result.total_duration == pytest.approx(2.5, rel=0.1)

    def test_plan_result_trajectories_order(self):
        """PlanResult.trajectories returns base first, then arm."""
        arm_traj = self._make_trajectory("left_arm", ["j1"], duration=1.0)
        base_traj = self._make_trajectory("left_base", ["linear"], duration=0.5)

        result = PlanResult(
            arm=None,
            arm_trajectory=arm_traj,
            base_trajectory=base_traj,
        )

        entities = [t.entity for t in result.trajectories]
        assert entities == ["left_base", "left_arm"]


class TestLinearTrajectory:
    """Tests for create_linear_trajectory function."""

    def test_trapezoidal_profile(self):
        """Linear trajectory uses trapezoidal velocity profile."""
        traj = create_linear_trajectory(
            start=0.0,
            end=0.5,
            vel_limit=0.1,
            acc_limit=0.2,
        )
        # Should have acceleration, cruise, deceleration phases
        assert traj.duration > 0
        assert traj.num_waypoints > 2

        # Check endpoints
        assert traj.positions[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert traj.positions[-1, 0] == pytest.approx(0.5, abs=1e-6)

        # Check velocities at endpoints are zero
        assert traj.velocities[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert traj.velocities[-1, 0] == pytest.approx(0.0, abs=1e-6)

    def test_zero_distance_trajectory(self):
        """Zero distance returns trivial trajectory."""
        traj = create_linear_trajectory(
            start=0.3,
            end=0.3,
            vel_limit=0.1,
            acc_limit=0.2,
        )
        assert traj.num_waypoints == 1
        assert traj.duration == 0.0
        assert traj.positions[0, 0] == pytest.approx(0.3)

    def test_reverse_direction(self):
        """Trajectory works for decreasing position."""
        traj = create_linear_trajectory(
            start=0.5,
            end=0.0,
            vel_limit=0.1,
            acc_limit=0.2,
        )
        assert traj.positions[0, 0] == pytest.approx(0.5, abs=1e-6)
        assert traj.positions[-1, 0] == pytest.approx(0.0, abs=1e-6)
        # Velocity should be negative during motion
        mid_idx = len(traj.velocities) // 2
        assert traj.velocities[mid_idx, 0] < 0
