"""Trajectory representation and time-optimal retiming using TOPP-RA."""

from dataclasses import dataclass

import numpy as np
import toppra
import toppra.algorithm as algo
import toppra.constraint as constraint


@dataclass
class Trajectory:
    """Time-parameterized robot trajectory.

    Stores positions, velocities, accelerations at discrete timestamps.
    Compatible with both MuJoCo simulation and real robot execution.

    The trajectory is dense (waypoints at control frequency) to avoid
    needing spline evaluation in the control loop.
    """

    timestamps: np.ndarray  # (N,) seconds from start
    positions: np.ndarray  # (N, dof) joint positions in radians
    velocities: np.ndarray  # (N, dof) joint velocities in rad/s
    accelerations: np.ndarray  # (N, dof) joint accelerations in rad/s²

    def __post_init__(self):
        """Validate trajectory dimensions."""
        n_waypoints = len(self.timestamps)
        if self.positions.shape[0] != n_waypoints:
            raise ValueError(
                f"Position shape {self.positions.shape} doesn't match "
                f"timestamps length {n_waypoints}"
            )
        if self.velocities.shape[0] != n_waypoints:
            raise ValueError(
                f"Velocity shape {self.velocities.shape} doesn't match "
                f"timestamps length {n_waypoints}"
            )
        if self.accelerations.shape[0] != n_waypoints:
            raise ValueError(
                f"Acceleration shape {self.accelerations.shape} doesn't match "
                f"timestamps length {n_waypoints}"
            )

        if self.positions.shape[1] != self.velocities.shape[1]:
            raise ValueError(
                f"DOF mismatch: positions {self.positions.shape[1]} "
                f"vs velocities {self.velocities.shape[1]}"
            )

    @property
    def duration(self) -> float:
        """Total duration of trajectory in seconds."""
        return float(self.timestamps[-1])

    @property
    def dof(self) -> int:
        """Degrees of freedom (number of joints)."""
        return self.positions.shape[1]

    @property
    def num_waypoints(self) -> int:
        """Number of waypoints in trajectory."""
        return len(self.timestamps)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate position, velocity, acceleration at time t.

        Args:
            t: Time in seconds (clamped to [0, duration])

        Returns:
            Tuple of (position, velocity, acceleration) arrays
        """
        # Clamp time to valid range
        t = np.clip(t, 0.0, self.duration)

        # Find surrounding waypoints
        idx = np.searchsorted(self.timestamps, t)

        # Handle edge cases
        if idx == 0:
            return self.positions[0], self.velocities[0], self.accelerations[0]
        if idx >= len(self.timestamps):
            return self.positions[-1], self.velocities[-1], self.accelerations[-1]

        # Linear interpolation between waypoints
        t0, t1 = self.timestamps[idx - 1], self.timestamps[idx]
        alpha = (t - t0) / (t1 - t0)

        pos = (1 - alpha) * self.positions[idx - 1] + alpha * self.positions[idx]
        vel = (1 - alpha) * self.velocities[idx - 1] + alpha * self.velocities[idx]
        acc = (1 - alpha) * self.accelerations[idx - 1] + alpha * self.accelerations[idx]

        return pos, vel, acc

    @classmethod
    def from_path(
        cls,
        path: list[np.ndarray],
        vel_limits: np.ndarray,
        acc_limits: np.ndarray,
        control_dt: float = 0.008,  # 125 Hz
    ) -> "Trajectory":
        """Create time-optimal trajectory from geometric path using TOPP-RA.

        Args:
            path: List of waypoint configurations (joint angles in radians)
            vel_limits: Joint velocity limits in rad/s (shape: (dof,))
            acc_limits: Joint acceleration limits in rad/s² (shape: (dof,))
            control_dt: Control timestep in seconds (default: 125 Hz)

        Returns:
            Trajectory with time-optimal parameterization respecting limits

        Raises:
            ValueError: If path is empty or has inconsistent dimensions
            RuntimeError: If TOPP-RA fails to find a valid parameterization
        """
        if not path:
            raise ValueError("Path cannot be empty")

        path_array = np.array(path)
        if path_array.ndim != 2:
            raise ValueError(f"Path must be 2D array, got shape {path_array.shape}")

        dof = path_array.shape[1]
        if len(vel_limits) != dof:
            raise ValueError(
                f"Velocity limits dimension {len(vel_limits)} doesn't match "
                f"path DOF {dof}"
            )
        if len(acc_limits) != dof:
            raise ValueError(
                f"Acceleration limits dimension {len(acc_limits)} doesn't match "
                f"path DOF {dof}"
            )

        # Remove consecutive duplicate waypoints that could cause numerical issues
        # Use a very small threshold to only remove true duplicates
        filtered_path = [path_array[0]]
        for i in range(1, len(path_array)):
            if not np.allclose(path_array[i], filtered_path[-1], atol=1e-10):
                filtered_path.append(path_array[i])

        path_array = np.array(filtered_path)

        # If all waypoints are the same (robot already at goal), return trivial trajectory
        if len(path_array) < 2:
            return cls(
                timestamps=np.array([0.0]),
                positions=path_array,
                velocities=np.zeros_like(path_array),
                accelerations=np.zeros_like(path_array),
            )

        # Create path parameterized from 0 to 1 (TOPP-RA convention)
        # SplineInterpolator uses cubic splines for smooth C2-continuous trajectories.
        # Note: splines can deviate ~3-4° from linear segments at direction changes,
        # so collision validation should check the retimed trajectory, not just waypoints.
        path_positions = toppra.SplineInterpolator(
            np.linspace(0, 1, len(path_array)), path_array
        )

        # Create velocity/acceleration limits in min/max format
        vel_limits_minmax = np.stack((-vel_limits, vel_limits)).T
        acc_limits_minmax = np.stack((-acc_limits, acc_limits)).T

        # Define velocity and acceleration constraints
        pc_vel = constraint.JointVelocityConstraint(vel_limits_minmax)
        pc_acc = constraint.JointAccelerationConstraint(acc_limits_minmax)

        # Run TOPP-RA algorithm
        instance = algo.TOPPRA(
            [pc_vel, pc_acc],
            path_positions,
            parametrizer="ParametrizeConstAccel",
        )

        # Compute time-optimal parameterization
        jnt_traj = instance.compute_trajectory()

        if jnt_traj is None:
            raise RuntimeError(
                "TOPP-RA failed to find valid trajectory. "
                "Path may violate velocity or acceleration constraints."
            )

        # Sample trajectory at control frequency
        duration = jnt_traj.duration
        # Start at t=0 to include the initial position
        timestamps = np.arange(0.0, duration, control_dt)
        # Ensure final timestamp is included
        if not np.isclose(timestamps[-1], duration, rtol=0.0, atol=1e-8):
            timestamps = np.append(timestamps, duration)

        # Evaluate trajectory at each timestamp
        positions = jnt_traj(timestamps)
        velocities = jnt_traj(timestamps, 1)  # First derivative
        accelerations = jnt_traj(timestamps, 2)  # Second derivative

        return cls(
            timestamps=timestamps,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
        )
