"""Trajectory execution for simulation and real robot."""

import time
from typing import Protocol

import mujoco
import numpy as np

from geodude.trajectory import Trajectory


class Executor(Protocol):
    """Protocol for trajectory execution.

    Executors take a time-parameterized trajectory and execute it,
    either in simulation or on real hardware.
    """

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory.

        Args:
            trajectory: Time-parameterized trajectory to execute

        Returns:
            True if execution completed successfully, False otherwise
        """
        ...


class KinematicExecutor:
    """Kinematic trajectory execution with perfect tracking.

    Directly sets joint positions and velocities without physics simulation.
    Uses mj_forward for forward kinematics only. Useful for:
    - Validating collision-free paths
    - Fast visualization without dynamics
    - Baseline comparison for physics-based execution
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: list[int],
        control_dt: float = 0.008,  # 125 Hz to match UR5e
        viewer=None,
    ):
        """Initialize kinematic executor.

        Args:
            model: MuJoCo model
            data: MuJoCo data (will be modified during execution)
            joint_qpos_indices: Indices into data.qpos for the arm joints
            control_dt: Control update rate in seconds (default: 125 Hz)
            viewer: Optional MuJoCo viewer to sync during execution
        """
        self.model = model
        self.data = data
        self.joint_qpos_indices = joint_qpos_indices
        self.control_dt = control_dt
        self.viewer = viewer

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory kinematically with perfect tracking.

        Directly sets joint positions and velocities at each waypoint.
        No physics simulation - pure kinematic motion.

        Args:
            trajectory: Time-parameterized trajectory

        Returns:
            True if execution completed successfully
        """
        if trajectory.dof != len(self.joint_qpos_indices):
            raise ValueError(
                f"Trajectory DOF {trajectory.dof} doesn't match "
                f"joint count {len(self.joint_qpos_indices)}"
            )

        # Execute trajectory kinematically
        for i in range(trajectory.num_waypoints):
            # Directly set positions and velocities
            for joint_idx, qpos_idx in enumerate(self.joint_qpos_indices):
                self.data.qpos[qpos_idx] = trajectory.positions[i, joint_idx]
                self.data.qvel[qpos_idx] = trajectory.velocities[i, joint_idx]

            # Forward kinematics only (no dynamics)
            mujoco.mj_forward(self.model, self.data)

            # Sync viewer if provided
            if self.viewer is not None:
                self.viewer.sync()

            # Wait for control period
            time.sleep(self.control_dt)

        # Set final state with zero velocity
        for joint_idx, qpos_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[qpos_idx] = trajectory.positions[-1, joint_idx]
            self.data.qvel[qpos_idx] = 0.0

        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        return True


class PhysicsExecutor:
    """Execute trajectories with physics simulation (open-loop control).

    Uses position-controlled actuators to track trajectory waypoints.
    Realistic dynamics with actuator bandwidth limitations (~5 Hz in MuJoCo).
    Useful for:
    - Contact-rich tasks requiring compliance
    - Realistic dynamics simulation
    - Testing with actuator limitations
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: list[int],
        actuator_ids: list[int],
        control_dt: float = 0.008,  # 125 Hz to match UR5e
        viewer=None,
    ):
        """Initialize simulation executor.

        Args:
            model: MuJoCo model
            data: MuJoCo data (will be modified during execution)
            joint_qpos_indices: Indices into data.qpos for the arm joints
            actuator_ids: MuJoCo actuator IDs for position control
            control_dt: Control update rate in seconds (default: 125 Hz)
            viewer: Optional MuJoCo viewer to sync during execution
        """
        self.model = model
        self.data = data
        self.joint_qpos_indices = joint_qpos_indices
        self.actuator_ids = actuator_ids
        self.control_dt = control_dt
        self.viewer = viewer

        # Calculate number of physics steps per control update
        # MuJoCo timestep is typically 0.002s (500 Hz)
        self.steps_per_control = max(1, int(control_dt / model.opt.timestep))

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory in simulation.

        Uses position-controlled actuators to track the trajectory.
        Sends position commands at control_dt frequency and steps physics
        to simulate realistic dynamics.

        Args:
            trajectory: Time-parameterized trajectory

        Returns:
            True if execution completed successfully
        """
        if trajectory.dof != len(self.joint_qpos_indices):
            raise ValueError(
                f"Trajectory DOF {trajectory.dof} doesn't match "
                f"joint count {len(self.joint_qpos_indices)}"
            )

        # Execute trajectory waypoint by waypoint
        for i in range(trajectory.num_waypoints):
            # Send position command to actuators
            for joint_idx, actuator_id in enumerate(self.actuator_ids):
                self.data.ctrl[actuator_id] = trajectory.positions[i, joint_idx]

            # Step physics multiple times for realistic dynamics
            for _ in range(self.steps_per_control):
                mujoco.mj_step(self.model, self.data)

            # Sync viewer if provided (for smooth visualization)
            if self.viewer is not None:
                self.viewer.sync()

            # Wait for control period to maintain real-time execution
            time.sleep(self.control_dt)

        # Send final position command and let physics settle
        for joint_idx, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = trajectory.positions[-1, joint_idx]

        # Step physics longer to settle at final position
        # With realistic actuator dynamics, we need more time to converge
        for _ in range(self.steps_per_control * 20):
            mujoco.mj_step(self.model, self.data)

        # Final viewer sync
        if self.viewer is not None:
            self.viewer.sync()

        return True


class ClosedLoopExecutor:
    """Execute trajectories with closed-loop PD feedback control (DEFAULT).

    Uses position-controlled actuators with PD (proportional-derivative) feedback.
    This is similar to how real robots work - constantly measuring position and
    velocity errors and adjusting commands. Provides much better tracking than
    open-loop PhysicsExecutor.

    This executor combines:
    - Position error feedback (P term): corrects position deviations
    - Velocity error feedback (D term): improves damping and velocity tracking
    - Physics simulation for realistic dynamics

    Default executor for most applications.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: list[int],
        actuator_ids: list[int],
        control_dt: float = 0.008,  # 125 Hz to match UR5e
        viewer=None,
        kp: float = 200.0,  # Position error gain (tuned for very aggressive tracking)
        ki: float = 0.0,    # Integral gain (0.0 optimal for trajectory tracking)
        kd: float = 20.0,   # Velocity error gain (strong damping to prevent overshoot)
    ):
        """Initialize closed-loop feedback executor.

        Args:
            model: MuJoCo model
            data: MuJoCo data (will be modified during execution)
            joint_qpos_indices: Indices into data.qpos for the arm joints
            actuator_ids: MuJoCo actuator IDs for position control
            control_dt: Control update rate in seconds (default: 125 Hz)
            viewer: Optional MuJoCo viewer to sync during execution
            kp: Position error gain for feedback control
            ki: Integral gain for accumulated position error
            kd: Velocity error gain for damping and velocity tracking
        """
        self.model = model
        self.data = data
        self.joint_qpos_indices = joint_qpos_indices
        self.actuator_ids = actuator_ids
        self.control_dt = control_dt
        self.viewer = viewer
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Calculate number of physics steps per control update
        self.steps_per_control = max(1, int(control_dt / model.opt.timestep))

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory with closed-loop PD feedback control.

        At each control cycle:
        1. Measure actual joint positions and velocities
        2. Compute position and velocity errors
        3. Send corrected command using PD feedback: q_cmd = q_des + kp*e_pos + kd*e_vel
        4. Step physics
        5. Repeat

        This mimics real robot servo control with PD feedback.

        Args:
            trajectory: Time-parameterized trajectory

        Returns:
            True if execution completed successfully
        """
        if trajectory.dof != len(self.joint_qpos_indices):
            raise ValueError(
                f"Trajectory DOF {trajectory.dof} doesn't match "
                f"joint count {len(self.joint_qpos_indices)}"
            )

        # Execute trajectory with closed-loop feedback
        for i in range(trajectory.num_waypoints):
            # Get desired state from trajectory
            q_desired = trajectory.positions[i]
            qd_desired = trajectory.velocities[i]

            # Measure actual state (feedback)
            q_actual = np.array([self.data.qpos[idx] for idx in self.joint_qpos_indices])
            qd_actual = np.array([self.data.qvel[idx] for idx in self.joint_qpos_indices])

            # Compute position and velocity errors
            position_error = q_desired - q_actual
            velocity_error = qd_desired - qd_actual

            # Compute corrected command with PD feedback
            # Command = desired + kp * position_error + kd * velocity_error
            # This implements outer-loop PD control on top of MuJoCo's actuator PD control
            q_command = q_desired + self.kp * position_error + self.kd * velocity_error

            # Send corrected position command to actuators
            for joint_idx, actuator_id in enumerate(self.actuator_ids):
                self.data.ctrl[actuator_id] = q_command[joint_idx]

            # Step physics multiple times for realistic dynamics
            for _ in range(self.steps_per_control):
                mujoco.mj_step(self.model, self.data)

            # Sync viewer if provided
            if self.viewer is not None:
                self.viewer.sync()

            # Wait for control period to maintain real-time execution
            time.sleep(self.control_dt)

        # Final settling with feedback control
        q_final = trajectory.positions[-1]
        qd_final = np.zeros(len(self.joint_qpos_indices))  # Desired velocity is zero at rest

        for _ in range(self.steps_per_control * 20):
            # Continue applying PD feedback control during settling
            q_actual = np.array([self.data.qpos[idx] for idx in self.joint_qpos_indices])
            qd_actual = np.array([self.data.qvel[idx] for idx in self.joint_qpos_indices])

            position_error = q_final - q_actual
            velocity_error = qd_final - qd_actual

            q_command = q_final + self.kp * position_error + self.kd * velocity_error

            for joint_idx, actuator_id in enumerate(self.actuator_ids):
                self.data.ctrl[actuator_id] = q_command[joint_idx]

            mujoco.mj_step(self.model, self.data)

        # Final viewer sync
        if self.viewer is not None:
            self.viewer.sync()

        return True


# Alias for backward compatibility
SimExecutor = ClosedLoopExecutor  # Default is now closed-loop


class RealExecutor:
    """Execute trajectories on real robot via ros_control/ur_rtde.

    Streams trajectory points to the robot controller at the same
    frequency used in simulation (125 Hz).

    NOTE: This is a placeholder for future implementation.
    Requires ur_rtde package and real robot connection.
    """

    def __init__(
        self,
        controller_interface: str = "scaled_pos_joint_traj_controller",
        robot_ip: str | None = None,
    ):
        """Initialize real robot executor.

        Args:
            controller_interface: ROS control interface name
            robot_ip: Robot IP address (e.g., "192.168.1.100")

        Raises:
            NotImplementedError: This executor is not yet implemented
        """
        raise NotImplementedError(
            "RealExecutor is not yet implemented. "
            "This will require ur_rtde integration and robot connection setup."
        )

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory on real robot.

        Args:
            trajectory: Time-parameterized trajectory

        Returns:
            True if execution completed successfully

        Raises:
            NotImplementedError: This executor is not yet implemented
        """
        raise NotImplementedError("RealExecutor not yet implemented")

    def stop(self) -> None:
        """Emergency stop - halt robot immediately.

        Raises:
            NotImplementedError: This executor is not yet implemented
        """
        raise NotImplementedError("RealExecutor not yet implemented")
