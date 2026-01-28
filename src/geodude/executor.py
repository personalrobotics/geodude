"""Trajectory execution for simulation and real robot."""

import time
from typing import Protocol

import mujoco
import numpy as np

from geodude.trajectory import Trajectory

# Import for type hints only - avoids circular import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geodude.grasp_manager import GraspManager


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

    This executor has NO physics - joints are set directly. There is no
    position hold because there are no dynamics to fight against.

    For manipulation tasks, set a GraspManager to automatically update
    attached object poses when the gripper moves.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: list[int],
        control_dt: float = 0.008,  # 125 Hz to match UR5e
        viewer=None,
        grasp_manager: "GraspManager | None" = None,
    ):
        """Initialize kinematic executor.

        Args:
            model: MuJoCo model
            data: MuJoCo data (will be modified during execution)
            joint_qpos_indices: Indices into data.qpos for the arm joints
            control_dt: Control update rate in seconds (default: 125 Hz)
            viewer: Optional MuJoCo viewer to sync during execution
            grasp_manager: Optional GraspManager for kinematic manipulation.
                          When set, attached objects move with the gripper.
        """
        self.model = model
        self.data = data
        self.joint_qpos_indices = joint_qpos_indices
        self.control_dt = control_dt
        self.viewer = viewer
        self.grasp_manager = grasp_manager

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

            # Update attached object poses
            if self.grasp_manager is not None:
                self.grasp_manager.update_attached_poses()
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

        # Final update of attached objects
        if self.grasp_manager is not None:
            self.grasp_manager.update_attached_poses()
            mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        return True

    def set_position(self, q: np.ndarray) -> None:
        """Set joint positions directly (kinematic - no physics).

        Also updates poses of any kinematically attached objects.

        Args:
            q: Joint positions to set
        """
        for joint_idx, qpos_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[qpos_idx] = q[joint_idx]
            self.data.qvel[qpos_idx] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Update attached object poses
        if self.grasp_manager is not None:
            self.grasp_manager.update_attached_poses()
            mujoco.mj_forward(self.model, self.data)


class PhysicsExecutor:
    """Execute trajectories with physics simulation and velocity feedforward.

    Uses position-controlled actuators with velocity feedforward, similar to
    the real UR5e's servoj command. MuJoCo's actuator PD (defined in XML)
    handles the low-level servo control.

    Key features:
    - Velocity feedforward: cmd = q_desired + lookahead_time * qd_desired
    - Position hold: When not executing, maintains last commanded position
    - step() method: Call each physics tick to apply control and step physics

    The lookahead_time parameter works like servoj's lookahead_time:
    - Projects current position forward based on trajectory velocity
    - Helps compensate for actuator lag
    - Higher values = smoother but more lag, lower = responsive but can overshoot

    Usage:
        executor = PhysicsExecutor(model, data, joint_indices, actuator_ids)

        # Option 1: Execute a full trajectory (blocking)
        executor.execute(trajectory)

        # Option 2: Manual stepping with position hold
        executor.set_target(q_desired)  # Set target position
        while running:
            executor.step()  # Steps physics and applies control
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: list[int],
        actuator_ids: list[int],
        control_dt: float = 0.008,  # 125 Hz to match UR5e
        lookahead_time: float = 0.1,  # Like servoj lookahead (0.03-0.2s range)
        viewer=None,
    ):
        """Initialize physics executor.

        Args:
            model: MuJoCo model
            data: MuJoCo data (will be modified during execution)
            joint_qpos_indices: Indices into data.qpos for the arm joints
            actuator_ids: MuJoCo actuator IDs for position control
            control_dt: Control update rate in seconds (default: 125 Hz)
            lookahead_time: Velocity feedforward gain in seconds (default: 0.1s).
                           Command = q_desired + lookahead_time * qd_desired.
                           Set to 0 for pure position control (no feedforward).
            viewer: Optional MuJoCo viewer to sync during execution
        """
        self.model = model
        self.data = data
        self.joint_qpos_indices = joint_qpos_indices
        self.actuator_ids = actuator_ids
        self.control_dt = control_dt
        self.lookahead_time = lookahead_time
        self.viewer = viewer

        # Calculate number of physics steps per control update
        # MuJoCo timestep is typically 0.002s (500 Hz)
        self.steps_per_control = max(1, int(control_dt / model.opt.timestep))

        # Position hold state - initialize to current position
        self._target_position = np.array([
            data.qpos[idx] for idx in joint_qpos_indices
        ])
        self._target_velocity = np.zeros(len(joint_qpos_indices))

    @property
    def target_position(self) -> np.ndarray:
        """Current target position being commanded."""
        return self._target_position.copy()

    def set_target(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Set target position (and optionally velocity) for position hold.

        This updates what position the executor commands. Call step() to
        actually apply the control and step physics.

        Args:
            position: Target joint positions
            velocity: Target joint velocities (default: zeros for stationary hold)
        """
        self._target_position = np.asarray(position).copy()
        if velocity is not None:
            self._target_velocity = np.asarray(velocity).copy()
        else:
            self._target_velocity = np.zeros(len(self.joint_qpos_indices))

    def step(self) -> None:
        """Apply control and step physics once.

        This method:
        1. Computes the control command with velocity feedforward
        2. Applies the command to actuators
        3. Steps physics for one control period

        Call this in your simulation loop to maintain position hold or
        track a trajectory that you're updating via set_target().
        """
        # Compute command with velocity feedforward
        q_command = self._target_position + self.lookahead_time * self._target_velocity

        # Apply to actuators
        for joint_idx, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = q_command[joint_idx]

        # Step physics
        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)

        # Sync viewer if provided
        if self.viewer is not None:
            self.viewer.sync()

    def hold(self) -> None:
        """Update target to current position (capture and hold).

        Useful when you want to hold the arm at its current position
        after it has been moved by external forces or another executor.
        """
        self._target_position = np.array([
            self.data.qpos[idx] for idx in self.joint_qpos_indices
        ])
        self._target_velocity = np.zeros(len(self.joint_qpos_indices))

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory with velocity feedforward.

        Uses position-controlled actuators with velocity feedforward to track
        the trajectory. Similar to real UR5e servoj command.

        Command at each step: ctrl = q_desired + lookahead_time * qd_desired

        After execution completes, the executor continues holding the final
        position. Call step() to maintain the hold.

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
            # Update target with trajectory state
            self._target_position = trajectory.positions[i]
            self._target_velocity = trajectory.velocities[i]

            # Apply control and step physics
            self.step()

            # Wait for control period to maintain real-time execution
            time.sleep(self.control_dt)

        # Set final target for position hold (zero velocity)
        self._target_position = trajectory.positions[-1].copy()
        self._target_velocity = np.zeros(len(self.joint_qpos_indices))

        # Settling period - continue holding final position
        for _ in range(self.steps_per_control * 20):
            q_command = self._target_position  # No feedforward - we want to stop
            for joint_idx, actuator_id in enumerate(self.actuator_ids):
                self.data.ctrl[actuator_id] = q_command[joint_idx]
            mujoco.mj_step(self.model, self.data)

        # Final viewer sync
        if self.viewer is not None:
            self.viewer.sync()

        return True

    def get_position(self) -> np.ndarray:
        """Get current actual joint positions.

        Returns:
            Current joint positions from qpos
        """
        return np.array([self.data.qpos[idx] for idx in self.joint_qpos_indices])

    def get_velocity(self) -> np.ndarray:
        """Get current actual joint velocities.

        Returns:
            Current joint velocities from qvel
        """
        return np.array([self.data.qvel[idx] for idx in self.joint_qpos_indices])

    def get_tracking_error(self) -> np.ndarray:
        """Get current position tracking error.

        Returns:
            Difference between target and actual position
        """
        return self._target_position - self.get_position()


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
