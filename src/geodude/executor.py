"""Trajectory execution for simulation and real robot."""

import logging
import time
from typing import Protocol

import mujoco
import numpy as np

from geodude.trajectory import Trajectory

logger = logging.getLogger(__name__)

# Import for type hints only - avoids circular import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geodude.grasp_manager import GraspManager
    from geodude.robot import Geodude


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
        recorder=None,
        viewer_sync_interval: float = 0.033,  # ~30 Hz
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
            recorder: Optional FrameRecorder to capture frames during execution
            viewer_sync_interval: Minimum time between viewer syncs in seconds.
                                 Lower = smoother but slower. 0 = sync every step.
        """
        self.model = model
        self.data = data
        self.joint_qpos_indices = joint_qpos_indices
        self.control_dt = control_dt
        self.viewer = viewer
        self.grasp_manager = grasp_manager
        self.recorder = recorder

        # Viewer sync throttling
        self._last_viewer_sync = 0.0
        self._viewer_sync_interval = viewer_sync_interval

        # Target position for step() - initialize to current
        self._target_position = np.array([
            data.qpos[idx] for idx in joint_qpos_indices
        ])

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

            # Sync viewer if provided (throttled to ~30Hz)
            if self.viewer is not None:
                now = time.time()
                if now - self._last_viewer_sync >= self._viewer_sync_interval:
                    self.viewer.sync()
                    self._last_viewer_sync = now

            # Capture frame for recording (every 8th frame for reasonable GIF size)
            if self.recorder is not None and i % 8 == 0:
                self.recorder.capture()

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
        self._target_position = np.asarray(q).copy()
        for joint_idx, qpos_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[qpos_idx] = q[joint_idx]
            self.data.qvel[qpos_idx] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Update attached object poses
        if self.grasp_manager is not None:
            self.grasp_manager.update_attached_poses()
            mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        """Apply current target position and sync viewer (kinematic).

        For cartesian control - set target with set_position() then call step().
        """
        # Apply current target
        for joint_idx, qpos_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[qpos_idx] = self._target_position[joint_idx]
            self.data.qvel[qpos_idx] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Update attached object poses
        if self.grasp_manager is not None:
            self.grasp_manager.update_attached_poses()
            mujoco.mj_forward(self.model, self.data)

        # Sync viewer (throttled to ~30Hz)
        if self.viewer is not None:
            now = time.time()
            if now - self._last_viewer_sync >= self._viewer_sync_interval:
                self.viewer.sync()
                self._last_viewer_sync = now


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
        joint_qvel_indices: list[int] | None = None,
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
            joint_qvel_indices: Indices into data.qvel for the arm joints.
                               If None, assumes same as qpos indices (valid for hinge joints).
        """
        self.model = model
        self.data = data
        self.joint_qpos_indices = joint_qpos_indices
        self.joint_qvel_indices = joint_qvel_indices if joint_qvel_indices is not None else joint_qpos_indices
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

        # Viewer sync throttling (~30Hz)
        self._last_viewer_sync = 0.0
        self._viewer_sync_interval = 0.033  # 30 Hz

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

        # Sync viewer if provided (throttled to ~30Hz)
        if self.viewer is not None:
            now = time.time()
            if now - self._last_viewer_sync >= self._viewer_sync_interval:
                self.viewer.sync()
                self._last_viewer_sync = now

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
        return np.array([self.data.qvel[idx] for idx in self.joint_qvel_indices])

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


class RobotPhysicsController:
    """Physics controller that manages all robot actuators.

    In physics simulation, ALL actuators must be controlled - otherwise joints
    will fall under gravity. This controller ensures that when one arm executes
    a trajectory, all other actuators hold their positions.

    Usage:
        controller = RobotPhysicsController(robot, viewer=viewer)

        # Execute on right arm (left arm automatically holds position)
        controller.execute_right(trajectory)

        # Or create an executor for a specific arm
        executor = controller.get_executor("right")
        executor.execute(trajectory)
    """

    def __init__(
        self,
        robot: "Geodude",
        viewer=None,
        recorder=None,
        initial_positions: dict[str, np.ndarray] | None = None,
        viewer_sync_interval: float = 0.033,  # ~30 Hz
    ):
        """Initialize physics controller for the robot.

        Args:
            robot: Geodude robot instance
            viewer: Optional MuJoCo viewer to sync
            recorder: Optional FrameRecorder to capture frames during execution
            initial_positions: Optional dict mapping arm names ("left", "right") to
                             initial joint positions. If not provided for an arm,
                             uses current qpos (which may be zeros if not initialized).
            viewer_sync_interval: Minimum time between viewer syncs in seconds.
                                 Lower = smoother but slower. 0 = sync every step.

        Physics parameters are read from robot.config.physics.
        """
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.viewer = viewer
        self.recorder = recorder

        # Get physics config from robot
        physics_config = robot.config.physics
        self.execution_config = physics_config.execution
        self.gripper_config = physics_config.gripper
        self.recovery_config = physics_config.recovery

        # Control timing from config
        self.control_dt = self.execution_config.control_dt
        self.lookahead_time = self.execution_config.lookahead_time
        self.steps_per_control = max(1, int(self.control_dt / self.model.opt.timestep))
        self._step_count = 0  # For frame capture interval

        # Viewer sync throttling
        self._last_viewer_sync = 0.0
        self._viewer_sync_interval = viewer_sync_interval

        # Collect actuator info for both arms and grippers
        self._arms = {}
        self._grippers = {}
        for arm_name in ["left", "right"]:
            arm = getattr(robot, f"{arm_name}_arm")
            # Actuator names: "left_ur5e/shoulder_pan", "left_ur5e/elbow", etc.
            # Joint names: "left_ur5e/shoulder_pan_joint", etc.
            # Strip "_joint" suffix and use the full joint name path
            actuator_ids = []
            for jname in arm.config.joint_names:
                # Remove "_joint" suffix if present
                act_name = jname.replace("_joint", "")
                actuator_ids.append(self.model.actuator(act_name).id)

            # Use provided initial position or fall back to current qpos
            if initial_positions and arm_name in initial_positions:
                target_pos = np.asarray(initial_positions[arm_name]).copy()
            else:
                target_pos = arm.get_joint_positions().copy()

            self._arms[arm_name] = {
                "arm": arm,
                "actuator_ids": actuator_ids,
                "joint_qpos_indices": arm.joint_qpos_indices,
                "joint_qvel_indices": arm.joint_qvel_indices,
                "target_position": target_pos,
                "target_velocity": np.zeros(len(actuator_ids)),
            }

            # Store gripper info
            gripper = arm.gripper
            if gripper and gripper.actuator_id is not None:
                self._grippers[arm_name] = {
                    "gripper": gripper,
                    "actuator_id": gripper.actuator_id,
                    "ctrl_open": gripper.ctrl_open,
                    "ctrl_closed": gripper.ctrl_closed,
                    "target_ctrl": self.data.ctrl[gripper.actuator_id],
                }

        # Track base actuators for coordinated control
        self._bases = {}
        for side in ["left", "right"]:
            base = getattr(robot, f"{side}_base", None)
            if base is not None:
                actuator_name = f"{side}_linear_actuator"
                try:
                    actuator_id = self.model.actuator(actuator_name).id
                    self._bases[side] = {
                        "actuator_id": actuator_id,
                        "target_height": base.height,
                    }
                except KeyError:
                    pass

        # Initialize qpos and ctrl to target positions to avoid violent jumps
        # This ensures physics starts with arms already at target, not at zeros
        for arm_name, info in self._arms.items():
            for i, qpos_idx in enumerate(info["joint_qpos_indices"]):
                self.data.qpos[qpos_idx] = info["target_position"][i]
            for qvel_idx in info["joint_qvel_indices"]:
                self.data.qvel[qvel_idx] = 0.0
            for i, actuator_id in enumerate(info["actuator_ids"]):
                self.data.ctrl[actuator_id] = info["target_position"][i]

        # Initialize base ctrl to target heights
        for info in self._bases.values():
            self.data.ctrl[info["actuator_id"]] = info["target_height"]

        # Update kinematics so everything is consistent
        mujoco.mj_forward(self.model, self.data)

    def hold_all(self) -> None:
        """Update all position hold targets to current positions."""
        for info in self._arms.values():
            info["target_position"] = np.array([
                self.data.qpos[idx] for idx in info["joint_qpos_indices"]
            ])
            info["target_velocity"] = np.zeros(len(info["actuator_ids"]))

    def set_arm_target(
        self,
        arm_name: str,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Set target position and velocity for an arm.

        Use this for cartesian control - set the target then call step().

        Args:
            arm_name: "left" or "right"
            position: Target joint positions (rad)
            velocity: Target joint velocities (rad/s), or None for zero velocity
        """
        if arm_name not in self._arms:
            raise ValueError(f"Unknown arm: {arm_name}")

        info = self._arms[arm_name]
        info["target_position"] = np.asarray(position).copy()
        if velocity is not None:
            info["target_velocity"] = np.asarray(velocity).copy()
        else:
            info["target_velocity"] = np.zeros(len(info["actuator_ids"]))

    def step(self) -> None:
        """Apply control to all actuators and step physics.

        Uses lookahead_time (default 0.1s) for trajectory following.
        For reactive/streaming control, use step_reactive() instead.
        """
        # Apply control to all arms
        for info in self._arms.values():
            q_command = info["target_position"] + self.lookahead_time * info["target_velocity"]
            for joint_idx, actuator_id in enumerate(info["actuator_ids"]):
                self.data.ctrl[actuator_id] = q_command[joint_idx]

        # Apply control to all grippers
        for info in self._grippers.values():
            self.data.ctrl[info["actuator_id"]] = info["target_ctrl"]

        # Apply control to all bases (hold positions)
        for info in self._bases.values():
            self.data.ctrl[info["actuator_id"]] = info["target_height"]

        # Step physics
        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)

        # Sync viewer (throttled to ~30Hz)
        if self.viewer is not None:
            now = time.time()
            if now - self._last_viewer_sync >= self._viewer_sync_interval:
                self.viewer.sync()
                self._last_viewer_sync = now

        # Capture frame for recording (every 4th step)
        self._step_count += 1
        if self.recorder is not None and self._step_count % 4 == 0:
            self.recorder.capture()

    def step_reactive(
        self,
        arm_name: str,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Step physics with reactive/streaming control for one arm.

        Unlike step() which uses a large lookahead (0.1s) for trajectory
        following, this uses control_dt lookahead - treating each step as
        a mini-trajectory segment. This gives smooth motion without overshoot.

        The commanded position leads the target by control_dt * velocity,
        so the PD controller smoothly interpolates rather than chasing
        discrete position jumps.

        Args:
            arm_name: "left" or "right"
            position: Target joint positions (rad)
            velocity: Target joint velocities (rad/s) for feedforward.
                     If provided, enables smooth streaming motion.
        """
        if arm_name not in self._arms:
            raise ValueError(f"Unknown arm: {arm_name}")

        info = self._arms[arm_name]

        # Update stored target for this arm
        info["target_position"] = np.asarray(position).copy()
        if velocity is not None:
            info["target_velocity"] = np.asarray(velocity).copy()
        else:
            info["target_velocity"] = np.zeros(len(info["actuator_ids"]))

        # Compute command with small lookahead (2x control_dt = 16ms, not 0.1s)
        # This treats each step as a mini-trajectory segment.
        # Using 2x control_dt gives the PD controller enough "heads up" for smooth
        # interpolation while staying small enough to avoid overshoot.
        reactive_lookahead = 2.0 * self.control_dt
        q_command = info["target_position"] + reactive_lookahead * info["target_velocity"]

        # Apply to this arm
        for joint_idx, actuator_id in enumerate(info["actuator_ids"]):
            self.data.ctrl[actuator_id] = q_command[joint_idx]

        # Apply control to other arms (hold position, no velocity feedforward)
        for other_name, other_info in self._arms.items():
            if other_name != arm_name:
                for joint_idx, actuator_id in enumerate(other_info["actuator_ids"]):
                    self.data.ctrl[actuator_id] = other_info["target_position"][joint_idx]

        # Apply control to all grippers
        for grip_info in self._grippers.values():
            self.data.ctrl[grip_info["actuator_id"]] = grip_info["target_ctrl"]

        # Apply control to all bases (hold positions)
        for base_info in self._bases.values():
            self.data.ctrl[base_info["actuator_id"]] = base_info["target_height"]

        # Step physics
        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)

        # Sync viewer (throttled)
        if self.viewer is not None:
            now = time.time()
            if now - self._last_viewer_sync >= self._viewer_sync_interval:
                self.viewer.sync()
                self._last_viewer_sync = now

        # Capture frame for recording
        self._step_count += 1
        if self.recorder is not None and self._step_count % 4 == 0:
            self.recorder.capture()

    def _wait_for_convergence(
        self,
        arm_name: str,
        position_tolerance: float | None = None,
        velocity_tolerance: float | None = None,
        timeout_steps: int | None = None,
    ) -> bool:
        """Wait for arm to converge to target position.

        Args:
            arm_name: "left" or "right"
            position_tolerance: Max position error in radians (default from config)
            velocity_tolerance: Max velocity in rad/s (default from config)
            timeout_steps: Max steps to wait (default from config)

        Returns:
            True if converged, False if timeout
        """
        # Use config defaults if not specified
        if position_tolerance is None:
            position_tolerance = self.execution_config.position_tolerance
        if velocity_tolerance is None:
            velocity_tolerance = self.execution_config.velocity_tolerance
        if timeout_steps is None:
            timeout_steps = self.execution_config.convergence_timeout_steps

        info = self._arms[arm_name]

        for step in range(timeout_steps):
            self.step()

            # Check position error
            current_pos = np.array([
                self.data.qpos[idx] for idx in info["joint_qpos_indices"]
            ])
            pos_error = np.abs(info["target_position"] - current_pos)

            # Check velocity (use qvel indices, not qpos indices)
            current_vel = np.array([
                self.data.qvel[idx] for idx in info["joint_qvel_indices"]
            ])

            if np.all(pos_error < position_tolerance) and np.all(np.abs(current_vel) < velocity_tolerance):
                return True

        # Log final error on timeout for debugging
        max_pos_err = np.max(pos_error)
        max_vel = np.max(np.abs(current_vel))
        logger.warning(
            f"Convergence timeout for {arm_name}: "
            f"max_pos_err={np.rad2deg(max_pos_err):.2f}° (limit {np.rad2deg(position_tolerance):.2f}°), "
            f"max_vel={max_vel:.3f} rad/s (limit {velocity_tolerance:.3f})"
        )
        return False

    def execute(self, arm_name: str, trajectory: Trajectory) -> bool:
        """Execute trajectory on specified arm while others hold position.

        Args:
            arm_name: "left" or "right"
            trajectory: Trajectory to execute

        Returns:
            True if execution completed and converged, False if timeout
        """
        if arm_name not in self._arms:
            raise ValueError(f"Unknown arm: {arm_name}")

        info = self._arms[arm_name]

        if trajectory.dof != len(info["joint_qpos_indices"]):
            raise ValueError(
                f"Trajectory DOF {trajectory.dof} doesn't match "
                f"arm joint count {len(info['joint_qpos_indices'])}"
            )

        # Execute trajectory
        for i in range(trajectory.num_waypoints):
            info["target_position"] = trajectory.positions[i]
            info["target_velocity"] = trajectory.velocities[i]
            self.step()
            time.sleep(self.control_dt)

        # Set final target for position hold
        info["target_position"] = trajectory.positions[-1].copy()
        info["target_velocity"] = np.zeros(len(info["actuator_ids"]))

        # Convergence-based settling (replaces fixed-step settling)
        converged = self._wait_for_convergence(arm_name)

        return converged

    def execute_right(self, trajectory: Trajectory) -> bool:
        """Execute trajectory on right arm."""
        return self.execute("right", trajectory)

    def execute_left(self, trajectory: Trajectory) -> bool:
        """Execute trajectory on left arm."""
        return self.execute("left", trajectory)

    def execute_base(self, side: str, trajectory: Trajectory) -> bool:
        """Execute base trajectory while maintaining arm positions.

        Args:
            side: "left" or "right"
            trajectory: Base trajectory (1 DOF)

        Returns:
            True if execution completed
        """
        if side not in self._bases:
            raise ValueError(f"Unknown base: {side}")

        info = self._bases[side]

        # Execute trajectory waypoints
        for i in range(trajectory.num_waypoints):
            info["target_height"] = trajectory.positions[i, 0]
            self.step()
            time.sleep(self.control_dt)

        # Set final target for position hold
        info["target_height"] = trajectory.positions[-1, 0]

        # Brief settling period for base (simpler than arm convergence)
        for _ in range(self.execution_config.base_settling_steps):
            self.step()

        return True

    def set_base_height(self, side: str, height: float) -> None:
        """Update base target height for position hold.

        Args:
            side: "left" or "right"
            height: Target height in meters
        """
        if side in self._bases:
            self._bases[side]["target_height"] = height

    def close_gripper(self, arm_name: str, steps: int | None = None) -> str | None:
        """Close gripper while maintaining all arm positions.

        Monitors for contact during closing and continues for a firm grip
        once contact is detected.

        Args:
            arm_name: "left" or "right"
            steps: Number of control steps for closing (default from config)

        Returns:
            Name of grasped object, or None
        """
        # Use config defaults
        cfg = self.gripper_config
        if steps is None:
            steps = cfg.close_steps

        if arm_name not in self._grippers:
            logger.warning(f"No gripper found for {arm_name} arm")
            return None

        gripper_info = self._grippers[arm_name]
        gripper = gripper_info["gripper"]

        from geodude.grasp_manager import detect_grasped_object

        # Always start from open position to ensure gripper actually closes
        # (using target_ctrl could be stale if gripper was already closed)
        start_ctrl = gripper_info["ctrl_open"]
        end_ctrl = gripper_info["ctrl_closed"]

        # First open the gripper to ensure clean start
        gripper_info["target_ctrl"] = start_ctrl
        for _ in range(cfg.pre_open_steps):
            self.step()
        if cfg.debug:
            logger.info(f"Gripper {arm_name}: opening before close, ctrl={start_ctrl}")
        contacts_detected = False
        grasped = None

        if cfg.debug:
            logger.info(f"Gripper {arm_name}: closing from {start_ctrl} to {end_ctrl}")

        for i in range(steps):
            t = (i + 1) / steps
            gripper_info["target_ctrl"] = start_ctrl + t * (end_ctrl - start_ctrl)
            self.step()

            # Check for contacts periodically
            # Use require_bilateral=False during closing - we just want to know if contact started
            if i % cfg.contact_check_interval == 0 and not contacts_detected:
                grasped = detect_grasped_object(
                    self.model,
                    self.data,
                    gripper.gripper_body_names,
                    gripper._candidate_objects,
                    require_bilateral=False,  # Less strict during closing
                    debug=cfg.debug,
                )
                if grasped:
                    contacts_detected = True
                    if cfg.debug:
                        logger.info(f"Gripper {arm_name}: contact detected at step {i}")

            time.sleep(self.control_dt * 0.5)  # Slower for gripper

        if cfg.debug:
            logger.info(f"Gripper {arm_name}: close complete, contacts_detected={contacts_detected}")

        # If contacts detected, continue closing for firm grip
        if contacts_detected:
            for _ in range(cfg.firm_grip_steps):
                self.step()
                time.sleep(self.control_dt * 0.5)

        # Check gripper position for logging
        gripper_pos = gripper.get_actual_position()
        if cfg.debug:
            logger.info(f"Gripper {arm_name}: actual_position={gripper_pos:.3f}")

        # Final grasp detection with bilateral contact for robust grasping
        # Do contact detection FIRST - a thin object like a can may allow gripper
        # to close to 0.95+ and still be successfully grasped
        grasped = detect_grasped_object(
            self.model,
            self.data,
            gripper.gripper_body_names,
            gripper._candidate_objects,
            require_bilateral=True,  # Require both fingers touching
            debug=cfg.debug,
        )

        if not grasped:
            # No bilateral contact - try relaxed detection as fallback
            grasped = detect_grasped_object(
                self.model,
                self.data,
                gripper.gripper_body_names,
                gripper._candidate_objects,
                require_bilateral=False,
                debug=cfg.debug,
            )
            if grasped:
                logger.warning(f"Gripper {arm_name}: only unilateral contact with {grasped} - grasp may be unstable")

        # Only consider "fully closed = lost grasp" if there are NO contacts at all
        # This catches cases where gripper completely missed or object slipped out
        if not grasped and gripper_pos > cfg.fully_closed_threshold:
            logger.warning(f"Gripper {arm_name}: fully closed (pos={gripper_pos:.3f}) with no contacts - missed or lost object")

        logger.debug(f"Gripper {arm_name}: final grasp detection = {grasped}")

        if grasped:
            gripper.grasp_manager.mark_grasped(grasped, arm_name)

        return grasped

    def open_gripper(self, arm_name: str, steps: int | None = None) -> None:
        """Open gripper while maintaining all arm positions.

        Args:
            arm_name: "left" or "right"
            steps: Number of control steps for opening (default from config)
        """
        if steps is None:
            steps = self.gripper_config.open_steps

        if arm_name not in self._grippers:
            return

        gripper_info = self._grippers[arm_name]
        gripper = gripper_info["gripper"]

        # Release any grasped objects first
        for obj in gripper.grasp_manager.get_grasped_by(arm_name):
            gripper.grasp_manager.mark_released(obj)

        # Gradually open gripper
        start_ctrl = gripper_info["target_ctrl"]
        end_ctrl = gripper_info["ctrl_open"]

        for i in range(steps):
            t = (i + 1) / steps
            gripper_info["target_ctrl"] = start_ctrl + t * (end_ctrl - start_ctrl)
            self.step()
            time.sleep(self.control_dt * 0.5)

    def get_executor(self, arm_name: str) -> "ArmPhysicsExecutor":
        """Get an executor interface for a specific arm.

        Returns an object with the standard Executor interface that internally
        uses this controller.
        """
        return ArmPhysicsExecutor(self, arm_name)


class ArmPhysicsExecutor:
    """Executor interface for a single arm, backed by RobotPhysicsController.

    This provides the standard Executor interface while ensuring all other
    actuators hold their positions during execution.
    """

    def __init__(self, controller: RobotPhysicsController, arm_name: str):
        self.controller = controller
        self.arm_name = arm_name

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory on this arm."""
        return self.controller.execute(self.arm_name, trajectory)
