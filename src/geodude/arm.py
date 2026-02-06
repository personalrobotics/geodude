"""Single arm control with planning and execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from geodude.collision import CollisionChecker, GraspAwareCollisionChecker
from geodude.config import ArmConfig
from geodude.executor import KinematicExecutor, PhysicsExecutor
from geodude.grasp_manager import GraspManager
from geodude.gripper import Gripper
from geodude.planning import PlanResult
from geodude.trajectory import Trajectory

# Optional pycbirrt imports for motion planning
try:
    from pycbirrt import CBiRRT, CBiRRTConfig, PlanningError

    PYCBIRRT_AVAILABLE = True
except ImportError:
    PYCBIRRT_AVAILABLE = False
    PlanningError = Exception  # Fallback for type hints

# Optional EAIK import for analytical IK
try:
    from pycbirrt.backends.eaik import EAIKSolver

    EAIK_AVAILABLE = True
except ImportError:
    EAIK_AVAILABLE = False

if TYPE_CHECKING:
    from geodude.robot import Geodude


class ArmRobotModel:
    """Adapter that wraps an Arm to provide the pycbirrt RobotModel interface.

    This allows an Arm to be used with the CBiRRT planner.
    """

    def __init__(self, arm: "Arm"):
        """Initialize the adapter.

        Args:
            arm: The Arm instance to wrap
        """
        self._arm = arm

    @property
    def dof(self) -> int:
        """Number of degrees of freedom."""
        return self._arm.dof

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint limits as (lower, upper) bounds arrays."""
        return self._arm.get_joint_limits()

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector pose from joint configuration.

        Args:
            q: Joint configuration array of shape (dof,)

        Returns:
            4x4 homogeneous transform of end-effector in world frame
        """
        # Use temporary MjData to avoid visible teleportation during planning
        # This method is called hundreds of times by the planner!
        return self._arm._get_ee_pose_at_config(q)


class ContextRobotModel:
    """Robot model adapter using a PlanningContext's data for thread-safe FK.

    Unlike ArmRobotModel which uses the shared Arm data, this uses an isolated
    MjData from a PlanningContext, making it safe for parallel planning.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: list[int],
        ee_site_id: int,
        joint_limits: tuple[np.ndarray, np.ndarray],
    ):
        """Initialize the context-based robot model.

        Args:
            model: MuJoCo model (shared, read-only)
            data: MjData for this context (private copy)
            joint_qpos_indices: Indices into qpos for arm joints
            ee_site_id: Site ID for end-effector
            joint_limits: (lower, upper) joint limit arrays
        """
        self._model = model
        self._data = data
        self._joint_qpos_indices = joint_qpos_indices
        self._ee_site_id = ee_site_id
        self._joint_limits = joint_limits

    @property
    def dof(self) -> int:
        """Number of degrees of freedom."""
        return len(self._joint_qpos_indices)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint limits as (lower, upper) bounds arrays."""
        return self._joint_limits

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector pose from joint configuration.

        Uses this context's MjData (thread-safe).

        Args:
            q: Joint configuration array of shape (dof,)

        Returns:
            4x4 homogeneous transform of end-effector in world frame
        """
        # Set arm joints in our private data
        for i, qpos_idx in enumerate(self._joint_qpos_indices):
            self._data.qpos[qpos_idx] = q[i]

        # Compute FK
        mujoco.mj_forward(self._model, self._data)

        # Read EE pose
        pos = self._data.site_xpos[self._ee_site_id]
        rot_mat = self._data.site_xmat[self._ee_site_id].reshape(3, 3)

        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = pos
        return transform


class ArmIKSolver:
    """Adapter that wraps an Arm's IK to provide the pycbirrt IKSolver interface.

    This allows an Arm's inverse_kinematics method to be used with CBiRRT.
    """

    def __init__(self, arm: "Arm"):
        """Initialize the adapter.

        Args:
            arm: The Arm instance to wrap
        """
        self._arm = arm

    def solve(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Solve IK for a single end-effector pose (raw, unvalidated).

        Args:
            pose: 4x4 homogeneous transform of desired end-effector pose
            q_init: Ignored (EAIK finds all solutions analytically)

        Returns:
            List of joint configurations (may include invalid ones)
        """
        return self._arm.inverse_kinematics(pose, validate=False, sort_by_distance=False)

    def solve_valid(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Solve IK and return only valid solutions.

        Args:
            pose: 4x4 homogeneous transform of desired end-effector pose
            q_init: Ignored (EAIK finds all solutions analytically)

        Returns:
            List of valid joint configurations (may be empty)
        """
        return self._arm.inverse_kinematics(pose, validate=True, sort_by_distance=False)


class SimpleIKSolver:
    """IK solver with pre-computed base pose. No MjData access during solve.

    This is the preferred IK solver for parallel planning. It computes all
    transforms at construction time, so solve() is pure math with no MuJoCo calls.
    """

    def __init__(
        self,
        base_pose: np.ndarray,
        base_rotation: np.ndarray,
        ee_offset: np.ndarray,
        joint_limits: tuple[np.ndarray, np.ndarray],
    ):
        """Initialize the IK solver with pre-computed transforms.

        Args:
            base_pose: 4x4 transform of arm base in world frame
            base_rotation: 4x4 rotation from EAIK frame to MuJoCo base frame
            ee_offset: 4x4 transform from EAIK EE to MuJoCo EE site
            joint_limits: (lower, upper) joint limit arrays
        """
        self._base_pose = base_pose
        self._base_rotation = base_rotation
        self._ee_offset = ee_offset
        # Create EAIK solver without collision checker (planner will validate)
        self._eaik = EAIKSolver.for_ur5e(joint_limits=joint_limits, collision_checker=None)

    def solve(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Solve IK for a single end-effector pose (raw, unvalidated).

        Pure math - no MjData access. Thread-safe.

        Args:
            pose: 4x4 homogeneous transform of desired EE pose in world frame
            q_init: Ignored (EAIK finds all solutions analytically)

        Returns:
            List of joint configurations (may include invalid ones)
        """
        # Transform from world frame to MuJoCo base frame
        T_base = np.linalg.inv(self._base_pose) @ pose
        # Transform from MuJoCo base frame to EAIK frame
        T_eaik = np.linalg.inv(self._base_rotation) @ T_base @ np.linalg.inv(self._ee_offset)
        solutions = self._eaik.solve(T_eaik)
        return solutions if solutions else []

    def solve_valid(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Solve IK and return solutions (planner validates).

        Note: Returns same as solve() because validation is done by the
        CBiRRT planner using its collision checker, not here.

        Args:
            pose: 4x4 homogeneous transform of desired EE pose in world frame
            q_init: Ignored (EAIK finds all solutions analytically)

        Returns:
            List of joint configurations (planner will validate them)
        """
        return self.solve(pose, q_init)


class Arm:
    """Controls a single robot arm with planning and execution.

    Provides:
    - Forward/inverse kinematics
    - Motion planning via pycbirrt
    - Trajectory execution
    - Gripper control
    - High-level pick/place tasks
    """

    def __init__(
        self,
        robot: "Geodude",
        config: ArmConfig,
        grasp_manager: GraspManager,
    ):
        """Initialize arm controller.

        Args:
            robot: Parent Geodude robot instance
            config: Arm configuration
            grasp_manager: GraspManager for tracking grasp state
        """
        self.robot = robot
        self.config = config
        self.grasp_manager = grasp_manager
        self.model = robot.model
        self.data = robot.data

        # Get joint indices
        self.joint_ids = []
        self.joint_qpos_indices = []
        for name in config.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found in model")
            self.joint_ids.append(jid)
            self.joint_qpos_indices.append(self.model.jnt_qposadr[jid])

        # Get actuator IDs for position control
        # Find actuators that control these joints
        self.actuator_ids = []
        for jid in self.joint_ids:
            # Find actuator that controls this joint
            actuator_id = -1
            for act_id in range(self.model.nu):
                # Check if this actuator's transmission targets our joint
                if self.model.actuator_trnid[act_id, 0] == jid:
                    actuator_id = act_id
                    break
            if actuator_id == -1:
                raise ValueError(f"No actuator found for joint ID {jid}")
            self.actuator_ids.append(actuator_id)

        # Get EE site
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, config.ee_site
        )
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{config.ee_site}' not found in model")

        # Get the arm side for gripper and transforms
        # The UR5e base body is named "{side}_ur5e/base" where side is "left" or "right"
        side = "left" if "left" in config.name else "right"

        # Initialize gripper
        self.gripper = Gripper(
            self.model,
            self.data,
            side,  # Use side ("left"/"right"), not config.name ("left_arm"/"right_arm")
            config.gripper_actuator,
            config.gripper_bodies,
            grasp_manager,
        )
        base_body_name = f"{side}_ur5e/base"
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
        )
        if self._base_body_id == -1:
            # Fallback: try older naming convention
            base_body_name = f"{config.name}_ur5e/base"
            self._base_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
            )
        if self._base_body_id == -1:
            raise ValueError(f"Could not find base body for arm {config.name}. Tried: {side}_ur5e/base, {config.name}_ur5e/base")

        # Planner, IK solver, and executor will be set up lazily
        self._planner = None
        self._collision_checker = None
        self._ik_solver = None
        self._executor = None
        self._ee_offset = None  # Cached EE offset transform
        self._cached_base_rotation = None  # Cached base rotation

        # Eagerly compute IK frame constants (if EAIK available)
        # These are mounting constants that never change, so compute once at init
        # while still single-threaded
        if EAIK_AVAILABLE:
            self._cached_base_rotation = self._get_base_rotation()
            _ = self._get_ee_offset()  # Sets self._ee_offset

        # F/T sensor tare offsets (software zeroing)
        self._ft_tare_force = np.zeros(3)
        self._ft_tare_torque = np.zeros(3)

        # Cache F/T sensor IDs for efficient lookup
        # Use side ("left"/"right") not full name ("left_arm"/"right_arm")
        side = "left" if "left" in config.name else "right"
        force_sensor_name = f"{side}_ur5e/ft_sensor_force"
        torque_sensor_name = f"{side}_ur5e/ft_sensor_torque"
        self._ft_force_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, force_sensor_name
        )
        self._ft_torque_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, torque_sensor_name
        )

    @property
    def name(self) -> str:
        """Arm name (e.g., 'left_arm' or 'right_arm')."""
        return self.config.name

    @property
    def side(self) -> str:
        """Arm side ('left' or 'right'), extracted from name."""
        return "left" if "left" in self.config.name else "right"

    @property
    def dof(self) -> int:
        """Degrees of freedom."""
        return len(self.joint_ids)

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        return np.array([self.data.qpos[i] for i in self.joint_qpos_indices])

    def set_joint_positions(self, q: np.ndarray) -> None:
        """Set joint positions directly (no physics step)."""
        for i, qpos_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[qpos_idx] = q[i]
        mujoco.mj_forward(self.model, self.data)

    def get_ee_pose(self) -> np.ndarray:
        """Get current end-effector pose as 4x4 transform."""
        mujoco.mj_forward(self.model, self.data)

        pos = self.data.site_xpos[self.ee_site_id]
        rot_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)

        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = pos
        return transform

    def _get_ee_pose_at_config(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector pose at a specific configuration without modifying robot state.

        Creates a temporary MjData copy to avoid corrupting the shared robot state.
        This is essential during planning to avoid visible teleportation artifacts.

        Args:
            q: Joint configuration to compute FK at

        Returns:
            4x4 homogeneous transform of end-effector pose
        """
        # Create temporary data for FK computation
        temp_data = mujoco.MjData(self.model)

        # Copy current state to temp data
        temp_data.qpos[:] = self.data.qpos
        temp_data.qvel[:] = self.data.qvel

        # Set arm joints to target configuration in temp data
        for i, qpos_idx in enumerate(self.joint_qpos_indices):
            temp_data.qpos[qpos_idx] = q[i]

        # Compute FK in temp data (doesn't affect shared state)
        mujoco.mj_forward(self.model, temp_data)

        # Read EE pose from temp data
        pos = temp_data.site_xpos[self.ee_site_id]
        rot_mat = temp_data.site_xmat[self.ee_site_id].reshape(3, 3)

        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = pos
        return transform

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Get joint position limits."""
        lower = np.array([self.model.jnt_range[jid, 0] for jid in self.joint_ids])
        upper = np.array([self.model.jnt_range[jid, 1] for jid in self.joint_ids])
        return lower, upper

    # UR5e F/T sensor specifications (from datasheet)
    _FT_FORCE_RANGE = 50.0  # ±50 N
    _FT_TORQUE_RANGE = 10.0  # ±10 Nm

    def get_ft_sensor(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get force/torque sensor reading at the tool flange.

        Returns the F/T sensor values in the tool0 frame (Z+ out of flange,
        X+ left, Y+ up), with any tare offset subtracted. Values are clamped
        to the UR5e sensor range (±50N force, ±10Nm torque) to simulate
        hardware saturation. Noise matching real sensor precision (±3.5N,
        ±0.2Nm) is applied in the MuJoCo model.

        Note: Only meaningful with physics execution (mj_step). Returns None
        if physics has not been stepped yet (data.time == 0), since kinematic
        mode (mj_forward) produces artifacts, not real forces.

        Returns:
            Tuple of (force, torque) where:
                - force: np.ndarray of shape (3,) with [Fx, Fy, Fz] in Newtons
                - torque: np.ndarray of shape (3,) with [Tx, Ty, Tz] in Nm
            Returns None if physics simulation has not been run yet.

        Raises:
            RuntimeError: If F/T sensors are not available in the model
        """
        if self._ft_force_sensor_id == -1 or self._ft_torque_sensor_id == -1:
            raise RuntimeError(
                f"F/T sensors not found for arm '{self.config.name}'. "
                "Make sure you are using a geodude_assets version >= 0.1.1 "
                "that includes F/T sensor support."
            )

        # Check if physics has been stepped (time advances only with mj_step)
        if self.data.time == 0:
            return None

        # Read raw sensor values from sensordata (includes MuJoCo noise)
        force_adr = self.model.sensor_adr[self._ft_force_sensor_id]
        torque_adr = self.model.sensor_adr[self._ft_torque_sensor_id]

        raw_force = self.data.sensordata[force_adr : force_adr + 3].copy()
        raw_torque = self.data.sensordata[torque_adr : torque_adr + 3].copy()

        # Apply tare offset
        force = raw_force - self._ft_tare_force
        torque = raw_torque - self._ft_tare_torque

        # Clamp to sensor range (simulates hardware saturation)
        force = np.clip(force, -self._FT_FORCE_RANGE, self._FT_FORCE_RANGE)
        torque = np.clip(torque, -self._FT_TORQUE_RANGE, self._FT_TORQUE_RANGE)

        return force, torque

    def tare_ft_sensor(self) -> None:
        """Zero the F/T sensor by storing current reading as offset.

        After calling this method, subsequent get_ft_sensor() calls will
        return values relative to the current sensor state. This is useful
        for compensating gravity loads or zeroing drift.

        Typical usage:
            # Tare with gripper empty to compensate gripper weight
            robot.right_arm.tare_ft_sensor()

            # Now readings reflect only external forces
            force, torque = robot.right_arm.get_ft_sensor()

        Raises:
            RuntimeError: If F/T sensors are not available in the model
        """
        if self._ft_force_sensor_id == -1 or self._ft_torque_sensor_id == -1:
            raise RuntimeError(
                f"F/T sensors not found for arm '{self.config.name}'. "
                "Make sure you are using a geodude_assets version >= 0.1.1 "
                "that includes F/T sensor support."
            )

        # Read current raw values and store as tare offset
        force_adr = self.model.sensor_adr[self._ft_force_sensor_id]
        torque_adr = self.model.sensor_adr[self._ft_torque_sensor_id]

        self._ft_tare_force = self.data.sensordata[force_adr : force_adr + 3].copy()
        self._ft_tare_torque = self.data.sensordata[torque_adr : torque_adr + 3].copy()

    def _get_collision_checker(self) -> GraspAwareCollisionChecker:
        """Get or create the collision checker."""
        if self._collision_checker is None:
            self._collision_checker = GraspAwareCollisionChecker(
                self.model,
                self.data,
                self.config.joint_names,
                self.grasp_manager,
            )
        return self._collision_checker

    def _get_planner(self, config: "CBiRRTConfig | None" = None) -> "CBiRRT":
        """Get or create the CBiRRT motion planner.

        Args:
            config: Optional planner configuration. If provided, creates a new
                    planner with this config. If None, uses cached planner with
                    default config.

        Returns:
            CBiRRT planner instance

        Raises:
            ImportError: If pycbirrt is not installed
        """
        if not PYCBIRRT_AVAILABLE:
            raise ImportError(
                "pycbirrt not available. Install with: pip install pycbirrt[eaik]\n"
                "Or: uv add pycbirrt[eaik]"
            )

        # Create new planner if config provided or no cached planner
        if config is not None or self._planner is None:
            robot_model = ArmRobotModel(self)
            ik_solver = ArmIKSolver(self)
            collision_checker = self._get_collision_checker()

            defaults = self.config.planning_defaults
            planner_config = config or CBiRRTConfig(
                max_iterations=defaults.max_iterations,
                step_size=defaults.step_size,
                goal_bias=defaults.goal_bias,
                smoothing_iterations=defaults.smoothing_iterations,
            )

            planner = CBiRRT(
                robot=robot_model,
                ik_solver=ik_solver,
                collision_checker=collision_checker,
                config=planner_config,
            )

            if config is None:
                self._planner = planner
            return planner

        return self._planner

    def create_planner(
        self,
        config: "CBiRRTConfig | None" = None,
        *,
        base_joint_name: str | None = None,
        base_height: float | None = None,
    ) -> "CBiRRT":
        """Create an isolated CBiRRT planner for thread-safe parallel planning.

        Each call creates a planner with its own MjData copy. Safe for concurrent
        use - call from multiple threads for different arms or different goals.

        Args:
            config: Optional planner configuration. If None, uses default config.
            base_joint_name: Optional joint name for setting base height (e.g., "right_vention/joint")
            base_height: Optional height to set the base to (requires base_joint_name)

        Returns:
            CBiRRT planner instance with isolated state

        Raises:
            ImportError: If pycbirrt is not installed

        Example:
            # Plan both arms in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                left_future = executor.submit(
                    lambda: robot.left_arm.create_planner().plan(start, goal=left_goal)
                )
                right_future = executor.submit(
                    lambda: robot.right_arm.create_planner().plan(start, goal=right_goal)
                )
                left_path = left_future.result()
                right_path = right_future.result()

            # Plan at different base heights
            planner = arm.create_planner(base_joint_name="right_vention/joint", base_height=0.2)
            path = planner.plan(start, goal_tsrs=[tsr])
        """
        if not PYCBIRRT_AVAILABLE:
            raise ImportError(
                "pycbirrt not available. Install with: pip install pycbirrt[eaik]\n"
                "Or: uv add pycbirrt[eaik]"
            )

        # 1. Create isolated MjData copy
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.data.qpos

        # 2. Optionally set base height
        if base_height is not None and base_joint_name is not None:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, base_joint_name)
            if joint_id != -1:
                data.qpos[self.model.jnt_qposadr[joint_id]] = base_height

        # 3. Forward kinematics
        mujoco.mj_forward(self.model, data)

        # 4. Get base pose from this data (computed ONCE, not per IK call)
        base_pose = self._get_body_pose_from_data(data, self._base_body_id)

        # 5. Snapshot grasp state (just data, not a class)
        grasped = frozenset(self.grasp_manager.grasped.items())
        attachments = {k: (b, t.copy()) for k, (b, t) in self.grasp_manager._attachments.items()}

        # 6. Create components
        ik_solver = SimpleIKSolver(
            base_pose,
            self._cached_base_rotation,
            self._ee_offset,
            self.get_joint_limits(),
        )
        collision_checker = CollisionChecker(
            self.model,
            data,
            self.config.joint_names,
            grasped,
            attachments,
        )
        robot_model = ContextRobotModel(
            model=self.model,
            data=data,
            joint_qpos_indices=self.joint_qpos_indices,
            ee_site_id=self.ee_site_id,
            joint_limits=self.get_joint_limits(),
        )

        # 7. Create and return planner
        defaults = self.config.planning_defaults
        planner_config = config or CBiRRTConfig(
            max_iterations=defaults.max_iterations,
            step_size=defaults.step_size,
            goal_bias=defaults.goal_bias,
            smoothing_iterations=defaults.smoothing_iterations,
        )

        return CBiRRT(
            robot=robot_model,
            ik_solver=ik_solver,
            collision_checker=collision_checker,
            config=planner_config,
        )

    def _get_body_pose_from_data(self, data: mujoco.MjData, body_id: int) -> np.ndarray:
        """Get body pose from a specific MjData instance.

        Args:
            data: MjData to read from
            body_id: Body ID to get pose for

        Returns:
            4x4 homogeneous transform
        """
        if body_id == -1:
            return np.eye(4)

        pos = data.xpos[body_id]
        rot_mat = data.xmat[body_id].reshape(3, 3)

        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = pos
        return transform

    # Internal properties for parallel planning module
    @property
    def _model(self) -> mujoco.MjModel:
        """MuJoCo model (for parallel module access)."""
        return self.model

    @property
    def _data(self) -> mujoco.MjData:
        """MuJoCo data (for parallel module access)."""
        return self.data

    @property
    def _config(self) -> ArmConfig:
        """Arm configuration (for parallel module access)."""
        return self.config

    @property
    def _grasp_manager(self) -> GraspManager:
        """Grasp manager (for parallel module access)."""
        return self.grasp_manager

    def get_base_pose(self) -> np.ndarray:
        """Get the arm base pose in world frame as 4x4 transform.

        Returns:
            4x4 homogeneous transformation matrix from arm base to world
        """
        mujoco.mj_forward(self.model, self.data)

        if self._base_body_id == -1:
            # No base body found, assume identity (base at world origin)
            return np.eye(4)

        pos = self.data.xpos[self._base_body_id]
        rot_mat = self.data.xmat[self._base_body_id].reshape(3, 3)

        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = pos
        return transform

    def _get_ik_solver(self) -> "EAIKSolver":
        """Get or create the EAIK analytical IK solver.

        Returns:
            EAIKSolver instance configured for UR5e

        Raises:
            ImportError: If EAIK is not installed
        """
        if not EAIK_AVAILABLE:
            raise ImportError(
                "EAIK not available. Install with: pip install eaik\n"
                "Or: uv add eaik"
            )

        if self._ik_solver is None:
            self._ik_solver = EAIKSolver.for_ur5e(
                joint_limits=self.get_joint_limits(),
                collision_checker=self._get_collision_checker(),
            )
        return self._ik_solver

    def _get_executor(
        self,
        viewer=None,
        executor_type: str = "physics",
    ) -> KinematicExecutor | PhysicsExecutor:
        """Get or create the trajectory executor.

        Args:
            viewer: Optional MuJoCo viewer for visualization during execution
            executor_type: Type of executor to use:
                - "physics" (default): Physics simulation with velocity feedforward
                - "kinematic": Perfect tracking without physics (direct qpos setting)

        Returns:
            Executor instance configured for this arm
        """
        if executor_type == "kinematic":
            return KinematicExecutor(
                model=self.model,
                data=self.data,
                joint_qpos_indices=self.joint_qpos_indices,
                viewer=viewer,
            )
        elif executor_type == "physics":
            return PhysicsExecutor(
                model=self.model,
                data=self.data,
                joint_qpos_indices=self.joint_qpos_indices,
                actuator_ids=self.actuator_ids,
                viewer=viewer,
            )
        else:
            raise ValueError(
                f"Unknown executor_type: {executor_type}. "
                "Must be 'kinematic' or 'physics'"
            )

    def _get_base_rotation(self) -> np.ndarray:
        """Get the rotation from EAIK's base frame to MuJoCo's base body frame.

        EAIK uses the standard UR5e DH convention where the base is at identity.
        The MuJoCo model may have the arm mounted at a different orientation.

        This computes R such that:
            T_mjbase_mujoco = R @ T_eaikbase_eaik

        by comparing EAIK FK with MuJoCo FK at a reference configuration.

        Returns:
            4x4 rotation matrix R
        """
        if self._cached_base_rotation is not None:
            return self._cached_base_rotation

        # Compare FK at reference configuration to compute R
        q_ref = np.zeros(6)

        # Get EAIK FK at q_ref (returns pose in EAIK's base frame)
        ik_solver = self._get_ik_solver()
        T_eaikbase_eaik = ik_solver.forward_kinematics(q_ref)

        # Get MuJoCo FK at q_ref and convert to base body frame
        old_q = self.get_joint_positions()
        self.set_joint_positions(q_ref)
        T_world_mujoco = self.get_ee_pose()
        self.set_joint_positions(old_q)

        T_world_base = self.get_base_pose()
        T_mjbase_mujoco = np.linalg.inv(T_world_base) @ T_world_mujoco

        # Solve for R: T_mjbase_mujoco = R @ T_eaikbase_eaik @ T_eaik_mujoco
        # Since T_eaik_mujoco is nearly identity (computed in _get_ee_offset),
        # we can approximate: R ≈ T_mjbase_mujoco @ inv(T_eaikbase_eaik)
        # But this won't give a pure rotation. Instead, extract rotation only.
        #
        # For rotation part: R_mjbase_mujoco = R_rot @ R_eaikbase_eaik
        # So: R_rot = R_mjbase_mujoco @ inv(R_eaikbase_eaik)
        R_mjbase = T_mjbase_mujoco[:3, :3]
        R_eaik = T_eaikbase_eaik[:3, :3]
        R_rot = R_mjbase @ np.linalg.inv(R_eaik)

        R = np.eye(4)
        R[:3, :3] = R_rot
        self._cached_base_rotation = R
        return R

    def _get_ee_offset(self) -> np.ndarray:
        """Get the transform from EAIK EE frame to MuJoCo EE site.

        EAIK uses the standard UR5e DH convention which places the EE
        at the tool flange. MuJoCo's gripper_attachment_site may have
        an additional offset from there.

        This method computes the fixed offset by comparing EAIK FK
        with MuJoCo FK at a known configuration. The result is cached.

        Returns:
            4x4 transform T_eaik_mujoco such that:
            T_mjbase_mujoco = R @ T_eaikbase_eaik @ T_eaik_mujoco
            where R is the base rotation and T_mjbase_mujoco is the MuJoCo
            EE pose in MuJoCo's base body frame.
        """
        if self._ee_offset is not None:
            return self._ee_offset

        # Use a known configuration to compute the offset
        q_ref = np.zeros(6)

        # Get EAIK FK (in EAIK's base frame) - access solver directly to avoid recursion
        if not EAIK_AVAILABLE:
            raise ImportError(
                "EAIK not available. Install with: pip install eaik\n"
                "Or: uv add eaik"
            )
        if self._ik_solver is None:
            self._ik_solver = EAIKSolver.for_ur5e(
                joint_limits=self.get_joint_limits(),
                collision_checker=self._get_collision_checker(),
            )

        # EAIK FK gives pose in EAIK's base frame (standard UR5e orientation)
        T_eaikbase_eaik = self._ik_solver.forward_kinematics(q_ref)

        # Convert EAIK result to MuJoCo base frame
        R = self._get_base_rotation()
        T_mjbase_eaik = R @ T_eaikbase_eaik

        # Get MuJoCo FK and transform to MuJoCo base frame
        old_q = self.get_joint_positions()
        self.set_joint_positions(q_ref)
        T_world_mujoco = self.get_ee_pose()
        self.set_joint_positions(old_q)

        # Convert MuJoCo EE pose to MuJoCo base frame
        T_world_base = self.get_base_pose()
        T_mjbase_mujoco = np.linalg.inv(T_world_base) @ T_world_mujoco

        # Compute offset: T_eaik_mujoco = inv(T_mjbase_eaik) @ T_mjbase_mujoco
        self._ee_offset = np.linalg.inv(T_mjbase_eaik) @ T_mjbase_mujoco

        return self._ee_offset

    def inverse_kinematics(
        self,
        pose: np.ndarray,
        validate: bool = True,
        sort_by_distance: bool = True,
    ) -> list[np.ndarray]:
        """Compute inverse kinematics for an end-effector pose.

        Uses EAIK analytical IK solver which returns all kinematic solutions
        (up to 8 for a 6-DOF manipulator like UR5e).

        The target pose should be for the MuJoCo EE site (gripper_attachment_site).
        This method automatically handles the coordinate transformation between
        the MuJoCo EE site and EAIK's DH-convention EE frame.

        Args:
            pose: Target end-effector pose as 4x4 homogeneous transform in WORLD frame
                  (for the gripper_attachment_site, not the DH-convention EE)
            validate: If True, filter solutions by joint limits and collisions
            sort_by_distance: If True, sort solutions by distance from current config

        Returns:
            List of valid joint configurations (may be empty if no solution exists)
        """
        ik_solver = self._get_ik_solver()

        # Save current configuration for sorting (collision checker modifies state)
        q_current = self.get_joint_positions().copy()

        # Transform pose from world frame to MuJoCo base frame
        T_world_base = self.get_base_pose()
        T_mjbase_mujoco = np.linalg.inv(T_world_base) @ pose

        # Account for the offset between MuJoCo EE site and EAIK EE frame
        # T_mjbase_mujoco = R @ T_eaikbase_eaik @ T_eaik_mujoco
        # where R is the base rotation
        # => T_eaikbase_eaik = inv(R) @ T_mjbase_mujoco @ inv(T_eaik_mujoco)
        R = self._get_base_rotation()
        R_inv = np.linalg.inv(R)
        T_eaik_mujoco = self._get_ee_offset()
        T_eaikbase_eaik = R_inv @ T_mjbase_mujoco @ np.linalg.inv(T_eaik_mujoco)

        # Solve IK (EAIK expects pose in its base frame)
        if validate:
            solutions = ik_solver.solve_valid(T_eaikbase_eaik)
        else:
            solutions = ik_solver.solve(T_eaikbase_eaik)

        if not solutions:
            return []

        # Optionally sort by distance from original configuration
        if sort_by_distance and solutions:
            solutions = sorted(solutions, key=lambda q: np.linalg.norm(q - q_current))

        return solutions

    def forward_kinematics_eaik(self, q: np.ndarray) -> np.ndarray:
        """Compute forward kinematics using EAIK (in world frame).

        This uses EAIK's FK which may be faster than MuJoCo for batch operations.
        The result is transformed to world frame and accounts for the offset
        between EAIK's DH-convention EE and MuJoCo's gripper_attachment_site.

        Args:
            q: Joint configuration

        Returns:
            4x4 homogeneous transform of gripper_attachment_site in world frame
        """
        ik_solver = self._get_ik_solver()

        # EAIK FK returns pose in EAIK's base frame (DH convention EE)
        T_eaikbase_eaik = ik_solver.forward_kinematics(q)

        # Convert to MuJoCo base frame and apply EE offset
        # T_mjbase_mujoco = R @ T_eaikbase_eaik @ T_eaik_mujoco
        R = self._get_base_rotation()
        T_eaik_mujoco = self._get_ee_offset()
        T_mjbase_mujoco = R @ T_eaikbase_eaik @ T_eaik_mujoco

        # Transform to world frame
        T_world_base = self.get_base_pose()
        return T_world_base @ T_mjbase_mujoco

    def go_to(self, target: str | np.ndarray, speed: float = 1.0) -> bool:
        """Move to a named configuration or joint positions.

        If an execution context is active (inside `with robot.sim()`), plans
        and executes a collision-free trajectory. Otherwise, sets positions
        directly (kinematic mode, useful for initialization).

        Args:
            target: Named configuration (e.g., 'home') or joint positions array
            speed: Execution speed multiplier (not yet implemented)

        Returns:
            True if successful
        """
        if isinstance(target, str):
            # Look up named configuration
            if target not in self.robot.named_poses:
                raise ValueError(f"Unknown named pose: {target}")
            q_target = np.array(self.robot.named_poses[target][self.side])
        else:
            q_target = np.asarray(target)

        # Check if we have an active execution context
        ctx = self.robot._active_context
        if ctx is not None:
            # Plan and execute trajectory
            trajectory = self.plan_to(q_target)
            if trajectory is None:
                return False
            return ctx.execute(trajectory)
        else:
            # No context - set positions directly (kinematic initialization)
            self.set_joint_positions(q_target)
            return True

    def plan_to_configuration(
        self,
        q_goal: np.ndarray,
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> list[np.ndarray] | None:
        """Plan a collision-free path to a goal configuration using CBiRRT.

        Args:
            q_goal: Goal joint configuration
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility

        Returns:
            List of waypoints, or None if planning failed
        """
        if not PYCBIRRT_AVAILABLE:
            raise ImportError(
                "pycbirrt not available. Install with: pip install pycbirrt[eaik]\n"
                "Or: uv add pycbirrt[eaik]"
            )

        # Create fork to preserve state (planning corrupts self.data)
        state_snapshot = self.robot.env.fork()

        q_start = self.get_joint_positions()

        defaults = self.config.planning_defaults
        config = CBiRRTConfig(
            timeout=timeout,
            max_iterations=defaults.max_iterations,
            step_size=defaults.step_size,
            goal_bias=defaults.goal_bias,
            smoothing_iterations=defaults.smoothing_iterations,
        )
        planner = self._get_planner(config)

        try:
            # Pass goal configuration directly - no FK/IK round-trip needed
            path = planner.plan(start=q_start, goal=q_goal, seed=seed)
        except (ValueError, PlanningError):
            # Invalid configuration (collision or constraint violation)
            path = None
        finally:
            # Restore robot state from snapshot (planning corrupts state)
            self.robot.env.sync_from(state_snapshot)

        return path

    def plan_to_tsrs(
        self,
        tsrs: list,
        constraint_tsrs: list | None = None,
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> list[np.ndarray] | None:
        """Plan a path to reach any of the given TSRs.

        Uses CBiRRT with TSR constraints to find a collision-free path from
        the current configuration to a pose that satisfies any of the goal TSRs.

        Args:
            tsrs: List of TSR objects defining valid goal poses (union - any one)
            constraint_tsrs: Optional TSRs that constrain the entire path
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility

        Returns:
            List of waypoints, or None if planning failed
        """
        if not PYCBIRRT_AVAILABLE:
            raise ImportError(
                "pycbirrt not available. Install with: pip install pycbirrt[eaik]\n"
                "Or: uv add pycbirrt[eaik]"
            )

        # Create fork to preserve state (planning corrupts self.data)
        state_snapshot = self.robot.env.fork()

        q_start = self.get_joint_positions()

        defaults = self.config.planning_defaults
        config = CBiRRTConfig(
            timeout=timeout,
            max_iterations=defaults.max_iterations,
            step_size=defaults.step_size,
            goal_bias=defaults.goal_bias,
            smoothing_iterations=defaults.smoothing_iterations,
        )
        planner = self._get_planner(config)

        try:
            path = planner.plan(
                q_start,
                goal_tsrs=tsrs,
                constraint_tsrs=constraint_tsrs,
                seed=seed,
            )
        except (ValueError, PlanningError):
            # No valid configurations found (IK failed or all in collision)
            path = None
        finally:
            # Restore robot state from snapshot (planning corrupts state)
            self.robot.env.sync_from(state_snapshot)

        return path

    def execute(
        self,
        path: list[np.ndarray] | Trajectory,
        executor: KinematicExecutor | PhysicsExecutor | None = None,
        viewer=None,
        executor_type: str = "physics",
    ) -> bool:
        """Execute a planned path or pre-computed trajectory.

        Converts geometric paths to time-optimal trajectories using TOPP-RA
        and executes them respecting velocity/acceleration limits.

        Args:
            path: Either a list of waypoints (joint configurations) or
                  a pre-computed Trajectory object
            executor: Optional executor to use. If None, creates one based on executor_type
            viewer: Optional MuJoCo viewer to sync during execution for smooth visualization
            executor_type: Type of executor to create if executor=None:
                - "physics" (default): Physics simulation with velocity feedforward
                - "kinematic": Perfect tracking without physics

        Returns:
            True if execution completed successfully

        Raises:
            RuntimeError: If trajectory generation fails (path violates limits)
        """
        # Convert path to trajectory if needed
        if isinstance(path, list):
            trajectory = Trajectory.from_path(
                path,
                vel_limits=self.config.kinematic_limits.velocity,
                acc_limits=self.config.kinematic_limits.acceleration,
            )
        else:
            trajectory = path

        # Get or use provided executor
        if executor is None:
            executor = self._get_executor(viewer=viewer, executor_type=executor_type)

        # Execute trajectory
        return executor.execute(trajectory)

    def plan_to(
        self,
        goal: np.ndarray | list,
        *,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> Trajectory | PlanResult | None:
        """Plan to a goal (configuration, pose, or TSR).

        Unified planning method that dispatches to the appropriate planner
        based on goal type:
        - np.ndarray with shape (n,) where n == dof: configuration
        - np.ndarray with shape (4, 4): end-effector pose
        - TSR object: task space region
        - list of any of the above: multiple goals (planner picks one)

        Args:
            goal: Target configuration, pose, TSR, or list of goals
            base_heights: Optional list of base heights to search (tried in order).
                         If provided, returns PlanResult with both trajectories.
            strategy: Planning strategy:
                     - "first": Return first successful plan (fastest)
                     - "best": Try all heights, return shortest path
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility

        Returns:
            - Trajectory or PlanResult (with base_heights) if planning succeeded
            - None if planning failed
        """
        # Detect goal type and dispatch
        if hasattr(goal, "sample"):
            # TSR object (has sample method)
            return self.plan_to_tsr(
                goal,
                base_heights=base_heights,
                strategy=strategy,
                timeout=timeout,
                seed=seed,
            )
        elif isinstance(goal, list):
            # List of goals - check first element type
            if not goal:
                return None
            first = goal[0]
            if hasattr(first, "sample"):
                # List of TSRs
                return self.plan_to_tsr(
                    goal,
                    base_heights=base_heights,
                    strategy=strategy,
                    timeout=timeout,
                    seed=seed,
                )
            elif isinstance(first, np.ndarray):
                if first.shape == (4, 4):
                    # List of poses
                    return self.plan_to_pose(
                        goal,
                        base_heights=base_heights,
                        strategy=strategy,
                        timeout=timeout,
                        seed=seed,
                    )
                else:
                    # List of configurations
                    return self.plan_to_configurations(
                        goal,
                        base_heights=base_heights,
                        strategy=strategy,
                        timeout=timeout,
                        seed=seed,
                    )
            else:
                raise ValueError(f"Unknown goal type in list: {type(first)}")
        elif isinstance(goal, np.ndarray):
            if goal.shape == (4, 4):
                # Single pose
                return self.plan_to_pose(
                    goal,
                    base_heights=base_heights,
                    strategy=strategy,
                    timeout=timeout,
                    seed=seed,
                )
            elif goal.ndim == 1 and len(goal) == self.dof:
                # Single configuration
                return self.plan_to_configurations(
                    [goal],
                    base_heights=base_heights,
                    strategy=strategy,
                    timeout=timeout,
                    seed=seed,
                )
            else:
                raise ValueError(
                    f"Goal array has unexpected shape {goal.shape}. "
                    f"Expected (4, 4) for pose or ({self.dof},) for configuration."
                )
        else:
            raise ValueError(f"Unknown goal type: {type(goal)}")

    def plan_to_configurations(
        self,
        q_goals: list[np.ndarray],
        *,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> Trajectory | PlanResult | None:
        """Plan to one of multiple goal configurations.

        The planner will attempt to find a path to any of the given
        configurations, returning the first successful path.

        Args:
            q_goals: List of goal joint configurations
            base_heights: Optional list of base heights to search (tried in order).
                         If provided, returns PlanResult with base trajectory.
            strategy: Planning strategy:
                     - "first": Return first successful plan (fastest)
                     - "best": Try all heights, return shortest path
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility

        Returns:
            - Trajectory if planning succeeded (no base_heights)
            - PlanResult if base_heights was provided
            - None if planning failed
        """
        if not q_goals:
            return None

        # If base_heights provided, plan at each height
        if base_heights is not None:
            return self._plan_with_base_heights(
                poses=None,
                tsrs=None,
                configurations=q_goals,
                base_heights=base_heights,
                strategy=strategy,
                timeout=timeout,
                seed=seed,
            )

        # Try each goal configuration
        for q_goal in q_goals:
            path = self.plan_to_configuration(q_goal, timeout=timeout, seed=seed)
            if path is not None:
                # Convert path to trajectory with entity info
                trajectory = Trajectory.from_path(
                    path,
                    vel_limits=self.config.kinematic_limits.velocity,
                    acc_limits=self.config.kinematic_limits.acceleration,
                    entity=self.config.name,
                    joint_names=self.config.joint_names,
                )
                return trajectory

        return None

    def plan_to_pose(
        self,
        pose: np.ndarray | list[np.ndarray],
        *,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> Trajectory | PlanResult | None:
        """Plan to an end-effector pose (or one of multiple poses).

        Uses inverse kinematics to find valid configurations for each pose,
        then plans to reach any of them.

        Args:
            pose: Target 4x4 pose matrix, or list of poses (planner picks one)
            base_heights: Optional list of base heights to search (tried in order).
                         If provided, returns PlanResult with base trajectory.
            strategy: Planning strategy:
                     - "first": Return first successful plan (fastest)
                     - "best": Try all heights, return shortest path
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility

        Returns:
            - Trajectory if planning succeeded (no base_heights)
            - PlanResult if base_heights was provided
            - None if planning failed
        """
        # Convert single pose to list
        poses = [pose] if isinstance(pose, np.ndarray) and pose.shape == (4, 4) else pose

        # If base_heights provided, plan at each height
        if base_heights is not None:
            return self._plan_with_base_heights(
                poses=poses,
                tsrs=None,
                configurations=None,
                base_heights=base_heights,
                strategy=strategy,
                timeout=timeout,
                seed=seed,
            )

        # Solve IK for all poses to get candidate configurations
        q_candidates = []
        for p in poses:
            solutions = self.inverse_kinematics(p, validate=True)
            q_candidates.extend(solutions)

        if not q_candidates:
            return None

        # Plan to any of the candidate configurations
        return self.plan_to_configurations(
            q_candidates,
            timeout=timeout,
            seed=seed,
        )

    def plan_to_tsr(
        self,
        tsr,
        *,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> Trajectory | PlanResult | None:
        """Plan to a TSR (or one of multiple TSRs).

        Uses CBiRRT with TSR constraints to find a collision-free path.

        Args:
            tsr: Target TSR, or list of TSRs (planner picks one)
            base_heights: Optional list of base heights to search (tried in order).
                         If provided, returns PlanResult with base trajectory.
            strategy: Planning strategy:
                     - "first": Return first successful plan (fastest)
                     - "best": Try all heights, return shortest path
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility

        Returns:
            - Trajectory if planning succeeded (no base_heights)
            - PlanResult if base_heights was provided
            - None if planning failed
        """
        # Convert single TSR to list
        tsrs = [tsr] if not isinstance(tsr, list) else tsr

        # If base_heights provided, plan at each height
        if base_heights is not None:
            return self._plan_with_base_heights(
                poses=None,
                tsrs=tsrs,
                configurations=None,
                base_heights=base_heights,
                strategy=strategy,
                timeout=timeout,
                seed=seed,
            )

        # Plan using existing method
        path = self.plan_to_tsrs(tsrs, timeout=timeout, seed=seed)

        if path is None:
            return None

        # Convert path to trajectory with entity info
        trajectory = Trajectory.from_path(
            path,
            vel_limits=self.config.kinematic_limits.velocity,
            acc_limits=self.config.kinematic_limits.acceleration,
            entity=self.config.name,
            joint_names=self.config.joint_names,
        )

        return trajectory

    def _plan_with_base_heights(
        self,
        poses: list[np.ndarray] | None,
        tsrs: list | None,
        configurations: list[np.ndarray] | None,
        base_heights: list[float],
        strategy: str,
        timeout: float,
        seed: int | None,
    ) -> PlanResult | None:
        """Plan arm motion at different base heights.

        Pre-filters heights by collision checking, then tries planners at
        valid heights sequentially (in the order provided).

        Args:
            poses: List of target poses (or None if using TSRs/configurations)
            tsrs: List of target TSRs (or None if using poses/configurations)
            configurations: List of goal configurations (or None if using poses/TSRs)
            base_heights: List of base heights to try (in order)
            strategy: "first" returns first success, "best" tries all and picks shortest
            timeout: Per-planner timeout
            seed: Random seed for reproducibility

        Returns:
            PlanResult with arm and base trajectories, or None if failed
        """
        from geodude.trajectory import create_linear_trajectory

        if strategy not in ("first", "best"):
            raise ValueError(f"strategy must be 'first' or 'best', got {strategy!r}")

        # Get the base for this arm
        base = self.robot.left_base if "left" in self.config.name else self.robot.right_base

        if base is None:
            raise ValueError(
                f"base_heights requires a Vention base, but no base found for arm {self.config.name}"
            )

        # Read current arm configuration and filter reachable heights
        q_start = self.get_joint_positions().copy()
        current_height = base.height

        # Filter heights by collision checking (sequential, no threading needed)
        reachable_heights = [
            h for h in base_heights
            if base.is_path_collision_free(current_height, h)
        ]

        if not reachable_heights:
            return None

        # Determine goal candidates
        # For configurations: use directly
        # For poses: pre-compute IK candidates (same for all heights)
        q_candidates = None
        if configurations is not None:
            q_candidates = configurations
        elif poses is not None:
            q_candidates = []
            for p in poses:
                solutions = self.inverse_kinematics(p, validate=True)
                q_candidates.extend(solutions)
            if not q_candidates:
                return None

        defaults = self.config.planning_defaults

        def plan_at_height(height: float) -> list | None:
            """Plan at a single base height. Returns path or None."""
            try:
                config = CBiRRTConfig(
                    timeout=timeout,
                    max_iterations=defaults.max_iterations,
                    step_size=defaults.step_size,
                    goal_bias=defaults.goal_bias,
                    smoothing_iterations=defaults.smoothing_iterations,
                )
                planner = self.create_planner(
                    config=config,
                    base_joint_name=base.config.joint_name,
                    base_height=height,
                )

                if tsrs is not None:
                    return planner.plan(q_start, goal_tsrs=tsrs, seed=seed)
                elif q_candidates is not None:
                    # Try each IK candidate
                    for q_goal in q_candidates:
                        try:
                            path = planner.plan(q_start, goal=q_goal, seed=seed)
                            if path is not None:
                                return path
                        except Exception:
                            continue
                    return None
                else:
                    return None
            except Exception:
                return None

        def build_result(height: float, path: list) -> PlanResult:
            """Build PlanResult from successful path."""
            arm_trajectory = Trajectory.from_path(
                path,
                vel_limits=self.config.kinematic_limits.velocity,
                acc_limits=self.config.kinematic_limits.acceleration,
                entity=self.config.name,
                joint_names=self.config.joint_names,
            )

            base_trajectory = create_linear_trajectory(
                start=current_height,
                end=height,
                vel_limit=base.config.kinematic_limits.velocity,
                acc_limit=base.config.kinematic_limits.acceleration,
                entity=base.config.name,
                joint_names=base.config.joint_names,
            )

            return PlanResult(
                arm=self,
                arm_trajectory=arm_trajectory,
                base_trajectory=base_trajectory,
                base_height=height,
            )

        # Try heights sequentially (in order provided)
        if strategy == "first":
            # Return on first success
            for height in reachable_heights:
                path = plan_at_height(height)
                if path is not None:
                    return build_result(height, path)

            return None

        else:  # strategy == "best"
            # Try all heights, collect successes, pick shortest path
            successful_results: list[tuple[float, list]] = []

            for height in reachable_heights:
                path = plan_at_height(height)
                if path is not None:
                    successful_results.append((height, path))

            if not successful_results:
                return None

            # Pick shortest path (by path length in configuration space)
            def path_length(path: list) -> float:
                """Compute path length in configuration space."""
                total = 0.0
                for i in range(1, len(path)):
                    total += np.linalg.norm(path[i] - path[i - 1])
                return total

            best_height, best_path = min(successful_results, key=lambda x: path_length(x[1]))
            return build_result(best_height, best_path)

    def pickup(
        self,
        target: str | None = None,
        **kwargs,
    ) -> bool:
        """Pick up an object with this arm.

        Uses affordance-based planning via the active execution context.

        Args:
            target: Object name (e.g., "can_0"), or None for any pickable
            **kwargs: Additional args (see primitives.pickup)

        Returns:
            True if pickup succeeded

        Example:
            with robot.sim() as ctx:
                robot.right_arm.pickup("can_0")
        """
        from geodude.primitives import pickup

        return pickup(self._robot, target, arm=self, **kwargs)

    def place(
        self,
        destination: str,
        **kwargs,
    ) -> bool:
        """Place held object at a destination with this arm.

        Uses affordance-based planning via the active execution context.

        Args:
            destination: Destination name (e.g., "recycle_bin_0")
            **kwargs: Additional args (see primitives.place)

        Returns:
            True if place succeeded

        Example:
            with robot.sim() as ctx:
                robot.right_arm.pickup("can_0")
                robot.right_arm.place("recycle_bin_0")
        """
        from geodude.primitives import place

        return place(self._robot, destination, arm=self, **kwargs)

    def close_gripper(self, steps: int = 100) -> str | None:
        """Close the gripper and detect grasp.

        Returns:
            Name of grasped object, or None
        """
        return self.gripper.close(steps)

    def open_gripper(self, steps: int = 100) -> None:
        """Open the gripper and release any held object."""
        self.gripper.open(steps)

    def grasp(self, object_name: str, steps: int = 100) -> str | None:
        """Close gripper and mark object as grasped.

        High-level grasp operation that:
        1. Sets the object as a candidate for grasp detection
        2. Closes the gripper
        3. Detects if the object was grasped
        4. Updates the grasp manager to mark the object as held

        Args:
            object_name: Name of the object to grasp
            steps: Number of simulation steps for gripper closing

        Returns:
            Name of grasped object if successful, None otherwise

        Example:
            robot.right_arm.grasp("can_0")
        """
        if self.gripper is None:
            return None

        # Set candidate objects for detection
        self.gripper.set_candidate_objects([object_name])

        # Close gripper and detect grasp
        grasped = self.gripper.close(steps)

        if grasped:
            # Update grasp manager
            self.grasp_manager.mark_grasped(grasped, self.side)
            # Attach to gripper body for kinematic tracking
            attach_body = f"{self.side}_ur5e/gripper/right_follower"
            self.grasp_manager.attach_object(grasped, attach_body)

        return grasped

    def release(self, object_name: str | None = None, steps: int = 100) -> None:
        """Open gripper and release held object(s).

        High-level release operation that:
        1. Updates the grasp manager to mark object(s) as released
        2. Detaches object(s) from kinematic tracking
        3. Opens the gripper

        Args:
            object_name: Specific object to release, or None to release all
                        objects held by this arm.
            steps: Number of simulation steps for gripper opening

        Example:
            robot.right_arm.release("can_0")  # Release specific object
            robot.right_arm.release()  # Release all held objects
        """
        if self.gripper is None:
            return

        # Determine which objects to release
        if object_name is not None:
            objects_to_release = [object_name]
        else:
            objects_to_release = list(self.grasp_manager.get_grasped_by(self.side))

        # Release in grasp manager
        for obj in objects_to_release:
            self.grasp_manager.mark_released(obj)
            self.grasp_manager.detach_object(obj)

        # Open gripper
        self.gripper.open(steps)

    def pick(
        self,
        object_name: str,
        object_height: float = 0.05,
        gripper_standoff: float = 0.15,
        approach_distance: float = 0.1,
        lift_height: float = 0.1,
        grasp_type: str = "top",
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> bool:
        """High-level pick action.

        Plans approach to object, grasps, and lifts.

        Args:
            object_name: Name of object to pick
            object_height: Approximate height of the object (meters)
            gripper_standoff: Distance from gripper to object surface at grasp
            approach_distance: Distance to retract for approach pose
            lift_height: Height to lift after grasping
            grasp_type: Type of grasp ("top" or "side")
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility

        Returns:
            True if pick succeeded
        """
        # Import TSR utilities
        from geodude.tsr_utils import (
            create_top_grasp_tsr,
            create_approach_tsr,
            create_lift_tsr,
        )

        # Get object pose
        object_pose = self.robot.get_object_pose(object_name)

        # Create grasp TSR
        if grasp_type == "top":
            grasp_tsr = create_top_grasp_tsr(
                object_pose=object_pose,
                object_height=object_height,
                gripper_standoff=gripper_standoff,
            )
        else:
            raise ValueError(f"Unsupported grasp type: {grasp_type}")

        # 1. Open gripper to prepare for grasp
        # Note: Save arm position since gripper stepping may cause drift
        q_saved = self.get_joint_positions().copy()
        self.open_gripper()
        self.set_joint_positions(q_saved)

        # 2. Plan to grasp pose
        grasp_path = self.plan_to_tsrs([grasp_tsr], timeout=timeout, seed=seed)
        if grasp_path is None:
            return False

        # 3. Create approach pose (back along approach direction)
        # Get the final grasp pose from the planned path without modifying robot state
        final_q = grasp_path[-1]
        grasp_pose = self._get_ee_pose_at_config(final_q)

        approach_tsr = create_approach_tsr(
            target_ee_pose=grasp_pose,
            approach_distance=approach_distance,
        )

        # 4. Plan to approach pose first
        approach_path = self.plan_to_tsrs([approach_tsr], timeout=timeout, seed=seed)
        if approach_path is None:
            # If approach planning fails, try going directly to grasp
            pass
        else:
            # Execute approach path
            self.execute(approach_path)

        # 5. Execute the grasp path (from approach to grasp)
        # Re-plan from current position to grasp
        final_grasp_path = self.plan_to_configuration(final_q)
        if final_grasp_path is not None:
            self.execute(final_grasp_path)
        else:
            # Re-planning failed - this can happen if approach put us in a bad spot
            # Return failure rather than executing stale path from wrong start position
            return False

        # 6. Close gripper to grasp
        self.gripper.set_candidate_objects([object_name])
        grasped = self.close_gripper()

        if grasped != object_name:
            # Failed to grasp - retreat
            return False

        # 7. Lift the object
        current_ee_pose = self.get_ee_pose()
        lift_tsr = create_lift_tsr(
            current_ee_pose=current_ee_pose,
            lift_height=lift_height,
        )

        lift_path = self.plan_to_tsrs([lift_tsr], timeout=timeout, seed=seed)
        if lift_path is not None:
            self.execute(lift_path)

        return True

    def place(
        self,
        target: str | np.ndarray,
        object_height: float = 0.05,
        surface_height: float = 0.02,
        gripper_standoff: float = 0.15,
        approach_distance: float = 0.1,
        retract_height: float = 0.1,
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> bool:
        """High-level place action.

        Plans motion to place location and releases object.

        Args:
            target: Target surface name (body) or 4x4 pose matrix for placement
            object_height: Height of the held object
            surface_height: Height/thickness of the target surface
            gripper_standoff: Distance from gripper to object surface
            approach_distance: Distance to retract for approach pose
            retract_height: Height to retract after placing
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility

        Returns:
            True if place succeeded
        """
        from geodude.tsr_utils import (
            create_place_tsr,
            create_approach_tsr,
            create_lift_tsr,
        )

        # Get target surface pose
        if isinstance(target, str):
            surface_pose = self.robot.get_object_pose(target)
        else:
            surface_pose = np.asarray(target)

        # Create place TSR
        place_tsr = create_place_tsr(
            surface_pose=surface_pose,
            surface_height=surface_height,
            object_height=object_height,
            gripper_standoff=gripper_standoff,
        )

        # 1. Plan to place pose
        place_path = self.plan_to_tsrs([place_tsr], timeout=timeout, seed=seed)
        if place_path is None:
            return False

        # 2. Get the place pose for approach planning without modifying robot state
        final_q = place_path[-1]
        place_pose = self._get_ee_pose_at_config(final_q)

        # 3. Create approach pose
        approach_tsr = create_approach_tsr(
            target_ee_pose=place_pose,
            approach_distance=approach_distance,
        )

        # 4. Plan to approach first
        approach_path = self.plan_to_tsrs([approach_tsr], timeout=timeout, seed=seed)
        if approach_path is not None:
            self.execute(approach_path)

        # 5. Move to place pose
        final_place_path = self.plan_to_configuration(final_q)
        if final_place_path is not None:
            self.execute(final_place_path)
        else:
            # Re-planning failed - this can happen if approach put us in a bad spot
            # Return failure rather than executing stale path from wrong start position
            return False

        # 6. Open gripper to release object
        self.open_gripper()

        # 7. Retract upward
        current_ee_pose = self.get_ee_pose()
        retract_tsr = create_lift_tsr(
            current_ee_pose=current_ee_pose,
            lift_height=retract_height,
        )

        retract_path = self.plan_to_tsrs([retract_tsr], timeout=timeout, seed=seed)
        if retract_path is not None:
            self.execute(retract_path)

        return True

    def move_until_touch(
        self,
        direction: np.ndarray,
        distance: float,
        max_distance: float,
        max_force: float = 5.0,
        max_torque: float | None = None,
        speed: float = 0.02,
        frame: str = "world",
        push_distance: float = 0.0,
        physics: bool | None = None,
        viewer=None,
    ):
        """Move gripper in direction until contact or max_distance reached.

        Jacobian-based Cartesian velocity control for contact-based grasping.
        Moves the end-effector at constant velocity in the specified direction,
        stopping when contact is detected.

        In physics mode, contact is detected via F/T sensor force threshold.
        In kinematic mode, contact is detected via MuJoCo collision detection.

        Args:
            direction: Direction vector for motion (will be normalized)
            distance: Minimum distance to travel before contact checking (meters).
                     Use this to ignore initial collisions (e.g., object on table).
            max_distance: Maximum distance to travel - safety limit (meters).
                         Motion stops here even if no contact detected.
            max_force: Force magnitude threshold in Newtons (default 5.0).
                      Only used in physics mode.
            max_torque: Torque magnitude threshold in Nm (optional).
                       Only used in physics mode.
            speed: Cartesian velocity in m/s (default 0.02 = 2cm/s)
            frame: "world" or "hand" frame for direction vector
            push_distance: Distance to continue after contact (meters).
                          Use for push-grasp to seat object in gripper.
            physics: True for physics mode (F/T sensor), False for kinematic
                    (collision detection). If None, uses active context mode.
            viewer: Optional MuJoCo viewer to sync during motion

        Returns:
            MoveUntilTouchResult with:
                - success: True if contact was detected after min_distance
                - terminated_by: "contact", "max_distance", or "error"
                - distance_moved: Actual distance traveled
                - final_force/final_torque: F/T readings (physics mode)
                - contact_geom: Name of contacted geometry (kinematic mode)

        Example:
            # Approach object from above until contact
            result = robot.right_arm.move_until_touch(
                direction=[0, 0, -1],  # Down in world frame
                distance=0.01,         # Min 1cm before checking
                max_distance=0.05,     # Max 5cm
                max_force=3.0,         # Light touch
            )
            if result.success:
                robot.right_arm.close_gripper()

            # Push-grasp: continue 5mm after contact to seat object
            result = robot.right_arm.move_until_touch(
                direction=[0, 0, 1],   # Forward in gripper frame
                distance=0.01,
                max_distance=0.08,
                max_force=5.0,
                frame="hand",
                push_distance=0.005,   # Push 5mm after contact
            )
        """
        from geodude.cartesian import move_until_touch, MoveUntilTouchResult

        # Determine physics mode from context if not specified
        if physics is None:
            ctx = self.robot._active_context
            if ctx is not None:
                physics = ctx._physics
            else:
                physics = True  # Default to physics mode

        # Get viewer from context if not provided
        if viewer is None:
            ctx = self.robot._active_context
            if ctx is not None:
                viewer = ctx._viewer

        # Call the cartesian module function
        result = move_until_touch(
            arm=self,
            direction=np.asarray(direction),
            distance=distance,
            max_distance=max_distance,
            max_force=max_force,
            max_torque=max_torque,
            speed=speed,
            frame=frame,
            physics=physics,
            viewer=viewer,
        )

        # Handle push_distance if contact was detected
        if result.success and push_distance > 0:
            from geodude.cartesian import execute_twist

            # Continue moving in the same direction for push_distance
            direction = np.asarray(direction, dtype=float)
            direction = direction / np.linalg.norm(direction)
            twist = np.zeros(6)
            twist[:3] = direction * speed

            execute_twist(
                arm=self,
                twist=twist,
                frame=frame,
                max_distance=push_distance,
                physics=physics,
                viewer=viewer,
            )

            # Update distance_moved in result
            result = MoveUntilTouchResult(
                success=result.success,
                terminated_by=result.terminated_by,
                distance_moved=result.distance_moved + push_distance,
                final_force=result.final_force,
                final_torque=result.final_torque,
                contact_geom=result.contact_geom,
            )

        return result
