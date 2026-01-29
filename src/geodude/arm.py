"""Single arm control with planning and execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from geodude.collision import GraspAwareCollisionChecker
from geodude.config import ArmConfig
from geodude.executor import KinematicExecutor, PhysicsExecutor
from geodude.grasp_manager import GraspManager
from geodude.gripper import Gripper
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

        # Initialize gripper
        self.gripper = Gripper(
            self.model,
            self.data,
            config.name,
            config.gripper_actuator,
            config.gripper_bodies,
            grasp_manager,
        )

        # Get the arm base body for coordinate transforms
        # The UR5e base body is named "{arm_name}_ur5e/base"
        base_body_name = f"{config.name}_ur5e/base"
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
        )
        if self._base_body_id == -1:
            # Fallback: try without the trailing slash naming convention
            base_body_name = f"{config.name}_ur5e"
            self._base_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
            )
        # Note: base_body_id can be -1 if not found; we'll handle this in get_base_pose

        # Planner, IK solver, and executor will be set up lazily
        self._planner = None
        self._collision_checker = None
        self._ik_solver = None
        self._executor = None
        self._ee_offset = None  # Cached EE offset transform

    @property
    def name(self) -> str:
        """Arm name ('left' or 'right')."""
        return self.config.name

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

            planner_config = config or CBiRRTConfig(
                max_iterations=5000,
                step_size=0.1,
                goal_bias=0.1,
                ik_num_seeds=1,  # EAIK finds all solutions analytically
                smoothing_iterations=100,  # Default smoothing
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
        if hasattr(self, '_cached_base_rotation'):
            return self._cached_base_rotation

        # Compare FK at reference configuration to compute R
        # Debug: print that we're computing R
        print(f"[DEBUG] Computing base rotation R for {self.config.name} arm...", flush=True)
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
            q_target = np.array(self.robot.named_poses[target][self.name])
        else:
            q_target = np.asarray(target)

        # For now, just set directly (no planning/interpolation)
        # TODO: Plan and execute trajectory
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

        config = CBiRRTConfig(
            timeout=timeout,
            step_size=0.1,
            goal_bias=0.1,
            ik_num_seeds=1,
            smoothing_iterations=100,  # Default smoothing
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

        config = CBiRRTConfig(
            max_iterations=5000,
            step_size=0.1,
            goal_bias=0.1,
            ik_num_seeds=1,
            timeout=timeout,
            smoothing_iterations=100,  # Default smoothing
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

    def close_gripper(self, steps: int = 100) -> str | None:
        """Close the gripper and detect grasp.

        Returns:
            Name of grasped object, or None
        """
        return self.gripper.close(steps)

    def open_gripper(self, steps: int = 100) -> None:
        """Open the gripper and release any held object."""
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
