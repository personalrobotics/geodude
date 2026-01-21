"""Single arm control with planning and execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from geodude.collision import GraspAwareCollisionChecker
from geodude.config import ArmConfig
from geodude.grasp_manager import GraspManager
from geodude.gripper import Gripper

if TYPE_CHECKING:
    from geodude.robot import Geodude


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

        # Planner will be set up lazily
        self._planner = None
        self._collision_checker = None

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

    def plan_to_configuration(self, q_goal: np.ndarray) -> list[np.ndarray] | None:
        """Plan a path to a goal configuration.

        Args:
            q_goal: Goal joint configuration

        Returns:
            List of waypoints, or None if planning failed
        """
        # This would use pycbirrt, but for now just return direct path
        # if collision-free
        q_start = self.get_joint_positions()
        checker = self._get_collision_checker()

        # Simple straight-line check
        n_steps = 20
        for i in range(n_steps + 1):
            alpha = i / n_steps
            q = q_start + alpha * (q_goal - q_start)
            if not checker.is_valid(q):
                return None  # Collision detected

        return [q_start, q_goal]

    def plan_to_tsrs(self, tsrs: list, timeout: float = 30.0) -> list[np.ndarray] | None:
        """Plan a path to reach any of the given TSRs.

        Args:
            tsrs: List of TSR objects defining valid goal poses
            timeout: Planning timeout in seconds

        Returns:
            List of waypoints, or None if planning failed
        """
        # TODO: Integrate with pycbirrt
        # For now, this is a placeholder
        raise NotImplementedError("TSR planning requires pycbirrt integration")

    def execute(self, path: list[np.ndarray], speed: float = 1.0) -> bool:
        """Execute a planned path.

        Args:
            path: List of waypoints (joint configurations)
            speed: Execution speed multiplier

        Returns:
            True if execution completed
        """
        # Simple execution: interpolate and step
        for i in range(len(path) - 1):
            q_start = path[i]
            q_end = path[i + 1]

            # Interpolate between waypoints
            n_steps = 50
            for step in range(n_steps):
                alpha = step / n_steps
                q = q_start + alpha * (q_end - q_start)
                self.set_joint_positions(q)
                mujoco.mj_step(self.model, self.data)

        # Final position
        self.set_joint_positions(path[-1])
        return True

    def close_gripper(self, steps: int = 100) -> str | None:
        """Close the gripper and detect grasp.

        Returns:
            Name of grasped object, or None
        """
        return self.gripper.close(steps)

    def open_gripper(self, steps: int = 100) -> None:
        """Open the gripper and release any held object."""
        self.gripper.open(steps)

    def pick(self, object_name: str) -> bool:
        """High-level pick action.

        Plans approach to object, grasps, and lifts.

        Args:
            object_name: Name of object to pick

        Returns:
            True if pick succeeded
        """
        # TODO: Full implementation with TSR planning
        # 1. Get grasp TSRs for object
        # 2. Plan to grasp pose
        # 3. Execute approach
        # 4. Close gripper
        # 5. Lift

        # For now, just try to close gripper and see if we grasp something
        self.gripper.set_candidate_objects([object_name])
        grasped = self.close_gripper()
        return grasped == object_name

    def place(self, object_name: str, on: str | np.ndarray | None = None) -> bool:
        """High-level place action.

        Plans motion to place location and releases object.

        Args:
            object_name: Name of object being placed
            on: Target surface/object name or position

        Returns:
            True if place succeeded
        """
        # TODO: Full implementation with TSR planning
        # 1. Get place TSRs for target
        # 2. Plan to place pose (using grasp-aware collision)
        # 3. Execute
        # 4. Open gripper

        # For now, just open gripper
        self.open_gripper()
        return True
