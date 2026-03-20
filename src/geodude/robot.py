"""Main Geodude robot interface.

Composes mj_manipulator Arms, RobotiqGrippers, and VentionBases into a
bimanual robot with high-level planning and manipulation primitives.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mj_environment import Environment
from mj_manipulator import (
    Arm,
    GraspManager,
    PlanResult,
    RobotiqGripper,
    SimContext,
    Trajectory,
)
from mj_manipulator.arms.eaik_solver import MuJoCoEAIKSolver
from mj_manipulator.arms.ur5e import (
    UR5E_ACCELERATION_LIMITS,
    UR5E_VELOCITY_LIMITS,
)
from mj_manipulator.config import ArmConfig, KinematicLimits

from geodude.config import GeodudConfig, GeodudeArmSpec, setup_logging
from geodude.vention_base import VentionBase

logger = logging.getLogger(__name__)


class Geodude:
    """High-level interface for the Geodude bimanual robot.

    Provides:
    - Two UR5e arms with Robotiq 2F-140 grippers (from mj_manipulator)
    - Optional Vention linear actuator bases
    - Bimanual planning with arm/height interleaving
    - Named configurations from MuJoCo keyframes
    - Affordance-driven pickup/place primitives

    Example::

        robot = Geodude(objects={"can": 2, "recycle_bin": 1})
        with robot.sim(physics=True) as ctx:
            robot.pickup("can_0")
            robot.place("recycle_bin_0")
    """

    def __init__(
        self,
        config: GeodudConfig | None = None,
        objects: dict[str, int] | None = None,
    ):
        self.config = config or GeodudConfig.default()

        # Load MuJoCo model via mj_environment
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"MuJoCo model not found: {self.config.model_path}\n"
                "Make sure geodude_assets is available."
            )

        if objects:
            from prl_assets import OBJECTS_DIR

            scene_config = self._create_temp_scene_config(objects)
            self._env = Environment(
                base_scene_xml=str(self.config.model_path),
                objects_dir=str(OBJECTS_DIR),
                scene_config_yaml=scene_config,
            )
        else:
            self._env = Environment(
                base_scene_xml=str(self.config.model_path),
                objects_dir=None,
                scene_config_yaml=None,
            )

        self.model = self._env.model
        self.data = self._env.data

        # Shared grasp manager
        self.grasp_manager = GraspManager(self.model, self.data)

        # Create arms from mj_manipulator
        self._left_arm = self._create_arm(self.config.left_arm, "left")
        self._right_arm = self._create_arm(self.config.right_arm, "right")

        # Create bases (if configured)
        self._left_base: VentionBase | None = None
        self._right_base: VentionBase | None = None
        if self.config.left_base is not None:
            self._left_base = VentionBase(
                self.model, self.data, self.config.left_base, self._left_arm,
            )
        if self.config.right_base is not None:
            self._right_base = VentionBase(
                self.model, self.data, self.config.right_base, self._right_arm,
            )

        # Named poses from keyframes
        keyframe_poses = self._load_keyframe_poses()
        self._named_poses = {**self.config.named_poses, **keyframe_poses}

        # Initialize state
        mujoco.mj_forward(self.model, self.data)
        setup_logging(self.config.debug)

        # Active execution context (set by sim() context manager)
        self._context: SimContext | None = None

    def _create_arm(self, spec: GeodudeArmSpec, name: str) -> Arm:
        """Create an mj_manipulator Arm from a GeodudeArmSpec."""
        joint_names = self.config.joint_names(spec)
        arm_config = ArmConfig(
            name=name,
            entity_type="arm",
            joint_names=joint_names,
            kinematic_limits=KinematicLimits(
                velocity=UR5E_VELOCITY_LIMITS.copy(),
                acceleration=UR5E_ACCELERATION_LIMITS.copy(),
            ),
            ee_site=spec.ee_site,
        )

        # Create arm first to get joint indices for IK solver
        arm = Arm(self._env, arm_config)

        # Create EAIK solver
        joint_limits = arm.get_joint_limits()
        first_joint_body = self._env.model.jnt_bodyid[arm.joint_ids[0]]
        base_body_id = self._env.model.body_parentid[first_joint_body]

        ik_solver = MuJoCoEAIKSolver(
            model=self._env.model,
            data=self._env.data,
            joint_ids=list(arm.joint_ids),
            joint_qpos_indices=arm.joint_qpos_indices,
            ee_site_id=arm.ee_site_id,
            base_body_id=base_body_id,
            joint_limits=joint_limits,
        )

        # Create Robotiq gripper
        gripper = RobotiqGripper(
            self.model, self.data, name, prefix=spec.gripper_prefix,
            grasp_manager=self.grasp_manager,
        )

        return Arm(
            self._env, arm_config,
            ik_solver=ik_solver, gripper=gripper,
            grasp_manager=self.grasp_manager,
        )

    # -- Properties ----------------------------------------------------------

    @property
    def left_arm(self) -> Arm:
        return self._left_arm

    @property
    def right_arm(self) -> Arm:
        return self._right_arm

    @property
    def left_base(self) -> VentionBase | None:
        return self._left_base

    @property
    def right_base(self) -> VentionBase | None:
        return self._right_base

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def named_poses(self) -> dict[str, dict[str, list[float]]]:
        return self._named_poses

    @property
    def _active_context(self) -> SimContext | None:
        """Currently active execution context (set by sim())."""
        return self._context

    @_active_context.setter
    def _active_context(self, ctx: SimContext | None) -> None:
        self._context = ctx

    def get_arm_spec(self, arm: Arm) -> GeodudeArmSpec:
        """Get the GeodudeArmSpec for an arm (has hand_type, prefix, etc.)."""
        if arm is self._left_arm:
            return self.config.left_arm
        return self.config.right_arm

    # -- Arm resolution ------------------------------------------------------

    def _resolve_arms(self, arm: Arm | str | None) -> list[Arm]:
        """Resolve arm specification to list of Arm instances."""
        if arm is None:
            return [self._right_arm, self._left_arm]
        if isinstance(arm, str):
            if arm in ("left", "left_arm"):
                return [self._left_arm]
            if arm in ("right", "right_arm"):
                return [self._right_arm]
            raise ValueError(f"Unknown arm name: {arm}")
        if isinstance(arm, Arm):
            return [arm]
        raise ValueError(f"Invalid arm specification: {arm}")

    def _resolve_arm(self, arm: Arm | str) -> Arm:
        """Resolve a single arm specification to an Arm instance."""
        if isinstance(arm, Arm):
            return arm
        if arm in ("left", "left_arm"):
            return self._left_arm
        if arm in ("right", "right_arm"):
            return self._right_arm
        raise ValueError(f"Unknown arm name: {arm}")

    def _get_base_for_arm(self, arm: Arm) -> VentionBase | None:
        """Get the base associated with an arm."""
        if arm is self._left_arm:
            return self._left_base
        return self._right_base

    def arm_name(self, arm: Arm) -> str:
        """Get the side name for an arm ('left' or 'right')."""
        if arm is self._left_arm:
            return "left"
        return "right"

    # -- Simulation context --------------------------------------------------

    def sim(
        self,
        physics: bool = True,
        viewer=None,
        viewer_fps: float = 30.0,
        headless: bool = False,
    ) -> SimContext:
        """Create simulation execution context.

        Returns a context manager for executing trajectories in MuJoCo.

        Example::

            with robot.sim(physics=True) as ctx:
                path = robot.left_arm.plan_to_pose(target)
                traj = robot.left_arm.retime(path)
                ctx.execute(traj)
                ctx.arm("left").grasp("can_0")
        """
        arms = {"left": self._left_arm, "right": self._right_arm}
        return SimContext(
            self.model, self.data, arms,
            physics=physics, headless=headless,
            viewer=viewer, viewer_fps=viewer_fps,
        )

    # -- Named poses ---------------------------------------------------------

    def go_to(self, pose_name: str, ctx: SimContext | None = None) -> bool:
        """Move both arms to a named configuration.

        Args:
            pose_name: Name of the configuration (e.g., 'ready').
            ctx: Execution context. If None, uses the active context.

        Returns:
            True if both arms planned and executed successfully.
        """
        if pose_name not in self.named_poses:
            raise ValueError(f"Unknown named pose: {pose_name}")

        ctx = ctx or self._active_context
        if ctx is None:
            raise RuntimeError("No active execution context. Use robot.sim().")

        pose = self.named_poses[pose_name]
        success = True
        for side, arm in [("left", self._left_arm), ("right", self._right_arm)]:
            if side in pose:
                q_goal = np.array(pose[side])
                path = arm.plan_to_configuration(q_goal)
                if path is not None:
                    traj = arm.retime(path)
                    ctx.execute(traj)
                else:
                    logger.warning("Failed to plan %s arm to '%s'", side, pose_name)
                    success = False
        return success

    # -- Planning (bimanual with base height search) -------------------------

    def plan_to_tsrs(
        self,
        goal_tsrs,
        *,
        arm: Arm | str | None = None,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> PlanResult | None:
        """Plan to TSR goals with arm/height interleaving.

        If no arm is specified, tries both arms. If base_heights are given,
        tries each arm at each height.

        Returns:
            PlanResult with arm trajectory (and base trajectory if height changed),
            or None if all attempts failed.
        """
        return self._plan_with_sequence(
            goal_tsrs=goal_tsrs,
            arm=arm,
            base_heights=base_heights,
            strategy=strategy,
            timeout=timeout,
            seed=seed,
        )

    def plan_to_pose(
        self,
        pose: np.ndarray,
        *,
        arm: Arm | str | None = None,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> PlanResult | None:
        """Plan to an EE pose with arm/height interleaving."""
        return self._plan_with_sequence(
            pose=pose,
            arm=arm,
            base_heights=base_heights,
            strategy=strategy,
            timeout=timeout,
            seed=seed,
        )

    def _plan_with_sequence(
        self,
        *,
        goal_tsrs=None,
        pose: np.ndarray | None = None,
        arm: Arm | str | None = None,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> PlanResult | None:
        """Core planning: try arm/height combinations, return first or best."""
        arms = self._resolve_arms(arm)

        # Build (arm, height) sequence
        heights = base_heights if base_heights else [None]
        sequence: list[tuple[Arm, float | None]] = []

        # Randomize arm order for fairness
        arms_shuffled = list(arms)
        random.shuffle(arms_shuffled)

        for h in heights:
            for a in arms_shuffled:
                sequence.append((a, h))

        # Try each combination
        results: list[PlanResult] = []
        for a, h in sequence:
            result = self._plan_single(a, h, goal_tsrs=goal_tsrs, pose=pose,
                                       timeout=timeout, seed=seed)
            if result is not None:
                if strategy == "first":
                    return result
                results.append(result)

        if not results:
            return None

        # "best" strategy: pick shortest arm trajectory
        return min(results, key=lambda r: r.arm_trajectory.duration)

    def _plan_single(
        self,
        arm: Arm,
        height: float | None,
        *,
        goal_tsrs=None,
        pose: np.ndarray | None = None,
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> PlanResult | None:
        """Plan with a single arm at a specific base height."""
        base = self._get_base_for_arm(arm)

        # Move base if requested
        base_traj = None
        if height is not None and base is not None:
            current_height = base.get_height()
            if abs(current_height - height) > 0.001:
                base_traj = base.plan_to(height, arm)
                if base_traj is None:
                    return None  # Base move would cause collision

        # Plan arm motion
        try:
            if goal_tsrs is not None:
                tsrs = goal_tsrs if isinstance(goal_tsrs, list) else [goal_tsrs]
                path = arm.plan_to_tsrs(tsrs, timeout=timeout, seed=seed)
            elif pose is not None:
                path = arm.plan_to_pose(pose, timeout=timeout, seed=seed)
            else:
                raise ValueError("Must provide goal_tsrs or pose")
        except Exception:
            return None

        if path is None:
            return None

        arm_traj = arm.retime(path)

        return PlanResult(
            arm_name=arm.config.name,
            arm_trajectory=arm_traj,
            base_trajectory=base_traj,
        )

    # -- Primitives (delegate to primitives module) --------------------------

    def pickup(self, object_name: str, grasp_tsrs: list, **kwargs) -> bool:
        """Pick up an object using caller-provided grasp TSRs."""
        from geodude.primitives import pickup
        return pickup(self, object_name, grasp_tsrs, **kwargs)

    def place(self, place_tsrs: list, **kwargs) -> bool:
        """Place held object using caller-provided place TSRs."""
        from geodude.primitives import place
        return place(self, place_tsrs, **kwargs)

    # -- State management ----------------------------------------------------

    def forward(self) -> None:
        """Run forward kinematics to update positions."""
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """Reset simulation to initial state (ready keyframe if available)."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        for obj in list(self.grasp_manager.grasped.keys()):
            self.grasp_manager.mark_released(obj)

    def reset_to_keyframe(self, name: str) -> None:
        """Reset robot to a MuJoCo keyframe by name."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id == -1:
            raise ValueError(f"Keyframe '{name}' not found in model")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def get_object_pose(self, object_name: str) -> np.ndarray:
        """Get the 4x4 pose of an object in the scene."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id == -1:
            raise ValueError(f"Object '{object_name}' not found in model")
        self.forward()
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[body_id]
        return T

    # -- Internal helpers ----------------------------------------------------

    def _load_keyframe_poses(self) -> dict[str, dict[str, list[float]]]:
        """Extract named poses from MuJoCo keyframes."""
        poses: dict[str, dict[str, list[float]]] = {}
        for key_id in range(self.model.nkey):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, key_id)
            if name is None:
                continue
            key_qpos = self.model.key_qpos[key_id]
            left_qpos = [float(key_qpos[i]) for i in self._left_arm.joint_qpos_indices]
            right_qpos = [float(key_qpos[i]) for i in self._right_arm.joint_qpos_indices]
            poses[name] = {"left": left_qpos, "right": right_qpos}
        return poses

    def _create_temp_scene_config(self, objects: dict[str, int]) -> str:
        """Create temporary scene_config.yaml from objects dict."""
        import tempfile

        import yaml

        config = {
            "objects": {
                obj_type: {"count": count} for obj_type, count in objects.items()
            }
        }
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        )
        yaml.dump(config, temp_file)
        temp_file.close()
        return temp_file.name
