"""Main Geodude robot interface."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mj_environment import Environment

from geodude.arm import Arm
from geodude.config import GeodudConfig
from geodude.grasp_manager import GraspManager
from geodude.planning import PlanResult
from geodude.trajectory import Trajectory
from geodude.vention_base import VentionBase


class Geodude:
    """High-level interface for the Geodude bimanual robot.

    Provides:
    - Easy access to left and right arms
    - Grasp state management
    - Environment interaction with optional object management
    - Named configurations

    Example:
        # Robot only
        robot = Geodude()
        robot.right_arm.go_to('ready')

        # With manipulable objects from prl_assets
        robot = Geodude(objects={"can": 2, "recycle_bin": 1})
        robot.env.registry.activate("can", pos=[0.4, -0.2, 0.81])
    """

    def __init__(
        self,
        config: GeodudConfig | None = None,
        objects: dict[str, int] | None = None,
    ):
        """Initialize the Geodude robot.

        Args:
            config: Robot configuration. If None, uses default configuration.
            objects: Optional dict mapping object type to count, e.g.
                {"can": 2, "recycle_bin": 1}. Objects are loaded from prl_assets
                and start hidden. Use robot.env.registry.activate() to show them.
        """
        self.config = config or GeodudConfig.default()

        # Load MuJoCo model via mj_environment
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"MuJoCo model not found: {self.config.model_path}\n"
                "Make sure geodude_assets is available."
            )

        if objects:
            # Load with object management from prl_assets
            from prl_assets import OBJECTS_DIR

            scene_config = self._create_temp_scene_config(objects)
            self._env = Environment(
                base_scene_xml=str(self.config.model_path),
                objects_dir=str(OBJECTS_DIR),
                scene_config_yaml=scene_config,
            )
        else:
            # Robot-only mode (no objects)
            self._env = Environment(
                base_scene_xml=str(self.config.model_path),
                objects_dir=None,
                scene_config_yaml=None,
            )
        self.model = self._env.model
        self.data = self._env.data

        # Initialize grasp manager
        self.grasp_manager = GraspManager(self.model, self.data)

        # Initialize arms
        self._left_arm = Arm(self, self.config.left_arm, self.grasp_manager)
        self._right_arm = Arm(self, self.config.right_arm, self.grasp_manager)

        # Initialize bases (if configured)
        self._left_base: VentionBase | None = None
        self._right_base: VentionBase | None = None

        if self.config.left_base is not None:
            self._left_base = VentionBase(
                self.model, self.data, self.config.left_base, self._left_arm
            )
        if self.config.right_base is not None:
            self._right_base = VentionBase(
                self.model, self.data, self.config.right_base, self._right_arm
            )

        # Load named poses: keyframes take precedence over config
        keyframe_poses = self._load_keyframe_poses()
        self._named_poses = {**self.config.named_poses, **keyframe_poses}

        # Run forward to initialize state
        mujoco.mj_forward(self.model, self.data)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Geodude":
        """Create Geodude from a YAML configuration file.

        Args:
            config_path: Path to configuration YAML file.

        Returns:
            Initialized Geodude instance.
        """
        config = GeodudConfig.from_yaml(Path(config_path))
        return cls(config)

    @classmethod
    def from_xml(cls, xml_path: str | Path) -> "Geodude":
        """Create Geodude from a MuJoCo XML file with default joint names.

        Args:
            xml_path: Path to MuJoCo XML file.

        Returns:
            Initialized Geodude instance.
        """
        config = GeodudConfig.default()
        config.model_path = Path(xml_path)
        return cls(config)

    @property
    def left_arm(self) -> Arm:
        """Left arm controller."""
        return self._left_arm

    @property
    def right_arm(self) -> Arm:
        """Right arm controller."""
        return self._right_arm

    @property
    def left_base(self) -> VentionBase | None:
        """Left Vention base controller (linear actuator)."""
        return self._left_base

    @property
    def right_base(self) -> VentionBase | None:
        """Right Vention base controller (linear actuator)."""
        return self._right_base

    @property
    def env(self) -> Environment:
        """MuJoCo environment wrapper for forking and state management."""
        return self._env

    @property
    def named_poses(self) -> dict[str, dict[str, list[float]]]:
        """Named pose configurations (from keyframes + config)."""
        return self._named_poses

    def go_to(self, pose_name: str, speed: float = 1.0) -> bool:
        """Move both arms to a named configuration.

        Args:
            pose_name: Name of the configuration (e.g., 'ready')
            speed: Execution speed multiplier

        Returns:
            True if successful
        """
        if pose_name not in self.named_poses:
            raise ValueError(f"Unknown named pose: {pose_name}")

        left_success = self.left_arm.go_to(pose_name, speed)
        right_success = self.right_arm.go_to(pose_name, speed)
        return left_success and right_success

    def step(self, n_steps: int = 1) -> None:
        """Step the simulation forward.

        Args:
            n_steps: Number of simulation steps
        """
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        """Run forward kinematics to update positions."""
        mujoco.mj_forward(self.model, self.data)

    def get_time(self) -> float:
        """Get current simulation time."""
        return self.data.time

    def reset(self) -> None:
        """Reset simulation to initial state (ready keyframe if available)."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Clear grasp state
        for obj in list(self.grasp_manager.grasped.keys()):
            self.grasp_manager.mark_released(obj)

    def reset_to_keyframe(self, name: str) -> None:
        """Reset robot to a MuJoCo keyframe by name.

        Args:
            name: Name of the keyframe (e.g., 'ready')

        Raises:
            ValueError: If keyframe not found in model
        """
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id == -1:
            raise ValueError(f"Keyframe '{name}' not found in model")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def plan_to_pose(
        self,
        pose: np.ndarray | list[np.ndarray],
        *,
        arm: Arm | str | None = None,
        execute: bool = True,
        base_heights: list[float] | None = None,
        timeout: float = 30.0,
        seed: int | None = None,
        viewer=None,
        executor_type: str = "physics",
    ) -> PlanResult | Trajectory | None:
        """Plan to an end-effector pose with either arm.

        If no arm is specified, races both arms in parallel and returns first success.

        Args:
            pose: Target 4x4 pose matrix, or list of poses (planner picks one)
            arm: Which arm to use: Arm instance, "left", "right", or None (race both)
            execute: If True (default), execute trajectory after planning.
                    If False, return trajectory without executing.
            base_heights: Optional list of base heights to search in parallel.
                         If provided, returns PlanResult with base trajectory.
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility
            viewer: Optional MuJoCo viewer for execution visualization
            executor_type: "physics" or "kinematic" for execution

        Returns:
            - Trajectory or PlanResult if planning succeeded
            - None if planning failed with all arms
        """
        arms = self._resolve_arms(arm)

        # Single arm: plan directly
        if len(arms) == 1:
            return arms[0].plan_to_pose(
                pose,
                execute=execute,
                base_heights=base_heights,
                timeout=timeout,
                seed=seed,
                viewer=viewer,
                executor_type=executor_type,
            )

        # Multiple arms: race in parallel, first success wins
        def plan_arm(a: Arm) -> PlanResult | Trajectory | None:
            return a.plan_to_pose(
                pose,
                execute=False,  # Don't execute in thread
                base_heights=base_heights,
                timeout=timeout,
                seed=seed,
                viewer=None,
                executor_type=executor_type,
            )

        with ThreadPoolExecutor(max_workers=len(arms)) as executor:
            futures = {executor.submit(plan_arm, a): a for a in arms}

            while futures:
                done, _ = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    if result is not None:
                        # Cancel remaining
                        for f in futures:
                            f.cancel()

                        # Execute if requested
                        if execute:
                            if isinstance(result, PlanResult):
                                arm_obj = result.arm
                                base = self.left_base if "left" in arm_obj.config.name else self.right_base
                                if result.base_trajectory is not None and base is not None:
                                    base.move_to(result.base_height, viewer=viewer, executor_type=executor_type)
                                arm_obj.execute(result.arm_trajectory, viewer=viewer, executor_type=executor_type)
                            else:
                                # Plain Trajectory - need to find which arm
                                winning_arm = futures[future]
                                winning_arm.execute(result, viewer=viewer, executor_type=executor_type)

                        return result
                    del futures[future]

        return None

    def plan_to_tsr(
        self,
        tsr,
        *,
        arm: Arm | str | None = None,
        execute: bool = True,
        base_heights: list[float] | None = None,
        timeout: float = 30.0,
        seed: int | None = None,
        viewer=None,
        executor_type: str = "physics",
    ) -> PlanResult | Trajectory | None:
        """Plan to a TSR with either arm.

        If no arm is specified, races both arms in parallel and returns first success.

        Args:
            tsr: Target TSR, or list of TSRs (planner picks one)
            arm: Which arm to use: Arm instance, "left", "right", or None (race both)
            execute: If True (default), execute trajectory after planning.
                    If False, return trajectory without executing.
            base_heights: Optional list of base heights to search in parallel.
                         If provided, returns PlanResult with base trajectory.
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility
            viewer: Optional MuJoCo viewer for execution visualization
            executor_type: "physics" or "kinematic" for execution

        Returns:
            - Trajectory or PlanResult if planning succeeded
            - None if planning failed with all arms
        """
        arms = self._resolve_arms(arm)

        # Single arm: plan directly
        if len(arms) == 1:
            return arms[0].plan_to_tsr(
                tsr,
                execute=execute,
                base_heights=base_heights,
                timeout=timeout,
                seed=seed,
                viewer=viewer,
                executor_type=executor_type,
            )

        # Multiple arms: race in parallel, first success wins
        def plan_arm(a: Arm) -> PlanResult | Trajectory | None:
            return a.plan_to_tsr(
                tsr,
                execute=False,  # Don't execute in thread
                base_heights=base_heights,
                timeout=timeout,
                seed=seed,
                viewer=None,
                executor_type=executor_type,
            )

        with ThreadPoolExecutor(max_workers=len(arms)) as executor:
            futures = {executor.submit(plan_arm, a): a for a in arms}

            while futures:
                done, _ = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    if result is not None:
                        # Cancel remaining
                        for f in futures:
                            f.cancel()

                        # Execute if requested
                        if execute:
                            if isinstance(result, PlanResult):
                                arm_obj = result.arm
                                base = self.left_base if "left" in arm_obj.config.name else self.right_base
                                if result.base_trajectory is not None and base is not None:
                                    base.move_to(result.base_height, viewer=viewer, executor_type=executor_type)
                                arm_obj.execute(result.arm_trajectory, viewer=viewer, executor_type=executor_type)
                            else:
                                # Plain Trajectory - need to find which arm
                                winning_arm = futures[future]
                                winning_arm.execute(result, viewer=viewer, executor_type=executor_type)

                        return result
                    del futures[future]

        return None

    def plan_to(
        self,
        goal: np.ndarray | list,
        *,
        arm: Arm | str | None = None,
        execute: bool = True,
        base_heights: list[float] | None = None,
        timeout: float = 30.0,
        seed: int | None = None,
        viewer=None,
        executor_type: str = "physics",
    ) -> PlanResult | Trajectory | None:
        """Plan to a goal (configuration, pose, or TSR) with either arm.

        Unified planning method that dispatches based on goal type.
        If no arm is specified, races both arms in parallel and returns first success.

        Args:
            goal: Target configuration, pose, TSR, or list of goals
            arm: Which arm to use: Arm instance, "left", "right", or None (race both)
            execute: If True (default), execute trajectory after planning.
                    If False, return trajectory without executing.
            base_heights: Optional list of base heights to search in parallel.
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility
            viewer: Optional MuJoCo viewer for execution visualization
            executor_type: "physics" or "kinematic" for execution

        Returns:
            - Trajectory or PlanResult if planning succeeded
            - None if planning failed
        """
        arms = self._resolve_arms(arm)

        # Single arm: plan directly
        if len(arms) == 1:
            return arms[0].plan_to(
                goal,
                execute=execute,
                base_heights=base_heights,
                timeout=timeout,
                seed=seed,
                viewer=viewer,
                executor_type=executor_type,
            )

        # Multiple arms: race in parallel, first success wins
        def plan_arm(a: Arm) -> PlanResult | Trajectory | None:
            return a.plan_to(
                goal,
                execute=False,  # Don't execute in thread
                base_heights=base_heights,
                timeout=timeout,
                seed=seed,
                viewer=None,
                executor_type=executor_type,
            )

        with ThreadPoolExecutor(max_workers=len(arms)) as executor:
            futures = {executor.submit(plan_arm, a): a for a in arms}

            while futures:
                done, _ = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    if result is not None:
                        # Cancel remaining
                        for f in futures:
                            f.cancel()

                        # Execute if requested
                        if execute:
                            if isinstance(result, PlanResult):
                                arm_obj = result.arm
                                base = self.left_base if "left" in arm_obj.config.name else self.right_base
                                if result.base_trajectory is not None and base is not None:
                                    base.move_to(result.base_height, viewer=viewer, executor_type=executor_type)
                                arm_obj.execute(result.arm_trajectory, viewer=viewer, executor_type=executor_type)
                            else:
                                # Plain Trajectory - need to find which arm
                                winning_arm = futures[future]
                                winning_arm.execute(result, viewer=viewer, executor_type=executor_type)

                        return result
                    del futures[future]

        return None

    def _resolve_arms(self, arm: Arm | str | None) -> list[Arm]:
        """Resolve arm specification to list of Arm instances.

        Args:
            arm: Arm instance, "left", "right", or None (both arms)

        Returns:
            List of Arm instances to try
        """
        if arm is None:
            return [self._right_arm, self._left_arm]  # Try right first (has gripper)
        elif isinstance(arm, str):
            if arm == "left" or arm == "left_arm":
                return [self._left_arm]
            elif arm == "right" or arm == "right_arm":
                return [self._right_arm]
            else:
                raise ValueError(f"Unknown arm name: {arm}")
        elif isinstance(arm, Arm):
            return [arm]
        else:
            raise ValueError(f"Invalid arm specification: {arm}")

    def _load_keyframe_poses(self) -> dict[str, dict[str, list[float]]]:
        """Extract named poses from MuJoCo keyframes.

        Reads keyframes from the model and extracts arm-specific joint values
        for use with go_to() and other pose-based methods.

        Returns:
            Dict mapping keyframe names to arm poses:
            {"ready": {"left": [6 floats], "right": [6 floats]}}
        """
        poses: dict[str, dict[str, list[float]]] = {}

        for key_id in range(self.model.nkey):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, key_id)
            if name is None:
                continue

            # Extract arm joint values from full keyframe qpos
            key_qpos = self.model.key_qpos[key_id]
            left_qpos = [float(key_qpos[idx]) for idx in self._left_arm.joint_qpos_indices]
            right_qpos = [float(key_qpos[idx]) for idx in self._right_arm.joint_qpos_indices]

            poses[name] = {
                "left": left_qpos,
                "right": right_qpos,
            }

        return poses

    # Environment interaction (placeholder for mj_environment integration)

    def get_object_pose(self, object_name: str) -> np.ndarray:
        """Get the pose of an object in the scene.

        Args:
            object_name: Name of the object body

        Returns:
            4x4 transformation matrix
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id == -1:
            raise ValueError(f"Object '{object_name}' not found in model")

        self.forward()

        pos = self.data.xpos[body_id]
        rot = self.data.xmat[body_id].reshape(3, 3)

        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        return transform

    def set_object_pose(self, object_name: str, pose: np.ndarray) -> None:
        """Set the pose of a free-floating object.

        Args:
            object_name: Name of the object body
            pose: 4x4 transformation matrix
        """
        # Find the joint for this body (assumes freejoint)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id == -1:
            raise ValueError(f"Object '{object_name}' not found in model")

        # Find freejoint
        joint_id = None
        for jid in range(self.model.njnt):
            if self.model.jnt_bodyid[jid] == body_id:
                if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                    joint_id = jid
                    break

        if joint_id is None:
            raise ValueError(f"Object '{object_name}' has no freejoint")

        qpos_adr = self.model.jnt_qposadr[joint_id]

        # Set position
        self.data.qpos[qpos_adr:qpos_adr + 3] = pose[:3, 3]

        # Convert rotation matrix to quaternion
        quat = self._mat_to_quat(pose[:3, :3])
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = quat

        self.forward()

    def _mat_to_quat(self, mat: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
        # MuJoCo uses (w, x, y, z) quaternion convention
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return quat

    def _create_temp_scene_config(self, objects: dict[str, int]) -> str:
        """Create temporary scene_config.yaml from objects dict.

        Args:
            objects: Dict mapping object type to count, e.g. {"can": 2}

        Returns:
            Path to temporary YAML file.
        """
        import tempfile

        import yaml

        config = {
            "objects": {
                obj_type: {"count": count} for obj_type, count in objects.items()
            }
        }

        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.dump(config, temp_file)
        temp_file.close()
        return temp_file.name
