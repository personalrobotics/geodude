"""Main Geodude robot interface."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from mj_environment import Environment

from geodude.affordances import AffordanceRegistry
from geodude.arm import Arm
from geodude.config import GeodudConfig, setup_logging
from geodude.grasp_manager import GraspManager
from geodude.planning import PlanResult
from geodude.trajectory import Trajectory
from geodude.vention_base import VentionBase

if TYPE_CHECKING:
    from geodude.execution import SimContext


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

        # Initialize affordance registry
        tsr_templates_dir = Path(__file__).parent.parent.parent / "tsr_templates"
        self._affordance_registry = AffordanceRegistry([tsr_templates_dir])

        # Run forward to initialize state
        mujoco.mj_forward(self.model, self.data)

        # Configure debug logging based on config (reads from env if not set)
        setup_logging(self.config.debug)

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

    @property
    def affordances(self) -> AffordanceRegistry:
        """Affordance registry for manipulation discovery."""
        return self._affordance_registry

    @property
    def _active_context(self) -> "SimContext | None":
        """Currently active execution context, if any.

        This is set by the context manager (robot.sim() or robot.hardware())
        and used by primitives like pickup() and place() to execute actions.
        """
        return getattr(self, "_context", None)

    @_active_context.setter
    def _active_context(self, ctx: "SimContext | None") -> None:
        """Set the active execution context."""
        self._context = ctx

    def apply_debug_config(self) -> None:
        """Re-apply debug configuration to loggers.

        Call this after modifying robot.config.debug to apply changes.

        Example:
            robot.config.debug.enable("executor", "gripper")
            robot.apply_debug_config()
        """
        setup_logging(self.config.debug)

        # Sync gripper debug flag with the debug config
        # This allows the per-call verbose logging in close_gripper to work
        self.config.physics.gripper.debug = self.config.debug.gripper

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

    def pickup(
        self,
        target: str | None = None,
        **kwargs,
    ) -> bool:
        """Pick up an object using affordance-based planning.

        Uses the AffordanceRegistry to discover grasp TSRs, then plans and
        executes via the active execution context.

        Args:
            target: Object name (e.g., "can_0"), or None for any pickable
            **kwargs: Additional args passed to primitives.pickup():
                - object_type: Filter by type if target is None
                - arm: Specific arm ("left"/"right") or None for auto
                - base_heights: Heights to search (default [0.2, 0.0, 0.4])
                - lift_height: Height to lift after grasping (default 0.1m)
                - timeout: Planning timeout (default 30s)

        Returns:
            True if pickup succeeded

        Raises:
            RuntimeError: If no execution context is active

        Example:
            with robot.sim() as ctx:
                robot.pickup("can_0")
                robot.pickup(object_type="can")
        """
        from geodude.primitives import pickup

        return pickup(self, target, **kwargs)

    def place(
        self,
        destination: str,
        **kwargs,
    ) -> bool:
        """Place held object at a destination.

        Uses the AffordanceRegistry to discover place TSRs, then plans and
        executes via the active execution context.

        Args:
            destination: Destination name (e.g., "recycle_bin_0")
            **kwargs: Additional args passed to primitives.place():
                - arm: Arm holding object, or None for auto-detect
                - base_heights: Heights to search (default [0.2, 0.0, 0.4])
                - timeout: Planning timeout (default 30s)

        Returns:
            True if place succeeded

        Raises:
            RuntimeError: If no execution context is active

        Example:
            with robot.sim() as ctx:
                robot.pickup("can_0")
                robot.place("recycle_bin_0")
        """
        from geodude.primitives import place

        return place(self, destination, **kwargs)

    def get_pickable_objects(self, object_type: str | None = None) -> list[str]:
        """Get names of objects that can be picked up.

        Args:
            object_type: Filter by type (e.g., "can")

        Returns:
            List of pickable object instance names
        """
        from geodude.primitives import get_pickable_objects

        return get_pickable_objects(self, object_type)

    def get_place_destinations(self, object_type: str) -> list[str]:
        """Get valid place destinations for an object type.

        Args:
            object_type: Type of object being placed

        Returns:
            List of destination instance names
        """
        from geodude.primitives import get_place_destinations

        return get_place_destinations(self, object_type)

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
        sequence: list[tuple[Arm | str, float]] | None = None,
        arm: Arm | str | None = None,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> PlanResult | Trajectory | None:
        """Plan to an end-effector pose with either arm.

        If no arm is specified, tries both arms with interleaved heights.

        Args:
            pose: Target 4x4 pose matrix, or list of poses (planner picks one)
            sequence: Optional explicit list of (arm, height) tuples to try in order.
                     Each tuple is (Arm | "left" | "right", height_float).
                     If provided, base_heights and arm are ignored.
            arm: Which arm to use: Arm instance, "left", "right", or None (try both)
            base_heights: Optional list of base heights to search.
                         Default sequence interleaves arms at each height.
            strategy: Planning strategy:
                     - "first": Return first successful plan (fastest)
                     - "best": Try all options, return shortest path
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility

        Returns:
            - Trajectory or PlanResult if planning succeeded
            - None if planning failed with all arms
        """
        return self._plan_with_sequence(
            goal=pose,
            goal_type="pose",
            sequence=sequence,
            arm=arm,
            base_heights=base_heights,
            strategy=strategy,
            timeout=timeout,
            seed=seed,
        )

    def plan_to_tsr(
        self,
        tsr,
        *,
        sequence: list[tuple[Arm | str, float]] | None = None,
        arm: Arm | str | None = None,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> PlanResult | Trajectory | None:
        """Plan to a TSR with either arm.

        If no arm is specified, tries both arms with interleaved heights.

        Args:
            tsr: Target TSR, or list of TSRs (planner picks one)
            sequence: Optional explicit list of (arm, height) tuples to try in order.
                     Each tuple is (Arm | "left" | "right", height_float).
                     If provided, base_heights and arm are ignored.
            arm: Which arm to use: Arm instance, "left", "right", or None (try both)
            base_heights: Optional list of base heights to search.
                         Default sequence interleaves arms at each height.
            strategy: Planning strategy:
                     - "first": Return first successful plan (fastest)
                     - "best": Try all options, return shortest path
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility

        Returns:
            - Trajectory or PlanResult if planning succeeded
            - None if planning failed with all arms

        Example:
            # Default: interleaved arms at each height, random first arm
            robot.plan_to_tsr(tsr, base_heights=[0.2, 0.0, 0.4])

            # Explicit sequence for fine control
            robot.plan_to_tsr(tsr, sequence=[
                ("right", 0.2),
                ("left", 0.2),
                ("right", 0.0),
            ])
        """
        return self._plan_with_sequence(
            goal=tsr,
            goal_type="tsr",
            sequence=sequence,
            arm=arm,
            base_heights=base_heights,
            strategy=strategy,
            timeout=timeout,
            seed=seed,
        )

    def plan_to(
        self,
        goal: np.ndarray | list,
        *,
        sequence: list[tuple[Arm | str, float]] | None = None,
        arm: Arm | str | None = None,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float = 30.0,
        seed: int | None = None,
    ) -> PlanResult | Trajectory | None:
        """Plan to a goal (configuration, pose, or TSR) with either arm.

        Unified planning method that dispatches based on goal type.
        If no arm is specified, tries both arms with interleaved heights.

        Args:
            goal: Target configuration, pose, TSR, or list of goals
            sequence: Optional explicit list of (arm, height) tuples to try in order.
                     Each tuple is (Arm | "left" | "right", height_float).
                     If provided, base_heights and arm are ignored.
            arm: Which arm to use: Arm instance, "left", "right", or None (try both)
            base_heights: Optional list of base heights to search.
                         Default sequence interleaves arms at each height.
            strategy: Planning strategy:
                     - "first": Return first successful plan (fastest)
                     - "best": Try all options, return shortest path
            timeout: Planning timeout in seconds
            seed: Random seed for reproducibility

        Returns:
            - Trajectory or PlanResult if planning succeeded
            - None if planning failed
        """
        return self._plan_with_sequence(
            goal=goal,
            goal_type="goal",
            sequence=sequence,
            arm=arm,
            base_heights=base_heights,
            strategy=strategy,
            timeout=timeout,
            seed=seed,
        )

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

    def _resolve_arm(self, arm: Arm | str) -> Arm:
        """Resolve a single arm specification to an Arm instance."""
        if isinstance(arm, Arm):
            return arm
        elif arm == "left" or arm == "left_arm":
            return self._left_arm
        elif arm == "right" or arm == "right_arm":
            return self._right_arm
        else:
            raise ValueError(f"Unknown arm name: {arm}")

    def _get_base_for_arm(self, arm: Arm) -> VentionBase | None:
        """Get the base associated with an arm."""
        if "left" in arm.config.name:
            return self.left_base
        else:
            return self.right_base

    def _build_default_sequence(
        self,
        arms: list[Arm],
        base_heights: list[float] | None,
    ) -> list[tuple[Arm, float]]:
        """Build default interleaved sequence of (arm, height) pairs.

        Default behavior:
        - Randomly pick which arm goes first
        - At each height level, try both arms before moving to next height
        - If no base_heights, use current height only

        Args:
            arms: List of arms to include
            base_heights: Heights to try (None = current height only)

        Returns:
            List of (arm, height) tuples in order to try
        """
        # Randomly shuffle arm order
        arms = list(arms)  # Copy to avoid mutation
        random.shuffle(arms)

        # Determine heights to use
        if base_heights is None or len(base_heights) == 0:
            # Use current height for each arm (or 0.0 if no base)
            heights = [0.0]  # Will be ignored if arm has no base
        else:
            heights = base_heights

        # Build interleaved sequence: for each height, try all arms
        sequence = []
        for height in heights:
            for arm in arms:
                sequence.append((arm, height))

        return sequence

    def _plan_with_sequence(
        self,
        goal,
        goal_type: str,
        sequence: list[tuple[Arm | str, float]] | None,
        arm: Arm | str | None,
        base_heights: list[float] | None,
        strategy: str,
        timeout: float,
        seed: int | None,
    ) -> PlanResult | Trajectory | None:
        """Core planning implementation with sequence support.

        Args:
            goal: The planning goal (TSR, pose, or configuration)
            goal_type: "tsr" or "goal" to dispatch to correct arm method
            sequence: Explicit (arm, height) sequence, or None to build default
            arm: Arm specification (used if sequence is None)
            base_heights: Heights to search (used if sequence is None)
            strategy: "first" or "best"
            timeout: Per-attempt timeout
            seed: Random seed

        Returns:
            PlanResult, Trajectory, or None
        """
        if strategy not in ("first", "best"):
            raise ValueError(f"strategy must be 'first' or 'best', got {strategy!r}")

        # Build or resolve the sequence
        if sequence is not None:
            # Explicit sequence provided - resolve arm strings to Arm instances
            resolved_sequence = [(self._resolve_arm(a), h) for a, h in sequence]
        else:
            # Build default sequence from arms and heights
            arms = self._resolve_arms(arm)

            # Special case: single arm without base_heights - delegate directly
            if len(arms) == 1 and base_heights is None:
                a = arms[0]
                if goal_type == "tsr":
                    return a.plan_to_tsr(
                        goal,
                        base_heights=None,
                        strategy=strategy,
                        timeout=timeout,
                        seed=seed,
                    )
                elif goal_type == "pose":
                    return a.plan_to_pose(
                        goal,
                        base_heights=None,
                        strategy=strategy,
                        timeout=timeout,
                        seed=seed,
                    )
                else:  # goal_type == "goal"
                    return a.plan_to(
                        goal,
                        base_heights=None,
                        strategy=strategy,
                        timeout=timeout,
                        seed=seed,
                    )

            resolved_sequence = self._build_default_sequence(arms, base_heights)

        # Plan at each position in sequence
        def plan_at(arm: Arm, height: float) -> PlanResult | Trajectory | None:
            """Plan with a specific arm at a specific height."""
            # Use base_heights=[height] to get a PlanResult with height info
            if goal_type == "tsr":
                return arm.plan_to_tsr(
                    goal,
                    base_heights=[height],
                    strategy="first",  # Single height, so first == only
                    timeout=timeout,
                    seed=seed,
                )
            elif goal_type == "pose":
                return arm.plan_to_pose(
                    goal,
                    base_heights=[height],
                    strategy="first",
                    timeout=timeout,
                    seed=seed,
                )
            else:  # goal_type == "goal"
                return arm.plan_to(
                    goal,
                    base_heights=[height],
                    strategy="first",
                    timeout=timeout,
                    seed=seed,
                )

        if strategy == "first":
            # Try sequence in order, return first success
            for arm, height in resolved_sequence:
                try:
                    result = plan_at(arm, height)
                    if result is not None:
                        return result
                except Exception:
                    continue
            return None

        else:  # strategy == "best"
            # Try all, collect successes, pick shortest
            successful: list[tuple[Arm, PlanResult | Trajectory]] = []

            for arm, height in resolved_sequence:
                try:
                    result = plan_at(arm, height)
                    if result is not None:
                        successful.append((arm, result))
                except Exception:
                    continue

            if not successful:
                return None

            # Pick shortest trajectory
            def trajectory_duration(item: tuple[Arm, PlanResult | Trajectory]) -> float:
                _, result = item
                if isinstance(result, PlanResult):
                    return result.arm_trajectory.duration
                return result.duration

            _, best_result = min(successful, key=trajectory_duration)
            return best_result

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

    def sim(
        self,
        physics: bool = True,
        viewer=None,
        viewer_fps: float = 30.0,
    ) -> "SimContext":
        """Create simulation execution context.

        Returns a context manager that provides a unified interface for
        executing trajectories in MuJoCo simulation.

        Args:
            physics: If True (default), use physics simulation with realistic
                    dynamics. If False, use kinematic execution (perfect
                    tracking, no dynamics).
            viewer: Optional MuJoCo viewer. If None, a viewer is created
                   when entering the context.
            viewer_fps: Target viewer refresh rate in Hz (default 30).
                       Higher values = smoother but slower execution.
                       Set to 0 for unlimited (sync every physics step).

        Returns:
            SimContext that can be used as a context manager.

        Example:
            robot = Geodude(objects={"can": 1})

            with robot.sim(physics=True) as ctx:
                # Plan and execute
                result = robot.plan_to_tsr(grasp_tsr, base_heights=[0.2, 0.0])
                ctx.execute(result)

                # Grasp operations
                ctx.arm("right").grasp("can_0")

                # Main loop
                while ctx.is_running():
                    ctx.sync()
        """
        from geodude.execution import SimContext

        return SimContext(self, viewer=viewer, physics=physics, viewer_fps=viewer_fps)

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
