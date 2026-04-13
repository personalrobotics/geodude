# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Main Geodude robot interface.

Composes mj_manipulator Arms, RobotiqGrippers, and VentionBases into a
bimanual robot with high-level planning and manipulation primitives.
"""

from __future__ import annotations

import logging
import random

import mujoco
import numpy as np
from mj_environment import Environment
from mj_manipulator import (
    Arm,
    GraspManager,
    PlanResult,
    RobotiqGripper,
    SimContext,
)
from mj_manipulator.arms.eaik_solver import MuJoCoEAIKSolver
from mj_manipulator.arms.ur5e import (
    UR5E_ACCELERATION_LIMITS,
    UR5E_VELOCITY_LIMITS,
)
from mj_manipulator.config import ArmConfig, KinematicLimits
from mj_manipulator.grasp_verifier import GraspVerifier

from geodude.config import GeodudConfig, GeodudeArmSpec, setup_logging
from geodude.vention_base import VentionBase

logger = logging.getLogger(__name__)


class _GeodudeSimContext:
    """Wrapper that sets robot._active_context on enter/exit."""

    def __init__(self, inner: SimContext, robot: "Geodude"):
        self._inner = inner
        self._robot = robot

    def __enter__(self):
        ctx = self._inner.__enter__()
        self._robot._active_context = ctx
        return ctx

    def __exit__(self, *args):
        self._robot._active_context = None
        return self._inner.__exit__(*args)


class _ArmScope:
    """Unified arm interface — high-level primitives + low-level Arm access.

    Returned by ``robot.right`` / ``robot.left``. Provides geodude-specific
    methods (pickup, place, close, open) and delegates everything else to
    the underlying mj_manipulator Arm via ``__getattr__``::

        robot.left.pickup("can")        # geodude primitive
        robot.left.close()              # gripper control
        robot.left.get_ee_pose()        # Arm method (delegated)
        robot.left.get_ft_wrench()      # Arm method (delegated)
        robot.left.plan_to_pose(target) # Arm method (delegated)

    Class-level annotations expose delegated Arm methods to IPython/Jedi
    for tab completion (Jedi uses static analysis and can't follow __getattr__).
    """

    def __init__(self, robot: "Geodude", side: str):
        self._robot = robot
        self._side = side

    @property
    def _arm(self):
        return self._robot._resolve_arm(self._side)

    def __getattr__(self, name):
        # Delegate unknown attributes to the underlying Arm
        return getattr(self._arm, name)

    def __dir__(self):
        # Combine _ArmScope methods + Arm methods for tab completion
        return sorted(set(super().__dir__()) | set(dir(self._arm)))

    # -- Delegated Arm methods (explicit for IPython tab completion) --
    # IPython's runtime completer only sees methods defined on the class,
    # not __getattr__ delegated ones. These thin wrappers make the key
    # methods tab-completable.

    def get_ee_pose(self):
        """Current end-effector pose as 4x4 homogeneous transform."""
        return self._arm.get_ee_pose()

    def get_ft_wrench(self):
        """Wrist F/T reading [fx,fy,fz,tx,ty,tz] in sensor local frame."""
        return self._arm.get_ft_wrench()

    def get_ft_wrench_world(self):
        """Wrist F/T reading [fx,fy,fz,tx,ty,tz] in world frame."""
        return self._arm.get_ft_wrench_world()

    def tare_ft(self):
        """Zero the F/T sensor at the current reading."""
        return self._arm.tare_ft()

    def get_joint_positions(self):
        """Current joint positions (rad)."""
        return self._arm.get_joint_positions()

    def set_joint_positions(self, q):
        """Set joint positions directly (sim only). Use plan_to_configuration() on hardware."""
        return self._arm.set_joint_positions(q, ctx=self._robot._active_context)

    def get_joint_velocities(self):
        """Current joint velocities (rad/s)."""
        return self._arm.get_joint_velocities()

    def get_joint_limits(self):
        """Joint position limits as (lower, upper) arrays."""
        return self._arm.get_joint_limits()

    def forward_kinematics(self, q):
        """Compute end-effector pose for a given joint configuration."""
        return self._arm.forward_kinematics(q)

    def plan_to_pose(self, pose, **kwargs):
        """Plan a collision-free path to an end-effector pose."""
        return self._arm.plan_to_pose(pose, **kwargs)

    def plan_to_configuration(self, q_goal, **kwargs):
        """Plan a collision-free path to a joint configuration."""
        return self._arm.plan_to_configuration(q_goal, **kwargs)

    def plan_to_tsrs(self, goal_tsrs, **kwargs):
        """Plan a collision-free path to a TSR-defined goal region."""
        return self._arm.plan_to_tsrs(goal_tsrs, **kwargs)

    def retime(self, path, **kwargs):
        """Retime a joint path into a smooth trajectory."""
        return self._arm.retime(path, **kwargs)

    def check_collisions(self):
        """Check current configuration for collisions. Prints a summary."""
        return self._arm.check_collisions()

    @property
    def has_ft_sensor(self) -> bool:
        """Whether this arm has a wrist F/T sensor configured."""
        return self._arm.has_ft_sensor

    @property
    def gripper(self) -> RobotiqGripper:
        """The arm's gripper."""
        return self._arm.gripper

    @property
    def config(self) -> ArmConfig:
        """Arm configuration (joint names, limits, ee_site, etc.)."""
        return self._arm.config

    @property
    def grasp_manager(self) -> GraspManager:
        """GraspManager for grasp state queries."""
        return self._arm.grasp_manager

    @property
    def ee_site_id(self) -> int:
        """MuJoCo site ID for the end-effector."""
        return self._arm.ee_site_id

    def pickup(self, target: str | None = None, **kwargs) -> bool:
        return self._robot.pickup(target, arm=self._side, **kwargs)

    def place(self, destination: str | None = None, **kwargs) -> bool:
        return self._robot.place(destination, arm=self._side, **kwargs)

    def go_home(self, **kwargs) -> bool:
        from geodude.primitives import go_home

        return go_home(self._robot, arm=self._side, **kwargs)

    def close(self) -> str | None:
        """Close the gripper. Grasps whatever is between the fingers.

        Returns:
            Name of grasped object, or None if nothing detected.
        """
        ctx = self._robot._active_context
        if ctx is None:
            raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")
        return ctx.arm(self._side).grasp()

    def open(self) -> None:
        """Open the gripper. Releases whatever is held."""
        ctx = self._robot._active_context
        if ctx is None:
            raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")
        ctx.arm(self._side).release()


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

    config: GeodudConfig
    model: mujoco.MjModel
    data: mujoco.MjData
    grasp_manager: GraspManager

    def __init__(
        self,
        config: GeodudConfig | None = None,
        objects: dict[str, int] | None = None,
    ):
        self.config: GeodudConfig = config or GeodudConfig.default()

        # Load MuJoCo model via mj_environment
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"MuJoCo model not found: {self.config.model_path}\nMake sure geodude_assets is available."
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

        self.model: mujoco.MjModel = self._env.model
        self.data: mujoco.MjData = self._env.data

        # Shared grasp manager
        self.grasp_manager: GraspManager = GraspManager(self.model, self.data)

        # Create arms from mj_manipulator
        self._left_arm = self._create_arm(self.config.left_arm, "left")
        self._right_arm = self._create_arm(self.config.right_arm, "right")

        # Cache freejoint qpos addresses (for hide-all in reset)
        self._freejoint_qpos_addrs = [
            self.model.jnt_qposadr[i]
            for i in range(self.model.njnt)
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        ]

        # Create bases (if configured)
        self._left_base: VentionBase | None = None
        self._right_base: VentionBase | None = None
        if self.config.left_base is not None:
            self._left_base = VentionBase(
                self.model,
                self.data,
                self.config.left_base,
                self._left_arm,
            )
        if self.config.right_base is not None:
            self._right_base = VentionBase(
                self.model,
                self.data,
                self.config.right_base,
                self._right_arm,
            )

        # Named poses from keyframes
        keyframe_poses = self._load_keyframe_poses()
        self._named_poses = {**self.config.named_poses, **keyframe_poses}

        # Perception service — runs the full pipeline (mock detection →
        # alias resolution → tracker → env.update) same as hardware.
        from mj_manipulator.perception import SimPerceptionService

        asset_mgr = None
        try:
            from asset_manager import AssetManager
            from prl_assets import OBJECTS_DIR

            asset_mgr = AssetManager(str(OBJECTS_DIR))
        except ImportError:
            pass

        self._perception = SimPerceptionService(
            self._env,
            grasp_manager=self.grasp_manager,
            asset_manager=asset_mgr,
        )
        # fixture_types is populated later by setup_scene when the
        # demo config is known. Until then, refresh() preserves
        # nothing as fixtures (empty set).

        # Initialize state
        mujoco.mj_forward(self.model, self.data)
        setup_logging(self.config.debug)

        # Active execution context (set by sim() context manager)
        self._context: SimContext | None = None

        # Abort flag (thread-safe, shared between terminal and Viser)
        import threading

        self._abort_event = threading.Event()

    # -----------------------------------------------------------------
    # Abort mechanism
    # -----------------------------------------------------------------

    def request_abort(self) -> None:
        """Signal all running operations to stop.

        When an ownership registry is available (tick-driven mode), aborts
        all arms via per-arm flags. Falls back to global event otherwise.
        """
        if self._context is not None and self._context.ownership is not None:
            self._context.ownership.abort_all()
        self._abort_event.set()

    def clear_abort(self) -> None:
        """Clear the abort flag (call before starting a new operation)."""
        if self._context is not None and self._context.ownership is not None:
            self._context.ownership.clear_all()
        self._abort_event.clear()

    def is_abort_requested(self) -> bool:
        """Check if a global abort has been requested."""
        return self._abort_event.is_set()

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
            ft_force_sensor=f"{spec.prefix}/ft_sensor_force",
            ft_torque_sensor=f"{spec.prefix}/ft_sensor_torque",
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
            self.model,
            self.data,
            name,
            prefix=spec.gripper_prefix,
            grasp_manager=self.grasp_manager,
        )

        arm = Arm(
            self._env,
            arm_config,
            ik_solver=ik_solver,
            gripper=gripper,
            grasp_manager=self.grasp_manager,
        )

        # Attach a sensor-based grasp verifier so gripper.is_holding /
        # gripper.held_object reflect real signals instead of stale
        # GraspManager bookkeeping.
        #
        # The grasp verifier uses no load signals — the only check is
        # the decisive-negative branch (gripper at mechanical stop →
        # nothing held), which reads gripper.get_actual_position()
        # directly inside _collect_facts, not through the signal list.
        #
        # GripperPositionSignal was previously in the signal list, but
        # the load-drop check (|val| < |baseline| * 0.7) is wrong for
        # position: tiny compliance drift (fingers flex ~0.06 rad
        # under load) triggers false LOST on objects grasped at low
        # position values (pop_tarts_case_0 at 0.209 → 0.146 drift).
        # The object is still firmly held; the fingers just flexed.
        #
        # With an empty signal list, the verifier has one check:
        # "did the gripper reach the mechanical stop?" If yes → LOST.
        # If no → HOLDING. Delayed detection of a mid-transport drop
        # (object falls out, fingers close to stop) is acceptable —
        # the decisive-negative fires within a few ticks once the
        # fingers close through the now-empty space.
        gripper.grasp_verifier = GraspVerifier(
            gripper=gripper,
            signals=[],
        )

        return arm

    # -- Properties ----------------------------------------------------------

    @property
    def left(self) -> _ArmScope:
        """Left arm: ``robot.left.pickup("can")``, ``robot.left.get_ee_pose()``."""
        return _ArmScope(self, "left")

    @property
    def right(self) -> _ArmScope:
        """Right arm: ``robot.right.pickup("can")``, ``robot.right.get_ee_pose()``."""
        return _ArmScope(self, "right")

    @property
    def left_arm(self) -> Arm:
        import warnings

        warnings.warn(
            "robot.left_arm is deprecated. Use robot.left instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._left_arm

    @property
    def right_arm(self) -> Arm:
        import warnings

        warnings.warn(
            "robot.right_arm is deprecated. Use robot.right instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
    def arms(self) -> dict[str, Arm]:
        """All arms, keyed by side name."""
        return {"left": self._left_arm, "right": self._right_arm}

    @property
    def grasp_source(self):
        """GraspSource backed by prl_assets geometry."""
        if not hasattr(self, "_grasp_source"):
            from geodude.grasp_source import GeodueGraspSource

            self._grasp_source = GeodueGraspSource(self)
        return self._grasp_source

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

    # (see _GeodudeSimContext below)

    def sim(
        self,
        physics: bool = True,
        viewer=None,
        viewer_fps: float = 30.0,
        headless: bool = False,
        event_loop=None,
    ) -> _GeodudeSimContext:
        """Create simulation execution context.

        Returns a context manager for executing trajectories in MuJoCo.
        Sets robot._active_context on enter, clears on exit.

        Example::

            with robot.sim(physics=True) as ctx:
                path = robot.left_arm.plan_to_pose(target)
                traj = robot.left_arm.retime(path)
                ctx.execute(traj)
                ctx.arm("left").grasp("can_0")
        """
        arms = {"left": self._left_arm, "right": self._right_arm}
        entities = {}
        if self._left_base is not None:
            entities[self._left_base.config.name] = self._left_base
        if self._right_base is not None:
            entities[self._right_base.config.name] = self._right_base
        inner = SimContext(
            self.model,
            self.data,
            arms,
            physics=physics,
            headless=headless,
            viewer=viewer,
            viewer_fps=viewer_fps,
            entities=entities,
            abort_fn=self.is_abort_requested,
            event_loop=event_loop,
        )
        return _GeodudeSimContext(inner, self)

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
        timeout: float | None = None,
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
        timeout: float | None = None,
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
        timeout: float | None = None,
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
            result = self._plan_single(a, h, goal_tsrs=goal_tsrs, pose=pose, timeout=timeout, seed=seed)
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
        timeout: float | None = None,
        seed: int | None = None,
    ) -> PlanResult | None:
        """Plan with a single arm at a specific base height.

        Teleports the base for planning (so IK/RRT sees the correct
        workspace), then restores it. The base trajectory is included
        in the PlanResult so ctx.execute() moves the base properly
        through physics/kinematics.
        """
        base = self._get_base_for_arm(arm)
        original_height = None
        base_traj = None

        # Plan base trajectory and teleport for arm planning
        if height is not None and base is not None:
            current_height = base.get_height()
            if abs(current_height - height) > 0.001:
                base_traj = base.plan_to(height, check_collisions=True)
                if base_traj is None:
                    return None  # path blocked by collision
                original_height = current_height
                base.set_height(height)  # teleport for arm planning

        # Resolve timeout from config if not specified
        if timeout is None:
            timeout = self.config.planning.timeout

        # Plan arm motion (with base at target height)
        try:
            if goal_tsrs is not None:
                tsrs = goal_tsrs if isinstance(goal_tsrs, list) else [goal_tsrs]
                path = arm.plan_to_tsrs(tsrs, timeout=timeout, seed=seed)
            elif pose is not None:
                path = arm.plan_to_pose(pose, timeout=timeout, seed=seed)
            else:
                raise ValueError("Must provide goal_tsrs or pose")
        except Exception as e:
            logger.info("Planning failed: %s", e)
            path = None

        if path is None:
            # Restore base height on failure
            if original_height is not None:
                base.set_height(original_height)
            return None

        # Restore base — execution will move it properly
        if original_height is not None:
            base.set_height(original_height)

        arm_traj = arm.retime(path)

        return PlanResult(
            arm_name=arm.config.name,
            arm_trajectory=arm_traj,
            base_trajectory=base_traj,
            base_height=height,
        )

    # -- Scene setup ---------------------------------------------------------

    def setup_scene(
        self,
        fixtures: dict[str, list[list[float]]] | None = None,
    ) -> None:
        """Set up the scene: place fixtures and ready the robot.

        Activates fixture objects at specified positions, sets bases to
        midpoint height, and arms to "ready" keyframe.

        Args:
            fixtures: Stationary objects and their positions, e.g.
                ``{"recycle_bin": [[0.85, -0.35, 0.01], [-0.85, -0.35, 0.01]]}``
        """
        fixtures = fixtures or {}
        self._fixtures = fixtures

        # Tell the perception service which types are fixtures so
        # refresh() preserves them (they're not detected by perception).
        self._perception._fixture_types = set(fixtures.keys())

        # 1. Activate fixtures at specified positions
        for obj_type, positions in fixtures.items():
            for pos in positions:
                self._env.registry.activate(obj_type, pos=list(pos))

        # 2. Set bases to midpoint
        for base in [self._left_base, self._right_base]:
            if base is not None:
                base.set_height(0.25)

        # 3. Set arms to ready keyframe
        if "ready" in self._named_poses:
            for side, arm in [("left", self._left_arm), ("right", self._right_arm)]:
                q = np.array(self._named_poses["ready"][side])
                for i, idx in enumerate(arm.joint_qpos_indices):
                    self.data.qpos[idx] = q[i]

        self.forward()

    def holding(self) -> tuple[str, str] | None:
        """Return which arm is holding an object.

        Returns:
            ``(side, object_name)`` if either arm is holding, else ``None``.

        Example::

            result = robot.holding()
            if result:
                side, obj = result
                print(f"{side} arm is holding {obj}")
        """
        for side in ("left", "right"):
            held = list(self.grasp_manager.get_grasped_by(side))
            if held:
                return (side, held[0])
        return None

    # -- Scene queries -------------------------------------------------------

    def find_objects(self, target: str | None = None) -> list[str]:
        """Find objects in the scene.

        Args:
            target: "can_0" (specific), "can" (type), None (all graspable).

        Returns:
            List of body names on the table (active, not grasped, not hidden).

        Example::

            robot.find_objects()         # ['can_0', 'can_1', 'spam_can_0']
            robot.find_objects("can")    # ['can_0', 'can_1']
        """
        objects = self.grasp_source.get_graspable_objects()
        if target is not None:
            # Filter: exact match (instance) or prefix match (type)
            objects = [o for o in objects if o == target or o.startswith(target + "_")]
        return objects

    # -- Primitives (delegate to primitives module) --------------------------

    def pickup(self, target: str | None = None, **kwargs) -> bool:
        """Pick up an object.

        Args:
            target: "can_0" (specific), "can" (any can), None (anything).
            arm: "left", "right", or None (try both).
            verbose: Show BT tree status.
        """
        from geodude.primitives import pickup

        return pickup(self, target, **kwargs)

    def place(self, destination: str | None = None, **kwargs) -> bool:
        """Place held object at a destination.

        Args:
            destination: "recycle_bin_0" (specific), "recycle_bin" (any), None (any).
            arm: "left", "right", or None (auto-detect).
            verbose: Show BT tree status.
        """
        from geodude.primitives import place

        return place(self, destination, **kwargs)

    def go_home(self) -> bool:
        """Return all arms to ready configuration."""
        from geodude.primitives import go_home

        return go_home(self)

    # -- State management ----------------------------------------------------

    def forward(self) -> None:
        """Run forward kinematics and sync viewer."""
        mujoco.mj_forward(self.model, self.data)
        if self._context is not None:
            self._context.sync()

    def reset(self) -> None:
        """Reset the scene to its initial state.

        Hides all objects, releases grasps, restores fixtures to their
        original positions, and sets the robot to the ready keyframe.
        Call ``_spawn_manipulable_objects`` after to re-scatter objects.

        For just returning the robot to home, use ``robot.go_home()``
        which plans and executes through the context.
        """
        # Reset MuJoCo state to keyframe
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Sync controller targets to new positions (prevents violent corrections)
        if self._context is not None:
            self._context.hold()

        # Release grasps and clear F/T tare
        for obj in list(self.grasp_manager.grasped.keys()):
            self.grasp_manager.mark_released(obj)
        for arm in [self._left_arm, self._right_arm]:
            arm._ft_tare_offset = np.zeros(6)

        # Move ALL freejoint bodies to hide position. The keyframe reset put
        # them at the origin; setup_scene and _spawn_manipulable_objects will
        # activate the ones that should be visible.
        if self._env.registry is not None:
            hide_pos = self._env.hide_pos
            for qpos_adr in self._freejoint_qpos_addrs:
                self.data.qpos[qpos_adr : qpos_adr + 3] = hide_pos
            # Clear the registry's active state
            for name in list(self._env.registry.active_objects):
                self._env.registry.hide(name)

        # Re-setup scene (fixtures + robot pose — calls forward() internally)
        self.setup_scene(fixtures=getattr(self, "_fixtures", None))

    def reset_to_keyframe(self, name: str) -> None:
        """Reset robot to a MuJoCo keyframe by name."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id == -1:
            raise ValueError(f"Keyframe '{name}' not found in model")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.forward()

    @property
    def perception(self):
        """Perception service for object pose queries."""
        return self._perception

    def get_object_pose(self, object_name: str) -> np.ndarray:
        """Get the 4x4 pose of an object in the scene."""
        pose = self._perception.get_pose(object_name)
        if pose is not None:
            return pose
        # Fallback for non-registry bodies (fixtures, robot links)
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

        config = {"objects": {obj_type: {"count": count} for obj_type, count in objects.items()}}
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        )
        yaml.dump(config, temp_file)
        temp_file.close()
        return temp_file.name
