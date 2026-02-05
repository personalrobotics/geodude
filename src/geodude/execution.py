"""Execution context abstraction for simulation and hardware.

Provides a unified API for executing trajectories in simulation or on real
hardware. The context handles routing trajectories to the correct entity
(arm, base, gripper) and manages state synchronization.

Usage:
    from geodude import Geodude

    robot = Geodude(objects={"can": 1})

    # Simulation context
    with robot.sim(physics=True) as ctx:
        result = robot.plan_to_tsr(grasp_tsr, base_heights=[0.2, 0.0, 0.4])
        ctx.execute(result)
        ctx.arm("right").grasp("can_0")

    # Hardware context (future)
    with robot.hardware() as ctx:
        result = robot.plan_to_tsr(grasp_tsr, base_heights=[0.2, 0.0, 0.4])
        ctx.execute(result)
        ctx.arm("right").grasp("can_0")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import mujoco

if TYPE_CHECKING:
    from geodude.arm import Arm
    from geodude.planning import PlanResult
    from geodude.robot import Geodude
    from geodude.trajectory import Trajectory


class ArmController(Protocol):
    """Protocol for arm controllers within an execution context.

    Provides high-level operations like grasp and release that combine
    gripper control with grasp manager updates.
    """

    def grasp(self, object_name: str) -> str | None:
        """Close gripper and attempt to grasp object.

        Args:
            object_name: Name of the object to grasp

        Returns:
            Name of grasped object if successful, None otherwise
        """
        ...

    def release(self, object_name: str | None = None) -> None:
        """Open gripper and release any held object.

        Args:
            object_name: Optional specific object to release. If None,
                        releases all objects held by this arm.
        """
        ...


class ExecutionContext(Protocol):
    """Protocol for execution contexts (simulation or hardware).

    An execution context provides a unified interface for executing
    trajectories and managing the robot state. It handles:
    - Routing trajectories to the correct entity (arm, base)
    - Synchronizing state (simulation forward kinematics, sensor reads)
    - Providing arm controllers for grasp/release operations
    """

    def execute(self, item: "Trajectory | PlanResult") -> bool:
        """Execute a trajectory or plan result.

        For PlanResult, executes all trajectories in order (base first,
        then arm). Routes each trajectory to the appropriate entity
        based on its entity field.

        Args:
            item: Single trajectory or PlanResult with multiple trajectories

        Returns:
            True if execution completed successfully
        """
        ...

    def sync(self) -> None:
        """Synchronize state with the environment.

        In simulation: calls mj_forward and viewer.sync
        In hardware: reads sensor values
        """
        ...

    def is_running(self) -> bool:
        """Check if the context is still active.

        In simulation: returns True if viewer is open
        In hardware: returns True if no e-stop triggered
        """
        ...

    def arm(self, name: str) -> ArmController:
        """Get arm controller for grasp/release operations.

        Args:
            name: Arm identifier ("left" or "right", or full name
                 "left_arm"/"right_arm")

        Returns:
            ArmController for the specified arm
        """
        ...


class SimArmController:
    """Arm controller for simulation context.

    Provides grasp/release operations that work in both kinematic
    and physics simulation modes.
    """

    def __init__(
        self,
        arm: "Arm",
        context: "SimContext",
    ):
        """Initialize arm controller.

        Args:
            arm: Arm instance
            context: Parent simulation context
        """
        self._arm = arm
        self._context = context

    def grasp(self, object_name: str) -> str | None:
        """Close gripper and attempt to grasp object.

        In physics mode, uses the physics controller for realistic gripper
        closing. In kinematic mode, directly closes the gripper.

        Args:
            object_name: Name of the object to grasp

        Returns:
            Name of grasped object if successful, None otherwise
        """
        gripper = self._arm.gripper
        if gripper is None:
            return None

        # Set candidate objects for detection
        gripper.set_candidate_objects([object_name])

        # Get arm side for physics controller
        side = self._arm.side

        if self._context._physics and self._context._controller is not None:
            # Physics mode: use controller for realistic gripper motion
            grasped = self._context._controller.close_gripper(side, steps=200)
        else:
            # Kinematic mode: direct gripper close and assume grasp succeeds
            gripper.kinematic_close()
            # In kinematic mode, we assume the grasp succeeds if the object
            # is in the candidate list (no physics contact detection)
            grasped = object_name

        if grasped:
            # Update grasp manager
            self._arm.grasp_manager.mark_grasped(grasped, side)
            # Attach to gripper body for kinematic tracking
            attach_body = f"{side}_ur5e/gripper/right_follower"
            self._arm.grasp_manager.attach_object(grasped, attach_body)

        self._context.sync()
        return grasped

    def release(self, object_name: str | None = None) -> None:
        """Open gripper and release held object(s).

        Args:
            object_name: Specific object to release, or None to release all
        """
        gripper = self._arm.gripper
        if gripper is None:
            return

        side = self._arm.side

        # Determine which objects to release
        if object_name is not None:
            objects_to_release = [object_name]
        else:
            objects_to_release = list(self._arm.grasp_manager.get_grasped_by(side))

        # Release in grasp manager
        for obj in objects_to_release:
            self._arm.grasp_manager.mark_released(obj)
            self._arm.grasp_manager.detach_object(obj)

        if self._context._physics and self._context._controller is not None:
            # Physics mode: use controller for realistic gripper motion
            self._context._controller.open_gripper(side, steps=100)
        else:
            # Kinematic mode: direct gripper open
            gripper.kinematic_open()

        self._context.sync()


class SimContext:
    """Simulation execution context using MuJoCo.

    Provides a unified interface for executing trajectories in MuJoCo
    simulation, with support for both kinematic and physics modes.

    Example:
        with robot.sim(physics=True) as ctx:
            result = robot.plan_to_tsr(grasp_tsr)
            ctx.execute(result)
            ctx.arm("right").grasp("can_0")

            while ctx.is_running():
                # Main loop
                ctx.sync()
    """

    def __init__(
        self,
        robot: "Geodude",
        viewer=None,
        physics: bool = True,
    ):
        """Initialize simulation context.

        Args:
            robot: Geodude robot instance
            viewer: Optional MuJoCo viewer (created internally if None and
                   context is used as context manager)
            physics: If True, use physics simulation. If False, use kinematic
                    execution (perfect tracking, no dynamics).
        """
        self._robot = robot
        self._viewer = viewer
        self._physics = physics
        self._controller = None
        self._executors: dict[str, object] = {}
        self._arm_controllers: dict[str, SimArmController] = {}
        self._owns_viewer = False

    def __enter__(self) -> "SimContext":
        """Enter context manager, optionally creating viewer."""
        # Create viewer if not provided
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(
                self._robot.model, self._robot.data
            )
            self._owns_viewer = True

            # Set preferred camera view
            self._viewer.cam.azimuth = -90
            self._viewer.cam.elevation = -26.5
            self._viewer.cam.distance = 2.96
            self._viewer.cam.lookat[:] = [0.188, 0.001, 1.141]

        # Initialize executors
        self._setup_executors()

        # Set as active context on robot (for primitives)
        self._robot._active_context = self

        # Initial sync
        self.sync()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, cleaning up resources."""
        # Clear active context on robot
        self._robot._active_context = None

        if self._owns_viewer and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._controller = None
        self._executors.clear()
        self._arm_controllers.clear()
        return False

    def _setup_executors(self) -> None:
        """Set up executors based on physics mode."""
        from geodude.executor import KinematicExecutor, RobotPhysicsController

        if self._physics:
            # Physics mode: use RobotPhysicsController
            self._controller = RobotPhysicsController(
                self._robot, viewer=self._viewer
            )
            self._executors["left_arm"] = self._controller.get_executor("left")
            self._executors["right_arm"] = self._controller.get_executor("right")
        else:
            # Kinematic mode: use KinematicExecutor
            for arm_name in ["left", "right"]:
                arm = getattr(self._robot, f"{arm_name}_arm")
                self._executors[f"{arm_name}_arm"] = KinematicExecutor(
                    self._robot.model,
                    self._robot.data,
                    arm.joint_qpos_indices,
                    viewer=self._viewer,
                    grasp_manager=self._robot.grasp_manager,
                )

    def execute(self, item: "Trajectory | PlanResult") -> bool:
        """Execute a trajectory or plan result.

        Routes trajectories to the appropriate executor based on entity.

        Args:
            item: Trajectory or PlanResult to execute

        Returns:
            True if execution completed successfully
        """
        from geodude.planning import PlanResult
        from geodude.trajectory import Trajectory

        if isinstance(item, PlanResult):
            # Execute all trajectories in order
            for traj in item.trajectories:
                if not self._execute_trajectory(traj):
                    return False
            return True
        elif isinstance(item, Trajectory):
            return self._execute_trajectory(item)
        else:
            raise TypeError(f"Cannot execute {type(item)}")

    def _execute_trajectory(self, trajectory: "Trajectory") -> bool:
        """Execute a single trajectory."""
        entity = trajectory.entity

        if entity is None:
            raise ValueError("Trajectory has no entity set")

        # Route to appropriate executor
        if entity in self._executors:
            return self._executors[entity].execute(trajectory)
        elif entity.endswith("_base"):
            # Base trajectories - execute via base.move_to
            side = "left" if "left" in entity else "right"
            base = getattr(self._robot, f"{side}_base")
            if base is not None and trajectory.num_waypoints > 0:
                target_height = trajectory.positions[-1, 0]
                base.move_to(
                    target_height,
                    viewer=self._viewer,
                    executor_type="physics" if self._physics else "kinematic",
                )
            return True
        else:
            raise ValueError(f"Unknown entity: {entity}")

    def sync(self) -> None:
        """Synchronize state with simulation."""
        mujoco.mj_forward(self._robot.model, self._robot.data)
        if self._viewer is not None:
            self._viewer.sync()

    def is_running(self) -> bool:
        """Check if viewer is still open."""
        if self._viewer is None:
            return True  # No viewer, assume running
        return self._viewer.is_running()

    def arm(self, name: str) -> SimArmController:
        """Get arm controller for grasp/release operations.

        Args:
            name: "left", "right", "left_arm", or "right_arm"

        Returns:
            SimArmController for the specified arm
        """
        # Normalize name
        if name in ("left", "left_arm"):
            key = "left"
        elif name in ("right", "right_arm"):
            key = "right"
        else:
            raise ValueError(f"Unknown arm: {name}")

        if key not in self._arm_controllers:
            arm = getattr(self._robot, f"{key}_arm")
            self._arm_controllers[key] = SimArmController(arm, self)

        return self._arm_controllers[key]

    @property
    def viewer(self):
        """Get the MuJoCo viewer."""
        return self._viewer

    @property
    def robot(self) -> "Geodude":
        """Get the robot instance."""
        return self._robot
