"""Gripper control and grasp detection."""

import mujoco
import numpy as np

from geodude.grasp_manager import GraspManager, detect_grasped_object


class Gripper:
    """Controls a gripper and detects grasp state.

    Handles:
    - Opening and closing the gripper via MuJoCo actuator
    - Detecting when an object is grasped (via contacts or geometry)
    - Updating grasp state in GraspManager
    - Kinematic attachment for non-physics execution

    For physics-based execution, use open()/close() which simulate gripper motion.
    For kinematic execution, use kinematic_open()/kinematic_close() which set
    gripper position directly and use geometric grasp detection.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_name: str,
        actuator_name: str,
        gripper_body_names: list[str],
        grasp_manager: GraspManager,
        ctrl_open: float = 0.0,
        ctrl_closed: float = 255.0,
        gripper_site_name: str | None = None,
    ):
        """Initialize gripper controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            arm_name: Name of arm this gripper belongs to ("left" or "right")
            actuator_name: Name of gripper actuator in MuJoCo
            gripper_body_names: Names of gripper bodies for contact detection
            grasp_manager: GraspManager for tracking grasp state
            ctrl_open: Control value for open gripper
            ctrl_closed: Control value for closed gripper
            gripper_site_name: Name of gripper site for pose queries and attachment
        """
        self.model = model
        self.data = data
        self.arm_name = arm_name
        self.gripper_body_names = gripper_body_names
        self.grasp_manager = grasp_manager
        self.ctrl_open = ctrl_open
        self.ctrl_closed = ctrl_closed
        self.gripper_site_name = gripper_site_name

        # Get actuator index
        if actuator_name:
            self.actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if self.actuator_id == -1:
                raise ValueError(f"Actuator '{actuator_name}' not found in model")
        else:
            self.actuator_id = None

        self._candidate_objects: list[str] | None = None
        # Grasp detection threshold for kinematic mode (distance in meters)
        self.kinematic_grasp_threshold = 0.05

    def set_candidate_objects(self, objects: list[str] | None) -> None:
        """Set the list of objects that could be grasped.

        This helps grasp detection by limiting which bodies are considered.

        Args:
            objects: List of object body names, or None to consider all
        """
        self._candidate_objects = objects

    def open(self, steps: int = 100) -> None:
        """Open the gripper.

        Args:
            steps: Number of simulation steps to run after commanding open
        """
        if self.actuator_id is None:
            return

        # Release any currently grasped objects first
        for obj in self.grasp_manager.get_grasped_by(self.arm_name):
            self.grasp_manager.mark_released(obj)

        # Command gripper open
        self.data.ctrl[self.actuator_id] = self.ctrl_open

        # Step simulation to let gripper open
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def close(self, steps: int = 100) -> str | None:
        """Close the gripper and detect grasp.

        Args:
            steps: Number of simulation steps to run after commanding close

        Returns:
            Name of grasped object, or None if nothing grasped
        """
        if self.actuator_id is None:
            return None

        # Command gripper closed
        self.data.ctrl[self.actuator_id] = self.ctrl_closed

        # Step simulation to let gripper close
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

        # Detect what we grasped
        grasped = detect_grasped_object(
            self.model,
            self.data,
            self.gripper_body_names,
            self._candidate_objects,
        )

        # Update grasp state
        if grasped:
            self.grasp_manager.mark_grasped(grasped, self.arm_name)

        return grasped

    def get_position(self) -> float:
        """Get current gripper position (0=open, 1=closed)."""
        if self.actuator_id is None:
            return 0.0

        ctrl = self.data.ctrl[self.actuator_id]
        return (ctrl - self.ctrl_open) / (self.ctrl_closed - self.ctrl_open)

    def set_position(self, position: float) -> None:
        """Set gripper position (0=open, 1=closed)."""
        if self.actuator_id is None:
            return

        ctrl = self.ctrl_open + position * (self.ctrl_closed - self.ctrl_open)
        self.data.ctrl[self.actuator_id] = ctrl

    @property
    def is_holding(self) -> bool:
        """Check if gripper is currently holding an object."""
        return len(self.grasp_manager.get_grasped_by(self.arm_name)) > 0

    @property
    def held_object(self) -> str | None:
        """Get the name of the currently held object, or None."""
        held = self.grasp_manager.get_grasped_by(self.arm_name)
        return held[0] if held else None

    def kinematic_close(self) -> str | None:
        """Close gripper kinematically (no physics) and detect grasp geometrically.

        Sets the gripper to closed position directly without simulation.
        Uses geometric proximity to detect if an object is grasped, then
        attaches it for kinematic manipulation.

        Returns:
            Name of grasped object, or None if nothing within grasp distance
        """
        # Set gripper to closed position
        if self.actuator_id is not None:
            self.data.ctrl[self.actuator_id] = self.ctrl_closed

        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)

        # Detect grasp geometrically
        grasped = self._detect_grasp_geometric()

        if grasped:
            # Update grasp state (collision groups)
            self.grasp_manager.mark_grasped(grasped, self.arm_name)

            # Attach object for kinematic movement
            attach_body = self.gripper_body_names[0] if self.gripper_body_names else None
            if attach_body:
                self.grasp_manager.attach_object(grasped, attach_body)

        return grasped

    def kinematic_open(self) -> None:
        """Open gripper kinematically (no physics) and release any attached object.

        Sets the gripper to open position directly without simulation.
        Detaches and releases any currently held object.
        """
        # Release any currently grasped objects
        for obj in self.grasp_manager.get_grasped_by(self.arm_name):
            self.grasp_manager.detach_object(obj)
            self.grasp_manager.mark_released(obj)

        # Set gripper to open position
        if self.actuator_id is not None:
            self.data.ctrl[self.actuator_id] = self.ctrl_open

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

    def _detect_grasp_geometric(self) -> str | None:
        """Detect grasp using geometric proximity (no physics contacts).

        Checks if any candidate object's center is close enough to the
        gripper position to be considered grasped.

        Returns:
            Name of closest object within threshold, or None
        """
        if not self._candidate_objects:
            return None

        # Get gripper position
        gripper_pos = self._get_gripper_position()
        if gripper_pos is None:
            return None

        closest_obj = None
        closest_dist = float("inf")

        for obj_name in self._candidate_objects:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name
            )
            if body_id == -1:
                continue

            obj_pos = self.data.xpos[body_id]
            dist = np.linalg.norm(obj_pos - gripper_pos)

            if dist < self.kinematic_grasp_threshold and dist < closest_dist:
                closest_obj = obj_name
                closest_dist = dist

        return closest_obj

    def _get_gripper_position(self) -> np.ndarray | None:
        """Get the gripper's center position.

        Uses gripper site if available, otherwise averages gripper body positions.

        Returns:
            3D position array, or None if not available
        """
        # Try site first
        if self.gripper_site_name:
            site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self.gripper_site_name
            )
            if site_id != -1:
                return self.data.site_xpos[site_id].copy()

        # Fall back to average of gripper body positions
        if not self.gripper_body_names:
            return None

        positions = []
        for body_name in self.gripper_body_names:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id != -1:
                positions.append(self.data.xpos[body_id])

        if not positions:
            return None

        return np.mean(positions, axis=0)
