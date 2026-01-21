"""Gripper control and grasp detection."""

import mujoco
import numpy as np

from geodude.grasp_manager import GraspManager, detect_grasped_object


class Gripper:
    """Controls a gripper and detects grasp state.

    Handles:
    - Opening and closing the gripper via MuJoCo actuator
    - Detecting when an object is grasped (via contacts)
    - Updating grasp state in GraspManager
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
        """
        self.model = model
        self.data = data
        self.arm_name = arm_name
        self.gripper_body_names = gripper_body_names
        self.grasp_manager = grasp_manager
        self.ctrl_open = ctrl_open
        self.ctrl_closed = ctrl_closed

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
