"""Main Geodude robot interface."""

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from geodude.arm import Arm
from geodude.config import GeodudConfig
from geodude.grasp_manager import GraspManager
from geodude.vention_base import VentionBase


class Geodude:
    """High-level interface for the Geodude bimanual robot.

    Provides:
    - Easy access to left and right arms
    - Grasp state management
    - Environment interaction
    - Named configurations

    Example:
        robot = Geodude()
        robot.right_arm.go_to('home')
        robot.right_arm.pick('cup')
        robot.right_arm.place('cup', on='table')
    """

    def __init__(self, config: GeodudConfig | None = None):
        """Initialize the Geodude robot.

        Args:
            config: Robot configuration. If None, uses default configuration.
        """
        self.config = config or GeodudConfig.default()

        # Load MuJoCo model
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"MuJoCo model not found: {self.config.model_path}\n"
                "Make sure geodude_assets is available."
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.config.model_path))
        self.data = mujoco.MjData(self.model)

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
    def named_poses(self) -> dict[str, dict[str, list[float]]]:
        """Named pose configurations."""
        return self.config.named_poses

    def go_to(self, pose_name: str, speed: float = 1.0) -> bool:
        """Move both arms to a named configuration.

        Args:
            pose_name: Name of the configuration (e.g., 'home', 'ready')
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
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Clear grasp state
        for obj in list(self.grasp_manager.grasped.keys()):
            self.grasp_manager.mark_released(obj)

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
