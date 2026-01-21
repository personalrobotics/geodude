"""Robot configuration for Geodude."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Try to import geodude_assets for model paths, fall back to None if not installed
try:
    from geodude_assets import get_model_path
except ImportError:
    get_model_path = None


@dataclass
class ArmConfig:
    """Configuration for a single arm."""

    name: str
    joint_names: list[str]
    ee_site: str
    gripper_actuator: str
    gripper_bodies: list[str]  # Bodies that are part of gripper (for collision filtering)


@dataclass
class GeodudConfig:
    """Full robot configuration."""

    model_path: Path
    left_arm: ArmConfig
    right_arm: ArmConfig
    named_poses: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "GeodudConfig":
        """Create default configuration for Geodude with Robotiq grippers.

        Requires geodude_assets package to be installed:
            uv add geodude_assets
        Or install from local clone:
            uv pip install -e path/to/geodude_assets
        """
        if get_model_path is None:
            raise ImportError(
                "geodude_assets package not found. Install it with:\n"
                "  uv add geodude_assets\n"
                "Or from a local clone:\n"
                "  uv pip install -e /path/to/geodude_assets"
            )
        return cls(
            model_path=get_model_path(),
            left_arm=ArmConfig(
                name="left",
                joint_names=[
                    "left_ur5e/shoulder_pan_joint",
                    "left_ur5e/shoulder_lift_joint",
                    "left_ur5e/elbow_joint",
                    "left_ur5e/wrist_1_joint",
                    "left_ur5e/wrist_2_joint",
                    "left_ur5e/wrist_3_joint",
                ],
                ee_site="left_ur5e/gripper_attachment_site",
                gripper_actuator="",  # No gripper in default geodude.xml
                gripper_bodies=[],
            ),
            right_arm=ArmConfig(
                name="right",
                joint_names=[
                    "right_ur5e/shoulder_pan_joint",
                    "right_ur5e/shoulder_lift_joint",
                    "right_ur5e/elbow_joint",
                    "right_ur5e/wrist_1_joint",
                    "right_ur5e/wrist_2_joint",
                    "right_ur5e/wrist_3_joint",
                ],
                ee_site="right_ur5e/gripper_attachment_site",
                gripper_actuator="right_ur5e/gripper/fingers_actuator",
                gripper_bodies=[
                    "right_ur5e/gripper/right_follower",
                    "right_ur5e/gripper/left_follower",
                    "right_ur5e/gripper/right_pad",
                    "right_ur5e/gripper/left_pad",
                ],
            ),
            named_poses={
                "home": {
                    "left": [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0],
                    "right": [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0],
                },
                "ready": {
                    "left": [-1.0, -1.5708, 1.5708, -1.5708, -1.5708, 0],
                    "right": [-2.14, -1.5708, 1.5708, -1.5708, -1.5708, 0],
                },
            },
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "GeodudConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            model_path=Path(data["model_path"]),
            left_arm=ArmConfig(**data["left_arm"]),
            right_arm=ArmConfig(**data["right_arm"]),
            named_poses=data.get("named_poses", {}),
        )
