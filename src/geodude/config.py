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
class VentionBaseConfig:
    """Configuration for a Vention linear actuator."""

    name: str  # "left" or "right"
    joint_name: str  # MuJoCo joint name
    actuator_name: str  # MuJoCo actuator name
    height_range: tuple[float, float] = (0.0, 0.5)  # meters (min, max)
    collision_check_resolution: float = 0.01  # meters between collision checks


@dataclass
class GeodudConfig:
    """Full robot configuration."""

    model_path: Path
    left_arm: ArmConfig
    right_arm: ArmConfig
    left_base: VentionBaseConfig | None = None
    right_base: VentionBaseConfig | None = None
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
            left_base=VentionBaseConfig(
                name="left",
                joint_name="left_arm_linear_vention",
                actuator_name="left_linear_actuator",
            ),
            right_base=VentionBaseConfig(
                name="right",
                joint_name="right_arm_linear_vention",
                actuator_name="right_linear_actuator",
            ),
            named_poses={
                "home": {
                    # Arms tucked in, elbows bent up
                    "left": [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0],
                    "right": [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0],
                },
                "ready": {
                    # Arms extended forward, gripper pointing down, ready to work
                    # shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3
                    "left": [-1.57, -1.2, 0.8, -1.17, -1.57, 0],
                    "right": [-1.57, -1.2, 0.8, -1.17, -1.57, 0],
                },
            },
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "GeodudConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        left_base = None
        if "left_base" in data:
            left_base = VentionBaseConfig(**data["left_base"])

        right_base = None
        if "right_base" in data:
            right_base = VentionBaseConfig(**data["right_base"])

        return cls(
            model_path=Path(data["model_path"]),
            left_arm=ArmConfig(**data["left_arm"]),
            right_arm=ArmConfig(**data["right_arm"]),
            left_base=left_base,
            right_base=right_base,
            named_poses=data.get("named_poses", {}),
        )
