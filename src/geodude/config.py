"""Robot configuration for Geodude."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Try to import geodude_assets for model paths, fall back to None if not installed
try:
    from geodude_assets import get_model_path
except ImportError:
    get_model_path = None


@dataclass
class TrackingThresholds:
    """Error thresholds for trajectory execution.

    Execution aborts if any single joint exceeds the max_error threshold.
    Per-joint thresholds may be added in the future.
    """
    max_error: float = np.deg2rad(10.0)  # Abort if any joint error exceeds this

    @classmethod
    def default(cls) -> "TrackingThresholds":
        """Default thresholds."""
        return cls()


@dataclass
class KinematicLimits:
    """Velocity and acceleration limits for trajectory planning.

    These limits come from the robot manufacturer's specifications (UR5e datasheet)
    and are used by TOPP-RA for trajectory retiming. They are NOT physics parameters
    in MuJoCo - the simulation doesn't enforce them, but planners must respect them
    to ensure trajectories are executable on real hardware.

    Note: Position limits are in the MuJoCo model XML (joint range attributes)
    and can be read via model.jnt_range. Velocity/acceleration limits are not
    supported by MuJoCo XML, so we define them here.

    Recommended speed scales:
    - 10%: Conservative for initial simulation testing
    - 50%: Standard for real robot operation (safe and efficient) - DEFAULT
    - 75-100%: High-speed operation
    """
    velocity: np.ndarray  # rad/s per joint
    acceleration: np.ndarray  # rad/s² per joint

    @classmethod
    def ur5e_default(cls, vel_scale: float = 0.5, acc_scale: float = 0.5) -> "KinematicLimits":
        """Create default UR5e limits from datasheet.

        Args:
            vel_scale: Safety scaling factor for velocity (default 50% of max)
            acc_scale: Safety scaling factor for acceleration (default 50% of max)

        Returns:
            KinematicLimits with scaled UR5e specifications
        """
        # UR5e official velocity limits from datasheet
        base_vel = np.array([
            3.14,  # shoulder_pan: ±180°/s
            3.14,  # shoulder_lift: ±180°/s
            3.14,  # elbow: ±180°/s
            6.28,  # wrist_1: ±360°/s
            6.28,  # wrist_2: ±360°/s
            6.28,  # wrist_3: ±360°/s
        ])

        # UR5e acceleration limits (conservative defaults)
        base_acc = np.array([
            2.5, 2.5, 2.5,  # shoulder/elbow
            5.0, 5.0, 5.0,  # wrist joints (can accelerate faster)
        ])

        return cls(
            velocity=base_vel * vel_scale,
            acceleration=base_acc * acc_scale,
        )


@dataclass
class ArmConfig:
    """Configuration for a single arm."""

    name: str
    joint_names: list[str]
    ee_site: str
    gripper_actuator: str
    gripper_bodies: list[str]  # Bodies that are part of gripper (for collision filtering)
    kinematic_limits: KinematicLimits = field(default_factory=KinematicLimits.ur5e_default)
    tracking_thresholds: TrackingThresholds = field(default_factory=TrackingThresholds.default)


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
            # Named poses are loaded from MuJoCo keyframes in the model XML.
            # To add custom poses here, use format:
            #   named_poses={
            #       "pose_name": {
            #           "left": [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3],
            #           "right": [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3],
            #       },
            #   }
            named_poses={},
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "GeodudConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        def parse_arm_config(arm_data: dict) -> ArmConfig:
            """Parse arm config with optional kinematic limits and tracking thresholds."""
            kinematic_limits = KinematicLimits.ur5e_default()
            if "kinematic_limits" in arm_data:
                limits_data = arm_data.pop("kinematic_limits")
                kinematic_limits = KinematicLimits(
                    velocity=np.array(limits_data["velocity"]),
                    acceleration=np.array(limits_data["acceleration"]),
                )
            tracking_thresholds = TrackingThresholds.default()
            if "tracking_thresholds" in arm_data:
                thresh_data = arm_data.pop("tracking_thresholds")
                tracking_thresholds = TrackingThresholds(
                    max_error=np.deg2rad(thresh_data.get("max_error_deg", 10.0)),
                )
            return ArmConfig(
                **arm_data,
                kinematic_limits=kinematic_limits,
                tracking_thresholds=tracking_thresholds,
            )

        left_base = None
        if "left_base" in data:
            left_base = VentionBaseConfig(**data["left_base"])

        right_base = None
        if "right_base" in data:
            right_base = VentionBaseConfig(**data["right_base"])

        return cls(
            model_path=Path(data["model_path"]),
            left_arm=parse_arm_config(data["left_arm"]),
            right_arm=parse_arm_config(data["right_arm"]),
            left_base=left_base,
            right_base=right_base,
            named_poses=data.get("named_poses", {}),
        )
