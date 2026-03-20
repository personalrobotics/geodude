"""Robot configuration for Geodude."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Try to import geodude_assets for model paths, fall back to None if not installed
try:
    from geodude_assets import get_model_path
except ImportError:
    get_model_path = None


# ---------------------------------------------------------------------------
# Arm specification (Geodude-specific, used to create mj_manipulator Arms)
# ---------------------------------------------------------------------------


@dataclass
class GeodudeArmSpec:
    """Specification for one Geodude arm.

    Stores the Geodude-specific naming/prefixes needed to create an
    mj_manipulator Arm with the correct joint names, end-effector site,
    and gripper configuration.
    """

    prefix: str  # e.g., "left_ur5e" or "right_ur5e"
    ee_site: str = ""  # MuJoCo site name for end-effector
    gripper_prefix: str = ""  # e.g., "left_ur5e/gripper/"
    hand_type: str = "robotiq_2f_140"  # For affordance matching


# ---------------------------------------------------------------------------
# Vention base (Geodude hardware)
# ---------------------------------------------------------------------------


@dataclass
class VentionKinematicLimits:
    """Velocity and acceleration limits for Vention linear actuator.

    Based on Vention MachineMotion hardware specs:
    - Motor: NEMA 34 stepper servo with 7.2Nm torque
    - Actuator: Enclosed lead screw (50mm/s default safe limit)
    - Range: 0-0.5m (500mm)
    """

    velocity: float  # m/s
    acceleration: float  # m/s²

    @classmethod
    def default(cls) -> VentionKinematicLimits:
        """Default limits: 100 mm/s velocity, 200 mm/s² acceleration."""
        return cls(velocity=0.1, acceleration=0.2)


@dataclass
class VentionBaseConfig:
    """Configuration for a Vention linear actuator."""

    name: str  # e.g., "left_base"
    joint_name: str  # MuJoCo joint name
    actuator_name: str = ""  # MuJoCo actuator name
    height_range: tuple[float, float] = (0.0, 0.5)  # meters (min, max)
    collision_check_resolution: float = 0.01  # meters between collision checks
    kinematic_limits: VentionKinematicLimits = field(
        default_factory=VentionKinematicLimits.default
    )


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------


@dataclass
class DebugConfig:
    """Debug logging configuration.

    Controls which subsystems emit debug-level log messages.
    Use GEODUDE_DEBUG=subsystem1,subsystem2 or GEODUDE_DEBUG=all.
    """

    planning: bool = False
    primitives: bool = False
    affordances: bool = False

    show_timestamps: bool = True
    show_module: bool = True

    def enable_all(self) -> None:
        self.planning = True
        self.primitives = True
        self.affordances = True

    def get_enabled_subsystems(self) -> list[str]:
        return [s for s in ("planning", "primitives", "affordances") if getattr(self, s)]

    @classmethod
    def from_env(cls) -> DebugConfig:
        """Create config from GEODUDE_DEBUG environment variable."""
        config = cls()
        debug_env = os.environ.get("GEODUDE_DEBUG", "")
        if debug_env:
            if debug_env.lower() == "all":
                config.enable_all()
            else:
                for s in debug_env.split(","):
                    s = s.strip()
                    if s and hasattr(config, s):
                        setattr(config, s, True)
        return config


_SUBSYSTEM_LOGGERS = {
    "planning": "geodude.robot",
    "primitives": "geodude.primitives",
    "affordances": "geodude.affordances",
}


def setup_logging(config: DebugConfig | None = None) -> None:
    """Configure geodude loggers based on debug config."""
    if config is None:
        config = DebugConfig.from_env()

    root_logger = logging.getLogger("geodude")
    root_logger.propagate = False

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        fmt_parts = []
        if config.show_timestamps:
            fmt_parts.append("%(asctime)s")
        fmt_parts.append("%(levelname)s")
        if config.show_module:
            fmt_parts.append("[%(name)s]")
        fmt_parts.append("%(message)s")
        handler.setFormatter(logging.Formatter(" - ".join(fmt_parts)))
        root_logger.addHandler(handler)

    root_logger.setLevel(logging.WARNING)

    for subsystem in config.get_enabled_subsystems():
        logger_name = _SUBSYSTEM_LOGGERS.get(subsystem)
        if logger_name:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Top-level Geodude configuration
# ---------------------------------------------------------------------------

# UR5e joint name suffixes (combined with arm prefix)
_UR5E_JOINT_SUFFIXES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@dataclass
class GeodudConfig:
    """Full robot configuration."""

    model_path: Path
    left_arm: GeodudeArmSpec
    right_arm: GeodudeArmSpec
    left_base: VentionBaseConfig | None = None
    right_base: VentionBaseConfig | None = None
    named_poses: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    debug: DebugConfig = field(default_factory=DebugConfig.from_env)

    def joint_names(self, arm_spec: GeodudeArmSpec) -> list[str]:
        """Get prefixed UR5e joint names for an arm spec."""
        return [f"{arm_spec.prefix}/{j}" for j in _UR5E_JOINT_SUFFIXES]

    @classmethod
    def default(cls) -> GeodudConfig:
        """Create default configuration for Geodude with Robotiq grippers."""
        if get_model_path is None:
            raise ImportError(
                "geodude_assets package not found. Install it with:\n"
                "  uv add geodude_assets"
            )
        return cls(
            model_path=get_model_path(),
            left_arm=GeodudeArmSpec(
                prefix="left_ur5e",
                ee_site="left_ur5e/gripper_attachment_site",
                gripper_prefix="left_ur5e/gripper/",
            ),
            right_arm=GeodudeArmSpec(
                prefix="right_ur5e",
                ee_site="right_ur5e/gripper_attachment_site",
                gripper_prefix="right_ur5e/gripper/",
            ),
            left_base=VentionBaseConfig(
                name="left_base",
                joint_name="left_arm_linear_vention",
                actuator_name="left_linear_actuator",
            ),
            right_base=VentionBaseConfig(
                name="right_base",
                joint_name="right_arm_linear_vention",
                actuator_name="right_linear_actuator",
            ),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> GeodudConfig:
        """Load configuration from YAML file."""
        import yaml

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
            left_arm=GeodudeArmSpec(**data["left_arm"]),
            right_arm=GeodudeArmSpec(**data["right_arm"]),
            left_base=left_base,
            right_base=right_base,
            named_poses=data.get("named_poses", {}),
        )
