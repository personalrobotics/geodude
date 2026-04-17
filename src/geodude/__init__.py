# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Geodude: High-level API for bimanual robot manipulation.

Built on mj_manipulator for generic arm control, planning, and execution.
"""

__version__ = "3.0.0"


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    # Geodude-specific
    if name == "Geodude":
        from geodude.robot import Geodude

        return Geodude
    if name == "GeodudConfig":
        from geodude.config import GeodudConfig

        return GeodudConfig
    if name == "VentionBase":
        from geodude.vention_base import VentionBase

        return VentionBase
    if name == "VentionBaseConfig":
        from geodude.config import VentionBaseConfig

        return VentionBaseConfig
    if name == "DebugConfig":
        from geodude.config import DebugConfig

        return DebugConfig
    if name == "setup_logging":
        from geodude.config import setup_logging

        return setup_logging
    # Re-exports from mj_manipulator (convenience)
    _MJ_REEXPORTS = {
        "Arm": ("mj_manipulator", "Arm"),
        "Trajectory": ("mj_manipulator", "Trajectory"),
        "PlanResult": ("mj_manipulator", "PlanResult"),
        "SimContext": ("mj_manipulator", "SimContext"),
        "GraspManager": ("mj_manipulator", "GraspManager"),
    }
    if name in _MJ_REEXPORTS:
        mod, attr = _MJ_REEXPORTS[name]
        import importlib

        m = importlib.import_module(mod)
        return getattr(m, attr)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Geodude-specific
    "Geodude",
    "GeodudConfig",
    "VentionBase",
    "VentionBaseConfig",
    "DebugConfig",
    "setup_logging",
    # Re-exports from mj_manipulator
    "Arm",
    "Trajectory",
    "PlanResult",
    "SimContext",
    "GraspManager",
]
