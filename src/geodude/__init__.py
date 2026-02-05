"""Geodude: High-level API for bimanual robot manipulation."""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies during testing
def __getattr__(name):
    if name == "Geodude":
        from geodude.robot import Geodude
        return Geodude
    if name == "VentionBase":
        from geodude.vention_base import VentionBase
        return VentionBase
    if name == "VentionBaseConfig":
        from geodude.config import VentionBaseConfig
        return VentionBaseConfig
    # TSR utilities
    if name in (
        "create_top_grasp_tsr",
        "create_side_grasp_tsr",
        "create_place_tsr",
        "create_lift_tsr",
        "create_retract_tsr",
        "create_approach_tsr",
    ):
        from geodude import tsr_utils
        return getattr(tsr_utils, name)
    # Parallel planning utilities
    if name in ("plan_first_success", "plan_best_of_all", "plan_with_base_heights"):
        from geodude import parallel
        return getattr(parallel, name)
    # Planning result types
    if name == "PlanResult":
        from geodude.planning import PlanResult
        return PlanResult
    # Trajectory
    if name == "Trajectory":
        from geodude.trajectory import Trajectory
        return Trajectory
    # Entity config
    if name == "EntityConfig":
        from geodude.config import EntityConfig
        return EntityConfig
    # Planning defaults
    if name == "PlanningDefaults":
        from geodude.config import PlanningDefaults
        return PlanningDefaults
    # Execution context
    if name == "SimContext":
        from geodude.execution import SimContext
        return SimContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Geodude",
    "VentionBase",
    "VentionBaseConfig",
    "EntityConfig",
    "PlanningDefaults",
    "Trajectory",
    "PlanResult",
    "SimContext",
    # TSR utilities
    "create_top_grasp_tsr",
    "create_side_grasp_tsr",
    "create_place_tsr",
    "create_lift_tsr",
    "create_retract_tsr",
    "create_approach_tsr",
    # Parallel planning utilities
    "plan_first_success",
    "plan_best_of_all",
    "plan_with_base_heights",
]
