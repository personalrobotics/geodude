"""Planning result types and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geodude.arm import Arm
    from geodude.trajectory import Trajectory


@dataclass
class PlanResult:
    """Result of a planning operation with optional base motion.

    When planning with `base_heights` parameter, the result includes both
    an arm trajectory and potentially a base trajectory. The base trajectory
    should be executed first to position the arm at the correct height.

    Attributes:
        arm: The arm that will execute the motion
        arm_trajectory: Trajectory for the arm joints
        base_trajectory: Optional trajectory for base height adjustment
        base_height: The base height used for planning (if base_heights was provided)

    Example:
        result = arm.plan_to_pose(pose, base_heights=[0.1, 0.2, 0.3])
        if result:
            # Execute base motion first (if any)
            for traj in result.trajectories:
                # Hardware deployment would execute each trajectory
                pass
    """

    arm: "Arm"
    arm_trajectory: "Trajectory"
    base_trajectory: "Trajectory | None" = None
    base_height: float | None = None

    @property
    def trajectories(self) -> list["Trajectory"]:
        """All trajectories in execution order (base first, then arm)."""
        result = []
        if self.base_trajectory is not None:
            result.append(self.base_trajectory)
        result.append(self.arm_trajectory)
        return result

    @property
    def success(self) -> bool:
        """Whether planning succeeded (arm_trajectory is not None)."""
        return self.arm_trajectory is not None

    @property
    def total_duration(self) -> float:
        """Total duration of all trajectories in seconds."""
        return sum(t.duration for t in self.trajectories)
