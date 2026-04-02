"""Geodude teleop panel — thin wrapper around mj_viser.TeleopPanel.

Supplies Robotiq gripper body prefix and arm reference for left/right arms.
Wired into the console's Viser tab layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geodude.robot import Geodude
    from mj_manipulator.sim_context import SimContext
    from mj_viser import MujocoViewer, TeleopPanel


def create_teleop_panel(
    robot: Geodude,
    ctx: SimContext,
    side: str = "right",
) -> TeleopPanel:
    """Create a TeleopPanel for a Geodude arm.

    Args:
        robot: Geodude instance.
        ctx: Active execution context.
        side: "left" or "right".

    Returns:
        TeleopPanel ready for setup() in a Viser tab.
    """
    from mj_manipulator.teleop import TeleopController
    from mj_viser import TeleopPanel

    arm = robot._resolve_arm(side)
    controller = TeleopController(arm, ctx)

    return TeleopPanel(
        arm=arm,
        controller=controller,
        model=robot.model,
        data=robot.data,
        gripper_body_prefix=f"{side}_ur5e/gripper/",
        arm_label=f"{side.title()} Arm",
    )
