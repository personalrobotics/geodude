"""Status HUD overlay for the Viser browser viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import viser
from mj_viser import MujocoViewer, PanelBase

if TYPE_CHECKING:
    from geodude.robot import Geodude


class StatusHud(PanelBase):
    """Compact status overlay on the 3D viewport.

    Shows held objects, mode, and last action result.
    """

    def __init__(self, robot: Geodude, mode: str) -> None:
        self._robot = robot
        self._mode = mode

    def name(self) -> str:
        return "StatusHud"

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        self._viewer = viewer
        # Initial HUD
        viewer.set_hud("status", self._build_status(), "bottom-left")

    def on_sync(self, viewer: MujocoViewer) -> None:
        viewer.set_hud("status", self._build_status(), "bottom-left")

    def _build_status(self) -> str:
        robot = self._robot
        import numpy as np

        parts = []
        for side, arm in [("L", robot._left_arm), ("R", robot._right_arm)]:
            # F/T magnitude
            wrench = arm.get_ft_wrench()
            if np.isnan(wrench[0]):
                force_str = "—"
            else:
                force_mag = float(np.linalg.norm(wrench[:3]))
                force_str = f"{force_mag:.0f}N"

            # Held object
            held = robot.grasp_manager.get_grasped_by(
                "left" if side == "L" else "right",
            )
            held_str = held[0] if held else "—"

            parts.append(f"<b>{side}</b>: [{force_str}] {held_str}")

        return (
            " &nbsp;|&nbsp; ".join(parts)
            + f" &nbsp;|&nbsp; {self._mode}"
        )
