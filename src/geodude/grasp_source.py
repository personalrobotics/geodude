# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""GraspSource for Geodude — delegates to mj_manipulator's PrlAssetsGraspSource.

Thin wrapper that constructs the shared PrlAssetsGraspSource with
Geodude's model, data, grasp manager, arms, and environment registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geodude.robot import Geodude


class GeodueGraspSource:
    """GraspSource backed by prl_assets geometry.

    Delegates to mj_manipulator's PrlAssetsGraspSource, which handles
    all TSR generation from prl_assets object metadata.
    """

    def __init__(self, robot: Geodude) -> None:
        from mj_manipulator.grasp_sources.prl_assets import PrlAssetsGraspSource

        registry = robot.env.registry if hasattr(robot.env, "registry") else None
        self._inner = PrlAssetsGraspSource(
            robot.model,
            robot.data,
            robot.grasp_manager,
            robot.arms,
            registry=registry,
        )

    def get_grasps(self, object_name: str, hand_type: str) -> list:
        return self._inner.get_grasps(object_name, hand_type)

    def get_placements(self, destination: str, object_name: str) -> list:
        return self._inner.get_placements(destination, object_name)

    def get_graspable_objects(self) -> list[str]:
        return self._inner.get_graspable_objects()

    def get_place_destinations(self, object_name: str) -> list[str]:
        return self._inner.get_place_destinations(object_name)
