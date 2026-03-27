"""Geodude-specific behavior tree nodes and subtrees.

Extends mj_manipulator.bt with bimanual coordination, VentionBase
control, and automatic TSR generation from prl_assets geometry.
"""

from geodude.bt.nodes import GenerateGrasps, GeneratePlaceTSRs
from geodude.bt.subtrees import (
    geodude_pickup,
    geodude_place,
)

__all__ = [
    "GenerateGrasps",
    "GeneratePlaceTSRs",
    "geodude_pickup",
    "geodude_place",
]
