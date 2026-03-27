"""Geodude-specific behavior tree nodes and subtrees.

Extends mj_manipulator.bt with bimanual coordination, VentionBase
control, and automatic TSR generation from prl_assets geometry.
"""

from geodude.bt.nodes import GenerateDropZone, GenerateGrasps, GeneratePlaceTSRs
from geodude.bt.subtrees import (
    geodude_pickup,
    geodude_place,
)

__all__ = [
    "GenerateGrasps",
    "GeneratePlaceTSRs",
    "GenerateDropZone",  # deprecated alias
    "geodude_pickup",
    "geodude_place",
]
