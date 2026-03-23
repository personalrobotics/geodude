"""Geodude-specific behavior tree leaf nodes.

These nodes handle TSR generation from prl_assets geometry, so the
student API doesn't need to know about TSRs or object dimensions.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import py_trees
from py_trees.common import Access, Status

if TYPE_CHECKING:
    pass


class GenerateGrasps(py_trees.behaviour.Behaviour):
    """Generate side-grasp TSRs from object geometry in prl_assets.

    Reads the object's geometric_properties (type, radius, height) from
    prl_assets and generates Robotiq2F140 grasp templates.

    Reads: ``{ns}/object_name``, ``{ns}/robot``
    Writes: ``{ns}/grasp_tsrs``
    """

    def __init__(self, ns: str = "", name: str = "GenerateGrasps"):
        super().__init__(name)
        self.ns = ns
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(key=f"{ns}/object_name", access=Access.READ)
        self.bb.register_key(key=f"{ns}/robot", access=Access.READ)
        self.bb.register_key(key=f"{ns}/grasp_tsrs", access=Access.WRITE)

    def update(self) -> Status:
        from asset_manager import AssetManager
        from prl_assets import OBJECTS_DIR
        from tsr.hands import Robotiq2F140

        robot = self.bb.get(f"{self.ns}/robot")
        object_name = self.bb.get(f"{self.ns}/object_name")

        # Get object pose from MuJoCo
        try:
            obj_pose = robot.get_object_pose(object_name)
        except ValueError:
            self.feedback_message = f"Object '{object_name}' not found in scene"
            return Status.FAILURE

        # Extract object type from instance name (strip trailing _N)
        match = re.match(r"^(.+?)(?:_\d+)?$", object_name)
        obj_type = match.group(1) if match else object_name

        # Look up geometry from prl_assets
        assets = AssetManager(str(OBJECTS_DIR))
        try:
            gp = assets.get(obj_type)["geometric_properties"]
        except (KeyError, TypeError):
            self.feedback_message = f"No geometry in prl_assets for '{obj_type}'"
            return Status.FAILURE

        # Generate TSRs based on geometry type
        hand = Robotiq2F140()
        if gp.get("type") == "cylinder":
            T_bottom = obj_pose.copy()
            T_bottom[2, 3] -= gp["height"] / 2
            templates = hand.grasp_cylinder_side(gp["radius"], gp["height"])
            tsrs = [t.instantiate(T_bottom) for t in templates]
        else:
            self.feedback_message = f"Unsupported geometry type '{gp.get('type')}' for '{obj_type}'"
            return Status.FAILURE

        self.bb.set(f"{self.ns}/grasp_tsrs", tsrs)
        return Status.SUCCESS


class GenerateDropZone(py_trees.behaviour.Behaviour):
    """Generate drop-zone TSRs for a container from prl_assets geometry.

    Reads the container's geometric_properties (outer_dimensions,
    wall_thickness) and creates a TSR above the opening.

    Reads: ``{ns}/destination``, ``{ns}/robot``
    Writes: ``{ns}/place_tsrs``
    """

    def __init__(self, ns: str = "", name: str = "GenerateDropZone"):
        super().__init__(name)
        self.ns = ns
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(key=f"{ns}/destination", access=Access.READ)
        self.bb.register_key(key=f"{ns}/robot", access=Access.READ)
        self.bb.register_key(key=f"{ns}/place_tsrs", access=Access.WRITE)

    def update(self) -> Status:
        from asset_manager import AssetManager
        from prl_assets import OBJECTS_DIR
        from tsr import TSR

        robot = self.bb.get(f"{self.ns}/robot")
        destination = self.bb.get(f"{self.ns}/destination")

        try:
            dest_pose = robot.get_object_pose(destination)
        except ValueError:
            self.feedback_message = f"Destination '{destination}' not found in scene"
            return Status.FAILURE

        # Extract type
        match = re.match(r"^(.+?)(?:_\d+)?$", destination)
        dest_type = match.group(1) if match else destination

        assets = AssetManager(str(OBJECTS_DIR))
        try:
            meta = assets.get(dest_type)
            gp = meta["geometric_properties"]
            policy = meta.get("policy", {}).get("placement", {})
        except (KeyError, TypeError):
            self.feedback_message = f"No geometry in prl_assets for '{dest_type}'"
            return Status.FAILURE

        if gp.get("type") not in ("open_box", "tote"):
            self.feedback_message = f"Unsupported destination type '{gp.get('type')}' for '{dest_type}'"
            return Status.FAILURE

        outer = gp["outer_dimensions"]
        wall = gp.get("wall_thickness", 0.003)
        margin = policy.get("drop_zone_margin", 0.05)

        hx = (outer[0] / 2) - wall - margin
        hy = (outer[1] / 2) - wall - margin
        drop_z = dest_pose[2, 3] + outer[2] + 0.15  # 15cm above rim

        T0_w = np.array([
            [1,  0,  0, dest_pose[0, 3]],
            [0, -1,  0, dest_pose[1, 3]],
            [0,  0, -1, drop_z],
            [0,  0,  0, 1],
        ], dtype=float)

        Bw = np.zeros((6, 2))
        Bw[0, :] = [-hx, hx]
        Bw[1, :] = [-hy, hy]
        Bw[2, :] = [-0.02, 0.05]
        Bw[5, :] = [-np.pi, np.pi]

        self.bb.set(f"{self.ns}/place_tsrs", [TSR(T0_w=T0_w, Bw=Bw)])
        return Status.SUCCESS
