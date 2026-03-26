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


def _find_scene_objects(robot, target: str | None) -> list[tuple[str, str]]:
    """Find objects in the scene matching a target specification.

    Args:
        robot: Geodude instance.
        target: One of:
            - "can_0" (specific instance) → [("can_0", "can")]
            - "can" (type name) → [("can_0", "can"), ("can_1", "can"), ...]
            - None (any graspable) → all objects with prl_assets geometry

    Returns:
        List of (body_name, object_type) tuples.
    """
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR

    assets = AssetManager(str(OBJECTS_DIR))
    model = robot.model
    gm = robot.grasp_manager

    # Collect all visible body names in the scene
    registry = robot.env.registry if hasattr(robot.env, 'registry') else None
    all_bodies = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name:
            continue
        # Skip hidden/inactive objects
        if registry is not None:
            try:
                if not registry.is_active(name):
                    continue
            except Exception:
                pass  # not a registry-managed object
        all_bodies.append(name)

    if target is not None:
        # Check if it's a specific instance (has _N suffix)
        instance_match = re.match(r"^(.+)_(\d+)$", target)
        if instance_match:
            obj_type = instance_match.group(1)
            if target in all_bodies:
                return [(target, obj_type)]
            return []

        # It's a type name — find all instances of this type
        matches = []
        for body in all_bodies:
            m = re.match(r"^(.+?)_(\d+)$", body)
            if m and m.group(1) == target:
                if not gm.is_grasped(body):
                    matches.append((body, target))
        return matches

    # None — find all graspable objects
    matches = []
    known_types = set()
    for body in all_bodies:
        m = re.match(r"^(.+?)_(\d+)$", body)
        if not m:
            continue
        obj_type = m.group(1)
        if gm.is_grasped(body):
            continue
        if obj_type not in known_types:
            try:
                assets.get(obj_type)["geometric_properties"]
                known_types.add(obj_type)
            except (KeyError, TypeError):
                continue
        if obj_type in known_types:
            matches.append((body, obj_type))
    return matches


def _generate_tsrs_for_object(robot, body_name: str, obj_type: str) -> list:
    """Generate grasp TSRs for a single object from its prl_assets geometry.

    Returns list of TSRs, or empty list if geometry is unsupported.
    """
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR
    from tsr.hands import Robotiq2F140

    assets = AssetManager(str(OBJECTS_DIR))
    try:
        gp = assets.get(obj_type)["geometric_properties"]
    except (KeyError, TypeError):
        return []

    try:
        obj_pose = robot.get_object_pose(body_name)
    except ValueError:
        return []

    hand = Robotiq2F140()
    if gp.get("type") == "cylinder":
        # Shift to bottom face along the object's local Z axis
        T_bottom = obj_pose.copy()
        local_z = obj_pose[:3, 2]  # object's Z axis in world frame
        T_bottom[:3, 3] -= local_z * (gp["height"] / 2)
        templates = hand.grasp_cylinder_side(gp["radius"], gp["height"])
        return [t.instantiate(T_bottom) for t in templates]
    elif gp.get("type") == "box":
        size = gp["size"]  # [x, y, z]
        T_bottom = obj_pose.copy()
        local_z = obj_pose[:3, 2]
        T_bottom[:3, 3] -= local_z * (size[2] / 2)
        templates = hand.grasp_box(size[0], size[1], size[2])
        return [t.instantiate(T_bottom) for t in templates]

    return []


def _generate_drop_tsrs(robot, body_name: str, dest_type: str) -> list:
    """Generate drop-zone TSRs for a container from prl_assets geometry."""
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR
    from tsr import TSR

    assets = AssetManager(str(OBJECTS_DIR))
    try:
        meta = assets.get(dest_type)
        gp = meta["geometric_properties"]
        policy = meta.get("policy", {}).get("placement", {})
    except (KeyError, TypeError):
        return []

    if gp.get("type") not in ("open_box", "tote"):
        return []

    try:
        dest_pose = robot.get_object_pose(body_name)
    except ValueError:
        return []

    outer = gp["outer_dimensions"]
    wall = gp.get("wall_thickness", 0.003)
    margin = policy.get("drop_zone_margin", 0.05)

    hx = (outer[0] / 2) - wall - margin
    hy = (outer[1] / 2) - wall - margin

    # Drop point: above the container opening along its local Z axis
    local_z = dest_pose[:3, 2]
    drop_pos = dest_pose[:3, 3] + local_z * (outer[2] + 0.15)

    # Approach direction: gripper Z points opposite to container's local Z (downward into it)
    # Gripper X follows container's local X
    approach = -local_z
    gripper_x = dest_pose[:3, 0]
    gripper_y = np.cross(approach, gripper_x)

    T0_w = np.eye(4)
    T0_w[:3, 0] = gripper_x
    T0_w[:3, 1] = gripper_y
    T0_w[:3, 2] = approach
    T0_w[:3, 3] = drop_pos

    Bw = np.zeros((6, 2))
    Bw[0, :] = [-hx, hx]
    Bw[1, :] = [-hy, hy]
    Bw[2, :] = [-0.02, 0.05]
    Bw[5, :] = [-np.pi, np.pi]

    return [TSR(T0_w=T0_w, Bw=Bw)]


class GenerateGrasps(py_trees.behaviour.Behaviour):
    """Generate grasp TSRs for one or more objects in the scene.

    Supports smart object resolution:
    - "can_0" → specific instance
    - "can" → all can instances in scene
    - None → all graspable objects

    Combines TSRs from all matching objects with a tsr_to_object mapping
    so the Grasp node knows which object was reached.

    Reads: ``{ns}/object_name``, ``{ns}/robot``
    Writes: ``{ns}/grasp_tsrs``, ``{ns}/tsr_to_object``
    """

    def __init__(self, ns: str = "", name: str = "GenerateGrasps"):
        super().__init__(name)
        self.ns = ns
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(key=f"{ns}/object_name", access=Access.READ)
        self.bb.register_key(key=f"{ns}/robot", access=Access.READ)
        self.bb.register_key(key=f"{ns}/grasp_tsrs", access=Access.WRITE)
        self.bb.register_key(key=f"{ns}/tsr_to_object", access=Access.WRITE)

    def update(self) -> Status:
        robot = self.bb.get(f"{self.ns}/robot")
        target = self.bb.get(f"{self.ns}/object_name")

        objects = _find_scene_objects(robot, target)
        if not objects:
            self.feedback_message = f"No graspable objects found for '{target}'"
            return Status.FAILURE

        all_tsrs = []
        tsr_to_object = []
        for body_name, obj_type in objects:
            tsrs = _generate_tsrs_for_object(robot, body_name, obj_type)
            for _ in tsrs:
                tsr_to_object.append(body_name)
            all_tsrs.extend(tsrs)

        if not all_tsrs:
            self.feedback_message = f"No TSRs generated for {[b for b, _ in objects]}"
            return Status.FAILURE

        self.bb.set(f"{self.ns}/grasp_tsrs", all_tsrs)
        self.bb.set(f"{self.ns}/tsr_to_object", tsr_to_object)
        return Status.SUCCESS


class GenerateDropZone(py_trees.behaviour.Behaviour):
    """Generate drop-zone TSRs for one or more containers in the scene.

    Supports smart resolution like GenerateGrasps.

    Reads: ``{ns}/destination``, ``{ns}/robot``
    Writes: ``{ns}/place_tsrs``, ``{ns}/tsr_to_object``
    """

    def __init__(self, ns: str = "", name: str = "GenerateDropZone"):
        super().__init__(name)
        self.ns = ns
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(key=f"{ns}/destination", access=Access.READ)
        self.bb.register_key(key=f"{ns}/robot", access=Access.READ)
        self.bb.register_key(key=f"{ns}/place_tsrs", access=Access.WRITE)

    def update(self) -> Status:
        robot = self.bb.get(f"{self.ns}/robot")
        target = self.bb.get(f"{self.ns}/destination")

        # Find matching destinations
        objects = _find_scene_objects(robot, target)
        if not objects:
            self.feedback_message = f"No destinations found for '{target}'"
            return Status.FAILURE

        all_tsrs = []
        for body_name, dest_type in objects:
            tsrs = _generate_drop_tsrs(robot, body_name, dest_type)
            all_tsrs.extend(tsrs)

        if not all_tsrs:
            self.feedback_message = f"No drop-zone TSRs for {[b for b, _ in objects]}"
            return Status.FAILURE

        self.bb.set(f"{self.ns}/place_tsrs", all_tsrs)
        return Status.SUCCESS


class LiftBase(py_trees.behaviour.Behaviour):
    """Lift the Vention base to clear worktop clutter after grasping.

    Plans a collision-free base trajectory 15cm upward (clamped to range)
    and executes it through the context.

    Reads: ``{ns}/robot``, ``{ns}/arm``, ``/context``
    """

    LIFT_AMOUNT = 0.15  # meters

    def __init__(self, ns: str = "", name: str = "LiftBase"):
        super().__init__(name)
        self.ns = ns
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(key=f"{ns}/robot", access=Access.READ)
        self.bb.register_key(key=f"{ns}/arm", access=Access.READ)
        self.bb.register_key(key="/context", access=Access.READ)

    def update(self) -> Status:
        import logging

        robot = self.bb.get(f"{self.ns}/robot")
        arm = self.bb.get(f"{self.ns}/arm")
        ctx = self.bb.get("/context")

        base = robot._get_base_for_arm(arm)
        if base is None:
            return Status.SUCCESS  # no base, nothing to do

        current = base.get_height()
        target_h = min(current + self.LIFT_AMOUNT, base.height_range[1])
        if target_h - current < 0.01:
            return Status.SUCCESS  # already at max

        base_traj = base.plan_to(target_h, check_collisions=True)
        if base_traj is None:
            logging.getLogger(__name__).info(
                "Base lift to %.2fm blocked by collision", target_h,
            )
            return Status.SUCCESS  # non-critical, don't fail pickup

        ctx.execute(base_traj)
        ctx.sync()
        return Status.SUCCESS
