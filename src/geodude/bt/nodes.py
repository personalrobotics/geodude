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
        # Try each grasp mode — some may fail for thin objects
        import logging
        _logger = logging.getLogger(__name__)
        templates = []
        for grasp_fn in [
            hand.grasp_box_face_x,
            hand.grasp_box_face_y,
            hand.grasp_box_top,
            hand.grasp_box_bottom,
        ]:
            try:
                templates.extend(grasp_fn(size[0], size[1], size[2]))
            except ValueError as e:
                _logger.info(
                    "%s: skipping %s — %s (size=%.0f×%.0f×%.0fmm)",
                    body_name, grasp_fn.__name__, e,
                    size[0] * 1000, size[1] * 1000, size[2] * 1000,
                )
        if not templates:
            _logger.warning(
                "%s: no valid grasps — object too small for gripper "
                "(size=%.0f×%.0f×%.0fmm)",
                body_name, size[0] * 1000, size[1] * 1000, size[2] * 1000,
            )
        return [t.instantiate(T_bottom) for t in templates]

    return []


def _get_held_object_height(robot) -> float:
    """Get the height of the currently held object, or 0 if unknown."""
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR

    held = robot.holding()
    if not held:
        return 0.0

    _, obj_name = held
    assets = AssetManager(str(OBJECTS_DIR))
    # Extract type from instance name (e.g. "cracker_box_0" -> "cracker_box")
    import re
    m = re.match(r"^(.+?)_(\d+)$", obj_name)
    if not m:
        return 0.0

    try:
        gp = assets.get(m.group(1))["geometric_properties"]
        if gp["type"] == "cylinder":
            return max(gp["height"], gp["radius"] * 2)
        elif gp["type"] == "box":
            return max(gp["size"])
    except (KeyError, TypeError):
        pass
    return 0.0


def _generate_container_drop_tsrs(
    robot, body_name: str, dest_type: str, held_height: float = 0.0,
) -> list:
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
    clearance = max(0.25, held_height + 0.10)
    drop_pos = dest_pose[:3, 3] + local_z * (outer[2] + clearance)

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


def _generate_surface_place_tsrs(
    robot,
    surface_pose: np.ndarray,
    surface_hx: float,
    surface_hy: float,
    held_obj_type: str | None,
    T_gripper_object: np.ndarray | None = None,
) -> list:
    """Generate stable placement TSRs for placing a held object on a flat surface.

    Uses StablePlacer to compute the resting pose, then lifts Bw[2] by a
    clearance buffer so the planner targets a collision-free pose above the
    surface.  Physics drops the object the last few mm on release.

    If ``T_gripper_object`` is provided, the grasp offset is composed into
    ``Tw_e`` so that the TSR samples end-effector (gripper) poses rather than
    object poses — which is what pycbirrt expects for IK.

    Args:
        robot: Geodude instance (unused, kept for API consistency).
        surface_pose: 4x4 world-frame pose of the surface center (z up).
        surface_hx: Surface half-extent along x (m).
        surface_hy: Surface half-extent along y (m).
        held_obj_type: prl_assets type of the held object, or None.
        T_gripper_object: 4x4 grasp transform (gripper frame → object frame),
            or None to skip the correction (e.g. for spawning / tests).
    """
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR
    from tsr.placement import StablePlacer

    if held_obj_type is None:
        return []

    assets = AssetManager(str(OBJECTS_DIR))
    try:
        gp = assets.get(held_obj_type)["geometric_properties"]
    except (KeyError, TypeError):
        return []

    # Shrink placement area by a margin to keep objects away from edges
    margin = 0.05
    placer = StablePlacer(
        table_x=max(0.01, surface_hx - margin),
        table_y=max(0.01, surface_hy - margin),
    )

    geo_type = gp.get("type")
    if geo_type == "cylinder":
        templates = placer.place_cylinder(gp["radius"], gp["height"], subject=held_obj_type)
    elif geo_type == "box":
        templates = placer.place_box(gp["size"][0], gp["size"][1], gp["size"][2], subject=held_obj_type)
    elif geo_type == "sphere":
        templates = placer.place_sphere(gp["radius"], subject=held_obj_type)
    else:
        return []

    # Only use the most natural resting pose (first template, typically -z face down)
    if not templates:
        return []
    template = templates[0]

    # Apply grasp offset: convert object-frame TSR → gripper-frame TSR
    # so pycbirrt plans the end-effector to the right pose.
    #   Tw_e_corrected = Tw_e_object @ inv(T_site_object)
    if T_gripper_object is not None:
        import dataclasses
        T_object_gripper = np.linalg.inv(T_gripper_object)
        template = dataclasses.replace(template, Tw_e=template.Tw_e @ T_object_gripper)

    tsr = template.instantiate(surface_pose)

    # Lift z-bounds above the surface so the planner avoids collision.
    # The object will settle the last few mm on release via physics.
    clearance = 0.005  # 5mm buffer
    tsr.Bw[2, :] += clearance

    return [tsr]


_UPWARD_THRESHOLD = 0.95  # dot(normal, [0,0,1]) > this ≈ <18° from vertical


def _get_upward_faces(
    dest_pose: np.ndarray, gp: dict,
) -> list[tuple[np.ndarray, float, float]]:
    """Enumerate flat faces of a destination object that currently point upward.

    Returns a list of ``(surface_pose, half_x, half_y)`` for each face whose
    world-frame normal has ``dot(n, [0,0,1]) > _UPWARD_THRESHOLD``.

    Args:
        dest_pose: 4x4 world-frame pose of the destination (origin at center).
        gp: ``geometric_properties`` dict from prl_assets metadata.
    """
    R = dest_pose[:3, :3]
    origin = dest_pose[:3, 3]
    geo_type = gp.get("type")

    # Build candidate faces: (local_normal, offset_along_normal, half_x, half_y)
    # Object frame has origin at center; faces are at ±half-extent along each axis.
    candidates: list[tuple[np.ndarray, float, float, float]] = []

    if geo_type == "box":
        lx, ly, lz = gp["size"]
        # ±z faces
        candidates.append((np.array([0, 0, +1.0]), lz / 2, lx / 2, ly / 2))
        candidates.append((np.array([0, 0, -1.0]), lz / 2, lx / 2, ly / 2))
        # ±y faces
        candidates.append((np.array([0, +1.0, 0]), ly / 2, lx / 2, lz / 2))
        candidates.append((np.array([0, -1.0, 0]), ly / 2, lx / 2, lz / 2))
        # ±x faces
        candidates.append((np.array([+1.0, 0, 0]), lx / 2, ly / 2, lz / 2))
        candidates.append((np.array([-1.0, 0, 0]), lx / 2, ly / 2, lz / 2))

    elif geo_type == "cylinder":
        r, h = gp["radius"], gp["height"]
        # ±z end caps (circular, approximate extents as r × r)
        candidates.append((np.array([0, 0, +1.0]), h / 2, r, r))
        candidates.append((np.array([0, 0, -1.0]), h / 2, r, r))

    # sphere: no flat faces → candidates stays empty

    results = []
    up = np.array([0.0, 0.0, 1.0])
    for local_normal, offset, hx, hy in candidates:
        normal_world = R @ local_normal
        if normal_world @ up < _UPWARD_THRESHOLD:
            continue

        # Surface pose: origin at the face center, Z = upward normal
        face_center = origin + R @ (local_normal * offset)

        # Build surface orientation: Z = normal_world (≈ up),
        # X/Y from the destination's rotation projected onto the face plane.
        # Use the destination's local axes that span the face.
        surface_pose = np.eye(4)
        surface_pose[:3, 3] = face_center
        surface_pose[:3, 2] = normal_world

        # Pick X axis: use dest_pose's first axis that isn't the face normal
        abs_normal = np.abs(local_normal)
        if abs_normal[0] < 0.5:
            local_x = np.array([1.0, 0, 0])
        elif abs_normal[1] < 0.5:
            local_x = np.array([0, 1.0, 0])
        else:
            local_x = np.array([0, 0, 1.0])
        surface_x = R @ local_x
        # Orthogonalize against normal
        surface_x -= normal_world * (surface_x @ normal_world)
        surface_x /= np.linalg.norm(surface_x)
        surface_pose[:3, 0] = surface_x
        surface_pose[:3, 1] = np.cross(normal_world, surface_x)

        results.append((surface_pose, hx, hy))

    return results


def _generate_place_tsrs(
    robot, body_name: str, dest_type: str, held_height: float = 0.0,
    T_gripper_object: np.ndarray | None = None,
) -> list:
    """Generate placement TSRs — dispatches between container drop and surface placement."""
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR

    assets = AssetManager(str(OBJECTS_DIR))
    try:
        gp = assets.get(dest_type)["geometric_properties"]
    except (KeyError, TypeError):
        return []

    geo_type = gp.get("type")

    # Container drop (existing behavior)
    if geo_type in ("open_box", "tote"):
        return _generate_container_drop_tsrs(robot, body_name, dest_type, held_height)

    # Surface placement on any upward-facing flat face
    try:
        dest_pose = robot.get_object_pose(body_name)
    except (ValueError, AttributeError):
        return []

    faces = _get_upward_faces(dest_pose, gp)
    if not faces:
        return []

    held_type = _get_held_object_type(robot)
    all_tsrs = []
    for surface_pose, hx, hy in faces:
        tsrs = _generate_surface_place_tsrs(
            robot, surface_pose, hx, hy, held_type,
            T_gripper_object=T_gripper_object,
        )
        all_tsrs.extend(tsrs)
    return all_tsrs


def _get_held_object_type(robot) -> str | None:
    """Get the prl_assets type of the currently held object, or None."""
    held = robot.holding()
    if not held:
        return None
    _, obj_name = held
    m = re.match(r"^(.+?)_(\d+)$", obj_name)
    return m.group(1) if m else None


def _get_worktop_surface(robot) -> tuple[np.ndarray, float, float] | None:
    """Get the worktop surface pose and half-extents, or None if no worktop site."""
    try:
        wt_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    except Exception:
        return None
    if wt_id < 0:
        return None
    wt_size = robot.model.site_size[wt_id]
    surface_pose = np.eye(4)
    surface_pose[:3, 3] = robot.data.site_xpos[wt_id].copy()
    return surface_pose, float(wt_size[0]), float(wt_size[1])


def _get_grasp_transform(robot) -> np.ndarray | None:
    """Get T_site_object (grasp site → object) for the currently held object.

    GraspManager stores T_body_object (attachment body → object), but
    pycbirrt targets the grasp_site, not the body.  We convert:
        T_site_object = inv(T_body_site) @ T_body_object
    """
    held = robot.holding()
    if not held:
        return None
    side, obj_name = held

    T_body_object = robot.grasp_manager.get_grasp_transform(obj_name)
    if T_body_object is None:
        return None

    # Get the arm for the holding side to find site and body poses
    arm = robot._left_arm if side == "left" else robot._right_arm

    # Compute T_body_site from world poses of body and site
    body_name = arm.gripper.attachment_body
    body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    site_id = arm.ee_site_id

    T_world_body = np.eye(4)
    T_world_body[:3, :3] = robot.data.xmat[body_id].reshape(3, 3)
    T_world_body[:3, 3] = robot.data.xpos[body_id]

    T_world_site = np.eye(4)
    T_world_site[:3, :3] = robot.data.site_xmat[site_id].reshape(3, 3)
    T_world_site[:3, 3] = robot.data.site_xpos[site_id]

    T_body_site = np.linalg.inv(T_world_body) @ T_world_site
    T_site_object = np.linalg.inv(T_body_site) @ T_body_object
    return T_site_object


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


class GeneratePlaceTSRs(py_trees.behaviour.Behaviour):
    """Generate placement TSRs for containers, surfaces, or worktop.

    Supports smart resolution like GenerateGrasps.  When no destination is
    specified, tries containers first, then falls back to the worktop surface.

    Reads: ``{ns}/destination``, ``{ns}/robot``
    Writes: ``{ns}/place_tsrs``
    """

    def __init__(self, ns: str = "", name: str = "GeneratePlaceTSRs"):
        super().__init__(name)
        self.ns = ns
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(key=f"{ns}/destination", access=Access.READ)
        self.bb.register_key(key=f"{ns}/robot", access=Access.READ)
        self.bb.register_key(key=f"{ns}/place_tsrs", access=Access.WRITE)

    def update(self) -> Status:
        robot = self.bb.get(f"{self.ns}/robot")
        target = self.bb.get(f"{self.ns}/destination")

        held_height = _get_held_object_height(robot)
        held_type = _get_held_object_type(robot)

        # Look up grasp transform so surface placement TSRs target the
        # gripper (what pycbirrt expects) rather than the object.
        T_gripper_object = _get_grasp_transform(robot)

        all_tsrs = []

        # Special case: "worktop" targets the worktop surface directly
        if target == "worktop":
            wt = _get_worktop_surface(robot)
            if wt is not None:
                surface_pose, hx, hy = wt
                all_tsrs = _generate_surface_place_tsrs(
                    robot, surface_pose, hx, hy, held_type,
                    T_gripper_object=T_gripper_object,
                )
            if not all_tsrs:
                self.feedback_message = "No worktop surface found"
                return Status.FAILURE
            self.bb.set(f"{self.ns}/place_tsrs", all_tsrs)
            return Status.SUCCESS

        # Find matching destinations from scene objects
        objects = _find_scene_objects(robot, target)
        for body_name, dest_type in objects:
            tsrs = _generate_place_tsrs(
                robot, body_name, dest_type, held_height=held_height,
                T_gripper_object=T_gripper_object,
            )
            all_tsrs.extend(tsrs)

        # No-destination fallback: try worktop if no container TSRs were found
        if not all_tsrs and target is None:
            wt = _get_worktop_surface(robot)
            if wt is not None:
                surface_pose, hx, hy = wt
                all_tsrs = _generate_surface_place_tsrs(
                    robot, surface_pose, hx, hy, held_type,
                    T_gripper_object=T_gripper_object,
                )

        if not all_tsrs:
            self.feedback_message = f"No placement TSRs for destination '{target}'"
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
