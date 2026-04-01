"""High-level manipulation primitives for Geodude.

Clean function API backed by py_trees behavior trees. Students write::

    robot = Geodude(objects={"can": 1, "recycle_bin": 1})
    with robot.sim() as ctx:
        robot.pickup("can_0")
        robot.place("recycle_bin_0")
        robot.go_home()

Under the hood, each function builds a py_trees tree with recovery,
ticks it to completion, and returns True/False.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import py_trees
from py_trees.common import Access, Status

from geodude.bt.subtrees import geodude_pickup, geodude_place

if TYPE_CHECKING:
    from geodude.robot import Geodude

logger = logging.getLogger(__name__)

_CONTAINER_TYPES = frozenset(("open_box", "tote"))


def _set_hud_action(robot, arm: str, text: str) -> None:
    """Update the HUD status for an arm (no-op if HUD not active)."""
    hud = getattr(robot, "_status_hud", None)
    if hud is not None:
        hud.set_action(arm, text)


def _sync_viewer(robot) -> None:
    """Force a viewer sync so HUD updates are visible immediately."""
    ctx = robot._active_context
    if ctx is not None and hasattr(ctx, 'sync'):
        try:
            ctx.sync()
        except Exception:
            pass


def _is_container_destination(destination: str | None) -> bool:
    """Check if a destination name refers to a container (bin, tote).

    Returns True for container types where the object should be hidden
    after placement (simulating recycling/disposal). Returns False for
    surface destinations where the object stays in the scene.
    """
    if destination is None:
        return False

    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR

    # Strip instance suffix (e.g. "recycle_bin_0" → "recycle_bin")
    import re
    m = re.match(r"^(.+?)_(\d+)$", destination)
    obj_type = m.group(1) if m else destination

    if obj_type == "worktop":
        return False

    assets = AssetManager(str(OBJECTS_DIR))
    try:
        gp = assets.get(obj_type)["geometric_properties"]
        return gp.get("type") in _CONTAINER_TYPES
    except (KeyError, TypeError):
        return False


def _setup_blackboard(robot: Geodude, ns: str) -> py_trees.blackboard.Client:
    """Set up blackboard with robot state for a given arm namespace."""
    ctx = robot._active_context
    arm = robot._resolve_arm(ns.strip("/"))

    bb = py_trees.blackboard.Client(name=f"primitives{ns}")
    keys = [
        "/context",
        f"{ns}/robot",
        f"{ns}/arm", f"{ns}/arm_name",
        f"{ns}/grasp_tsrs", f"{ns}/place_tsrs",
        f"{ns}/timeout", f"{ns}/object_name", f"{ns}/destination",
        f"{ns}/goal_config", f"{ns}/grasped",
        f"{ns}/path", f"{ns}/trajectory",
        f"{ns}/twist", f"{ns}/distance",
        f"{ns}/goal_tsr_index", f"{ns}/tsr_to_object",
        f"{ns}/plan_failure_reason",
    ]
    for key in keys:
        try:
            bb.register_key(key=key, access=Access.WRITE)
        except KeyError:
            pass  # already registered by another client

    bb.set("/context", ctx)
    bb.set(f"{ns}/robot", robot)
    bb.set(f"{ns}/arm", arm)
    bb.set(f"{ns}/arm_name", arm.config.name)
    bb.set(f"{ns}/timeout", robot.config.planning.timeout)

    # Clear stale results from previous runs
    for stale_key in [
        "path", "trajectory", "grasped", "grasp_tsrs", "place_tsrs",
        "tsr_to_object", "goal_tsr_index", "plan_failure_reason",
    ]:
        bb.set(f"{ns}/{stale_key}", None)

    # Home config for recovery
    side = arm.config.name
    if "ready" in robot.named_poses and side in robot.named_poses["ready"]:
        bb.set(f"{ns}/goal_config", np.array(robot.named_poses["ready"][side]))

    return bb


def _tick_tree(root: py_trees.behaviour.Behaviour, verbose: bool = False) -> bool:
    """Reset and tick a tree to completion. Returns True if SUCCESS."""
    for node in root.iterate():
        node.status = Status.INVALID
    tree = py_trees.trees.BehaviourTree(root=root)
    tree.tick()

    if verbose:
        print(py_trees.display.ascii_tree(root, show_status=True))

    if root.status != Status.SUCCESS:
        tip = root.tip()
        if tip is not None and tip.feedback_message:
            logger.info("%s: %s", tip.name, tip.feedback_message)

    return root.status == Status.SUCCESS


def pickup(
    robot: Geodude,
    target: str | None = None,
    *,
    arm: str | None = None,
    verbose: bool | None = None,
) -> bool:
    """Pick up an object.

    Automatically generates grasp TSRs from object geometry in prl_assets.

    Args:
        robot: Geodude instance with active execution context.
        target: What to pick up:
            - "can_0" — specific instance
            - "can" — any can in the scene
            - None — anything graspable
        arm: "left", "right", or None (try both).
        verbose: Show BT tree status. None = use robot.config.debug.verbose.

    Returns:
        True if pickup succeeded.
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    if verbose is None:
        verbose = robot.config.debug.verbose

    # Quick check: are there any matching objects?
    from geodude.bt.nodes import _find_scene_objects
    if not _find_scene_objects(robot, target):
        desc = f"'{target}'" if target else "any object"
        logger.info("Pickup failed: no graspable objects found for %s", desc)
        return False

    def _try_pickup(side: str) -> bool:
        ns = f"/{side}"
        bb = _setup_blackboard(robot, ns)
        bb.set(f"{ns}/object_name", target)
        desc = target or "any"
        _set_hud_action(robot, side, f"⟳ pickup({desc})")
        if not _tick_tree(geodude_pickup(ns), verbose=verbose):
            _set_hud_action(robot, side, f"✗ pickup({desc})")
            return False
        _set_hud_action(robot, side, f"✓ pickup({desc})")
        return True

    def _pickup_details(side: str) -> tuple[list[str], str | None, bool, bool, str | None]:
        """Get attempted objects, the specific object reached, and grasp result.

        Returns:
            (attempted_objects, reached_object, plan_succeeded, grasp_succeeded,
             plan_failure_reason)
        """
        ns = f"/{side}"
        attempted: list[str] = []
        reached: str | None = None
        plan_failed = False
        grasped = False
        plan_reason: str | None = None
        try:
            bb = py_trees.blackboard.Client(name=f"pickup_report{ns}")
            bb.register_key(key=f"{ns}/tsr_to_object", access=Access.READ)
            bb.register_key(key=f"{ns}/object_name", access=Access.READ)
            bb.register_key(key=f"{ns}/grasped", access=Access.READ)
            bb.register_key(key=f"{ns}/plan_failure_reason", access=Access.READ)
            mapping = bb.get(f"{ns}/tsr_to_object")
            if mapping:
                attempted = sorted(set(mapping))
            obj = bb.get(f"{ns}/object_name")
            if obj:
                reached = obj
            plan_reason = bb.get(f"{ns}/plan_failure_reason")
            plan_failed = plan_reason is not None
            grasped = bool(bb.get(f"{ns}/grasped"))
        except (KeyError, RuntimeError):
            pass
        return attempted, reached, not plan_failed, grasped, plan_reason

    def _report_failure(sides_tried: list[str]) -> None:
        all_attempted: set[str] = set()
        plan_failures: list[str] = []
        grasp_failures: list[str] = []
        for side in sides_tried:
            attempted, reached, planned, grasped, plan_reason = _pickup_details(side)
            all_attempted.update(attempted)
            if reached and planned and not grasped:
                grasp_failures.append(f"{reached} ({side} arm)")
                _set_hud_action(robot, side, f"✗ pickup: grasp failed")
            elif reached and not planned:
                detail = f"{reached} ({side} arm)"
                short = plan_reason.split(":")[0] if plan_reason else "plan failed"
                if plan_reason:
                    detail += f": {plan_reason}"
                plan_failures.append(detail)
                _set_hud_action(robot, side, f"✗ pickup: {short}")

        if grasp_failures:
            msg = f"Pickup failed: reached {', '.join(grasp_failures)} but grasp failed"
            logger.info(msg)
        elif plan_failures:
            logger.info(
                "Pickup failed: could not plan to %s",
                "; ".join(plan_failures),
            )
        elif all_attempted:
            logger.info(
                "Pickup failed: could not plan to %s",
                ", ".join(sorted(all_attempted)),
            )
        else:
            desc = f"'{target}'" if target else "any object"
            logger.info("Pickup failed: no graspable %s found", desc)

    if arm is not None:
        if _try_pickup(arm):
            _sync_viewer(robot)
            return True
        _report_failure([arm])
        _sync_viewer(robot)
        return False

    import random
    sides = ["right", "left"]
    random.shuffle(sides)
    for i, side in enumerate(sides):
        if _try_pickup(side):
            _sync_viewer(robot)
            return True
        # Before trying the other arm, ensure this arm is home
        # so it doesn't block the workspace
        if i < len(sides) - 1:
            from geodude.primitives import go_home
            go_home(robot, arm=side)

    _report_failure(sides)
    _sync_viewer(robot)
    return False


def place(
    robot: Geodude,
    destination: str | None = None,
    *,
    arm: str | None = None,
    verbose: bool | None = None,
) -> bool:
    """Place the held object at a destination.

    Automatically generates drop-zone TSRs from destination geometry in prl_assets.

    Args:
        robot: Geodude instance with active execution context.
        destination: Where to place:
            - "recycle_bin_0" — specific instance
            - "recycle_bin" — any recycle bin in scene
            - None — any valid destination
        arm: "left", "right", or None (auto-detect holding arm).
        verbose: Show BT tree status. None = use robot.config.debug.verbose.

    Returns:
        True if place succeeded.
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    # Find which arm is holding
    if arm is None:
        for side in ("left", "right"):
            if robot.grasp_manager.get_grasped_by(side):
                arm = side
                break
        if arm is None:
            logger.info("Place failed: no arm is holding an object")
            return False

    if verbose is None:
        verbose = robot.config.debug.verbose

    # Capture held object before the tree releases the grasp
    held = list(robot.grasp_manager.get_grasped_by(arm))
    held_object = held[0] if held else None

    ns = f"/{arm}"
    bb = _setup_blackboard(robot, ns)
    bb.set(f"{ns}/destination", destination)
    desc = destination or "any"
    _set_hud_action(robot, arm, f"⟳ place({desc})")
    ok = _tick_tree(geodude_place(ns), verbose=verbose)
    if not ok:
        # Read failure reason from blackboard
        try:
            _bb = py_trees.blackboard.Client(name=f"place_report{ns}")
            _bb.register_key(key=f"{ns}/plan_failure_reason", access=Access.READ)
            reason = _bb.get(f"{ns}/plan_failure_reason")
        except (KeyError, RuntimeError):
            reason = None

        detail = f": {reason}" if reason else ""
        logger.info("Place failed: %s arm could not place at '%s'%s", arm, destination, detail)
        short_reason = reason.split(":")[0] if reason else "failed"
        _set_hud_action(robot, arm, f"✗ place({desc}): {short_reason}")
    else:
        _set_hud_action(robot, arm, f"✓ place({desc})")

    # Force viewer sync so HUD updates immediately
    _sync_viewer(robot)

    # Hide object only if placed into a container (recycled) — surface placements stay
    if ok and held_object and _is_container_destination(destination):
        if robot.env.registry.is_active(held_object):
            robot.env.registry.hide(held_object)
            robot.forward()

    return ok


def go_home(robot: Geodude, *, arm: str | None = None, verbose: bool | None = None) -> bool:
    """Return arms to ready configuration.

    Args:
        robot: Geodude instance with active execution context.
        arm: "left", "right", or None (both arms).
        verbose: Show debug info.

    Returns:
        True if all specified arms returned to ready.
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    if verbose is None:
        verbose = robot.config.debug.verbose

    from mj_manipulator.cartesian import CartesianController

    if arm is not None:
        arms = [(arm, robot._resolve_arm(arm))]
    else:
        arms = [("left", robot._left_arm), ("right", robot._right_arm)]

    success = True
    for side, arm_obj in arms:
        if "ready" not in robot.named_poses or side not in robot.named_poses["ready"]:
            continue
        ready = np.array(robot.named_poses["ready"][side])
        try:
            path = arm_obj.plan_to_configuration(ready)
        except Exception as e:
            logger.info("go_home %s arm: initial plan failed: %s", side, e)
            path = None

        if path is None:
            # Retract up first, then retry
            arm_name = arm_obj.config.name

            def _step_fn(q, qd):
                ctx.step_cartesian(arm_name, q, qd)

            ctrl = CartesianController.from_arm(arm_obj, step_fn=_step_fn)
            ctrl.move(
                np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0]),
                dt=0.008, max_distance=0.10,
            )
            try:
                path = arm_obj.plan_to_configuration(ready)
            except Exception as e:
                logger.info("go_home %s arm: retry after retract failed: %s", side, e)
                path = None

        if path is not None:
            traj = arm_obj.retime(path)
            ctx.execute(traj)
        else:
            if verbose:
                logger.debug("go_home: %s arm failed to plan", side)
            logger.info("Could not plan %s arm to ready", side)
            success = False
    # Return bases to starting height (0.25 midpoint)
    for _, arm_obj in arms:
        base = robot._get_base_for_arm(arm_obj)
        if base is not None and abs(base.get_height() - 0.25) > 0.01:
            base_traj = base.plan_to(0.25, check_collisions=True)
            if base_traj is not None:
                ctx.execute(base_traj)

    ctx.sync()
    return success
