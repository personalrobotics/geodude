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


def _setup_blackboard(robot: Geodude, ns: str) -> py_trees.blackboard.Client:
    """Set up blackboard with robot state for a given arm namespace."""
    ctx = robot._active_context
    arm = robot._resolve_arm(ns.strip("/"))
    step_fn = ctx.step if hasattr(ctx, '_controller') and ctx._controller is not None else None

    bb = py_trees.blackboard.Client(name=f"primitives{ns}")
    keys = [
        "/context",
        f"{ns}/robot",
        f"{ns}/arm", f"{ns}/arm_name",
        f"{ns}/grasp_tsrs", f"{ns}/place_tsrs",
        f"{ns}/timeout", f"{ns}/object_name", f"{ns}/destination",
        f"{ns}/goal_config", f"{ns}/step_fn", f"{ns}/grasped",
        f"{ns}/path", f"{ns}/trajectory",
        f"{ns}/twist", f"{ns}/distance",
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
    bb.set(f"{ns}/step_fn", step_fn)

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
            logger.warning("%s: %s", tip.name, tip.feedback_message)

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
        logger.warning("Pickup failed: no graspable objects found for %s", desc)
        return False

    def _try_pickup(side: str) -> bool:
        ns = f"/{side}"
        bb = _setup_blackboard(robot, ns)
        bb.set(f"{ns}/object_name", target)
        if not _tick_tree(geodude_pickup(ns), verbose=verbose):
            return False
        # Raise base to clear worktop clutter
        arm_obj = robot._resolve_arm(side)
        base = robot._get_base_for_arm(arm_obj)
        if base is not None:
            current = base.get_height()
            target_h = min(current + 0.15, base.height_range[1])
            if target_h > current + 0.01:
                viewer = getattr(ctx, '_viewer', None)
                ok = base.move_to(target_h, check_collisions=True, viewer=viewer)
                if not ok:
                    logger.info("Base lift to %.2fm blocked by collision", target_h)
            ctx.sync()
        return True

    if arm is not None:
        if _try_pickup(arm):
            return True
        logger.warning("Pickup failed: %s arm could not pick up '%s'", arm, target)
        return False

    import random
    sides = ["right", "left"]
    random.shuffle(sides)
    for side in sides:
        if _try_pickup(side):
            return True

    desc = f"'{target}'" if target else "any object"
    logger.warning("Pickup failed: neither arm could pick up %s (tried %s)", desc, ", ".join(sides))
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
            logger.warning("Place failed: no arm is holding an object")
            return False

    if verbose is None:
        verbose = robot.config.debug.verbose

    ns = f"/{arm}"
    bb = _setup_blackboard(robot, ns)
    bb.set(f"{ns}/destination", destination)
    ok = _tick_tree(geodude_place(ns), verbose=verbose)
    if not ok:
        logger.warning("Place failed: %s arm could not place at '%s'", arm, destination)
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
        arms = [("left", robot.left_arm), ("right", robot.right_arm)]

    success = True
    for side, arm_obj in arms:
        if "ready" not in robot.named_poses or side not in robot.named_poses["ready"]:
            continue
        ready = np.array(robot.named_poses["ready"][side])
        try:
            path = arm_obj.plan_to_configuration(ready)
        except Exception:
            path = None

        if path is None:
            # Retract up first, then retry
            ctrl = CartesianController.from_arm(arm_obj)
            ctrl.move(
                np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0]),
                dt=0.008, max_distance=0.10,
            )
            try:
                path = arm_obj.plan_to_configuration(ready)
            except Exception:
                path = None

        if path is not None:
            traj = arm_obj.retime(path)
            ctx.execute(traj)
        else:
            if verbose:
                logger.debug("go_home: %s arm failed to plan", side)
            logger.warning("Could not plan %s arm to ready", side)
            success = False
    # Return bases to starting height (0.25 midpoint)
    viewer = getattr(ctx, '_viewer', None)
    for _, arm_obj in arms:
        base = robot._get_base_for_arm(arm_obj)
        if base is not None and abs(base.get_height() - 0.25) > 0.01:
            base.move_to(0.25, check_collisions=True, viewer=viewer)

    ctx.sync()
    return success
