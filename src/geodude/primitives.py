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


def _tick_tree(root: py_trees.behaviour.Behaviour) -> bool:
    """Reset and tick a tree to completion. Returns True if SUCCESS."""
    for node in root.iterate():
        node.status = Status.INVALID
    tree = py_trees.trees.BehaviourTree(root=root)
    tree.tick()
    if root.status != Status.SUCCESS:
        # Find the deepest failed node and log its feedback
        tip = root.tip()
        if tip is not None and tip.feedback_message:
            logger.warning("%s: %s", tip.name, tip.feedback_message)
    return root.status == Status.SUCCESS


def pickup(
    robot: Geodude,
    object_name: str,
    *,
    arm: str | None = None,
) -> bool:
    """Pick up an object by name.

    Automatically generates grasp TSRs from the object's geometry in
    prl_assets. Tries the specified arm, or both arms if not specified.

    Args:
        robot: Geodude instance with active execution context.
        object_name: MuJoCo body name (e.g., "can_0").
        arm: "left", "right", or None (try both).

    Returns:
        True if pickup succeeded.
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    if arm is not None:
        # Single arm
        ns = f"/{arm}"
        bb = _setup_blackboard(robot, ns)
        bb.set(f"{ns}/object_name", object_name)
        return _tick_tree(geodude_pickup(ns))

    # Try both arms (random order)
    import random
    sides = ["right", "left"]
    random.shuffle(sides)
    for side in sides:
        ns = f"/{side}"
        bb = _setup_blackboard(robot, ns)
        bb.set(f"{ns}/object_name", object_name)
        if _tick_tree(geodude_pickup(ns)):
            return True

    return False


def place(
    robot: Geodude,
    destination: str,
    *,
    arm: str | None = None,
) -> bool:
    """Place the held object at a destination.

    Automatically generates drop-zone TSRs from the destination's
    geometry in prl_assets. Auto-detects which arm is holding.

    Args:
        robot: Geodude instance with active execution context.
        destination: MuJoCo body name (e.g., "recycle_bin_0").
        arm: "left", "right", or None (auto-detect holding arm).

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
            logger.warning("No arm is holding an object")
            return False

    ns = f"/{arm}"
    bb = _setup_blackboard(robot, ns)
    bb.set(f"{ns}/destination", destination)
    return _tick_tree(geodude_place(ns))


def go_home(robot: Geodude) -> bool:
    """Return all arms to ready configuration.

    Args:
        robot: Geodude instance with active execution context.

    Returns:
        True if all arms returned to ready.
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    from mj_manipulator.cartesian import CartesianController

    success = True
    for side, arm_obj in [("left", robot.left_arm), ("right", robot.right_arm)]:
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
            logger.warning("Could not plan %s arm to ready", side)
            success = False
    ctx.sync()
    return success
