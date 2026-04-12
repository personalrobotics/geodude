# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""High-level manipulation primitives for Geodude.

Delegates to mj_manipulator's generic primitives for core logic.
Adds Geodude-specific behavior: VentionBase homing in go_home,
and uses geodude_pickup/geodude_place subtrees (which include LiftBase).
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import py_trees
from mj_manipulator.primitives import (
    _arm_preempted,
    _deactivate_teleop_for_arms,
    _report_pickup_failure,
    _set_hud_action,
    _setup_blackboard,
    _sync_viewer,
    _tick_tree,
)
from py_trees.common import Access

if TYPE_CHECKING:
    from geodude.robot import Geodude

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pickup
# ---------------------------------------------------------------------------


def pickup(
    robot: Geodude,
    target: str | None = None,
    *,
    arm: str | None = None,
    verbose: bool | None = None,
) -> bool:
    """Pick up an object.

    Uses geodude_pickup subtree (includes LiftBase for VentionBase).
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    if verbose is None:
        verbose = robot.config.debug.verbose

    robot.clear_abort()
    _deactivate_teleop_for_arms(robot)
    try:
        return _pickup_inner(robot, target, arm=arm, verbose=verbose)
    except KeyboardInterrupt:
        robot.request_abort()
        logger.warning("Pickup interrupted by user")
        for side in ("left", "right"):
            _set_hud_action(robot, side, "⊘ interrupted")
        _sync_viewer(robot)
        return False
    finally:
        robot.clear_abort()


def _pickup_inner(
    robot: Geodude,
    target: str | None = None,
    *,
    arm: str | None = None,
    verbose: bool = False,
) -> bool:
    """Core pickup logic."""
    from geodude.bt.subtrees import geodude_pickup

    ctx = robot._active_context

    # Quick check: are there any matching objects?
    if not robot.find_objects(target):
        desc = f"'{target}'" if target else "any object"
        logger.warning("Pickup failed: no graspable objects found for %s", desc)
        return False

    def _try_pickup(side: str) -> bool:
        arm_obj = robot.arms[side]
        ns = f"/{side}"
        _setup_blackboard(robot, ctx, side, arm_obj, ns)

        bb = py_trees.blackboard.Client(name=f"pickup_target{ns}")
        bb.register_key(key=f"{ns}/object_name", access=Access.WRITE)
        bb.set(f"{ns}/object_name", target)

        desc = target or "any"
        _set_hud_action(robot, side, f"⟳ pickup({desc})")
        if not _tick_tree(geodude_pickup(ns), verbose=verbose):
            _set_hud_action(robot, side, f"✗ pickup({desc})")
            return False
        _set_hud_action(robot, side, f"✓ pickup({desc})")
        return True

    if arm is not None:
        if _try_pickup(arm):
            _sync_viewer(robot)
            return True
        _report_pickup_failure(robot, [arm], target)
        _sync_viewer(robot)
        return False

    sides = ["right", "left"]
    random.shuffle(sides)
    sides_tried = []
    for i, side in enumerate(sides):
        if _try_pickup(side):
            _sync_viewer(robot)
            return True
        sides_tried.append(side)
        if robot.is_abort_requested() or _arm_preempted(robot, side):
            _sync_viewer(robot)
            return False
        # Before trying the other arm, send this arm home
        if i < len(sides) - 1 and not _arm_preempted(robot, side):
            go_home(robot, arm=side)

    _report_pickup_failure(robot, sides_tried, target)
    for side in sides_tried:
        if not _arm_preempted(robot, side):
            go_home(robot, arm=side)
    _sync_viewer(robot)
    return False


# ---------------------------------------------------------------------------
# Place
# ---------------------------------------------------------------------------


def place(
    robot: Geodude,
    destination: str | None = None,
    *,
    arm: str | None = None,
    verbose: bool | None = None,
) -> bool:
    """Place the held object."""
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    if arm is None:
        held = robot.holding()
        if held:
            arm = held[0]
        else:
            logger.warning("Place failed: no arm is holding an object")
            return False

    if verbose is None:
        verbose = robot.config.debug.verbose

    robot.clear_abort()
    _deactivate_teleop_for_arms(robot)
    try:
        return _place_inner(robot, destination, arm=arm, verbose=verbose)
    except KeyboardInterrupt:
        robot.request_abort()
        logger.warning("Place interrupted by user")
        _set_hud_action(robot, arm, "⊘ interrupted")
        _sync_viewer(robot)
        return False
    finally:
        robot.clear_abort()


def _place_inner(
    robot: Geodude,
    destination: str | None,
    *,
    arm: str,
    verbose: bool = False,
) -> bool:
    """Core place logic."""
    from mj_manipulator.primitives import _maybe_hide_in_container

    from geodude.bt.subtrees import geodude_place

    ctx = robot._active_context
    arm_obj = robot.arms[arm]
    ns = f"/{arm}"
    _setup_blackboard(robot, ctx, arm, arm_obj, ns)

    bb = py_trees.blackboard.Client(name=f"place_target{ns}")
    bb.register_key(key=f"{ns}/destination", access=Access.WRITE)
    bb.register_key(key=f"{ns}/object_name", access=Access.WRITE)
    bb.set(f"{ns}/destination", destination)

    held_name = None
    if arm_obj.gripper:
        held_name = arm_obj.gripper.held_object
    bb.set(f"{ns}/object_name", held_name)

    desc = destination or "auto"
    _set_hud_action(robot, arm, f"⟳ place({desc})")
    ok = _tick_tree(geodude_place(ns), verbose=verbose)

    if ok:
        _set_hud_action(robot, arm, f"✓ place({desc})")
        if held_name:
            _maybe_hide_in_container(robot, ns, destination, held_name)
    else:
        _set_hud_action(robot, arm, f"✗ place({desc})")
        logger.warning("Place failed for destination '%s'", destination)

    _sync_viewer(robot)
    return ok


# ---------------------------------------------------------------------------
# Go home (with VentionBase homing)
# ---------------------------------------------------------------------------


def go_home(robot: Geodude, *, arm: str | None = None, verbose: bool | None = None) -> bool:
    """Return arms to ready, including VentionBase homing."""
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    if verbose is None:
        verbose = robot.config.debug.verbose

    robot.clear_abort()
    arms_to_home = [arm] if arm is not None else ["left", "right"]
    _deactivate_teleop_for_arms(robot, arms_to_home)
    try:
        return _go_home_inner(robot, ctx, arm=arm, verbose=verbose)
    except KeyboardInterrupt:
        robot.request_abort()
        logger.warning("go_home interrupted by user")
        return False
    finally:
        robot.clear_abort()


def _go_home_inner(
    robot: Geodude,
    ctx,
    *,
    arm: str | None = None,
    verbose: bool = False,
) -> bool:
    """Core go_home with VentionBase homing."""
    from mj_manipulator.cartesian import CartesianController

    if arm is not None:
        arms = [(arm, robot._resolve_arm(arm))]
    else:
        arms = [("left", robot._left_arm), ("right", robot._right_arm)]

    def abort_fn():
        return robot.is_abort_requested()

    success = True
    for side, arm_obj in arms:
        if "ready" not in robot.named_poses or side not in robot.named_poses["ready"]:
            logger.warning("go_home: no 'ready' pose for %s arm, skipping", side)
            continue
        ready = np.array(robot.named_poses["ready"][side])
        try:
            path = arm_obj.plan_to_configuration(ready, abort_fn=abort_fn)
        except Exception as e:
            logger.warning("go_home %s arm: plan failed: %s", side, e)
            path = None

        if path is None:
            logger.warning("go_home %s arm: retract up and retry", side)
            arm_name = arm_obj.config.name

            def _step_fn(q, qd):
                ctx.step_cartesian(arm_name, q, qd)

            ctrl = CartesianController.from_arm(arm_obj, step_fn=_step_fn)
            ctrl.move(
                np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0]),
                dt=ctx.control_dt,
                max_distance=0.10,
                stop_condition=abort_fn,
            )
            try:
                path = arm_obj.plan_to_configuration(ready, abort_fn=abort_fn)
            except Exception as e:
                logger.warning("go_home %s arm: retry failed: %s", side, e)
                path = None

        if path is not None:
            traj = arm_obj.retime(path)
            ctx.execute(traj)
        else:
            logger.warning("go_home: could not plan %s arm to ready", side)
            success = False

    # Return bases to starting height (Geodude-specific)
    for _, arm_obj in arms:
        base = robot._get_base_for_arm(arm_obj)
        if base is not None and abs(base.get_height() - 0.25) > 0.01:
            base_traj = base.plan_to(0.25, check_collisions=True)
            if base_traj is not None:
                ctx.execute(base_traj)

    ctx.sync()
    return success
