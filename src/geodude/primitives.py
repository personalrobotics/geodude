"""High-level manipulation primitives for Geodude.

Provides pickup() and place() that take explicit TSRs from the caller.
TSR generation is the caller's responsibility — typically using tsr.hands
for grasps (from object geometry in prl_assets) and task-specific logic
for placements.

Example::

    from tsr.hands import Robotiq2F140
    hand = Robotiq2F140()
    grasp_templates = hand.grasp_cylinder(radius, height)
    grasp_tsrs = [t.instantiate(object_pose) for t in grasp_templates]

    with robot.sim() as ctx:
        robot.pickup("can_0", grasp_tsrs=grasp_tsrs)
        robot.place("recycle_bin_0", place_tsrs=place_tsrs)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator import Arm, PlanResult
from mj_manipulator.cartesian import CartesianControlConfig, CartesianController

if TYPE_CHECKING:
    from geodude.robot import Geodude

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step_fn(robot: Geodude):
    """Create a step function for Cartesian control."""
    ctx = robot._active_context
    if ctx is not None and hasattr(ctx, '_controller') and ctx._controller is not None:
        def step_fn():
            ctx.step()
        return step_fn

    def step_fn():
        mujoco.mj_forward(robot.model, robot.data)
    return step_fn


def _return_to_ready(robot: Geodude, arm: Arm) -> bool:
    """Return arm to ready position after failure."""
    ctx = robot._active_context
    if ctx is None:
        return False

    side = arm.config.name
    if "ready" not in robot.named_poses or side not in robot.named_poses["ready"]:
        logger.warning("No ready pose defined for %s arm", side)
        return False

    ready_config = np.array(robot.named_poses["ready"][side])

    try:
        path = arm.plan_to_configuration(ready_config)
    except Exception:
        path = None

    if path is not None:
        traj = arm.retime(path)
        return ctx.execute(traj)

    # Fallback: Cartesian lift then re-plan
    logger.info("Direct path to ready failed for %s, trying retract...", side)
    ctrl = CartesianController.from_arm(arm)
    ctrl.move(
        np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0]),
        dt=0.008, max_distance=0.15, step_fn=_make_step_fn(robot),
    )

    try:
        path = arm.plan_to_configuration(ready_config)
    except Exception:
        path = None

    if path is not None:
        traj = arm.retime(path)
        return ctx.execute(traj)

    logger.warning("Could not plan recovery path for %s arm", side)
    return False


# ---------------------------------------------------------------------------
# Pickup
# ---------------------------------------------------------------------------


def pickup(
    robot: Geodude,
    object_name: str,
    grasp_tsrs: list,
    *,
    arm: Arm | str | None = None,
    base_heights: list[float] | None = None,
    lift_height: float = 0.05,
    timeout: float = 30.0,
) -> bool:
    """Pick up an object using caller-provided grasp TSRs.

    Args:
        robot: The robot instance.
        object_name: MuJoCo body name of the object (e.g., "can_0").
        grasp_tsrs: List of TSR objects defining valid grasp poses.
            Generate these from tsr.hands using object geometry.
        arm: Arm to use ("left", "right", Arm instance), or None for both.
        base_heights: Base heights to search (default: [0.2, 0.0, 0.4]).
        lift_height: Height to lift after grasping (meters).
        timeout: Planning timeout per attempt (seconds).

    Returns:
        True if pickup succeeded.
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    if base_heights is None:
        base_heights = [0.2, 0.0, 0.4]

    # Try each arm
    arms = robot._resolve_arms(arm)
    for a in arms:
        side = a.config.name
        logger.info("Trying %s arm for %s...", side, object_name)

        # Plan approach
        result = robot.plan_to_tsrs(
            grasp_tsrs, arm=a, base_heights=base_heights, timeout=timeout,
        )
        if result is None:
            logger.warning("Planning FAILED for %s arm", side)
            _return_to_ready(robot, a)
            continue

        # Execute approach
        if not ctx.execute(result):
            logger.warning("Execution FAILED for %s arm", side)
            ctx.arm(side).release()
            _return_to_ready(robot, a)
            continue

        # Move forward until contact
        gripper = a.gripper
        if gripper is not None:
            ctrl = CartesianController.from_arm(a)
            touch_result = ctrl.move_until_contact(
                np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0]),  # 3cm/s in hand Z
                dt=0.008,
                gripper_body_names=gripper.gripper_body_names,
                max_distance=0.10,
                step_fn=_make_step_fn(robot),
            )
            if touch_result.success:
                logger.info("Contact at %.1fcm", touch_result.distance_moved * 100)

        # Grasp
        logger.info("Closing gripper on %s...", object_name)
        grasped = ctx.arm(side).grasp(object_name)
        if not grasped:
            logger.warning("Grasp FAILED for %s", object_name)
            ctx.arm(side).release()
            _return_to_ready(robot, a)
            continue

        # Lift
        if lift_height > 0:
            ctrl = CartesianController.from_arm(a)
            lift_result = ctrl.move(
                np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0]),
                dt=0.008, max_distance=lift_height,
                step_fn=_make_step_fn(robot),
            )
            robot.grasp_manager.update_attached_poses()
            logger.info("Lifted %.1fcm", lift_result.distance_moved * 100)

        logger.info("Picked up %s with %s arm", object_name, side)
        return True

    logger.warning("All pickup attempts failed for %s", object_name)
    return False


# ---------------------------------------------------------------------------
# Place
# ---------------------------------------------------------------------------


def place(
    robot: Geodude,
    place_tsrs: list,
    *,
    arm: Arm | str | None = None,
    base_heights: list[float] | None = None,
    timeout: float = 30.0,
) -> bool:
    """Place the held object using caller-provided place TSRs.

    Args:
        robot: The robot instance.
        place_tsrs: List of TSR objects defining valid place poses.
        arm: Arm holding the object, or None to auto-detect.
        base_heights: Base heights to search.
        timeout: Planning timeout (seconds).

    Returns:
        True if place succeeded.
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    # Find arm holding object
    if arm is None:
        for side in ("left", "right"):
            if robot.grasp_manager.get_grasped_by(side):
                a = robot.left_arm if side == "left" else robot.right_arm
                break
        else:
            logger.warning("No arm is holding an object")
            return False
    else:
        resolved = robot._resolve_arms(arm)
        a = resolved[0]

    side = a.config.name
    held_objects = list(robot.grasp_manager.get_grasped_by(side))
    if not held_objects:
        logger.warning("%s arm is not holding any object", side)
        return False
    held_object = held_objects[0]

    # Plan
    result = robot.plan_to_tsrs(
        place_tsrs, arm=a, base_heights=base_heights, timeout=timeout,
    )
    if result is None:
        logger.warning("Place planning FAILED")
        _return_to_ready(robot, a)
        return False

    # Execute
    if not ctx.execute(result):
        logger.warning("Place execution FAILED")
        _return_to_ready(robot, a)
        return False

    # Release
    ctx.arm(side).release(held_object)
    _return_to_ready(robot, a)
    logger.info("Placed %s", held_object)
    return True
