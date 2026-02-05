"""High-level manipulation primitives.

Provides pickup() and place() functions that use affordance discovery
to automatically plan and execute manipulation tasks.

Example:
    with robot.sim() as ctx:
        robot.pickup("can_0")           # Pick up specific object
        robot.place("recycle_bin_0")    # Place in specific destination

        robot.pickup(object_type="can") # Pick up any can
        robot.pickup()                  # Pick up any pickable object
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from geodude.arm import Arm
    from geodude.robot import Geodude

logger = logging.getLogger(__name__)


def _get_object_type(robot: "Geodude", object_name: str) -> str:
    """Extract object type from instance name.

    Args:
        robot: Robot instance
        object_name: Instance name like "can_0" or "recycle_bin_1"

    Returns:
        Object type like "can" or "recycle_bin"
    """
    # Strip trailing _N suffix
    match = re.match(r"^(.+?)(?:_\d+)?$", object_name)
    if match:
        return match.group(1)
    return object_name


def _get_object_pose(robot: "Geodude", object_name: str) -> np.ndarray | None:
    """Get object pose from MuJoCo.

    Args:
        robot: Robot instance
        object_name: Instance name

    Returns:
        3D position array, or None if not found
    """
    import mujoco

    try:
        body_id = mujoco.mj_name2id(
            robot.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        if body_id < 0:
            return None
        return robot.data.xpos[body_id].copy()
    except Exception:
        return None


def _find_arm_holding_object(robot: "Geodude") -> "Arm | None":
    """Find which arm is currently holding an object.

    Args:
        robot: Robot instance

    Returns:
        Arm holding an object, or None
    """
    for side in ["left", "right"]:
        grasped = robot.grasp_manager.get_grasped_by(side)
        if grasped:
            return robot.left_arm if side == "left" else robot.right_arm
    return None


def pickup(
    robot: "Geodude",
    target: str | None = None,
    *,
    object_type: str | None = None,
    arm: "Arm | str | None" = None,
    base_heights: list[float] | None = None,
    lift_height: float = 0.10,
    timeout: float = 30.0,
) -> bool:
    """Pick up an object using affordance-based planning.

    Uses the AffordanceRegistry to discover grasp TSRs for the target object,
    then plans and executes a grasp motion via the active execution context.

    Args:
        robot: The robot instance
        target: Specific object name (e.g., "can_0"), or None to pick up
               any object matching object_type or any pickable object
        object_type: Filter by object type (e.g., "can") if target is None
        arm: Specific arm to use ("left", "right", Arm instance),
            or None to let robot choose based on affordances
        base_heights: Base heights to search (default: [0.2, 0.0, 0.4])
        lift_height: Height to lift after grasping (meters)
        timeout: Planning timeout per attempt (seconds)

    Returns:
        True if pickup succeeded, False otherwise

    Raises:
        RuntimeError: If no execution context is active

    Example:
        with robot.sim() as ctx:
            # Pick up specific object
            success = robot.pickup("can_0")

            # Pick up any can
            success = robot.pickup(object_type="can")

            # Pick up with specific arm
            success = robot.pickup("can_0", arm="right")
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError(
            "No active execution context. Use 'with robot.sim() as ctx:'"
        )

    # Default base heights
    if base_heights is None:
        base_heights = [0.2, 0.0, 0.4]

    # Resolve target object
    if target is None:
        # Find any pickable object
        pickable = get_pickable_objects(robot, object_type=object_type)
        if not pickable:
            logger.warning("No pickable objects found")
            return False
        target = pickable[0]
        logger.info(f"Auto-selected target: {target}")

    # Get object info
    obj_type = _get_object_type(robot, target)
    obj_pose = _get_object_pose(robot, target)

    if obj_pose is None:
        logger.warning(f"Object not found in scene: {target}")
        return False

    # Get grasp affordances
    affordances = robot.affordances.get_grasp_affordances(obj_type)
    if not affordances:
        logger.warning(f"No grasp affordances found for object type: {obj_type}")
        return False

    # Resolve arms to try (randomize order for variety)
    import random

    arms = robot._resolve_arms(arm)
    random.shuffle(arms)

    # Group compatible affordances by arm
    from geodude.affordances import hand_types_compatible

    arm_affordances: dict["Arm", list] = {}
    for a in arms:
        hand_type = a.config.hand_type
        compatible = [
            aff for aff in affordances
            if hand_types_compatible(aff.hand_type, hand_type)
        ]
        if compatible:
            arm_affordances[a] = compatible

    if not arm_affordances:
        logger.warning(
            f"No affordances compatible with available hands for {obj_type}"
        )
        return False

    # Try each arm with all its compatible TSRs at once
    for a, affs in arm_affordances.items():
        aff_names = [aff.name for aff in affs]
        logger.debug(f"Trying {a.side} arm with TSRs: {aff_names}")

        # Create all TSRs at object pose - CBiRRT will find path to any of them
        grasp_tsrs = [aff.create_tsr(obj_pose) for aff in affs]

        # Plan approach (planner picks best TSR)
        result = a.plan_to_tsr(
            grasp_tsrs,
            base_heights=base_heights,
            timeout=timeout,
        )

        if result is None:
            logger.debug(f"Planning failed for {a.side} arm")
            continue

        # Execute approach
        ctx.execute(result)

        # Grasp
        grasped = ctx.arm(a.side).grasp(target)
        if not grasped:
            logger.debug(f"Grasp failed for {target}")
            continue

        # Lift
        if lift_height > 0:
            lift_pose = a.get_ee_pose().copy()
            lift_pose[2, 3] += lift_height
            lift_result = a.plan_to_pose(lift_pose, timeout=5.0)
            if lift_result:
                ctx.execute(lift_result)
                robot.grasp_manager.update_attached_poses()

        logger.info(f"Successfully picked up {target} with {a.side} arm")
        return True

    logger.warning(f"All pickup attempts failed for {target}")
    return False


def place(
    robot: "Geodude",
    destination: str,
    *,
    arm: "Arm | str | None" = None,
    base_heights: list[float] | None = None,
    timeout: float = 30.0,
) -> bool:
    """Place a held object at a destination.

    Uses the AffordanceRegistry to discover place TSRs for the destination,
    then plans and executes a place motion via the active execution context.

    Args:
        robot: The robot instance
        destination: Destination name (e.g., "recycle_bin_0", "table")
        arm: Arm holding the object, or None to auto-detect
        base_heights: Base heights to search (default: [0.2, 0.0, 0.4])
        timeout: Planning timeout (seconds)

    Returns:
        True if place succeeded, False otherwise

    Raises:
        RuntimeError: If no execution context is active

    Example:
        with robot.sim() as ctx:
            robot.pickup("can_0")
            success = robot.place("recycle_bin_0")
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError(
            "No active execution context. Use 'with robot.sim() as ctx:'"
        )

    # Note: Unlike pickup(), we don't default base_heights for place.
    # Moving the base while holding an object is more complex and the
    # arm usually has enough reach from the grasp position.

    # Find which arm is holding something
    if arm is None:
        holding_arm = _find_arm_holding_object(robot)
        if holding_arm is None:
            logger.warning("No arm is currently holding an object")
            return False
        a = holding_arm
    else:
        resolved = robot._resolve_arms(arm)
        if not resolved:
            logger.warning(f"Invalid arm specification: {arm}")
            return False
        a = resolved[0]

    # Get held object info
    held_objects = list(robot.grasp_manager.get_grasped_by(a.side))
    if not held_objects:
        logger.warning(f"{a.side} arm is not holding any object")
        return False

    held_object = held_objects[0]
    held_type = _get_object_type(robot, held_object)

    # Get destination pose
    dest_pose = _get_object_pose(robot, destination)
    if dest_pose is None:
        logger.warning(f"Destination not found in scene: {destination}")
        return False

    dest_type = _get_object_type(robot, destination)

    # Get place affordances for destination
    affordances = robot.affordances.get_place_affordances(
        object_type=held_type,
        destination_type=dest_type,
        hand_type=a.config.hand_type,
    )

    if not affordances:
        logger.warning(
            f"No place affordances found for {held_type} at {dest_type}"
        )
        return False

    # Create all place TSRs - CBiRRT will find path to any of them
    aff_names = [aff.name for aff in affordances]
    logger.debug(f"Trying place with TSRs: {aff_names}")

    place_tsrs = [aff.create_tsr(dest_pose) for aff in affordances]

    # Plan place motion (planner picks best TSR)
    result = a.plan_to_tsr(
        place_tsrs,
        base_heights=base_heights,
        timeout=timeout,
    )

    if result is None:
        logger.warning(f"All place attempts failed for {destination}")
        return False

    # Execute place motion
    ctx.execute(result)

    # Release
    ctx.arm(a.side).release(held_object)

    logger.info(f"Successfully placed {held_object} at {destination}")
    return True


def get_pickable_objects(
    robot: "Geodude",
    object_type: str | None = None,
) -> list[str]:
    """Get names of objects that can be picked up.

    Queries the scene for objects that have grasp affordances registered
    and are compatible with the robot's hands.

    Args:
        robot: Robot instance
        object_type: Filter by object type (e.g., "can")

    Returns:
        List of object instance names that can be picked up
    """
    import mujoco

    # Get object types with grasp affordances
    registry = robot.affordances
    pickable_types = set()

    for obj_type in registry.get_object_types():
        grasps = registry.get_grasp_affordances(obj_type)
        if grasps:
            # Check if any affordance is compatible with our hands
            for arm in [robot.left_arm, robot.right_arm]:
                hand_type = arm.config.hand_type
                from geodude.affordances import hand_types_compatible

                for aff in grasps:
                    if hand_types_compatible(aff.hand_type, hand_type):
                        pickable_types.add(obj_type)
                        break

    if object_type:
        pickable_types = {object_type} if object_type in pickable_types else set()

    # Find instances of these types in the scene
    pickable = []
    n_bodies = robot.model.nbody
    for i in range(n_bodies):
        name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            body_type = _get_object_type(robot, name)
            if body_type in pickable_types:
                # Skip if already grasped
                if not robot.grasp_manager.is_grasped(name):
                    pickable.append(name)

    return pickable


def get_place_destinations(
    robot: "Geodude",
    object_type: str,
) -> list[str]:
    """Get valid place destinations for an object type.

    Args:
        robot: Robot instance
        object_type: Type of object being placed (e.g., "can")

    Returns:
        List of destination instance names
    """
    import mujoco

    # Get destination types with place affordances for this object
    registry = robot.affordances
    dest_types = set()

    # Look through all place affordances
    for task in registry.PLACE_TASKS:
        for obj_type in registry.get_object_types():
            affordances = registry.get_affordances(obj_type, task=task)
            # The template's reference (obj_type) is the destination
            if affordances:
                dest_types.add(obj_type)

    # Find instances of destination types in the scene
    destinations = []
    n_bodies = robot.model.nbody
    for i in range(n_bodies):
        name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            body_type = _get_object_type(robot, name)
            if body_type in dest_types:
                destinations.append(name)

    return destinations
