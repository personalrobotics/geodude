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
    """Get object pose (4x4 transformation matrix) from MuJoCo.

    Args:
        robot: Robot instance
        object_name: Instance name

    Returns:
        4x4 homogeneous transformation matrix, or None if not found
    """
    import mujoco

    try:
        body_id = mujoco.mj_name2id(
            robot.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        if body_id < 0:
            return None

        # Get position and orientation
        pos = robot.data.xpos[body_id].copy()
        rot_mat = robot.data.xmat[body_id].reshape(3, 3).copy()

        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos
        return T
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


def _has_grasped_object_arm_collision(robot: "Geodude", arm: "Arm") -> bool:
    """Check if grasped object is colliding with arm (not gripper).

    In physics mode, the grasped object can swing into the forearm.
    This is a recoverable condition - we can try to separate them.

    Args:
        robot: Robot instance
        arm: Arm to check

    Returns:
        True if grasped object is touching non-gripper arm parts
    """
    import mujoco

    collision_checker = arm._get_collision_checker()
    model = collision_checker.model
    data = collision_checker.data

    # Get grasped objects for this arm
    grasped = list(robot.grasp_manager.get_grasped_by(arm.side))
    if not grasped:
        return False

    grasped_set = set(grasped)

    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]

        body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
        body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)

        # Check if one is grasped and other is arm (but not gripper)
        body1_is_grasped = body1_name in grasped_set
        body2_is_grasped = body2_name in grasped_set
        body1_is_arm = body1 in collision_checker._arm_body_ids
        body2_is_arm = body2 in collision_checker._arm_body_ids
        body1_is_gripper = body1_is_arm and body1_name and "gripper" in body1_name
        body2_is_gripper = body2_is_arm and body2_name and "gripper" in body2_name

        # Grasped object touching non-gripper arm part
        if body1_is_grasped and body2_is_arm and not body2_is_gripper:
            logger.debug(f"Grasped-to-arm collision: {body1_name} <-> {body2_name}")
            return True
        if body2_is_grasped and body1_is_arm and not body1_is_gripper:
            logger.debug(f"Grasped-to-arm collision: {body2_name} <-> {body1_name}")
            return True

    return False


def _separate_grasped_object(robot: "Geodude", arm: "Arm") -> bool:
    """Try to separate grasped object from arm with small motions.

    When the grasped object is touching the forearm, try lifting up
    and/or wiggling to separate them.

    Args:
        robot: Robot instance
        arm: Arm with grasped object

    Returns:
        True if separation successful, False otherwise
    """
    from geodude.cartesian import execute_twist, CartesianControlConfig

    ctx = robot._active_context
    if ctx is None:
        return False

    logger.info(f"  Attempting to separate grasped object from {arm.side} arm...")

    # Try lifting up a bit more
    lift_config = CartesianControlConfig(min_progress=0.05)
    twist = np.array([0.0, 0.0, 0.15, 0.0, 0.0, 0.0])  # 15 cm/s upward

    lift_result = execute_twist(
        arm=arm,
        twist=twist,
        max_distance=0.05,  # 5cm additional lift
        frame="world",
        config=lift_config,
    )

    if lift_result.distance_moved > 0.02:
        logger.info(f"    Lifted additional {lift_result.distance_moved*100:.1f}cm")
        robot.grasp_manager.update_attached_poses()

    # Check if separated
    if not _has_grasped_object_arm_collision(robot, arm):
        logger.info(f"  Separation successful")
        return True

    # Try a small rotation to shift the object
    logger.debug(f"  Trying rotation to separate...")
    rotate_config = CartesianControlConfig(min_progress=0.05)
    # Small wrist rotation
    rotate_twist = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0])  # 0.3 rad/s rotation

    rotate_result = execute_twist(
        arm=arm,
        twist=rotate_twist,
        duration=0.3,  # Short rotation
        frame="hand",
        config=rotate_config,
    )

    robot.grasp_manager.update_attached_poses()

    # Check if separated
    if not _has_grasped_object_arm_collision(robot, arm):
        logger.info(f"  Separation successful after rotation")
        return True

    logger.warning(f"  Could not separate grasped object from arm")
    return False


def _return_to_ready(robot: "Geodude", arm: "Arm") -> bool:
    """Return arm to ready position after failure.

    Used for recovery when a grasp fails, so the arm doesn't block
    the other arm from attempting the same target.

    If planning to ready fails, attempts a simple retract (lift up) first.
    As a last resort in physics mode, directly commands the arm to ready.

    Args:
        robot: Robot instance
        arm: Arm to return to ready

    Returns:
        True if successful, False if planning or execution failed
    """
    import numpy as np

    ctx = robot._active_context
    if ctx is None:
        return False

    if "ready" not in robot.named_poses or arm.side not in robot.named_poses["ready"]:
        logger.warning(f"No ready pose defined for {arm.side} arm")
        return False

    # Get recovery config
    recovery_cfg = robot.config.physics.recovery

    ready_config = np.array(robot.named_poses["ready"][arm.side])
    trajectory = arm.plan_to(ready_config)

    if trajectory is None:
        # Fallback: use Cartesian velocity to lift up first, then plan to ready
        # execute_twist is more robust than planning - just moves until it can't
        logger.info(f"Direct path to ready failed for {arm.side} arm, trying retract first...")
        from geodude.cartesian import execute_twist, CartesianControlConfig

        # Move straight up in world frame
        retract_config = CartesianControlConfig(min_progress=0.1)
        twist = np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0])  # 10 cm/s upward
        retract_result = execute_twist(
            arm=arm,
            twist=twist,
            max_distance=recovery_cfg.retract_height,
            frame="world",
            config=retract_config,
        )

        if retract_result.distance_moved > 0.01:  # Moved at least 1cm
            logger.info(f"  Retracted {retract_result.distance_moved*100:.1f}cm")
            # Now try planning to ready again
            trajectory = arm.plan_to(ready_config)

    if trajectory is None:
        # Planning failed - try forced recovery in physics mode
        # First lift arm up (move shoulder_lift and elbow to clear workspace)
        # then move to ready. This is safer than direct interpolation.
        if hasattr(ctx, '_physics') and ctx._physics and hasattr(ctx, '_controller'):
            logger.warning(f"Planning failed - attempting safe forced recovery for {arm.side} arm")
            controller = ctx._controller
            if controller is not None and arm.side in controller._arms:
                arm_info = controller._arms[arm.side]
                current_pos = arm_info["target_position"].copy()

                # Phase 1: Lift arm by moving shoulder_lift and elbow to clear workspace
                lift_config = current_pos.copy()
                lift_config[1] = recovery_cfg.lift_shoulder_angle
                lift_config[2] = recovery_cfg.lift_elbow_angle

                # Interpolate to lift position
                steps = recovery_cfg.interpolation_steps
                for i in range(steps):
                    t = (i + 1) / steps
                    t_smooth = t * t * (3 - 2 * t)  # Smoothstep
                    arm_info["target_position"] = current_pos + t_smooth * (lift_config - current_pos)
                    controller.step()

                # Phase 2: Move to ready position
                lift_pos = arm_info["target_position"].copy()
                for i in range(steps):
                    t = (i + 1) / steps
                    t_smooth = t * t * (3 - 2 * t)
                    arm_info["target_position"] = lift_pos + t_smooth * (ready_config - lift_pos)
                    controller.step()

                ctx.sync()
                return True
        logger.warning(f"Could not plan recovery path for {arm.side} arm")
        return False

    return ctx.execute(trajectory)


def pickup(
    robot: "Geodude",
    target: str | None = None,
    *,
    object_type: str | None = None,
    arm: "Arm | str | None" = None,
    base_heights: list[float] | None = None,
    lift_height: float = 0.05,  # 5cm lift (collision-checked)
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
        # Re-read object pose before each attempt (object may have moved/fallen)
        obj_pose = _get_object_pose(robot, target)
        if obj_pose is None:
            logger.warning(f"Object {target} no longer found in scene")
            return False

        aff_names = [aff.name for aff in affs]
        logger.info(f"Trying {a.side} arm with TSRs: {aff_names}")

        # Create all TSRs at CURRENT object pose - CBiRRT will find path to any of them
        grasp_tsrs = [aff.create_tsr(obj_pose) for aff in affs]

        # Plan approach (planner picks best TSR)
        logger.info(f"  Planning approach to {target}...")
        result = a.plan_to_tsr(
            grasp_tsrs,
            base_heights=base_heights,
            timeout=timeout,
        )

        if result is None:
            logger.warning(f"  Planning FAILED for {a.side} arm (timeout or collision)")
            # Return to ready even on planning failure to clear the workspace
            _return_to_ready(robot, a)
            continue

        # Execute approach
        # result can be Trajectory or PlanResult - get waypoint count for logging
        from geodude.planning import PlanResult
        if isinstance(result, PlanResult):
            waypoints = result.arm_trajectory.num_waypoints
        else:
            waypoints = result.num_waypoints
        logger.info(f"  Executing approach trajectory ({waypoints} waypoints)...")
        exec_success = ctx.execute(result)
        if not exec_success:
            logger.warning(f"  Execution FAILED - arm did not converge to target")
            # Recovery: return to ready
            ctx.arm(a.side).release()
            _return_to_ready(robot, a)
            continue
        logger.info(f"  Approach execution completed")

        # Move forward until contact (close the standoff gap)
        from geodude.cartesian import move_until_touch

        # Check for pre-existing arm collision before starting motion
        # In physics mode, the arm might have sagged into collision with the base
        collision_checker = a._get_collision_checker()
        if collision_checker.is_arm_in_collision():
            logger.warning(
                f"  Arm already in collision before approach - skipping to other arm"
            )
            _return_to_ready(robot, a)
            continue

        logger.info(f"  Moving forward until contact...")
        touch_result = move_until_touch(
            arm=a,
            direction=[0, 0, 1],  # Forward in gripper frame (+Z is gripper approach)
            distance=0.01,        # Min 1cm before checking (avoid false positives)
            max_distance=0.10,    # Max 10cm (slightly more than typical standoff)
            max_force=10.0,       # Contact force threshold
            speed=0.03,           # 3cm/s - slow for precision
            frame="hand",
            check_arm_collision=True,  # Stop if arm hits environment
        )

        if touch_result.success:
            logger.info(f"  Contact detected at {touch_result.distance_moved*100:.1f}cm")
        elif touch_result.terminated_by == "arm_collision":
            logger.warning(
                f"  Arm collision during approach at {touch_result.distance_moved*100:.1f}cm"
            )
            # Return to ready and try another arm
            _return_to_ready(robot, a)
            continue
        else:
            logger.info(
                f"  Move stopped: {touch_result.terminated_by} "
                f"at {touch_result.distance_moved*100:.1f}cm"
            )

        # Grasp
        logger.info(f"  Closing gripper on {target}...")
        grasped = ctx.arm(a.side).grasp(target)
        if not grasped:
            logger.warning(f"  Grasp FAILED - no contact detected with {target}")
            # Check if object moved/fell (Z is at [2,3] in 4x4 pose matrix)
            new_pose = _get_object_pose(robot, target)
            if new_pose is not None:
                old_z = obj_pose[2, 3]
                new_z = new_pose[2, 3]
                if new_z < old_z - 0.1:  # Object fell more than 10cm
                    logger.warning(f"  Object {target} appears to have fallen (z: {old_z:.3f} -> {new_z:.3f})")
            # Open gripper and return to ready before trying another arm
            ctx.arm(a.side).release()
            _return_to_ready(robot, a)
            continue
        logger.info(f"  Grasp succeeded - holding {grasped}")

        # Lift using Cartesian velocity control (simpler and more robust than planning)
        if lift_height > 0:
            logger.info(f"  Lifting {lift_height*100:.0f}cm...")

            from geodude.cartesian import execute_twist, CartesianControlConfig

            # Move straight up in world frame
            # Use lower min_progress to allow slower motion near joint limits
            # Use higher velocity (20cm/s) to overcome contact forces in physics mode
            lift_config = CartesianControlConfig(min_progress=0.05)
            twist = np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0])  # 20 cm/s upward
            lift_result = execute_twist(
                arm=a,
                twist=twist,
                max_distance=lift_height,
                frame="world",
                config=lift_config,
                check_arm_collision=True,  # Stop if arm hits environment
            )

            if lift_result.terminated_by == "arm_collision":
                logger.warning(
                    f"  Arm collision during lift at {lift_result.distance_moved*100:.1f}cm - aborting grasp"
                )
                # Release object and try another arm
                ctx.arm(a.side).release()
                _return_to_ready(robot, a)
                continue

            if lift_result.distance_moved >= lift_height * 0.9:
                robot.grasp_manager.update_attached_poses()
                logger.info(f"  Lift completed ({lift_result.distance_moved*100:.1f}cm)")
            else:
                # Partial lift is OK - update attached poses and continue
                robot.grasp_manager.update_attached_poses()
                logger.warning(
                    f"  Lift incomplete: {lift_result.distance_moved*100:.1f}cm "
                    f"(target: {lift_height*100:.0f}cm, stopped by: {lift_result.terminated_by})"
                )

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
    logger.info(f"Placing {held_object} at {destination} with TSRs: {aff_names}")

    place_tsrs = [aff.create_tsr(dest_pose) for aff in affordances]

    # Log arm state before planning (debug level)
    q_current = a.get_joint_positions()
    ee_pose = a.get_ee_pose()
    logger.debug(f"  Current joint positions: {q_current.round(3)}")
    logger.debug(f"  Current EE position: {ee_pose[:3, 3].round(3)}")

    # Check if current configuration is in collision
    collision_checker = a._get_collision_checker()
    is_collision_free = collision_checker.is_valid(q_current)
    logger.debug(f"  Current config collision-free: {is_collision_free}")

    # Debug contacts if in collision
    if not is_collision_free:
        logger.debug(f"  Collision detected - dumping contacts:")
        collision_checker.debug_contacts(q_current)

    # Log what object is being held (debug level)
    logger.debug(f"  Holding object: {held_object} (type: {held_type})")
    logger.debug(f"  Destination: {destination} (type: {dest_type})")
    logger.debug(f"  Destination pose:\n{dest_pose.round(3)}")

    # Plan place motion (planner picks best TSR)
    # May need retry if grasped object is colliding with arm
    max_retries = 2
    result = None

    for attempt in range(max_retries):
        if attempt > 0:
            logger.info(f"  Retry {attempt}: Planning place motion...")
        else:
            logger.info(f"  Planning place motion...")

        result = a.plan_to_tsr(
            place_tsrs,
            base_heights=base_heights,
            timeout=timeout,
        )

        if result is not None:
            break

        # Planning failed - check if recoverable
        logger.warning(f"  Place planning FAILED for {destination} (timeout or collision)")
        logger.debug(f"  Held objects by {a.side}: {list(robot.grasp_manager.get_grasped_by(a.side))}")

        # Check if failure is due to grasped-object-to-arm collision
        if attempt < max_retries - 1 and _has_grasped_object_arm_collision(robot, a):
            logger.info(f"  Detected grasped object touching arm - attempting recovery...")
            if _separate_grasped_object(robot, a):
                continue  # Retry planning
            else:
                break  # Recovery failed

    if result is None:
        # All retries exhausted
        _return_to_ready(robot, a)
        return False

    # Execute place motion (with retry on failure)
    # result can be Trajectory or PlanResult
    from geodude.planning import PlanResult
    if isinstance(result, PlanResult):
        waypoints = result.arm_trajectory.num_waypoints
    else:
        waypoints = result.num_waypoints

    exec_success = False
    for exec_attempt in range(2):
        if exec_attempt > 0:
            logger.info(f"  Retry {exec_attempt}: Executing place trajectory...")
            # Let physics settle before retry
            for _ in range(50):
                ctx.step()
        else:
            logger.info(f"  Executing place trajectory ({waypoints} waypoints)...")

        exec_success = ctx.execute(result)
        if exec_success:
            break

        logger.warning(f"  Place execution failed (attempt {exec_attempt + 1}) - arm did not converge")

    if not exec_success:
        logger.warning(f"  Place execution FAILED after retries")
        # Return to ready with object still held
        _return_to_ready(robot, a)
        return False
    logger.info(f"  Place execution completed")

    # Release
    logger.info(f"  Opening gripper to release {held_object}...")
    ctx.arm(a.side).release(held_object)

    # Return to ready after successful place
    logger.info(f"  Returning to ready position...")
    _return_to_ready(robot, a)

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
