"""Jacobian-based Cartesian velocity control.

Provides closed-loop Cartesian control primitives like move_until_touch
for contact-based grasping. Works in both physics and kinematic modes.

Physics mode: Uses F/T sensor for contact detection
Kinematic mode: Uses MuJoCo collision detection (data.contact)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import mujoco
import numpy as np

if TYPE_CHECKING:
    from geodude.arm import Arm

logger = logging.getLogger(__name__)


@dataclass
class MoveUntilTouchResult:
    """Result of a move_until_touch operation.

    Attributes:
        success: True if contact was detected after min_distance
        terminated_by: What caused termination ("contact", "max_distance", "error")
        distance_moved: Actual distance traveled in meters
        final_force: F/T force reading at termination (physics mode only)
        final_torque: F/T torque reading at termination (physics mode only)
        contact_geom: Name of contacted geometry (kinematic mode only)
    """

    success: bool
    terminated_by: Literal["contact", "max_distance", "error"]
    distance_moved: float
    final_force: np.ndarray | None = None
    final_torque: np.ndarray | None = None
    contact_geom: str | None = None


@dataclass
class TwistExecutionResult:
    """Result of a twist execution operation.

    Attributes:
        terminated_by: What caused termination
        distance_moved: Total distance traveled by EE
        duration: Actual execution time
        final_pose: EE pose at termination
    """

    terminated_by: Literal["duration", "distance", "condition", "error"]
    distance_moved: float
    duration: float
    final_pose: np.ndarray


def get_ee_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    joint_vel_indices: list[int],
) -> np.ndarray:
    """Compute the 6x6 end-effector Jacobian for an arm.

    Uses MuJoCo's mj_jacSite to get the Jacobian at the EE site,
    then extracts columns for the arm's joints.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        ee_site_id: Site ID for end-effector
        joint_vel_indices: Indices into qvel for arm joints

    Returns:
        6x6 Jacobian matrix where:
        - Rows 0-2: Linear velocity (dx, dy, dz)
        - Rows 3-5: Angular velocity (wx, wy, wz)
    """
    # Get full Jacobians (3 x nv each)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)

    # Extract columns for this arm's joints
    n_joints = len(joint_vel_indices)
    J_pos = jacp[:, joint_vel_indices]  # 3 x n_joints
    J_rot = jacr[:, joint_vel_indices]  # 3 x n_joints

    # Stack to form 6 x n_joints Jacobian
    return np.vstack([J_pos, J_rot])


def step_twist(
    arm: "Arm",
    twist: np.ndarray,
    frame: str = "world",
    dt: float = 0.004,
) -> np.ndarray:
    """Execute one timestep of Cartesian velocity control.

    Uses the Jacobian pseudoinverse to compute joint velocities that
    achieve the desired Cartesian twist. Maintains orientation while
    translating by using the full 6x6 Jacobian.

    Args:
        arm: Arm instance
        twist: 6D twist [vx, vy, vz, wx, wy, wz] in m/s and rad/s
        frame: "world" or "hand" (tool frame)
        dt: Timestep duration in seconds

    Returns:
        New joint positions after the step
    """
    model = arm.model
    data = arm.data

    # Get Jacobian
    # Need joint velocity indices (same as qpos indices for revolute joints)
    joint_vel_indices = arm.joint_qpos_indices  # For UR5e, qvel = qpos indices
    J = get_ee_jacobian(model, data, arm.ee_site_id, joint_vel_indices)

    # Transform twist to world frame if needed
    if frame == "hand":
        # Get EE rotation matrix
        R = data.site_xmat[arm.ee_site_id].reshape(3, 3)
        # Transform linear and angular velocities
        twist_world = np.zeros(6)
        twist_world[:3] = R @ twist[:3]
        twist_world[3:] = R @ twist[3:]
        twist = twist_world

    # Compute joint velocities using pseudoinverse
    # J @ q_dot = x_dot  =>  q_dot = J^+ @ x_dot
    q_dot = np.linalg.pinv(J) @ twist

    # Integrate to get new positions
    q_current = arm.get_joint_positions()
    q_new = q_current + q_dot * dt

    # Clamp to joint limits
    lower, upper = arm.get_joint_limits()
    q_new = np.clip(q_new, lower, upper)

    return q_new


def check_gripper_contact_kinematic(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_body_names: list[str],
) -> str | None:
    """Check for gripper contact using MuJoCo collision detection.

    Scans data.contact for any contacts involving gripper bodies.
    Used in kinematic mode where F/T sensors aren't meaningful.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        gripper_body_names: Body names that are part of the gripper

    Returns:
        Name of contacted geom, or None if no contact
    """
    # Get body IDs for gripper
    gripper_body_ids = set()
    for name in gripper_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            gripper_body_ids.add(body_id)

    # Scan contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_body = model.geom_bodyid[contact.geom1]
        geom2_body = model.geom_bodyid[contact.geom2]

        # Check if either geom belongs to gripper
        if geom1_body in gripper_body_ids:
            # Contact! Return the other geom's name
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if geom2_body in gripper_body_ids:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)

    return None


def move_until_touch(
    arm: "Arm",
    direction: np.ndarray,
    distance: float,
    max_distance: float,
    max_force: float = 5.0,
    max_torque: float | None = None,
    speed: float = 0.02,
    frame: str = "world",
    physics: bool = True,
    viewer=None,
) -> MoveUntilTouchResult:
    """Move gripper in direction until contact or max_distance reached.

    Inspired by herbpy's MoveUntilTouch API. Moves the end-effector at
    constant velocity in the specified direction, stopping when:
    - Physics mode: F/T sensor force exceeds threshold
    - Kinematic mode: Collision detected in data.contact

    The `distance` parameter specifies the minimum distance to travel
    before contact detection activates (avoids false positives from
    initial contacts).

    Args:
        arm: Arm instance to control
        direction: Unit vector for direction of motion
        distance: Minimum distance to travel before contact checking (meters)
        max_distance: Maximum distance to travel - required safety limit (meters)
        max_force: Force magnitude threshold in Newtons (default 5.0)
        max_torque: Torque magnitude threshold in Nm (optional)
        speed: Cartesian velocity in m/s (default 0.02 = 2cm/s)
        frame: "world" or "hand" frame for direction vector
        physics: True for physics mode (F/T), False for kinematic (collision)
        viewer: Optional MuJoCo viewer to sync

    Returns:
        MoveUntilTouchResult with success status and details

    Example:
        # Move forward in gripper frame until contact
        result = move_until_touch(
            arm=robot.right_arm,
            direction=[0, 0, 1],  # Forward in gripper frame
            distance=0.01,        # Min 1cm before checking
            max_distance=0.05,    # Max 5cm
            max_force=3.0,        # Light touch
            frame="hand",
        )
        if result.success:
            robot.right_arm.close_gripper()
    """
    model = arm.model
    data = arm.data

    # Normalize direction
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    # Create twist (linear velocity only, preserve orientation)
    twist = np.zeros(6)
    twist[:3] = direction * speed

    # Control parameters
    dt = 0.004  # 4ms control timestep (250 Hz)
    total_distance = 0.0
    start_pos = data.site_xpos[arm.ee_site_id].copy()

    # Get gripper body names for kinematic contact detection
    gripper_body_names = arm.config.gripper_bodies if hasattr(arm.config, 'gripper_bodies') else []

    logger.debug(
        f"move_until_touch: direction={direction}, distance={distance:.3f}, "
        f"max_distance={max_distance:.3f}, speed={speed:.3f}, physics={physics}"
    )

    while total_distance < max_distance:
        # Compute new joint positions
        q_new = step_twist(arm, twist, frame=frame, dt=dt)

        if physics:
            # Physics mode: set joint targets and step simulation
            for i, qpos_idx in enumerate(arm.joint_qpos_indices):
                data.qpos[qpos_idx] = q_new[i]

            # Step physics
            mujoco.mj_step(model, data)

            # Sync viewer if provided
            if viewer is not None:
                viewer.sync()

            # Check F/T sensor after min_distance
            if total_distance >= distance:
                ft_result = arm.get_ft_sensor()
                if ft_result is not None:
                    force, torque = ft_result
                    force_magnitude = np.linalg.norm(force)

                    if force_magnitude > max_force:
                        logger.debug(
                            f"move_until_touch: contact detected! "
                            f"force={force_magnitude:.2f}N > {max_force:.2f}N"
                        )
                        return MoveUntilTouchResult(
                            success=True,
                            terminated_by="contact",
                            distance_moved=total_distance,
                            final_force=force,
                            final_torque=torque,
                        )

                    if max_torque is not None:
                        torque_magnitude = np.linalg.norm(torque)
                        if torque_magnitude > max_torque:
                            logger.debug(
                                f"move_until_touch: contact detected! "
                                f"torque={torque_magnitude:.2f}Nm > {max_torque:.2f}Nm"
                            )
                            return MoveUntilTouchResult(
                                success=True,
                                terminated_by="contact",
                                distance_moved=total_distance,
                                final_force=force,
                                final_torque=torque,
                            )
        else:
            # Kinematic mode: set positions directly and run forward kinematics
            for i, qpos_idx in enumerate(arm.joint_qpos_indices):
                data.qpos[qpos_idx] = q_new[i]

            mujoco.mj_forward(model, data)

            # Sync viewer if provided
            if viewer is not None:
                viewer.sync()

            # Check collision after min_distance
            if total_distance >= distance:
                contact_geom = check_gripper_contact_kinematic(
                    model, data, gripper_body_names
                )
                if contact_geom is not None:
                    logger.debug(
                        f"move_until_touch: contact detected with {contact_geom}"
                    )
                    return MoveUntilTouchResult(
                        success=True,
                        terminated_by="contact",
                        distance_moved=total_distance,
                        contact_geom=contact_geom,
                    )

        # Update distance
        current_pos = data.site_xpos[arm.ee_site_id]
        total_distance = np.linalg.norm(current_pos - start_pos)

    # Reached max_distance without contact
    logger.debug(
        f"move_until_touch: max_distance reached ({total_distance:.3f}m)"
    )

    # Get final F/T reading if physics
    final_force = None
    final_torque = None
    if physics:
        ft_result = arm.get_ft_sensor()
        if ft_result is not None:
            final_force, final_torque = ft_result

    return MoveUntilTouchResult(
        success=False,
        terminated_by="max_distance",
        distance_moved=total_distance,
        final_force=final_force,
        final_torque=final_torque,
    )


def execute_twist(
    arm: "Arm",
    twist: np.ndarray,
    frame: str = "world",
    duration: float | None = None,
    max_distance: float | None = None,
    until: Callable[[], bool] | None = None,
    physics: bool = True,
    viewer=None,
) -> TwistExecutionResult:
    """Execute Cartesian twist until a termination condition.

    Lower-level method that runs a constant twist (Cartesian velocity)
    until one of the termination conditions is met.

    Args:
        arm: Arm instance to control
        twist: 6D twist [vx, vy, vz, wx, wy, wz] in m/s and rad/s
        frame: "world" or "hand" frame for twist
        duration: Stop after this many seconds
        max_distance: Stop after EE moves this far (meters)
        until: Custom termination predicate (returns True to stop)
        physics: True for physics stepping, False for kinematic
        viewer: Optional MuJoCo viewer to sync

    Returns:
        TwistExecutionResult with termination info

    Example:
        # Move up at 5cm/s for 2 seconds
        result = execute_twist(
            arm=robot.right_arm,
            twist=[0, 0, 0.05, 0, 0, 0],  # 5cm/s up
            duration=2.0,
        )
    """
    model = arm.model
    data = arm.data

    dt = 0.004  # 4ms control timestep
    elapsed = 0.0
    start_pos = data.site_xpos[arm.ee_site_id].copy()

    logger.debug(
        f"execute_twist: twist={twist}, duration={duration}, "
        f"max_distance={max_distance}"
    )

    while True:
        # Check termination conditions
        current_pos = data.site_xpos[arm.ee_site_id]
        distance = np.linalg.norm(current_pos - start_pos)
        final_pose = arm.get_ee_pose()

        if duration is not None and elapsed >= duration:
            return TwistExecutionResult(
                terminated_by="duration",
                distance_moved=distance,
                duration=elapsed,
                final_pose=final_pose,
            )

        if max_distance is not None and distance >= max_distance:
            return TwistExecutionResult(
                terminated_by="distance",
                distance_moved=distance,
                duration=elapsed,
                final_pose=final_pose,
            )

        if until is not None and until():
            return TwistExecutionResult(
                terminated_by="condition",
                distance_moved=distance,
                duration=elapsed,
                final_pose=final_pose,
            )

        # Execute one step
        q_new = step_twist(arm, twist, frame=frame, dt=dt)

        for i, qpos_idx in enumerate(arm.joint_qpos_indices):
            data.qpos[qpos_idx] = q_new[i]

        if physics:
            mujoco.mj_step(model, data)
        else:
            mujoco.mj_forward(model, data)

        if viewer is not None:
            viewer.sync()

        elapsed += dt
