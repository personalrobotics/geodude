"""Jacobian-based Cartesian velocity control.

Provides robust Cartesian control primitives for contact-based grasping.
The core function `twist_to_joint_velocity()` handles:
- Singularity robustness via damped least squares
- Joint velocity limit scaling (preserves Cartesian direction)
- Joint position limit avoidance via null-space repulsion

Works in both physics mode (F/T sensor) and kinematic mode (data.contact).
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


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CartesianControlConfig:
    """Configuration for Cartesian velocity control.

    User-friendly parameters that control the behavior of twist execution.
    All parameters have sensible defaults for manipulation tasks.

    Attributes:
        velocity_scale: Fraction of robot's max joint velocity to use (0-1).
                       Lower = safer/slower, higher = faster but riskier.
                       Default 0.5 = use half of max velocity.

        joint_margin_deg: Degrees from joint limits to start slowing down.
                         When a joint is within this margin of its limit,
                         null-space motion pushes it away.
                         Default 10° provides good safety margin.

        singularity_robustness: How much to prioritize stability over accuracy
                               near singularities (0-1).
                               0 = pure accuracy (may be unstable)
                               1 = pure stability (may be slow)
                               Default 0.3 = slight preference for stability.

        min_progress_threshold: Minimum fraction of commanded velocity that
                               must be achievable to continue (0-1).
                               If we can only achieve less than this, declare
                               that we can't make meaningful progress.
                               Default 0.1 = require at least 10% progress.
    """

    velocity_scale: float = 0.5
    joint_margin_deg: float = 10.0
    singularity_robustness: float = 0.3
    min_progress_threshold: float = 0.1

    def __post_init__(self):
        """Validate configuration values."""
        if not 0 < self.velocity_scale <= 1:
            raise ValueError(f"velocity_scale must be in (0, 1], got {self.velocity_scale}")
        if self.joint_margin_deg < 0:
            raise ValueError(f"joint_margin_deg must be >= 0, got {self.joint_margin_deg}")
        if not 0 <= self.singularity_robustness <= 1:
            raise ValueError(f"singularity_robustness must be in [0, 1], got {self.singularity_robustness}")


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class TwistStepResult:
    """Result of a single twist-to-joint-velocity computation.

    Provides interpretable metrics about the robot's current state and
    how well it can execute the commanded twist.

    Attributes:
        joint_velocities: Computed joint velocities (rad/s)
        achieved_fraction: Fraction of commanded twist that will be achieved (0-1).
                          1.0 = full twist achieved
                          <1.0 = had to scale down due to limits
                          0.0 = cannot make progress

        manipulability: Ability to move in all Cartesian directions (0-1).
                       High (>0.5) = good dexterity
                       Low (<0.1) = near singularity, motion constrained

        joint_headroom: How far from the closest joint limit (0-1).
                       1.0 = all joints at center of range
                       0.0 = at least one joint at its limit

        limiting_factor: What's currently limiting performance.
                        None = no limitations
                        "velocity" = joint velocity limit reached
                        "singularity" = near singular configuration
                        "joint_limit" = near joint position limit
    """

    joint_velocities: np.ndarray
    achieved_fraction: float
    manipulability: float
    joint_headroom: float
    limiting_factor: str | None = None


@dataclass
class MoveUntilTouchResult:
    """Result of a move_until_touch operation.

    Attributes:
        success: True if contact was detected after min_distance
        terminated_by: What caused termination ("contact", "max_distance", "no_progress")
        distance_moved: Actual distance traveled in meters
        final_force: F/T force reading at termination (physics mode only)
        final_torque: F/T torque reading at termination (physics mode only)
        contact_geom: Name of contacted geometry (kinematic mode only)
    """

    success: bool
    terminated_by: Literal["contact", "max_distance", "no_progress"]
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

    terminated_by: Literal["duration", "distance", "condition", "no_progress"]
    distance_moved: float
    duration: float
    final_pose: np.ndarray


# =============================================================================
# Core Jacobian Functions
# =============================================================================


def get_ee_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    joint_vel_indices: list[int],
) -> np.ndarray:
    """Compute the 6xN end-effector Jacobian for an arm.

    Uses MuJoCo's mj_jacSite to get the Jacobian at the EE site,
    then extracts columns for the arm's joints.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        ee_site_id: Site ID for end-effector
        joint_vel_indices: Indices into qvel for arm joints

    Returns:
        6xN Jacobian matrix where:
        - Rows 0-2: Linear velocity (dx, dy, dz)
        - Rows 3-5: Angular velocity (wx, wy, wz)
    """
    # Get full Jacobians (3 x nv each)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)

    # Extract columns for this arm's joints
    J_pos = jacp[:, joint_vel_indices]  # 3 x n_joints
    J_rot = jacr[:, joint_vel_indices]  # 3 x n_joints

    # Stack to form 6 x n_joints Jacobian
    return np.vstack([J_pos, J_rot])


def compute_manipulability(J: np.ndarray) -> float:
    """Compute Yoshikawa's manipulability measure.

    μ = sqrt(det(J @ J.T))

    This measures the robot's ability to move in all Cartesian directions.
    - μ > 0.1: Good manipulability
    - μ ≈ 0: At or near a singularity

    Args:
        J: 6xN Jacobian matrix

    Returns:
        Manipulability measure (0 to ~1, higher is better)
    """
    JJT = J @ J.T
    det_val = np.linalg.det(JJT)
    if det_val <= 0:
        return 0.0
    # Normalize to roughly 0-1 range for UR5e
    # UR5e has max manipulability around 0.01, so scale up
    return min(1.0, np.sqrt(det_val) * 10)


def compute_joint_headroom(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute minimum distance to any joint limit as a fraction.

    Returns 1.0 if all joints are at the center of their range,
    0.0 if any joint is at its limit.

    Args:
        q: Current joint positions
        lower: Lower joint limits
        upper: Upper joint limits

    Returns:
        Headroom fraction (0-1)
    """
    ranges = upper - lower
    # Distance from lower limit as fraction of range
    from_lower = (q - lower) / ranges
    # Distance from upper limit as fraction of range
    from_upper = (upper - q) / ranges
    # Minimum headroom for each joint (0.5 = centered)
    headroom_per_joint = np.minimum(from_lower, from_upper)
    # Return minimum across all joints, scaled to 0-1 (0.5 -> 1.0)
    return float(np.min(headroom_per_joint) * 2)


def compute_joint_limit_repulsion(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    margin_rad: float,
) -> np.ndarray:
    """Compute repulsive velocity to push joints away from limits.

    When a joint is within `margin_rad` of its limit, this returns a
    velocity that pushes it back toward the center of its range.

    Args:
        q: Current joint positions
        lower: Lower joint limits
        upper: Upper joint limits
        margin_rad: Distance from limit to start pushing (radians)

    Returns:
        Repulsive joint velocities (rad/s)
    """
    n = len(q)
    repulsion = np.zeros(n)

    for i in range(n):
        dist_lower = q[i] - lower[i]
        dist_upper = upper[i] - q[i]

        # Repel from lower limit
        if dist_lower < margin_rad:
            strength = 1.0 - (dist_lower / margin_rad)
            repulsion[i] += strength * 0.5  # 0.5 rad/s max repulsion

        # Repel from upper limit
        if dist_upper < margin_rad:
            strength = 1.0 - (dist_upper / margin_rad)
            repulsion[i] -= strength * 0.5

    return repulsion


# =============================================================================
# Core Control Function
# =============================================================================


def twist_to_joint_velocity(
    J: np.ndarray,
    twist: np.ndarray,
    q_current: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    velocity_limits: np.ndarray,
    config: CartesianControlConfig | None = None,
) -> TwistStepResult:
    """Convert a Cartesian twist to joint velocities with constraint handling.

    This is the core function that handles:
    1. Singularity robustness via damped least squares
    2. Joint velocity limit scaling (preserves Cartesian direction)
    3. Joint position limit avoidance via null-space repulsion

    Args:
        J: 6xN Jacobian matrix
        twist: 6D twist [vx, vy, vz, wx, wy, wz] in m/s and rad/s
        q_current: Current joint positions
        lower_limits: Lower joint position limits
        upper_limits: Upper joint position limits
        velocity_limits: Maximum joint velocities (rad/s)
        config: Control configuration (uses defaults if None)

    Returns:
        TwistStepResult with joint velocities and metrics
    """
    if config is None:
        config = CartesianControlConfig()

    n_joints = J.shape[1]
    m_task = J.shape[0]

    # Scale velocity limits by config
    scaled_vel_limits = velocity_limits * config.velocity_scale

    # ==========================================================================
    # Step 1: Compute metrics
    # ==========================================================================

    manipulability = compute_manipulability(J)
    joint_headroom = compute_joint_headroom(q_current, lower_limits, upper_limits)

    # ==========================================================================
    # Step 2: Damped Least Squares
    # ==========================================================================

    # Adaptive damping based on manipulability and config
    # Low manipulability -> higher damping for stability
    # High singularity_robustness -> higher base damping
    lambda_base = 0.01 + 0.1 * config.singularity_robustness

    if manipulability < 0.1:
        # Near singularity - increase damping significantly
        lambda_damp = lambda_base + 0.2 * (1 - manipulability / 0.1)
    else:
        lambda_damp = lambda_base

    # Compute damped pseudoinverse: J^T (J J^T + λ²I)^-1
    JJT = J @ J.T
    damped_inv = J.T @ np.linalg.inv(JJT + lambda_damp**2 * np.eye(m_task))

    # Primary task: achieve desired twist
    q_dot_task = damped_inv @ twist

    # ==========================================================================
    # Step 3: Null-space repulsion from joint limits
    # ==========================================================================

    margin_rad = np.deg2rad(config.joint_margin_deg)
    q_dot_repulsion = compute_joint_limit_repulsion(
        q_current, lower_limits, upper_limits, margin_rad
    )

    # Project repulsion into null space (won't affect Cartesian motion)
    J_pinv = np.linalg.pinv(J)
    null_projector = np.eye(n_joints) - J_pinv @ J
    q_dot_null = null_projector @ q_dot_repulsion

    # Combine task and null-space motion
    q_dot = q_dot_task + q_dot_null

    # ==========================================================================
    # Step 4: Scale for velocity limits (preserve Cartesian direction)
    # ==========================================================================

    limiting_factor = None
    achieved_fraction = 1.0

    # Check which joints exceed velocity limits
    abs_qd = np.abs(q_dot)
    violations = abs_qd > scaled_vel_limits

    if np.any(violations):
        # Compute scale factors needed for each violating joint
        scale_factors = scaled_vel_limits[violations] / abs_qd[violations]
        scale = float(np.min(scale_factors))

        # Scale down entire solution to maintain direction
        q_dot = q_dot * scale
        achieved_fraction = scale
        limiting_factor = "velocity"

    # ==========================================================================
    # Step 5: Check for singularity limitation
    # ==========================================================================

    if manipulability < 0.05 and limiting_factor is None:
        limiting_factor = "singularity"
        # Estimate achieved fraction from manipulability
        achieved_fraction = min(achieved_fraction, manipulability / 0.05)

    # ==========================================================================
    # Step 6: Check for joint limit limitation
    # ==========================================================================

    if joint_headroom < 0.1 and limiting_factor is None:
        limiting_factor = "joint_limit"

    return TwistStepResult(
        joint_velocities=q_dot,
        achieved_fraction=achieved_fraction,
        manipulability=manipulability,
        joint_headroom=joint_headroom,
        limiting_factor=limiting_factor,
    )


# =============================================================================
# Higher-Level Control Functions
# =============================================================================


def step_twist(
    arm: "Arm",
    twist: np.ndarray,
    frame: str = "world",
    dt: float = 0.004,
    config: CartesianControlConfig | None = None,
) -> tuple[np.ndarray, TwistStepResult]:
    """Execute one timestep of Cartesian velocity control.

    Uses the robust `twist_to_joint_velocity()` function to compute joint
    velocities that achieve the desired Cartesian twist while respecting
    all constraints.

    Args:
        arm: Arm instance
        twist: 6D twist [vx, vy, vz, wx, wy, wz] in m/s and rad/s
        frame: "world" or "hand" (tool frame)
        dt: Timestep duration in seconds
        config: Control configuration (uses defaults if None)

    Returns:
        Tuple of (new_joint_positions, step_result)
    """
    model = arm.model
    data = arm.data

    # Get Jacobian
    joint_vel_indices = arm.joint_qpos_indices
    J = get_ee_jacobian(model, data, arm.ee_site_id, joint_vel_indices)

    # Transform twist to world frame if needed
    if frame == "hand":
        R = data.site_xmat[arm.ee_site_id].reshape(3, 3)
        twist_world = np.zeros(6)
        twist_world[:3] = R @ twist[:3]
        twist_world[3:] = R @ twist[3:]
        twist = twist_world

    # Get current state and limits
    q_current = arm.get_joint_positions()
    lower, upper = arm.get_joint_limits()

    # Get velocity limits from arm config
    vel_limits = np.array(arm.config.kinematic_limits.velocity)

    # Compute joint velocities with constraint handling
    result = twist_to_joint_velocity(
        J=J,
        twist=twist,
        q_current=q_current,
        lower_limits=lower,
        upper_limits=upper,
        velocity_limits=vel_limits,
        config=config,
    )

    # Integrate to get new positions
    q_new = q_current + result.joint_velocities * dt

    # Final clamp to joint limits (safety)
    q_new = np.clip(q_new, lower, upper)

    return q_new, result


# =============================================================================
# Contact Detection
# =============================================================================


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
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if geom2_body in gripper_body_ids:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)

    return None


# =============================================================================
# High-Level Primitives
# =============================================================================


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
    config: CartesianControlConfig | None = None,
) -> MoveUntilTouchResult:
    """Move gripper in direction until contact or max_distance reached.

    Moves the end-effector at constant velocity in the specified direction,
    stopping when:
    - Physics mode: F/T sensor force exceeds threshold
    - Kinematic mode: Collision detected in data.contact

    The `distance` parameter specifies the minimum distance to travel
    before contact detection activates (avoids false positives from
    initial contacts).

    Args:
        arm: Arm instance to control
        direction: Direction vector for motion (will be normalized)
        distance: Minimum distance before contact checking (meters)
        max_distance: Maximum distance - safety limit (meters)
        max_force: Force threshold in Newtons (default 5.0)
        max_torque: Torque threshold in Nm (optional)
        speed: Cartesian velocity in m/s (default 0.02 = 2cm/s)
        frame: "world" or "hand" frame for direction vector
        physics: True for physics mode, False for kinematic
        viewer: Optional MuJoCo viewer to sync
        config: Control configuration (uses defaults if None)

    Returns:
        MoveUntilTouchResult with success status and details

    Example:
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
    gripper_body_names = getattr(arm.config, 'gripper_bodies', [])

    # Use provided config or defaults
    if config is None:
        config = CartesianControlConfig()

    logger.debug(
        f"move_until_touch: direction={direction}, distance={distance:.3f}, "
        f"max_distance={max_distance:.3f}, speed={speed:.3f}, physics={physics}"
    )

    while total_distance < max_distance:
        # Compute new joint positions with constraint handling
        q_new, step_result = step_twist(arm, twist, frame=frame, dt=dt, config=config)

        # Check if we can make meaningful progress
        if step_result.achieved_fraction < config.min_progress_threshold:
            logger.debug(
                f"move_until_touch: cannot make progress "
                f"(achieved={step_result.achieved_fraction:.2f}, "
                f"limiting={step_result.limiting_factor})"
            )
            return MoveUntilTouchResult(
                success=False,
                terminated_by="no_progress",
                distance_moved=total_distance,
            )

        if physics:
            # Physics mode: set joint positions and step simulation
            for i, qpos_idx in enumerate(arm.joint_qpos_indices):
                data.qpos[qpos_idx] = q_new[i]

            mujoco.mj_step(model, data)

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
                            f"move_until_touch: contact! "
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
                                f"move_until_touch: contact! "
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
            # Kinematic mode: set positions and run forward kinematics
            for i, qpos_idx in enumerate(arm.joint_qpos_indices):
                data.qpos[qpos_idx] = q_new[i]

            mujoco.mj_forward(model, data)

            if viewer is not None:
                viewer.sync()

            # Check collision after min_distance
            if total_distance >= distance:
                contact_geom = check_gripper_contact_kinematic(
                    model, data, gripper_body_names
                )
                if contact_geom is not None:
                    logger.debug(
                        f"move_until_touch: contact with {contact_geom}"
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
    logger.debug(f"move_until_touch: max_distance reached ({total_distance:.3f}m)")

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
    config: CartesianControlConfig | None = None,
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
        config: Control configuration (uses defaults if None)

    Returns:
        TwistExecutionResult with termination info

    Example:
        result = execute_twist(
            arm=robot.right_arm,
            twist=[0, 0, 0.05, 0, 0, 0],  # 5cm/s up
            duration=2.0,
        )
    """
    model = arm.model
    data = arm.data

    if config is None:
        config = CartesianControlConfig()

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

        # Execute one step with constraint handling
        q_new, step_result = step_twist(arm, twist, frame=frame, dt=dt, config=config)

        # Check if we can make meaningful progress
        if step_result.achieved_fraction < config.min_progress_threshold:
            logger.debug(
                f"execute_twist: cannot make progress "
                f"(achieved={step_result.achieved_fraction:.2f})"
            )
            return TwistExecutionResult(
                terminated_by="no_progress",
                distance_moved=distance,
                duration=elapsed,
                final_pose=final_pose,
            )

        for i, qpos_idx in enumerate(arm.joint_qpos_indices):
            data.qpos[qpos_idx] = q_new[i]

        if physics:
            mujoco.mj_step(model, data)
        else:
            mujoco.mj_forward(model, data)

        if viewer is not None:
            viewer.sync()

        elapsed += dt
