"""Jacobian-based Cartesian velocity control.

Provides robust Cartesian control primitives for contact-based grasping.
The core function `twist_to_joint_velocity()` solves a constrained QP:

    min  (1/2)||J*q_dot - v_d||_W^2 + (λ/2)||q_dot||^2
    s.t. ℓ <= q_dot <= u

where bounds ℓ,u incorporate both velocity limits AND position limits
(converted to per-timestep velocity bounds).

Works in both physics mode (F/T sensor) and kinematic mode (data.contact).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import mujoco
import numpy as np
from scipy.linalg import cho_factor, cho_solve

if TYPE_CHECKING:
    from geodude.arm import Arm

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CartesianControlConfig:
    """Configuration for Cartesian velocity control.

    Based on constrained QP formulation for Jacobian IK.

    Attributes:
        length_scale: Length scale L for twist weighting (meters).
                     W = diag(1,1,1, 1/L², 1/L², 1/L²)
                     Balances linear velocity (m/s) vs angular (rad/s).
                     Default 0.1m works well for manipulation.

        damping: Regularization λ for joint velocity.
                Prevents large joint motions when twist is small.
                Default 1e-4 is good for most cases.

        joint_margin_deg: Degrees from joint limits to treat as buffer.
                         Position limits are shrunk by this margin.
                         Default 5° provides safety margin.

        velocity_scale: Fraction of max joint velocity to use (0-1).
                       Default 1.0 uses full velocity limits.
                       Lower values give slower, safer motion.

        min_progress: Minimum achieved_fraction to continue motion (0-1).
                     If the achieved twist falls below this fraction of
                     the desired twist, motion stops with "no_progress".
                     Default 0.5 requires at least 50% of desired velocity.
    """

    length_scale: float = 0.1
    damping: float = 1e-4
    joint_margin_deg: float = 5.0
    velocity_scale: float = 1.0
    min_progress: float = 0.5

    def __post_init__(self):
        """Validate configuration values."""
        if self.length_scale <= 0:
            raise ValueError(f"length_scale must be > 0, got {self.length_scale}")
        if self.damping < 0:
            raise ValueError(f"damping must be >= 0, got {self.damping}")
        if self.joint_margin_deg < 0:
            raise ValueError(f"joint_margin_deg must be >= 0, got {self.joint_margin_deg}")
        if not 0 < self.velocity_scale <= 1:
            raise ValueError(f"velocity_scale must be in (0, 1], got {self.velocity_scale}")
        if not 0 <= self.min_progress <= 1:
            raise ValueError(f"min_progress must be in [0, 1], got {self.min_progress}")


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class TwistStepResult:
    """Result of a single twist-to-joint-velocity computation.

    Attributes:
        joint_velocities: Computed joint velocities (rad/s)
        twist_error: Residual ||J*q_dot - v_d||_W after optimization
        achieved_fraction: Approximate fraction of desired twist achieved (0-1)
        limiting_factor: What's constraining the solution, if any
    """

    joint_velocities: np.ndarray
    twist_error: float
    achieved_fraction: float
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


# =============================================================================
# Core Control Function (Constrained QP)
# =============================================================================


def twist_to_joint_velocity(
    J: np.ndarray,
    twist: np.ndarray,
    q_current: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    qd_max: np.ndarray,
    dt: float,
    config: CartesianControlConfig | None = None,
    q_dot_prev: np.ndarray | None = None,
) -> TwistStepResult:
    """Convert a Cartesian twist to joint velocities via constrained QP.

    Solves:
        min  (1/2)||J*q_dot - v_d||_W^2 + (λ/2)||q_dot||^2
        s.t. ℓ <= q_dot <= u

    where:
        W = diag(1,1,1, 1/L², 1/L², 1/L²) balances linear/angular
        ℓ, u combine velocity limits with position-derived bounds

    Args:
        J: 6xN Jacobian matrix
        twist: 6D desired twist [vx, vy, vz, wx, wy, wz] in m/s and rad/s
        q_current: Current joint positions (rad)
        q_min: Lower joint position limits (rad)
        q_max: Upper joint position limits (rad)
        qd_max: Maximum joint velocities (rad/s), assumed symmetric
        dt: Controller timestep (seconds)
        config: Control configuration (uses defaults if None)
        q_dot_prev: Previous solution for warm-starting (optional).
                   Significantly speeds up convergence in streaming control.

    Returns:
        TwistStepResult with joint velocities and diagnostics
    """
    if config is None:
        config = CartesianControlConfig()

    n_joints = J.shape[1]
    L = config.length_scale
    lam = config.damping
    margin = np.deg2rad(config.joint_margin_deg)

    # Scale velocity limits
    qd_max_scaled = qd_max * config.velocity_scale

    # =========================================================================
    # Step 1: Convert position limits to velocity bounds
    # =========================================================================
    # One-step safe bounds: ensure q + q_dot*dt stays within [q_min+m, q_max-m]
    ell_pos = ((q_min + margin) - q_current) / dt
    u_pos = ((q_max - margin) - q_current) / dt

    # Combine with velocity limits
    ell = np.maximum(-qd_max_scaled, ell_pos)
    u = np.minimum(+qd_max_scaled, u_pos)

    # Check for infeasible bounds (joint already past limit + margin)
    infeasible = ell > u
    if np.any(infeasible):
        # Relax bounds for infeasible joints (allow them to move back)
        ell[infeasible] = np.minimum(ell[infeasible], 0)
        u[infeasible] = np.maximum(u[infeasible], 0)

    # =========================================================================
    # Step 2: Build QP matrices
    # =========================================================================
    # Twist weighting: W = diag(1,1,1, 1/L², 1/L², 1/L²)
    w_diag = np.array([1.0, 1.0, 1.0, 1.0/L**2, 1.0/L**2, 1.0/L**2])
    W = np.diag(w_diag)

    # H = J^T W J + λI
    # g = -J^T W v_d
    JtW = J.T @ W
    H = JtW @ J + lam * np.eye(n_joints)
    g = -JtW @ twist

    # =========================================================================
    # Step 3: Solve box-constrained QP via projected gradient descent
    # =========================================================================
    # min (1/2) q_dot^T H q_dot + g^T q_dot  s.t. ell <= q_dot <= u
    #
    # For small box-constrained QP (6 DOF), projected gradient descent is fast
    # and predictable - typically converges in 5-15 iterations. Much better than
    # L-BFGS-B which has unpredictable iteration counts.

    # Unconstrained solution via Cholesky (H is SPD by construction)
    try:
        cho = cho_factor(H)
        qd_unconstrained = cho_solve(cho, -g)
    except np.linalg.LinAlgError:
        # Fallback if Cholesky fails (shouldn't happen with λ > 0)
        qd_unconstrained = np.linalg.solve(H, -g)

    # If unconstrained solution is feasible, we're done
    if np.all(qd_unconstrained >= ell) and np.all(qd_unconstrained <= u):
        q_dot = qd_unconstrained
    else:
        # Projected gradient descent for box-constrained QP
        # Very efficient - typically converges in 2-5 iterations with warm start

        # Warm start: use previous solution if provided, else clamped unconstrained
        if q_dot_prev is not None:
            q_dot = np.clip(q_dot_prev, ell, u)
        else:
            q_dot = np.clip(qd_unconstrained, ell, u)

        # Compute step size from eigenvalue bounds (H is well-conditioned due to λI)
        # Step size = 1 / max_eigenvalue(H) ≈ 1 / (||J||² + λ)
        alpha = 1.0 / (np.linalg.norm(H, 2) + 1e-6)

        for _ in range(20):  # Max 20 iterations (typically 2-5 with warm start)
            grad = H @ q_dot + g
            q_new = np.clip(q_dot - alpha * grad, ell, u)

            # Check convergence: gradient projected onto feasible set
            if np.linalg.norm(q_new - q_dot) < 1e-8:
                break

            q_dot = q_new

    # =========================================================================
    # Step 4: Compute diagnostics
    # =========================================================================
    # Twist error: ||J*q_dot - v_d||_W
    twist_achieved = J @ q_dot
    twist_diff = twist_achieved - twist
    twist_error = float(np.sqrt(twist_diff @ W @ twist_diff))

    # Achieved fraction: compare achieved twist magnitude to desired
    twist_norm = float(np.sqrt(twist @ W @ twist))
    if twist_norm > 1e-10:
        # Project achieved onto desired direction
        achieved_fraction = float(np.dot(twist_achieved, W @ twist) / (twist_norm**2))
        achieved_fraction = max(0.0, min(1.0, achieved_fraction))
    else:
        achieved_fraction = 1.0  # No motion requested

    # Determine limiting factor
    limiting_factor = None
    at_lower = np.abs(q_dot - ell) < 1e-6
    at_upper = np.abs(q_dot - u) < 1e-6
    at_bound = at_lower | at_upper

    if np.any(at_bound):
        # Check if it's position or velocity limit
        at_pos_lower = at_lower & (ell > -qd_max_scaled + 1e-6)
        at_pos_upper = at_upper & (u < qd_max_scaled - 1e-6)
        if np.any(at_pos_lower | at_pos_upper):
            limiting_factor = "joint_limit"
        else:
            limiting_factor = "velocity"

    return TwistStepResult(
        joint_velocities=q_dot,
        twist_error=twist_error,
        achieved_fraction=achieved_fraction,
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
    q_dot_prev: np.ndarray | None = None,
) -> tuple[np.ndarray, TwistStepResult]:
    """Execute one timestep of Cartesian velocity control.

    Uses the constrained QP solver to compute joint velocities that
    achieve the desired Cartesian twist while respecting all limits.

    Args:
        arm: Arm instance
        twist: 6D twist [vx, vy, vz, wx, wy, wz] in m/s and rad/s
        frame: "world" or "hand" (tool frame)
        dt: Timestep duration in seconds
        config: Control configuration (uses defaults if None)
        q_dot_prev: Previous joint velocities for warm-starting (optional).
                   Pass the joint_velocities from previous TwistStepResult.

    Returns:
        Tuple of (new_joint_positions, step_result)
    """
    model = arm.model
    data = arm.data

    # Get Jacobian - use qvel indices (dof space), not qpos indices
    # MuJoCo's mj_jacSite returns Jacobian in velocity space (nv)
    joint_vel_indices = arm.joint_qvel_indices
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
    q_min, q_max = arm.get_joint_limits()
    qd_max = np.array(arm.config.kinematic_limits.velocity)

    # Solve constrained QP (warm start from previous solution if provided)
    result = twist_to_joint_velocity(
        J=J,
        twist=twist,
        q_current=q_current,
        q_min=q_min,
        q_max=q_max,
        qd_max=qd_max,
        dt=dt,
        config=config,
        q_dot_prev=q_dot_prev,
    )

    # Integrate to get new positions
    q_new = q_current + result.joint_velocities * dt

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


def get_arm_body_ids(
    model: mujoco.MjModel,
    arm: "Arm",
) -> set[int]:
    """Get all body IDs that are part of an arm (links + gripper).

    Collects body IDs from:
    - Bodies associated with each arm joint (the UR5e links)
    - Gripper bodies from the arm configuration

    Args:
        model: MuJoCo model
        arm: Arm instance

    Returns:
        Set of body IDs belonging to this arm
    """
    body_ids = set()

    # Get bodies from joints (each joint is attached to a body)
    for joint_name in arm.config.joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            # Joint's body is stored in jnt_bodyid
            body_id = model.jnt_bodyid[joint_id]
            body_ids.add(body_id)

    # Add gripper bodies
    gripper_body_names = getattr(arm.config, 'gripper_bodies', [])
    for name in gripper_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            body_ids.add(body_id)

    return body_ids


def check_arm_contact_kinematic(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: "Arm",
    exclude_self_collision: bool = True,
) -> str | None:
    """Check for arm contact using MuJoCo collision detection.

    Scans data.contact for any contacts involving ANY arm body (links + gripper).
    Used in kinematic mode where F/T sensors aren't meaningful, and also
    useful in physics mode for early collision detection.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        arm: Arm instance to check contacts for
        exclude_self_collision: If True, ignores contacts between arm bodies
                               (arm self-collision). Default True.

    Returns:
        Name of contacted geom (the non-arm geom), or None if no contact
    """
    # Get all body IDs for this arm
    arm_body_ids = get_arm_body_ids(model, arm)

    # Scan contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_body = model.geom_bodyid[contact.geom1]
        geom2_body = model.geom_bodyid[contact.geom2]

        geom1_is_arm = geom1_body in arm_body_ids
        geom2_is_arm = geom2_body in arm_body_ids

        # Skip self-collision if requested
        if exclude_self_collision and geom1_is_arm and geom2_is_arm:
            continue

        # Check if either geom belongs to arm (but not both, unless we want self-collision)
        if geom1_is_arm and not geom2_is_arm:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if geom2_is_arm and not geom1_is_arm:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)

    return None


def check_arm_contact_after_move(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: "Arm",
    q_new: np.ndarray,
) -> str | None:
    """Check if moving to q_new would cause arm collision.

    Predictive collision check: temporarily sets arm to proposed position,
    runs collision detection, then restores original position. This prevents
    moving into known obstacles.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        arm: Arm instance
        q_new: Proposed joint positions to check

    Returns:
        Name of contacted geom if collision would occur, None if safe
    """
    # Save current positions
    q_saved = np.array([data.qpos[idx] for idx in arm.joint_qpos_indices])

    # Temporarily set to proposed position
    for i, idx in enumerate(arm.joint_qpos_indices):
        data.qpos[idx] = q_new[i]

    # Run forward kinematics to update collision geometry
    mujoco.mj_forward(model, data)

    # Check for contacts
    contact_geom = check_arm_contact_kinematic(model, data, arm)

    # Restore original positions
    for i, idx in enumerate(arm.joint_qpos_indices):
        data.qpos[idx] = q_saved[i]

    # Restore collision state
    mujoco.mj_forward(model, data)

    return contact_geom


# =============================================================================
# High-Level Primitives
# =============================================================================


def move_until_touch(
    arm: "Arm",
    direction: np.ndarray,
    distance: float,
    max_distance: float,
    max_force: float = 20.0,
    max_torque: float | None = None,
    speed: float = 0.02,
    frame: str = "world",
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

    Requires an active SimContext on the robot.

    Args:
        arm: Arm instance to control
        direction: Direction vector for motion (will be normalized)
        distance: Minimum distance before contact checking (meters)
        max_distance: Maximum distance - safety limit (meters)
        max_force: Force threshold in Newtons (default 5.0)
        max_torque: Torque threshold in Nm (optional)
        speed: Cartesian velocity in m/s (default 0.02 = 2cm/s)
        frame: "world" or "hand" frame for direction vector
        config: Control configuration (uses defaults if None)

    Returns:
        MoveUntilTouchResult with success status and details

    Raises:
        RuntimeError: If no active SimContext

    Example:
        with robot.sim(physics=True) as ctx:
            result = move_until_touch(
                arm=robot.right_arm,
                direction=[0, 0, 1],  # Forward in gripper frame
                distance=0.01,        # Min 1cm before checking
                max_distance=0.05,    # Max 5cm
                max_force=3.0,        # Light touch
                frame="hand",
            )
            if result.success:
                ctx.arm("right").grasp("object")
    """
    # Get context - required for execution
    ctx = arm.robot._active_context
    if ctx is None:
        raise RuntimeError(
            "move_until_touch requires an active SimContext. "
            "Use 'with robot.sim() as ctx:' to create one."
        )

    model = arm.model
    data = arm.data
    physics = ctx._physics

    # Normalize direction
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    # Create twist (linear velocity only, preserve orientation)
    twist = np.zeros(6)
    twist[:3] = direction * speed

    # Control parameters - use context's control_dt to match physics timing
    dt = ctx.control_dt
    total_distance = 0.0
    start_pos = data.site_xpos[arm.ee_site_id].copy()

    # Note: We use check_arm_contact_kinematic which checks ALL arm bodies,
    # not just gripper. This prevents arm links from penetrating obstacles.

    # Use provided config or defaults
    if config is None:
        config = CartesianControlConfig()

    logger.debug(
        f"move_until_touch: direction={direction}, distance={distance:.3f}, "
        f"max_distance={max_distance:.3f}, speed={speed:.3f}, physics={physics}"
    )

    # Warm-start: track previous solution for faster QP convergence
    q_dot_prev = None

    while total_distance < max_distance:
        # Compute new joint positions with QP solver (warm-started)
        q_new, step_result = step_twist(
            arm, twist, frame=frame, dt=dt, config=config, q_dot_prev=q_dot_prev
        )
        q_dot_prev = step_result.joint_velocities  # Save for next iteration

        # Check if we can make meaningful progress
        if step_result.achieved_fraction < config.min_progress:
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

        # Kinematic mode only: check if next position would cause collision BEFORE moving
        # This prevents arm from moving into known obstacles.
        # In physics mode, we can't do this check (it would corrupt physics state),
        # so we rely on F/T sensor for contact detection.
        if not physics:
            contact_geom = check_arm_contact_after_move(model, data, arm, q_new)
            if contact_geom is not None:
                logger.debug(
                    f"move_until_touch: would collide with {contact_geom}, stopping"
                )
                return MoveUntilTouchResult(
                    success=False,
                    terminated_by="no_progress",
                    distance_moved=total_distance,
                    contact_geom=contact_geom,
                )

        # Step through context (handles physics/kinematic, viewer sync, other actuators)
        # Pass velocity for smooth streaming - step_reactive uses control_dt lookahead
        # (not the 0.1s trajectory lookahead) to avoid overshoot while staying smooth.
        ctx.step_cartesian(arm.side, q_new, step_result.joint_velocities)

        # Check for actual contact after min_distance (physics mode: F/T sensor)
        if total_distance >= distance and physics:
            ft_result = arm.get_ft_sensor()
            if ft_result is not None:
                force, torque = ft_result
                force_magnitude = np.linalg.norm(force)

                if force_magnitude > max_force:
                    logger.debug(
                        f"move_until_touch: F/T contact! "
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
                            f"move_until_touch: F/T contact! "
                            f"torque={torque_magnitude:.2f}Nm > {max_torque:.2f}Nm"
                        )
                        return MoveUntilTouchResult(
                            success=True,
                            terminated_by="contact",
                            distance_moved=total_distance,
                            final_force=force,
                            final_torque=torque,
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
    config: CartesianControlConfig | None = None,
) -> TwistExecutionResult:
    """Execute Cartesian twist until a termination condition.

    Lower-level method that runs a constant twist (Cartesian velocity)
    until one of the termination conditions is met.

    Requires an active SimContext on the robot.

    Args:
        arm: Arm instance to control
        twist: 6D twist [vx, vy, vz, wx, wy, wz] in m/s and rad/s
        frame: "world" or "hand" frame for twist
        duration: Stop after this many seconds
        max_distance: Stop after EE moves this far (meters)
        until: Custom termination predicate (returns True to stop)
        config: Control configuration (uses defaults if None)

    Returns:
        TwistExecutionResult with termination info

    Raises:
        RuntimeError: If no active SimContext

    Example:
        with robot.sim(physics=True) as ctx:
            result = execute_twist(
                arm=robot.right_arm,
                twist=[0, 0, 0.05, 0, 0, 0],  # 5cm/s up
                duration=2.0,
            )
    """
    # Get context - required for execution
    ctx = arm.robot._active_context
    if ctx is None:
        raise RuntimeError(
            "execute_twist requires an active SimContext. "
            "Use 'with robot.sim() as ctx:' to create one."
        )

    data = arm.data

    if config is None:
        config = CartesianControlConfig()

    # Use context's control_dt to match physics timing
    dt = ctx.control_dt
    elapsed = 0.0
    start_pos = data.site_xpos[arm.ee_site_id].copy()

    logger.debug(
        f"execute_twist: twist={twist}, duration={duration}, "
        f"max_distance={max_distance}, physics={ctx._physics}"
    )

    # Track for stall detection
    last_pos = start_pos.copy()
    stall_count = 0
    stall_threshold = 50  # Stop if no movement for 50 iterations
    step_count = 0

    # Warm-start: track previous solution for faster QP convergence
    q_dot_prev = None

    while True:
        step_count += 1
        # Check termination conditions
        current_pos = data.site_xpos[arm.ee_site_id].copy()
        distance = np.linalg.norm(current_pos - start_pos)
        final_pose = arm.get_ee_pose()

        # Check for stall (arm not moving despite commands)
        step_movement = np.linalg.norm(current_pos - last_pos)
        if step_movement < 1e-5:  # Less than 0.01mm movement
            stall_count += 1
        else:
            stall_count = 0
        last_pos = current_pos.copy()

        # Log progress periodically
        if step_count % 100 == 0:
            logger.debug(
                f"execute_twist step {step_count}: distance={distance*100:.2f}cm, "
                f"stall_count={stall_count}"
            )

        if stall_count >= stall_threshold:
            logger.debug(
                f"execute_twist: stalled after {step_count} steps "
                f"(no movement for {stall_count} iterations)"
            )
            return TwistExecutionResult(
                terminated_by="no_progress",
                distance_moved=distance,
                duration=elapsed,
                final_pose=final_pose,
            )

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

        # Execute one step with QP solver (warm-started)
        q_new, step_result = step_twist(
            arm, twist, frame=frame, dt=dt, config=config, q_dot_prev=q_dot_prev
        )
        q_dot_prev = step_result.joint_velocities  # Save for next iteration

        # Check if we can make meaningful progress
        if step_result.achieved_fraction < config.min_progress:
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

        # Step through context (handles physics/kinematic, viewer sync, other actuators)
        # Pass velocity for smooth streaming - step_reactive uses control_dt lookahead
        # (not the 0.1s trajectory lookahead) to avoid overshoot while staying smooth.
        ctx.step_cartesian(arm.side, q_new, step_result.joint_velocities)

        elapsed += dt
