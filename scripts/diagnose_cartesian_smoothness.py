#!/usr/bin/env python3
"""Diagnose cartesian control smoothness.

Measures motion quality metrics without viewer:
- Tracking error (commanded vs actual position)
- Velocity smoothness (jerk magnitude)
- Position oscillation

Usage:
    uv run python scripts/diagnose_cartesian_smoothness.py
"""

import numpy as np
import mujoco

from geodude import Geodude
from geodude.cartesian import CartesianControlConfig, step_twist


def measure_cartesian_motion(
    arm,
    ctx,
    direction: np.ndarray,
    duration: float = 1.0,
    speed: float = 0.05,
) -> dict:
    """Execute cartesian motion and record metrics.

    Returns dict with:
        - positions: recorded joint positions over time
        - commanded: commanded joint positions over time
        - times: timestamps
        - ee_positions: end-effector positions over time
    """
    model = arm.model
    data = arm.data
    dt = ctx.control_dt

    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    twist = np.zeros(6)
    twist[:3] = direction * speed

    config = CartesianControlConfig()

    # Recording arrays
    positions = []
    commanded = []
    times = []
    ee_positions = []
    velocities = []

    t = 0.0
    steps = int(duration / dt)

    for i in range(steps):
        # Record current state BEFORE step
        q_actual = np.array([data.qpos[idx] for idx in arm.joint_qpos_indices])
        qd_actual = np.array([data.qvel[idx] for idx in arm.joint_qpos_indices])
        ee_pos = data.site_xpos[arm.ee_site_id].copy()

        positions.append(q_actual)
        ee_positions.append(ee_pos)
        velocities.append(qd_actual)
        times.append(t)

        # Compute next target
        q_new, step_result = step_twist(arm, twist, frame="world", dt=dt, config=config)
        commanded.append(q_new.copy())

        # Step
        ctx.step_cartesian(arm.side, q_new, step_result.joint_velocities)

        t += dt

    return {
        "positions": np.array(positions),
        "commanded": np.array(commanded),
        "times": np.array(times),
        "ee_positions": np.array(ee_positions),
        "velocities": np.array(velocities),
        "dt": dt,
    }


def compute_smoothness_metrics(data: dict) -> dict:
    """Compute smoothness metrics from recorded motion data."""
    positions = data["positions"]
    commanded = data["commanded"]
    velocities = data["velocities"]
    ee_positions = data["ee_positions"]
    dt = data["dt"]

    # 1. Tracking error: how well does actual follow commanded?
    # Use position at t+1 vs commanded at t (since command is for next step)
    tracking_errors = []
    for i in range(len(commanded) - 1):
        error = np.linalg.norm(positions[i + 1] - commanded[i])
        tracking_errors.append(error)
    tracking_errors = np.array(tracking_errors)

    # 2. Joint velocity smoothness: compute jerk (derivative of acceleration)
    # First compute acceleration from velocity
    accelerations = np.diff(velocities, axis=0) / dt
    # Then compute jerk
    jerks = np.diff(accelerations, axis=0) / dt
    jerk_magnitude = np.linalg.norm(jerks, axis=1)

    # 3. EE position smoothness: check for oscillations
    # Compute EE velocity
    ee_velocities = np.diff(ee_positions, axis=0) / dt
    ee_velocity_magnitude = np.linalg.norm(ee_velocities, axis=1)

    # Compute EE acceleration
    ee_accelerations = np.diff(ee_velocities, axis=0) / dt
    ee_accel_magnitude = np.linalg.norm(ee_accelerations, axis=1)

    # 4. Velocity sign changes (oscillation indicator)
    # Count how often velocity changes sign per joint
    sign_changes_per_joint = []
    for j in range(velocities.shape[1]):
        joint_vel = velocities[:, j]
        signs = np.sign(joint_vel[1:]) != np.sign(joint_vel[:-1])
        # Only count changes when velocity is significant (not near zero)
        significant = np.abs(joint_vel[:-1]) > 0.01
        sign_changes = np.sum(signs & significant)
        sign_changes_per_joint.append(sign_changes)

    return {
        "tracking_error_mean": np.mean(tracking_errors),
        "tracking_error_max": np.max(tracking_errors),
        "tracking_error_std": np.std(tracking_errors),
        "jerk_mean": np.mean(jerk_magnitude),
        "jerk_max": np.max(jerk_magnitude),
        "jerk_std": np.std(jerk_magnitude),
        "jerk_array": jerk_magnitude,  # For distribution analysis
        "ee_accel_mean": np.mean(ee_accel_magnitude),
        "ee_accel_max": np.max(ee_accel_magnitude),
        "ee_accel_std": np.std(ee_accel_magnitude),
        "ee_accel_array": ee_accel_magnitude,  # For distribution analysis
        "velocity_sign_changes": sign_changes_per_joint,
        "total_sign_changes": sum(sign_changes_per_joint),
        "ee_velocity_mean": np.mean(ee_velocity_magnitude),
        "ee_velocity_std": np.std(ee_velocity_magnitude),
    }


def diagnose_smoothness():
    """Run smoothness diagnosis."""
    print("Cartesian Control Smoothness Diagnosis")
    print("=" * 50)

    # Test both modes
    for mode_name, physics_mode in [("PHYSICS", True), ("KINEMATIC", False)]:
        print(f"\n{'#' * 50}")
        print(f"# {mode_name} MODE")
        print(f"{'#' * 50}")

        robot = Geodude()
        robot.go_to("ready")
        mujoco.mj_forward(robot.model, robot.data)

        _run_mode_tests(robot, physics_mode)


def _run_mode_tests(robot, physics_mode: bool):
    """Run tests for a single mode."""
    # Test with headless (no viewer)
    with robot.sim(physics=physics_mode, viewer=False) as ctx:

        print(f"\nControl dt: {ctx.control_dt * 1000:.1f} ms")
        print(f"Physics timestep: {robot.model.opt.timestep * 1000:.1f} ms")

        # Test directions
        directions = [
            ([1, 0, 0], "+X"),
            ([0, 1, 0], "+Y"),
            ([0, 0, -1], "-Z (down)"),
            ([0.707, 0.707, 0], "diagonal XY"),
        ]

        all_metrics = []

        for direction, name in directions:
            # Reset to ready position
            robot.go_to("ready")
            mujoco.mj_forward(robot.model, robot.data)

            # Let physics settle (physics mode only)
            if physics_mode and ctx._controller is not None:
                for _ in range(50):
                    ctx._controller.step()

            print(f"\nTesting direction: {name}")
            print("-" * 30)

            # Measure motion
            data = measure_cartesian_motion(
                robot.right_arm,
                ctx,
                direction=direction,
                duration=0.5,  # 500ms of motion
                speed=0.05,    # 5 cm/s
            )

            metrics = compute_smoothness_metrics(data)
            all_metrics.append((name, metrics))

            # Print metrics
            print(f"  Tracking error:  mean={metrics['tracking_error_mean']*1000:.3f}ms, "
                  f"max={metrics['tracking_error_max']*1000:.3f}ms")
            print(f"  Joint jerk:      mean={metrics['jerk_mean']:.1f}, "
                  f"max={metrics['jerk_max']:.1f} rad/s³")
            print(f"  EE acceleration: mean={metrics['ee_accel_mean']:.2f}, "
                  f"max={metrics['ee_accel_max']:.2f} m/s²")
            print(f"  Velocity sign changes: {metrics['total_sign_changes']} "
                  f"(per joint: {metrics['velocity_sign_changes']})")
            print(f"  EE velocity:     mean={metrics['ee_velocity_mean']*100:.1f} cm/s, "
                  f"std={metrics['ee_velocity_std']*100:.2f} cm/s")

    # Summary assessment
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    # Aggregate metrics
    avg_jerk = np.mean([m["jerk_mean"] for _, m in all_metrics])
    max_jerk = np.max([m["jerk_max"] for _, m in all_metrics])
    avg_tracking = np.mean([m["tracking_error_mean"] for _, m in all_metrics])
    total_oscillations = sum([m["total_sign_changes"] for _, m in all_metrics])
    avg_ee_accel = np.mean([m["ee_accel_mean"] for _, m in all_metrics])

    # Jerk distribution analysis
    print("\nJerk distribution (shows if spikes are isolated or frequent):")
    for name, metrics in all_metrics:
        jerk_arr = metrics.get("jerk_array")
        if jerk_arr is not None:
            pct_above_100 = np.sum(jerk_arr > 100) / len(jerk_arr) * 100
            pct_above_500 = np.sum(jerk_arr > 500) / len(jerk_arr) * 100
            print(f"  {name}: {pct_above_100:.1f}% above 100 rad/s³, {pct_above_500:.1f}% above 500 rad/s³")

    print(f"\nAverage jerk: {avg_jerk:.1f} rad/s³")
    print(f"Maximum jerk: {max_jerk:.1f} rad/s³")
    print(f"Average tracking error: {avg_tracking*1000:.3f} rad (in 1000ths)")
    print(f"Total velocity oscillations: {total_oscillations}")
    print(f"Average EE acceleration: {avg_ee_accel:.2f} m/s²")

    # Thresholds for "good" motion
    # These are heuristic - may need tuning
    jerk_threshold = 500  # rad/s³
    oscillation_threshold = 10  # per direction
    accel_threshold = 2.0  # m/s² (should be low for constant velocity)

    issues = []
    if avg_jerk > jerk_threshold:
        issues.append(f"High jerk ({avg_jerk:.0f} > {jerk_threshold})")
    if total_oscillations > oscillation_threshold * len(directions):
        issues.append(f"Excessive oscillations ({total_oscillations})")
    if avg_ee_accel > accel_threshold:
        issues.append(f"High EE acceleration ({avg_ee_accel:.2f} > {accel_threshold})")

    if issues:
        print(f"\n⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nMotion may appear jittery.")
    else:
        print(f"\n✓ Motion appears smooth based on metrics.")


if __name__ == "__main__":
    diagnose_smoothness()
