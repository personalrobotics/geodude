"""Visual demo: Arm movement, IK, and Vention base control.

Demonstrates:
- Random joint configurations with collision-free planning
- IK to random Cartesian targets (using all solutions)
- Vention base height changes
- Time-optimal trajectory execution using TOPP-RA (respects velocity/acceleration limits)
- All visualized in the MuJoCo viewer

Usage:
    uv run mjpython examples/basic_movement.py
"""

import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from geodude import Geodude
from geodude.config import KinematicLimits


def plan_and_execute(robot, arm, q_goal, viewer, seed=42):
    """Plan to a configuration and execute, handling state corruption from planner.

    IMPORTANT: Planners/collision checkers modify robot state. This function
    ensures the robot is at path[0] before execution starts.

    Returns:
        True if planning and execution both succeeded, False otherwise
    """
    # Save current position
    q_before = arm.get_joint_positions().copy()

    # Plan
    path = arm.plan_to_configuration(q_goal, seed=seed)

    if path is None:
        print(f"      FAILURE: Planning failed - no collision-free path found")
        sys.stdout.flush()
        return False

    # Planner already restored state to path[0] - just sync actuators
    # Set actuator commands to match current position (which is already at path[0])
    for joint_idx, actuator_id in enumerate(arm.actuator_ids):
        robot.data.ctrl[actuator_id] = arm.get_joint_positions()[joint_idx]
    # Hold other arm
    other_arm = robot.left_arm if arm.name == "right" else robot.right_arm
    for joint_idx, actuator_id in enumerate(other_arm.actuator_ids):
        robot.data.ctrl[actuator_id] = other_arm.get_joint_positions()[joint_idx]
    mujoco.mj_forward(robot.model, robot.data)

    # Execute
    return execute_path_with_viewer(robot, arm, path, viewer)


def execute_path_with_viewer(robot, arm, path: list[np.ndarray], viewer):
    """Execute a path using TOPP-RA retiming with viewer updates.

    The trajectory is automatically retimed to respect velocity and acceleration
    limits. The viewer is synced during execution for smooth visualization.

    IMPORTANT: Actively holds the other arm in place with feedback control
    to prevent flailing while this arm moves.
    """
    from geodude.trajectory import Trajectory

    # State should be clean now (planner restores it), but verify
    q_current = arm.get_joint_positions()
    q_start = path[0]
    start_discontinuity = np.linalg.norm(q_start - q_current)

    if start_discontinuity > 0.01:  # More than ~0.5 degrees
        print(f"      WARNING: Unexpected discontinuity {np.rad2deg(start_discontinuity):.2f}° (planner should have restored state!)")
        sys.stdout.flush()

    # Set actuator commands to match current position for both arms
    other_arm = robot.left_arm if arm.name == "right" else robot.right_arm

    for joint_idx, actuator_id in enumerate(arm.actuator_ids):
        robot.data.ctrl[actuator_id] = arm.get_joint_positions()[joint_idx]
    for joint_idx, actuator_id in enumerate(other_arm.actuator_ids):
        robot.data.ctrl[actuator_id] = other_arm.get_joint_positions()[joint_idx]

    # Create trajectory with TOPP-RA retiming
    trajectory = Trajectory.from_path(
        path,
        arm.config.kinematic_limits.velocity,
        arm.config.kinematic_limits.acceleration,
    )

    # Get executor for moving arm
    executor = arm._get_executor(viewer=viewer, executor_type="closed_loop")

    # Get the other arm to hold it in place during execution
    other_arm = robot.left_arm if arm.name == "right" else robot.right_arm
    other_executor = other_arm._get_executor(viewer=None, executor_type="closed_loop")
    # Capture hold position AFTER any transition (not before)
    other_q_hold = other_arm.get_joint_positions()

    # Track errors for both arms
    moving_arm_errors = []
    other_arm_errors = []
    moving_arm_vel_errors = []
    other_arm_vel_errors = []

    # Execute trajectory with both arms controlled
    print(f"      Executing {trajectory.num_waypoints} waypoints (kp={executor.kp}, kd={executor.kd})...")
    sys.stdout.flush()

    # Tracking quality thresholds
    max_acceptable_error = np.deg2rad(15.0)  # 15 degrees max instantaneous error
    catastrophic_error = np.deg2rad(45.0)  # 45 degrees = immediate abort

    for i in range(trajectory.num_waypoints):
        # === Control moving arm with feedback ===
        q_desired = trajectory.positions[i]
        qd_desired = trajectory.velocities[i]

        q_actual = np.array([robot.data.qpos[idx] for idx in executor.joint_qpos_indices])
        qd_actual = np.array([robot.data.qvel[idx] for idx in executor.joint_qpos_indices])

        position_error = q_desired - q_actual
        velocity_error = qd_desired - qd_actual
        q_command = q_desired + executor.kp * position_error + executor.kd * velocity_error

        for joint_idx, actuator_id in enumerate(executor.actuator_ids):
            robot.data.ctrl[actuator_id] = q_command[joint_idx]

        # === Hold other arm in place with feedback ===
        other_q_actual = np.array([robot.data.qpos[idx] for idx in other_executor.joint_qpos_indices])
        other_qd_actual = np.array([robot.data.qvel[idx] for idx in other_executor.joint_qpos_indices])

        other_position_error = other_q_hold - other_q_actual
        other_velocity_error = -other_qd_actual  # Target velocity is zero
        other_q_command = other_q_hold + other_executor.kp * other_position_error + other_executor.kd * other_velocity_error

        for joint_idx, actuator_id in enumerate(other_executor.actuator_ids):
            robot.data.ctrl[actuator_id] = other_q_command[joint_idx]

        # Step physics
        for _ in range(executor.steps_per_control):
            mujoco.mj_step(robot.model, robot.data)

        # Check for collisions after physics step
        if robot.data.ncon > 0:
            # Check if any contacts involve the moving arm
            collision_detected = False
            for contact_idx in range(robot.data.ncon):
                contact = robot.data.contact[contact_idx]
                geom1_body = robot.model.geom_bodyid[contact.geom1]
                geom2_body = robot.model.geom_bodyid[contact.geom2]

                # Get body names
                body1_name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, geom1_body)
                body2_name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, geom2_body)

                # Check if moving arm is involved in unexpected collision
                # (arm name should be in one of the body names)
                if arm.name in (body1_name or "") or arm.name in (body2_name or ""):
                    # This is a contact involving the moving arm
                    # Check if it's with base, table, or other unexpected object
                    if "base" in (body1_name or "").lower() or "base" in (body2_name or "").lower():
                        collision_detected = True
                        print(f"        ABORT: Collision detected between {body1_name} and {body2_name} at waypoint {i}")
                        sys.stdout.flush()
                        break

            if collision_detected:
                return False

        # Sync viewer
        if viewer is not None:
            viewer.sync()

        # Track errors after physics
        q_actual = np.array([robot.data.qpos[idx] for idx in executor.joint_qpos_indices])
        qd_actual = np.array([robot.data.qvel[idx] for idx in executor.joint_qpos_indices])
        other_q_actual = np.array([robot.data.qpos[idx] for idx in other_executor.joint_qpos_indices])
        other_qd_actual = np.array([robot.data.qvel[idx] for idx in other_executor.joint_qpos_indices])

        current_error = np.linalg.norm(trajectory.positions[i] - q_actual)
        moving_arm_errors.append(current_error)
        moving_arm_vel_errors.append(np.linalg.norm(trajectory.velocities[i] - qd_actual))
        other_arm_errors.append(np.linalg.norm(other_q_hold - other_q_actual))
        other_arm_vel_errors.append(np.linalg.norm(other_qd_actual))  # Should be ~0

        # Check for catastrophic tracking failure
        if current_error > catastrophic_error:
            print(f"        ABORT: Catastrophic tracking error {np.rad2deg(current_error):.1f}° at waypoint {i}")
            sys.stdout.flush()
            return False

        # Print real-time error every 20 waypoints
        if i > 0 and i % 20 == 0:
            recent_pos_error = np.mean(moving_arm_errors[-20:])
            print(f"        Waypoint {i}/{trajectory.num_waypoints}: pos_error={np.rad2deg(recent_pos_error):.2f}°")
            sys.stdout.flush()

        time.sleep(executor.control_dt)

    # Final settling with both arms controlled
    q_final = trajectory.positions[-1]

    for _ in range(executor.steps_per_control * 20):
        # Moving arm feedback to final position
        q_actual = np.array([robot.data.qpos[idx] for idx in executor.joint_qpos_indices])
        qd_actual = np.array([robot.data.qvel[idx] for idx in executor.joint_qpos_indices])

        position_error = q_final - q_actual
        velocity_error = -qd_actual
        q_command = q_final + executor.kp * position_error + executor.kd * velocity_error

        for joint_idx, actuator_id in enumerate(executor.actuator_ids):
            robot.data.ctrl[actuator_id] = q_command[joint_idx]

        # Other arm feedback to hold position
        other_q_actual = np.array([robot.data.qpos[idx] for idx in other_executor.joint_qpos_indices])
        other_qd_actual = np.array([robot.data.qvel[idx] for idx in other_executor.joint_qpos_indices])

        other_position_error = other_q_hold - other_q_actual
        other_velocity_error = -other_qd_actual
        other_q_command = other_q_hold + other_executor.kp * other_position_error + other_executor.kd * other_velocity_error

        for joint_idx, actuator_id in enumerate(other_executor.actuator_ids):
            robot.data.ctrl[actuator_id] = other_q_command[joint_idx]

        mujoco.mj_step(robot.model, robot.data)

    # Final viewer sync
    if viewer is not None:
        viewer.sync()

    # Report tracking statistics
    moving_arm_errors = np.array(moving_arm_errors)
    other_arm_errors = np.array(other_arm_errors)
    moving_arm_vel_errors = np.array(moving_arm_vel_errors)
    other_arm_vel_errors = np.array(other_arm_vel_errors)

    avg_pos_error = np.rad2deg(np.mean(moving_arm_errors))
    max_pos_error = np.rad2deg(np.max(moving_arm_errors))
    avg_hold_error = np.rad2deg(np.mean(other_arm_errors))
    max_hold_error = np.rad2deg(np.max(other_arm_errors))

    print(f"      {arm.name.upper()} arm tracking:")
    print(f"        Position: avg={avg_pos_error:.2f}°, max={max_pos_error:.2f}°")
    print(f"        Velocity: avg={np.rad2deg(np.mean(moving_arm_vel_errors)):.2f}°/s, max={np.rad2deg(np.max(moving_arm_vel_errors)):.2f}°/s")
    print(f"      {other_arm.name.upper()} arm hold error:")
    print(f"        Position: avg={avg_hold_error:.2f}°, max={max_hold_error:.2f}°")
    print(f"        Velocity: avg={np.rad2deg(np.mean(other_arm_vel_errors)):.2f}°/s, max={np.rad2deg(np.max(other_arm_vel_errors)):.2f}°/s")

    # Determine success based on tracking quality
    # Success criteria:
    # - Average position error < 1° (excellent tracking)
    # - Max position error < 10° (no catastrophic failures)
    # - Stationary arm hold error < 1° (good coordination)
    success = (avg_pos_error < 1.0 and
               max_pos_error < 10.0 and
               max_hold_error < 1.0)

    if success:
        print(f"      SUCCESS: Trajectory executed successfully")
    else:
        print(f"      FAILURE: Trajectory execution quality issues:")
        if avg_pos_error >= 1.0:
            print(f"        - Poor average tracking: {avg_pos_error:.2f}° (threshold: 1.0°)")
        if max_pos_error >= 10.0:
            print(f"        - Large tracking spikes: {max_pos_error:.2f}° (threshold: 10.0°)")
        if max_hold_error >= 1.0:
            print(f"        - Stationary arm drift: {max_hold_error:.2f}° (threshold: 1.0°)")

    sys.stdout.flush()

    return success


def animate_base_move(robot, base, target_height: float, viewer, steps: int = 50):
    """Animate base movement to target height."""
    start_height = base.height
    for h in np.linspace(start_height, target_height, steps):
        base.set_height(h)
        mujoco.mj_forward(robot.model, robot.data)
        viewer.sync()
        time.sleep(0.015)


def setup_camera(viewer):
    """Set camera to view robot from the front, looking down at 45 degrees."""
    viewer.cam.azimuth = 90  # View from +X direction (front of workspace)
    viewer.cam.elevation = -35  # 35 degrees down
    viewer.cam.distance = 2.2  # Distance from lookat point
    viewer.cam.lookat[:] = [0.25, 0.0, 1.1]  # Center of workspace


def random_valid_config(arm, rng: np.random.Generator) -> np.ndarray:
    """Generate a random joint configuration within safe limits."""
    lower, upper = arm.get_joint_limits()
    margin = 0.3
    return rng.uniform(lower + margin, upper - margin)


def plan_to_random_config(
    arm, rng: np.random.Generator, seed: int, max_attempts: int = 15
):
    """Try planning to random configurations until one succeeds.

    First checks if the random config is collision-free before attempting
    to plan, which saves time on invalid goals.
    """
    checker = arm._get_collision_checker()

    for attempt in range(max_attempts):
        q_target = random_valid_config(arm, rng)

        # Only try planning if the goal configuration is valid
        if not checker.is_valid(q_target):
            continue

        path = arm.plan_to_configuration(q_target, seed=seed + attempt)
        if path is not None:
            return path

    return None


def random_ee_pose(arm, rng: np.random.Generator) -> np.ndarray:
    """Generate a random EE pose near current position."""
    current = arm.get_ee_pose()
    target = current.copy()
    # Random position offset
    target[0, 3] += rng.uniform(-0.15, 0.15)  # x
    target[1, 3] += rng.uniform(-0.15, 0.15)  # y
    target[2, 3] += rng.uniform(-0.10, 0.10)  # z
    return target


def plan_to_pose(arm, target_pose, seed=42):
    """Plan to a Cartesian pose using IK + planning.

    Gets all IK solutions and tries planning to each until one succeeds.
    """
    try:
        # Get all IK solutions (up to 8 for UR5e)
        solutions = arm.inverse_kinematics(target_pose, validate=True)
        if not solutions:
            return None

        # Try planning to each solution until one works
        for i, q_goal in enumerate(solutions):
            path = arm.plan_to_configuration(q_goal, seed=seed + i)
            if path is not None:
                return path

        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Visual demo of Geodude capabilities")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("Geodude Visual Demo")
    print("=" * 60)
    print("\nInitializing robot...")
    sys.stdout.flush()

    robot = Geodude()

    # Override default 10% speed with 50% for better visualization
    limits_50 = KinematicLimits.ur5e_default(vel_scale=0.5, acc_scale=0.5)
    robot.left_arm.config.kinematic_limits = limits_50
    robot.right_arm.config.kinematic_limits = limits_50

    print(f"Model: {robot.config.model_path.name}")
    print(f"Arms: {robot.left_arm.dof} DOF each")
    print(f"Bases: {robot.left_base.height_range[1]}m travel")
    sys.stdout.flush()

    # Start at home
    robot.go_to("home")
    # Initialize actuator commands to match home position
    q_left_home = robot.left_arm.get_joint_positions()
    q_right_home = robot.right_arm.get_joint_positions()
    for joint_idx, actuator_id in enumerate(robot.left_arm.actuator_ids):
        robot.data.ctrl[actuator_id] = q_left_home[joint_idx]
    for joint_idx, actuator_id in enumerate(robot.right_arm.actuator_ids):
        robot.data.ctrl[actuator_id] = q_right_home[joint_idx]

    print("\nLaunching viewer...")
    print("Watch the robot explore random configurations.")
    print("Close the window to exit.\n")
    sys.stdout.flush()

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        setup_camera(viewer)
        viewer.sync()
        time.sleep(1.0)

        # ---- Setup: Move arms to ready positions first ----
        # This gets both arms out of each other's way for subsequent movements
        print("0. Moving arms to ready positions...")

        # Move left arm to ready first (direct set since we're at home)
        q_ready_left = np.array(robot.named_poses["ready"]["left"])
        robot.left_arm.set_joint_positions(q_ready_left)
        # Set actuator commands to match position
        for joint_idx, actuator_id in enumerate(robot.left_arm.actuator_ids):
            robot.data.ctrl[actuator_id] = q_ready_left[joint_idx]
        # Also set right arm actuators to hold current position
        q_right = robot.right_arm.get_joint_positions()
        for joint_idx, actuator_id in enumerate(robot.right_arm.actuator_ids):
            robot.data.ctrl[actuator_id] = q_right[joint_idx]
        mujoco.mj_forward(robot.model, robot.data)
        viewer.sync()
        time.sleep(0.3)

        # Now plan right arm to ready (left arm is out of the way)
        q_ready_right = np.array(robot.named_poses["ready"]["right"])
        plan_and_execute(robot, robot.right_arm, q_ready_right, viewer, seed=args.seed)
        time.sleep(0.5)

        # ---- Demo 1: Random joint configurations ----
        print("\n1. Exploring random joint configurations...")
        sys.stdout.flush()

        q_ready_right = np.array(robot.named_poses["ready"]["right"])
        q_ready_left = np.array(robot.named_poses["ready"]["left"])

        for i in range(3):
            # Right arm to random config (with retries)
            print(f"   Right arm: random config {i + 1}")
            path = plan_to_random_config(
                robot.right_arm, rng, seed=args.seed + i * 20, max_attempts=10
            )
            if path:
                success = execute_path_with_viewer(robot, robot.right_arm, path, viewer)
                time.sleep(0.2)
                # Only return to ready if the trajectory succeeded
                if success:
                    path_back = robot.right_arm.plan_to_configuration(
                        q_ready_right, seed=args.seed + i * 20 + 5
                    )
                    if path_back:
                        execute_path_with_viewer(robot, robot.right_arm, path_back, viewer)
                else:
                    print("      SKIPPING return to ready due to trajectory failure")
                    sys.stdout.flush()
            else:
                print("      (no valid path found)")
            time.sleep(0.3)

            # Left arm to random config (with retries)
            print(f"   Left arm: random config {i + 1}")
            path = plan_to_random_config(
                robot.left_arm, rng, seed=args.seed + i * 20 + 100, max_attempts=10
            )
            if path:
                success = execute_path_with_viewer(robot, robot.left_arm, path, viewer)
                time.sleep(0.2)
                # Only return to ready if the trajectory succeeded
                if success:
                    path_back = robot.left_arm.plan_to_configuration(
                        q_ready_left, seed=args.seed + i * 20 + 105
                    )
                    if path_back:
                        execute_path_with_viewer(robot, robot.left_arm, path_back, viewer)
                else:
                    print("      SKIPPING return to ready due to trajectory failure")
                    sys.stdout.flush()
            else:
                print("      (no valid path found)")
            time.sleep(0.3)

        # ---- Demo 2: Vention base movement ----
        print("\n2. Moving Vention bases...")

        # Left base up
        print("   Left base: up to 0.35m")
        animate_base_move(robot, robot.left_base, 0.35, viewer)
        time.sleep(0.2)

        # Right base up
        print("   Right base: up to 0.45m")
        animate_base_move(robot, robot.right_base, 0.45, viewer)
        time.sleep(0.2)

        # Left base down
        print("   Left base: down to 0.15m")
        animate_base_move(robot, robot.left_base, 0.15, viewer)
        time.sleep(0.2)

        # Right base down
        print("   Right base: down to 0.1m")
        animate_base_move(robot, robot.right_base, 0.1, viewer)
        time.sleep(0.3)

        # ---- Demo 3: IK to random Cartesian poses ----
        print("\n3. IK to random Cartesian targets...")

        # Move both arms to ready for good starting positions
        print("   Moving arms to ready positions...")
        q_ready_right = np.array(robot.named_poses["ready"]["right"])
        q_ready_left = np.array(robot.named_poses["ready"]["left"])

        path = robot.right_arm.plan_to_configuration(q_ready_right, seed=args.seed)
        if path:
            execute_path_with_viewer(robot, robot.right_arm, path, viewer)

        path = robot.left_arm.plan_to_configuration(q_ready_left, seed=args.seed)
        if path:
            execute_path_with_viewer(robot, robot.left_arm, path, viewer)
        time.sleep(0.3)

        # IK targets for right arm
        for i in range(4):
            target_pose = random_ee_pose(robot.right_arm, rng)
            print(f"   Right arm: IK target {i + 1}")

            path = plan_to_pose(robot.right_arm, target_pose, seed=args.seed + i * 10)
            if path:
                execute_path_with_viewer(robot, robot.right_arm, path, viewer)
                time.sleep(0.25)
            else:
                print("      (no valid path found)")

        # IK targets for left arm
        for i in range(4):
            target_pose = random_ee_pose(robot.left_arm, rng)
            print(f"   Left arm: IK target {i + 1}")

            path = plan_to_pose(robot.left_arm, target_pose, seed=args.seed + i * 10 + 200)
            if path:
                execute_path_with_viewer(robot, robot.left_arm, path, viewer)
                time.sleep(0.25)
            else:
                print("      (no valid path found)")

        # ---- Demo 4: More base movement ----
        print("\n4. More base movement...")

        print("   Left base: up to 0.4m")
        animate_base_move(robot, robot.left_base, 0.4, viewer, steps=40)
        time.sleep(0.2)

        print("   Right base: up to 0.35m")
        animate_base_move(robot, robot.right_base, 0.35, viewer, steps=40)
        time.sleep(0.3)

        # ---- Demo 5: Return home ----
        print("\n5. Returning home...")

        # Right arm home
        print("   Right arm: home")
        q_home_right = np.array(robot.named_poses["home"]["right"])
        path = robot.right_arm.plan_to_configuration(q_home_right, seed=args.seed)
        if path:
            execute_path_with_viewer(robot, robot.right_arm, path, viewer)
        time.sleep(0.3)

        # Left arm home
        print("   Left arm: home")
        q_home_left = np.array(robot.named_poses["home"]["left"])
        path = robot.left_arm.plan_to_configuration(q_home_left, seed=args.seed)
        if path:
            execute_path_with_viewer(robot, robot.left_arm, path, viewer)
        time.sleep(0.3)

        # Bases to 0
        print("   Left base: down to 0")
        animate_base_move(robot, robot.left_base, 0.0, viewer, steps=35)
        print("   Right base: down to 0")
        animate_base_move(robot, robot.right_base, 0.0, viewer, steps=35)

        print("\n" + "=" * 60)
        print("Demo complete! Close the viewer window to exit.")
        print("=" * 60)

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)


if __name__ == "__main__":
    main()
