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

    IMPORTANT: Actively holds the other arm in place during execution.
    """
    from geodude.trajectory import Trajectory

    # Get tracking threshold from config (abort if any joint exceeds this)
    max_joint_error_threshold = arm.config.tracking_thresholds.max_error

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

    # Get executor for moving arm (physics for realistic dynamics)
    executor = arm._get_executor(viewer=viewer, executor_type="physics")

    # Get the other arm to hold it in place during execution
    other_arm = robot.left_arm if arm.name == "right" else robot.right_arm
    other_executor = other_arm._get_executor(viewer=None, executor_type="physics")
    # Capture hold position AFTER any transition (not before)
    other_q_hold = other_arm.get_joint_positions()

    # Track per-joint errors for reporting
    moving_arm_errors = []

    # Detect if using kinematic or physics execution
    from geodude.executor import KinematicExecutor
    use_kinematic = isinstance(executor, KinematicExecutor)

    if use_kinematic:
        print(f"      Executing {trajectory.num_waypoints} waypoints (KINEMATIC - perfect tracking)...")
    else:
        lookahead_time = executor.lookahead_time
        print(f"      Executing {trajectory.num_waypoints} waypoints (lookahead={lookahead_time}s)...")
    sys.stdout.flush()

    for i in range(trajectory.num_waypoints):
        q_desired = trajectory.positions[i]
        qd_desired = trajectory.velocities[i]

        if use_kinematic:
            # === KINEMATIC: Directly set joint positions (perfect tracking) ===
            for joint_idx, qpos_idx in enumerate(executor.joint_qpos_indices):
                robot.data.qpos[qpos_idx] = q_desired[joint_idx]
                robot.data.qvel[qpos_idx] = qd_desired[joint_idx]

            # Hold other arm at its position
            for joint_idx, qpos_idx in enumerate(other_executor.joint_qpos_indices):
                robot.data.qpos[qpos_idx] = other_q_hold[joint_idx]
                robot.data.qvel[qpos_idx] = 0.0

            # Forward kinematics only (no dynamics)
            mujoco.mj_forward(robot.model, robot.data)
        else:
            # === PHYSICS: Send commands to actuators with feedforward ===
            q_command = q_desired + lookahead_time * qd_desired

            for joint_idx, actuator_id in enumerate(executor.actuator_ids):
                robot.data.ctrl[actuator_id] = q_command[joint_idx]

            # Hold other arm in place
            for joint_idx, actuator_id in enumerate(other_executor.actuator_ids):
                robot.data.ctrl[actuator_id] = other_q_hold[joint_idx]

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

        # Track errors after physics (per-joint)
        q_actual = np.array([robot.data.qpos[idx] for idx in executor.joint_qpos_indices])
        joint_errors = np.abs(trajectory.positions[i] - q_actual)
        max_joint_error = np.max(joint_errors)
        moving_arm_errors.append(joint_errors)

        # Check if any joint exceeds threshold - abort if so
        if max_joint_error > max_joint_error_threshold:
            worst_joint = np.argmax(joint_errors)
            print(f"        ABORT: Joint {worst_joint} error {np.rad2deg(max_joint_error):.1f}° exceeds {np.rad2deg(max_joint_error_threshold):.0f}° at waypoint {i}")
            sys.stdout.flush()
            return False

        # Print real-time error every 20 waypoints
        if i > 0 and i % 20 == 0:
            recent_errors = np.array(moving_arm_errors[-20:])
            recent_max = np.rad2deg(np.max(recent_errors))
            print(f"        Waypoint {i}/{trajectory.num_waypoints}: max_joint_error={recent_max:.2f}°")
            sys.stdout.flush()

        time.sleep(executor.control_dt)

    # Final settling
    q_final = trajectory.positions[-1]

    if use_kinematic:
        # Kinematic: just set final position (already there)
        for joint_idx, qpos_idx in enumerate(executor.joint_qpos_indices):
            robot.data.qpos[qpos_idx] = q_final[joint_idx]
            robot.data.qvel[qpos_idx] = 0.0
        mujoco.mj_forward(robot.model, robot.data)
    else:
        # Physics: let actuators settle
        for _ in range(executor.steps_per_control * 20):
            for joint_idx, actuator_id in enumerate(executor.actuator_ids):
                robot.data.ctrl[actuator_id] = q_final[joint_idx]
            for joint_idx, actuator_id in enumerate(other_executor.actuator_ids):
                robot.data.ctrl[actuator_id] = other_q_hold[joint_idx]
            mujoco.mj_step(robot.model, robot.data)

    # Final viewer sync
    if viewer is not None:
        viewer.sync()

    # Report tracking statistics (per-joint)
    all_errors = np.array(moving_arm_errors)  # Shape: (num_waypoints, num_joints)
    avg_per_joint = np.rad2deg(np.mean(all_errors, axis=0))
    max_per_joint = np.rad2deg(np.max(all_errors, axis=0))

    print(f"      {arm.name.upper()} arm tracking (per joint):")
    print(f"        Avg: {avg_per_joint.round(2)}°")
    print(f"        Max: {max_per_joint.round(2)}°")
    print(f"      SUCCESS: Trajectory completed (threshold: {np.rad2deg(max_joint_error_threshold):.0f}°)")

    sys.stdout.flush()

    return True


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
