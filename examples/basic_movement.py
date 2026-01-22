"""Visual demo: Arm movement, IK, and Vention base control.

Demonstrates:
- Random joint configurations with collision-free planning
- IK to random Cartesian targets (using all solutions)
- Vention base height changes
- All visualized in the MuJoCo viewer

Usage:
    uv run mjpython examples/basic_movement.py
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from geodude import Geodude


def execute_path(robot, arm, path: list[np.ndarray], viewer, dt: float = 0.01):
    """Execute a path with visualization."""
    for q in path:
        arm.set_joint_positions(q)
        mujoco.mj_forward(robot.model, robot.data)
        viewer.sync()
        time.sleep(dt)


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

    robot = Geodude()

    print(f"Model: {robot.config.model_path.name}")
    print(f"Arms: {robot.left_arm.dof} DOF each")
    print(f"Bases: {robot.left_base.height_range[1]}m travel")

    # Start at home
    robot.go_to("home")

    print("\nLaunching viewer...")
    print("Watch the robot explore random configurations.")
    print("Close the window to exit.\n")

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
        mujoco.mj_forward(robot.model, robot.data)
        viewer.sync()
        time.sleep(0.3)

        # Now plan right arm to ready (left arm is out of the way)
        q_ready_right = np.array(robot.named_poses["ready"]["right"])
        path = robot.right_arm.plan_to_configuration(q_ready_right, seed=args.seed)
        if path:
            execute_path(robot, robot.right_arm, path, viewer, dt=0.015)
        time.sleep(0.5)

        # ---- Demo 1: Random joint configurations ----
        print("\n1. Exploring random joint configurations...")

        q_ready_right = np.array(robot.named_poses["ready"]["right"])
        q_ready_left = np.array(robot.named_poses["ready"]["left"])

        for i in range(3):
            # Right arm to random config (with retries)
            print(f"   Right arm: random config {i + 1}")
            path = plan_to_random_config(
                robot.right_arm, rng, seed=args.seed + i * 20, max_attempts=10
            )
            if path:
                execute_path(robot, robot.right_arm, path, viewer, dt=0.015)
                time.sleep(0.2)
                # Return to ready
                path_back = robot.right_arm.plan_to_configuration(
                    q_ready_right, seed=args.seed + i * 20 + 5
                )
                if path_back:
                    execute_path(robot, robot.right_arm, path_back, viewer, dt=0.015)
            else:
                print("      (no valid path found)")
            time.sleep(0.3)

            # Left arm to random config (with retries)
            print(f"   Left arm: random config {i + 1}")
            path = plan_to_random_config(
                robot.left_arm, rng, seed=args.seed + i * 20 + 100, max_attempts=10
            )
            if path:
                execute_path(robot, robot.left_arm, path, viewer, dt=0.015)
                time.sleep(0.2)
                # Return to ready
                path_back = robot.left_arm.plan_to_configuration(
                    q_ready_left, seed=args.seed + i * 20 + 105
                )
                if path_back:
                    execute_path(robot, robot.left_arm, path_back, viewer, dt=0.015)
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
            execute_path(robot, robot.right_arm, path, viewer, dt=0.015)

        path = robot.left_arm.plan_to_configuration(q_ready_left, seed=args.seed)
        if path:
            execute_path(robot, robot.left_arm, path, viewer, dt=0.015)
        time.sleep(0.3)

        # IK targets for right arm
        for i in range(4):
            target_pose = random_ee_pose(robot.right_arm, rng)
            print(f"   Right arm: IK target {i + 1}")

            path = plan_to_pose(robot.right_arm, target_pose, seed=args.seed + i * 10)
            if path:
                execute_path(robot, robot.right_arm, path, viewer, dt=0.012)
                time.sleep(0.25)
            else:
                print("      (no valid path found)")

        # IK targets for left arm
        for i in range(4):
            target_pose = random_ee_pose(robot.left_arm, rng)
            print(f"   Left arm: IK target {i + 1}")

            path = plan_to_pose(robot.left_arm, target_pose, seed=args.seed + i * 10 + 200)
            if path:
                execute_path(robot, robot.left_arm, path, viewer, dt=0.012)
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
            execute_path(robot, robot.right_arm, path, viewer, dt=0.015)
        time.sleep(0.3)

        # Left arm home
        print("   Left arm: home")
        q_home_left = np.array(robot.named_poses["home"]["left"])
        path = robot.left_arm.plan_to_configuration(q_home_left, seed=args.seed)
        if path:
            execute_path(robot, robot.left_arm, path, viewer, dt=0.015)
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
