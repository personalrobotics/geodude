"""Test symmetric grasping at different Vention mount heights.

Computes IK solutions for both arms to grasp at a fixed position in front
of the robot, while varying the Vention linear actuator heights.

Usage:
    uv run mjpython examples/symmetric_grasp_test.py
"""

import mujoco
import numpy as np
from pathlib import Path
from geodude import Geodude


def pose_from_position_and_z_axis(position: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """Create a 4x4 pose matrix from position and z-axis (gripper approach direction)."""
    z = z_axis / np.linalg.norm(z_axis)

    if abs(np.dot(z, [0, 1, 0])) < 0.9:
        x = np.cross([0, 1, 0], z)
    else:
        x = np.cross([0, 0, 1], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = position
    return pose


def save_screenshot(model, data, filename: str, width: int = 640, height: int = 480):
    """Save a screenshot of the current scene from the front."""
    renderer = mujoco.Renderer(model, width=width, height=height)

    # Camera setup - looking from in front of the robot (from +X toward -X)
    # The robot faces +X, so we want to look at it from in front
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 1.2]  # Look at workspace center
    camera.distance = 2.5
    camera.azimuth = -90  # Camera at -X looking toward +X (front view of robot)
    camera.elevation = -5

    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)

    pixels = renderer.render()

    try:
        from PIL import Image
        img = Image.fromarray(pixels)
        img.save(filename)
    except ImportError:
        import mediapy as media
        media.write_image(filename, pixels)

    print(f"  Saved: {filename}")


def main():
    # Create output directory
    output_dir = Path("snapshots")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Symmetric Grasp Test - Varying Vention Heights")
    print("=" * 60)
    print()

    # Fixed grasp position (both arms reach to symmetric positions)
    grasp_height = 1.0  # Fixed z height for grasping
    x_forward = 0.5     # Distance in front
    y_offset = 0.35     # Symmetric left/right offset

    # Gripper orientations for side grasps (grippers pointing inward)
    z_axis_inward_left = np.array([0, -1, 0])   # Left gripper pointing right
    z_axis_inward_right = np.array([0, 1, 0])   # Right gripper pointing left

    # Test different Vention heights (the linear actuator range is 0 to 0.5)
    vention_heights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    results = []

    for vention_h in vention_heights:
        print(f"\n--- Vention height offset: {vention_h}m ---")

        # Create fresh robot instance for each test
        robot = Geodude()

        # Set the Vention linear actuator positions
        # Find the joint indices
        left_vention_jnt = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "left_arm_linear_vention")
        right_vention_jnt = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "right_arm_linear_vention")

        # Get qpos addresses
        left_qpos_adr = robot.model.jnt_qposadr[left_vention_jnt]
        right_qpos_adr = robot.model.jnt_qposadr[right_vention_jnt]

        # Set both arms to same Vention height
        robot.data.qpos[left_qpos_adr] = vention_h
        robot.data.qpos[right_qpos_adr] = vention_h

        mujoco.mj_forward(robot.model, robot.data)

        # Target positions for grasping
        left_pos = np.array([x_forward, y_offset, grasp_height])
        right_pos = np.array([x_forward, -y_offset, grasp_height])

        print(f"  Left target:  [{left_pos[0]:.2f}, {left_pos[1]:.2f}, {left_pos[2]:.2f}]")
        print(f"  Right target: [{right_pos[0]:.2f}, {right_pos[1]:.2f}, {right_pos[2]:.2f}]")

        # Side grasp poses
        left_pose = pose_from_position_and_z_axis(left_pos, z_axis_inward_left)
        right_pose = pose_from_position_and_z_axis(right_pos, z_axis_inward_right)

        # Solve IK
        left_solutions = robot.left_arm.inverse_kinematics(left_pose)
        right_solutions = robot.right_arm.inverse_kinematics(right_pose)

        left_ok = len(left_solutions) > 0
        right_ok = len(right_solutions) > 0

        print(f"  Left IK:  {'SUCCESS' if left_ok else 'FAILED'} ({len(left_solutions)} solutions)")
        print(f"  Right IK: {'SUCCESS' if right_ok else 'FAILED'} ({len(right_solutions)} solutions)")

        if left_ok and right_ok:
            # Use first solution
            left_q = left_solutions[0]
            right_q = right_solutions[0]

            robot.left_arm.set_joint_positions(left_q)
            robot.right_arm.set_joint_positions(right_q)
            mujoco.mj_forward(robot.model, robot.data)

            # Save screenshot
            filename = output_dir / f"vention_height_{vention_h:.1f}m.png"
            save_screenshot(robot.model, robot.data, str(filename))

            results.append({
                'vention_height': vention_h,
                'left_q': left_q.tolist(),
                'right_q': right_q.tolist(),
                'success': True
            })
        else:
            results.append({
                'vention_height': vention_h,
                'left_ok': left_ok,
                'right_ok': right_ok,
                'success': False
            })

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    success_count = sum(1 for r in results if r['success'])
    print(f"Successful poses: {success_count}/{len(vention_heights)}")

    for r in results:
        h = r['vention_height']
        if r['success']:
            print(f"  Vention +{h:.1f}m: OK")
        else:
            left_status = 'OK' if r.get('left_ok') else 'FAIL'
            right_status = 'OK' if r.get('right_ok') else 'FAIL'
            print(f"  Vention +{h:.1f}m: FAILED (left={left_status}, right={right_status})")

    print()
    print(f"Screenshots saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
