"""Interactive viewer for finding joint configurations.

Launch this to manually position the robot. The robot joints are free
(no actuators controlling them) so you can double-click and drag to
move them.

Usage:
    uv run mjpython examples/interactive_viewer.py
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
from geodude import Geodude


def main():
    robot = Geodude()

    # Disable actuators/motors so joints can be moved freely
    # This is done by setting ctrl to 0 and letting the model have no damping
    robot.model.dof_damping[:] = 0.1  # Small damping for stability

    # Start at home
    q_right = np.array(robot.named_poses["home"]["right"])
    q_left = np.array(robot.named_poses["home"]["left"])

    robot.right_arm.set_joint_positions(q_right)
    robot.left_arm.set_joint_positions(q_left)
    mujoco.mj_forward(robot.model, robot.data)

    print("=" * 60)
    print("Interactive Robot Viewer")
    print("=" * 60)
    print()
    print("To move joints:")
    print("  1. Double-click on a robot link to select it")
    print("  2. Ctrl+Right-click and drag to apply force/torque")
    print("  3. The joint will move and stay in the new position")
    print()
    print("Or use the perturbation controls in the UI.")
    print()
    print("Close the window when done to see joint values.")
    print("=" * 60)

    viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

    while viewer.is_running():
        # Step physics to let the robot respond to perturbations
        mujoco.mj_step(robot.model, robot.data)
        viewer.sync()
        time.sleep(0.002)

    viewer.close()

    # Print final positions
    print()
    print("=" * 60)
    print("Final Joint Positions")
    print("=" * 60)

    q_right = robot.right_arm.get_joint_positions()
    q_left = robot.left_arm.get_joint_positions()

    print()
    print("Right arm:")
    print(f"  {[round(float(x), 4) for x in q_right]}")
    print()
    print("Left arm:")
    print(f"  {[round(float(x), 4) for x in q_left]}")
    print()
    print("For config.py, use:")
    print(f'  "left": {[round(float(x), 2) for x in q_left]},')
    print(f'  "right": {[round(float(x), 2) for x in q_right]},')


if __name__ == "__main__":
    main()
