#!/usr/bin/env python3
"""Cartesian Velocity Control Demo.

Demonstrates the Jacobian-based Cartesian velocity controller by moving
each arm in random directions until it can't progress further (joint limits,
singularities, workspace boundary, or collision).

Usage:
    uv run mjpython examples/cartesian_demo.py
"""

import time

import mujoco
import numpy as np

from geodude import Geodude
from geodude.cartesian import execute_twist, CartesianControlConfig


def random_direction():
    """Generate a random unit direction vector."""
    d = np.random.randn(3)
    return d / np.linalg.norm(d)


def check_arm_collision(robot, arm) -> bool:
    """Check if arm is in collision with anything.

    Returns True if collision detected (motion should stop).
    """
    # Use the arm's collision checker
    if arm._collision_checker is not None:
        q = arm.get_joint_positions()
        return not arm._collision_checker.is_valid(q)
    return False


def main():
    print("Cartesian Velocity Control Demo", flush=True)
    print("=" * 50, flush=True)
    print("Each arm will move in random directions until blocked.", flush=True)
    print("Stops on: joint limits, singularities, or collisions.", flush=True)
    print("Close the viewer window to exit.\n", flush=True)

    robot = Geodude()
    robot.go_to("ready")
    mujoco.mj_forward(robot.model, robot.data)

    # Slower, safer config for demo
    config = CartesianControlConfig(
        velocity_scale=0.8,
        joint_margin_deg=10.0,
    )

    arms = [robot.right_arm, robot.left_arm]
    arm_names = ["Right", "Left"]

    with robot.sim(physics=False) as ctx:
        ctx.sync()

        iteration = 0
        while ctx.is_running():
            iteration += 1

            for arm, name in zip(arms, arm_names):
                if not ctx.is_running():
                    break

                # Pick random direction
                direction = random_direction()

                # Create twist (linear velocity only)
                speed = 0.05  # 5 cm/s
                twist = np.zeros(6)
                twist[:3] = direction * speed

                # Get starting position
                start_pos = robot.data.site_xpos[arm.ee_site_id].copy()

                print(f"\n[{iteration}] {name} arm: moving in direction "
                      f"[{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]",
                      flush=True)

                # Collision check function for this arm
                def collision_check():
                    return check_arm_collision(robot, arm)

                # Execute twist until blocked, collision, or max distance
                result = execute_twist(
                    arm=arm,
                    twist=twist,
                    frame="world",
                    max_distance=0.60,  # Max 60cm per move
                    until=collision_check,
                    physics=False,
                    viewer=ctx._viewer,
                    config=config,
                )

                end_pos = robot.data.site_xpos[arm.ee_site_id].copy()
                actual_displacement = end_pos - start_pos

                # Check if we stopped due to collision
                termination = result.terminated_by
                if termination == "condition":
                    termination = "collision"

                print(f"    Terminated by: {termination}", flush=True)
                print(f"    Distance moved: {result.distance_moved*100:.1f} cm", flush=True)
                print(f"    Displacement: [{actual_displacement[0]*100:.1f}, "
                      f"{actual_displacement[1]*100:.1f}, {actual_displacement[2]*100:.1f}] cm",
                      flush=True)

                ctx.sync()
                time.sleep(0.3)  # Brief pause between moves

            # Return to ready between iterations
            print(f"\n[{iteration}] Returning to ready...", flush=True)
            robot.go_to("ready")
            mujoco.mj_forward(robot.model, robot.data)
            ctx.sync()
            time.sleep(0.5)

    print("\nDemo complete.", flush=True)


if __name__ == "__main__":
    main()
