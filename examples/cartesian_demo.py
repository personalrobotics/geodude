#!/usr/bin/env python3
"""Cartesian Velocity Control Demo.

Demonstrates the Jacobian-based Cartesian velocity controller by moving
each arm in random directions until it can't progress further (joint limits,
singularities, workspace boundary, or collision).

Runs continuously until the viewer window is closed.

Usage:
    uv run mjpython examples/cartesian_demo.py            # Kinematic mode
    uv run mjpython examples/cartesian_demo.py --physics  # Physics mode
"""

import argparse
import logging
import time

import mujoco
import numpy as np

from geodude import Geodude
from geodude.cartesian import CartesianControlConfig, move_until_touch

# Enable debug logging for cartesian module
logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
logging.getLogger("geodude.cartesian").setLevel(logging.DEBUG)


def random_direction():
    """Generate a random unit direction vector."""
    d = np.random.randn(3)
    return d / np.linalg.norm(d)


def main():
    parser = argparse.ArgumentParser(description="Cartesian velocity control demo")
    parser.add_argument("--physics", action="store_true", help="Enable physics simulation")
    args = parser.parse_args()

    mode = "Physics" if args.physics else "Kinematic"
    print(f"Cartesian Velocity Control Demo - {mode} Mode", flush=True)
    print("=" * 50, flush=True)
    print("Each arm will move in random directions until blocked.", flush=True)
    print("Stops on: joint limits, singularities, or collisions.", flush=True)
    print("Close the viewer window to exit.\n", flush=True)

    robot = Geodude()
    robot.go_to("ready")
    mujoco.mj_forward(robot.model, robot.data)

    # Config for demo
    config = CartesianControlConfig(
        velocity_scale=0.8,
        joint_margin_deg=10.0,
        length_scale=0.5,
    )

    arms = [robot.right_arm, robot.left_arm]
    arm_names = ["Right", "Left"]

    # Track consecutive stuck counts per arm
    stuck_counts = {"Right": 0, "Left": 0}
    max_stuck_before_reset = 3  # Reset arm after 3 consecutive stuck moves

    with robot.sim(physics=args.physics) as ctx:
        ctx.sync()

        iteration = 0
        while ctx.is_running():
            iteration += 1

            for arm, name in zip(arms, arm_names):
                if not ctx.is_running():
                    break

                # Pick random direction
                direction = random_direction()

                # Get starting position
                start_pos = robot.data.site_xpos[arm.ee_site_id].copy()

                print(f"\n[{iteration}] {name} arm: moving in direction "
                      f"[{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]",
                      flush=True)

                # Use move_until_touch - it handles collision detection via data.contact
                # in kinematic mode and F/T sensor in physics mode
                print(f"    Starting move_until_touch...", flush=True)
                try:
                    result = move_until_touch(
                        arm=arm,
                        direction=direction,
                        distance=0.03,        # Ignore collisions for first 3cm
                        max_distance=0.60,    # Max 60cm per move
                        max_force=20.0,       # Force threshold (physics mode) - 20N to avoid inertial false positives
                        speed=0.05,           # 5 cm/s
                        frame="world",
                        config=config,
                    )
                    print(f"    move_until_touch returned: {result.terminated_by}", flush=True)
                    if result.final_force is not None:
                        force_mag = np.linalg.norm(result.final_force)
                        print(f"    Final force: {force_mag:.1f}N (threshold: 20N)", flush=True)
                except Exception as e:
                    print(f"    ERROR in move_until_touch: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue

                end_pos = robot.data.site_xpos[arm.ee_site_id].copy()
                actual_displacement = end_pos - start_pos

                print(f"    Terminated by: {result.terminated_by}", flush=True)
                print(f"    Distance moved: {result.distance_moved*100:.1f} cm", flush=True)
                print(f"    Displacement: [{actual_displacement[0]*100:.1f}, "
                      f"{actual_displacement[1]*100:.1f}, {actual_displacement[2]*100:.1f}] cm",
                      flush=True)

                # Track stuck count for recovery
                # Count as "stuck" if:
                # - Less than 1cm moved (truly stuck)
                # - Stopped at min_distance with contact (likely false positive)
                min_distance = 0.03  # Must match the value passed to move_until_touch
                is_stuck = (
                    result.distance_moved < 0.01 or
                    (result.terminated_by == "contact" and result.distance_moved < min_distance + 0.005)
                )
                if is_stuck:
                    stuck_counts[name] += 1
                    print(f"    Stuck count: {stuck_counts[name]} (likely false positive)", flush=True)
                else:
                    stuck_counts[name] = 0

                # Reset arm to ready if stuck too many times
                if stuck_counts[name] >= max_stuck_before_reset:
                    print(f"    {name} arm stuck {stuck_counts[name]} times, resetting to ready...",
                          flush=True)
                    ready_pos = np.array(robot.named_poses["ready"][name.lower()])

                    if args.physics and ctx._controller is not None:
                        # Use controller to reset - ensures all actuators are coordinated
                        ctx._controller.set_arm_target(name.lower(), ready_pos)
                        for _ in range(300):  # ~0.6s settling
                            ctx._controller.step()
                    else:
                        # Kinematic: use step_cartesian
                        ctx.step_cartesian(name.lower(), ready_pos)

                    stuck_counts[name] = 0
                    print(f"    Reset complete", flush=True)

                ctx.sync()
                time.sleep(0.1)  # Brief pause between moves

    print("\nDemo complete.", flush=True)


if __name__ == "__main__":
    main()
