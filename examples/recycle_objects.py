#!/usr/bin/env python3
"""Bimanual Recycling Demo - Both arms race to pick and place objects.

Usage:
    uv run mjpython examples/recycle_objects.py
    uv run mjpython examples/recycle_objects.py --physics
    uv run mjpython examples/recycle_objects.py --base-physics  # Use physics executor for base
"""

import argparse
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from geodude import Geodude
from geodude.executor import KinematicExecutor, RobotPhysicsController
from geodude.trajectory import Trajectory
from geodude.tsr_utils import compensate_tsr_for_gripper
from pycbirrt import CBiRRTConfig
from tsr import TSR
from tsr.core.tsr_primitive import load_template_file

TSR_DIR = Path(__file__).parent.parent / "tsr_templates"

# Bin positions (symmetric)
RIGHT_BIN_POS = [0.75, -0.35, 0.50]
LEFT_BIN_POS = [-0.75, -0.35, 0.50]

# Base heights to try in parallel
BASE_HEIGHTS = [0.0, 0.2, 0.4]


def sample_can_placement(model, data):
    """Sample random can placement on worktop (upright or flipped)."""
    import random

    templates = [
        load_template_file(str(TSR_DIR / "places" / "can_on_table_upright.yaml")),
        load_template_file(str(TSR_DIR / "places" / "can_on_table_flipped.yaml")),
    ]
    template = random.choice(templates)

    worktop_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    worktop_pos = data.site_xpos[worktop_id].copy()

    # Narrower bounds for reachability
    Bw = template.Bw.copy()
    Bw[0, :] = [-0.4, 0.4]
    Bw[1, :] = [-0.25, 0.25]

    tsr = TSR(Bw=Bw)
    xyzrpy = tsr.sample_xyzrpy()
    pos = worktop_pos + xyzrpy[:3]

    rot = TSR.rpy_to_rot(xyzrpy[3:6])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())

    return pos, quat


def load_grasp_tsr(object_pos):
    """Load side grasp TSR for can at given position."""
    template = load_template_file(str(TSR_DIR / "grasps" / "can_side_grasp.yaml"))

    T0_w = np.eye(4)
    T0_w[:3, 3] = object_pos

    tsr = TSR(T0_w=T0_w, Tw_e=template.Tw_e, Bw=template.Bw)
    return compensate_tsr_for_gripper(tsr, template.subject)


def load_place_tsr(model, data, bin_name):
    """Load drop TSR for recycle bin."""
    template = load_template_file(str(TSR_DIR / "places" / "recycle_bin_drop.yaml"))

    bin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bin_name)
    T0_w = np.eye(4)
    T0_w[:3, 3] = data.xpos[bin_id].copy()

    tsr = TSR(T0_w=T0_w, Tw_e=template.Tw_e, Bw=template.Bw)
    return compensate_tsr_for_gripper(tsr, template.subject)


def plan_bimanual_grasp(robot, grasp_tsr, timeout=15.0):
    """Plan grasp with both arms at multiple heights. First success wins."""
    config = CBiRRTConfig(timeout=timeout, max_iterations=5000, step_size=0.1, goal_bias=0.1)

    def plan_arm_at_height(arm_name, height):
        try:
            arm = robot.left_arm if arm_name == "left" else robot.right_arm
            base = robot.left_base if arm_name == "left" else robot.right_base

            planner = arm.create_planner(config, base_joint_name=base.config.joint_name, base_height=height)
            path = planner.plan(arm.get_joint_positions(), goal_tsrs=[grasp_tsr])
            return (arm_name, height, path)
        except Exception as e:
            print(f"   {arm_name} @ {height:.2f}m: {e}", flush=True)
            return (arm_name, height, None)

    # All combinations: 2 arms x 3 heights = 6 parallel planners
    tasks = [(arm, h) for arm in ["left", "right"] for h in BASE_HEIGHTS]

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(plan_arm_at_height, arm, h): (arm, h) for arm, h in tasks}

        while futures:
            done, _ = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
            for future in done:
                arm_name, height, path = future.result()
                if path is not None:
                    for f in futures:
                        f.cancel()
                    return arm_name, height, path
                del futures[future]

    return None, None, None


def move_base(base, target, viewer, executor_type="kinematic"):
    """Move base with hardware-realistic motion profile."""
    base.move_to(target, viewer=viewer, executor_type=executor_type)


def execute_path(arm, path, executor):
    """Execute path with time-optimal retiming."""
    traj = Trajectory.from_path(path, arm.config.kinematic_limits.velocity, arm.config.kinematic_limits.acceleration)
    executor.execute(traj)


def run_cycle(robot, executors, viewer, use_physics, controller=None, cycle=1, base_executor="kinematic"):
    """Run one pick-and-place cycle. Returns True on success."""
    print(f"\n{'=' * 40}\nCycle {cycle}\n{'=' * 40}", flush=True)

    model, data = robot.model, robot.data

    # Get can position
    can_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "can_0")
    can_pos = data.xpos[can_id].copy()
    print(f"Can at {can_pos.round(3)}", flush=True)

    # 1. Plan grasp (both arms race)
    print("\n1. Planning grasp...", flush=True)
    grasp_tsr = load_grasp_tsr(can_pos)

    t0 = time.perf_counter()
    arm_name, height, path = plan_bimanual_grasp(robot, grasp_tsr)

    if path is None:
        print("   Failed!", flush=True)
        return False

    print(f"   {arm_name.upper()} @ {height:.1f}m in {time.perf_counter()-t0:.1f}s ({len(path)} wp)", flush=True)

    # Get winning arm's components
    arm = robot.left_arm if arm_name == "left" else robot.right_arm
    base = robot.left_base if arm_name == "left" else robot.right_base
    gripper = arm.gripper
    executor = executors[arm_name]
    bin_name = "recycle_bin_1" if arm_name == "left" else "recycle_bin_0"

    # Move base
    move_base(base, height, viewer, base_executor)

    # 2. Execute grasp
    print(f"\n2. Grasping ({arm_name})...", flush=True)
    execute_path(arm, path, executor)
    time.sleep(0.2)

    # 3. Close gripper
    gripper.set_candidate_objects(["can_0"])
    if use_physics:
        controller.close_gripper(arm_name, steps=200)
    else:
        gripper.kinematic_close()
    robot.grasp_manager.mark_grasped("can_0", arm_name)
    robot.grasp_manager.attach_object("can_0", f"{arm_name}_ur5e/gripper/right_follower")
    viewer.sync()

    # 4. Lift
    print("\n3. Lifting...", flush=True)
    lift_pose = arm.get_ee_pose().copy()
    lift_pose[2, 3] += 0.05
    ik = arm.inverse_kinematics(lift_pose, validate=False)
    if ik:
        q_start = arm.get_joint_positions()
        lift_path = [q_start + (i / 20) * (ik[0] - q_start) for i in range(21)]
        execute_path(arm, lift_path, executor)
        if not use_physics:
            robot.grasp_manager.update_attached_poses()
    viewer.sync()

    # 5. Plan to bin
    print(f"\n4. Planning to {bin_name}...", flush=True)
    place_tsr = load_place_tsr(model, data, bin_name)
    path = arm.plan_to_tsrs([place_tsr], timeout=15.0)

    if not path:
        print("   Failed!", flush=True)
        return False

    print(f"   {len(path)} waypoints", flush=True)
    execute_path(arm, path, executor)
    time.sleep(0.2)

    # 6. Release
    print("\n5. Releasing...", flush=True)
    if use_physics:
        controller.open_gripper(arm_name, steps=100)
    else:
        gripper.kinematic_open()
    robot.grasp_manager.mark_released("can_0")
    robot.grasp_manager.detach_object("can_0")
    viewer.sync()

    # Hide can
    robot.env.registry.hide("can_0")
    mujoco.mj_forward(model, data)
    viewer.sync()

    # 7. Return to ready
    print(f"\n6. Returning to ready...", flush=True)
    ready_config = np.array(robot.named_poses["ready"][arm_name])
    path = arm.plan_to_configuration(ready_config, timeout=10.0)
    if path:
        execute_path(arm, path, executor)

    print(f"\nDone! {arm_name.upper()} arm recycled can.", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Bimanual Recycling Demo")
    parser.add_argument("--physics", action="store_true", help="Use physics simulation for arms")
    parser.add_argument("--base-physics", action="store_true", help="Use physics executor for base")
    args = parser.parse_args()

    base_executor = "physics" if args.base_physics else "kinematic"

    # Create robot with objects
    robot = Geodude(objects={"can": 1, "recycle_bin": 2})
    model, data = robot.model, robot.data

    # Place can and bins
    can_pos, can_quat = sample_can_placement(model, data)
    print(f"Can at {can_pos.round(3)}")
    robot.env.registry.activate("can", pos=can_pos, quat=can_quat)
    robot.env.registry.activate("recycle_bin", pos=RIGHT_BIN_POS)
    robot.env.registry.activate("recycle_bin", pos=LEFT_BIN_POS)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Preferred camera view for paper-quality screenshots
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -26.5
        viewer.cam.distance = 2.96
        viewer.cam.lookat[:] = [0.188, 0.001, 1.141]

        mujoco.mj_forward(model, data)
        viewer.sync()

        robot.go_to("ready")
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.5)

        # Create executors
        if args.physics:
            controller = RobotPhysicsController(robot, viewer=viewer)
            executors = {"left": controller.get_executor("left"), "right": controller.get_executor("right")}
        else:
            controller = None
            executors = {
                "left": KinematicExecutor(
                    model, data, robot.left_arm.joint_qpos_indices,
                    viewer=viewer, grasp_manager=robot.grasp_manager
                ),
                "right": KinematicExecutor(
                    model, data, robot.right_arm.joint_qpos_indices,
                    viewer=viewer, grasp_manager=robot.grasp_manager
                ),
            }

        print(f"Bimanual Recycling Demo ({'Physics' if args.physics else 'Kinematic'})")
        print("Close viewer or Ctrl+C to exit\n")

        try:
            cycle = 1
            while viewer.is_running():
                success = run_cycle(robot, executors, viewer, args.physics, controller, cycle, base_executor)

                if not viewer.is_running():
                    break

                # Spawn new can
                print("\nSpawning new can...", flush=True)
                can_pos, can_quat = sample_can_placement(model, data)
                print(f"Can at {can_pos.round(3)}")
                robot.env.registry.activate("can", pos=can_pos, quat=can_quat)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.5)

                if success:
                    cycle += 1

        except KeyboardInterrupt:
            print("\nInterrupted")

        print(f"\nCompleted {cycle - 1} cycles.")


if __name__ == "__main__":
    main()
