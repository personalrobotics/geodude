#!/usr/bin/env python3
"""Recycling Demo - Pick objects and place them in a bin.

Demonstrates:
- TSR-based grasp and place planning
- Parallel planning at different Vention base heights
- Interchangeable kinematic/physics execution

Usage:
    uv run mjpython examples/recycle_objects.py
    uv run mjpython examples/recycle_objects.py --physics
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
from tsr import TSR
from tsr.core.tsr_primitive import load_template_file

GEODUDE_TSR_DIR = Path(__file__).parent.parent / "tsr_templates"


def create_scene_with_object():
    """Load Geodude model and add a can and recycle bin."""
    import re
    import tempfile

    from geodude_assets import get_model_path
    from prl_assets import OBJECTS_DIR

    xml_path = get_model_path()
    with open(xml_path) as f:
        xml_content = f.read()

    meshdir = str(xml_path.parent) + "/"
    xml_content = xml_content.replace(
        '<compiler autolimits="true" angle="radian"/>',
        f'<compiler autolimits="true" angle="radian" meshdir="{meshdir}"/>',
    )

    # Load recycle bin from prl_assets
    bin_xml_path = OBJECTS_DIR / "recycle_bin" / "recycle_bin.xml"
    with open(bin_xml_path) as f:
        bin_xml = f.read()

    bin_body = re.search(r'(<body name="recycle_bin".*?</body>)', bin_xml, re.DOTALL)
    bin_material = re.search(r'(<material name="bin_blue"[^/]*/?>)', bin_xml)

    if bin_material:
        if "<asset>" in xml_content:
            xml_content = xml_content.replace("<asset>", f"<asset>\n    {bin_material.group(1)}")
        else:
            xml_content = xml_content.replace(
                "<worldbody>", f"<asset>\n    {bin_material.group(1)}\n  </asset>\n\n  <worldbody>"
            )

    objects_xml = f"""
    <!-- Graspable can -->
    <body name="can" pos="0.4 -0.2 0.81">
      <freejoint name="can_joint"/>
      <geom name="can_geom" type="cylinder" size="0.033 0.06"
            rgba="0.8 0.1 0.1 1" mass="0.05"
            contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>
    </body>

    <!-- Recycle bin -->
    <body name="recycle_bin_base" pos="0.75 -0.35 0.50">
      {bin_body.group(1) if bin_body else ""}
    </body>
  </worldbody>"""

    xml_content = xml_content.replace("</worldbody>", objects_xml)

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    temp_file.write(xml_content)
    temp_file.close()
    return temp_file.name


def load_grasp_tsrs(object_pos: np.ndarray, pregrasp_standoff: float = 0.15):
    """Load can side grasp TSRs (pregrasp + grasp)."""
    template = load_template_file(str(GEODUDE_TSR_DIR / "grasps" / "can_side_grasp.yaml"))

    # T0_w is the object frame - use MuJoCo body position directly (center origin)
    object_pose = np.eye(4)
    object_pose[:3, 3] = object_pos

    grasp_tsr = TSR(T0_w=object_pose, Tw_e=template.Tw_e, Bw=template.Bw)

    Tw_e_pregrasp = template.Tw_e.copy()
    Tw_e_pregrasp[0, 3] += pregrasp_standoff
    pregrasp_tsr = TSR(T0_w=object_pose, Tw_e=Tw_e_pregrasp, Bw=template.Bw)

    grasp_tsr = compensate_tsr_for_gripper(grasp_tsr, template.subject)
    pregrasp_tsr = compensate_tsr_for_gripper(pregrasp_tsr, template.subject)

    return pregrasp_tsr, grasp_tsr


def load_place_tsr(model, data):
    """Load recycle bin drop TSR."""
    template = load_template_file(str(GEODUDE_TSR_DIR / "places" / "recycle_bin_drop.yaml"))

    bin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "recycle_bin")
    bin_pose = np.eye(4)
    bin_pose[:3, 3] = data.xpos[bin_id].copy()

    place_tsr = TSR(T0_w=bin_pose, Tw_e=template.Tw_e, Bw=template.Bw)
    place_tsr = compensate_tsr_for_gripper(place_tsr, template.subject)

    return place_tsr


def plan_with_base_heights(arm, base, tsr, base_heights, timeout=15.0):
    """Plan to TSR trying different base heights in parallel. First success wins."""
    from pycbirrt import CBiRRTConfig

    q_start = arm.get_joint_positions().copy()
    base_joint = base.config.joint_name

    config = CBiRRTConfig(
        timeout=timeout,
        max_iterations=5000,
        step_size=0.1,
        goal_bias=0.1,
        smoothing_iterations=100,
    )

    def plan_at_height(height):
        try:
            planner = arm.create_planner(config, base_joint_name=base_joint, base_height=height)
            path = planner.plan(q_start, goal_tsrs=[tsr])
            return (height, path)
        except Exception as e:
            print(f"   Height {height:.2f}m: {e}", flush=True)
            return (height, None)

    with ThreadPoolExecutor(max_workers=len(base_heights)) as executor:
        futures = {executor.submit(plan_at_height, h): h for h in base_heights}

        while futures:
            done, _ = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
            for future in done:
                height, path = future.result()
                if path is not None:
                    for f in futures:
                        f.cancel()
                    return height, path
                del futures[future]

    return None, None


def move_base(base, target, viewer, model, data):
    """Animate base movement."""
    start = base.height
    for i in range(30):
        h = start + (i / 29) * (target - start)
        base.set_height(h)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.02)


def execute_path(arm, path, executor):
    """Execute a planned path with time-optimal retiming."""
    trajectory = Trajectory.from_path(
        path,
        arm.config.kinematic_limits.velocity,
        arm.config.kinematic_limits.acceleration,
    )
    executor.execute(trajectory)


def linear_move(arm, q_target, executor, steps=50):
    """Linear interpolation to target (for approach/retract without collision check)."""
    q_start = arm.get_joint_positions()
    path = [q_start + (i / steps) * (q_target - q_start) for i in range(steps + 1)]
    execute_path(arm, path, executor)


def run_demo(robot, executor, viewer, use_physics: bool, already_at_ready: bool = False, controller=None):
    """Run the pick and place demo."""
    print("Recycling Demo", flush=True)
    print("=" * 40, flush=True)
    print(f"Mode: {'Physics' if use_physics else 'Kinematic'}", flush=True)

    model = robot.model
    data = robot.data
    arm = robot.right_arm
    base = robot.right_base
    gripper = arm.gripper

    # Get object position (MuJoCo body frame is at object center)
    can_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "can")
    object_pos = data.xpos[can_id].copy()
    print(f"\nCan position: {object_pos.round(3)}", flush=True)

    # 1. Go to ready (skip if already there from physics mode setup)
    print("\n1. Moving to ready pose...", flush=True)
    if not already_at_ready:
        robot.go_to("ready")
    viewer.sync()
    time.sleep(0.5)

    # 2. Plan to grasp with parallel base heights
    print("\n2. Planning to grasp...", flush=True)
    _, grasp_tsr = load_grasp_tsrs(object_pos, pregrasp_standoff=0.20)

    current_height = base.height if base else 0.0
    base_heights = sorted(set([current_height, 0.0, 0.2, 0.4]))
    print(f"   Trying base heights: {base_heights}", flush=True)

    if base:
        t0 = time.perf_counter()
        winning_height, path = plan_with_base_heights(arm, base, grasp_tsr, base_heights)
        if path:
            print(f"   Height {winning_height:.2f}m won in {time.perf_counter()-t0:.2f}s", flush=True)
            move_base(base, winning_height, viewer, model, data)
    else:
        path = arm.plan_to_tsrs([grasp_tsr], timeout=15.0)

    if not path:
        print("   Planning failed!", flush=True)
        return

    print(f"   Path: {len(path)} waypoints", flush=True)

    # 3. Execute to grasp
    print("\n3. Executing to grasp...", flush=True)
    execute_path(arm, path, executor)
    time.sleep(0.3)

    # 4. Close gripper
    print("\n4. Closing gripper...", flush=True)
    gripper.set_candidate_objects(["can"])
    if use_physics:
        # Use controller's gripper method to maintain arm positions during close
        controller.close_gripper("right", steps=200)
    else:
        gripper.kinematic_close()
    # Mark grasp for collision checking during planning (both modes)
    robot.grasp_manager.mark_grasped("can", "right")
    robot.grasp_manager.attach_object("can", "right_ur5e/gripper/right_follower")
    viewer.sync()
    time.sleep(0.3)

    # 5. Lift to clear table
    print("\n5. Lifting...", flush=True)
    lift_pose = arm.get_ee_pose().copy()
    lift_pose[2, 3] += 0.05
    solutions = arm.inverse_kinematics(lift_pose, validate=False)
    if solutions:
        linear_move(arm, solutions[0], executor, steps=20)
        if not use_physics:
            robot.grasp_manager.update_attached_poses()
    viewer.sync()
    time.sleep(0.2)

    # 6. Plan to place
    print("\n6. Planning to place (recycle bin)...", flush=True)
    place_tsr = load_place_tsr(model, data)
    path = arm.plan_to_tsrs([place_tsr], timeout=15.0)

    if not path:
        print("   Planning failed!", flush=True)
        return

    print(f"   Path: {len(path)} waypoints", flush=True)
    execute_path(arm, path, executor)
    time.sleep(0.3)

    # 7. Release
    print("\n7. Opening gripper...", flush=True)
    if use_physics:
        # Use controller's gripper method to maintain arm positions during open
        controller.open_gripper("right", steps=100)
    else:
        gripper.kinematic_open()
    # Mark release (both modes)
    robot.grasp_manager.mark_released("can")
    robot.grasp_manager.detach_object("can")
    viewer.sync()
    time.sleep(0.3)

    # 8. Retract
    print("\n8. Retracting...", flush=True)
    retract_pose = arm.get_ee_pose().copy()
    retract_pose[2, 3] += 0.15
    solutions = arm.inverse_kinematics(retract_pose, validate=True, sort_by_distance=True)
    if solutions:
        path = arm.plan_to_configuration(solutions[0], timeout=10.0)
        if path:
            execute_path(arm, path, executor)

    # Final position
    final_pos = data.xpos[can_id].copy()
    print(f"\nCan final position: {final_pos.round(3)}", flush=True)
    print(f"Displacement: {np.linalg.norm(final_pos - object_pos):.3f}m", flush=True)
    print("\nDemo complete. Close viewer to exit.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Recycling Demo")
    parser.add_argument("--physics", action="store_true", help="Use physics execution")
    args = parser.parse_args()

    xml_path = create_scene_with_object()
    robot = Geodude.from_xml(xml_path)
    model = robot.model
    data = robot.data
    arm = robot.right_arm

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0.4, 0, 0.8]

        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.5)

        # Create executor based on mode
        if args.physics:
            # In physics mode, go to ready BEFORE creating controller
            # so controller captures the ready positions (not initial positions)
            robot.go_to("ready")
            mujoco.mj_forward(model, data)
            viewer.sync()

            # RobotPhysicsController manages ALL actuators - arms not executing
            # trajectories automatically hold their positions
            controller = RobotPhysicsController(robot, viewer=viewer)
            executor = controller.get_executor("right")
            already_at_ready = True
        else:
            controller = None
            executor = KinematicExecutor(
                model=model,
                data=data,
                joint_qpos_indices=arm.joint_qpos_indices,
                viewer=viewer,
                grasp_manager=robot.grasp_manager,
            )
            already_at_ready = False

        try:
            run_demo(robot, executor, viewer, args.physics, already_at_ready, controller)
            while viewer.is_running():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted", flush=True)


if __name__ == "__main__":
    main()
