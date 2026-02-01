#!/usr/bin/env python3
"""Recycling Demo - Pick objects and place them in a bin.

Demonstrates:
- TSR-based grasp and place planning
- Parallel planning at different Vention base heights
- Interchangeable kinematic/physics execution
- Object management via mj_environment

Usage:
    uv run mjpython examples/recycle_objects.py
    uv run mjpython examples/recycle_objects.py --physics
    uv run mjpython examples/recycle_objects.py --record  # Save GIF
"""

import argparse
import subprocess
import tempfile
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

# Fixed positions
RECYCLE_BIN_POS = [0.75, -0.35, 0.50]  # On floor


def sample_can_placement(model, data):
    """Sample a random can placement from upright or flipped TSR (50/50)."""
    import random

    # Load both placement TSRs
    upright = load_template_file(str(GEODUDE_TSR_DIR / "places" / "can_on_table_upright.yaml"))
    flipped = load_template_file(str(GEODUDE_TSR_DIR / "places" / "can_on_table_flipped.yaml"))

    # Choose randomly
    template = random.choice([upright, flipped])
    is_flipped = template.name == "Can on Table (Flipped)"

    # Get worktop site pose (TSR reference frame)
    worktop_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    worktop_pos = data.site_xpos[worktop_id].copy()

    # Sample position within TSR bounds (relative to worktop)
    x = random.uniform(-0.4, 0.4)  # Narrower than full bounds for reachability
    y = random.uniform(-0.25, 0.25)
    z = 0.0615  # Half-height offset so bottom sits on surface

    # Transform to world coordinates
    pos = worktop_pos + np.array([x, y, z])

    # Quaternion: identity for upright, 180° pitch for flipped
    if is_flipped:
        # Rotation of 180° around x-axis: [cos(90°), sin(90°), 0, 0] = [0, 1, 0, 0]
        quat = np.array([0, 1, 0, 0], dtype=float)
    else:
        quat = np.array([1, 0, 0, 0], dtype=float)

    return pos, quat, is_flipped


def create_robot_with_objects():
    """Create Geodude robot with can and recycle bin from prl_assets."""
    robot = Geodude(objects={"can": 1, "recycle_bin": 1})

    # Sample can placement from TSR
    can_pos, can_quat, is_flipped = sample_can_placement(robot.model, robot.data)
    orientation = "flipped" if is_flipped else "upright"
    print(f"Can placement: {orientation} at {can_pos.round(3)}")

    # Activate objects
    robot.env.registry.activate("can", pos=can_pos, quat=can_quat)
    robot.env.registry.activate("recycle_bin", pos=RECYCLE_BIN_POS)
    mujoco.mj_forward(robot.model, robot.data)

    return robot


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

    bin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "recycle_bin_0")
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
    can_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "can_0")
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
    gripper.set_candidate_objects(["can_0"])
    if use_physics:
        # Use controller's gripper method to maintain arm positions during close
        controller.close_gripper("right", steps=200)
    else:
        gripper.kinematic_close()
    # Mark grasp for collision checking during planning (both modes)
    robot.grasp_manager.mark_grasped("can_0", "right")
    robot.grasp_manager.attach_object("can_0", "right_ur5e/gripper/right_follower")
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
    robot.grasp_manager.mark_released("can_0")
    robot.grasp_manager.detach_object("can_0")
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


class FrameRecorder:
    """Records frames for GIF creation."""

    def __init__(self, model, data, width=640, height=480):
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.model = model
        self.data = data
        self.frames = []
        self.frame_dir = tempfile.mkdtemp()

        # Set up camera for front view centered on right arm
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = 270
        self.camera.elevation = -15
        self.camera.distance = 1.8
        self.camera.lookat[:] = [0.5, -0.25, 0.85]

    def capture(self):
        """Capture current frame."""
        self.renderer.update_scene(self.data, camera=self.camera)
        frame = self.renderer.render()
        self.frames.append(frame.copy())

    def save_gif(self, output_path: str, fps: int = 15):
        """Save captured frames as GIF using ffmpeg."""
        import os

        # Save frames as PNGs
        for i, frame in enumerate(self.frames):
            from PIL import Image

            img = Image.fromarray(frame)
            img.save(f"{self.frame_dir}/frame_{i:04d}.png")

        # Use ffmpeg to create GIF with palette for quality
        palette_path = f"{self.frame_dir}/palette.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                f"{self.frame_dir}/frame_%04d.png",
                "-vf",
                "palettegen=stats_mode=diff",
                palette_path,
            ],
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                f"{self.frame_dir}/frame_%04d.png",
                "-i",
                palette_path,
                "-lavfi",
                "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                output_path,
            ],
            capture_output=True,
        )

        # Cleanup
        for f in os.listdir(self.frame_dir):
            os.remove(f"{self.frame_dir}/{f}")
        os.rmdir(self.frame_dir)

        print(f"Saved GIF to {output_path} ({len(self.frames)} frames)")


class DummyViewer:
    """Dummy viewer for headless recording mode."""

    def sync(self):
        pass

    def is_running(self):
        return False


def main():
    parser = argparse.ArgumentParser(description="Recycling Demo")
    parser.add_argument("--physics", action="store_true", help="Use physics execution")
    parser.add_argument("--record", action="store_true", help="Record and save as GIF (headless)")
    parser.add_argument("--output", default="recycle_demo.gif", help="Output GIF path")
    args = parser.parse_args()

    robot = create_robot_with_objects()
    model = robot.model
    data = robot.data
    arm = robot.right_arm

    # Setup recorder if requested (headless mode)
    recorder = None
    if args.record:
        recorder = FrameRecorder(model, data, width=480, height=360)
        print("Recording in headless mode...", flush=True)

        # Run headless
        viewer = DummyViewer()
        mujoco.mj_forward(model, data)

        controller = None
        executor = KinematicExecutor(
            model=model,
            data=data,
            joint_qpos_indices=arm.joint_qpos_indices,
            viewer=viewer,
            grasp_manager=robot.grasp_manager,
            recorder=recorder,
        )

        run_demo(robot, executor, viewer, use_physics=False, already_at_ready=False, controller=None)
        recorder.save_gif(args.output)
        return

    # Normal mode with viewer (requires mjpython on macOS)
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
