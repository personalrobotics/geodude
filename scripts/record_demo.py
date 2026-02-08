#!/usr/bin/env python3
"""Record recycling demo as animated GIF for README.

Captures frames using MuJoCo's offscreen renderer.

Usage:
    uv run mjpython scripts/record_demo.py
"""

import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from geodude import Geodude
from geodude.primitives import _find_arm_holding_object
from tsr import TSR
from tsr.core.tsr_primitive import load_template_file

TSR_DIR = Path(__file__).parent.parent / "tsr_templates"

# Bin positions
RIGHT_BIN_POS = [0.85, -0.35, 0.01]
LEFT_BIN_POS = [-0.85, -0.35, 0.01]


def sample_can_placement(robot):
    """Sample random can placement on worktop."""
    import random

    templates = [
        load_template_file(str(TSR_DIR / "places" / "can_on_table_upright.yaml")),
        load_template_file(str(TSR_DIR / "places" / "can_on_table_flipped.yaml")),
    ]
    template = random.choice(templates)

    worktop_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    worktop_pos = robot.data.site_xpos[worktop_id].copy()

    Bw = template.Bw.copy()
    Bw[0, :] = [-0.3, 0.3]
    Bw[1, :] = [-0.15, 0.15]

    tsr = TSR(Bw=Bw)
    xyzrpy = tsr.sample_xyzrpy()
    pos = worktop_pos + xyzrpy[:3]

    rot = TSR.rpy_to_rot(xyzrpy[3:6])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())

    return pos, quat


class FrameRecorder:
    """Records frames from MuJoCo simulation."""

    def __init__(self, model, data, width=640, height=480):
        self.model = model
        self.data = data
        self.renderer = mujoco.Renderer(model, height, width)
        self.frames = []

        # Camera settings (from CLAUDE.md preferred view)
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = -90
        self.camera.elevation = -26.5
        self.camera.distance = 2.96
        self.camera.lookat[:] = [0.188, 0.001, 1.141]

    def capture(self, n=1):
        """Capture n frames of current state."""
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera)
        frame = self.renderer.render()
        img = Image.fromarray(frame)
        for _ in range(n):
            self.frames.append(img.copy())

    def save_gif(self, path, fps=15):
        """Save frames as GIF."""
        if not self.frames:
            print("No frames to save!")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            duration=1000 // fps,
            loop=0,
            optimize=True,
        )
        print(f"Saved {len(self.frames)} frames to {path}", flush=True)
        print(f"File size: {path.stat().st_size / 1024:.1f} KB", flush=True)


def execute_with_recording(ctx, trajectory, recorder, frames_per_waypoint=2):
    """Execute trajectory while capturing frames."""
    if trajectory is None:
        return False

    # Sample waypoints for smooth animation
    n_waypoints = trajectory.num_waypoints
    step = max(1, n_waypoints // 30)  # ~30 frames per trajectory

    for i in range(0, n_waypoints, step):
        pos = trajectory.positions[i]

        # Set joint positions based on entity
        entity = trajectory.entity
        if entity and "arm" in entity:
            side = "left" if "left" in entity else "right"
            arm = ctx._robot.left_arm if side == "left" else ctx._robot.right_arm
            arm.set_joint_positions(pos)
        elif entity and "base" in entity:
            side = "left" if "left" in entity else "right"
            base = getattr(ctx._robot, f"{side}_base")
            if base:
                base.set_height(pos[0])

        recorder.capture(frames_per_waypoint)

    return True


def main():
    print("Recording recycling demo...", flush=True)

    output_path = Path(__file__).parent.parent / "docs" / "images" / "recycle_demo.gif"
    cycles = 3
    fps = 15

    # Create robot
    robot = Geodude(objects={"can": 1, "recycle_bin": 2})

    # Place can and bins
    can_pos, can_quat = sample_can_placement(robot)
    robot.env.registry.activate("can", pos=can_pos, quat=can_quat)
    robot.env.registry.activate("recycle_bin", pos=RIGHT_BIN_POS)
    robot.env.registry.activate("recycle_bin", pos=LEFT_BIN_POS)

    robot.go_to("ready")
    mujoco.mj_forward(robot.model, robot.data)

    # Create recorder
    recorder = FrameRecorder(robot.model, robot.data)

    # Capture initial frames
    recorder.capture(fps)  # 1 second at start

    with robot.sim(physics=False, viewer=False) as ctx:
        for cycle in range(1, cycles + 1):
            print(f"Cycle {cycle}/{cycles}...", flush=True)

            pickable = robot.get_pickable_objects()
            if not pickable:
                can_pos, can_quat = sample_can_placement(robot)
                robot.env.registry.activate("can", pos=can_pos, quat=can_quat)
                mujoco.mj_forward(robot.model, robot.data)
                recorder.capture(fps // 2)
                continue

            target = pickable[0]

            # === PICKUP ===
            print(f"  Picking up {target}...", flush=True)
            if not robot.pickup(target):
                print(f"  Pickup failed!", flush=True)
                break

            # Capture post-pickup state
            recorder.capture(fps // 2)

            # === PLACE ===
            holding_arm = _find_arm_holding_object(robot)
            bin_name = "recycle_bin_0" if holding_arm.side == "right" else "recycle_bin_1"

            print(f"  Placing in {bin_name}...", flush=True)
            if not robot.place(bin_name):
                print(f"  Place failed!", flush=True)
                break

            # Capture post-place state
            recorder.capture(fps // 2)

            # === RETURN TO READY ===
            robot.env.registry.hide(target)
            ready_config = np.array(robot.named_poses["ready"][holding_arm.side])
            ready_result = holding_arm.plan_to(ready_config)
            if ready_result is not None:
                execute_with_recording(ctx, ready_result, recorder)

            recorder.capture(fps // 4)

            # === SPAWN NEW CAN ===
            if cycle < cycles:
                can_pos, can_quat = sample_can_placement(robot)
                robot.env.registry.activate("can", pos=can_pos, quat=can_quat)
                mujoco.mj_forward(robot.model, robot.data)
                recorder.capture(fps // 2)

    # Final pause
    recorder.capture(fps)

    print(f"\nTotal frames: {len(recorder.frames)}", flush=True)
    recorder.save_gif(output_path, fps=fps)


if __name__ == "__main__":
    main()
