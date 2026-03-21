#!/usr/bin/env python3
"""Recycling demo — bimanual pick and place with Geodude.

Demonstrates the full stack:
  prl_assets      — can + recycle bin models with geometry metadata
  asset_manager   — reads geometric_properties for TSR generation
  tsr.hands       — Robotiq2F140 grasp templates from cylinder geometry
  mj_manipulator  — Arm, SimContext, CartesianController, GraspManager
  geodude         — bimanual orchestration, VentionBase, primitives

The robot picks up soda cans from the worktop and drops them into
floor-standing recycle bins. Grasp TSRs are generated programmatically
from the can's physical dimensions; the drop zone TSR is constructed
from the bin's opening geometry.

Usage:
    uv run mjpython examples/recycle.py
    uv run mjpython examples/recycle.py --physics
    uv run mjpython examples/recycle.py --cycles 3 --headless
"""

from __future__ import annotations

import argparse
import logging
import random

import mujoco
import numpy as np
from asset_manager import AssetManager
from prl_assets import OBJECTS_DIR
from tsr import TSR
from tsr.hands import Robotiq2F140

from geodude import Geodude
from geodude.primitives import _return_to_ready

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logging.getLogger("toppra").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Object geometry (read once from prl_assets metadata)
# ---------------------------------------------------------------------------

_ASSETS = AssetManager(str(OBJECTS_DIR))
_CAN_GP = _ASSETS.get("can")["geometric_properties"]
_BIN_META = _ASSETS.get("recycle_bin")
_BIN_GP = _BIN_META["geometric_properties"]
_BIN_POLICY = _BIN_META.get("policy", {}).get("placement", {})
_HAND = Robotiq2F140()

# Bin positions: floor-standing, one per side
RIGHT_BIN_POS = np.array([0.85, -0.35, 0.01])
LEFT_BIN_POS = np.array([-0.85, -0.35, 0.01])


# ---------------------------------------------------------------------------
# TSR generation
# ---------------------------------------------------------------------------


def make_grasp_tsrs(can_pose: np.ndarray) -> list[TSR]:
    """Side-grasp TSRs for a soda can, generated from prl_assets geometry.

    TSR convention: reference at the can bottom centre, z up.
    can_pose is the MuJoCo body pose (geometric centre); we shift to
    the bottom face before calling instantiate().
    """
    T_bottom = can_pose.copy()
    T_bottom[2, 3] -= _CAN_GP["height"] / 2
    templates = _HAND.grasp_cylinder_side(_CAN_GP["radius"], _CAN_GP["height"])
    return [t.instantiate(T_bottom) for t in templates]


def make_drop_tsrs(bin_pos: np.ndarray) -> list[TSR]:
    """Drop-zone TSR above the recycle bin opening.

    Defines a region where the gripper can release:
    - XY within the bin opening (with safety margin)
    - Z above the rim (clearance for held object)
    - Palm-down, free yaw rotation
    """
    outer = _BIN_GP["outer_dimensions"]
    wall = _BIN_GP["wall_thickness"]
    margin = _BIN_POLICY.get("drop_zone_margin", 0.05)

    hx = (outer[0] / 2) - wall - margin
    hy = (outer[1] / 2) - wall - margin
    drop_z = bin_pos[2] + outer[2] + 0.15  # 15cm above rim

    T0_w = np.array([
        [1,  0,  0, bin_pos[0]],
        [0, -1,  0, bin_pos[1]],
        [0,  0, -1, drop_z],
        [0,  0,  0, 1],
    ], dtype=float)

    Bw = np.zeros((6, 2))
    Bw[0, :] = [-hx, hx]
    Bw[1, :] = [-hy, hy]
    Bw[2, :] = [-0.02, 0.05]
    Bw[5, :] = [-np.pi, np.pi]

    return [TSR(T0_w=T0_w, Bw=Bw)]


def sample_can_position(worktop_pos: np.ndarray) -> np.ndarray:
    """Random position on the worktop surface."""
    return np.array([
        worktop_pos[0] + random.uniform(-0.15, 0.15),
        worktop_pos[1] + random.uniform(-0.08, 0.08),
        worktop_pos[2] + _CAN_GP["height"] / 2 + 0.005,
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Geodude recycling demo")
    parser.add_argument("--physics", action="store_true", help="Physics simulation")
    parser.add_argument("--headless", action="store_true", help="No viewer")
    parser.add_argument("--cycles", type=int, default=5, help="Pick-place cycles")
    parser.add_argument("--debug", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    mode = "Physics" if args.physics else "Kinematic"
    print(f"\n{'='*60}")
    print(f"  Geodude Recycling Demo — {mode} Mode")
    print(f"{'='*60}\n")

    robot = Geodude(objects={"can": 1, "recycle_bin": 2})

    # Worktop position for can spawning
    worktop_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    worktop_pos = robot.data.site_xpos[worktop_id].copy()

    # Place bins and set arms to ready
    robot.env.registry.activate("recycle_bin", pos=list(RIGHT_BIN_POS))
    robot.env.registry.activate("recycle_bin", pos=list(LEFT_BIN_POS))
    for side, arm in [("left", robot.left_arm), ("right", robot.right_arm)]:
        q = np.array(robot.named_poses["ready"][side])
        for i, idx in enumerate(arm.joint_qpos_indices):
            robot.data.qpos[idx] = q[i]

    # Spawn first can
    can_pos = sample_can_position(worktop_pos)
    robot.env.registry.activate("can", pos=list(can_pos))
    mujoco.mj_forward(robot.model, robot.data)

    with robot.sim(physics=args.physics, headless=args.headless) as ctx:
        for cycle in range(1, args.cycles + 1):
            if not ctx.is_running():
                break

            print(f"\n--- Cycle {cycle} ---")
            can_pose = robot.get_object_pose("can_0")
            print(f"  Can at {can_pose[:3, 3].round(3)}")

            # --- Pick up ---
            grasp_tsrs = make_grasp_tsrs(can_pose)
            if not robot.pickup("can_0", grasp_tsrs):
                print("  Pickup FAILED")
                break

            # Determine which arm picked up
            holding_side = None
            for side in ("left", "right"):
                if robot.grasp_manager.get_grasped_by(side):
                    holding_side = side
                    break
            print(f"  Picked up with {holding_side} arm")

            # --- Place in bin on same side ---
            bin_pos = RIGHT_BIN_POS if holding_side == "right" else LEFT_BIN_POS
            drop_tsrs = make_drop_tsrs(bin_pos)
            if not robot.place(drop_tsrs):
                print("  Place FAILED")
                break

            # Hide can after release so the pool is free for next cycle
            robot.env.registry.hide("can_0")
            mujoco.mj_forward(robot.model, robot.data)
            ctx.sync()
            print(f"  Dropped into {holding_side} bin")

            arm = robot.left_arm if holding_side == "left" else robot.right_arm
            _return_to_ready(robot, arm)
            ctx.sync()

            # Spawn next can
            if cycle < args.cycles:
                can_pos = sample_can_position(worktop_pos)
                robot.env.registry.activate("can", pos=list(can_pos))
                mujoco.mj_forward(robot.model, robot.data)

        print(f"\nCompleted {cycle} cycles.")


if __name__ == "__main__":
    main()
