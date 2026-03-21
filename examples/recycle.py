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
# Object geometry from prl_assets
# ---------------------------------------------------------------------------

_ASSETS = AssetManager(str(OBJECTS_DIR))
_CAN = _ASSETS.get("can")["geometric_properties"]  # type=cylinder, radius, height
_BIN = _ASSETS.get("recycle_bin")

_BIN_GP = _BIN["geometric_properties"]  # type=open_box, outer_dimensions, wall_thickness
_BIN_POLICY = _BIN.get("policy", {}).get("placement", {})

_HAND = Robotiq2F140()

# Bin positions: floor-standing, one per side for bimanual reach
RIGHT_BIN_POS = np.array([0.85, -0.35, 0.01])
LEFT_BIN_POS = np.array([-0.85, -0.35, 0.01])


# ---------------------------------------------------------------------------
# TSR generation
# ---------------------------------------------------------------------------


def make_grasp_tsrs(can_pose: np.ndarray) -> list[TSR]:
    """Generate grasp TSRs for a can from its geometry.

    TSR convention: reference frame at the can's bottom centre, z up.
    can_pose is the MuJoCo body pose (geometric centre); shift down to
    bottom face before instantiating.
    """
    T_bottom = can_pose.copy()
    T_bottom[2, 3] -= _CAN["height"] / 2
    templates = _HAND.grasp_cylinder(_CAN["radius"], _CAN["height"])
    return [t.instantiate(T_bottom) for t in templates]


def make_drop_tsrs(bin_pos: np.ndarray) -> list[TSR]:
    """Generate drop-zone TSRs above the recycle bin opening.

    The TSR defines a region above the bin where the gripper can release:
    - XY: within the bin opening (with margin)
    - Z: fixed height above the rim (clearance for the held object)
    - Orientation: palm-down (z pointing down)
    """
    outer = _BIN_GP["outer_dimensions"]
    wall = _BIN_GP["wall_thickness"]
    margin = _BIN_POLICY.get("drop_zone_margin", 0.05)

    # Inner half-extents minus safety margin
    hx = (outer[0] / 2) - wall - margin
    hy = (outer[1] / 2) - wall - margin

    # Drop height: above the rim + clearance for can
    bin_height = outer[2]
    clearance = 0.10  # 10cm above rim
    drop_z = bin_pos[2] + bin_height + clearance

    # Palm-down orientation (gripper z points down)
    T0_w = np.array([
        [1,  0,  0, bin_pos[0]],
        [0, -1,  0, bin_pos[1]],
        [0,  0, -1, drop_z],
        [0,  0,  0, 1],
    ], dtype=float)

    # Bounds: allow XY sliding within opening, small Z range,
    # free rotation about vertical (yaw)
    Bw = np.zeros((6, 2))
    Bw[0, :] = [-hx, hx]  # x
    Bw[1, :] = [-hy, hy]  # y
    Bw[2, :] = [-0.02, 0.05]  # z (small range around drop height)
    Bw[5, :] = [-np.pi, np.pi]  # yaw (free rotation)

    return [TSR(T0_w=T0_w, Bw=Bw)]


def sample_can_position(worktop_pos: np.ndarray) -> np.ndarray:
    """Random position on the worktop for spawning a can."""
    x = worktop_pos[0] + random.uniform(-0.15, 0.15)
    y = worktop_pos[1] + random.uniform(-0.08, 0.08)
    z = worktop_pos[2] + _CAN["height"] / 2 + 0.005  # rest on surface
    return np.array([x, y, z])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Geodude recycling demo")
    parser.add_argument("--physics", action="store_true", help="Enable physics simulation")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--cycles", type=int, default=5, help="Number of pick-place cycles")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    mode = "Physics" if args.physics else "Kinematic"
    print(f"\n{'='*60}")
    print(f"  Geodude Recycling Demo — {mode} Mode")
    print(f"{'='*60}\n")

    # Create robot with objects from prl_assets
    robot = Geodude(objects={"can": 1, "recycle_bin": 2})

    # Get worktop position for can placement
    worktop_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    worktop_pos = robot.data.site_xpos[worktop_id].copy()

    # Place bins
    robot.env.registry.activate("recycle_bin", pos=list(RIGHT_BIN_POS))
    robot.env.registry.activate("recycle_bin", pos=list(LEFT_BIN_POS))

    # Set arms to ready pose before starting
    for side, arm in [("left", robot.left_arm), ("right", robot.right_arm)]:
        q = np.array(robot.named_poses["ready"][side])
        for i, idx in enumerate(arm.joint_qpos_indices):
            robot.data.qpos[idx] = q[i]
    mujoco.mj_forward(robot.model, robot.data)

    # Place first can
    can_pos = sample_can_position(worktop_pos)
    robot.env.registry.activate("can", pos=list(can_pos))
    mujoco.mj_forward(robot.model, robot.data)

    with robot.sim(physics=args.physics, headless=args.headless) as ctx:
        for cycle in range(1, args.cycles + 1):
            if not ctx.is_running():
                break

            print(f"\n--- Cycle {cycle} ---")

            # Get can pose
            can_pose = robot.get_object_pose("can_0")
            print(f"  Can at: {can_pose[:3, 3].round(3)}")

            # Generate grasp TSRs from can geometry
            grasp_tsrs = make_grasp_tsrs(can_pose)
            print(f"  Generated {len(grasp_tsrs)} grasp TSRs")

            # Pick up
            if not robot.pickup("can_0", grasp_tsrs):
                print("  Pickup FAILED")
                break
            print("  Picked up can")

            # Choose bin on same side as the arm that picked up
            for side in ("left", "right"):
                if robot.grasp_manager.get_grasped_by(side):
                    holding_side = side
                    break
            bin_pos = RIGHT_BIN_POS if holding_side == "right" else LEFT_BIN_POS

            # Generate drop zone TSRs from bin geometry
            drop_tsrs = make_drop_tsrs(bin_pos)

            # Place
            if not robot.place(drop_tsrs):
                print("  Place FAILED")
                break
            print(f"  Dropped can into {holding_side} bin")

            # Hide can (kinematic mode) or let it fall (physics mode)
            if not args.physics:
                robot.env.registry.hide("can_0")

            # Return holding arm to ready
            arm = robot.left_arm if holding_side == "left" else robot.right_arm
            _return_to_ready(robot, arm)
            ctx.sync()

            # Respawn can for next cycle
            if cycle < args.cycles:
                can_pos = sample_can_position(worktop_pos)
                robot.env.registry.activate("can", pos=list(can_pos))
                mujoco.mj_forward(robot.model, robot.data)
                print(f"  Spawned new can at {can_pos.round(3)}")

        print(f"\nCompleted {cycle} cycles.")


if __name__ == "__main__":
    main()
