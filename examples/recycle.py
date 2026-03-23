#!/usr/bin/env python3
"""Recycling demo — bimanual pick and place with Geodude.

Demonstrates the full student API::

    robot.pickup()                  # pick up anything
    robot.pickup("can")             # pick up any can
    robot.right.pickup("can")      # right arm picks any can
    robot.place("recycle_bin")      # place in any bin
    robot.go_home()

Usage:
    uv run mjpython examples/recycle.py
    uv run mjpython examples/recycle.py --physics
    uv run mjpython examples/recycle.py --headless --cycles 10
"""

from __future__ import annotations

import argparse
import logging

import mujoco
import numpy as np
from asset_manager import AssetManager
from prl_assets import OBJECTS_DIR
from tsr.placement import TablePlacer

from geodude import Geodude

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logging.getLogger("toppra").setLevel(logging.WARNING)

# Bin positions: floor-standing, one per side
RIGHT_BIN_POS = [0.85, -0.35, 0.01]
LEFT_BIN_POS = [-0.85, -0.35, 0.01]

# Object geometry from prl_assets
_ASSETS = AssetManager(str(OBJECTS_DIR))


def _has_object_collision(model, data, body_name: str) -> bool:
    """Check if a body is in contact with any other object (not floor)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return False
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if (b1 == body_id or b2 == body_id) and c.dist < 0:
            other = b2 if b1 == body_id else b1
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other)
            if other_name and other_name != "world":
                return True
    return False


def _spawn_objects(robot, worktop_pos, object_specs: list[tuple[str, int]]):
    """Spawn objects on the worktop, resampling if in collision.

    Uses MuJoCo's collision checker — place object, mj_forward, check
    contacts, resample if colliding.

    Args:
        robot: Geodude instance.
        worktop_pos: [x, y, z] of worktop surface center.
        object_specs: list of (object_type, count) e.g. [("can", 3), ("potted_meat_can", 1)]
    """
    table_hx, table_hy = 0.15, 0.08
    placer = TablePlacer(table_hx, table_hy)

    table_surface = np.eye(4)
    table_surface[:3, 3] = worktop_pos

    for obj_type, count in object_specs:
        gp = _ASSETS.get(obj_type)["geometric_properties"]

        if gp["type"] == "cylinder":
            templates = placer.place_cylinder(gp["radius"], gp["height"])
        elif gp["type"] == "box":
            templates = placer.place_box(gp["size"][0], gp["size"][1], gp["size"][2])
        else:
            continue

        for _ in range(count):
            tsr = templates[0].instantiate(table_surface)
            # Activate at a temporary position, then resample until collision-free
            pos = tsr.sample()[:3, 3]
            instance_name = robot.env.registry.activate(obj_type, pos=list(pos))
            mujoco.mj_forward(robot.model, robot.data)

            for _attempt in range(50):
                if not _has_object_collision(robot.model, robot.data, instance_name):
                    break
                # Resample position via freejoint qpos
                pos = tsr.sample()[:3, 3]
                body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, instance_name)
                jnt_id = robot.model.body_jntadr[body_id]
                qpos_adr = robot.model.jnt_qposadr[jnt_id]
                robot.data.qpos[qpos_adr:qpos_adr + 3] = pos
                mujoco.mj_forward(robot.model, robot.data)


def main():
    parser = argparse.ArgumentParser(description="Geodude recycling demo")
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--cans", type=int, default=3)
    args = parser.parse_args()

    mode = "Physics" if args.physics else "Kinematic"
    print(f"\n{'='*60}")
    print(f"  Geodude Recycling Demo — {mode} Mode")
    print(f"  {args.cans} cans + 1 potted meat can, up to {args.cycles} cycles")
    print(f"{'='*60}\n")

    robot = Geodude(objects={"can": args.cans, "potted_meat_can": 1, "recycle_bin": 2})

    # Worktop position for spawning
    worktop_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    worktop_pos = robot.data.site_xpos[worktop_id].copy()

    # Place bins, raise bases to midpoint, set arms to ready
    robot.env.registry.activate("recycle_bin", pos=RIGHT_BIN_POS)
    robot.env.registry.activate("recycle_bin", pos=LEFT_BIN_POS)
    for base in [robot.left_base, robot.right_base]:
        if base is not None:
            base.set_height(0.25)  # midpoint of 0–0.5m range
    for side, arm in [("left", robot.left_arm), ("right", robot.right_arm)]:
        q = np.array(robot.named_poses["ready"][side])
        for i, idx in enumerate(arm.joint_qpos_indices):
            robot.data.qpos[idx] = q[i]

    # Spawn objects on worktop (non-overlapping, TSR-sampled)
    _spawn_objects(robot, worktop_pos, [
        ("can", args.cans),
        ("potted_meat_can", 1),
    ])
    mujoco.mj_forward(robot.model, robot.data)

    with robot.sim(physics=args.physics, headless=args.headless) as ctx:
        for cycle in range(1, args.cycles + 1):
            if not ctx.is_running():
                break

            print(f"\n--- Cycle {cycle} ---")

            # ====================================
            # THE STUDENT API
            # ====================================

            # Pick up anything on the table
            if not robot.pickup():
                print("  Nothing left to pick up")
                break

            # Which arm grabbed what?
            for side in ("left", "right"):
                held = list(robot.grasp_manager.get_grasped_by(side))
                if held:
                    holding_side = side
                    held_object = held[0]
                    break

            print(f"  {holding_side} arm picked up {held_object}")

            # Place in same-side bin
            bin_type = "recycle_bin_0" if holding_side == "right" else "recycle_bin_1"
            if not robot.place(bin_type):
                print("  Place FAILED")
                # In kinematic mode, the released object is floating — hide it
                if not args.physics:
                    robot.env.registry.hide(held_object)
                    mujoco.mj_forward(robot.model, robot.data)
                robot.go_home()
                continue

            # Hide object and go home
            robot.env.registry.hide(held_object)
            mujoco.mj_forward(robot.model, robot.data)
            ctx.sync()
            print(f"  Placed in {holding_side} bin")

            robot.go_home()

        print(f"\nCompleted {cycle} cycles.")


if __name__ == "__main__":
    main()
