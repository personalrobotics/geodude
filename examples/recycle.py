#!/usr/bin/env python3
"""Recycling demo — bimanual pick and place with Geodude.

Demonstrates the student API::

    robot.pickup()                  # pick up anything
    robot.pickup("can")             # pick up any can
    robot.right.pickup("can")       # right arm picks any can
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


def spawn_objects(robot: Geodude, object_specs: list[tuple[str, int]]) -> None:
    """Simulate perception: scatter objects on the worktop.

    Places objects at random collision-free positions on the worktop surface
    using TSR sampling. In a real setup, object poses would come from a
    perception system via ``robot.env.registry.activate()``.

    Args:
        robot: Geodude instance (not yet in sim context).
        object_specs: List of (object_type, count), e.g.
            ``[("can", 3), ("potted_meat_can", 1)]``.
    """
    assets = AssetManager(str(OBJECTS_DIR))

    wt_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    wt_size = robot.model.site_size[wt_id]
    worktop_pos = robot.data.site_xpos[wt_id].copy()

    placer = TablePlacer(wt_size[0] - 0.05, wt_size[1] - 0.05)
    table_surface = np.eye(4)
    table_surface[:3, 3] = worktop_pos

    for obj_type, count in object_specs:
        gp = assets.get(obj_type)["geometric_properties"]

        if gp["type"] == "cylinder":
            templates = placer.place_cylinder(gp["radius"], gp["height"])
        elif gp["type"] == "box":
            templates = placer.place_box(gp["size"][0], gp["size"][1], gp["size"][2])
        else:
            continue

        for _ in range(count):
            tsr = templates[0].instantiate(table_surface)
            pos = tsr.sample()[:3, 3]
            name = robot.env.registry.activate(obj_type, pos=list(pos))
            mujoco.mj_forward(robot.model, robot.data)

            # Resample until collision-free
            body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jnt_id = robot.model.body_jntadr[body_id]
            qpos_adr = robot.model.jnt_qposadr[jnt_id]
            for _ in range(50):
                if not _has_object_collision(robot.model, robot.data, name):
                    break
                robot.data.qpos[qpos_adr : qpos_adr + 3] = tsr.sample()[:3, 3]
                mujoco.mj_forward(robot.model, robot.data)

    mujoco.mj_forward(robot.model, robot.data)


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
    robot.setup_scene(fixtures={"recycle_bin": [RIGHT_BIN_POS, LEFT_BIN_POS]})
    spawn_objects(robot, [("can", args.cans), ("potted_meat_can", 1)])

    with robot.sim(physics=args.physics, headless=args.headless) as ctx:
        for cycle in range(1, args.cycles + 1):
            if not ctx.is_running():
                break

            print(f"\n--- Cycle {cycle} ---")

            if not robot.pickup():
                print("  Nothing left to pick up")
                break

            robot.place("recycle_bin")
            robot.go_home()

        print(f"\nCompleted {cycle} cycles.")


if __name__ == "__main__":
    main()
