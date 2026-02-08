#!/usr/bin/env python3
"""Visualize collision geometries and contacts in MuJoCo.

This script launches the viewer with collision geometry visualization enabled
to help debug collision checking issues.

Usage:
    uv run mjpython scripts/visualize_collisions.py

Keyboard controls in viewer:
    F1: Toggle help
    F2: Toggle info
    0-5: Toggle geom groups (0 is usually collision, 1-2 are visual)
    C: Toggle contact points
    F: Toggle contact forces
    T: Toggle transparent mode
    W: Toggle wireframe
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from geodude import Geodude


def main():
    print("Loading robot...", flush=True)
    robot = Geodude(objects={"can": 1})

    # Move to ready position
    robot.go_to("ready")

    print("\nLaunching viewer with collision visualization...", flush=True)
    print("\nViewer keyboard shortcuts:", flush=True)
    print("  0-5: Toggle geom groups (0=collision, 1-2=visual)", flush=True)
    print("  C: Toggle contact point visualization", flush=True)
    print("  F: Toggle contact force visualization", flush=True)
    print("  T: Toggle transparent mode", flush=True)
    print("  W: Toggle wireframe mode", flush=True)
    print("  Tab: Cycle through render modes", flush=True)
    print("\nTo see ONLY collision geometry:", flush=True)
    print("  Press 1, 2, 3 to hide visual groups", flush=True)
    print("  Press 0 to show collision group", flush=True)

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        # Set camera
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -26.5
        viewer.cam.distance = 2.96
        viewer.cam.lookat[:] = [0.188, 0.001, 1.141]

        # Enable contact visualization by default
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        # Show collision geom group (group 0) by default
        # geomgroup is a numpy array of 6 bools
        viewer.opt.geomgroup[0] = True  # Collision geoms

        print("\nGeom groups enabled:", flush=True)
        for i in range(6):
            status = "ON" if viewer.opt.geomgroup[i] else "OFF"
            print(f"  Group {i}: {status}", flush=True)

        # List geom info
        print("\n--- Gripper geom info ---", flush=True)
        for i in range(robot.model.ngeom):
            name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and ("gripper" in name.lower() or "pad" in name.lower() or "follower" in name.lower()):
                group = robot.model.geom_group[i]
                contype = robot.model.geom_contype[i]
                conaffinity = robot.model.geom_conaffinity[i]
                body_id = robot.model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                print(f"  {name}:", flush=True)
                print(f"    group={group}, contype={contype}, conaffinity={conaffinity}", flush=True)
                print(f"    body={body_name}", flush=True)

        print("\n--- Can geom info ---", flush=True)
        for i in range(robot.model.ngeom):
            name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "can" in name.lower():
                group = robot.model.geom_group[i]
                contype = robot.model.geom_contype[i]
                conaffinity = robot.model.geom_conaffinity[i]
                body_id = robot.model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                geom_type = robot.model.geom_type[i]
                geom_size = robot.model.geom_size[i]
                print(f"  {name}:", flush=True)
                print(f"    type={geom_type}, size={geom_size}", flush=True)
                print(f"    group={group}, contype={contype}, conaffinity={conaffinity}", flush=True)
                print(f"    body={body_name}", flush=True)

        print("\nViewer running. Press Ctrl+C or close window to exit.", flush=True)

        while viewer.is_running():
            mujoco.mj_step(robot.model, robot.data)
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()
