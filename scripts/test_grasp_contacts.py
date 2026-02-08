#!/usr/bin/env python3
"""Test grasp contact detection."""

import time
import numpy as np
import mujoco
import mujoco.viewer

from geodude import Geodude


def main():
    print("Loading robot with can...", flush=True)
    robot = Geodude(objects={"can": 1})

    # Place can at center of worktop
    can_pos = [0.2, -0.35, 0.816]
    robot.env.registry.activate("can", pos=can_pos)

    robot.go_to("ready")
    mujoco.mj_forward(robot.model, robot.data)

    # Check can collision settings
    can_body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "can_0")
    geom_adr = robot.model.body_geomadr[can_body_id]
    geom_num = robot.model.body_geomnum[can_body_id]
    print(f"\nCan collision settings:", flush=True)
    for i in range(geom_num):
        geom_id = geom_adr + i
        name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        print(f"  {name}: contype={robot.model.geom_contype[geom_id]}, conaffinity={robot.model.geom_conaffinity[geom_id]}")

    # Check gripper collision settings
    print(f"\nGripper collision settings:", flush=True)
    for body_name in ["right_ur5e/gripper/right_follower", "right_ur5e/gripper/left_follower"]:
        body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        geom_adr = robot.model.body_geomadr[body_id]
        geom_num = robot.model.body_geomnum[body_id]
        for i in range(geom_num):
            geom_id = geom_adr + i
            name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name:
                print(f"  {name}: contype={robot.model.geom_contype[geom_id]}, conaffinity={robot.model.geom_conaffinity[geom_id]}")

    print("\nLaunching viewer - move arm manually to test contacts", flush=True)
    print("Press 'c' to close gripper, 'o' to open", flush=True)

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -26.5
        viewer.cam.distance = 2.96
        viewer.cam.lookat[:] = [0.188, 0.001, 1.141]

        # Enable contact visualization
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        gripper_ctrl_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_ur5e/gripper/fingers_actuator")

        last_print = 0
        while viewer.is_running():
            mujoco.mj_step(robot.model, robot.data)

            # Print contact info periodically
            if time.time() - last_print > 1.0:
                last_print = time.time()
                gripper_pos = robot.right_arm.gripper.get_actual_position()
                print(f"\nContacts: {robot.data.ncon}, Gripper pos: {gripper_pos:.3f}", flush=True)

                # Check for can-gripper contacts
                for i in range(robot.data.ncon):
                    contact = robot.data.contact[i]
                    body1 = robot.model.geom_bodyid[contact.geom1]
                    body2 = robot.model.geom_bodyid[contact.geom2]
                    name1 = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, body1)
                    name2 = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, body2)
                    if name1 and name2 and ("can" in name1 or "can" in name2):
                        print(f"  Can contact: {name1} <-> {name2}", flush=True)

            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()
