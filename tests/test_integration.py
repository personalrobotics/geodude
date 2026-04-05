# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Integration tests for geodude with geodude_assets.

These tests verify the geodude package works correctly with the actual robot model.
"""

import mujoco
import numpy as np
import pytest
from geodude_assets import get_model_path

from geodude.config import GeodudConfig
from geodude.robot import Geodude

GEODUDE_XML = get_model_path()


class TestConfigMatchesModel:
    """Verify config matches actual model structure."""

    def test_joint_names_exist(self):
        """All configured joint names exist in model."""
        model = mujoco.MjModel.from_xml_path(str(GEODUDE_XML))
        config = GeodudConfig.default()

        for joint_name in config.joint_names(config.left_arm):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            assert jid != -1, f"Joint {joint_name} not found"

        for joint_name in config.joint_names(config.right_arm):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            assert jid != -1, f"Joint {joint_name} not found"

    def test_ee_sites_exist(self):
        """EE sites exist in model."""
        model = mujoco.MjModel.from_xml_path(str(GEODUDE_XML))
        config = GeodudConfig.default()

        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.left_arm.ee_site)
        assert sid != -1, f"Site {config.left_arm.ee_site} not found"

        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.right_arm.ee_site)
        assert sid != -1, f"Site {config.right_arm.ee_site} not found"


class TestCollisionGroupsWithRealModel:
    """Test collision group logic with real model."""

    @pytest.fixture
    def model_and_data(self):
        model = mujoco.MjModel.from_xml_path(str(GEODUDE_XML))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_gripper_pads_exist(self, model_and_data):
        """Gripper pad bodies exist."""
        model, data = model_and_data
        for body_name in [
            "right_ur5e/gripper/right_follower",
            "right_ur5e/gripper/left_follower",
        ]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            assert body_id != -1, f"Body {body_name} not found"


class TestGraspManagerWithRealModel:
    """Test GraspManager with real model."""

    @pytest.fixture
    def model_with_object(self):
        with open(GEODUDE_XML) as f:
            xml_content = f.read()

        meshdir = str(GEODUDE_XML.parent) + "/"
        xml_content = xml_content.replace(
            '<compiler autolimits="true" angle="radian"/>',
            f'<compiler autolimits="true" angle="radian" meshdir="{meshdir}"/>',
        )

        object_xml = """
    <body name="test_object" pos="0.5 0 1.0">
      <freejoint name="test_object_joint"/>
      <geom name="test_object_geom" type="box" size="0.02 0.02 0.02" mass="0.1"
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>"""

        modified_xml = xml_content.replace("</worldbody>", object_xml)
        model = mujoco.MjModel.from_xml_string(modified_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_grasp_manager_with_object(self, model_with_object):
        """GraspManager can mark objects as grasped."""
        from mj_manipulator import GraspManager

        model, data = model_with_object
        gm = GraspManager(model, data)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_object")
        assert body_id != -1

        gm.mark_grasped("test_object", "right")
        assert gm.is_grasped("test_object")

        gm.mark_released("test_object")
        assert not gm.is_grasped("test_object")


class TestGeodudeFull:
    """End-to-end tests with Geodude."""

    def test_geodude_init_and_arms(self):
        """Full Geodude initialization with arms and bases."""
        robot = Geodude()
        assert robot.left.dof == 6
        assert robot.right.dof == 6
        assert robot.left_base is not None
        assert robot.right_base is not None

    def test_sim_context(self):
        """SimContext can be created and entered."""
        robot = Geodude()
        with robot.sim(headless=True) as ctx:
            assert ctx.is_running()
            ctx.sync()

    def test_ee_poses_reachable(self):
        """EE poses are valid transforms in ready config."""
        robot = Geodude()
        for arm in (robot.left, robot.right):
            pose = arm.get_ee_pose()
            assert pose.shape == (4, 4)
            det = np.linalg.det(pose[:3, :3])
            assert np.isclose(det, 1.0, atol=1e-6)
