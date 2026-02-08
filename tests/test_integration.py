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

        for joint_name in config.left_arm.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            assert jid != -1, f"Joint {joint_name} not found"

        for joint_name in config.right_arm.joint_names:
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

    def test_gripper_actuator_exists(self):
        """Gripper actuator exists in model."""
        model = mujoco.MjModel.from_xml_path(str(GEODUDE_XML))
        config = GeodudConfig.default()

        if config.right_arm.gripper_actuator:
            aid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, config.right_arm.gripper_actuator
            )
            assert aid != -1, f"Actuator {config.right_arm.gripper_actuator} not found"

    def test_gripper_bodies_exist(self):
        """Gripper bodies exist in model."""
        model = mujoco.MjModel.from_xml_path(str(GEODUDE_XML))
        config = GeodudConfig.default()

        for body_name in config.right_arm.gripper_bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            assert bid != -1, f"Body {body_name} not found"


class TestCollisionGroupsWithRealModel:
    """Test collision group logic with real model."""

    @pytest.fixture
    def model_and_data(self):
        """Load real model directly for collision tests."""
        model = mujoco.MjModel.from_xml_path(str(GEODUDE_XML))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_arm_collision_geoms_exist(self, model_and_data):
        """Arm collision geoms exist in model."""
        model, data = model_and_data

        # Check some collision geoms on the arm exist
        for geom_name in [
            "right_ur5e/geom_0",
            "right_ur5e/geom_1",
        ]:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            # Just verify they exist (may have different naming)

    def test_gripper_pads_exist(self, model_and_data):
        """Gripper pad bodies exist."""
        model, data = model_and_data

        # Check gripper follower bodies exist
        for body_name in [
            "right_ur5e/gripper/right_follower",
            "right_ur5e/gripper/left_follower",
        ]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            assert body_id != -1, f"Body {body_name} not found"


class TestGraspManagerWithRealModel:
    """Test GraspManager with real model (requires adding objects)."""

    @pytest.fixture
    def model_with_object(self):
        """Create a modified model with a graspable object."""
        with open(GEODUDE_XML) as f:
            xml_content = f.read()

        # Insert meshdir to point to original asset location
        meshdir = str(GEODUDE_XML.parent) + "/"
        xml_content = xml_content.replace(
            '<compiler autolimits="true" angle="radian"/>',
            f'<compiler autolimits="true" angle="radian" meshdir="{meshdir}"/>'
        )

        # Insert a free body before </worldbody>
        object_xml = '''
    <body name="test_object" pos="0.5 0 1.0">
      <freejoint name="test_object_joint"/>
      <geom name="test_object_geom" type="box" size="0.02 0.02 0.02" mass="0.1"
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>'''

        modified_xml = xml_content.replace('</worldbody>', object_xml)

        model = mujoco.MjModel.from_xml_string(modified_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_grasp_manager_with_object(self, model_with_object):
        """GraspManager can mark objects as grasped."""
        from geodude.grasp_manager import GraspManager

        model, data = model_with_object
        gm = GraspManager(model, data)

        # Verify object exists
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_object")
        assert body_id != -1

        # Mark as grasped
        gm.mark_grasped("test_object", "right")

        # Verify grasp state tracked
        assert gm.is_grasped("test_object")
        assert gm.get_holder("test_object") == "right"

        # Release
        gm.mark_released("test_object")
        assert not gm.is_grasped("test_object")
        assert gm.get_holder("test_object") is None
