# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for the Robotiq 2F-140 gripper as used by geodude.

The ``RobotiqGripper`` class itself lives in ``mj_manipulator``, but
its scene XML lives in ``geodude_assets``. These tests previously sat
in ``mj_manipulator/tests/test_grippers.py`` and caused CI collection
errors there because ``geodude_assets`` is not a declared dependency
of ``mj_manipulator`` — see personalrobotics/mj_manipulator#87.

The right place for them is here in geodude, which declares both
``mj_manipulator`` and ``geodude_assets`` as dependencies.
"""

import mujoco
import pytest
from mj_manipulator.grasp_manager import GraspManager
from mj_manipulator.grippers.robotiq import RobotiqGripper
from mj_manipulator.protocols import Gripper

# geodude_assets is optional — skip the whole module cleanly if it's
# not importable (e.g. running the test file against a standalone
# geodude checkout without the asset package).
try:
    import geodude_assets
except ImportError:
    pytest.skip("geodude_assets not available", allow_module_level=True)


# Load the bare 2F-140 model directly via MuJoCo rather than wrapping
# it in mj_environment.Environment, because:
#
#  1. RobotiqGripper only needs a MjModel + MjData pair — the Environment
#     wrapper's asset-management features (object spawning, scene config,
#     placement) aren't exercised by these unit-level tests.
#
#  2. Environment(path) builds an in-memory XML string and an explicit
#     mesh-asset dict at construction time, which has subtle interactions
#     with the meshdir="assets" relative-path resolution in scene.xml
#     that broke under pip-install-from-git in CI.
#     mujoco.MjModel.from_xml_path() resolves meshdir natively against
#     the file's directory, with no string-buffering layer to confuse it.
#
# Use 2f140.xml (the gripper-only model) rather than scene.xml so there
# are no extra worldbody/lighting/floor things to worry about.
_ROBOTIQ_XML = geodude_assets.MODELS_DIR / "robotiq_2f140" / "2f140.xml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def robotiq_model_data():
    if not _ROBOTIQ_XML.exists():
        pytest.skip(f"Robotiq XML not found at {_ROBOTIQ_XML}")
    model = mujoco.MjModel.from_xml_path(str(_ROBOTIQ_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def robotiq_gripper(robotiq_model_data):
    model, data = robotiq_model_data
    return RobotiqGripper(model, data, "test_arm")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRobotiqGripperConstruction:
    def test_constructs(self, robotiq_gripper):
        assert robotiq_gripper is not None

    def test_satisfies_protocol(self, robotiq_gripper):
        assert isinstance(robotiq_gripper, Gripper)

    def test_arm_name(self, robotiq_gripper):
        assert robotiq_gripper.arm_name == "test_arm"

    def test_actuator_id(self, robotiq_gripper):
        assert robotiq_gripper.actuator_id is not None
        assert robotiq_gripper.actuator_id >= 0

    def test_ctrl_range(self, robotiq_gripper):
        assert robotiq_gripper.ctrl_open == 0.0
        assert robotiq_gripper.ctrl_closed == 255.0

    def test_body_names(self, robotiq_gripper):
        names = robotiq_gripper.gripper_body_names
        assert len(names) == 12
        assert "base_mount" in names
        assert "right_follower" in names
        assert "left_follower" in names

    def test_attachment_body(self, robotiq_gripper):
        assert robotiq_gripper.attachment_body == "base_mount"

    def test_initially_not_holding(self, robotiq_gripper):
        assert not robotiq_gripper.is_holding
        assert robotiq_gripper.held_object is None

    def test_invalid_actuator_raises(self, robotiq_model_data):
        model, data = robotiq_model_data
        with pytest.raises(ValueError, match="not found"):
            RobotiqGripper(
                model,
                data,
                "arm",
                prefix="nonexistent/",
            )


# ---------------------------------------------------------------------------
# Kinematic control
# ---------------------------------------------------------------------------


class TestRobotiqGripperKinematic:
    def test_kinematic_open(self, robotiq_gripper):
        robotiq_gripper.kinematic_open()
        assert robotiq_gripper.get_actual_position() == pytest.approx(0.0, abs=0.05)

    def test_kinematic_position_closed(self, robotiq_gripper):
        robotiq_gripper.set_kinematic_position(1.0)
        pos = robotiq_gripper.get_actual_position()
        assert pos > 0.9

    def test_kinematic_position_midpoint(self, robotiq_gripper):
        robotiq_gripper.set_kinematic_position(0.5)
        pos = robotiq_gripper.get_actual_position()
        assert 0.3 < pos < 0.7

    def test_kinematic_close_no_object(self, robotiq_gripper):
        result = robotiq_gripper.kinematic_close()
        # No candidate objects set, and the scene has an object but the
        # gripper isn't positioned near it — may or may not contact.
        # Just verify it doesn't crash and returns str or None.
        assert result is None or isinstance(result, str)

    def test_get_actual_position_range(self, robotiq_gripper):
        robotiq_gripper.set_kinematic_position(0.0)
        pos_open = robotiq_gripper.get_actual_position()
        assert 0.0 <= pos_open <= 0.1

        robotiq_gripper.set_kinematic_position(1.0)
        pos_closed = robotiq_gripper.get_actual_position()
        assert 0.9 <= pos_closed <= 1.0


# ---------------------------------------------------------------------------
# GraspManager integration
# ---------------------------------------------------------------------------


class TestRobotiqGripperGraspManager:
    def test_is_holding_with_grasp_manager(self, robotiq_model_data):
        model, data = robotiq_model_data
        gm = GraspManager(model, data)
        gripper = RobotiqGripper(
            model,
            data,
            "arm",
            grasp_manager=gm,
        )
        assert not gripper.is_holding

        gm.mark_grasped("box", "arm")
        assert gripper.is_holding
        assert gripper.held_object == "box"

        gm.mark_released("box")
        assert not gripper.is_holding
        assert gripper.held_object is None


# ---------------------------------------------------------------------------
# Construction with arm factory
# ---------------------------------------------------------------------------


class TestUR5eIntegration:
    def test_ur5e_with_gripper_constructs(self, robotiq_model_data):
        """RobotiqGripper satisfies the Gripper protocol when constructed
        against the 2F-140 model, ready to be attached to a UR5e arm."""
        model, data = robotiq_model_data
        gripper = RobotiqGripper(
            model,
            data,
            "ur5e",
        )
        assert isinstance(gripper, Gripper)
        assert gripper.arm_name == "ur5e"
