"""Pytest fixtures for geodude tests."""

import mujoco
import numpy as np
import pytest

# geodude_assets is optional - only needed for full integration tests
try:
    from geodude_assets import get_model_path
    GEODUDE_XML = get_model_path()
    GEODUDE_ASSETS_AVAILABLE = True
except ImportError:
    GEODUDE_XML = None
    GEODUDE_ASSETS_AVAILABLE = False


def _load_geodude_with_objects():
    """Load geodude model with test objects added for grasp testing."""
    if not GEODUDE_ASSETS_AVAILABLE:
        pytest.skip("geodude_assets not available")
    with open(GEODUDE_XML) as f:
        xml_content = f.read()

    # Insert meshdir to point to original asset location
    meshdir = str(GEODUDE_XML.parent) + "/"
    xml_content = xml_content.replace(
        '<compiler autolimits="true" angle="radian"/>',
        f'<compiler autolimits="true" angle="radian" meshdir="{meshdir}"/>',
    )

    # Insert test objects before </worldbody>
    objects_xml = """
    <!-- Test objects for grasp testing -->
    <body name="box1" pos="0.5 0 0.8">
      <freejoint name="box1_joint"/>
      <geom name="box1_geom" type="box" size="0.015 0.015 0.015" rgba="1 0.5 0 1" mass="0.1"
            contype="1" conaffinity="1"/>
    </body>

    <body name="box2" pos="0.5 0.1 0.8">
      <freejoint name="box2_joint"/>
      <geom name="box2_geom" type="box" size="0.015 0.015 0.015" rgba="0 0.5 1 1" mass="0.1"
            contype="1" conaffinity="1"/>
    </body>

    <body name="obstacle" pos="0.4 0 0.9">
      <geom name="obstacle_geom" type="box" size="0.02 0.02 0.02" rgba="1 0 0 1"
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>"""

    modified_xml = xml_content.replace("</worldbody>", objects_xml)
    return modified_xml


@pytest.fixture
def geodude_xml():
    """Return the modified geodude XML with test objects."""
    return _load_geodude_with_objects()


@pytest.fixture
def mujoco_model(geodude_xml):
    """Create MuJoCo model from geodude with test objects."""
    model = mujoco.MjModel.from_xml_string(geodude_xml)
    return model


@pytest.fixture
def mujoco_data(mujoco_model):
    """Create MuJoCo data for the geodude model."""
    return mujoco.MjData(mujoco_model)


@pytest.fixture
def mujoco_model_and_data(mujoco_model, mujoco_data):
    """Return both model and data as a tuple."""
    # Run forward to initialize positions
    mujoco.mj_forward(mujoco_model, mujoco_data)
    return mujoco_model, mujoco_data


@pytest.fixture
def arm_joint_names():
    """Joint names for the right arm."""
    return [
        "right_ur5e/shoulder_pan_joint",
        "right_ur5e/shoulder_lift_joint",
        "right_ur5e/elbow_joint",
        "right_ur5e/wrist_1_joint",
        "right_ur5e/wrist_2_joint",
        "right_ur5e/wrist_3_joint",
    ]


@pytest.fixture
def gripper_body_names():
    """Body names for the right gripper fingers."""
    return [
        "right_ur5e/gripper/right_follower",
        "right_ur5e/gripper/left_follower",
        "right_ur5e/gripper/right_pad",
        "right_ur5e/gripper/left_pad",
    ]


@pytest.fixture
def left_arm_joint_names():
    """Joint names for the left arm."""
    return [
        "left_ur5e/shoulder_pan_joint",
        "left_ur5e/shoulder_lift_joint",
        "left_ur5e/elbow_joint",
        "left_ur5e/wrist_1_joint",
        "left_ur5e/wrist_2_joint",
        "left_ur5e/wrist_3_joint",
    ]


@pytest.fixture
def right_arm_joint_names():
    """Joint names for the right arm (alias for arm_joint_names)."""
    return [
        "right_ur5e/shoulder_pan_joint",
        "right_ur5e/shoulder_lift_joint",
        "right_ur5e/elbow_joint",
        "right_ur5e/wrist_1_joint",
        "right_ur5e/wrist_2_joint",
        "right_ur5e/wrist_3_joint",
    ]
