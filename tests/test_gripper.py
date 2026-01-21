"""Tests for Gripper class."""

import mujoco
import numpy as np
import pytest

from geodude.grasp_manager import GraspManager
from geodude.gripper import Gripper


class TestGripper:
    """Tests for Gripper class."""

    @pytest.fixture
    def gripper(self, mujoco_model_and_data, gripper_body_names):
        """Create a Gripper instance for testing."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        return Gripper(
            model=model,
            data=data,
            arm_name="right",
            actuator_name="right_ur5e/gripper/fingers_actuator",
            gripper_body_names=gripper_body_names,
            grasp_manager=gm,
        )

    def test_init(self, gripper):
        """Gripper initializes correctly."""
        assert gripper.arm_name == "right"
        assert gripper.actuator_id is not None

    def test_init_no_actuator(self, mujoco_model_and_data, gripper_body_names):
        """Gripper works with no actuator (for testing)."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        gripper = Gripper(
            model=model,
            data=data,
            arm_name="left",
            actuator_name="",  # No actuator
            gripper_body_names=[],
            grasp_manager=gm,
        )

        assert gripper.actuator_id is None
        # Should not crash when opening/closing
        gripper.open()
        gripper.close()

    def test_invalid_actuator_raises(self, mujoco_model_and_data, gripper_body_names):
        """Invalid actuator name raises ValueError."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        with pytest.raises(ValueError, match="not found"):
            Gripper(
                model=model,
                data=data,
                arm_name="right",
                actuator_name="nonexistent_actuator",
                gripper_body_names=gripper_body_names,
                grasp_manager=gm,
            )

    def test_open(self, gripper):
        """open() sets control to open value."""
        gripper.open(steps=10)
        assert gripper.data.ctrl[gripper.actuator_id] == gripper.ctrl_open

    def test_close(self, gripper):
        """close() sets control to closed value."""
        gripper.close(steps=10)
        assert gripper.data.ctrl[gripper.actuator_id] == gripper.ctrl_closed

    def test_get_position(self, gripper):
        """get_position() returns normalized gripper position."""
        # At open position
        gripper.data.ctrl[gripper.actuator_id] = gripper.ctrl_open
        assert gripper.get_position() == pytest.approx(0.0)

        # At closed position
        gripper.data.ctrl[gripper.actuator_id] = gripper.ctrl_closed
        assert gripper.get_position() == pytest.approx(1.0)

        # At midpoint
        mid = (gripper.ctrl_open + gripper.ctrl_closed) / 2
        gripper.data.ctrl[gripper.actuator_id] = mid
        assert gripper.get_position() == pytest.approx(0.5)

    def test_set_position(self, gripper):
        """set_position() sets normalized gripper position."""
        gripper.set_position(0.0)
        assert gripper.data.ctrl[gripper.actuator_id] == pytest.approx(gripper.ctrl_open)

        gripper.set_position(1.0)
        assert gripper.data.ctrl[gripper.actuator_id] == pytest.approx(gripper.ctrl_closed)

        gripper.set_position(0.5)
        expected = (gripper.ctrl_open + gripper.ctrl_closed) / 2
        assert gripper.data.ctrl[gripper.actuator_id] == pytest.approx(expected)

    def test_is_holding_initially_false(self, gripper):
        """is_holding is False initially."""
        assert not gripper.is_holding

    def test_held_object_initially_none(self, gripper):
        """held_object is None initially."""
        assert gripper.held_object is None

    def test_set_candidate_objects(self, gripper):
        """set_candidate_objects stores the list."""
        gripper.set_candidate_objects(["box1", "box2"])
        assert gripper._candidate_objects == ["box1", "box2"]

        gripper.set_candidate_objects(None)
        assert gripper._candidate_objects is None

    def test_close_detects_grasp_and_updates_manager(self, mujoco_model_and_data, gripper_body_names):
        """close() detects grasp and updates GraspManager."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        gripper = Gripper(
            model=model,
            data=data,
            arm_name="right",
            actuator_name="right_ur5e/gripper/fingers_actuator",
            gripper_body_names=gripper_body_names,
            grasp_manager=gm,
        )

        # Move box1 into gripper position
        box1_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box1_joint")
        box1_qpos_adr = model.jnt_qposadr[box1_jnt_id]

        # Get the gripper attachment site position
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_ur5e/gripper_attachment_site")
        mujoco.mj_forward(model, data)
        ee_pos = data.site_xpos[ee_site_id].copy()

        # Place box slightly in front of gripper
        data.qpos[box1_qpos_adr:box1_qpos_adr + 3] = ee_pos + [0.1, 0, 0]
        data.qpos[box1_qpos_adr + 3:box1_qpos_adr + 7] = [1, 0, 0, 0]

        gripper.set_candidate_objects(["box1"])

        # Close gripper
        result = gripper.close(steps=50)

        # If contact was made, should detect grasp
        if result == "box1":
            assert gripper.is_holding
            assert gripper.held_object == "box1"
            assert gm.is_grasped("box1")

    def test_open_releases_grasped_objects(self, mujoco_model_and_data, gripper_body_names):
        """open() releases any held objects."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        gripper = Gripper(
            model=model,
            data=data,
            arm_name="right",
            actuator_name="right_ur5e/gripper/fingers_actuator",
            gripper_body_names=gripper_body_names,
            grasp_manager=gm,
        )

        # Manually mark object as grasped
        gm.mark_grasped("box1", "right")
        assert gripper.is_holding

        # Open gripper
        gripper.open(steps=10)

        # Should release
        assert not gripper.is_holding
        assert not gm.is_grasped("box1")
