"""Tests for kinematic manipulation (grasping without physics)."""

import mujoco
import numpy as np
import pytest

from geodude.executor import KinematicExecutor
from geodude.grasp_manager import GraspManager
from geodude.gripper import Gripper
from geodude.trajectory import Trajectory


class TestGraspManagerAttachment:
    """Tests for kinematic object attachment in GraspManager."""

    def test_attach_object(self, mujoco_model_and_data, gripper_body_names):
        """Attaching an object stores relative transform."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        gripper_body = gripper_body_names[0]
        gm.attach_object("box1", gripper_body)

        assert gm.is_attached("box1")
        assert "box1" in gm.get_attached_objects()

    def test_detach_object(self, mujoco_model_and_data, gripper_body_names):
        """Detaching an object removes attachment."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        gripper_body = gripper_body_names[0]
        gm.attach_object("box1", gripper_body)
        gm.detach_object("box1")

        assert not gm.is_attached("box1")
        assert "box1" not in gm.get_attached_objects()

    def test_detach_nonexistent_is_noop(self, mujoco_model_and_data):
        """Detaching non-attached object is a no-op."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        gm.detach_object("box1")  # Should not raise
        assert not gm.is_attached("box1")

    def test_update_attached_poses(self, mujoco_model_and_data, gripper_body_names):
        """Attached objects move when update_attached_poses is called."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        gripper_body = gripper_body_names[0]

        # Get initial positions
        box1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box1")
        gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gripper_body)

        initial_box_pos = data.xpos[box1_id].copy()

        # Attach object
        gm.attach_object("box1", gripper_body)

        # Move the gripper (simulated by moving arm joint)
        # Get the right arm shoulder pan joint
        jnt_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "right_ur5e/shoulder_pan_joint"
        )
        qpos_adr = model.jnt_qposadr[jnt_id]
        data.qpos[qpos_adr] = 0.5  # Rotate shoulder

        mujoco.mj_forward(model, data)

        # Gripper should have moved
        new_gripper_pos = data.xpos[gripper_id].copy()

        # Update attached poses
        gm.update_attached_poses()
        mujoco.mj_forward(model, data)

        # Box should have moved
        new_box_pos = data.xpos[box1_id].copy()
        assert not np.allclose(initial_box_pos, new_box_pos, atol=0.01)


class TestGripperKinematic:
    """Tests for kinematic gripper operations."""

    @pytest.fixture
    def gripper(self, mujoco_model_and_data, gripper_body_names):
        """Create a Gripper for the right arm."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        gripper = Gripper(
            model=model,
            data=data,
            arm_name="right",
            actuator_name="right_ur5e/gripper/fingers_actuator",
            gripper_body_names=gripper_body_names,
            grasp_manager=gm,
            gripper_site_name="right_ur5e/gripper_attachment_site",
        )
        return gripper

    def test_kinematic_close_no_objects(self, gripper):
        """Kinematic close with no candidate objects returns None."""
        gripper.set_candidate_objects([])
        result = gripper.kinematic_close()
        assert result is None

    def test_kinematic_close_object_not_in_range(self, gripper, mujoco_model_and_data):
        """Kinematic close returns None when objects are far away."""
        model, data = mujoco_model_and_data

        # Objects are at their initial positions (far from gripper)
        gripper.set_candidate_objects(["box1", "box2"])
        result = gripper.kinematic_close()

        # box1 is at [0.5, 0, 0.8], gripper is at a different position
        # Should not detect grasp
        assert result is None

    def test_kinematic_close_detects_nearby_object(
        self, gripper, mujoco_model_and_data
    ):
        """Kinematic close detects object within threshold."""
        model, data = mujoco_model_and_data

        # Move box1 to gripper position
        box1_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box1_joint")
        box1_qpos_adr = model.jnt_qposadr[box1_jnt_id]

        # Get gripper site position
        ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_ur5e/gripper_attachment_site"
        )
        mujoco.mj_forward(model, data)
        ee_pos = data.site_xpos[ee_site_id].copy()

        # Move box to gripper position (within threshold)
        data.qpos[box1_qpos_adr : box1_qpos_adr + 3] = ee_pos
        data.qpos[box1_qpos_adr + 3 : box1_qpos_adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        gripper.set_candidate_objects(["box1", "box2"])
        result = gripper.kinematic_close()

        assert result == "box1"
        assert gripper.grasp_manager.is_grasped("box1")
        assert gripper.grasp_manager.is_attached("box1")

    def test_kinematic_open_releases_object(self, gripper, mujoco_model_and_data):
        """Kinematic open releases held object."""
        model, data = mujoco_model_and_data

        # Move box1 to gripper and grasp
        box1_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box1_joint")
        box1_qpos_adr = model.jnt_qposadr[box1_jnt_id]
        ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_ur5e/gripper_attachment_site"
        )
        mujoco.mj_forward(model, data)
        ee_pos = data.site_xpos[ee_site_id].copy()

        data.qpos[box1_qpos_adr : box1_qpos_adr + 3] = ee_pos
        data.qpos[box1_qpos_adr + 3 : box1_qpos_adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        gripper.set_candidate_objects(["box1"])
        gripper.kinematic_close()

        assert gripper.is_holding

        # Now open
        gripper.kinematic_open()

        assert not gripper.is_holding
        assert not gripper.grasp_manager.is_grasped("box1")
        assert not gripper.grasp_manager.is_attached("box1")


class TestKinematicExecutorWithAttachment:
    """Tests for KinematicExecutor updating attached objects."""

    @pytest.fixture
    def executor_with_grasp(self, mujoco_model_and_data, arm_joint_names):
        """Create a KinematicExecutor with GraspManager."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        joint_qpos_indices = []
        for name in arm_joint_names:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            joint_qpos_indices.append(model.jnt_qposadr[jnt_id])

        executor = KinematicExecutor(
            model=model,
            data=data,
            joint_qpos_indices=joint_qpos_indices,
            control_dt=0.001,  # Fast for testing
            grasp_manager=gm,
        )
        return executor, gm

    def test_set_position_updates_attached_objects(
        self, executor_with_grasp, mujoco_model_and_data, gripper_body_names
    ):
        """set_position updates attached object poses."""
        model, data = mujoco_model_and_data
        executor, gm = executor_with_grasp

        gripper_body = gripper_body_names[0]

        # Get initial box position
        box1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box1")
        initial_pos = data.xpos[box1_id].copy()

        # Attach box to gripper
        gm.attach_object("box1", gripper_body)

        # Move arm via executor
        q_current = np.array(
            [data.qpos[idx] for idx in executor.joint_qpos_indices]
        )
        q_new = q_current.copy()
        q_new[0] = 0.5  # Rotate shoulder

        executor.set_position(q_new)

        # Box should have moved
        new_pos = data.xpos[box1_id].copy()
        assert not np.allclose(initial_pos, new_pos, atol=0.01)

    def test_execute_trajectory_updates_attached_objects(
        self, executor_with_grasp, mujoco_model_and_data, gripper_body_names
    ):
        """execute() updates attached object poses along trajectory."""
        model, data = mujoco_model_and_data
        executor, gm = executor_with_grasp

        gripper_body = gripper_body_names[0]

        # Attach box to gripper
        gm.attach_object("box1", gripper_body)

        # Get initial box position
        box1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box1")
        initial_pos = data.xpos[box1_id].copy()

        # Create a short trajectory
        q_start = np.array(
            [data.qpos[idx] for idx in executor.joint_qpos_indices]
        )
        q_end = q_start.copy()
        q_end[0] = 0.3  # Move shoulder

        positions = np.linspace(q_start, q_end, 5)
        velocities = np.zeros_like(positions)
        accelerations = np.zeros_like(positions)
        timestamps = np.linspace(0, 0.02, 5)

        traj = Trajectory(
            timestamps=timestamps,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
        )

        executor.execute(traj)

        # Box should have moved
        final_pos = data.xpos[box1_id].copy()
        assert not np.allclose(initial_pos, final_pos, atol=0.01)

    def test_executor_without_grasp_manager(
        self, mujoco_model_and_data, arm_joint_names
    ):
        """Executor works without grasp manager (no attachments)."""
        model, data = mujoco_model_and_data

        joint_qpos_indices = []
        for name in arm_joint_names:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            joint_qpos_indices.append(model.jnt_qposadr[jnt_id])

        executor = KinematicExecutor(
            model=model,
            data=data,
            joint_qpos_indices=joint_qpos_indices,
            control_dt=0.001,
            # No grasp_manager
        )

        # Should work without error
        q = np.array([data.qpos[idx] for idx in joint_qpos_indices])
        q[0] = 0.2
        executor.set_position(q)  # Should not raise


class TestKinematicPickAndPlace:
    """Integration test for full kinematic pick and place cycle."""

    def test_pick_move_place_cycle(
        self, mujoco_model_and_data, arm_joint_names, gripper_body_names
    ):
        """Full kinematic manipulation: pick object, move, place."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        # Create executor
        joint_qpos_indices = []
        for name in arm_joint_names:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            joint_qpos_indices.append(model.jnt_qposadr[jnt_id])

        executor = KinematicExecutor(
            model=model,
            data=data,
            joint_qpos_indices=joint_qpos_indices,
            control_dt=0.001,
            grasp_manager=gm,
        )

        # Create gripper
        gripper = Gripper(
            model=model,
            data=data,
            arm_name="right",
            actuator_name="right_ur5e/gripper/fingers_actuator",
            gripper_body_names=gripper_body_names,
            grasp_manager=gm,
            gripper_site_name="right_ur5e/gripper_attachment_site",
        )

        # Position box1 at gripper
        box1_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box1_joint")
        box1_qpos_adr = model.jnt_qposadr[box1_jnt_id]
        ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_ur5e/gripper_attachment_site"
        )
        mujoco.mj_forward(model, data)
        ee_pos = data.site_xpos[ee_site_id].copy()

        data.qpos[box1_qpos_adr : box1_qpos_adr + 3] = ee_pos
        data.qpos[box1_qpos_adr + 3 : box1_qpos_adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        # 1. Pick: close gripper kinematically
        gripper.set_candidate_objects(["box1"])
        grasped = gripper.kinematic_close()
        assert grasped == "box1"

        # Get box position after grasp
        box1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box1")
        pos_after_grasp = data.xpos[box1_id].copy()

        # 2. Move: change arm position
        q_current = np.array(
            [data.qpos[idx] for idx in joint_qpos_indices]
        )
        q_moved = q_current.copy()
        q_moved[0] = 0.5  # Rotate shoulder
        q_moved[1] = -0.5  # Lift shoulder

        executor.set_position(q_moved)

        # Box should have moved with arm
        pos_after_move = data.xpos[box1_id].copy()
        assert not np.allclose(pos_after_grasp, pos_after_move, atol=0.01)

        # 3. Place: open gripper
        gripper.kinematic_open()

        # Box should stay at new position (not attached anymore)
        pos_after_release = data.xpos[box1_id].copy()
        np.testing.assert_allclose(pos_after_move, pos_after_release, atol=0.001)

        # Move arm away - box should NOT move
        executor.set_position(q_current)
        pos_after_arm_moved = data.xpos[box1_id].copy()
        np.testing.assert_allclose(pos_after_release, pos_after_arm_moved, atol=0.001)
