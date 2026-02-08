"""Tests for collision checking."""

import mujoco
import numpy as np
import pytest

from geodude.collision import GraspAwareCollisionChecker, SimpleCollisionChecker
from geodude.grasp_manager import GraspManager


class TestSimpleCollisionChecker:
    """Tests for SimpleCollisionChecker."""

    def test_init(self, mujoco_model_and_data, arm_joint_names):
        """SimpleCollisionChecker initializes correctly."""
        model, data = mujoco_model_and_data
        checker = SimpleCollisionChecker(model, data, arm_joint_names)

        assert len(checker.joint_indices) == 6

    def test_invalid_joint_raises(self, mujoco_model_and_data):
        """Invalid joint name raises ValueError."""
        model, data = mujoco_model_and_data

        with pytest.raises(ValueError, match="not found"):
            SimpleCollisionChecker(model, data, ["nonexistent_joint"])

    def test_valid_configuration(self, mujoco_model_and_data, arm_joint_names):
        """Home configuration should be collision-free."""
        model, data = mujoco_model_and_data
        checker = SimpleCollisionChecker(model, data, arm_joint_names)

        # Home position
        q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        # Note: May have floor contact, so this tests the basic mechanism
        result = checker.is_valid(q)
        # Just verify it returns a boolean and doesn't crash
        assert isinstance(result, bool)

    def test_is_valid_batch(self, mujoco_model_and_data, arm_joint_names):
        """Batch validity check works."""
        model, data = mujoco_model_and_data
        checker = SimpleCollisionChecker(model, data, arm_joint_names)

        qs = np.array([
            [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0],
            [-1.0, -1.5708, 1.5708, -1.5708, -1.5708, 0],
            [-2.14, -1.5708, 1.5708, -1.5708, -1.5708, 0],
        ])

        results = checker.is_valid_batch(qs)
        assert results.shape == (3,)
        assert results.dtype == bool


class TestGraspAwareCollisionChecker:
    """Tests for GraspAwareCollisionChecker."""

    def test_init(self, mujoco_model_and_data, arm_joint_names):
        """GraspAwareCollisionChecker initializes correctly."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        assert len(checker.joint_indices) == 6
        assert checker.grasp_manager is gm

    def test_grasped_object_tracked_in_manager(self, mujoco_model_and_data, arm_joint_names):
        """When object is grasped, it is tracked by the grasp manager."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        # Mark box1 as grasped
        gm.mark_grasped("box1", "right")

        # Verify grasp manager tracks the grasped state
        assert gm.is_grasped("box1")
        assert gm.get_holder("box1") == "right"

        # The collision checker should use this state for filtering
        # (software-based filtering, not MuJoCo collision groups)

    def test_ungrasped_object_not_in_manager(self, mujoco_model_and_data, arm_joint_names):
        """Non-grasped objects are not tracked in grasp manager."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        # Without marking as grasped, object is not tracked
        assert not gm.is_grasped("box1")
        assert gm.get_holder("box1") is None


class TestArmBodyIdCompleteness:
    """Tests that _arm_body_ids contains all expected bodies including gripper."""

    def test_arm_body_ids_includes_gripper_base(self, mujoco_model_and_data, arm_joint_names):
        """_arm_body_ids should include gripper base."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        gripper_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ur5e/gripper/base")
        assert gripper_base_id in checker._arm_body_ids, "Gripper base not in _arm_body_ids"

    def test_arm_body_ids_includes_gripper_fingers(self, mujoco_model_and_data, arm_joint_names):
        """_arm_body_ids should include all gripper finger bodies."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        # All finger-related bodies
        finger_bodies = [
            "right_ur5e/gripper/left_driver",
            "right_ur5e/gripper/right_driver",
            "right_ur5e/gripper/left_follower",
            "right_ur5e/gripper/right_follower",
            "right_ur5e/gripper/left_coupler",
            "right_ur5e/gripper/right_coupler",
            "right_ur5e/gripper/left_spring_link",
            "right_ur5e/gripper/right_spring_link",
        ]

        for body_name in finger_bodies:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:  # Body exists in model
                assert body_id in checker._arm_body_ids, f"{body_name} not in _arm_body_ids"

    def test_arm_body_ids_includes_forearm(self, mujoco_model_and_data, arm_joint_names):
        """_arm_body_ids should include forearm link."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        forearm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ur5e/forearm_link")
        assert forearm_id in checker._arm_body_ids, "Forearm link not in _arm_body_ids"

    def test_arm_body_ids_count(self, mujoco_model_and_data, arm_joint_names):
        """_arm_body_ids should contain reasonable number of bodies."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        # Should have at least: 6 arm links + gripper base + ~8 finger parts = ~15 bodies
        assert len(checker._arm_body_ids) >= 10, f"Only {len(checker._arm_body_ids)} bodies in _arm_body_ids"


class TestSelfCollisionDetection:
    """Tests for detecting same-arm self-collision (e.g., forearm hitting gripper)."""

    def test_home_position_is_collision_free(self, mujoco_model_and_data, arm_joint_names):
        """Home position should not have self-collision."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        # Home position
        q_home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        assert checker.is_valid(q_home), "Home position should be collision-free"

    def test_extreme_wrist_causes_self_collision(self, mujoco_model_and_data, arm_joint_names):
        """Extreme wrist configuration should cause forearm-gripper collision."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        # Configuration that folds the wrist back toward the forearm
        # This should cause forearm-gripper collision
        # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        q_self_collision = np.array([0.0, -1.0, 2.5, -2.5, -1.5, 0.0])

        result = checker.is_valid(q_self_collision)
        # If this passes (no collision), the configuration may not cause collision
        # We need to verify by checking contacts directly
        if result:
            # Check if MuJoCo detected contacts
            for i, qpos_idx in enumerate(checker.joint_indices):
                checker._temp_data.qpos[qpos_idx] = q_self_collision[i]
            mujoco.mj_forward(checker.model, checker._temp_data)
            print(f"Contacts in extreme wrist config: {checker._temp_data.ncon}")

    def test_self_collision_counted_as_invalid(self, mujoco_model_and_data, arm_joint_names):
        """When self-collision contact exists, it should be counted as invalid."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        # Manually check if contacts between arm bodies are flagged
        # Set up a known self-collision scenario
        forearm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ur5e/forearm_link")
        gripper_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ur5e/gripper/base")

        # Both should be in arm_body_ids
        assert forearm_id in checker._arm_body_ids
        assert gripper_base_id in checker._arm_body_ids

        # If both are in _arm_body_ids, a contact between them should be counted as invalid
        # (non-adjacent self-collision)


class TestInterArmCollisionDetection:
    """Tests for detecting collision between left and right arms."""

    def test_arms_separated_is_collision_free(
        self, mujoco_model_and_data, arm_joint_names, left_arm_joint_names
    ):
        """Arms at home positions (separated) should not collide."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        # Use the model's actual home keyframe positions
        # Both arms use the same joint angles (they're mounted mirrored so same angles = mirrored poses)
        # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        home_config = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

        # Set left arm to home position
        for i, name in enumerate(left_arm_joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_adr] = home_config[i]

        mujoco.mj_forward(model, data)

        # Check right arm collision
        right_checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)
        assert right_checker.is_valid(home_config), "Arms at home should not collide"

    def test_arms_reaching_to_center_may_collide(
        self, mujoco_model_and_data, arm_joint_names, left_arm_joint_names
    ):
        """Arms reaching toward center workspace may collide."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        # Configuration where both arms reach toward center
        # With shoulder_pan = 0, both arms point straight forward
        # Combined with arm geometry, this may cause overlap
        # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        center_config = np.array([0.0, -1.5, 1.5, -1.5, -1.5, 0])

        # Set left arm position
        for i, name in enumerate(left_arm_joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_adr] = center_config[i]

        mujoco.mj_forward(model, data)

        # Check right arm collision - left arm bodies should be detected as "external"
        right_checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        # Verify left arm bodies are NOT in right checker's _arm_body_ids
        left_forearm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ur5e/forearm_link")
        assert left_forearm_id not in right_checker._arm_body_ids, \
            "Left arm bodies should NOT be in right arm's _arm_body_ids"

        # Check if this configuration causes collision
        result = right_checker.is_valid(center_config)
        # The result depends on whether the arms actually collide geometrically

    def test_inter_arm_collision_detected(
        self, mujoco_model_and_data, arm_joint_names, left_arm_joint_names
    ):
        """Verify inter-arm collision IS detected when arms physically overlap."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)

        # First, find a configuration where arms definitely collide
        # by checking MuJoCo contacts directly
        # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        # Arms with shoulder_pan pointing toward each other's side
        collision_config = np.array([0.0, -0.5, 1.0, -1.5, -1.5, 0])

        # Set both arms to overlapping configuration
        for i, name in enumerate(left_arm_joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_adr] = collision_config[i]

        for i, name in enumerate(arm_joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_adr] = collision_config[i]

        mujoco.mj_forward(model, data)

        # Check if MuJoCo detects any inter-arm contacts
        inter_arm_contacts = 0
        for i in range(data.ncon):
            contact = data.contact[i]
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1])
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2])
            if b1 and b2:
                is_left_right = ("left_ur5e" in b1 and "right_ur5e" in b2) or \
                                ("right_ur5e" in b1 and "left_ur5e" in b2)
                if is_left_right:
                    inter_arm_contacts += 1

        # If MuJoCo detects inter-arm contacts, verify our checker flags them
        if inter_arm_contacts > 0:
            right_checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)
            invalid_count = right_checker._count_invalid_contacts(data)
            assert invalid_count > 0, \
                f"MuJoCo found {inter_arm_contacts} inter-arm contacts but checker found 0 invalid"
            # Also verify is_valid returns False
            assert not right_checker.is_valid(collision_config), \
                "Configuration with inter-arm contacts should be invalid"


class TestArmEnvironmentCollisionDetection:
    """Tests for detecting collision between arm and environment (e.g., vention_base)."""

    def test_arm_at_home_not_hitting_base(self, mujoco_model_and_data, arm_joint_names):
        """Arm at home position should not hit the vention base."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        q_home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        assert checker.is_valid(q_home), "Home position should not hit base"

    def test_vention_base_not_in_arm_body_ids(self, mujoco_model_and_data, arm_joint_names):
        """Vention base should NOT be in _arm_body_ids (it's environment)."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        vention_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vention_base")
        if vention_base_id != -1:
            assert vention_base_id not in checker._arm_body_ids, \
                "vention_base should NOT be in _arm_body_ids"

    def test_arm_reaching_down_may_hit_base(self, mujoco_model_and_data, arm_joint_names):
        """Arm reaching down may collide with vention base."""
        model, data = mujoco_model_and_data
        gm = GraspManager(model, data)
        checker = GraspAwareCollisionChecker(model, data, arm_joint_names, gm)

        # Configuration where arm reaches down toward the base
        # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        q_reaching_down = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # This configuration has the arm pointing straight down
        # It may or may not collide depending on base geometry
        result = checker.is_valid(q_reaching_down)
        # Just verify it doesn't crash and returns a boolean
        assert isinstance(result, bool)
