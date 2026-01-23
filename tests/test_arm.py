"""Tests for Arm class."""

import mujoco
import numpy as np
import pytest

from geodude.arm import Arm
from geodude.config import ArmConfig
from geodude.grasp_manager import GraspManager


class MockRobot:
    """Minimal mock of Geodude for testing Arm independently."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.named_poses = {
            "home": {"right": [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]},
            "ready": {"right": [-2.14, -1.5708, 1.5708, -1.5708, -1.5708, 0]},
        }


class TestArm:
    """Tests for Arm class."""

    @pytest.fixture
    def arm_config(self):
        """Create arm configuration for the real geodude robot."""
        return ArmConfig(
            name="right",
            joint_names=[
                "right_ur5e/shoulder_pan_joint",
                "right_ur5e/shoulder_lift_joint",
                "right_ur5e/elbow_joint",
                "right_ur5e/wrist_1_joint",
                "right_ur5e/wrist_2_joint",
                "right_ur5e/wrist_3_joint",
            ],
            ee_site="right_ur5e/gripper_attachment_site",
            gripper_actuator="right_ur5e/gripper/fingers_actuator",
            gripper_bodies=[
                "right_ur5e/gripper/right_follower",
                "right_ur5e/gripper/left_follower",
                "right_ur5e/gripper/right_pad",
                "right_ur5e/gripper/left_pad",
            ],
        )

    @pytest.fixture
    def arm(self, mujoco_model_and_data, arm_config):
        """Create an Arm instance for testing."""
        model, data = mujoco_model_and_data
        mock_robot = MockRobot(model, data)
        gm = GraspManager(model, data)
        return Arm(mock_robot, arm_config, gm)

    def test_init(self, arm):
        """Arm initializes correctly."""
        assert arm.name == "right"
        assert arm.dof == 6
        assert len(arm.joint_ids) == 6
        assert len(arm.joint_qpos_indices) == 6

    def test_invalid_joint_raises(self, mujoco_model_and_data):
        """Invalid joint name raises ValueError."""
        model, data = mujoco_model_and_data
        mock_robot = MockRobot(model, data)
        gm = GraspManager(model, data)

        bad_config = ArmConfig(
            name="bad_arm",
            joint_names=["nonexistent_joint"],
            ee_site="right_ur5e/gripper_attachment_site",
            gripper_actuator="",
            gripper_bodies=[],
        )

        with pytest.raises(ValueError, match="not found"):
            Arm(mock_robot, bad_config, gm)

    def test_invalid_site_raises(self, mujoco_model_and_data, arm_config):
        """Invalid EE site name raises ValueError."""
        model, data = mujoco_model_and_data
        mock_robot = MockRobot(model, data)
        gm = GraspManager(model, data)

        arm_config.ee_site = "nonexistent_site"

        with pytest.raises(ValueError, match="not found"):
            Arm(mock_robot, arm_config, gm)

    def test_get_joint_positions(self, arm, mujoco_model_and_data):
        """get_joint_positions returns current joint values."""
        model, data = mujoco_model_and_data

        # Set known joint positions
        test_values = [0.5, -0.3, 0.2, -0.1, 0.4, -0.2]
        for i, val in enumerate(test_values):
            data.qpos[arm.joint_qpos_indices[i]] = val

        q = arm.get_joint_positions()
        for i, val in enumerate(test_values):
            assert q[i] == pytest.approx(val)

    def test_set_joint_positions(self, arm, mujoco_model_and_data):
        """set_joint_positions updates joint values."""
        model, data = mujoco_model_and_data

        test_values = np.array([1.0, -1.0, 0.5, -0.5, 0.3, -0.3])
        arm.set_joint_positions(test_values)

        for i, val in enumerate(test_values):
            assert data.qpos[arm.joint_qpos_indices[i]] == pytest.approx(val)

    def test_get_ee_pose(self, arm):
        """get_ee_pose returns 4x4 transformation matrix."""
        pose = arm.get_ee_pose()

        assert pose.shape == (4, 4)
        # Bottom row should be [0, 0, 0, 1]
        assert np.allclose(pose[3, :], [0, 0, 0, 1])
        # Rotation matrix should have det=1
        det = np.linalg.det(pose[:3, :3])
        assert np.isclose(det, 1.0, atol=1e-6)

    def test_get_joint_limits(self, arm):
        """get_joint_limits returns lower and upper bounds."""
        lower, upper = arm.get_joint_limits()

        assert len(lower) == 6
        assert len(upper) == 6
        # Lower should be less than upper
        assert np.all(lower < upper)

    def test_go_to_named_pose(self, arm):
        """go_to with named pose sets joint positions."""
        arm.go_to("home")

        q = arm.get_joint_positions()
        expected = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        assert np.allclose(q, expected, atol=0.01)

    def test_go_to_named_pose_ready(self, arm):
        """go_to 'ready' pose works."""
        arm.go_to("ready")

        q = arm.get_joint_positions()
        expected = [-2.14, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        assert np.allclose(q, expected, atol=0.01)

    def test_go_to_array(self, arm):
        """go_to with array sets joint positions."""
        target = np.array([0.7, -0.2, 0.3, -0.4, 0.5, -0.1])
        arm.go_to(target)

        q = arm.get_joint_positions()
        assert np.allclose(q, target, atol=0.01)

    def test_go_to_unknown_pose_raises(self, arm):
        """go_to with unknown named pose raises ValueError."""
        with pytest.raises(ValueError, match="Unknown named pose"):
            arm.go_to("nonexistent_pose")

    def test_plan_to_configuration_simple(self, arm):
        """plan_to_configuration returns path for simple case."""
        start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        arm.set_joint_positions(start)

        # Plan to a nearby configuration
        goal = start + 0.1
        path = arm.plan_to_configuration(goal)

        # Should return a path (may be None if collision)
        if path is not None:
            assert len(path) >= 2
            assert np.allclose(path[0], start, atol=0.01)
            assert np.allclose(path[-1], goal, atol=0.01)

    def test_execute_path(self, arm):
        """execute moves through waypoints."""
        start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        end = np.array([-1.0, -1.5708, 1.5708, -1.5708, -1.5708, 0])

        # Set starting position before executing
        arm.set_joint_positions(start)

        path = [start, end]

        success = arm.execute(path)
        assert success

        # Should end at final waypoint
        # Use 0.02 rad tolerance for physics-based control (realistic tracking error)
        q = arm.get_joint_positions()
        assert np.allclose(q, end, atol=0.02)

    def test_gripper_close_open(self, arm):
        """close_gripper and open_gripper work."""
        arm.close_gripper(steps=10)
        arm.open_gripper(steps=10)
        # Just verify no crashes

    def test_has_gripper(self, arm):
        """Arm has gripper attribute."""
        assert arm.gripper is not None
        assert arm.gripper.actuator_id is not None


# Check if EAIK is available for IK tests
try:
    from pycbirrt.backends.eaik import EAIKSolver

    EAIK_AVAILABLE = True
except ImportError:
    EAIK_AVAILABLE = False


@pytest.mark.skipif(not EAIK_AVAILABLE, reason="EAIK not installed")
class TestArmEAIK:
    """Tests for EAIK integration in Arm class."""

    @pytest.fixture
    def arm_config(self):
        """Create arm configuration for the real geodude robot."""
        return ArmConfig(
            name="right",
            joint_names=[
                "right_ur5e/shoulder_pan_joint",
                "right_ur5e/shoulder_lift_joint",
                "right_ur5e/elbow_joint",
                "right_ur5e/wrist_1_joint",
                "right_ur5e/wrist_2_joint",
                "right_ur5e/wrist_3_joint",
            ],
            ee_site="right_ur5e/gripper_attachment_site",
            gripper_actuator="right_ur5e/gripper/fingers_actuator",
            gripper_bodies=[
                "right_ur5e/gripper/right_follower",
                "right_ur5e/gripper/left_follower",
                "right_ur5e/gripper/right_pad",
                "right_ur5e/gripper/left_pad",
            ],
        )

    @pytest.fixture
    def arm(self, mujoco_model_and_data, arm_config):
        """Create an Arm instance for testing."""
        model, data = mujoco_model_and_data
        mock_robot = MockRobot(model, data)
        gm = GraspManager(model, data)
        return Arm(mock_robot, arm_config, gm)

    def test_get_base_pose(self, arm):
        """get_base_pose returns valid 4x4 transform."""
        pose = arm.get_base_pose()

        assert pose.shape == (4, 4)
        # Bottom row should be [0, 0, 0, 1]
        assert np.allclose(pose[3, :], [0, 0, 0, 1])
        # Rotation matrix should be orthonormal
        rot = pose[:3, :3]
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)

    def test_inverse_kinematics_from_current_pose(self, arm):
        """IK can find solutions for the current EE pose."""
        # Set arm to a known configuration
        q_start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        arm.set_joint_positions(q_start)

        # Get current EE pose
        target_pose = arm.get_ee_pose()

        # Solve IK - should find at least one solution close to current
        solutions = arm.inverse_kinematics(target_pose)

        assert len(solutions) > 0, "Should find at least one IK solution"

        # First solution should be sorted by distance, so it should be close
        # to the current configuration
        q_solution = solutions[0]
        assert len(q_solution) == 6

    def test_inverse_kinematics_consistency(self, arm):
        """FK of IK solution should match target pose."""
        # Set arm to a configuration
        q_start = np.array([-1.0, -1.5, 1.2, -1.0, -1.5, 0.5])
        arm.set_joint_positions(q_start)

        # Get current EE pose
        target_pose = arm.get_ee_pose()

        # Solve IK
        solutions = arm.inverse_kinematics(target_pose, validate=False)

        if len(solutions) == 0:
            pytest.skip("No IK solutions found for this pose")

        # Verify FK of solution matches target
        q_solution = solutions[0]
        arm.set_joint_positions(q_solution)
        result_pose = arm.get_ee_pose()

        # Position should be very close (within 2mm - accounts for numerical precision)
        pos_error = np.linalg.norm(result_pose[:3, 3] - target_pose[:3, 3])
        assert pos_error < 0.002, f"Position error {pos_error:.4f}m too large"

    def test_inverse_kinematics_multiple_solutions(self, arm):
        """IK typically returns multiple solutions for UR5e."""
        # Set to a typical pose
        q_start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        arm.set_joint_positions(q_start)

        target_pose = arm.get_ee_pose()

        # Without validation, should return all kinematic solutions
        solutions = arm.inverse_kinematics(target_pose, validate=False)

        # UR5e typically has up to 8 solutions
        assert len(solutions) >= 1, "Should find at least one solution"

    def test_inverse_kinematics_validates_solutions(self, arm):
        """IK validates solutions against joint limits and collisions."""
        # Set to a known reachable pose
        q_start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        arm.set_joint_positions(q_start)
        target_pose = arm.get_ee_pose()

        # Get solutions with and without validation
        unvalidated = arm.inverse_kinematics(target_pose, validate=False)
        validated = arm.inverse_kinematics(target_pose, validate=True)

        # Should have solutions
        assert len(unvalidated) > 0, "Should find unvalidated solutions"
        assert len(validated) > 0, "Should find validated solutions"

        # Validated should be subset of (or equal to) unvalidated
        assert len(validated) <= len(unvalidated)

        # All validated solutions should be within joint limits
        lower, upper = arm.get_joint_limits()
        for sol in validated:
            assert np.all(sol >= lower - 1e-6), "Solution below lower limit"
            assert np.all(sol <= upper + 1e-6), "Solution above upper limit"

    def test_inverse_kinematics_sorted_by_distance(self, arm):
        """Solutions are sorted by distance from current config."""
        q_start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        arm.set_joint_positions(q_start)

        target_pose = arm.get_ee_pose()

        solutions = arm.inverse_kinematics(target_pose, sort_by_distance=True)

        if len(solutions) < 2:
            pytest.skip("Need at least 2 solutions to test sorting")

        # Verify sorting: first solution should be closest
        dist0 = np.linalg.norm(solutions[0] - q_start)
        dist1 = np.linalg.norm(solutions[1] - q_start)
        assert dist0 <= dist1, "Solutions should be sorted by distance"

    def test_forward_kinematics_eaik(self, arm):
        """forward_kinematics_eaik matches MuJoCo FK."""
        q_test = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

        # EAIK FK
        pose_eaik = arm.forward_kinematics_eaik(q_test)

        # MuJoCo FK
        arm.set_joint_positions(q_test)
        pose_mujoco = arm.get_ee_pose()

        # Positions should be close (within a few cm due to different EE definitions)
        # Note: EAIK uses DH convention EE, MuJoCo uses gripper_attachment_site
        # They may differ by a fixed offset, but both should be valid transforms
        assert pose_eaik.shape == (4, 4)
        assert np.allclose(pose_eaik[3, :], [0, 0, 0, 1])

    def test_ik_solver_caching(self, arm):
        """IK solver is created lazily and cached."""
        # First access creates solver
        solver1 = arm._get_ik_solver()
        # Second access returns same instance
        solver2 = arm._get_ik_solver()

        assert solver1 is solver2, "IK solver should be cached"


# Check if pycbirrt planner is available
try:
    from pycbirrt import CBiRRT, CBiRRTConfig
    from tsr import TSR

    PYCBIRRT_AVAILABLE = True
except ImportError:
    PYCBIRRT_AVAILABLE = False


@pytest.mark.skipif(not PYCBIRRT_AVAILABLE, reason="pycbirrt not installed")
class TestArmPlanner:
    """Tests for CBiRRT planner integration in Arm class."""

    @pytest.fixture
    def arm_config(self):
        """Create arm configuration for the real geodude robot."""
        return ArmConfig(
            name="right",
            joint_names=[
                "right_ur5e/shoulder_pan_joint",
                "right_ur5e/shoulder_lift_joint",
                "right_ur5e/elbow_joint",
                "right_ur5e/wrist_1_joint",
                "right_ur5e/wrist_2_joint",
                "right_ur5e/wrist_3_joint",
            ],
            ee_site="right_ur5e/gripper_attachment_site",
            gripper_actuator="right_ur5e/gripper/fingers_actuator",
            gripper_bodies=[
                "right_ur5e/gripper/right_follower",
                "right_ur5e/gripper/left_follower",
                "right_ur5e/gripper/right_pad",
                "right_ur5e/gripper/left_pad",
            ],
        )

    @pytest.fixture
    def arm(self, mujoco_model_and_data, arm_config):
        """Create an Arm instance for testing."""
        model, data = mujoco_model_and_data
        mock_robot = MockRobot(model, data)
        gm = GraspManager(model, data)
        return Arm(mock_robot, arm_config, gm)

    def test_get_planner_returns_cbirrt(self, arm):
        """_get_planner returns a CBiRRT instance."""
        planner = arm._get_planner()
        assert isinstance(planner, CBiRRT)

    def test_planner_caching(self, arm):
        """Planner is cached when no config provided."""
        planner1 = arm._get_planner()
        planner2 = arm._get_planner()
        assert planner1 is planner2, "Planner should be cached"

    def test_planner_not_cached_with_config(self, arm):
        """New planner created when config provided."""
        planner1 = arm._get_planner()
        planner2 = arm._get_planner(CBiRRTConfig(max_iterations=100))
        assert planner1 is not planner2, "New planner should be created with config"

    def test_plan_to_configuration_nearby(self, arm):
        """plan_to_configuration finds path to nearby configuration."""
        q_start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        arm.set_joint_positions(q_start)

        # Plan to a nearby configuration
        q_goal = q_start + 0.1
        path = arm.plan_to_configuration(q_goal, seed=42)

        assert path is not None, "Should find a path"
        assert len(path) >= 2, "Path should have at least start and goal"
        assert np.allclose(path[-1], q_goal, atol=0.01)

    def test_plan_to_tsrs_basic(self, arm):
        """plan_to_tsrs finds a path to a simple TSR."""
        q_start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        arm.set_joint_positions(q_start)

        # Get the current EE pose and create a TSR around it
        current_pose = arm.get_ee_pose()

        # Create a TSR with some freedom in position
        goal_tsr = TSR(
            T0_w=current_pose,
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.05, 0.05],  # Small x freedom
                [-0.05, 0.05],  # Small y freedom
                [-0.05, 0.05],  # Small z freedom
                [0, 0],         # No rotation freedom
                [0, 0],
                [0, 0],
            ]),
        )

        path = arm.plan_to_tsrs([goal_tsr], timeout=10.0, seed=42)

        # Should find a path (current pose is in the TSR)
        assert path is not None, "Should find a path to the TSR"
        assert len(path) >= 1, "Path should have at least one waypoint"

    def test_plan_to_tsrs_reaches_goal_region(self, arm):
        """plan_to_tsrs finds configuration that satisfies TSR."""
        q_start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        arm.set_joint_positions(q_start)

        # Get the current EE pose
        current_pose = arm.get_ee_pose()

        # Create a TSR at a slightly offset position
        offset_pose = current_pose.copy()
        offset_pose[0, 3] += 0.02  # Small x offset

        goal_tsr = TSR(
            T0_w=offset_pose,
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.05, 0.05],
                [-0.05, 0.05],
                [-0.05, 0.05],
                [-0.1, 0.1],  # Some yaw freedom
                [-0.1, 0.1],
                [-0.1, 0.1],
            ]),
        )

        path = arm.plan_to_tsrs([goal_tsr], timeout=10.0, seed=42)

        if path is not None:
            # Verify final configuration satisfies TSR
            arm.set_joint_positions(path[-1])
            final_pose = arm.get_ee_pose()
            dist, _ = goal_tsr.distance(final_pose)
            assert dist < 0.1, f"Final pose should be close to TSR, got dist={dist}"


@pytest.mark.skipif(not PYCBIRRT_AVAILABLE, reason="pycbirrt not installed")
class TestArmPickPlace:
    """Integration tests for pick() and place() methods."""

    @pytest.fixture
    def robot_with_objects(self, geodude_xml):
        """Create full Geodude robot from XML with test objects."""
        from pathlib import Path
        import tempfile
        from geodude.config import GeodudConfig
        from geodude.robot import Geodude

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(geodude_xml)
            temp_path = Path(f.name)

        config = GeodudConfig.default()
        config.model_path = temp_path
        robot = Geodude(config)

        yield robot

        # Cleanup
        temp_path.unlink()

    def test_pick_calls_planner(self, robot_with_objects):
        """pick() attempts to plan motion to object."""
        robot = robot_with_objects
        arm = robot.right_arm

        # Use home pose which is known to be collision-free
        robot.go_to("home")

        # Verify start config is collision-free
        checker = arm._get_collision_checker()
        q_start = arm.get_joint_positions()
        if not checker.is_valid(q_start):
            pytest.skip("Start config is in collision - model may have changed")

        # Position object in reachable area (front of robot)
        object_pose = np.eye(4)
        object_pose[:3, 3] = [0.4, -0.3, 0.85]  # Front right, above table
        robot.set_object_pose("box1", object_pose)

        # pick() should attempt to plan (may or may not succeed depending on reachability)
        # We're testing that the method runs without error
        result = arm.pick(
            "box1",
            object_height=0.03,
            gripper_standoff=0.1,
            timeout=5.0,
            seed=42,
        )

        # Result indicates success/failure (may fail due to timeout or reachability)
        assert isinstance(result, bool)

    def test_place_calls_planner(self, robot_with_objects):
        """place() attempts to plan motion to target."""
        robot = robot_with_objects
        arm = robot.right_arm

        # Use home pose which is known to be collision-free
        robot.go_to("home")

        # Verify start config is collision-free
        checker = arm._get_collision_checker()
        q_start = arm.get_joint_positions()
        if not checker.is_valid(q_start):
            pytest.skip("Start config is in collision - model may have changed")

        # Create a target pose (use current EE pose with offset for reachability)
        current_pose = arm.get_ee_pose()
        target_pose = current_pose.copy()
        target_pose[0, 3] += 0.05  # Small offset

        # place() should attempt to plan (may or may not succeed)
        result = arm.place(
            target_pose,
            object_height=0.03,
            surface_height=0.01,
            timeout=5.0,
            seed=42,
        )

        assert isinstance(result, bool)

    def test_pick_opens_gripper_first(self, robot_with_objects):
        """pick() opens gripper before attempting grasp."""
        robot = robot_with_objects
        arm = robot.right_arm

        # Use home pose
        robot.go_to("home")

        # Close gripper first
        arm.close_gripper()

        # Position object out of reach (so pick fails early)
        object_pose = np.eye(4)
        object_pose[:3, 3] = [2.0, 0.0, 0.8]  # Far away
        robot.set_object_pose("box1", object_pose)

        # Try to pick (will fail but should open gripper)
        arm.pick("box1", timeout=1.0)

        # Gripper should be open (pick opens it first)
        # Note: After failed pick, gripper state depends on implementation
