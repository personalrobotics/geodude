#!/usr/bin/env python3
"""Recycling demo: pick up objects and place them elsewhere.

Demonstrates manipulation using both execution modes:
- Kinematic: Direct position setting with object attachment (fast, no physics)
- Physics: Full dynamics simulation with friction-based grasping (realistic)

This is the first step toward a full recycling demo where cans are
picked from a worktop and discarded into a bin.

Usage:
    uv run mjpython examples/recycle_objects.py [--physics]

Arguments:
    --physics    Use physics-based execution (default: kinematic)
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from geodude import Geodude
from geodude.executor import KinematicExecutor, PhysicsExecutor
from geodude.grasp_manager import GraspManager
from geodude.gripper import Gripper
from geodude.trajectory import Trajectory
from geodude.tsr_utils import compensate_tsr_for_gripper
from tsr import TSR
from tsr.core.tsr_primitive import load_template_file
from pathlib import Path

# Path to geodude TSR templates
GEODUDE_TSR_DIR = Path(__file__).parent.parent / "tsr_templates"


def create_scene_with_object():
    """Load Geodude model and add a graspable object and recycle bin.

    Returns:
        Path to temporary XML file with objects added.
    """
    import tempfile
    from geodude_assets import get_model_path
    from prl_assets import OBJECTS_DIR

    xml_path = get_model_path()
    with open(xml_path) as f:
        xml_content = f.read()

    # Add meshdir for assets
    meshdir = str(xml_path.parent) + "/"
    xml_content = xml_content.replace(
        '<compiler autolimits="true" angle="radian"/>',
        f'<compiler autolimits="true" angle="radian" meshdir="{meshdir}"/>',
    )

    # Read the recycle bin XML from prl_assets
    recycle_bin_xml_path = OBJECTS_DIR / "recycle_bin" / "recycle_bin.xml"
    with open(recycle_bin_xml_path) as f:
        bin_xml_content = f.read()

    # Extract just the body definition from the bin XML (skip mujoco/worldbody tags)
    # The bin XML has the body at pos="0 0 0", we'll wrap it with our desired position
    import re
    bin_body_match = re.search(r'(<body name="recycle_bin".*?</body>)', bin_xml_content, re.DOTALL)
    if not bin_body_match:
        raise ValueError("Could not find recycle_bin body in XML")
    bin_body_xml = bin_body_match.group(1)

    # Also extract the material definition
    bin_material_match = re.search(r'(<material name="bin_blue"[^/]*/?>)', bin_xml_content)
    bin_material_xml = bin_material_match.group(1) if bin_material_match else ""

    # Add material to asset section if it exists
    if bin_material_xml:
        # Insert material into existing asset section or create one
        if "<asset>" in xml_content:
            xml_content = xml_content.replace("<asset>", f"<asset>\n    {bin_material_xml}")
        else:
            # Insert asset section before worldbody
            xml_content = xml_content.replace("<worldbody>", f"<asset>\n    {bin_material_xml}\n  </asset>\n\n  <worldbody>")

    # Position the bin within reach of the right arm
    # Right arm base is at [0.33, -0.858, 1.12] with ~0.85m reach
    # Bin reference is at its base (z=0 in bin coords), walls extend up to z≈0.30
    # Place bin at x=0.75 (forward, clear of vention base), y=-0.35 (right side), z=0.50 (raised)
    # This puts bin top at z≈0.80, gripper at z≈1.0-1.05 with TSR offset
    bin_position = "0.75 -0.35 0.50"

    # Add a red cylinder (can) in front of the right arm
    # Can: radius=0.033m, half-height=0.06m -> place at table + half-height
    # Position: x=0.4 (in front), y=-0.2 (right side), z=0.81 (on table)
    object_xml = f"""
    <!-- Graspable object -->
    <body name="can" pos="0.4 -0.2 0.81">
      <freejoint name="can_joint"/>
      <geom name="can_geom" type="cylinder" size="0.033 0.06"
            rgba="0.8 0.1 0.1 1" mass="0.05"
            contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>
    </body>

    <!-- Recycle bin from prl_assets (with collision enabled) -->
    <!-- Bin base at floor level, walls extend up to z≈0.30 -->
    <body name="recycle_bin_base" pos="{bin_position}">
      {bin_body_xml}
    </body>
  </worldbody>"""

    modified_xml = xml_content.replace("</worldbody>", object_xml)

    # Write to temp file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
    temp_file.write(modified_xml)
    temp_file.close()
    return temp_file.name


def get_object_position(data, object_name="can"):
    """Get current object position."""
    model = data.model
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    return data.xpos[body_id].copy()


def print_status(msg, flush=True):
    """Print with flush for mjpython."""
    print(msg, flush=flush)


def solve_ik(arm, pose: np.ndarray, validate: bool = True):
    """Solve IK and return first solution or None.

    Wraps arm.inverse_kinematics which returns a list of all solutions.

    Args:
        arm: The arm to solve IK for
        pose: 4x4 target pose
        validate: Whether to check for collisions (set False for grasp poses)
    """
    solutions = arm.inverse_kinematics(pose, validate=validate, sort_by_distance=True)
    if solutions is not None and len(solutions) > 0:
        return solutions[0]
    return None


def load_can_grasp_tsrs(object_pos: np.ndarray, pregrasp_standoff: float = 0.15):
    """Load can side grasp TSR template and create grasp + pre-grasp TSRs.

    The pre-grasp TSR has extra standoff for collision-free planning.
    The grasp TSR is for the final grasp pose.

    Applies gripper frame compensation for the Robotiq 2F-140.

    Args:
        object_pos: [x, y, z] position of object center
        pregrasp_standoff: Extra standoff for pre-grasp (meters)

    Returns:
        Tuple of (pregrasp_tsr, grasp_tsr)
    """
    # Load template from geodude/tsr/grasps/
    template_path = GEODUDE_TSR_DIR / "grasps" / "can_side_grasp.yaml"
    template = load_template_file(str(template_path))

    # Create object pose matrix (object at given position, no rotation)
    object_pose = np.eye(4)
    object_pose[:3, 3] = object_pos

    # Create grasp TSR with template parameters
    grasp_tsr = TSR(T0_w=object_pose, Tw_e=template.Tw_e, Bw=template.Bw)

    # Pre-grasp TSR: increase standoff by backing up along approach direction
    # With radial approach, gripper z points toward -x (toward object center).
    # Backing up means moving along +x (away from object).
    # In Tw_e, the translation is in TSR frame. The standoff is along gripper -z,
    # which is +x in TSR frame. So we ADD to x component.
    Tw_e_pregrasp = template.Tw_e.copy()
    Tw_e_pregrasp[0, 3] += pregrasp_standoff  # Back up along +x (away from object)

    pregrasp_tsr = TSR(T0_w=object_pose, Tw_e=Tw_e_pregrasp, Bw=template.Bw)

    # Apply gripper frame compensation for Robotiq 2F-140
    # The template uses canonical TSR convention (y=fingers, z=approach),
    # but Robotiq has an internal -90° z rotation that needs compensation.
    grasp_tsr = compensate_tsr_for_gripper(grasp_tsr, template.subject)
    pregrasp_tsr = compensate_tsr_for_gripper(pregrasp_tsr, template.subject)

    return pregrasp_tsr, grasp_tsr


def load_place_tsr(model, data):
    """Load the recycle bin drop TSR.

    Args:
        model: MuJoCo model
        data: MuJoCo data

    Returns:
        TSR for placing above recycle bin
    """
    # Load template
    template_path = GEODUDE_TSR_DIR / "places" / "recycle_bin_drop.yaml"
    template = load_template_file(str(template_path))

    # Get recycle bin position from model
    bin_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "recycle_bin")
    if bin_body_id == -1:
        raise ValueError("recycle_bin body not found in model")
    bin_pos = data.xpos[bin_body_id].copy()

    # Create reference pose (bin at its position, no rotation)
    bin_pose = np.eye(4)
    bin_pose[:3, 3] = bin_pos

    # Create place TSR
    place_tsr = TSR(T0_w=bin_pose, Tw_e=template.Tw_e, Bw=template.Bw)

    # Apply gripper frame compensation
    place_tsr = compensate_tsr_for_gripper(place_tsr, template.subject)

    return place_tsr


def parallel_plan_to_place(robot, arm, place_tsr, timeout=10.0):
    """Plan to place TSR using parallel planning at different transit heights.

    Creates separate environment forks with independent MjData instances for each
    planning attempt. This enables true parallel planning with threads.

    Args:
        robot: Geodude robot instance (for forking)
        arm: Robot arm
        place_tsr: TSR for the place location
        timeout: Total timeout for all planning attempts

    Returns:
        Tuple of (transit_height, q_transit, path) or (None, None, None) if all fail
    """
    import concurrent.futures
    from geodude.collision import GraspAwareCollisionChecker
    from pycbirrt import CBiRRT, CBiRRTConfig

    # Transit heights to try in parallel (absolute z-coordinates)
    transit_heights = [1.15, 1.1, 1.05, 1.0, 0.95]

    # Sample a place pose to get x,y location and orientation
    sample_pose = place_tsr.sample()

    # Capture current arm joint positions (to start planning from)
    q_start = arm.get_joint_positions().copy()

    # Pre-create environment forks for parallel planning
    # Each fork has its own MjData instance, enabling true thread parallelism
    print_status(f"   Creating {len(transit_heights)} planning forks...")
    forks = robot.env.fork(n=len(transit_heights))

    def plan_from_height(args):
        """Worker function: try planning from a specific transit height using a fork."""
        height, fork = args

        # Create transit pose at this height (same x,y,orientation as place)
        transit_pose = sample_pose.copy()
        transit_pose[2, 3] = height

        # Create a collision checker that uses this fork's data
        checker = GraspAwareCollisionChecker(
            model=fork.model,
            data=fork.data,
            joint_names=arm.config.joint_names,
            grasp_manager=arm.grasp_manager,
        )

        # Check if transit pose is reachable and collision-free using local IK
        from geodude.arm import ArmIKSolver, ArmRobotModel

        # Create arm-like interface for the fork
        class ForkArm:
            """Minimal arm interface for planning on a fork."""
            def __init__(self, arm, fork, checker):
                self._arm = arm
                self._fork = fork
                self._checker = checker
                self.model = fork.model
                self.data = fork.data
                self.config = arm.config
                self.joint_qpos_indices = arm.joint_qpos_indices
                self.ee_site_id = arm.ee_site_id

            def get_joint_positions(self):
                return np.array([self._fork.data.qpos[i] for i in self.joint_qpos_indices])

            def get_joint_limits(self):
                return self._arm.get_joint_limits()

            def _get_ee_pose_at_config(self, q):
                """Compute EE pose without modifying fork state."""
                temp_data = mujoco.MjData(self.model)
                temp_data.qpos[:] = self._fork.data.qpos
                for i, qpos_idx in enumerate(self.joint_qpos_indices):
                    temp_data.qpos[qpos_idx] = q[i]
                mujoco.mj_forward(self.model, temp_data)
                pos = temp_data.site_xpos[self.ee_site_id]
                rot_mat = temp_data.site_xmat[self.ee_site_id].reshape(3, 3)
                transform = np.eye(4)
                transform[:3, :3] = rot_mat
                transform[:3, 3] = pos
                return transform

            def inverse_kinematics(self, pose, validate=True, sort_by_distance=True):
                # Use fork-local IK to avoid shared state issues
                # Create a fresh EAIK solver for this fork
                from pycbirrt.backends.eaik import EAIKSolver
                if not hasattr(self, '_local_ik_solver'):
                    self._local_ik_solver = EAIKSolver.for_ur5e(
                        joint_limits=self._arm.get_joint_limits(),
                        collision_checker=self._checker,
                    )

                # Transform pose to EAIK frame (same as arm.inverse_kinematics)
                T_world_base = self._arm.get_base_pose()
                T_mjbase_mujoco = np.linalg.inv(T_world_base) @ pose
                R = self._arm._get_base_rotation()
                R_inv = np.linalg.inv(R)
                T_eaik_mujoco = self._arm._get_ee_offset()
                T_eaikbase_eaik = R_inv @ T_mjbase_mujoco @ np.linalg.inv(T_eaik_mujoco)

                if validate:
                    solutions = self._local_ik_solver.solve_valid(T_eaikbase_eaik)
                else:
                    solutions = self._local_ik_solver.solve(T_eaikbase_eaik)

                if sort_by_distance and solutions:
                    q_current = self.get_joint_positions()
                    solutions = sorted(solutions, key=lambda q: np.linalg.norm(q - q_current))

                return solutions if solutions else []

            def _get_collision_checker(self):
                return self._checker

            @property
            def dof(self):
                return self._arm.dof

        fork_arm = ForkArm(arm, fork, checker)

        # Set the fork to current arm state
        for i, qpos_idx in enumerate(arm.joint_qpos_indices):
            fork.data.qpos[qpos_idx] = q_start[i]
        mujoco.mj_forward(fork.model, fork.data)

        # Check if transit pose is reachable via IK (use fork-local IK)
        q_transit_solutions = fork_arm.inverse_kinematics(transit_pose, validate=False, sort_by_distance=True)
        q_transit = None
        for sol in (q_transit_solutions or []):
            if checker.is_valid(sol):
                q_transit = sol
                break

        if q_transit is None:
            return (height, None, None)

        # Create planner with fork-local collision checker
        robot_model = ArmRobotModel(fork_arm)
        ik_solver = ArmIKSolver(fork_arm)

        planner_config = CBiRRTConfig(
            max_iterations=5000,
            step_size=0.1,
            goal_bias=0.1,
            ik_num_seeds=1,
            timeout=timeout / len(transit_heights),
            smoothing_iterations=100,
        )

        planner = CBiRRT(
            robot=robot_model,
            ik_solver=ik_solver,
            collision_checker=checker,
            config=planner_config,
        )

        # Plan from current position to place TSR
        try:
            path = planner.plan(
                q_start,
                goal_tsrs=[place_tsr],
                seed=int(height * 100),
            )
            if path is not None:
                return (height, q_transit, path)
        except Exception as e:
            pass

        return (height, q_transit, None)

    # Use ThreadPoolExecutor for true parallel planning
    # Each worker has its own fork with independent MjData
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(transit_heights)) as executor:
        # Submit all planning tasks
        futures = {
            executor.submit(plan_from_height, (h, f)): h
            for h, f in zip(transit_heights, forks)
        }

        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                height, q_transit, path = future.result()
                if path is not None:
                    print_status(f"   Fork at z={height} found path with {len(path)} waypoints!")
                    # Cancel remaining futures (best effort)
                    for f in futures:
                        f.cancel()
                    return height, q_transit, path
                elif q_transit is not None:
                    print_status(f"   Fork at z={height}: transit valid but no path found")
                else:
                    print_status(f"   Fork at z={height}: transit pose unreachable")
            except Exception as e:
                print_status(f"   Fork error: {e}")

    return None, None, None


def run_kinematic_demo(robot, viewer):
    """Run manipulation demo with kinematic execution."""
    print_status("\n=== Kinematic Manipulation Demo ===")
    print_status("Objects move via attachment (no physics)")

    # Set seed for reproducibility
    np.random.seed(42)

    model = robot.model
    data = robot.data
    arm = robot.right_arm

    # Use robot's existing grasp manager
    grasp_manager = robot.grasp_manager

    gripper_body_names = [
        "right_ur5e/gripper/right_follower",
        "right_ur5e/gripper/left_follower",
        "right_ur5e/gripper/right_pad",
        "right_ur5e/gripper/left_pad",
    ]

    gripper = Gripper(
        model=model,
        data=data,
        arm_name="right",
        actuator_name="right_ur5e/gripper/fingers_actuator",
        gripper_body_names=gripper_body_names,
        grasp_manager=grasp_manager,
        gripper_site_name="right_ur5e/gripper_attachment_site",
    )

    # Create kinematic executor with grasp manager
    executor = KinematicExecutor(
        model=model,
        data=data,
        joint_qpos_indices=arm.joint_qpos_indices,
        control_dt=0.016,  # 60 Hz for smoother visualization
        viewer=viewer,
        grasp_manager=grasp_manager,
    )

    # Set candidate objects for grasp detection
    gripper.set_candidate_objects(["can"])

    # Get object initial position
    initial_pos = get_object_position(data)
    print_status(f"Object initial position: {initial_pos}")

    # Go to ready pose
    print_status("\n1. Moving to ready pose...")
    robot.go_to("ready")
    viewer.sync()
    time.sleep(0.5)

    # Create pre-grasp and grasp TSRs
    # Pre-grasp has extra standoff for collision-free planning
    print_status("\n2. Planning path to pre-grasp...")
    pregrasp_tsr, grasp_tsr = load_can_grasp_tsrs(initial_pos, pregrasp_standoff=0.20)
    print_status(f"   Object at: {initial_pos}")

    # Plan to pre-grasp TSR (collision-free path)
    # Debug: check current config validity
    collision_checker = arm._get_collision_checker()
    q_home = arm.get_joint_positions()
    print_status(f"   Home config valid: {collision_checker.is_valid(q_home)}")

    # Debug: sample a pregrasp pose and check IK
    pregrasp_sample = pregrasp_tsr.sample()
    print_status(f"   Pregrasp sample: {pregrasp_sample[:3, 3]}")
    pregrasp_ik = arm.inverse_kinematics(pregrasp_sample, validate=False)
    print_status(f"   Pregrasp IK solutions: {len(pregrasp_ik) if pregrasp_ik else 0}")
    if pregrasp_ik:
        for i, q in enumerate(pregrasp_ik[:3]):
            valid = collision_checker.is_valid(q)
            print_status(f"   IK solution {i} valid: {valid}")
            if not valid:
                collision_checker.debug_contacts(q)

    try:
        path = arm.plan_to_tsrs([pregrasp_tsr], timeout=10.0, seed=42)
    except Exception as e:
        print_status(f"   Planning exception: {e}")
        import traceback
        traceback.print_exc()
        path = None

    if path is None:
        print_status("   Planning to pre-grasp failed!")
        return

    print_status(f"   Found path with {len(path)} waypoints")

    # Convert to trajectory and execute to pre-grasp
    print_status("\n3. Executing to pre-grasp...")
    trajectory = Trajectory.from_path(
        path,
        arm.config.kinematic_limits.velocity,
        arm.config.kinematic_limits.acceleration,
    )

    # Execute trajectory using proper time parameterization
    t = 0.0
    while t <= trajectory.duration:
        pos, vel, acc = trajectory.sample(t)
        executor.set_position(pos)
        viewer.sync()
        time.sleep(executor.control_dt)
        t += executor.control_dt

    time.sleep(0.3)

    # Approach: move from pre-grasp to grasp (straight line)
    print_status("\n4. Approaching grasp pose...")
    q_pregrasp = arm.get_joint_positions()

    # Sample a grasp pose from the TSR and solve IK (no validation - we expect contact)
    grasp_pose = grasp_tsr.sample()
    q_grasp_target = solve_ik(arm, grasp_pose, validate=False)
    if q_grasp_target is None:
        print_status("   IK for grasp pose failed!")
        return

    # Interpolate to grasp (direct approach, may contact object)
    interpolate_to(executor, q_pregrasp, q_grasp_target, steps=30, viewer=viewer)
    time.sleep(0.2)

    # Close gripper and attach object (kinematic - assume grasp succeeds)
    print_status("\n5. Closing gripper...")
    gripper.kinematic_close()
    # In kinematic mode, just attach the object directly
    grasp_manager.mark_grasped("can", "right")
    grasp_manager.attach_object("can", "right_ur5e/gripper/right_follower")
    print_status("   Attached: can")
    viewer.sync()
    time.sleep(0.3)

    # Short unvalidated lift to break contact with support surface
    # This is needed because at grasp pose, the object is still touching the table
    print_status("\n6. Breaking contact with support surface...")
    grasp_ee_pose = arm.get_ee_pose()
    clearance_pose = grasp_ee_pose.copy()
    clearance_pose[2, 3] += 0.02  # Lift 2cm to clear surface
    q_clearance = solve_ik(arm, clearance_pose, validate=False)
    if q_clearance is not None:
        interpolate_to(executor, arm.get_joint_positions(), q_clearance, steps=15, viewer=viewer)
        grasp_manager.update_attached_poses()
        viewer.sync()
    time.sleep(0.2)

    # Plan to place TSR (recycle bin drop)
    print_status("\n7. Planning to place TSR (recycle bin)...")

    # Debug: compare EAIK FK with MuJoCo FK at multiple configurations
    print_status("   === EAIK vs MuJoCo FK Comparison ===")
    ik_solver = arm._get_ik_solver()
    T_world_base = arm.get_base_pose()
    R = arm._get_base_rotation()
    T_eaik_mujoco = arm._get_ee_offset()

    test_configs = [
        np.zeros(6),
        np.array([0.5, -1.0, 1.0, -1.57, -1.57, 0]),
        np.array([-0.5, -1.5, 1.5, -1.0, -1.57, 0.5]),
        arm.get_joint_positions(),  # current config
    ]
    config_names = ["q=0", "test1", "test2", "current"]

    for name, q in zip(config_names, test_configs):
        # EAIK FK
        T_eaikbase_eaik = ik_solver.forward_kinematics(q)
        T_mjbase_eaik = R @ T_eaikbase_eaik
        T_mjbase_mujoco_eaik = T_mjbase_eaik @ T_eaik_mujoco
        T_world_mujoco_eaik = T_world_base @ T_mjbase_mujoco_eaik
        eaik_pos = T_world_mujoco_eaik[:3, 3]

        # MuJoCo FK
        mj_pos = arm._get_ee_pose_at_config(q)[:3, 3]

        # Compare
        diff = np.linalg.norm(eaik_pos - mj_pos)
        print_status(f"   {name}: MJ={mj_pos.round(3)}, EAIK={eaik_pos.round(3)}, diff={diff:.4f}m")

    print_status("   === End FK Comparison ===")

    place_tsr = load_place_tsr(model, data)

    # Debug: verify TSR samples and IK round-trip
    print_status(f"   TSR T0_w position (bin): {place_tsr.T0_w[:3, 3]}")
    sample = place_tsr.sample()
    print_status(f"   TSR sample pose: {sample[:3, 3]}")

    # Check IK round-trip with detailed frame transformation debug
    print_status("\n   === IK Frame Transformation Debug ===")

    # Get frame transforms
    T_world_base = arm.get_base_pose()
    print_status(f"   T_world_base position: {T_world_base[:3, 3]}")
    print_status(f"   T_world_base rotation:\n{T_world_base[:3, :3].round(3)}")

    # Compute pose in base frame
    T_mjbase_mujoco = np.linalg.inv(T_world_base) @ sample
    print_status(f"   T_mjbase_mujoco position: {T_mjbase_mujoco[:3, 3]}")

    # Get base rotation
    R = arm._get_base_rotation()
    R_inv = np.linalg.inv(R)
    print_status(f"   R:\n{R[:3, :3].round(3)}")

    # Get EE offset
    T_eaik_mujoco = arm._get_ee_offset()
    print_status(f"   T_eaik_mujoco position: {T_eaik_mujoco[:3, 3]}")

    # Compute EAIK target pose
    T_eaikbase_eaik = R_inv @ T_mjbase_mujoco @ np.linalg.inv(T_eaik_mujoco)
    print_status(f"   T_eaikbase_eaik position: {T_eaikbase_eaik[:3, 3]}")

    ik_solutions = arm.inverse_kinematics(sample, validate=False)
    if ik_solutions:
        q_ik = ik_solutions[0]
        ee_from_ik = arm._get_ee_pose_at_config(q_ik)
        print_status(f"   IK solution FK: {ee_from_ik[:3, 3]}")
        print_status(f"   IK error: {np.linalg.norm(ee_from_ik[:3, 3] - sample[:3, 3]):.4f}m")

        # Also check EAIK FK step by step
        ik_solver = arm._get_ik_solver()
        T_eaikbase_eaik_fk = ik_solver.forward_kinematics(q_ik)
        print_status(f"   EAIK FK (eaik base frame): {T_eaikbase_eaik_fk[:3, 3]}")

        T_mjbase_eaik_fk = R @ T_eaikbase_eaik_fk
        print_status(f"   After R rotation (mj base frame): {T_mjbase_eaik_fk[:3, 3]}")

        T_mjbase_mujoco_fk = T_mjbase_eaik_fk @ T_eaik_mujoco
        print_status(f"   After EE offset (mj base frame): {T_mjbase_mujoco_fk[:3, 3]}")

        T_world_mujoco_fk = T_world_base @ T_mjbase_mujoco_fk
        print_status(f"   Final world frame: {T_world_mujoco_fk[:3, 3]}")
        print_status(f"   Target was: {sample[:3, 3]}")
        print_status(f"   EAIK FK error: {np.linalg.norm(T_world_mujoco_fk[:3, 3] - sample[:3, 3]):.4f}m")

        # Double-check: what does MuJoCo think the EE is at this config?
        mj_fk = arm._get_ee_pose_at_config(q_ik)
        print_status(f"   MuJoCo FK at same q: {mj_fk[:3, 3]}")
        print_status(f"   MuJoCo vs EAIK chain diff: {np.linalg.norm(mj_fk[:3, 3] - T_world_mujoco_fk[:3, 3]):.4f}m")
    else:
        print_status("   IK found no solutions!")

    print_status("   === End IK Debug ===")

    # Debug: verify collision checker detects vention base
    collision_checker = arm._get_collision_checker()

    # Test current config (should be valid)
    q_current = arm.get_joint_positions()
    is_valid_now = collision_checker.is_valid(q_current)
    print_status(f"   Current config valid: {is_valid_now}")
    if not is_valid_now:
        print_status("   DEBUG: Current config contacts (blocking planning):")
        collision_checker.debug_contacts(q_current)

    # Debug: check what bodies are in the arm set
    print_status(f"   Arm body IDs: {len(collision_checker._arm_body_ids)} bodies")

    # Use direct planning (parallel planning has collision checking issues with vention base)
    try:
        path = arm.plan_to_tsrs([place_tsr], timeout=30.0, seed=42)
    except Exception as e:
        print_status(f"   Planning exception: {e}")
        import traceback
        traceback.print_exc()
        path = None

    # Debug: check if the path endpoint is valid
    if path is not None:
        q_end = path[-1]
        is_end_valid = collision_checker.is_valid(q_end)
        print_status(f"   Path endpoint valid: {is_end_valid}")
        if not is_end_valid:
            print_status("   DEBUG: Path endpoint contacts:")
            collision_checker.debug_contacts(q_end)

        # Check path validity along the way
        invalid_waypoints = []
        for i, q in enumerate(path):
            if not collision_checker.is_valid(q):
                invalid_waypoints.append(i)
        if invalid_waypoints:
            print_status(f"   WARNING: {len(invalid_waypoints)} invalid waypoints in path!")
            print_status(f"   Invalid waypoint indices: {invalid_waypoints[:10]}...")
            # Debug first invalid waypoint
            first_invalid = invalid_waypoints[0]
            print_status(f"   DEBUG: First invalid waypoint ({first_invalid}) contacts:")
            collision_checker.debug_contacts(path[first_invalid])

    if path is None:
        print_status("   Planning to place TSR failed!")
        return

    print_status(f"   Found path with {len(path)} waypoints")

    # Debug: where should the path end?
    final_q = path[-1]
    expected_ee = arm._get_ee_pose_at_config(final_q)
    print_status(f"   Path endpoint EE: {expected_ee[:3, 3]}")
    print_status(f"   Bin is at: [0.75, -0.35, 0.50], expect EE at ~[0.75, -0.35, 1.0]")

    execute_path(arm, path, executor, viewer)

    # Debug: where did we actually end up?
    actual_ee = arm.get_ee_pose()
    print_status(f"   Actual EE after execution: {actual_ee[:3, 3]}")

    # Debug: check if final config is in collision
    q_final = arm.get_joint_positions()
    is_final_valid = collision_checker.is_valid(q_final)
    print_status(f"   Final config valid: {is_final_valid}")
    if not is_final_valid:
        print_status("   DEBUG: Final config contacts:")
        collision_checker.debug_contacts(q_final)

    time.sleep(0.3)

    # Release
    print_status("\n8. Opening gripper...")
    gripper.kinematic_open()
    grasp_manager.mark_released("can")
    grasp_manager.detach_object("can")
    viewer.sync()
    time.sleep(0.3)

    # Retract up
    print_status("\n9. Retracting...")
    current_ee_pose = arm.get_ee_pose()
    retract_pose = current_ee_pose.copy()
    retract_pose[2, 3] += 0.15  # Lift 15cm
    q_retract = solve_ik(arm, retract_pose)
    if q_retract is None:
        print_status("   IK failed for retract pose!")
        return

    path = arm.plan_to_configuration(q_retract, timeout=10.0)
    if not path:
        print_status("   Planning failed!")
        return
    print_status(f"   Planned path with {len(path)} waypoints")
    execute_path(arm, path, executor, viewer)

    final_pos = get_object_position(data)
    print_status(f"\nObject final position: {final_pos}")
    print_status(f"Total displacement: {np.linalg.norm(final_pos - initial_pos):.3f}m")

    print_status("\n=== Kinematic Demo Complete ===")


def run_physics_demo(robot, viewer):
    """Run manipulation demo with physics-based execution."""
    print_status("\n=== Physics-Based Manipulation Demo ===")
    print_status("Objects held by friction (full dynamics)")

    model = robot.env.model
    data = robot.env.data
    arm = robot.right_arm

    # Create grasp manager (for collision group management)
    grasp_manager = GraspManager(model, data)

    gripper_body_names = [
        "right_ur5e/gripper/right_follower",
        "right_ur5e/gripper/left_follower",
        "right_ur5e/gripper/right_pad",
        "right_ur5e/gripper/left_pad",
    ]

    gripper = Gripper(
        model=model,
        data=data,
        arm_name="right",
        actuator_name="right_ur5e/gripper/fingers_actuator",
        gripper_body_names=gripper_body_names,
        grasp_manager=grasp_manager,
    )

    # Create physics executor
    joint_qpos_indices = [model.jnt_qposadr[model.joint(name).id] for name in [
        "right_ur5e/shoulder_pan_joint",
        "right_ur5e/shoulder_lift_joint",
        "right_ur5e/elbow_joint",
        "right_ur5e/wrist_1_joint",
        "right_ur5e/wrist_2_joint",
        "right_ur5e/wrist_3_joint",
    ]]

    actuator_ids = [model.actuator(name).id for name in [
        "right_ur5e/shoulder_pan_actuator",
        "right_ur5e/shoulder_lift_actuator",
        "right_ur5e/elbow_actuator",
        "right_ur5e/wrist_1_actuator",
        "right_ur5e/wrist_2_actuator",
        "right_ur5e/wrist_3_actuator",
    ]]

    executor = PhysicsExecutor(
        model=model,
        data=data,
        joint_qpos_indices=joint_qpos_indices,
        actuator_ids=actuator_ids,
        control_dt=0.008,
        viewer=viewer,
    )

    # Set candidate objects
    gripper.set_candidate_objects(["can"])

    # Get object initial position
    initial_pos = get_object_position(data)
    print_status(f"Object initial position: {initial_pos}")

    # Go to ready pose
    print_status("\n1. Moving to ready pose...")
    robot.go_to("ready")
    settle_physics(executor, steps=100)
    time.sleep(0.3)

    # Move gripper above object
    print_status("\n2. Moving above object...")
    q_above = np.array([-0.8, -1.2, 1.8, -2.2, -1.57, 0.0])
    move_physics(executor, q_above, steps=200)
    time.sleep(0.2)

    # Move down to grasp position
    print_status("\n3. Moving down to grasp...")
    q_grasp = np.array([-0.8, -1.0, 1.6, -2.2, -1.57, 0.0])
    move_physics(executor, q_grasp, steps=150)
    time.sleep(0.2)

    # Close gripper (physics - uses contacts)
    print_status("\n4. Closing gripper...")
    grasped = gripper.close(steps=200)
    viewer.sync()
    if grasped:
        print_status(f"   Grasped: {grasped}")
    else:
        print_status("   No object detected in contact")
    time.sleep(0.2)

    # Lift up
    print_status("\n5. Lifting object...")
    q_lift = np.array([-0.8, -1.5, 1.5, -1.5, -1.57, 0.0])
    move_physics(executor, q_lift, steps=200)

    lifted_pos = get_object_position(data)
    print_status(f"   Object position after lift: {lifted_pos}")
    print_status(f"   Height change: {lifted_pos[2] - initial_pos[2]:.3f}m")
    time.sleep(0.3)

    # Move to side
    print_status("\n6. Moving to side...")
    q_side = np.array([-0.3, -1.5, 1.5, -1.5, -1.57, 0.0])
    move_physics(executor, q_side, steps=200)
    time.sleep(0.2)

    # Lower
    print_status("\n7. Lowering object...")
    q_place = np.array([-0.3, -1.0, 1.6, -2.2, -1.57, 0.0])
    move_physics(executor, q_place, steps=150)
    time.sleep(0.2)

    # Release
    print_status("\n8. Opening gripper...")
    gripper.open(steps=100)
    viewer.sync()
    time.sleep(0.2)

    # Move away
    print_status("\n9. Retracting...")
    q_retract = np.array([-0.3, -1.5, 1.5, -1.5, -1.57, 0.0])
    move_physics(executor, q_retract, steps=150)

    # Let object settle
    settle_physics(executor, steps=200)

    final_pos = get_object_position(data)
    print_status(f"\nObject final position: {final_pos}")
    print_status(f"Total displacement: {np.linalg.norm(final_pos - initial_pos):.3f}m")

    print_status("\n=== Physics Demo Complete ===")


def interpolate_to(executor, q_start, q_end, steps=50, viewer=None):
    """Interpolate between configurations (kinematic)."""
    for i in range(steps + 1):
        t = i / steps
        q = q_start + t * (q_end - q_start)
        executor.set_position(q)
        if viewer:
            viewer.sync()
        time.sleep(executor.control_dt)


def execute_path(arm, path, executor, viewer):
    """Execute a planned path with proper time parameterization."""
    trajectory = Trajectory.from_path(
        path,
        arm.config.kinematic_limits.velocity,
        arm.config.kinematic_limits.acceleration,
    )
    t = 0.0
    while t <= trajectory.duration:
        pos, vel, acc = trajectory.sample(t)
        executor.set_position(pos)
        if viewer:
            viewer.sync()
        time.sleep(executor.control_dt)
        t += executor.control_dt


def move_physics(executor, q_target, steps=100):
    """Move to target using physics executor."""
    executor.set_target(q_target)
    for _ in range(steps):
        executor.step()
        time.sleep(0.001)  # Small delay for visualization


def settle_physics(executor, steps=100):
    """Let physics settle at current position."""
    executor.hold()
    for _ in range(steps):
        executor.step()


def main():
    parser = argparse.ArgumentParser(description="Manipulation demo")
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Use physics-based execution (default: kinematic)",
    )
    args = parser.parse_args()

    # Load scene with object
    xml_path = create_scene_with_object()

    # Create robot interface from modified XML
    robot = Geodude.from_xml(xml_path)
    model = robot.model
    data = robot.data

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera view
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0.4, 0, 0.8]

        mujoco.mj_forward(model, data)
        viewer.sync()

        print_status("Manipulation Demo")
        print_status("=" * 40)
        mode = "Physics" if args.physics else "Kinematic"
        print_status(f"Mode: {mode}")
        print_status("Press Ctrl+C to exit\n")

        time.sleep(1.0)

        try:
            if args.physics:
                run_physics_demo(robot, viewer)
            else:
                run_kinematic_demo(robot, viewer)

            # Keep viewer open
            print_status("\nDemo complete. Close viewer to exit.")
            while viewer.is_running():
                time.sleep(0.1)

        except KeyboardInterrupt:
            print_status("\nInterrupted")


if __name__ == "__main__":
    main()
