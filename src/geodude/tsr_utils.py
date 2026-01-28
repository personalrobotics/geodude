"""TSR utilities for grasp and place planning.

This module provides helper functions to create Task Space Regions (TSRs) for
common manipulation tasks like grasping and placing objects.

It also handles gripper-specific frame compensation. The TSR library uses a
canonical gripper convention (z = approach, y = finger opening), but specific
grippers may have different internal frame conventions that require compensation.
"""

from __future__ import annotations

import numpy as np

# Optional TSR import
try:
    from tsr import TSR

    TSR_AVAILABLE = True
except ImportError:
    TSR_AVAILABLE = False


# =============================================================================
# Gripper Frame Compensation
# =============================================================================
#
# The TSR library uses a canonical gripper frame convention:
#   - z-axis: approach direction (where the gripper "looks")
#   - y-axis: finger opening direction
#   - x-axis: perpendicular (by right-hand rule)
#
# Specific grippers may have different internal conventions. For example,
# the Robotiq 2F-140 MuJoCo model has an internal -90° z rotation on its
# base body (see geodude_assets issue #5). This means poses computed with
# the canonical convention need to be rotated before being used with IK.
#
# The compensation is applied as: adjusted_pose = canonical_pose @ R_offset
# where R_offset converts from canonical frame to the gripper's mount frame.

def _rotation_z(angle_rad: float) -> np.ndarray:
    """Create a 4x4 rotation matrix around z-axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


# Known gripper frame offsets (from canonical TSR convention to mount frame)
# Each entry is a 4x4 rotation matrix to post-multiply with canonical poses.
GRIPPER_FRAME_OFFSETS: dict[str, np.ndarray] = {
    # Robotiq 2F-140: model has -90° z rotation internally, so we apply +90°
    # to convert from canonical (y=fingers) to mount frame (−x=fingers)
    "robotiq_2f140": _rotation_z(np.pi / 2),
}


def get_gripper_frame_offset(gripper_name: str) -> np.ndarray:
    """Get the frame compensation matrix for a specific gripper.

    The returned matrix should be post-multiplied with canonical TSR poses
    to get poses suitable for the gripper's mount frame:
        adjusted_pose = canonical_pose @ offset

    Args:
        gripper_name: Name of the gripper (e.g., "robotiq_2f140")

    Returns:
        4x4 rotation matrix for frame compensation (identity if unknown gripper)
    """
    return GRIPPER_FRAME_OFFSETS.get(gripper_name, np.eye(4))


def apply_gripper_frame_compensation(
    pose: np.ndarray,
    gripper_name: str,
) -> np.ndarray:
    """Apply gripper-specific frame compensation to a pose.

    Converts a pose from the canonical TSR convention (z=approach, y=fingers)
    to the specific gripper's mount frame convention.

    Args:
        pose: 4x4 homogeneous transform in canonical TSR convention
        gripper_name: Name of the gripper (e.g., "robotiq_2f140")

    Returns:
        4x4 homogeneous transform adjusted for the gripper's frame convention
    """
    offset = get_gripper_frame_offset(gripper_name)
    return pose @ offset


def compensate_tsr_for_gripper(
    tsr: "TSR",
    gripper_name: str,
) -> "TSR":
    """Create a new TSR with gripper-specific frame compensation applied to Tw_e.

    This modifies the Tw_e transform to account for the gripper's internal
    frame convention, so that sampled poses work correctly with IK.

    Args:
        tsr: Original TSR with canonical convention
        gripper_name: Name of the gripper (e.g., "robotiq_2f140")

    Returns:
        New TSR with compensated Tw_e
    """
    if not TSR_AVAILABLE:
        raise ImportError("TSR not available. Install with: pip install tsr")

    offset = get_gripper_frame_offset(gripper_name)

    # Apply compensation to Tw_e: new_Tw_e = Tw_e @ offset
    # This rotates the end-effector frame relative to the TSR frame
    compensated_Tw_e = tsr.Tw_e @ offset

    return TSR(
        T0_w=tsr.T0_w.copy(),
        Tw_e=compensated_Tw_e,
        Bw=tsr.Bw.copy(),
    )


def create_top_grasp_tsr(
    object_pose: np.ndarray,
    object_height: float,
    gripper_standoff: float = 0.15,
    position_tolerance: tuple[float, float, float] = (0.02, 0.02, 0.05),
    allow_any_yaw: bool = True,
) -> "TSR":
    """Create a TSR for top-down grasping of an object.

    The TSR defines valid gripper poses for approaching an object from above.
    The gripper's z-axis points down toward the object.

    Args:
        object_pose: 4x4 transform of the object center in world frame
        object_height: Height of the object (for calculating approach height)
        gripper_standoff: Distance from gripper attachment site to object top
        position_tolerance: (x, y, z) tolerance for gripper position
        allow_any_yaw: If True, allow any rotation around vertical axis

    Returns:
        TSR defining valid top-down grasp poses
    """
    if not TSR_AVAILABLE:
        raise ImportError("TSR not available. Install with: pip install tsr")

    # Gripper approaches from above
    # The gripper frame has z pointing down (180° rotation around x from world frame)
    Tw_e = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, gripper_standoff + object_height / 2],
        [0, 0, 0, 1],
    ])

    # Position bounds
    tol_x, tol_y, tol_z = position_tolerance
    yaw_bound = np.pi if allow_any_yaw else 0.01

    Bw = np.array([
        [-tol_x, tol_x],      # x tolerance
        [-tol_y, tol_y],      # y tolerance
        [0, tol_z],           # z: can be higher (standoff margin)
        [-0.01, 0.01],        # roll: small tolerance
        [-0.01, 0.01],        # pitch: small tolerance
        [-yaw_bound, yaw_bound],  # yaw: any rotation or small tolerance
    ])

    return TSR(
        T0_w=object_pose,
        Tw_e=Tw_e,
        Bw=Bw,
    )


def create_side_grasp_tsr(
    object_pose: np.ndarray,
    object_width: float,
    gripper_standoff: float = 0.15,
    approach_axis: str = "y",
    position_tolerance: tuple[float, float, float] = (0.02, 0.02, 0.02),
) -> "TSR":
    """Create a TSR for side grasping of an object.

    The TSR defines valid gripper poses for approaching an object from the side.

    Args:
        object_pose: 4x4 transform of the object center in world frame
        object_width: Width of the object along the approach axis
        gripper_standoff: Distance from gripper attachment site to object surface
        approach_axis: Axis along which to approach ("x", "y", "-x", "-y")
        position_tolerance: (x, y, z) tolerance for gripper position

    Returns:
        TSR defining valid side grasp poses
    """
    if not TSR_AVAILABLE:
        raise ImportError("TSR not available. Install with: pip install tsr")

    # Determine approach direction and rotation
    total_offset = gripper_standoff + object_width / 2

    if approach_axis == "y":
        # Approach from +y direction: gripper at +y, gripper z points toward -y
        Tw_e = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, total_offset],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ])
    elif approach_axis == "-y":
        # Approach from -y direction: gripper at -y, gripper z points toward +y
        Tw_e = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, -total_offset],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
    elif approach_axis == "x":
        # Approach from +x direction: gripper at +x, gripper z points toward -x
        Tw_e = np.array([
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, total_offset],
            [0, 0, 0, 1],
        ])
    elif approach_axis == "-x":
        # Approach from -x direction: gripper at -x, gripper z points toward +x
        Tw_e = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, -total_offset],
            [0, 0, 0, 1],
        ])
    else:
        raise ValueError(f"Invalid approach_axis: {approach_axis}")

    tol_x, tol_y, tol_z = position_tolerance

    Bw = np.array([
        [-tol_x, tol_x],
        [-tol_y, tol_y],
        [-tol_z, tol_z],
        [-0.1, 0.1],  # Small rotation tolerance
        [-0.1, 0.1],
        [-0.1, 0.1],
    ])

    return TSR(
        T0_w=object_pose,
        Tw_e=Tw_e,
        Bw=Bw,
    )


def create_place_tsr(
    surface_pose: np.ndarray,
    surface_height: float,
    object_height: float,
    gripper_standoff: float = 0.15,
    position_tolerance: tuple[float, float] = (0.1, 0.1),
    allow_any_yaw: bool = True,
) -> "TSR":
    """Create a TSR for placing an object on a surface.

    The TSR defines valid gripper poses for placing an object from above
    onto a surface (like a table).

    Args:
        surface_pose: 4x4 transform of the surface center in world frame
        surface_height: Height of the surface (z offset from surface_pose origin)
        object_height: Height of the object being placed
        gripper_standoff: Distance from gripper attachment site to object bottom
        position_tolerance: (x, y) tolerance for placement position
        allow_any_yaw: If True, allow any rotation around vertical axis

    Returns:
        TSR defining valid placement poses
    """
    if not TSR_AVAILABLE:
        raise ImportError("TSR not available. Install with: pip install tsr")

    # Place from above, gripper z points down
    # The gripper should be positioned so the object bottom is at surface height
    z_offset = surface_height + object_height / 2 + gripper_standoff

    Tw_e = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, z_offset],
        [0, 0, 0, 1],
    ])

    tol_x, tol_y = position_tolerance
    yaw_bound = np.pi if allow_any_yaw else 0.01

    Bw = np.array([
        [-tol_x, tol_x],  # x tolerance on surface
        [-tol_y, tol_y],  # y tolerance on surface
        [0, 0.02],        # Small z margin
        [-0.01, 0.01],    # roll
        [-0.01, 0.01],    # pitch
        [-yaw_bound, yaw_bound],  # yaw
    ])

    return TSR(
        T0_w=surface_pose,
        Tw_e=Tw_e,
        Bw=Bw,
    )


def create_lift_tsr(
    current_ee_pose: np.ndarray,
    lift_height: float = 0.1,
    tolerance: float = 0.02,
) -> "TSR":
    """Create a TSR for lifting the gripper straight up.

    This is useful after grasping an object to lift it clear of the surface.

    Args:
        current_ee_pose: Current 4x4 transform of the end-effector
        lift_height: Height to lift above current position
        tolerance: Position tolerance

    Returns:
        TSR defining the lifted pose region
    """
    if not TSR_AVAILABLE:
        raise ImportError("TSR not available. Install with: pip install tsr")

    # Create a TSR at the lifted position
    lifted_pose = current_ee_pose.copy()
    lifted_pose[2, 3] += lift_height

    Bw = np.array([
        [-tolerance, tolerance],
        [-tolerance, tolerance],
        [-tolerance, tolerance],
        [-0.05, 0.05],  # Small rotation tolerance
        [-0.05, 0.05],
        [-0.05, 0.05],
    ])

    return TSR(
        T0_w=lifted_pose,
        Tw_e=np.eye(4),
        Bw=Bw,
    )


def create_retract_tsr(
    current_ee_pose: np.ndarray,
    retract_distance: float = 0.1,
    tolerance: float = 0.02,
) -> "TSR":
    """Create a TSR for retracting the gripper away from current position.

    Retracts along the gripper's approach direction (negative z in gripper frame).

    Args:
        current_ee_pose: Current 4x4 transform of the end-effector
        retract_distance: Distance to retract
        tolerance: Position tolerance

    Returns:
        TSR defining the retracted pose region
    """
    if not TSR_AVAILABLE:
        raise ImportError("TSR not available. Install with: pip install tsr")

    # Retract along gripper's z-axis (approach direction)
    retract_pose = current_ee_pose.copy()
    # The gripper's z-axis is the third column of the rotation matrix
    z_axis = current_ee_pose[:3, 2]
    retract_pose[:3, 3] -= retract_distance * z_axis

    Bw = np.array([
        [-tolerance, tolerance],
        [-tolerance, tolerance],
        [-tolerance, tolerance],
        [-0.05, 0.05],
        [-0.05, 0.05],
        [-0.05, 0.05],
    ])

    return TSR(
        T0_w=retract_pose,
        Tw_e=np.eye(4),
        Bw=Bw,
    )


def create_approach_tsr(
    target_ee_pose: np.ndarray,
    approach_distance: float = 0.1,
    tolerance: float = 0.02,
) -> "TSR":
    """Create a TSR for the approach pose before a grasp/place.

    The approach pose is positioned back along the gripper's approach direction.

    Args:
        target_ee_pose: Target 4x4 transform of the end-effector (grasp/place pose)
        approach_distance: Distance to position before target
        tolerance: Position tolerance

    Returns:
        TSR defining the approach pose region
    """
    if not TSR_AVAILABLE:
        raise ImportError("TSR not available. Install with: pip install tsr")

    # Position back along gripper's z-axis
    approach_pose = target_ee_pose.copy()
    z_axis = target_ee_pose[:3, 2]
    approach_pose[:3, 3] -= approach_distance * z_axis

    Bw = np.array([
        [-tolerance, tolerance],
        [-tolerance, tolerance],
        [-tolerance, tolerance],
        [-0.05, 0.05],
        [-0.05, 0.05],
        [-0.05, 0.05],
    ])

    return TSR(
        T0_w=approach_pose,
        Tw_e=np.eye(4),
        Bw=Bw,
    )
