#!/usr/bin/env python3
"""Visualize TSR templates with object and gripper.

A general-purpose tool for visualizing any TSR template - grasps, placements,
or other manipulation tasks. Shows the canonical Bw=0 pose and optionally
samples additional poses from the TSR.

Displays:
- Reference object (cylinder, box, etc.) - the thing being grasped/placed
- Subject (gripper or object) at TSR pose(s)
- Coordinate frame axes: Red=X, Green=Y, Blue=Z, Yellow ball=origin

Usage:
    uv run mjpython examples/visualize_tsr.py <template_path> [options]

Examples:
    # Visualize can side grasp
    uv run mjpython examples/visualize_tsr.py tsr/grasps/can_side_grasp.yaml

    # Show 5 sampled poses
    uv run mjpython examples/visualize_tsr.py tsr/grasps/can_side_grasp.yaml --samples 5

    # Don't apply gripper compensation (show canonical frame)
    uv run mjpython examples/visualize_tsr.py tsr/grasps/can_side_grasp.yaml --no-compensation
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from tsr import TSR
from tsr.core.tsr_primitive import load_template_file
from geodude.tsr_utils import apply_gripper_frame_compensation

# Import geodude_assets for gripper model paths
try:
    from geodude_assets import get_model_path
    GEODUDE_ASSETS_AVAILABLE = True
except ImportError:
    GEODUDE_ASSETS_AVAILABLE = False
    get_model_path = None

# Import prl_assets for object model paths
try:
    from prl_assets import OBJECTS_DIR
    from asset_manager import AssetManager
    PRL_ASSETS_AVAILABLE = True
    _prl_asset_manager = None
except ImportError:
    PRL_ASSETS_AVAILABLE = False
    OBJECTS_DIR = None
    AssetManager = None
    _prl_asset_manager = None


def get_prl_asset_manager():
    """Get or create the prl_assets AssetManager singleton."""
    global _prl_asset_manager
    if _prl_asset_manager is None and PRL_ASSETS_AVAILABLE:
        _prl_asset_manager = AssetManager(str(OBJECTS_DIR))
    return _prl_asset_manager


def get_object_xml_path(name: str) -> Path | None:
    """Get the MuJoCo XML path for a prl_assets object."""
    manager = get_prl_asset_manager()
    if manager is None:
        return None
    path = manager.get_path(name, "mujoco")
    return Path(path) if path else None


# Default object dimensions for common references (fallback if asset not found)
OBJECT_DEFAULTS = {
    "can": {"type": "cylinder", "radius": 0.033, "height": 0.123},
    "bottle": {"type": "cylinder", "radius": 0.035, "height": 0.20},
    "mug": {"type": "cylinder", "radius": 0.04, "height": 0.10},
    "box": {"type": "box", "size": [0.05, 0.05, 0.05]},
    "table": {"type": "box", "size": [0.5, 0.5, 0.02]},
    "worktop": {"type": "box", "size": [1.2, 0.8, 0.002], "is_surface": True},  # worktop - thin for viz, origin on surface
}

# Map template subject names to (model_name, xml_filename) tuples for grippers
# xml_filename is the actual gripper XML (not scene.xml which has demo stuff)
GRIPPER_TO_MODEL = {
    "robotiq_2f140": ("robotiq_2f140", "2f140.xml"),
}


def inject_freejoint_into_gripper_xml(gripper_xml_path: Path) -> str:
    """Read gripper XML and inject a freejoint into base_mount.

    The gripper XML has base_mount with no joint, which means we can't
    reposition it in MuJoCo. This function injects a freejoint to make
    the gripper positionable.

    Args:
        gripper_xml_path: Path to the gripper XML file

    Returns:
        Modified XML string with freejoint added
    """
    import re

    xml_content = gripper_xml_path.read_text()

    # Find the base_mount body tag and inject a freejoint as its first child
    # Pattern matches: <body name="base_mount" ... >
    pattern = r'(<body\s+name="base_mount"[^>]*>)'

    def add_freejoint(match):
        body_tag = match.group(1)
        return body_tag + '\n      <freejoint name="gripper_freejoint"/>'

    modified_xml = re.sub(pattern, add_freejoint, xml_content)

    return modified_xml


def create_scene_xml(
    object_type: str = "cylinder",
    object_size: dict = None,
    num_frames: int = 2,
    include_files: list[str] = None,
    use_ref_object_include: bool = False,
    use_subject_include: bool = False,
    ref_is_surface: bool = False,
) -> str:
    """Create MuJoCo XML for visualization scene.

    Args:
        object_type: "cylinder" or "box" (for fallback geometry)
        object_size: Size parameters (radius/height for cylinder, size for box)
        num_frames: Number of coordinate frames to create
        include_files: List of XML files to include (for gripper, objects)
        use_ref_object_include: If True, reference object is loaded via include
        use_subject_include: If True, subject is loaded via include
        ref_is_surface: If True, reference is a surface (top at z=0); else object (bottom at z=0)

    Returns:
        MuJoCo XML string
    """
    object_size = object_size or {"radius": 0.033, "height": 0.123}
    include_files = include_files or []

    # Object geometry (fallback if not using include)
    ref_object_section = ""
    if not use_ref_object_include:
        if object_type == "cylinder":
            r = object_size.get("radius", 0.033)
            h = object_size.get("height", 0.123)
            ref_object_section = f'''
        <body name="ref_object_body" pos="0 0 0">
            <geom name="ref_object" type="cylinder" size="{r} {h/2}" pos="0 0 {h/2}" rgba="0.8 0.2 0.2 0.7" contype="0" conaffinity="0"/>
        </body>'''
        else:
            size = object_size.get("size", [0.05, 0.05, 0.05])
            sx, sy, sz = size[0]/2, size[1]/2, size[2]/2
            # For surfaces (worktop, table): top at z=0; for objects: bottom at z=0
            z_offset = -sz if ref_is_surface else sz
            ref_object_section = f'''
        <body name="ref_object_body" pos="0 0 0">
            <geom name="ref_object" type="box" size="{sx} {sy} {sz}" pos="0 0 {z_offset}" rgba="0.8 0.2 0.2 0.7" contype="0" conaffinity="0"/>
        </body>'''

    # Axis visualization
    axis_geoms = ""
    for i in range(num_frames):
        prefix = f"frame_{i}"
        axis_geoms += f'''
        <body name="{prefix}" pos="0 0 0">
            <freejoint name="{prefix}_joint"/>
            <geom name="{prefix}_origin" type="sphere" size="0.008" rgba="1 1 0 1" contype="0" conaffinity="0"/>
            <geom name="{prefix}_x" type="cylinder" size="0.003 0.03" pos="0.03 0 0" euler="0 1.5708 0" rgba="1 0 0 1" contype="0" conaffinity="0"/>
            <geom name="{prefix}_y" type="cylinder" size="0.003 0.03" pos="0 0.03 0" euler="1.5708 0 0" rgba="0 1 0 1" contype="0" conaffinity="0"/>
            <geom name="{prefix}_z" type="cylinder" size="0.003 0.03" pos="0 0 0.03" rgba="0 0 1 1" contype="0" conaffinity="0"/>
        </body>'''

    # Include sections
    includes = "\n    ".join(f'<include file="{f}"/>' for f in include_files)

    xml = f'''<mujoco model="tsr_visualization">
    <compiler angle="radian"/>
    <option gravity="0 0 0"/>

    <visual>
        <headlight ambient="0.5 0.5 0.5"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.4 0.4 0.4" rgb2="0.3 0.3 0.3" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    </asset>

    {includes}

    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <light pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5"/>

        <geom name="floor" type="plane" size="1 1 0.1" material="grid" pos="0 0 -0.01"/>
        {ref_object_section}
        {axis_geoms}
    </worldbody>
</mujoco>'''

    return xml


def pose_to_pos_quat(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 4x4 pose matrix to position and quaternion (w,x,y,z)."""
    pos = pose[:3, 3]
    R = pose[:3, :3]

    # Rotation matrix to quaternion
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z])
    quat = quat / np.linalg.norm(quat)
    return pos, quat


def set_body_pose(model, data, body_name: str, pose: np.ndarray) -> bool:
    """Set a freejoint body's pose."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        return False

    joint_id = model.body_jntadr[body_id]
    if joint_id == -1:
        return False

    qpos_adr = model.jnt_qposadr[joint_id]
    pos, quat = pose_to_pos_quat(pose)

    data.qpos[qpos_adr:qpos_adr+3] = pos
    data.qpos[qpos_adr+3:qpos_adr+7] = quat
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize TSR templates")
    parser.add_argument("template", type=str, help="Path to TSR template YAML file")
    parser.add_argument("--samples", type=int, default=0, help="Number of additional poses to sample")
    parser.add_argument("--no-compensation", action="store_true", help="Don't apply gripper frame compensation")
    parser.add_argument("--object-height", type=float, default=None, help="Override object height")
    parser.add_argument("--object-radius", type=float, default=None, help="Override object radius")
    args = parser.parse_args()

    # Load template
    template_path = Path(args.template)
    if not template_path.exists():
        # Try relative to geodude/tsr_templates
        alt_path = Path(__file__).parent.parent / "tsr_templates" / args.template
        if alt_path.exists():
            template_path = alt_path
        else:
            print(f"Template not found: {args.template}")
            return

    print(f"Loading template: {template_path}")
    template = load_template_file(str(template_path))

    print(f"  Name: {template.name}")
    print(f"  Task: {template.task}")
    print(f"  Subject: {template.subject}")
    print(f"  Reference: {template.reference}")

    # Determine reference object properties
    ref = template.reference.lower()
    if ref in OBJECT_DEFAULTS:
        obj_props = OBJECT_DEFAULTS[ref].copy()
    else:
        obj_props = {"type": "cylinder", "radius": 0.033, "height": 0.123}

    # Override from args
    if args.object_height:
        obj_props["height"] = args.object_height
    if args.object_radius:
        obj_props["radius"] = args.object_radius

    # Track what we need to include
    include_files = []
    temp_files = []
    working_dir = Path.cwd()
    use_ref_include = False
    use_subject_include = False
    subject_body_name = None  # Body to position at TSR pose

    # Find reference object model from prl_assets
    ref_xml_path = get_object_xml_path(template.reference)
    if ref_xml_path:
        print(f"  Reference model: {ref_xml_path}")
        # Copy with unique name, use absolute path for include
        ref_temp_name = "_ref_object.xml"
        ref_temp_path = ref_xml_path.parent / ref_temp_name
        ref_temp_path.write_text(ref_xml_path.read_text())
        include_files.append(str(ref_temp_path.absolute()))  # Use absolute path
        temp_files.append(ref_temp_path)
        use_ref_include = True
    else:
        print(f"  Reference: using default {obj_props['type']} geometry")

    # Find subject model (gripper from geodude_assets or object from prl_assets)
    gripper_dir = None
    if template.subject in GRIPPER_TO_MODEL:
        # It's a gripper - load from geodude_assets
        if GEODUDE_ASSETS_AVAILABLE:
            try:
                model_name, xml_filename = GRIPPER_TO_MODEL[template.subject]
                model_path = get_model_path(model_name)
                gripper_dir = model_path.parent
                gripper_xml_path = gripper_dir / xml_filename
                modified_gripper_xml = inject_freejoint_into_gripper_xml(gripper_xml_path)

                gripper_temp_name = "_gripper_modified.xml"
                gripper_temp_path = gripper_dir / gripper_temp_name
                gripper_temp_path.write_text(modified_gripper_xml)
                include_files.append(gripper_temp_name)  # Relative path (scene will be in gripper_dir)
                temp_files.append(gripper_temp_path)
                use_subject_include = True
                subject_body_name = "base_mount"
                working_dir = gripper_dir  # Scene must be in gripper dir for mesh paths
                print(f"  Subject (gripper): {gripper_xml_path}")
            except FileNotFoundError as e:
                print(f"  Gripper model not found: {e}")
        else:
            print("  Note: geodude_assets not available, gripper won't be shown")
    else:
        # It might be an object from prl_assets
        subject_xml_path = get_object_xml_path(template.subject)
        if subject_xml_path:
            print(f"  Subject model: {subject_xml_path}")
            subject_temp_name = "_subject_object.xml"
            subject_temp_path = subject_xml_path.parent / subject_temp_name
            subject_temp_path.write_text(subject_xml_path.read_text())
            include_files.append(str(subject_temp_path.absolute()))  # Use absolute path
            temp_files.append(subject_temp_path)
            use_subject_include = True
            subject_body_name = template.subject  # Body name matches asset name
        else:
            print(f"  Subject: {template.subject} (no model found)")

    # Create scene and load model
    num_frames = 2 + args.samples  # object frame + canonical + samples

    xml = create_scene_xml(
        object_type=obj_props["type"],
        object_size=obj_props,
        num_frames=num_frames,
        include_files=include_files,
        use_ref_object_include=use_ref_include,
        use_subject_include=use_subject_include,
        ref_is_surface=obj_props.get("is_surface", False),
    )

    # Write scene XML to working directory (for include resolution)
    scene_temp_path = working_dir / "_tsr_viz_scene.xml"
    scene_temp_path.write_text(xml)
    temp_files.append(scene_temp_path)

    try:
        model = mujoco.MjModel.from_xml_path(str(scene_temp_path))
    finally:
        # Clean up temp files
        for tf in temp_files:
            if tf.exists():
                tf.unlink()

    data = mujoco.MjData(model)

    # Create TSR with object at origin
    object_pose = np.eye(4)
    tsr = TSR(T0_w=object_pose, Tw_e=template.Tw_e, Bw=template.Bw)

    # Set object frame (frame_0) at origin
    set_body_pose(model, data, "frame_0", object_pose)

    # Compute canonical pose at Bw midpoint using TSR's sample() method
    canonical_xyzrpy = (tsr.Bw[:, 0] + tsr.Bw[:, 1]) / 2
    canonical_pose = tsr.sample(canonical_xyzrpy)

    # Apply gripper frame compensation if needed
    if not args.no_compensation:
        canonical_pose = apply_gripper_frame_compensation(canonical_pose, template.subject)
        print(f"  Gripper compensation: applied for {template.subject}")
    else:
        print("  Gripper compensation: disabled")

    print(f"\nCanonical pose (Bw midpoint):")
    print(f"  Position: [{canonical_pose[0,3]:.4f}, {canonical_pose[1,3]:.4f}, {canonical_pose[2,3]:.4f}]")

    # Set canonical frame (frame_1)
    set_body_pose(model, data, "frame_1", canonical_pose)

    # Position subject at canonical pose (if we have a model for it)
    if subject_body_name:
        set_body_pose(model, data, subject_body_name, canonical_pose)

    # Sample additional poses
    for i in range(args.samples):
        pose = tsr.sample()
        if not args.no_compensation:
            pose = apply_gripper_frame_compensation(pose, template.subject)

        frame_name = f"frame_{i+2}"
        set_body_pose(model, data, frame_name, pose)

    mujoco.mj_forward(model, data)

    print(f"\nVisualization:")
    if use_ref_include:
        print(f"  Reference object = {template.reference} (from prl_assets)")
    else:
        print(f"  Reference object = {template.reference} (default geometry)")
    if subject_body_name:
        print(f"  Subject = {template.subject} at TSR pose")
    print(f"  Frame 0 (at origin) = object frame")
    print(f"  Frame 1 = canonical pose (subject frame)")
    if args.samples > 0:
        print(f"  Frames 2-{1+args.samples} = sampled poses")
    print(f"\n  Axes: Red=X, Green=Y, Blue=Z, Yellow=origin")
    print(f"\nClose viewer window to exit.")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 0.5
        viewer.cam.lookat[:] = [0, 0, 0.1]

        while viewer.is_running():
            # No physics stepping - static visualization
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
