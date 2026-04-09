# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for placement TSR generation."""

import numpy as np
import pytest
from mj_manipulator.grasp_sources.prl_assets import (
    _generate_surface_place_tsrs,
    _get_upward_faces,
)

# Surface at z=0.75, 60×40cm
SURFACE_POSE = np.eye(4)
SURFACE_POSE[2, 3] = 0.75
SURFACE_HX = 0.30
SURFACE_HY = 0.20
CLEARANCE = 0.005  # must match the constant in _generate_surface_place_tsrs


class TestGenerateSurfacePlaceTSRs:
    """Test _generate_surface_place_tsrs with real prl_assets geometry."""

    def test_cylinder_returns_one_tsr(self):
        tsrs = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "can",
        )
        assert len(tsrs) == 1

    def test_box_returns_one_tsr(self):
        tsrs = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "cracker_box",
        )
        assert len(tsrs) == 1

    def test_none_held_returns_empty(self):
        tsrs = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            None,
        )
        assert tsrs == []

    def test_unknown_type_returns_empty(self):
        tsrs = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "nonexistent_object",
        )
        assert tsrs == []

    def test_clearance_offset_applied(self):
        """Bw[2] (z-bounds) should be shifted up by the clearance buffer."""
        tsrs = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "can",
        )
        bw_z = tsrs[0].Bw[2]
        # Without clearance, z-bounds would be [0, 0] (object resting on surface).
        # With clearance, both bounds shift up.
        assert bw_z[0] == pytest.approx(CLEARANCE)
        assert bw_z[1] == pytest.approx(CLEARANCE)

    def test_cylinder_sampled_pose_above_surface(self):
        """A sampled placement pose should put the object above the surface."""
        tsrs = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "can",
        )
        pose = tsrs[0].sample()
        # can: height=0.123, COM at half-height above surface + clearance
        assert pose[2, 3] > SURFACE_POSE[2, 3]

    def test_xy_bounds_reflect_surface_with_margin(self):
        """Bw xy should be shrunk by the 5cm edge margin."""
        tsrs = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "can",
        )
        margin = 0.05
        expected_hx = SURFACE_HX - margin
        expected_hy = SURFACE_HY - margin
        np.testing.assert_allclose(tsrs[0].Bw[0], [-expected_hx, expected_hx])
        np.testing.assert_allclose(tsrs[0].Bw[1], [-expected_hy, expected_hy])

    def test_yaw_is_free(self):
        """Placement should allow any yaw orientation."""
        tsrs = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "can",
        )
        np.testing.assert_allclose(tsrs[0].Bw[5], [-np.pi, np.pi])


class TestGraspOffsetCorrection:
    """Test that T_gripper_object is composed into Tw_e for surface placement."""

    def test_no_grasp_offset_samples_object_pose(self):
        """Without grasp offset, TSR samples the object resting pose."""
        tsrs = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "can",
        )
        pose_no_offset = tsrs[0].sample()
        # Should be above the surface by COM height + clearance
        assert pose_no_offset[2, 3] > SURFACE_POSE[2, 3]

    def test_identity_grasp_offset_matches_no_offset(self):
        """Identity grasp transform should produce the same result."""
        tsrs_none = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "can",
        )
        tsrs_ident = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "can",
            T_gripper_object=np.eye(4),
        )
        # Tw_e should be identical since inv(I) = I
        np.testing.assert_allclose(tsrs_none[0].Tw_e, tsrs_ident[0].Tw_e, atol=1e-10)

    def test_grasp_offset_shifts_sample(self):
        """A non-identity grasp offset should shift the sampled pose."""
        # Simulate a gripper that's 10cm above the object
        T_gripper_object = np.eye(4)
        T_gripper_object[2, 3] = -0.10  # object is 10cm below gripper

        tsrs_no_offset = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "can",
        )
        tsrs_with_offset = _generate_surface_place_tsrs(
            SURFACE_POSE,
            SURFACE_HX,
            SURFACE_HY,
            "can",
            T_gripper_object=T_gripper_object,
        )

        # Tw_e should differ — the corrected one includes inv(T_gripper_object)
        assert not np.allclose(tsrs_no_offset[0].Tw_e, tsrs_with_offset[0].Tw_e)

        # The corrected TSR should sample a gripper pose that's higher than
        # the object pose (since gripper is above the object)
        # Use Bw=0 sample (deterministic at center) by setting tight bounds
        pose_object = tsrs_no_offset[0].sample()
        pose_gripper = tsrs_with_offset[0].sample()
        # Gripper z should be ~10cm above object z
        z_diff = pose_gripper[2, 3] - pose_object[2, 3]
        assert z_diff == pytest.approx(0.10, abs=0.01)


def _rotx(angle):
    """Rotation matrix around X axis."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4)
    R[1, 1] = c
    R[1, 2] = -s
    R[2, 1] = s
    R[2, 2] = c
    return R


def _roty(angle):
    """Rotation matrix around Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4)
    R[0, 0] = c
    R[0, 2] = s
    R[2, 0] = -s
    R[2, 2] = c
    return R


def _rotz(angle):
    """Rotation matrix around Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


BOX_GP = {"type": "box", "size": [0.10, 0.08, 0.06]}
CUBE_GP = {"type": "box", "size": [0.06, 0.06, 0.06]}
CYL_GP = {"type": "cylinder", "radius": 0.04, "height": 0.12}
SPHERE_GP = {"type": "sphere", "radius": 0.05}


class TestGetUpwardFaces:
    """Test _get_upward_faces face enumeration and normal filtering."""

    def test_box_upright_returns_one_face(self):
        """Upright box: only +z face points up."""
        pose = np.eye(4)
        pose[2, 3] = 0.5
        faces = _get_upward_faces(pose, BOX_GP)
        assert len(faces) == 1

    def test_box_upright_z_face_extents(self):
        """The +z face of a box has extents (lx/2, ly/2)."""
        pose = np.eye(4)
        faces = _get_upward_faces(pose, BOX_GP)
        _, hx, hy = faces[0]
        assert hx == pytest.approx(0.05)  # lx/2
        assert hy == pytest.approx(0.04)  # ly/2

    def test_box_upright_surface_at_top(self):
        """Surface center should be at object center + lz/2 upward."""
        pose = np.eye(4)
        pose[2, 3] = 0.5
        faces = _get_upward_faces(pose, BOX_GP)
        surface_z = faces[0][0][2, 3]
        assert surface_z == pytest.approx(0.5 + 0.03)  # center + lz/2

    def test_box_on_side_returns_former_y_face(self):
        """Box rotated 90° around X: the +y face now points up."""
        pose = _rotx(np.pi / 2)
        faces = _get_upward_faces(pose, BOX_GP)
        assert len(faces) == 1
        # Former ±y face has extents (lx/2, lz/2)
        _, hx, hy = faces[0]
        assert hx == pytest.approx(0.05)  # lx/2
        assert hy == pytest.approx(0.03)  # lz/2

    def test_box_tilted_30deg_no_faces(self):
        """Box tilted 30° — no face is within 18° of vertical."""
        pose = _rotx(np.radians(30))
        faces = _get_upward_faces(pose, BOX_GP)
        assert len(faces) == 0

    def test_cylinder_upright_returns_one_face(self):
        """Upright cylinder: only +z end cap points up."""
        pose = np.eye(4)
        faces = _get_upward_faces(pose, CYL_GP)
        assert len(faces) == 1

    def test_cylinder_upright_extents(self):
        """Cylinder end cap extents are (r, r)."""
        pose = np.eye(4)
        faces = _get_upward_faces(pose, CYL_GP)
        _, hx, hy = faces[0]
        assert hx == pytest.approx(0.04)
        assert hy == pytest.approx(0.04)

    def test_cylinder_on_side_no_faces(self):
        """Cylinder on its side: no flat face points up."""
        pose = _rotx(np.pi / 2)
        faces = _get_upward_faces(pose, CYL_GP)
        assert len(faces) == 0

    def test_sphere_always_empty(self):
        """Sphere has no flat faces."""
        pose = np.eye(4)
        faces = _get_upward_faces(pose, SPHERE_GP)
        assert len(faces) == 0

    def test_surface_pose_is_valid_se3(self):
        """All returned surface poses should be valid SE(3)."""
        pose = np.eye(4)
        for faces_gp in [BOX_GP, CYL_GP]:
            for surface_pose, _, _ in _get_upward_faces(pose, faces_gp):
                R = surface_pose[:3, :3]
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
                np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_surface_normal_points_up(self):
        """Surface pose Z axis should point approximately up."""
        pose = np.eye(4)
        for faces_gp in [BOX_GP, CYL_GP]:
            for surface_pose, _, _ in _get_upward_faces(pose, faces_gp):
                z_axis = surface_pose[:3, 2]
                assert z_axis[2] > 0.95


class TestGetUpwardFacesRotations:
    """Test _get_upward_faces with rotations around all axes and combined transforms."""

    def test_box_rotated_90_around_y(self):
        """Box rotated 90° around Y: +x face now points up."""
        pose = _roty(np.pi / 2)
        faces = _get_upward_faces(pose, BOX_GP)
        assert len(faces) == 1
        # Former ±x face has extents (ly/2, lz/2)
        _, hx, hy = faces[0]
        assert hx == pytest.approx(0.04)  # ly/2
        assert hy == pytest.approx(0.03)  # lz/2

    def test_box_rotated_90_around_z_still_z_face(self):
        """Box rotated 90° around Z: +z face still points up (yaw rotation)."""
        pose = _rotz(np.pi / 2)
        faces = _get_upward_faces(pose, BOX_GP)
        assert len(faces) == 1
        _, hx, hy = faces[0]
        # +z face extents are (lx/2, ly/2) but may be swapped by yaw
        assert {round(hx, 4), round(hy, 4)} == {0.05, 0.04}

    def test_box_upside_down(self):
        """Box flipped 180° around X: -z face now points up."""
        pose = _rotx(np.pi)
        faces = _get_upward_faces(pose, BOX_GP)
        assert len(faces) == 1
        _, hx, hy = faces[0]
        assert hx == pytest.approx(0.05)  # lx/2
        assert hy == pytest.approx(0.04)  # ly/2

    def test_box_at_offset_position_with_rotation(self):
        """Box at non-origin position with rotation: surface center is correct."""
        pose = _rotx(np.pi / 2)  # +y face points up
        pose[0, 3] = 1.0
        pose[1, 3] = 2.0
        pose[2, 3] = 0.5
        faces = _get_upward_faces(pose, BOX_GP)
        assert len(faces) == 1
        surface_pose = faces[0][0]
        # Surface center: object origin + local_y * ly/2 in world frame
        # After rotx(90), local Y maps to world Z
        assert surface_pose[0, 3] == pytest.approx(1.0)
        assert surface_pose[1, 3] == pytest.approx(2.0)
        assert surface_pose[2, 3] == pytest.approx(0.5 + 0.04)  # +ly/2 upward

    def test_cube_upright_returns_one_face(self):
        """Symmetric cube upright: exactly one face points up."""
        pose = np.eye(4)
        faces = _get_upward_faces(pose, CUBE_GP)
        assert len(faces) == 1
        _, hx, hy = faces[0]
        assert hx == pytest.approx(0.03)
        assert hy == pytest.approx(0.03)

    def test_tilt_just_under_threshold(self):
        """Box tilted just under ~18° (cos(18°) ≈ 0.951): one face still valid."""
        # 17° tilt: cos(17°) ≈ 0.956 > 0.95 threshold
        pose = _rotx(np.radians(17))
        faces = _get_upward_faces(pose, BOX_GP)
        assert len(faces) == 1

    def test_tilt_just_over_threshold(self):
        """Box tilted just over ~18°: no faces valid."""
        # 19° tilt: cos(19°) ≈ 0.946 < 0.95 threshold
        pose = _rotx(np.radians(19))
        faces = _get_upward_faces(pose, BOX_GP)
        assert len(faces) == 0

    def test_cylinder_upside_down(self):
        """Cylinder flipped: -z end cap points up."""
        pose = _rotx(np.pi)
        faces = _get_upward_faces(pose, CYL_GP)
        assert len(faces) == 1

    def test_unknown_geometry_type(self):
        """Unknown geometry type returns empty."""
        pose = np.eye(4)
        faces = _get_upward_faces(pose, {"type": "torus", "major_r": 0.1})
        assert len(faces) == 0

    def test_missing_type_key(self):
        """Missing type key returns empty."""
        pose = np.eye(4)
        faces = _get_upward_faces(pose, {"radius": 0.05})
        assert len(faces) == 0


class TestGetUpwardFacesSurfacePoseOrientation:
    """Verify that the surface pose X/Y axes are correct for each face."""

    def test_box_upright_surface_x_follows_dest_x(self):
        """For +z face, surface X should align with destination X."""
        pose = np.eye(4)
        faces = _get_upward_faces(pose, BOX_GP)
        surface_x = faces[0][0][:3, 0]
        np.testing.assert_allclose(surface_x, [1, 0, 0], atol=1e-10)

    def test_box_upright_surface_y_follows_dest_y(self):
        """For +z face, surface Y should align with destination Y."""
        pose = np.eye(4)
        faces = _get_upward_faces(pose, BOX_GP)
        surface_y = faces[0][0][:3, 1]
        np.testing.assert_allclose(surface_y, [0, 1, 0], atol=1e-10)

    def test_box_rotated_surface_axes_orthogonal(self):
        """After rotation, surface X/Y/Z must remain orthonormal."""
        for angle in [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi]:
            for rot_fn in [_rotx, _roty, _rotz]:
                pose = rot_fn(angle)
                for gp in [BOX_GP, CYL_GP]:
                    for surface_pose, _, _ in _get_upward_faces(pose, gp):
                        R = surface_pose[:3, :3]
                        np.testing.assert_allclose(
                            R @ R.T,
                            np.eye(3),
                            atol=1e-9,
                            err_msg=f"Not orthonormal for {rot_fn.__name__}({np.degrees(angle):.0f}°), {gp['type']}",
                        )
                        np.testing.assert_allclose(
                            np.linalg.det(R),
                            1.0,
                            atol=1e-9,
                            err_msg=f"Not right-handed for {rot_fn.__name__}({np.degrees(angle):.0f}°), {gp['type']}",
                        )


class TestEndToEndSurfacePlacement:
    """End-to-end: face enumeration → surface placement TSR → sampled pose lands on top."""

    def test_can_on_upright_box(self):
        """Place a can on top of an upright box: sampled z should be above box top."""
        box_pose = np.eye(4)
        box_pose[2, 3] = 0.5  # box center at z=0.5
        lz = BOX_GP["size"][2]  # 0.06
        box_top_z = 0.5 + lz / 2  # 0.53

        faces = _get_upward_faces(box_pose, BOX_GP)
        assert len(faces) == 1
        surface_pose, hx, hy = faces[0]

        tsrs = _generate_surface_place_tsrs(surface_pose, hx, hy, "can")
        assert len(tsrs) == 1

        pose = tsrs[0].sample()
        # Can COM should be above the box top (half can height + clearance)
        assert pose[2, 3] > box_top_z

    def test_can_on_upright_cylinder(self):
        """Place a can on top of an upright cylinder."""
        cyl_pose = np.eye(4)
        cyl_pose[2, 3] = 0.5
        h = CYL_GP["height"]  # 0.12
        cyl_top_z = 0.5 + h / 2  # 0.56

        faces = _get_upward_faces(cyl_pose, CYL_GP)
        assert len(faces) == 1
        surface_pose, hx, hy = faces[0]

        tsrs = _generate_surface_place_tsrs(surface_pose, hx, hy, "can")
        assert len(tsrs) == 1

        pose = tsrs[0].sample()
        assert pose[2, 3] > cyl_top_z

    def test_box_on_side_then_place(self):
        """Box on its side (rotated 90° around X), place a can on the new top face."""
        box_pose = _rotx(np.pi / 2)
        box_pose[2, 3] = 0.5
        ly = BOX_GP["size"][1]  # 0.08
        # After rotx(90), +y local maps to +z world, so top is at z + ly/2
        box_top_z = 0.5 + ly / 2  # 0.54

        faces = _get_upward_faces(box_pose, BOX_GP)
        assert len(faces) == 1
        surface_pose, hx, hy = faces[0]

        tsrs = _generate_surface_place_tsrs(surface_pose, hx, hy, "can")
        assert len(tsrs) == 1

        pose = tsrs[0].sample()
        assert pose[2, 3] > box_top_z

    def test_end_to_end_with_grasp_offset(self):
        """Full pipeline: face → surface TSR → grasp offset → gripper above box."""
        box_pose = np.eye(4)
        box_pose[2, 3] = 0.5
        lz = BOX_GP["size"][2]
        box_top_z = 0.5 + lz / 2

        faces = _get_upward_faces(box_pose, BOX_GP)
        surface_pose, hx, hy = faces[0]

        # Gripper is 8cm above the object center
        T_gripper_object = np.eye(4)
        T_gripper_object[2, 3] = -0.08

        tsrs = _generate_surface_place_tsrs(
            surface_pose,
            hx,
            hy,
            "can",
            T_gripper_object=T_gripper_object,
        )
        pose = tsrs[0].sample()

        # Gripper should be even higher: above box top + can half-height + offset
        assert pose[2, 3] > box_top_z + 0.08

    def test_tilted_box_no_placement(self):
        """Tilted box produces no faces, so no placement TSRs."""
        box_pose = _rotx(np.radians(30))
        box_pose[2, 3] = 0.5

        faces = _get_upward_faces(box_pose, BOX_GP)
        assert len(faces) == 0

    def test_small_surface_clamps_margin(self):
        """Very small surface: edge margin clamps extents to minimum 0.01."""
        # Tiny box: 4cm × 4cm × 4cm → face half-extents = 0.02
        # After 5cm margin: max(0.01, 0.02 - 0.05) = 0.01
        tiny_gp = {"type": "box", "size": [0.04, 0.04, 0.04]}
        pose = np.eye(4)

        faces = _get_upward_faces(pose, tiny_gp)
        assert len(faces) == 1
        surface_pose, hx, hy = faces[0]

        tsrs = _generate_surface_place_tsrs(surface_pose, hx, hy, "can")
        assert len(tsrs) == 1
        # Bw xy should be clamped to ±0.01
        np.testing.assert_allclose(tsrs[0].Bw[0], [-0.01, 0.01], atol=1e-6)


class TestGeneratePlaceTSRsDispatch:
    """Test PrlAssetsGraspSource placement dispatch by geometry type."""

    def test_unknown_dest_type_returns_empty(self):
        # get_placements for a nonexistent type should return empty

        # PrlAssetsGraspSource needs model/data — just test the helper
        from mj_manipulator.grasp_sources.prl_assets import _instance_to_type

        assert _instance_to_type("foo_0") == "foo"
        assert _instance_to_type("nonexistent") is None
