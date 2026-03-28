"""Tests for placement TSR generation."""

import numpy as np
import pytest

from geodude.bt.nodes import (
    _generate_surface_place_tsrs,
    _generate_place_tsrs,
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
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "can",
        )
        assert len(tsrs) == 1

    def test_box_returns_one_tsr(self):
        tsrs = _generate_surface_place_tsrs(
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "cracker_box",
        )
        assert len(tsrs) == 1

    def test_none_held_returns_empty(self):
        tsrs = _generate_surface_place_tsrs(
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, None,
        )
        assert tsrs == []

    def test_unknown_type_returns_empty(self):
        tsrs = _generate_surface_place_tsrs(
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "nonexistent_object",
        )
        assert tsrs == []

    def test_clearance_offset_applied(self):
        """Bw[2] (z-bounds) should be shifted up by the clearance buffer."""
        tsrs = _generate_surface_place_tsrs(
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "can",
        )
        bw_z = tsrs[0].Bw[2]
        # Without clearance, z-bounds would be [0, 0] (object resting on surface).
        # With clearance, both bounds shift up.
        assert bw_z[0] == pytest.approx(CLEARANCE)
        assert bw_z[1] == pytest.approx(CLEARANCE)

    def test_cylinder_sampled_pose_above_surface(self):
        """A sampled placement pose should put the object above the surface."""
        tsrs = _generate_surface_place_tsrs(
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "can",
        )
        pose = tsrs[0].sample()
        # can: height=0.123, COM at half-height above surface + clearance
        assert pose[2, 3] > SURFACE_POSE[2, 3]

    def test_xy_bounds_reflect_surface_with_margin(self):
        """Bw xy should be shrunk by the 5cm edge margin."""
        tsrs = _generate_surface_place_tsrs(
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "can",
        )
        margin = 0.05
        expected_hx = SURFACE_HX - margin
        expected_hy = SURFACE_HY - margin
        np.testing.assert_allclose(tsrs[0].Bw[0], [-expected_hx, expected_hx])
        np.testing.assert_allclose(tsrs[0].Bw[1], [-expected_hy, expected_hy])

    def test_yaw_is_free(self):
        """Placement should allow any yaw orientation."""
        tsrs = _generate_surface_place_tsrs(
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "can",
        )
        np.testing.assert_allclose(tsrs[0].Bw[5], [-np.pi, np.pi])


class TestGraspOffsetCorrection:
    """Test that T_gripper_object is composed into Tw_e for surface placement."""

    def test_no_grasp_offset_samples_object_pose(self):
        """Without grasp offset, TSR samples the object resting pose."""
        tsrs = _generate_surface_place_tsrs(
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "can",
        )
        pose_no_offset = tsrs[0].sample()
        # Should be above the surface by COM height + clearance
        assert pose_no_offset[2, 3] > SURFACE_POSE[2, 3]

    def test_identity_grasp_offset_matches_no_offset(self):
        """Identity grasp transform should produce the same result."""
        tsrs_none = _generate_surface_place_tsrs(
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "can",
        )
        tsrs_ident = _generate_surface_place_tsrs(
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "can",
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
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "can",
        )
        tsrs_with_offset = _generate_surface_place_tsrs(
            None, SURFACE_POSE, SURFACE_HX, SURFACE_HY, "can",
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
    R[1, 1] = c; R[1, 2] = -s
    R[2, 1] = s; R[2, 2] = c
    return R


BOX_GP = {"type": "box", "size": [0.10, 0.08, 0.06]}
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


class TestGeneratePlaceTSRsDispatch:
    """Test _generate_place_tsrs dispatches correctly by geometry type."""

    def test_unknown_dest_type_returns_empty(self):
        tsrs = _generate_place_tsrs(None, "foo_0", "nonexistent_type", held_height=0.1)
        assert tsrs == []
