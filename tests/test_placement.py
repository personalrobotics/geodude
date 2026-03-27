"""Tests for placement TSR generation."""

import numpy as np
import pytest

from geodude.bt.nodes import _generate_surface_place_tsrs, _generate_place_tsrs


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


class TestGeneratePlaceTSRsDispatch:
    """Test _generate_place_tsrs dispatches correctly by geometry type."""

    def test_unknown_dest_type_returns_empty(self):
        tsrs = _generate_place_tsrs(None, "foo_0", "nonexistent_type", held_height=0.1)
        assert tsrs == []
