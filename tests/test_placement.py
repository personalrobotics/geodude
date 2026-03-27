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


class TestGeneratePlaceTSRsDispatch:
    """Test _generate_place_tsrs dispatches correctly by geometry type."""

    def test_unknown_dest_type_returns_empty(self):
        tsrs = _generate_place_tsrs(None, "foo_0", "nonexistent_type", held_height=0.1)
        assert tsrs == []
