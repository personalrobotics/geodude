"""Tests for TSR utilities."""

import numpy as np
import pytest

# Check if TSR is available
try:
    from tsr import TSR
    from geodude.tsr_utils import (
        create_top_grasp_tsr,
        create_side_grasp_tsr,
        create_place_tsr,
        create_lift_tsr,
        create_retract_tsr,
        create_approach_tsr,
    )

    TSR_AVAILABLE = True
except ImportError:
    TSR_AVAILABLE = False


@pytest.mark.skipif(not TSR_AVAILABLE, reason="TSR not installed")
class TestTopGraspTSR:
    """Tests for create_top_grasp_tsr."""

    def test_creates_valid_tsr(self):
        """Creates a valid TSR object."""
        object_pose = np.eye(4)
        object_pose[2, 3] = 0.5  # Object at z=0.5

        tsr = create_top_grasp_tsr(
            object_pose=object_pose,
            object_height=0.1,
            gripper_standoff=0.15,
        )

        assert isinstance(tsr, TSR)

    def test_gripper_above_object(self):
        """Gripper pose is above the object."""
        object_pose = np.eye(4)
        object_pose[2, 3] = 0.5

        tsr = create_top_grasp_tsr(
            object_pose=object_pose,
            object_height=0.1,
            gripper_standoff=0.15,
        )

        # Sample a pose from the TSR
        sample = tsr.sample()

        # Gripper should be above object
        assert sample[2, 3] > object_pose[2, 3], "Gripper should be above object"

    def test_gripper_z_points_down(self):
        """Gripper z-axis points downward."""
        object_pose = np.eye(4)

        tsr = create_top_grasp_tsr(
            object_pose=object_pose,
            object_height=0.1,
        )

        sample = tsr.sample()
        gripper_z = sample[:3, 2]

        # Gripper z should point down (negative world z)
        assert gripper_z[2] < 0, "Gripper z should point down"

    def test_any_yaw_freedom(self):
        """With allow_any_yaw=True, yaw bounds span full rotation."""
        object_pose = np.eye(4)

        tsr = create_top_grasp_tsr(
            object_pose=object_pose,
            object_height=0.1,
            allow_any_yaw=True,
        )

        # Yaw bounds should allow full rotation
        assert tsr.Bw[5, 0] == pytest.approx(-np.pi)
        assert tsr.Bw[5, 1] == pytest.approx(np.pi)

    def test_restricted_yaw(self):
        """With allow_any_yaw=False, yaw bounds are small."""
        object_pose = np.eye(4)

        tsr = create_top_grasp_tsr(
            object_pose=object_pose,
            object_height=0.1,
            allow_any_yaw=False,
        )

        # Yaw bounds should be small
        assert abs(tsr.Bw[5, 0]) < 0.1
        assert abs(tsr.Bw[5, 1]) < 0.1


@pytest.mark.skipif(not TSR_AVAILABLE, reason="TSR not installed")
class TestSideGraspTSR:
    """Tests for create_side_grasp_tsr."""

    def test_creates_valid_tsr(self):
        """Creates a valid TSR object."""
        object_pose = np.eye(4)

        tsr = create_side_grasp_tsr(
            object_pose=object_pose,
            object_width=0.1,
            approach_axis="y",
        )

        assert isinstance(tsr, TSR)

    def test_y_approach(self):
        """Approach from +y direction."""
        object_pose = np.eye(4)
        object_pose[0, 3] = 0.5  # Object at x=0.5

        tsr = create_side_grasp_tsr(
            object_pose=object_pose,
            object_width=0.1,
            gripper_standoff=0.15,
            approach_axis="y",
        )

        sample = tsr.sample()

        # Gripper should be offset in y direction
        assert sample[1, 3] > object_pose[1, 3], "Gripper should be at +y from object"

    def test_negative_y_approach(self):
        """Approach from -y direction."""
        object_pose = np.eye(4)

        tsr = create_side_grasp_tsr(
            object_pose=object_pose,
            object_width=0.1,
            gripper_standoff=0.15,
            approach_axis="-y",
        )

        sample = tsr.sample()

        # Gripper should be offset in negative y direction
        assert sample[1, 3] < object_pose[1, 3], "Gripper should be at -y from object"

    def test_invalid_axis_raises(self):
        """Invalid approach axis raises ValueError."""
        object_pose = np.eye(4)

        with pytest.raises(ValueError):
            create_side_grasp_tsr(
                object_pose=object_pose,
                object_width=0.1,
                approach_axis="z",  # Invalid
            )


@pytest.mark.skipif(not TSR_AVAILABLE, reason="TSR not installed")
class TestPlaceTSR:
    """Tests for create_place_tsr."""

    def test_creates_valid_tsr(self):
        """Creates a valid TSR object."""
        surface_pose = np.eye(4)

        tsr = create_place_tsr(
            surface_pose=surface_pose,
            surface_height=0.02,  # Table top thickness
            object_height=0.1,
            gripper_standoff=0.15,
        )

        assert isinstance(tsr, TSR)

    def test_position_above_surface(self):
        """Gripper positioned above surface."""
        surface_pose = np.eye(4)
        surface_pose[2, 3] = 0.5  # Table at z=0.5

        tsr = create_place_tsr(
            surface_pose=surface_pose,
            surface_height=0.02,
            object_height=0.1,
            gripper_standoff=0.15,
        )

        sample = tsr.sample()

        # Gripper should be above surface
        assert sample[2, 3] > surface_pose[2, 3], "Gripper should be above surface"


@pytest.mark.skipif(not TSR_AVAILABLE, reason="TSR not installed")
class TestLiftTSR:
    """Tests for create_lift_tsr."""

    def test_creates_valid_tsr(self):
        """Creates a valid TSR object."""
        current_pose = np.eye(4)

        tsr = create_lift_tsr(
            current_ee_pose=current_pose,
            lift_height=0.1,
        )

        assert isinstance(tsr, TSR)

    def test_lifted_position(self):
        """Lift TSR is above current pose."""
        current_pose = np.eye(4)
        current_pose[2, 3] = 0.5

        lift_height = 0.1
        tsr = create_lift_tsr(
            current_ee_pose=current_pose,
            lift_height=lift_height,
        )

        sample = tsr.sample()

        # Should be approximately lift_height above current
        expected_z = current_pose[2, 3] + lift_height
        assert sample[2, 3] == pytest.approx(expected_z, abs=0.05)


@pytest.mark.skipif(not TSR_AVAILABLE, reason="TSR not installed")
class TestRetractTSR:
    """Tests for create_retract_tsr."""

    def test_creates_valid_tsr(self):
        """Creates a valid TSR object."""
        current_pose = np.eye(4)

        tsr = create_retract_tsr(
            current_ee_pose=current_pose,
            retract_distance=0.1,
        )

        assert isinstance(tsr, TSR)

    def test_retracted_along_z(self):
        """Retract moves back along gripper z-axis."""
        # Create a pose with gripper pointing down (z = [0, 0, -1])
        current_pose = np.array([
            [1, 0, 0, 0.5],
            [0, -1, 0, 0.0],
            [0, 0, -1, 0.5],  # z points down
            [0, 0, 0, 1],
        ])

        retract_distance = 0.1
        tsr = create_retract_tsr(
            current_ee_pose=current_pose,
            retract_distance=retract_distance,
        )

        sample = tsr.sample()

        # Retracting along negative z (which points down) should move up
        expected_z = current_pose[2, 3] + retract_distance
        assert sample[2, 3] == pytest.approx(expected_z, abs=0.05)


@pytest.mark.skipif(not TSR_AVAILABLE, reason="TSR not installed")
class TestApproachTSR:
    """Tests for create_approach_tsr."""

    def test_creates_valid_tsr(self):
        """Creates a valid TSR object."""
        target_pose = np.eye(4)

        tsr = create_approach_tsr(
            target_ee_pose=target_pose,
            approach_distance=0.1,
        )

        assert isinstance(tsr, TSR)

    def test_approach_before_target(self):
        """Approach is positioned back from target."""
        # Target with gripper pointing down
        target_pose = np.array([
            [1, 0, 0, 0.5],
            [0, -1, 0, 0.0],
            [0, 0, -1, 0.5],  # z points down
            [0, 0, 0, 1],
        ])

        approach_distance = 0.1
        tsr = create_approach_tsr(
            target_ee_pose=target_pose,
            approach_distance=approach_distance,
        )

        sample = tsr.sample()

        # Approach should be above target (back along z which points down)
        expected_z = target_pose[2, 3] + approach_distance
        assert sample[2, 3] == pytest.approx(expected_z, abs=0.05)
