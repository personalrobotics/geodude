"""Tests for affordance discovery and registry."""

from pathlib import Path

import numpy as np
import pytest

from geodude.affordances import (
    HAND_ALIASES,
    Affordance,
    AffordanceRegistry,
    hand_types_compatible,
    normalize_hand_type,
)

# Path to TSR templates in the geodude package
TSR_TEMPLATES_DIR = Path(__file__).parent.parent / "tsr_templates"


class TestHandCompatibility:
    """Tests for hand type normalization and compatibility."""

    def test_normalize_robotiq_variants(self):
        """Normalizes Robotiq gripper name variants."""
        assert normalize_hand_type("robotiq_2f140") == "robotiq_2f_140"
        assert normalize_hand_type("robotiq_2f_140") == "robotiq_2f_140"
        assert normalize_hand_type("robotiq_2f-140") == "robotiq_2f_140"

    def test_normalize_unknown_returns_original(self):
        """Unknown hand types pass through unchanged."""
        assert normalize_hand_type("unknown_gripper") == "unknown_gripper"

    def test_compatible_same_hand(self):
        """Same hand type is compatible."""
        assert hand_types_compatible("robotiq_2f_140", "robotiq_2f_140")

    def test_compatible_alias(self):
        """Aliases are compatible with canonical form."""
        assert hand_types_compatible("robotiq_2f140", "robotiq_2f_140")
        assert hand_types_compatible("robotiq_2f_140", "robotiq_2f140")

    def test_incompatible_different_hands(self):
        """Different hands are not compatible."""
        assert not hand_types_compatible("robotiq_2f_140", "robotiq_2f_85")


class TestAffordance:
    """Tests for the Affordance dataclass."""

    @pytest.fixture
    def mock_template(self):
        """Create a mock template for testing."""
        from unittest.mock import MagicMock

        template = MagicMock()
        template.name = "Test Grasp"
        template.task = "grasp"
        template.subject = "robotiq_2f140"
        template.reference = "can"
        template.Tw_e = np.eye(4)
        template.Bw = np.zeros((6, 2))
        return template

    def test_affordance_creation(self, mock_template):
        """Creates affordance with correct attributes."""
        aff = Affordance(
            task="grasp",
            object_type="can",
            template=mock_template,
            template_path=Path("test.yaml"),
        )

        assert aff.task == "grasp"
        assert aff.object_type == "can"
        assert aff.hand_type == "robotiq_2f_140"  # Normalized from template
        assert aff.name == "Test Grasp"

    def test_create_tsr_from_position(self, mock_template):
        """Creates TSR from 3D position."""
        aff = Affordance(
            task="grasp",
            object_type="can",
            template=mock_template,
            template_path=Path("test.yaml"),
        )

        pos = np.array([1.0, 2.0, 3.0])
        tsr = aff.create_tsr(pos, apply_gripper_compensation=False)

        assert tsr.T0_w[0, 3] == 1.0
        assert tsr.T0_w[1, 3] == 2.0
        assert tsr.T0_w[2, 3] == 3.0

    def test_create_tsr_from_transform(self, mock_template):
        """Creates TSR from 4x4 transform."""
        aff = Affordance(
            task="grasp",
            object_type="can",
            template=mock_template,
            template_path=Path("test.yaml"),
        )

        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        tsr = aff.create_tsr(T, apply_gripper_compensation=False)

        np.testing.assert_array_equal(tsr.T0_w[:3, 3], [1.0, 2.0, 3.0])

    def test_create_tsr_invalid_shape(self, mock_template):
        """Raises ValueError for invalid pose shape."""
        aff = Affordance(
            task="grasp",
            object_type="can",
            template=mock_template,
            template_path=Path("test.yaml"),
        )

        with pytest.raises(ValueError, match="must be"):
            aff.create_tsr(np.array([1.0, 2.0]))  # Wrong shape


class TestAffordanceRegistry:
    """Tests for the AffordanceRegistry."""

    @pytest.fixture
    def registry(self):
        """Create registry loaded with real templates."""
        reg = AffordanceRegistry()
        reg.load_from_directory(TSR_TEMPLATES_DIR)
        return reg

    def test_load_templates(self, registry):
        """Loads templates from directory."""
        # Should have loaded some templates
        assert len(registry) > 0

    def test_load_nonexistent_directory(self):
        """Returns 0 for nonexistent directory."""
        reg = AffordanceRegistry()
        count = reg.load_from_directory(Path("/nonexistent/path"))
        assert count == 0

    def test_load_duplicate_directory(self, registry):
        """Skips already-loaded directories."""
        initial_count = len(registry)
        registry.load_from_directory(TSR_TEMPLATES_DIR)
        assert len(registry) == initial_count

    def test_get_grasp_affordances_for_can(self, registry):
        """Finds grasp affordances for can."""
        grasps = registry.get_grasp_affordances("can")
        assert len(grasps) > 0
        assert all(a.task == "grasp" for a in grasps)
        assert all(a.object_type == "can" for a in grasps)

    def test_get_grasp_affordances_with_hand_filter(self, registry):
        """Filters by hand type."""
        # All templates use robotiq
        grasps = registry.get_grasp_affordances("can", hand_type="robotiq_2f_140")
        assert len(grasps) > 0

        # No templates for this hand
        grasps = registry.get_grasp_affordances("can", hand_type="unknown_gripper")
        assert len(grasps) == 0

    def test_get_place_affordances(self, registry):
        """Finds place affordances."""
        # recycle_bin is a destination in the recycle task
        places = registry.get_place_affordances(
            object_type="can", destination_type="recycle_bin"
        )
        assert len(places) > 0

    def test_get_object_types(self, registry):
        """Lists all object types."""
        types = registry.get_object_types()
        assert "can" in types

    def test_get_tasks_for_object(self, registry):
        """Lists tasks available for an object."""
        tasks = registry.get_tasks_for_object("can")
        assert "grasp" in tasks

    def test_registry_repr(self, registry):
        """Has informative repr."""
        r = repr(registry)
        assert "AffordanceRegistry" in r
        assert "affordances" in r


class TestAffordanceRegistryWithRealTemplates:
    """Integration tests with actual TSR templates."""

    @pytest.fixture
    def registry(self):
        """Create registry with real templates."""
        return AffordanceRegistry([TSR_TEMPLATES_DIR])

    def test_can_grasp_template_structure(self, registry):
        """Can grasp templates have expected structure."""
        grasps = registry.get_grasp_affordances("can")
        assert len(grasps) >= 1

        grasp = grasps[0]
        assert grasp.template.Tw_e is not None
        assert grasp.template.Bw is not None
        assert grasp.template.Bw.shape == (6, 2)

    def test_create_tsr_at_origin(self, registry):
        """Creates valid TSR at origin."""
        grasps = registry.get_grasp_affordances("can")
        grasp = grasps[0]

        tsr = grasp.create_tsr(np.zeros(3))

        # TSR should be valid
        assert tsr.T0_w is not None
        assert tsr.Tw_e is not None
        assert tsr.Bw is not None

    def test_create_tsr_at_position(self, registry):
        """Creates TSR at specific position."""
        grasps = registry.get_grasp_affordances("can")
        grasp = grasps[0]

        pos = np.array([0.5, 0.0, 1.0])
        tsr = grasp.create_tsr(pos)

        # TSR origin should be at object position
        np.testing.assert_array_almost_equal(tsr.T0_w[:3, 3], pos)
