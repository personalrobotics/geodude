"""Affordance discovery for manipulation primitives.

This module provides automatic discovery of manipulation affordances (grasps, places)
from TSR templates. The AffordanceRegistry indexes templates by object type and task,
enabling queries like "how can I grasp a can?" or "where can I place this object?".

Example:
    from geodude.affordances import AffordanceRegistry

    registry = AffordanceRegistry()
    registry.load_from_directory(Path("tsr_templates"))

    # Find grasp affordances for a can
    affordances = registry.get_grasp_affordances("can", hand_type="robotiq_2f_140")

    # Create concrete TSR at object pose
    grasp_tsr = affordances[0].create_tsr(object_pose)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tsr import TSR
    from tsr.core.tsr_primitive import ParsedTemplate

logger = logging.getLogger(__name__)


# Hand compatibility mapping
# Maps canonical hand names to aliases found in TSR templates
HAND_ALIASES: dict[str, list[str]] = {
    "robotiq_2f_140": ["robotiq_2f_140", "robotiq_2f140", "robotiq_2f-140"],
    "robotiq_2f_85": ["robotiq_2f_85", "robotiq_2f85", "robotiq_2f-85"],
}


def normalize_hand_type(hand_type: str) -> str:
    """Normalize hand type to canonical form.

    Args:
        hand_type: Hand type string (may be an alias)

    Returns:
        Canonical hand type, or original if no mapping found
    """
    for canonical, aliases in HAND_ALIASES.items():
        if hand_type in aliases:
            return canonical
    return hand_type


def hand_types_compatible(template_subject: str, query_hand: str) -> bool:
    """Check if a template's subject is compatible with a query hand type.

    Args:
        template_subject: The 'subject' field from a TSR template
        query_hand: The hand type being queried

    Returns:
        True if compatible
    """
    # Normalize both to canonical forms
    template_canonical = normalize_hand_type(template_subject)
    query_canonical = normalize_hand_type(query_hand)
    return template_canonical == query_canonical


@dataclass
class Affordance:
    """A discovered manipulation affordance.

    Represents a way to interact with an object (grasp it, place it, etc.)
    as defined by a TSR template.

    Attributes:
        task: Task type ("grasp", "place", "recycle")
        object_type: Object category this applies to ("can", "bottle")
        template: The parsed TSR template
        template_path: Path to the template file
        hand_type: Canonical hand type this was designed for
    """

    task: str
    object_type: str
    template: "ParsedTemplate"
    template_path: Path
    hand_type: str = field(default="")

    def __post_init__(self):
        """Set hand_type from template if not provided."""
        if not self.hand_type and hasattr(self.template, "subject"):
            self.hand_type = normalize_hand_type(self.template.subject)

    def create_tsr(
        self,
        object_pose: np.ndarray,
        *,
        apply_gripper_compensation: bool = True,
    ) -> "TSR":
        """Create a concrete TSR at the given object pose.

        Args:
            object_pose: 4x4 transform or 3-element position of the object
            apply_gripper_compensation: If True, apply gripper frame compensation

        Returns:
            TSR positioned at the object
        """
        from tsr import TSR

        from geodude.tsr_utils import compensate_tsr_for_gripper

        # Handle position-only input
        if object_pose.shape == (3,):
            T0_w = np.eye(4)
            T0_w[:3, 3] = object_pose
        elif object_pose.shape == (4, 4):
            T0_w = object_pose.copy()
        else:
            raise ValueError(
                f"object_pose must be (3,) position or (4,4) transform, "
                f"got shape {object_pose.shape}"
            )

        # Create TSR at object pose
        tsr = TSR(T0_w=T0_w, Tw_e=self.template.Tw_e, Bw=self.template.Bw)

        # Apply gripper frame compensation if requested
        if apply_gripper_compensation:
            tsr = compensate_tsr_for_gripper(tsr, self.template.subject)

        return tsr

    @property
    def name(self) -> str:
        """Human-readable name from template."""
        return getattr(self.template, "name", self.template_path.stem)

    def __repr__(self) -> str:
        return (
            f"Affordance(task={self.task!r}, object={self.object_type!r}, "
            f"name={self.name!r}, hand={self.hand_type!r})"
        )


class AffordanceRegistry:
    """Registry for discovering manipulation affordances.

    Loads TSR templates from directories and indexes them by object type and task.
    Supports queries filtered by hand compatibility.

    Example:
        registry = AffordanceRegistry()
        registry.load_from_directory(Path("tsr_templates"))

        # Get all grasp affordances for cans
        grasps = registry.get_grasp_affordances("can")

        # Filter by hand type
        grasps = registry.get_grasp_affordances("can", hand_type="robotiq_2f_140")
    """

    # Task types that count as "grasp"
    GRASP_TASKS = {"grasp"}

    # Task types that count as "place"
    PLACE_TASKS = {"place", "recycle", "discard", "drop"}

    def __init__(self, template_dirs: list[Path] | None = None):
        """Initialize registry.

        Args:
            template_dirs: Directories to load templates from.
                          If None, must call load_from_directory() manually.
        """
        # Index: object_type -> list of affordances
        self._by_object: dict[str, list[Affordance]] = {}

        # Index: (object_type, task) -> list of affordances
        self._by_object_task: dict[tuple[str, str], list[Affordance]] = {}

        # All affordances
        self._affordances: list[Affordance] = []

        # Track loaded directories
        self._loaded_dirs: set[Path] = set()

        # Load from provided directories
        if template_dirs:
            for d in template_dirs:
                self.load_from_directory(d)

    def load_from_directory(self, path: Path) -> int:
        """Load all TSR templates from a directory.

        Expects structure like:
            path/
            ├── grasps/
            │   └── can_side_grasp.yaml
            └── places/
                └── can_on_table.yaml

        Args:
            path: Directory containing TSR templates

        Returns:
            Number of templates loaded
        """
        from tsr.core.tsr_primitive import load_template_file

        path = Path(path)
        if not path.exists():
            logger.warning(f"Template directory does not exist: {path}")
            return 0

        if path in self._loaded_dirs:
            logger.debug(f"Directory already loaded: {path}")
            return 0

        count = 0
        for yaml_path in path.rglob("*.yaml"):
            try:
                template = load_template_file(str(yaml_path))

                # Extract metadata
                task = getattr(template, "task", None)
                reference = getattr(template, "reference", None)

                if not task or not reference:
                    logger.debug(
                        f"Skipping {yaml_path}: missing task or reference field"
                    )
                    continue

                # Create affordance
                affordance = Affordance(
                    task=task,
                    object_type=reference,
                    template=template,
                    template_path=yaml_path,
                )

                self._add_affordance(affordance)
                count += 1
                logger.debug(f"Loaded affordance: {affordance}")

            except Exception as e:
                logger.warning(f"Failed to load template {yaml_path}: {e}")

        self._loaded_dirs.add(path)
        logger.info(f"Loaded {count} affordances from {path}")
        return count

    def _add_affordance(self, affordance: Affordance) -> None:
        """Add an affordance to the indices."""
        self._affordances.append(affordance)

        # Index by object type
        if affordance.object_type not in self._by_object:
            self._by_object[affordance.object_type] = []
        self._by_object[affordance.object_type].append(affordance)

        # Index by (object_type, task)
        key = (affordance.object_type, affordance.task)
        if key not in self._by_object_task:
            self._by_object_task[key] = []
        self._by_object_task[key].append(affordance)

    def get_affordances(
        self,
        object_type: str,
        *,
        task: str | None = None,
        hand_type: str | None = None,
    ) -> list[Affordance]:
        """Get affordances for an object type.

        Args:
            object_type: Object category (e.g., "can", "bottle")
            task: Filter by task type (e.g., "grasp", "place")
            hand_type: Filter by compatible hand type

        Returns:
            List of matching affordances
        """
        # Get base list
        if task:
            affordances = list(self._by_object_task.get((object_type, task), []))
        else:
            affordances = list(self._by_object.get(object_type, []))

        # Filter by hand type if specified
        if hand_type:
            affordances = [
                a
                for a in affordances
                if hand_types_compatible(a.hand_type, hand_type)
            ]

        return affordances

    def get_grasp_affordances(
        self,
        object_type: str,
        hand_type: str | None = None,
    ) -> list[Affordance]:
        """Get grasp affordances for an object type.

        Args:
            object_type: Object category (e.g., "can")
            hand_type: Filter by compatible hand type

        Returns:
            List of grasp affordances
        """
        result = []
        for task in self.GRASP_TASKS:
            result.extend(
                self.get_affordances(object_type, task=task, hand_type=hand_type)
            )
        return result

    def get_place_affordances(
        self,
        object_type: str,
        destination_type: str | None = None,
        hand_type: str | None = None,
    ) -> list[Affordance]:
        """Get place affordances for an object type.

        Args:
            object_type: Object being placed (e.g., "can")
            destination_type: Filter by destination (e.g., "recycle_bin").
                            If None, returns all place affordances.
            hand_type: Filter by compatible hand type

        Returns:
            List of place affordances
        """
        # For place affordances, object_type in the template is the destination
        # We need to search differently - look for templates where:
        # - task is a place task
        # - reference (object_type in template) matches destination_type

        result = []
        for task in self.PLACE_TASKS:
            if destination_type:
                # Filter by destination
                affordances = self.get_affordances(
                    destination_type, task=task, hand_type=hand_type
                )
            else:
                # Return all place affordances across all object types
                for obj_type in self._by_object:
                    affordances = self.get_affordances(
                        obj_type, task=task, hand_type=hand_type
                    )
                    result.extend(affordances)
                continue

            result.extend(affordances)

        return result

    def get_object_types(self) -> list[str]:
        """Get all object types with registered affordances."""
        return list(self._by_object.keys())

    def get_tasks_for_object(self, object_type: str) -> list[str]:
        """Get all task types available for an object."""
        tasks = set()
        for (obj, task), affordances in self._by_object_task.items():
            if obj == object_type and affordances:
                tasks.add(task)
        return list(tasks)

    def __len__(self) -> int:
        """Number of registered affordances."""
        return len(self._affordances)

    def __repr__(self) -> str:
        return (
            f"AffordanceRegistry({len(self._affordances)} affordances, "
            f"{len(self._by_object)} object types)"
        )
