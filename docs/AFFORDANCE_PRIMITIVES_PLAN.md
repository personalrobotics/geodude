# Affordance-Based Manipulation Primitives

**Issue**: #37
**Branch**: `feature/affordance-primitives`

## Overview

Implement reusable manipulation primitives (`pickup`, `place`) with automatic affordance discovery. TSRs stored with objects in prl_assets "just work" for any robot with compatible hands.

## Current State Analysis

### What Exists

| Component | Location | Status |
|-----------|----------|--------|
| TSR templates | `geodude/tsr_templates/` | YAML with task/subject/reference metadata |
| Template parser | `tsr/core/tsr_primitive.py` | `load_template_file()` returns `ParsedTemplate` |
| TSR library | `tsr/tsr_library_rel.py` | `TSRLibraryRelational` with query API |
| Object metadata | `prl_assets/objects/*/meta.yaml` | Policy hints, no TSR linkage |
| Gripper compensation | `geodude/tsr_utils.py` | Frame transforms for gripper types |
| Execution context | `geodude/execution.py` | `SimContext` with `execute()`, `arm().grasp()` |

### What's Missing

1. **Affordance Registry** - No centralized discovery of what actions are possible
2. **Object→TSR mapping** - Objects don't link to their TSR templates
3. **Hand compatibility** - TSRs don't declare which hands they work with
4. **High-level primitives** - No `robot.pickup()` or `robot.place()`

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Code                                │
│  robot.pickup("can_0")  /  robot.place("recycle_bin_0")        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   Manipulation Primitives                       │
│  pickup() / place() orchestration logic                        │
│  - Query affordances for object                                │
│  - Select compatible (arm, TSR) pair                           │
│  - Plan approach → grasp → lift                                │
│  - Execute via active context                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   Affordance Registry                           │
│  - Load TSR templates from prl_assets                          │
│  - Index by (object, task, hand)                               │
│  - Query: get_grasp_affordances(object, hand_type)             │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   Existing Components                           │
│  Arm.plan_to_tsr() / ExecutionContext / GraspManager           │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Affordance Registry

### 1.1 TSR Template Metadata Extension

Extend TSR templates to declare hand compatibility:

```yaml
# geodude/tsr_templates/grasps/can_side_grasp.yaml (existing + new fields)
name: "Can Side Grasp"
task: grasp
subject: robotiq_2f140          # Canonical gripper this was designed for
reference: can                  # Object type

# NEW: Hand compatibility (derived from subject or explicit)
compatible_hands:
  - robotiq_2f_140
  - robotiq_2f_85

# Existing geometric fields...
position:
  type: shell
  # ...
```

**Decision**: Store `compatible_hands` in template or infer from `subject`?

- Option A: Explicit `compatible_hands` list in each template
- Option B: Maintain hand→subject mapping in AffordanceRegistry
- **Recommendation**: Option B - keeps templates simple, compatibility logic centralized

### 1.2 Object Affordance Manifest

Add affordance metadata to prl_assets objects:

```yaml
# prl_assets/objects/can/affordances.yaml (NEW file)
object: can
affordances:
  grasp:
    - template: geodude://tsr_templates/grasps/can_side_grasp.yaml
      name: side_grasp
      preferred: true
    - template: geodude://tsr_templates/grasps/can_side_grasp_flipped.yaml
      name: side_grasp_flipped
  place:
    - template: geodude://tsr_templates/places/can_on_table_upright.yaml
      name: on_table_upright
      surface_types: [table, shelf]
    - template: geodude://tsr_templates/places/can_on_table_flipped.yaml
      name: on_table_flipped
      surface_types: [table]
```

**Alternative**: Store templates directly in prl_assets:
```
prl_assets/objects/can/
├── meta.yaml
├── mujoco/can.xml
└── tsrs/
    ├── grasps/
    │   └── side_grasp.yaml
    └── places/
        └── on_table_upright.yaml
```

**Recommendation**: Start with manifest pointing to geodude templates (easier migration), plan for object-local TSRs later.

### 1.3 AffordanceRegistry Class

```python
# geodude/src/geodude/affordances.py

@dataclass
class Affordance:
    """A discovered manipulation affordance."""
    task: str                    # "grasp", "place"
    object_type: str             # "can", "bottle"
    template: ParsedTemplate     # Loaded TSR template
    compatible_hands: list[str]  # ["robotiq_2f_140", "robotiq_2f_85"]

    def create_tsr(self, object_pose: np.ndarray) -> TSR:
        """Create concrete TSR at object pose."""
        T0_w = np.eye(4)
        T0_w[:3, :3] = object_pose[:3, :3] if object_pose.shape == (4, 4) else np.eye(3)
        T0_w[:3, 3] = object_pose[:3, 3] if object_pose.shape == (4, 4) else object_pose
        return TSR(T0_w=T0_w, Tw_e=self.template.Tw_e, Bw=self.template.Bw)


class AffordanceRegistry:
    """Registry for discovering manipulation affordances."""

    # Hand compatibility mapping
    HAND_COMPATIBILITY = {
        "robotiq_2f_140": ["robotiq_2f_140", "robotiq_2f140"],
        "robotiq_2f_85": ["robotiq_2f_85", "robotiq_2f85"],
        # Extensible for other hands
    }

    def __init__(self, template_dirs: list[Path] = None):
        """Initialize registry with template search paths."""
        self._templates: dict[str, list[Affordance]] = {}  # object_type -> affordances
        self._loaded = False
        self._template_dirs = template_dirs or []

    def load_from_directory(self, path: Path) -> None:
        """Load all TSR templates from directory structure."""
        # Scan grasps/, places/, etc.
        # Parse each template, extract metadata
        # Index by reference (object type)
        pass

    def get_grasp_affordances(
        self,
        object_type: str,
        hand_type: str | None = None,
    ) -> list[Affordance]:
        """Get grasp affordances for an object type.

        Args:
            object_type: Object category (e.g., "can", "bottle")
            hand_type: Filter by compatible hand (e.g., "robotiq_2f_140")

        Returns:
            List of compatible grasp affordances
        """
        pass

    def get_place_affordances(
        self,
        object_type: str,
        destination_type: str | None = None,
    ) -> list[Affordance]:
        """Get place affordances for an object type."""
        pass

    def get_affordances(
        self,
        object_type: str,
        task: str | None = None,
        hand_type: str | None = None,
    ) -> list[Affordance]:
        """Get all affordances for an object, optionally filtered."""
        pass
```

### 1.4 Integration with Geodude

```python
# In geodude/src/geodude/robot.py

class Geodude:
    def __init__(self, ...):
        # ... existing init ...

        # Initialize affordance registry
        self._affordance_registry = AffordanceRegistry([
            Path(__file__).parent.parent.parent / "tsr_templates",
            # Could add prl_assets paths here
        ])
        self._affordance_registry.load()

    @property
    def affordances(self) -> AffordanceRegistry:
        """Access the affordance registry."""
        return self._affordance_registry
```

### 1.5 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/geodude/affordances.py` | CREATE | `Affordance`, `AffordanceRegistry` classes |
| `src/geodude/robot.py` | MODIFY | Add `_affordance_registry`, `affordances` property |
| `tsr_templates/grasps/*.yaml` | VERIFY | Ensure task/subject/reference fields present |
| `tests/test_affordances.py` | CREATE | Unit tests for registry |

---

## Phase 2: Manipulation Primitives

### 2.1 Pickup Primitive

```python
# geodude/src/geodude/primitives.py

def pickup(
    robot: "Geodude",
    target: str | None = None,
    *,
    object_type: str | None = None,
    arm: "Arm | str | None" = None,
    base_heights: list[float] | None = None,
    lift_height: float = 0.10,
    timeout: float = 30.0,
) -> bool:
    """Pick up an object using affordance-based planning.

    Args:
        robot: The robot instance
        target: Specific object name (e.g., "can_0") or None for any pickable
        object_type: Filter by object type (e.g., "can") if target is None
        arm: Specific arm, or None to let robot choose
        base_heights: Base heights to search, or None for default
        lift_height: Height to lift after grasping
        timeout: Planning timeout per attempt

    Returns:
        True if pickup succeeded, False otherwise

    Raises:
        RuntimeError: If no execution context is active

    Example:
        with robot.sim() as ctx:
            robot.pickup("can_0")           # Specific object
            robot.pickup(object_type="can") # Any can
            robot.pickup()                  # Any pickable object
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    # 1. Resolve target object
    if target is None:
        target = _find_pickable_object(robot, object_type)
        if target is None:
            return False

    # 2. Get object info
    object_type = _get_object_type(robot, target)
    object_pose = robot.get_object_pose(target)

    # 3. Get grasp affordances
    affordances = robot.affordances.get_grasp_affordances(object_type)
    if not affordances:
        return False

    # 4. Filter by arm hand compatibility
    arms = robot._resolve_arms(arm)
    compatible = []
    for a in arms:
        hand_type = a.config.hand_type
        for aff in affordances:
            if hand_type in aff.compatible_hands:
                compatible.append((a, aff))

    if not compatible:
        return False

    # 5. Try each (arm, affordance) pair
    for a, aff in compatible:
        grasp_tsr = aff.create_tsr(object_pose)
        grasp_tsr = compensate_tsr_for_gripper(grasp_tsr, a.config.hand_type)

        # Plan approach
        result = a.plan_to_tsr(
            grasp_tsr,
            base_heights=base_heights,
            timeout=timeout,
        )

        if result is None:
            continue

        # Execute approach
        ctx.execute(result)

        # Grasp
        ctx.arm(a.side).grasp(target)

        # Lift
        lift_pose = a.get_ee_pose().copy()
        lift_pose[2, 3] += lift_height
        lift_result = a.plan_to_pose(lift_pose, timeout=5.0)
        if lift_result:
            ctx.execute(lift_result)
            robot.grasp_manager.update_attached_poses()

        return True

    return False
```

### 2.2 Place Primitive

```python
def place(
    robot: "Geodude",
    destination: str,
    *,
    arm: "Arm | str | None" = None,
    base_heights: list[float] | None = None,
    timeout: float = 30.0,
) -> bool:
    """Place a held object at a destination.

    Args:
        destination: Destination name (e.g., "recycle_bin_0", "table")
        arm: Arm holding the object, or None to auto-detect
        base_heights: Base heights to search
        timeout: Planning timeout

    Returns:
        True if place succeeded
    """
    ctx = robot._active_context
    if ctx is None:
        raise RuntimeError("No active execution context")

    # 1. Find which arm is holding something
    if arm is None:
        arm = _find_arm_holding_object(robot)
        if arm is None:
            return False
    else:
        arm = robot._resolve_arms(arm)[0]

    # 2. Get held object info
    held_object = robot.grasp_manager.get_grasped_object(arm.side)
    if held_object is None:
        return False

    object_type = _get_object_type(robot, held_object)

    # 3. Get place affordances
    dest_type = _get_object_type(robot, destination)
    affordances = robot.affordances.get_place_affordances(object_type, dest_type)

    if not affordances:
        return False

    # 4. Get destination pose
    dest_pose = robot.get_object_pose(destination)

    # 5. Try each affordance
    for aff in affordances:
        place_tsr = aff.create_tsr(dest_pose)
        place_tsr = compensate_tsr_for_gripper(place_tsr, arm.config.hand_type)

        result = arm.plan_to_tsr(
            place_tsr,
            base_heights=base_heights,
            timeout=timeout,
        )

        if result is None:
            continue

        # Execute place motion
        ctx.execute(result)

        # Release
        ctx.arm(arm.side).release(held_object)

        return True

    return False
```

### 2.3 Robot Integration

```python
# In geodude/src/geodude/robot.py

class Geodude:
    # ... existing code ...

    def pickup(
        self,
        target: str | None = None,
        **kwargs,
    ) -> bool:
        """Pick up an object. See primitives.pickup() for details."""
        from geodude.primitives import pickup
        return pickup(self, target, **kwargs)

    def place(
        self,
        destination: str,
        **kwargs,
    ) -> bool:
        """Place held object. See primitives.place() for details."""
        from geodude.primitives import place
        return place(self, destination, **kwargs)

    def get_pickable_objects(self) -> list[str]:
        """Get names of objects that can be picked up."""
        # Query scene for objects with grasp affordances
        pass

    def get_place_destinations(self, object_type: str) -> list[str]:
        """Get valid place destinations for an object type."""
        pass
```

### 2.4 Arm Integration

```python
# In geodude/src/geodude/arm.py

class Arm:
    # ... existing code ...

    def pickup(
        self,
        target: str | None = None,
        **kwargs,
    ) -> bool:
        """Pick up an object with this arm."""
        from geodude.primitives import pickup
        return pickup(self._robot, target, arm=self, **kwargs)

    def place(
        self,
        destination: str,
        **kwargs,
    ) -> bool:
        """Place held object with this arm."""
        from geodude.primitives import place
        return place(self._robot, destination, arm=self, **kwargs)
```

### 2.5 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/geodude/primitives.py` | CREATE | `pickup()`, `place()` functions |
| `src/geodude/robot.py` | MODIFY | Add `pickup()`, `place()`, query methods |
| `src/geodude/arm.py` | MODIFY | Add `pickup()`, `place()` to Arm |
| `src/geodude/config.py` | MODIFY | Add `hand_type` to ArmConfig |
| `tests/test_primitives.py` | CREATE | Integration tests |

---

## Phase 3: Multi-Robot Abstraction (Future)

### 3.1 Abstract Interfaces

```python
# robot_interface/robot.py (new package, future)

from abc import ABC, abstractmethod

class RobotInterface(ABC):
    """Abstract interface for manipulator robots."""

    @property
    @abstractmethod
    def arms(self) -> list["ArmInterface"]:
        """Available arms."""
        pass

    @property
    @abstractmethod
    def affordances(self) -> "AffordanceRegistry":
        """Affordance registry."""
        pass

    @abstractmethod
    def get_object_pose(self, name: str) -> np.ndarray:
        """Get pose of object in scene."""
        pass


class ArmInterface(ABC):
    """Abstract interface for robot arm."""

    @property
    @abstractmethod
    def hand_type(self) -> str:
        """Hand/gripper type identifier."""
        pass

    @abstractmethod
    def plan_to_tsr(self, tsr, **kwargs) -> "Trajectory | PlanResult | None":
        """Plan to TSR goal."""
        pass


class ExecutionContextInterface(ABC):
    """Abstract interface for execution context."""

    @abstractmethod
    def execute(self, trajectory) -> None:
        """Execute trajectory."""
        pass

    @abstractmethod
    def arm(self, side: str) -> "ArmContextInterface":
        """Get arm context for gripper operations."""
        pass
```

### 3.2 Geodude Implementation

```python
# geodude would implement these interfaces
class Geodude(RobotInterface):
    # Existing implementation satisfies the interface
    pass
```

This phase is deferred—focus on Geodude-specific implementation first, extract interfaces when adding second robot.

---

## Implementation Order

### Week 1: Affordance Registry
1. Create `affordances.py` with `Affordance`, `AffordanceRegistry`
2. Implement template loading from `tsr_templates/`
3. Add hand compatibility mapping
4. Write unit tests
5. Integrate into Geodude

### Week 2: Primitives
1. Create `primitives.py` with `pickup()`, `place()`
2. Add methods to Robot and Arm classes
3. Add `hand_type` to ArmConfig
4. Write integration tests
5. Update recycle demo to use primitives

### Week 3: Polish & Documentation
1. Add query methods (`get_pickable_objects()`, etc.)
2. Error handling and retry logic
3. Update README with primitives documentation
4. Create example script showing primitives API

---

## Testing Strategy

### Unit Tests (test_affordances.py)
```python
def test_load_templates():
    """Registry loads templates from directory."""

def test_get_grasp_affordances():
    """Returns affordances for object type."""

def test_hand_compatibility_filter():
    """Filters affordances by hand type."""

def test_create_tsr_from_affordance():
    """Creates concrete TSR at object pose."""
```

### Integration Tests (test_primitives.py)
```python
def test_pickup_specific_object():
    """robot.pickup("can_0") picks up the can."""

def test_pickup_any_can():
    """robot.pickup(object_type="can") picks up any can."""

def test_place_in_bin():
    """robot.place("recycle_bin_0") places held object."""

def test_pickup_place_cycle():
    """Full pickup → place cycle works."""
```

---

## Open Questions

1. **TSR storage location**: Keep in geodude/tsr_templates or move to prl_assets?
   - Recommendation: Keep geodude-specific TSRs in geodude, plan for object-local TSRs in prl_assets later

2. **Hand type naming**: Use exact gripper model (`robotiq_2f_140`) or abstract category (`parallel_jaw`)?
   - Recommendation: Exact model for now, add abstraction later if needed

3. **Error recovery**: How much retry logic in primitives vs. user code?
   - Recommendation: Basic retry (try all affordances), expose hooks for custom recovery

4. **Object type inference**: How to get object type from instance name ("can_0" → "can")?
   - Recommendation: Strip numeric suffix, or lookup in registry

---

## Success Criteria

- [ ] `robot.pickup("can_0")` works in recycling demo
- [ ] `robot.place("recycle_bin_0")` works in recycling demo
- [ ] Same code runs with `robot.sim()` and `robot.hardware()` contexts
- [ ] Affordance registry discovers TSRs automatically
- [ ] Unit test coverage for registry and primitives
