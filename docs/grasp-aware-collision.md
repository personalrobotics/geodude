# Grasp-Aware Collision Detection in MuJoCo

This document explains how Geodude manages collision detection when grasping objects. Understanding this system is essential for implementing robust pick-and-place manipulation.

## The Problem

When a robot grasps an object, we face a collision detection dilemma:

```
Before Grasp:                    After Grasp:

    Gripper                          Gripper
      |                                |
      v                                v
    [   ]                            [CAN]  <-- Object now "part of" robot
                                       |
                                       v
    [CAN]                           Arm links below
      |
      v
    Table
```

**Without special handling**, the grasped object would be in constant collision with the gripper fingers. Worse, as the arm moves, the object might intersect with the robot's forearm or other links during motion planning.

**The naive solution** of simply ignoring all collisions with the grasped object is dangerous - we still need to detect collisions between the grasped object and the environment (tables, bins, obstacles).

## Geodude's Solution: Software-Based Contact Filtering

Instead of manipulating MuJoCo's collision groups (which adds complexity and can cause bugs), Geodude uses **software-based contact filtering**. All collision checking logic lives in the `_count_invalid_contacts()` method.

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     Collision Check Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Set robot configuration (joints, base height)               │
│                         │                                       │
│                         v                                       │
│  2. Update grasped object poses (move with gripper)            │
│                         │                                       │
│                         v                                       │
│  3. Call mj_forward() → MuJoCo generates ALL contacts          │
│                         │                                       │
│                         v                                       │
│  4. Filter contacts in software:                                │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  For each contact:                                   │    │
│     │    • Gripper ↔ Grasped Object → ALLOWED             │    │
│     │    • Arm ↔ Grasped Object → INVALID                 │    │
│     │    • Grasped Object ↔ Environment → INVALID         │    │
│     │    • Everything else → INVALID                       │    │
│     └─────────────────────────────────────────────────────┘    │
│                         │                                       │
│                         v                                       │
│  5. Return: invalid_contacts == 0                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Filtering Logic

```python
def _count_invalid_contacts(self, data: mujoco.MjData) -> int:
    """Count contacts that represent actual collisions."""
    invalid = 0

    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = self.model.geom_bodyid[contact.geom1]
        body2 = self.model.geom_bodyid[contact.geom2]

        # Get body names
        name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
        name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)

        # Check if either body is grasped
        grasped1 = self.grasp_manager.is_grasped(name1) if name1 else False
        grasped2 = self.grasp_manager.is_grasped(name2) if name2 else False

        if grasped1 or grasped2:
            # One body is grasped - check if contact is with gripper
            grasped_name = name1 if grasped1 else name2
            other_name = name2 if grasped1 else name1

            # Allow gripper-to-grasped-object contacts
            if self._is_gripper_body(other_name, grasped_name):
                continue  # Valid contact

            # Any other contact with grasped object is invalid
            invalid += 1
        else:
            # Neither is grasped - any contact is invalid
            invalid += 1

    return invalid
```

### Why This Approach?

| Approach | Pros | Cons |
|----------|------|------|
| **Collision Group Manipulation** | Uses MuJoCo's native filtering | Complex state management, bugs when groups get out of sync, need to temporarily restore for planning |
| **Software Filtering** (Geodude's choice) | Simple, explicit, debuggable | Slightly more contacts to process (but negligible overhead) |

The software approach is simpler because:
1. **No state to manage** - Collision groups stay constant
2. **No bugs from group mismatch** - Logic is in one place
3. **Easy to debug** - Can print exactly why a contact was allowed/rejected

## MuJoCo's Collision System (Background)

For reference, MuJoCo uses bitmask-based collision filtering:

| Property | Description |
|----------|-------------|
| `contype` | "Contact type" - what collision group this geometry belongs to |
| `conaffinity` | "Contact affinity" - which groups this geometry can collide with |

MuJoCo generates a contact between geometry A and B if:
```
(A.contype & B.conaffinity) || (B.contype & A.conaffinity) != 0
```

Geodude leaves all objects with their default collision groups (`contype=1, conaffinity=1`) and handles filtering in software instead.

## The GraspManager

The `GraspManager` class tracks grasp state and handles kinematic attachments:

```python
class GraspManager:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.grasped: dict[str, str] = {}  # object_name -> arm_name
        self._attachments: dict[str, tuple[str, np.ndarray]] = {}
```

### Marking Objects as Grasped

When `mark_grasped()` is called, we simply record which arm is holding the object:

```python
def mark_grasped(self, object_name: str, arm: str) -> None:
    """Mark an object as grasped by the specified arm."""
    if object_name in self.grasped:
        return  # Already grasped
    self.grasped[object_name] = arm

def mark_released(self, object_name: str) -> None:
    """Mark an object as released."""
    if object_name not in self.grasped:
        return  # Not currently grasped
    del self.grasped[object_name]
```

The collision checker uses `grasp_manager.is_grasped()` to determine how to filter contacts.

## Kinematic Attachments

For kinematic (non-physics) execution, grasped objects don't automatically move with the gripper. The `GraspManager` handles **kinematic attachments**:

```python
def attach_object(self, object_name: str, gripper_body_name: str) -> None:
    """Compute and store the relative transform T_gripper_object."""
    T_world_gripper = self._get_body_pose(gripper_body_name)
    T_world_object = self._get_body_pose(object_name)

    # Relative transform: where is object in gripper frame?
    T_gripper_object = np.linalg.inv(T_world_gripper) @ T_world_object

    self._attachments[object_name] = (gripper_body_name, T_gripper_object)

def update_attached_poses(self, data: mujoco.MjData = None) -> None:
    """Move attached objects with their grippers."""
    for object_name, (gripper_body_name, T_gripper_object) in self._attachments.items():
        T_world_gripper = self._get_body_pose_from_data(gripper_body_name, data)
        T_world_object = T_world_gripper @ T_gripper_object
        self._set_body_pose_in_data(object_name, T_world_object, data)
```

During collision checking, `update_attached_poses()` is called to move grasped objects with the gripper before checking contacts.

## Complete Grasp Lifecycle

```
1. APPROACH
   ┌─────────────────────────────────────────────────────┐
   │  Object is not grasped                              │
   │  Robot plans path avoiding object (normal contact)  │
   └─────────────────────────────────────────────────────┘
                            │
                            v
2. GRASP DETECTED (gripper closes, contacts detected)
   ┌─────────────────────────────────────────────────────┐
   │  grasp_manager.mark_grasped("can_0", "right")      │
   │  grasp_manager.attach_object("can_0", gripper)     │
   │                                                     │
   │  Object now tracked as grasped by right arm        │
   └─────────────────────────────────────────────────────┘
                            │
                            v
3. MANIPULATION (lift, move, place planning)
   ┌─────────────────────────────────────────────────────┐
   │  Collision checker:                                 │
   │    • Allows gripper ↔ can contacts                 │
   │    • Rejects arm ↔ can contacts                    │
   │    • Rejects can ↔ environment contacts            │
   │                                                     │
   │  update_attached_poses() keeps object with gripper │
   └─────────────────────────────────────────────────────┘
                            │
                            v
4. RELEASE
   ┌─────────────────────────────────────────────────────┐
   │  grasp_manager.detach_object("can_0")              │
   │  grasp_manager.mark_released("can_0")              │
   │                                                     │
   │  Object no longer tracked - normal collision rules │
   └─────────────────────────────────────────────────────┘
```

## Summary

| Concept | Purpose |
|---------|---------|
| `GraspManager.grasped` | Tracks which objects are held by which arm |
| `GraspManager.is_grasped()` | Query used by collision checker |
| `_count_invalid_contacts()` | Software filtering of MuJoCo contacts |
| Gripper ↔ Grasped allowed | Prevents false positives from finger contact |
| Arm ↔ Grasped rejected | Detects self-collision during motion |
| Grasped ↔ Environment rejected | Detects obstacle collision during place |
| `attach_object()` | Set up kinematic attachment for object following |
| `update_attached_poses()` | Move grasped objects with gripper during planning |

## Further Reading

- [MuJoCo Documentation: Contact](https://mujoco.readthedocs.io/en/stable/modeling.html#contact)
- [grasp_manager.py](../src/geodude/grasp_manager.py) - Grasp state and attachment tracking
- [collision.py](../src/geodude/collision.py) - Collision checker with software filtering
