# Geodude

A Python library for bimanual robot manipulation with collision-free motion planning.

<p align="center">
  <img src="docs/images/recycle_demo.gif" alt="Pick and place demo" width="480">
</p>

## What It Does

Geodude controls a bimanual UR5e robot system—two arms on height-adjustable Vention rails with Robotiq grippers. It handles the hard parts of manipulation:

- **Motion planning**: Find collision-free paths using CBiRRT with TSR goals
- **Grasp-aware collision**: Objects you're holding don't collide with your arm
- **Unified planning API**: Simple `plan_to()` with automatic arm selection and height search
- **Time-optimal trajectories**: TOPP-RA retiming respects joint velocity/acceleration limits

## Installation

```bash
uv add geodude geodude_assets
```

## Quick Start

```python
from geodude import Geodude

# Initialize robot (loads MuJoCo model)
robot = Geodude()

# Move to a named pose
robot.go_to("ready")

# Plan and execute a motion
import numpy as np
goal = np.array([-1.0, -1.5, 1.5, -1.5, -1.5, 0])
path = robot.right_arm.plan_to_configuration(goal)
if path:
    robot.right_arm.execute(path)

# Gripper control
robot.right_arm.close_gripper()
robot.right_arm.open_gripper()
```

## TSR-Based Planning

Plan to grasp regions instead of fixed poses using Task Space Regions:

```python
from geodude.tsr_utils import create_side_grasp_tsr

# Get object pose
obj_pose = robot.get_object_pose("can")

# Create grasp TSR (allows rotation around object axis)
grasp_tsr = create_side_grasp_tsr(obj_pose, object_height=0.12)

# Plan to any valid grasp
path = robot.right_arm.plan_to_tsrs([grasp_tsr])
if path:
    robot.right_arm.execute(path)
    robot.right_arm.close_gripper()
```

## Unified Planning API

The `plan_to_tsr()` method handles arm selection and base height search automatically:

```python
# Plan with both arms at multiple base heights
# Default: randomly picks first arm, interleaves at each height level
result = robot.plan_to_tsr(
    grasp_tsr,
    base_heights=[0.2, 0.0, 0.4],  # Middle height first (most versatile)
    execute=False,
)

if result:
    print(f"Success: {result.arm.config.name} @ {result.base_height}m")
    # Execute manually or use execute=True
```

For explicit control over the search order:

```python
# Explicit (arm, height) sequence
result = robot.plan_to_tsr(
    grasp_tsr,
    sequence=[
        ("right", 0.2),
        ("left", 0.2),
        ("right", 0.0),
        ("left", 0.0),
    ],
)
```

Single-arm planning with height search:

```python
# Plan with one arm at multiple heights
result = robot.right_arm.plan_to_tsr(
    grasp_tsr,
    base_heights=[0.2, 0.0, 0.4],
)
```

## Grasp Management

When you grasp an object, collision checking updates automatically:

```python
# Mark object as grasped
robot.grasp_manager.mark_grasped("can", "right")
robot.grasp_manager.attach_object("can", "right_ur5e/gripper/right_follower")

# Now planning treats the can as part of the robot
# (won't report false collisions with the arm)
path = robot.right_arm.plan_to_tsrs([place_tsr])

# Release
robot.grasp_manager.mark_released("can")
robot.grasp_manager.detach_object("can")
```

## Execution Modes

```python
# Physics simulation (default) - realistic dynamics
robot.right_arm.execute(path)

# Kinematic - instant, perfect tracking for validation
robot.right_arm.execute(path, executor_type="kinematic")
```

## Architecture

```
Geodude
├── left_arm / right_arm (Arm)
│   ├── Planning (CBiRRT + EAIK)
│   ├── Execution (TOPP-RA + MuJoCo)
│   └── Gripper control
├── left_base / right_base (VentionBase)
│   └── Height adjustment
├── GraspManager
│   └── Tracks grasped objects, updates collision groups
└── Collision checkers
    └── Grasp-aware, thread-safe for parallel planning
```

## Examples

```bash
# Parallel planning with base heights
uv run python examples/arm_planning.py

# Pick and place with physics
uv run python examples/recycle_objects.py --physics
```

## Testing

```bash
uv run pytest
```

## Dependencies

- **pycbirrt**: CBiRRT motion planner with TSR constraints
- **tsr**: Task Space Region definitions
- **eaik**: Analytical inverse kinematics for UR robots
- **toppra**: Time-optimal path parameterization
- **mujoco**: Physics simulation
- **geodude_assets**: Robot models and meshes

## License

MIT
