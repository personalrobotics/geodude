# Geodude

A Python library for bimanual robot manipulation with collision-free motion planning.

<p align="center">
  <img src="docs/images/geodude.png" alt="Geodude bimanual robot" width="500">
</p>

## What It Does

Geodude controls a bimanual UR5e robot system—two arms on height-adjustable Vention rails with Robotiq grippers. It handles the hard parts of manipulation:

- **Motion planning**: Find collision-free paths using CBiRRT with TSR goals
- **Grasp-aware collision**: Objects you're holding don't collide with your arm
- **Parallel planning**: Plan both arms simultaneously, or try multiple goals at once
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

## Parallel Planning

Plan multiple goals simultaneously—first success wins:

```python
from geodude import plan_first_success

# Try multiple grasp approaches in parallel
path = plan_first_success(robot.right_arm, [tsr1, tsr2, tsr3], timeout=10.0)
```

Plan both arms at once:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    left = executor.submit(
        lambda: robot.left_arm.create_planner().plan(start, goal=left_goal)
    )
    right = executor.submit(
        lambda: robot.right_arm.create_planner().plan(start, goal=right_goal)
    )
    left_path, right_path = left.result(), right.result()
```

## Height-Adaptive Planning

The Vention bases allow height adjustment. Plan at multiple heights to find reachable goals:

```python
from geodude.parallel import plan_with_base_heights

# Try planning at different base heights
heights = [0.0, 0.2, 0.4]
winning_height, path = plan_with_base_heights(
    robot.right_arm,
    robot.right_base,
    grasp_tsr,
    heights
)
if path:
    robot.right_base.set_height(winning_height)
    robot.right_arm.execute(path)
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
# Basic motion demonstration
uv run python examples/basic_movement.py

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
