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
- **Execution context**: Same code works for simulation and hardware deployment
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

# Run in simulation context
with robot.sim() as ctx:
    robot.go_to("ready")
    ctx.sync()

    # Plan a motion (returns Trajectory)
    import numpy as np
    goal = np.array([-1.0, -1.5, 1.5, -1.5, -1.5, 0])
    trajectory = robot.right_arm.plan_to(goal)

    # Execute via context
    ctx.execute(trajectory)

    # Gripper control via context
    ctx.arm("right").grasp("object_name")
    ctx.arm("right").release("object_name")
```

## Execution Context

The execution context provides a unified interface for simulation and hardware:

```python
# Simulation (kinematic - instant, perfect tracking)
with robot.sim(physics=False) as ctx:
    trajectory = robot.right_arm.plan_to(goal)
    ctx.execute(trajectory)
    ctx.sync()  # Sync viewer

# Simulation (physics - realistic dynamics)
with robot.sim(physics=True) as ctx:
    trajectory = robot.right_arm.plan_to(goal)
    ctx.execute(trajectory)
```

The context handles:
- **Trajectory execution**: `ctx.execute(trajectory)` or `ctx.execute(plan_result)`
- **Gripper operations**: `ctx.arm("right").grasp("can")`, `ctx.arm("left").release("box")`
- **Viewer sync**: `ctx.sync()` updates the MuJoCo viewer
- **Run loop**: `while ctx.is_running():` continues until viewer is closed

## TSR-Based Planning

Plan to grasp regions instead of fixed poses using Task Space Regions:

```python
from geodude.tsr_utils import create_side_grasp_tsr

# Get object pose
obj_pose = robot.get_object_pose("can")

# Create grasp TSR (allows rotation around object axis)
grasp_tsr = create_side_grasp_tsr(obj_pose, object_height=0.12)

# Plan to any valid grasp (returns Trajectory)
with robot.sim() as ctx:
    trajectory = robot.right_arm.plan_to_tsr(grasp_tsr)
    ctx.execute(trajectory)
    ctx.arm("right").grasp("can")
```

## Unified Planning API

The `plan_to_tsr()` method handles arm selection and base height search automatically:

```python
with robot.sim() as ctx:
    # Plan with both arms at multiple base heights
    # Default: randomly picks first arm, interleaves at each height level
    result = robot.plan_to_tsr(
        grasp_tsr,
        base_heights=[0.2, 0.0, 0.4],  # Middle height first (most versatile)
    )

    if result:
        print(f"Success: {result.arm.side} @ {result.base_height}m")
        ctx.execute(result)  # Executes base + arm trajectories
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
# Returns PlanResult with arm_trajectory and base_trajectory
```

## Grasp Management

When you grasp an object, collision checking updates automatically:

```python
with robot.sim() as ctx:
    # Grasp via context (recommended)
    ctx.arm("right").grasp("can")

    # Now planning treats the can as part of the robot
    # (won't report false collisions with the arm)
    place_traj = robot.right_arm.plan_to_tsr(place_tsr)
    ctx.execute(place_traj)

    # Release via context
    ctx.arm("right").release("can")
```

For manual control:

```python
# Mark object as grasped
robot.grasp_manager.mark_grasped("can", "right")
robot.grasp_manager.attach_object("can", "right_ur5e/gripper/right_follower")

# Release
robot.grasp_manager.mark_released("can")
robot.grasp_manager.detach_object("can")
```

## Architecture

### Component Overview

```
Geodude
├── left_arm / right_arm (Arm)
│   ├── Planning (CBiRRT + EAIK)
│   ├── plan_to(), plan_to_tsr(), plan_to_pose()
│   └── Gripper control
├── left_base / right_base (VentionBase)
│   └── Height adjustment with collision checking
├── GraspManager
│   └── Tracks grasped objects, updates collision groups
├── SimContext (via robot.sim())
│   └── Unified execution for simulation
└── Collision checkers
    └── Grasp-aware collision checking
```

### Ownership Model

The framework separates **planning** (what to do) from **execution** (how to do it):

| Component | Owns | Responsibilities |
|-----------|------|------------------|
| **Robot/Arm** | Planning & State | Motion planning, IK, collision checking, grasp management, TSR registry |
| **Context** | Execution | Trajectory execution, gripper actuation, state synchronization, hardware abstraction |

**Robot/Arm** is stateless with respect to execution—it plans trajectories but doesn't know whether they'll run in simulation or on hardware. This keeps planning logic portable.

**Context** handles the messy reality of execution: timing, physics, hardware communication, error recovery. Different contexts (sim, hardware) implement the same interface.

**The context manager** (`robot.sim()`, `robot.hardware()`) wires them together:

```python
with robot.sim() as ctx:
    # Robot plans trajectories (doesn't know about sim vs hardware)
    trajectory = robot.right_arm.plan_to(goal)

    # Context executes them (knows exactly how)
    ctx.execute(trajectory)

    # Gripper ops go through context (updates robot's grasp state)
    ctx.arm("right").grasp("can")
```

### Design Principles

The architecture is designed for **multi-robot reusability**:

1. **Robots implement interfaces, not inheritance** — Different robots (Geodude, HERB, Fetch) can implement the same planning/execution interfaces without sharing a base class.

2. **TSRs live with objects, not robots** — Grasp and place TSRs are stored in `prl_assets` alongside object models. Any robot with a compatible hand can use them.

3. **Hand compatibility, not robot compatibility** — TSRs declare which hands they work with (`robotiq_2f_140`, `wsg_50`). The robot queries its hand type to find compatible TSRs.

4. **Context abstracts deployment** — The same manipulation code runs in kinematic sim, physics sim, or hardware by changing only the context.

This separation allows:
- Reusable manipulation primitives that work across robots
- Objects that "just work" when added to the asset manager
- Easy testing (kinematic sim) before deployment (hardware)

## Examples

```bash
# Planning with base height search
uv run mjpython examples/arm_planning.py

# Pick and place recycling demo
uv run mjpython examples/recycle_objects.py
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
