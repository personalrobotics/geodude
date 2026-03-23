# Geodude

Bimanual manipulation with the Geodude robot, built on [mj_manipulator](https://github.com/siddhss5/mj_manipulator).

## The Robot

```
                ┌─────────────────────────────────────┐
                │           Vention Frame             │
                └─────────────────────────────────────┘
                     │                       │
                ┌────┴────┐             ┌────┴────┐
                │ Linear  │             │ Linear  │
                │ Rail    │             │ Rail    │
                │ (0-50cm)│             │ (0-50cm)│
                └────┬────┘             └────┬────┘
                     │                       │
                ┌────┴────┐             ┌────┴────┐
                │  UR5e   │             │  UR5e   │
                │  Left   │             │  Right  │
                └────┬────┘             └────┬────┘
                     │                       │
                ┌────┴────┐             ┌────┴────┐
                │ Robotiq │             │ Robotiq │
                │ 2F-140  │             │ 2F-140  │
                └─────────┘             └─────────┘
```

- **2× UR5e arms** — 6-DOF manipulators (from mj_manipulator)
- **2× Vention linear actuators** — Height-adjustable bases (0–50cm)
- **2× Robotiq 2F-140 grippers** — Parallel-jaw, 140mm stroke (from mj_manipulator)

## Quick Start

```python
from geodude import Geodude

robot = Geodude(objects={"can": 1, "recycle_bin": 2})

with robot.sim() as ctx:
    robot.pickup("can_0")
    robot.place("recycle_bin_0")
    robot.go_home()
```

That's it. TSR generation, planning, execution, grasp detection, and recovery are all automatic.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Your code                                           │
│  robot.pickup("can_0")                               │
│  robot.place("recycle_bin_0")                        │
│  robot.go_home()                                     │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────┐
│  geodude  (this package)                             │
│  • Geodude class — compose two Arms + VentionBases   │
│  • py_trees behavior trees — pickup/place with       │
│    automatic recovery, bimanual arm selection         │
│  • Auto TSR generation from prl_assets geometry      │
│  • VentionBase — linear actuator with collision check│
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────┐
│  mj_manipulator  (generic manipulation)              │
│  • Arm, SimContext, ExecutionContext protocol         │
│  • BT leaf nodes (PlanToTSRs, Execute, Grasp, ...)   │
│  • CBiRRT planning, EAIK inverse kinematics          │
│  • CartesianController, GraspManager                 │
│  • RobotiqGripper, FrankaGripper                     │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────┐
│  tsr + prl_assets  (objects + geometry)               │
│  • tsr.hands.Robotiq2F140.grasp_cylinder_side(r, h)  │
│  • prl_assets: can, recycle_bin, ... with meta.yaml   │
└──────────────────────────────────────────────────────┘
```

## Recycling Demo

```bash
uv run mjpython examples/recycle.py
uv run mjpython examples/recycle.py --physics
uv run mjpython examples/recycle.py --headless --cycles 5
```

## Debugging

Pass `verbose=True` to see the behavior tree status after execution:

```python
robot.pickup("can_0", verbose=True)
```

```
{-} geodude_pickup [o]
    --> GenerateGrasps [o]
    {o} pickup_or_recover [o]
        {-} pickup [o]
            {-} plan_and_execute [o]
                --> PlanToTSRs [o]
                --> Retime [o]
                --> Execute [o]
            --> Sync [o]
            --> Grasp [o]
            ...
```

`[o]` = success, `[x]` = failure (with reason), `[-]` = not reached.

Enable globally:

```python
robot.config.debug.verbose = True  # all primitives show tree status
robot.config.debug.enable_all()    # verbose + all debug logging
```

## Bimanual Planning

The robot-level planner tries both arms with optional base height search:

```python
result = robot.plan_to_tsrs(grasp_tsrs, base_heights=[0.2, 0.0, 0.4])
if result is not None:
    ctx.execute(result)
```

## Configuration

```python
# Planning parameters (single source of truth)
robot.config.planning.timeout = 60.0        # seconds per planning attempt
robot.config.planning.base_heights = [0.2]  # heights to search
robot.config.planning.lift_height = 0.10    # meters to lift after grasping
```

## Package Structure

```
src/geodude/
├── robot.py          # Geodude class — bimanual composition
├── config.py         # PlanningConfig, VentionBaseConfig, DebugConfig
├── primitives.py     # pickup() / place() / go_home() — BT-backed
├── bt/
│   ├── nodes.py      # GenerateGrasps, GenerateDropZone
│   └── subtrees.py   # geodude_pickup, geodude_place
├── vention_base.py   # Linear actuator planning + collision checking
└── __init__.py       # Public API + mj_manipulator re-exports
```

## Testing

```bash
uv run pytest tests/ -v
```

## Dependencies

**Workspace packages:**

- [mj_manipulator](https://github.com/siddhss5/mj_manipulator) — Arm control, planning, execution, BT leaf nodes
- [geodude_assets](https://github.com/personalrobotics/geodude_assets) — MuJoCo models (UR5e + Robotiq)
- [prl_assets](https://github.com/personalrobotics/prl_assets) — Object models with geometry metadata
- [tsr](https://github.com/personalrobotics/tsr) — Task Space Regions + grasp generation
- [pycbirrt](https://github.com/personalrobotics/pycbirrt) — CBiRRT motion planner
- [mj_environment](https://github.com/personalrobotics/mj_environment) — MuJoCo environment wrapper
- [asset_manager](https://github.com/personalrobotics/asset_manager) — Object metadata loader

**External:**

- [py_trees](https://github.com/splintered-reality/py_trees) — Behavior tree engine
- [eaik](https://github.com/Verdant-Robotics/eaik) — Analytical IK for UR robots
- [mujoco](https://github.com/google-deepmind/mujoco) — Physics simulation
