# Geodude

Bimanual manipulation with the Geodude robot, built on [mj_manipulator](https://github.com/personalrobotics/mj_manipulator).

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

robot = Geodude(objects={"can": 3, "recycle_bin": 2})

with robot.sim() as ctx:
    robot.pickup("can")           # pick up any can
    robot.place("recycle_bin")    # place in any bin
    robot.go_home()
```

That's it. Object discovery, TSR generation, planning, execution, grasp detection, and recovery are all automatic.

## Smart Object Resolution

Primitives accept instance names, type names, or nothing:

```python
robot.pickup("can_0")         # specific instance
robot.pickup("can")           # any can in the scene
robot.pickup()                # anything graspable

robot.place("recycle_bin_0")  # specific bin
robot.place("recycle_bin")    # any bin
robot.place()                 # any valid destination
```

All matching objects' TSRs are combined and sent to the planner — it picks whichever is easiest to reach.

## Arm Selection

By default, both arms are tried (random order). Specify an arm explicitly:

```python
robot.right.pickup("can")        # right arm only
robot.left.place("recycle_bin")   # left arm only
robot.right.go_home()             # right arm only

robot.pickup("can", arm="right")  # equivalent to above
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Your code                                           │
│  robot.pickup("can")                                 │
│  robot.right.place("recycle_bin")                    │
│  robot.go_home()                                     │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────┐
│  geodude  (this package)                             │
│  • Smart object resolution (instance/type/any)       │
│  • py_trees behavior trees with automatic recovery   │
│  • Bimanual arm selection + arm-scoped primitives    │
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

## Interactive Console

The primary way to use Geodude. IPython REPL with tab completion, optional LLM chat, and a demo system.

```bash
geodude --demo recycling                              # headless console
geodude --list-demos                                  # see available demos
uv run mjpython -m geodude --demo recycling --viewer  # with MuJoCo viewer
```

```python
In [1]: robot.pickup()               # pick up nearest object
In [2]: robot.place("recycle_bin")   # place in any bin
In [3]: sort_all()                   # run the demo's built-in function
In [4]: reset()                      # restart the demo
In [5]: commands()                   # quick reference

# Natural language (requires ANTHROPIC_API_KEY + uv sync --extra chat)
In [6]: chat('clear the table')
```

### Creating demos

Demos are Python files in `src/geodude/demos/`. Create one interactively with `save_demo('name')`, or write a file:

```python
# demos/my_task.py
"""My task — do something useful."""
scene = {"objects": {"can": 4}, "fixtures": {}}

def do_task():
    while robot.pickup("can"):
        robot.place()
```

Then: `geodude --demo my_task`

## Debugging

Pass `verbose=True` to see the behavior tree status after execution:

```python
robot.pickup("can", verbose=True)
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

## Configuration

```python
robot.config.planning.timeout = 60.0        # seconds per planning attempt
robot.config.planning.base_heights = [0.2]  # heights to search
robot.config.planning.lift_height = 0.10    # meters to lift after grasping
```

## Package Structure

```
src/geodude/
├── robot.py          # Geodude class + _ArmScope for robot.right/left
├── config.py         # PlanningConfig, VentionBaseConfig, DebugConfig
├── primitives.py     # pickup() / place() / go_home() — BT-backed
├── bt/
│   ├── nodes.py      # GenerateGrasps, GeneratePlaceTSRs + smart resolution
│   └── subtrees.py   # geodude_pickup, geodude_place
├── vention_base.py   # Linear actuator planning + collision checking
├── cli.py            # geodude CLI entry point
├── console.py        # IPython console with chat, demos, save_demo
├── chat.py           # LLM chat integration (ChatSession, tools)
├── demo_loader.py    # Demo discovery, loading, scene setup
├── demos/            # Demo files (recycling.py, ...)
└── __init__.py       # Public API + mj_manipulator re-exports
```

## Testing

```bash
uv run pytest tests/ -v
```

## Dependencies

**Workspace packages:**

- [mj_manipulator](https://github.com/personalrobotics/mj_manipulator) — Arm control, planning, execution, BT leaf nodes
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
