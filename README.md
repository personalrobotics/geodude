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
    robot.place("recycle_bin")    # drop into any bin
    robot.place("worktop")       # place on the table
    robot.place("sugar_box")     # place on top of a sugar box
    robot.go_home()
```

That's it. Object discovery, TSR generation, planning, execution, grasp detection, and recovery are all automatic.

## Smart Object Resolution

Primitives accept instance names, type names, or nothing:

```python
robot.pickup("can_0")         # specific instance
robot.pickup("can")           # any can in the scene
robot.pickup()                # anything graspable

robot.place("recycle_bin_0")  # drop into a specific bin
robot.place("recycle_bin")    # drop into any bin
robot.place("sugar_box")     # place on top of any sugar box
robot.place("worktop")       # place on the table surface
robot.place()                 # auto-select: containers first, then worktop
```

All matching objects' TSRs are combined and sent to the planner — it picks whichever is easiest to reach.

## Placement

`place()` supports three destination types:

| Destination | Example | Behavior |
|---|---|---|
| Container (bin, tote) | `robot.place("recycle_bin")` | Drop from above, object removed from scene |
| Surface (any flat-topped object) | `robot.place("sugar_box")` | Stable placement on top, object stays in scene |
| Worktop | `robot.place("worktop")` | Place on the table surface |
| Auto | `robot.place()` | Tries containers first, then worktop |

Any object with an upward-facing flat face (box top, cylinder end) is a valid placement surface. The system automatically:
- Enumerates flat faces from the destination's geometry
- Filters to faces within ~18 degrees of vertical
- Computes the grasp offset so the held object lands upright
- Adds clearance so the planner avoids collision with the surface

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
uv run python -m geodude --demo recycling                # headless console
uv run python -m geodude --demo recycling --viser        # browser viewer (http://localhost:8080)
uv run mjpython -m geodude --demo recycling --viewer     # native MuJoCo viewer (macOS: requires mjpython)
uv run python -m geodude --list-demos                    # see available demos
```

### Python API

```python
In [1]: robot.pickup()               # pick up nearest object
In [2]: robot.place("recycle_bin")   # drop in any bin
In [3]: robot.place("worktop")      # place on the table
In [4]: sort_all()                   # run the demo's built-in function
In [5]: reset()                      # restart the demo
In [6]: commands()                   # quick reference
```

### LLM Chat

Natural language control via Claude (requires `ANTHROPIC_API_KEY` + `uv sync --extra chat`).

```python
In [1]: /chat "what's on the table?"
  -> get_objects({})
Geodude: There are 2 cans, a sugar box, and 2 recycle bins on the table.

In [2]: /chat "recycle all the cans"
  -> pickup({"target": "can"})
  -> place({"destination": "recycle_bin"})
  -> pickup({"target": "can"})
  -> place({"destination": "recycle_bin"})
Geodude: Done! Both cans have been recycled.

In [3]: /chat "stack the sugar box on the pop tarts case"
  -> pickup({"target": "sugar_box"})
  -> place({"destination": "pop_tarts_case"})
Geodude: Placed the sugar box on top of the pop tarts case.

In [4]: /chat "reset with 3 cans and a sugar box"
  -> reset_scene({"objects": {"can": 3, "sugar_box": 1}})
Geodude: Scene reset with 4 objects.
```

The LLM sees planning diagnostics behind the scenes (IK failures, collision details, grasp results) and adjusts its strategy automatically.

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
- [mj_viser](https://github.com/personalrobotics/mj_viser) — Browser-based viewer (optional, for `--viser`)
- [asset_manager](https://github.com/personalrobotics/asset_manager) — Object metadata loader

**External:**

- [py_trees](https://github.com/splintered-reality/py_trees) — Behavior tree engine
- [eaik](https://github.com/Verdant-Robotics/eaik) — Analytical IK for UR robots
- [mujoco](https://github.com/google-deepmind/mujoco) — Physics simulation
