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

## Architecture

Geodude is a thin bimanual orchestration layer (~1,300 LOC) on top of [mj_manipulator](https://github.com/siddhss5/mj_manipulator), which provides all generic manipulation:

```
┌──────────────────────────────────────────────────────┐
│  Your code / Demo scripts                            │
│  robot.pickup("can_0", grasp_tsrs)                   │
│  robot.place(drop_tsrs)                              │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────┐
│  geodude  (this package)                             │
│  • Geodude class — compose two Arms + VentionBases   │
│  • Bimanual planning — arm/height interleaving       │
│  • Primitives — pickup/place with explicit TSRs      │
│  • VentionBase — linear actuator with collision check│
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────┐
│  mj_manipulator  (generic manipulation)              │
│  • Arm, SimContext, ExecutionContext protocol         │
│  • CBiRRT planning, EAIK inverse kinematics          │
│  • CartesianController, GraspManager                 │
│  • RobotiqGripper, FrankaGripper                     │
│  • Trajectory retiming (TOPP-RA)                     │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────┐
│  tsr + prl_assets  (objects + geometry)               │
│  • TSR templates generated from object geometry       │
│  • tsr.hands.Robotiq2F140.grasp_cylinder(r, h)       │
│  • prl_assets: can, recycle_bin, ... with meta.yaml   │
└──────────────────────────────────────────────────────┘
```

## Installation

```bash
# In the robot-code workspace
git clone https://github.com/siddhss5/robot-code && ./setup.sh
```

## Quick Start

```python
from geodude import Geodude

robot = Geodude()

with robot.sim(physics=False) as ctx:
    # Plan and execute a joint-space motion
    path = robot.right_arm.plan_to_configuration(goal_q)
    traj = robot.right_arm.retime(path)
    ctx.execute(traj)

    # Gripper control
    ctx.arm("right").grasp("object_name")
    ctx.arm("right").release()
```

## Recycling Demo

The full pick-and-place demo generates TSRs programmatically from object geometry in [prl_assets](https://github.com/personalrobotics/prl_assets):

```bash
uv run mjpython examples/recycle.py
uv run mjpython examples/recycle.py --physics
uv run mjpython examples/recycle.py --headless --cycles 5
```

```python
from tsr.hands import Robotiq2F140
from asset_manager import AssetManager
from prl_assets import OBJECTS_DIR

# Read object geometry from prl_assets metadata
assets = AssetManager(str(OBJECTS_DIR))
can_gp = assets.get("can")["geometric_properties"]  # radius, height

# Generate grasp TSRs from geometry (no YAML templates needed)
hand = Robotiq2F140()
templates = hand.grasp_cylinder_side(can_gp["radius"], can_gp["height"])
grasp_tsrs = [t.instantiate(can_bottom_pose) for t in templates]

# Pick and place
with robot.sim() as ctx:
    robot.pickup("can_0", grasp_tsrs)
    robot.place(drop_tsrs)
```

## Bimanual Planning

The robot-level planner tries both arms with optional base height search:

```python
# Try both arms, interleaved at each height (randomized order)
result = robot.plan_to_tsrs(
    grasp_tsrs,
    base_heights=[0.2, 0.0, 0.4],
)

# Or specify a single arm
result = robot.plan_to_tsrs(grasp_tsrs, arm="right")

if result is not None:
    ctx.execute(result)
```

## Vention Base

Height-adjustable bases expand the workspace. Collision checking ensures the arm won't collide at the target height:

```python
# Direct height set (no animation)
robot.left_base.set_height(0.3)

# Plan base trajectory with collision checking
traj = robot.left_base.plan_to(0.3)
```

## Package Structure

```
src/geodude/
├── robot.py          # Geodude class — bimanual composition on mj_manipulator
├── config.py         # GeodudeArmSpec, VentionBaseConfig, DebugConfig
├── vention_base.py   # Linear actuator planning + collision checking
├── primitives.py     # pickup() / place() with explicit TSRs
└── __init__.py       # Public API + mj_manipulator re-exports
```

## Testing

```bash
uv run pytest tests/ -v
```

## Dependencies

**Workspace packages:**

- [mj_manipulator](https://github.com/siddhss5/mj_manipulator) — Generic arm control, planning, execution, grasping
- [geodude_assets](https://github.com/personalrobotics/geodude_assets) — MuJoCo models for Geodude (UR5e + Robotiq)
- [prl_assets](https://github.com/personalrobotics/prl_assets) — Object models with geometry metadata
- [tsr](https://github.com/personalrobotics/tsr) — Task Space Regions + grasp/place generation
- [pycbirrt](https://github.com/personalrobotics/pycbirrt) — CBiRRT motion planner
- [mj_environment](https://github.com/personalrobotics/mj_environment) — MuJoCo environment wrapper
- [asset_manager](https://github.com/personalrobotics/asset_manager) — Object metadata loader

**External:**

- [eaik](https://github.com/Verdant-Robotics/eaik) — Analytical IK for UR robots
- [mujoco](https://github.com/google-deepmind/mujoco) — Physics simulation
