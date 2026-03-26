"""Demo discovery, loading, and scene setup.

Demos are Python files with a ``scene`` dict and optional helper functions.
They live in ``geodude/demos/`` or any directory.

Example demo file::

    \"\"\"My demo — stack cans.\"\"\"
    scene = {
        "objects": {"can": 6},
        "fixtures": {},
    }

    def stack_all():
        while robot.pickup():
            robot.place()
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from geodude.robot import Geodude

logger = logging.getLogger(__name__)

DEMOS_DIR = Path(__file__).parent / "demos"


# -- Demo discovery and loading ----------------------------------------------


def discover_demos() -> dict[str, Path]:
    """Find all demos in the demos/ directory. Returns {name: path}."""
    demos: dict[str, Path] = {}
    if not DEMOS_DIR.is_dir():
        return demos
    for p in sorted(DEMOS_DIR.glob("*.py")):
        if p.name.startswith("_"):
            continue
        demos[p.stem] = p
    return demos


def _get_demo_description(path: Path) -> str:
    """Extract the first line of a demo file's docstring without importing."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('"""') or line.startswith("'''"):
                # Single-line docstring: """description"""
                quote = line[:3]
                content = line[3:]
                if content.endswith(quote):
                    return content[:-3].strip()
                return content.strip()
            if line and not line.startswith("#"):
                break
    return path.stem


def list_demos() -> None:
    """Print available demos."""
    found = discover_demos()
    if not found:
        print("No demos found.")
        return
    print("\nAvailable demos:\n")
    for name, path in found.items():
        desc = _get_demo_description(path)
        print(f"  {name:20s} — {desc}")
    print()


def load_demo(name_or_path: str) -> ModuleType:
    """Load a demo by name (from demos/) or file path."""
    path = Path(name_or_path)
    if path.is_file():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # Try built-in demos
    demos = discover_demos()
    if name_or_path in demos:
        return load_demo(str(demos[name_or_path]))

    raise ValueError(
        f"Demo '{name_or_path}' not found. "
        f"Available: {', '.join(demos.keys()) or 'none'}"
    )


def inject_robot(demo_module: ModuleType, robot: Geodude) -> None:
    """Inject robot into the demo module's namespace."""
    demo_module.robot = robot


def get_demo_functions(demo_module: ModuleType) -> dict[str, Callable]:
    """Return public functions defined in a demo module."""
    import inspect

    return {
        name: obj
        for name, obj in inspect.getmembers(demo_module, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == demo_module.__name__
    }


# -- Scene resolution -------------------------------------------------------


def resolve_scene(
    demo: str | None = None,
    objects_json: str | None = None,
) -> tuple[dict[str, int], dict[str, list[list[float]]], ModuleType | None]:
    """Resolve scene config. Returns (objects, fixtures, demo_module or None)."""
    import json as _json

    if objects_json:
        return _json.loads(objects_json), {}, None

    if demo:
        mod = load_demo(demo)
        scene = mod.scene
        return scene["objects"], scene.get("fixtures", {}), mod

    return _choose_scene()


def _choose_scene() -> tuple[dict[str, int], dict[str, list[list[float]]], ModuleType | None]:
    """Interactive scene picker listing discovered demos."""
    demos = discover_demos()
    demo_list = list(demos.items())

    print("\nWhat scene would you like?\n")
    for i, (name, path) in enumerate(demo_list, 1):
        desc = _get_demo_description(path)
        print(f"  {i}. {desc}")
    print(f"  {len(demo_list) + 1}. Custom\n")

    choice = input("> ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(demo_list):
            name, _ = demo_list[idx]
            mod = load_demo(name)
            scene = mod.scene
            return scene["objects"], scene.get("fixtures", {}), mod
    except ValueError:
        pass

    # Custom scene
    from prl_assets import OBJECTS_DIR

    available = sorted(
        d.name for d in OBJECTS_DIR.iterdir()
        if d.is_dir() and (d / "meta.yaml").exists()
    )
    print(f"\nAvailable objects: {', '.join(available)}")
    print('How many of each? (e.g. "4 cans, 2 recycle_bins")')
    spec = input("> ").strip()

    objects: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        if len(tokens) == 2:
            count_str, obj_type = tokens
            if obj_type.endswith("s") and obj_type[:-1] in available:
                obj_type = obj_type[:-1]
            try:
                objects[obj_type] = int(count_str)
            except ValueError:
                print(f"  Skipping '{part}' — expected 'N type'")

    return objects, {}, None


# -- Robot setup -------------------------------------------------------------


def setup_robot(
    objects: dict[str, int],
    fixtures: dict[str, list[list[float]]],
) -> Geodude:
    """Create and set up a Geodude robot with the given scene config."""
    from geodude.robot import Geodude

    robot = Geodude(objects=objects)
    robot.setup_scene(fixtures=fixtures if fixtures else None)
    fixture_types = set(fixtures.keys()) if fixtures else set()
    _spawn_manipulable_objects(robot, objects, fixture_types)
    return robot


def _spawn_manipulable_objects(
    robot: Geodude,
    objects: dict[str, int],
    fixture_types: set[str],
) -> None:
    """Scatter non-fixture objects on the worktop (simulates perception)."""
    import mujoco
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR
    from tsr.placement import TablePlacer

    specs = [(t, n) for t, n in objects.items() if t not in fixture_types]
    if not specs:
        return

    assets = AssetManager(str(OBJECTS_DIR))
    wt_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    wt_size = robot.model.site_size[wt_id]
    worktop_pos = robot.data.site_xpos[wt_id].copy()

    placer = TablePlacer(wt_size[0] - 0.05, wt_size[1] - 0.05)
    table_surface = np.eye(4)
    table_surface[:3, 3] = worktop_pos

    for obj_type, count in specs:
        gp = assets.get(obj_type)["geometric_properties"]
        if gp["type"] == "cylinder":
            templates = placer.place_cylinder(gp["radius"], gp["height"])
        elif gp["type"] == "box":
            templates = placer.place_box(gp["size"][0], gp["size"][1], gp["size"][2])
        else:
            continue

        for _ in range(count):
            tsr = templates[0].instantiate(table_surface)
            pos = tsr.sample()[:3, 3]
            name = robot.env.registry.activate(obj_type, pos=list(pos))
            mujoco.mj_forward(robot.model, robot.data)

            body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jnt_id = robot.model.body_jntadr[body_id]
            qpos_adr = robot.model.jnt_qposadr[jnt_id]
            for _ in range(50):
                if not _has_object_collision(robot.model, robot.data, name):
                    break
                robot.data.qpos[qpos_adr : qpos_adr + 3] = tsr.sample()[:3, 3]
                mujoco.mj_forward(robot.model, robot.data)


def _has_object_collision(model, data, body_name: str) -> bool:
    """Check if a body is in contact with any other object (not floor)."""
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return False
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if (b1 == body_id or b2 == body_id) and c.dist < 0:
            other = b2 if b1 == body_id else b1
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other)
            if other_name and other_name != "world":
                return True
    return False
