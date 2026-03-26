"""IPython console for Geodude with integrated LLM chat.

Provides tab completion, introspection, demo functions, and optional
natural language control via LLM.
"""

from __future__ import annotations

import inspect
import logging
import os
import textwrap
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geodude.robot import Geodude

logger = logging.getLogger(__name__)


def start_console(
    robot: Geodude,
    *,
    physics: bool = False,
    viewer: bool = False,
    model_name: str = "claude-sonnet-4-20250514",
    demo_module: ModuleType | None = None,
    objects: dict | None = None,
    fixtures: dict | None = None,
) -> None:
    """Launch the IPython console with robot, chat, and demo functions."""
    import numpy as np

    mode = "physics" if physics else "kinematic"

    # -- Chat (lazy) ---------------------------------------------------------
    chat_session = None

    def _get_chat():
        nonlocal chat_session
        if chat_session is not None:
            return chat_session
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Set ANTHROPIC_API_KEY to use chat()")
            return None
        try:
            from geodude.chat import ChatSession
            chat_session = ChatSession(
                robot, mode=mode, model_name=model_name,
                original_objects=objects or {},
                original_fixtures=fixtures or {},
            )
        except ImportError:
            print("Install chat dependencies: uv sync --extra chat")
            return None
        return chat_session

    def chat(message: str) -> None:
        """Send a message to the LLM. Usage: chat('clear the table')"""
        if not message.strip():
            print("Usage: chat('clear the table')")
            return
        session = _get_chat()
        if session is None:
            return
        response = session.send(message.strip())
        if response:
            print(f"\nGeodude [{mode}]: {response}\n")

    # -- Demo helpers --------------------------------------------------------
    def reset() -> None:
        """Reset the demo — robot to ready, objects re-scattered."""
        from geodude.demo_loader import _spawn_manipulable_objects
        robot.reset()
        fixture_types = set(fixtures.keys()) if fixtures else set()
        _spawn_manipulable_objects(robot, objects or {}, fixture_types)
        robot.forward()
        print("Scene reset.")

    def demos() -> None:
        """List available demos."""
        from geodude.demo_loader import list_demos
        list_demos()

    def save_demo(name: str, description: str = "") -> None:
        """Save current scene config + user-defined functions as a demo file."""
        from geodude.demo_loader import DEMOS_DIR

        if not description:
            description = input("Description (one line): ").strip() or name

        # Build scene dict from current config
        scene_dict = {}
        if objects:
            scene_dict["objects"] = objects
        if fixtures:
            scene_dict["fixtures"] = fixtures

        # Find user-defined functions (not in initial namespace)
        user_funcs = {}
        ip = get_ipython()  # noqa: F821 — available inside IPython
        for k, v in ip.user_ns.items():
            if (
                callable(v)
                and not k.startswith("_")
                and k not in _initial_ns
                and k not in ("chat", "commands", "demos", "save_demo", "exit", "quit", "get_ipython")
            ):
                try:
                    src = inspect.getsource(v)
                    user_funcs[k] = textwrap.dedent(src)
                except OSError:
                    print(f"  Warning: couldn't get source for '{k}', skipping")

        # Write demo file
        path = DEMOS_DIR / f"{name}.py"
        with open(path, "w") as f:
            f.write(f'"""{description}"""\n\n')
            f.write(f"scene = {scene_dict!r}\n")
            for func_name, src in user_funcs.items():
                f.write(f"\n\n{src}")
            f.write("\n")

        print(f"Saved to {path}")

    def commands() -> None:
        """Print a quick reference of available commands."""
        print("""
Geodude Quick Reference
=======================

Scene:
  robot.find_objects()              — list all objects on table
  robot.find_objects("can")         — find objects by type
  robot.get_object_pose("can_0")    — 4x4 pose matrix of an object
  robot.holding()                   — (side, name) or None

Actions:
  robot.pickup()                    — pick up nearest reachable object
  robot.pickup("can")               — pick up any can
  robot.pickup("can_0")             — pick up specific object
  robot.pickup("can", arm="left")   — use a specific arm
  robot.place("recycle_bin")        — place in any bin
  robot.place()                     — place in any destination
  robot.go_home()                   — return all arms to ready

Arms:
  robot.left_arm.get_ee_pose()      — left end-effector 4x4 pose
  robot.right_arm.get_ee_pose()     — right end-effector 4x4 pose
  robot.left_arm.get_ft_wrench()    — left wrist F/T sensor [fx,fy,fz,tx,ty,tz]
  robot.right_arm.get_ft_wrench()   — right wrist F/T sensor

Scoped shortcuts:
  robot.left.pickup("can")          — left arm picks up a can
  robot.right.place("recycle_bin")  — right arm places in bin

Demos:
  reset()                           — restart the demo (re-scatter objects)
  demos()                           — list available demos
  save_demo('name')                 — save current scene as a demo

LLM chat:
  chat('clear the table')           — natural language control
  %chat clear the table             — same, magic syntax

IPython:
  robot.<tab>                       — tab completion
  ?robot.pickup                     — docstring
  ??robot.pickup                    — source code
  whos                              — list all variables
""")

    # -- Build namespace -----------------------------------------------------
    user_ns: dict = {
        "robot": robot,
        "np": np,
        "chat": chat,
        "commands": commands,
        "demos": demos,
        "save_demo": save_demo,
        "reset": reset,
    }

    if demo_module is not None:
        from geodude.demo_loader import get_demo_functions, inject_robot
        inject_robot(demo_module, robot)
        for name_fn, func in get_demo_functions(demo_module).items():
            user_ns[name_fn] = func

    # Snapshot initial namespace (for save_demo to identify user functions)
    _initial_ns = set(user_ns.keys())

    # -- Banner --------------------------------------------------------------
    n_objects = len(robot.find_objects())
    demo_name = getattr(demo_module, "__name__", "").rsplit(".", 1)[-1] if demo_module else None
    demo_str = f" | demo: {demo_name}" if demo_name else ""

    banner = f"\n{'=' * 60}\n  Geodude [{mode}] | 2 arms | {n_objects} objects{demo_str}\n"
    if os.environ.get("ANTHROPIC_API_KEY"):
        banner += f"  LLM: {model_name}\n"
    banner += (
        f"{'=' * 60}\n\n"
        f"  commands()   — quick reference\n"
        f"  chat('msg')  — LLM control\n"
        f"  robot.<tab>  — tab completion\n"
    )

    if demo_module and demo_module.__doc__:
        desc = demo_module.__doc__.strip()
        banner += f"\n  {desc}\n"

    # -- Launch IPython inside sim context -----------------------------------
    from IPython.terminal.embed import InteractiveShellEmbed

    with robot.sim(physics=physics, headless=not viewer) as ctx:
        user_ns["ctx"] = ctx

        shell = InteractiveShellEmbed(
            header=banner,
            user_ns=user_ns,
            colors="neutral",
        )

        @shell.register_magic_function
        def chat_magic(line):
            """%chat <message> — send a message to the LLM."""
            chat(line)

        shell.magics_manager.register_alias("chat", "chat_magic")
        shell()
