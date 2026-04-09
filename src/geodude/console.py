# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""IPython console for Geodude with integrated LLM chat.

Delegates to mj_manipulator's generic console for physics event loop,
viser viewer, teleop panels, and IPython shell. Adds Geodude-specific
panels (chat, sensors, status HUD) and demo infrastructure via hooks.
"""

from __future__ import annotations

import inspect
import logging
import os
import textwrap
import time
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
    viser: bool = False,
    model_name: str = "claude-haiku-4-5-20251001",
    demo_module: ModuleType | None = None,
    objects: dict | None = None,
    fixtures: dict | None = None,
) -> None:
    """Launch the Geodude IPython console.

    Delegates to mj_manipulator.console.start_console with
    Geodude-specific panels and namespace entries.
    """
    from mj_manipulator.console import start_console as _start_console

    from geodude.primitives import go_home, pickup, place

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

            sc = demo_module.scene.get("spawn_count") if demo_module and hasattr(demo_module, "scene") else None
            chat_session = ChatSession(
                robot,
                mode=mode,
                model_name=model_name,
                original_objects=objects or {},
                original_fixtures=fixtures or {},
                spawn_count=sc,
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
    viser_viewer_ref = [None]  # mutable ref set by panel_setup

    def reset() -> None:
        """Reset the demo — robot to ready, objects re-scattered, chat history cleared."""
        nonlocal chat_session
        from geodude.demo_loader import _spawn_manipulable_objects

        robot.request_abort()
        time.sleep(0.1)
        robot.reset()
        robot.clear_abort()
        fixture_types = set(fixtures.keys()) if fixtures else set()
        spawn_count = None
        if demo_module and hasattr(demo_module, "scene"):
            spawn_count = demo_module.scene.get("spawn_count")
        _spawn_manipulable_objects(robot, objects or {}, fixture_types, spawn_count=spawn_count)
        if chat_session is not None:
            chat_session.messages.clear()
            chat_session.action_log.clear()
        if viser_viewer_ref[0] is not None:
            viser_viewer_ref[0]._scene_mgr.clear_selection()
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

        scene_dict = {}
        if objects:
            scene_dict["objects"] = objects
        if fixtures:
            scene_dict["fixtures"] = fixtures

        user_funcs = {}
        ip = get_ipython()  # noqa: F821
        for k, v in ip.user_ns.items():
            if (
                callable(v)
                and not k.startswith("_")
                and k
                not in (
                    "chat",
                    "commands",
                    "demos",
                    "save_demo",
                    "exit",
                    "quit",
                    "get_ipython",
                    "pickup",
                    "place",
                    "go_home",
                    "reset",
                    "token_usage",
                )
            ):
                try:
                    src = inspect.getsource(v)
                    user_funcs[k] = textwrap.dedent(src)
                except OSError:
                    print(f"  Warning: couldn't get source for '{k}', skipping")

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
        print(
            """
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
  robot.left.get_ee_pose()          — left end-effector 4x4 pose
  robot.right.get_ee_pose()         — right end-effector 4x4 pose
  robot.left.get_ft_wrench()        — left wrist F/T sensor [fx,fy,fz,tx,ty,tz]
  robot.left.close()                — close left gripper
  robot.left.open()                 — open left gripper
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
"""
        )

    def token_usage() -> None:
        """Print LLM token usage and estimated cost for this session."""
        s = _get_chat()
        if s is None:
            print("Chat not initialized.")
            return
        print(s.token_usage())

    # -- Extra namespace entries -----------------------------------------------
    extra_ns: dict = {
        "chat": chat,
        "commands": commands,
        "demos": demos,
        "save_demo": save_demo,
        "reset": reset,
        "token_usage": token_usage,
        "pickup": lambda target=None, **kw: pickup(robot, target, **kw),
        "place": lambda dest=None, **kw: place(robot, dest, **kw),
        "go_home": lambda **kw: go_home(robot, **kw),
    }

    if demo_module is not None:
        from geodude.demo_loader import get_demo_functions, inject_robot

        inject_robot(demo_module, robot)
        for name_fn, func in get_demo_functions(demo_module).items():
            extra_ns[name_fn] = func

    # -- Panel setup callback --------------------------------------------------
    def _setup_geodude_panels(gui, viewer, robot, event_loop, tabs):
        nonlocal chat_session
        viser_viewer_ref[0] = viewer

        # Sensor panels
        from mj_viser import SensorChannel, SensorPanel

        sensor_panels = []
        for side, arm in [("Left", robot._left_arm), ("Right", robot._right_arm)]:
            if arm.has_ft_sensor:
                force_adr = arm._ft_force_adr
                torque_adr = arm._ft_torque_adr
                sensor_panels.append(
                    SensorPanel(
                        title=f"{side} F/T",
                        use_folder=False,
                        channels=[
                            SensorChannel(force_adr + 0, "Fx", "#e74c3c"),
                            SensorChannel(force_adr + 1, "Fy", "#2ecc71"),
                            SensorChannel(force_adr + 2, "Fz", "#3498db"),
                            SensorChannel(torque_adr + 0, "Tx", "#e67e22"),
                            SensorChannel(torque_adr + 1, "Ty", "#9b59b6"),
                            SensorChannel(torque_adr + 2, "Tz", "#1abc9c"),
                        ],
                        window_seconds=5.0,
                        y_label="N / Nm",
                        aspect=1.2,
                    )
                )

        # Chat panel
        chat_panel = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            chat_session = _get_chat()
            if chat_session is not None:
                from geodude.panels.chat_panel import ChatPanel

                chat_panel = ChatPanel(chat_session)

        all_panels = []
        if chat_panel is not None:
            with tabs.add_tab("Chat"):
                chat_panel.setup(gui, viewer)
            all_panels.append(chat_panel)
        if sensor_panels:
            with tabs.add_tab("Sensors"):
                for sp in sensor_panels:
                    sp.setup(gui, viewer)
                    all_panels.append(sp)

        # Status HUD is created by the generic console (mj_manipulator.status_hud)
        viewer._panels.extend(all_panels)

    # -- Delegate to generic console -------------------------------------------
    _start_console(
        robot,
        physics=physics,
        viser=viser,
        headless=not viewer,
        robot_name="Geodude",
        extra_ns=extra_ns,
        panel_setup=_setup_geodude_panels if viser else None,
    )
