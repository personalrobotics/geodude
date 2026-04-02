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
    viser: bool = False,
    model_name: str = "claude-haiku-4-5-20251001",
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
            sc = demo_module.scene.get("spawn_count") if demo_module and hasattr(demo_module, "scene") else None
            chat_session = ChatSession(
                robot, mode=mode, model_name=model_name,
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
    def reset() -> None:
        """Reset the demo — robot to ready, objects re-scattered, chat history cleared."""
        nonlocal chat_session
        from geodude.demo_loader import _spawn_manipulable_objects
        robot.reset()
        fixture_types = set(fixtures.keys()) if fixtures else set()
        spawn_count = None
        if demo_module and hasattr(demo_module, "scene"):
            spawn_count = demo_module.scene.get("spawn_count")
        _spawn_manipulable_objects(robot, objects or {}, fixture_types, spawn_count=spawn_count)
        if chat_session is not None:
            chat_session.messages.clear()
            chat_session.action_log.clear()
        if viser_viewer is not None:
            viser_viewer._scene_mgr.clear_selection()
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
""")

    def token_usage() -> None:
        """Print LLM token usage and estimated cost for this session."""
        s = _get_chat()
        if s is None:
            print("Chat not initialized.")
            return
        print(s.token_usage())

    # -- Build namespace -----------------------------------------------------
    user_ns: dict = {
        "robot": robot,
        "np": np,
        "chat": chat,
        "commands": commands,
        "demos": demos,
        "save_demo": save_demo,
        "reset": reset,
        "token_usage": token_usage,
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

    viewer_str = " | viser" if viser else " | viewer" if viewer else ""
    banner = f"\n{'=' * 60}\n  Geodude [{mode}] | 2 arms | {n_objects} objects{demo_str}{viewer_str}\n"
    if viser:
        banner += f"  Browser: http://localhost:8080\n"
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

    # Launch viser viewer if requested (runs alongside sim context)
    viser_viewer = None
    if viser:
        from mj_viser import MujocoViewer
        viser_viewer = MujocoViewer(
            robot.model, robot.data,
            show_sim_controls=False,
            show_visibility=False,
        )

        # Build sensor panels (not added via add_panel — we'll set them up in tabs)
        from mj_viser import SensorChannel, SensorPanel
        sensor_panels = []
        for side, arm in [("Left", robot._left_arm), ("Right", robot._right_arm)]:
            if arm.has_ft_sensor:
                force_adr = arm._ft_force_adr
                torque_adr = arm._ft_torque_adr
                sensor_panels.append(SensorPanel(
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
                ))

        # Build chat panel
        chat_panel = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            chat_session = _get_chat()
            if chat_session is not None:
                from geodude.panels.chat_panel import ChatPanel
                chat_panel = ChatPanel(chat_session)

        # Set up tabbed layout — panels inside tabs to avoid vertical overflow.
        # We call setup() manually inside tab contexts, then register for
        # on_sync via _panels list (bypassing add_panel which would re-setup).
        gui = viser_viewer._server.gui

        # Stop button — above tabs so it's always visible
        stop_btn = gui.add_button("Stop", color="red")

        @stop_btn.on_click
        def _on_stop(event):
            robot.request_abort()
            from geodude.panels.status_hud import StatusHud
            hud = getattr(robot, "_status_hud", None)
            if hud is not None:
                hud.set_action("left", "⊘ STOP")
                hud.set_action("right", "⊘ STOP")

        tabs = gui.add_tab_group()

        all_panels = []
        if chat_panel is not None:
            with tabs.add_tab("Chat"):
                chat_panel.setup(gui, viser_viewer)
            all_panels.append(chat_panel)
        if sensor_panels:
            with tabs.add_tab("Sensors"):
                for sp in sensor_panels:
                    sp.setup(gui, viser_viewer)
                    all_panels.append(sp)

        # Status HUD overlay — store on robot so primitives can update it
        from geodude.panels.status_hud import StatusHud
        status_hud = StatusHud(robot, mode)
        robot._status_hud = status_hud
        all_panels.append(status_hud)

        # Build scene (no panels registered via add_panel — we set them up above)
        viser_viewer.launch_passive(open_browser=False)

        # Register panels for on_sync after launch
        viser_viewer._panels.extend(all_panels)

        # Initialize HUD after launch
        status_hud.setup(viser_viewer._server.gui, viser_viewer)

        print(f"  Viser viewer: http://localhost:8080")

    # Pass viser viewer to SimContext so executors can sync it during trajectories
    sim_viewer = viser_viewer if viser else None
    with robot.sim(physics=physics, headless=not viewer, viewer=sim_viewer) as ctx:
        user_ns["ctx"] = ctx

        # Teleop panel (needs ctx, so created after sim context)
        if viser and viser_viewer is not None:
            from geodude.panels.teleop_panel import create_teleop_panel
            gui = viser_viewer._server.gui
            teleop_panel = create_teleop_panel(robot, ctx, side="right")
            with tabs.add_tab("Teleop"):
                teleop_panel.setup(gui, viser_viewer)
            viser_viewer._panels.append(teleop_panel)

        from IPython.terminal.prompts import Prompts, Token

        class GeodudePrompts(Prompts):
            def in_prompt_tokens(self, cli=None):
                return [
                    (Token.Prompt, f"Geodude [{mode}] [{self.shell.execution_count}]: "),
                ]

            def out_prompt_tokens(self, cli=None):
                return [
                    (Token.OutPrompt, f"Out[{self.shell.execution_count}]: "),
                ]

        shell = InteractiveShellEmbed(
            header=banner,
            user_ns=user_ns,
            colors="neutral",
        )
        shell.prompts = GeodudePrompts(shell)

        @shell.register_magic_function
        def chat_magic(line):
            """%chat <message> — send a message to the LLM."""
            chat(line)

        shell.magics_manager.register_alias("chat", "chat_magic")
        shell()
