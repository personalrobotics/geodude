#!/usr/bin/env python3
"""Interactive console for Geodude with integrated LLM chat.

IPython REPL with tab completion, introspection, and history.
Use /chat to talk to the robot via an LLM.

Usage:
    uv run python examples/console.py --preset recycling
    uv run python examples/console.py --physics --preset recycling
    uv run mjpython examples/console.py --physics --viewer --preset recycling

Environment:
    ANTHROPIC_API_KEY: Required for /chat. Console works without it.
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Geodude interactive console")
    parser.add_argument(
        "--physics", action="store_true",
        help="Use physics simulation (default: kinematic)",
    )
    parser.add_argument(
        "--hardware", action="store_true",
        help="Connect to real robot hardware",
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        help="Scene preset name (e.g. 'recycling')",
    )
    parser.add_argument(
        "--objects", type=str, default=None,
        help='JSON object counts, e.g. \'{"can": 4, "recycle_bin": 2}\'',
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="LLM model name for /chat",
    )
    parser.add_argument(
        "--viewer", action="store_true",
        help="Launch MuJoCo viewer (requires mjpython)",
    )
    args = parser.parse_args()

    if args.hardware:
        print("Hardware mode not yet implemented. See geodude#91.")
        sys.exit(1)

    mode = "physics" if args.physics else "kinematic"

    # Determine scene
    from geodude.chat import SCENE_PRESETS, choose_scene

    if args.objects:
        objects = json.loads(args.objects)
        fixtures = {}
    elif args.preset:
        if args.preset not in SCENE_PRESETS:
            print(f"Unknown preset: {args.preset}")
            print(f"Available: {', '.join(SCENE_PRESETS.keys())}")
            sys.exit(1)
        preset = SCENE_PRESETS[args.preset]
        objects = preset["objects"]
        fixtures = preset["fixtures"]
    else:
        objects, fixtures = choose_scene()

    # Create robot
    import numpy as np

    from geodude.robot import Geodude

    print(f"\nLoading Geodude with {objects}...")
    robot = Geodude(objects=objects)
    robot.setup_scene(fixtures=fixtures if fixtures else None)

    from geodude.chat import _spawn_manipulable_objects
    fixture_types = set(fixtures.keys()) if fixtures else set()
    _spawn_manipulable_objects(robot, objects, fixture_types)

    # Start sim
    sim_ctx = robot.sim(physics=args.physics, headless=not args.viewer)
    ctx = sim_ctx.__enter__()

    # Set up chat session (lazy — only if API key is available)
    chat_session = None

    def _get_chat():
        nonlocal chat_session
        if chat_session is not None:
            return chat_session
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Set ANTHROPIC_API_KEY to use /chat")
            return None
        from geodude.chat import ChatSession
        chat_session = ChatSession(
            robot, mode=mode, model_name=args.model,
            original_objects=objects, original_fixtures=fixtures,
        )
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

    def quickref():
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

LLM chat:
  chat('clear the table')           — natural language control
  %chat clear the table             — same, magic syntax

IPython:
  robot.<tab>                       — tab completion
  ?robot.pickup                     — docstring
  ??robot.pickup                    — source code
  whos                              — list all variables
""")

    # Banner
    n_objects = len(robot.find_objects())
    banner = (
        f"\n{'=' * 60}\n"
        f"  Geodude [{mode}] | 2 arms | {n_objects} objects\n"
    )
    if os.environ.get("ANTHROPIC_API_KEY"):
        banner += f"  LLM: {args.model}\n"
    if mode == "hardware":
        banner += "  \u26a0 REAL ROBOT \u2014 commands will move hardware\n"
    banner += (
        f"{'=' * 60}\n"
        f"\n"
        f"  quickref()   — quick reference of all commands\n"
        f"  chat('msg')  — talk to the robot via LLM\n"
        f"  robot.<tab>  — tab completion\n"
    )

    # Namespace for IPython
    user_ns = {
        "robot": robot,
        "ctx": ctx,
        "np": np,
        "chat": chat,
        "quickref": quickref,
    }

    import IPython
    from IPython.terminal.embed import InteractiveShellEmbed

    shell = InteractiveShellEmbed(
        header=banner,
        user_ns=user_ns,
        colors="neutral",
    )

    # Register /chat magic now that the shell exists
    from IPython.core.magic import register_line_magic

    @shell.register_magic_function
    def chat_magic(line):
        """/chat <message> — send a message to the LLM."""
        chat(line)

    # Make /chat work (alias)
    shell.magics_manager.register_alias("chat", "chat_magic")

    shell()

    # Cleanup
    sim_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
