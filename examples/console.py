#!/usr/bin/env python3
"""Interactive console for Geodude with integrated LLM chat.

IPython REPL with tab completion, introspection, and history.
Use chat('message') or %chat message to talk to the robot via an LLM.

Usage:
    uv run python examples/console.py --preset recycling
    uv run python examples/console.py --physics --preset recycling
    uv run mjpython examples/console.py --physics --viewer --preset recycling

Environment:
    ANTHROPIC_API_KEY: Required for chat(). Console works without it.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Geodude interactive console")
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--objects", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    mode = "physics" if args.physics else "kinematic"

    from geodude.chat import resolve_scene, setup_robot

    objects, fixtures = resolve_scene(args.preset, args.objects)
    print(f"\nLoading Geodude with {objects}...")
    robot = setup_robot(objects, fixtures)

    import numpy as np

    # -- Chat integration (lazy) ---------------------------------------------
    chat_session = None

    def _get_chat():
        nonlocal chat_session
        if chat_session is not None:
            return chat_session
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Set ANTHROPIC_API_KEY to use chat()")
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

    def commands():
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

    # -- Banner --------------------------------------------------------------
    n_objects = len(robot.find_objects())
    banner = (
        f"\n{'=' * 60}\n"
        f"  Geodude [{mode}] | 2 arms | {n_objects} objects\n"
    )
    if os.environ.get("ANTHROPIC_API_KEY"):
        banner += f"  LLM: {args.model}\n"
    banner += (
        f"{'=' * 60}\n"
        f"\n"
        f"  commands()   — quick reference of all commands\n"
        f"  chat('msg')  — talk to the robot via LLM\n"
        f"  robot.<tab>  — tab completion\n"
    )

    # -- Launch IPython inside sim context -----------------------------------
    user_ns = {
        "robot": robot,
        "np": np,
        "chat": chat,
        "commands": commands,
    }

    import IPython
    from IPython.terminal.embed import InteractiveShellEmbed

    with robot.sim(physics=args.physics, headless=not args.viewer) as ctx:
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


if __name__ == "__main__":
    main()
