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

    # Register IPython magic for /chat
    from IPython.core.magic import register_line_magic

    @register_line_magic
    def chat(line):
        """/chat <message> — send a message to the LLM."""
        if not line.strip():
            print("Usage: /chat <message>")
            return
        session = _get_chat()
        if session is None:
            return
        response = session.send(line.strip())
        if response:
            print(f"\nGeodude [{mode}]: {response}\n")

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
        f"  Available: robot, ctx, np\n"
        f"  /chat <msg>  — talk to the robot via LLM\n"
        f"  robot.<tab>  — tab completion\n"
        f"  ?robot.pickup — help on any method\n"
    )

    # Namespace for IPython
    user_ns = {
        "robot": robot,
        "ctx": ctx,
        "np": np,
        "chat": chat,
    }

    import IPython
    IPython.embed(
        header=banner,
        user_ns=user_ns,
        colors="neutral",
    )

    # Cleanup
    sim_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
