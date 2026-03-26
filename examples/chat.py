#!/usr/bin/env python3
"""Interactive chat interface for Geodude.

Talk to the robot in natural language. An LLM translates your requests
into robot actions and executes them.

Usage:
    # Kinematic mode (default)
    uv run python examples/chat.py

    # Physics simulation
    uv run python examples/chat.py --physics

    # Choose a scene preset
    uv run python examples/chat.py --physics --preset recycling

    # Custom objects
    uv run python examples/chat.py --physics --objects '{"can": 4, "recycle_bin": 2}'

Environment:
    ANTHROPIC_API_KEY: Required. Your Anthropic API key.
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Chat with Geodude")
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
        help="Scene preset name (e.g. 'recycling', 'stacking')",
    )
    parser.add_argument(
        "--objects", type=str, default=None,
        help='JSON object counts, e.g. \'{"can": 4, "recycle_bin": 2}\'',
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="LLM model name (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--viewer", action="store_true",
        help="Launch MuJoCo viewer (requires mjpython)",
    )
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Export your key: export ANTHROPIC_API_KEY=sk-...")
        sys.exit(1)

    if args.hardware:
        print("Hardware mode not yet implemented. See geodude#91.")
        sys.exit(1)

    # Determine mode
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
    from geodude.robot import Geodude

    print(f"\nLoading Geodude with {objects}...")
    robot = Geodude(objects=objects)
    robot.setup_scene(fixtures=fixtures if fixtures else None)

    # Spawn non-fixture objects on worktop
    from geodude.chat import _spawn_manipulable_objects
    fixture_types = set(fixtures.keys()) if fixtures else set()
    _spawn_manipulable_objects(robot, objects, fixture_types)

    # Start sim and chat
    with robot.sim(physics=args.physics, headless=not args.viewer) as ctx:
        from geodude.chat import chat_loop
        chat_loop(
            robot, mode=mode, model_name=args.model,
            original_objects=objects, original_fixtures=fixtures,
        )


if __name__ == "__main__":
    main()
