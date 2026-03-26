#!/usr/bin/env python3
"""Chat-only interface for Geodude (no IPython). See console.py for the full REPL.

Usage:
    uv run python examples/chat.py --preset recycling
    uv run python examples/chat.py --physics --preset recycling

Environment:
    ANTHROPIC_API_KEY: Required. Your Anthropic API key.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Chat with Geodude")
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--objects", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    from geodude.chat import ChatSession, resolve_scene, setup_robot

    objects, fixtures = resolve_scene(args.preset, args.objects)
    print(f"\nLoading Geodude with {objects}...")
    robot = setup_robot(objects, fixtures)
    mode = "physics" if args.physics else "kinematic"

    with robot.sim(physics=args.physics, headless=not args.viewer) as ctx:
        session = ChatSession(
            robot, mode=mode, model_name=args.model,
            original_objects=objects, original_fixtures=fixtures,
        )

        print(f"\nGeodude [{mode}] | Type naturally, 'quit' to exit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break
            response = session.send(user_input)
            if response:
                print(f"\nGeodude [{mode}]: {response}\n")

    print("Bye!")


if __name__ == "__main__":
    main()
