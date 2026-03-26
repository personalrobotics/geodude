"""Geodude CLI entry point.

Usage::

    uv run geodude                          # interactive scene picker
    uv run geodude --demo recycling         # load a demo
    uv run geodude --demo recycling --physics --viewer
    uv run geodude --list-demos
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="geodude",
        description="Geodude interactive console",
    )
    parser.add_argument("--demo", type=str, default=None, help="Demo name or path to demo file")
    parser.add_argument("--physics", action="store_true", help="Use physics simulation")
    parser.add_argument("--viewer", action="store_true", help="Launch MuJoCo viewer (requires mjpython)")
    parser.add_argument("--objects", type=str, default=None, help='JSON object spec, e.g. \'{"can": 4}\'')
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="LLM model name")
    parser.add_argument("--list-demos", action="store_true", help="List available demos and exit")
    args = parser.parse_args()

    if args.list_demos:
        from geodude.demo_loader import discover_demos, load_demo
        found = discover_demos()
        if not found:
            print("No demos found.")
            sys.exit(0)
        print("\nAvailable demos:\n")
        for name in found:
            mod = load_demo(name)
            desc = (mod.__doc__ or name).strip().split("\n")[0]
            print(f"  {name:20s} — {desc}")
        print()
        sys.exit(0)

    from geodude.demo_loader import resolve_scene, setup_robot

    objects, fixtures, demo_module = resolve_scene(args.demo, args.objects)
    print(f"\nLoading Geodude with {objects}...")
    robot = setup_robot(objects, fixtures)

    from geodude.console import start_console
    start_console(
        robot,
        physics=args.physics,
        viewer=args.viewer,
        model_name=args.model,
        demo_module=demo_module,
        objects=objects,
        fixtures=fixtures,
    )
