"""Geodude CLI entry point.

Usage::

    geodude --demo recycling              # headless (default)
    geodude --demo recycling --viewer     # native MuJoCo viewer (requires mjpython)
    geodude --demo recycling --viser      # browser viewer (http://localhost:8080)
    geodude --list-demos
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="geodude",
        description="Geodude interactive console",
    )
    parser.add_argument("--demo", type=str, default=None, help="Demo name or path")
    parser.add_argument("--physics", action="store_true", help="Physics simulation")
    parser.add_argument("--viewer", action="store_true", help="Launch native MuJoCo viewer (requires mjpython)")
    parser.add_argument("--viser", action="store_true", help="Launch browser viewer at http://localhost:8080")
    parser.add_argument("--objects", type=str, default=None, help='JSON, e.g. \'{"can": 4}\'')
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="LLM model")
    parser.add_argument("--list-demos", action="store_true", help="List demos and exit")
    args = parser.parse_args()

    if args.list_demos:
        from geodude.demo_loader import list_demos
        list_demos()
        sys.exit(0)

    from geodude.demo_loader import resolve_scene, setup_robot

    objects, fixtures, demo_module = resolve_scene(args.demo, args.objects)
    spawn_count = None
    if demo_module and hasattr(demo_module, "scene"):
        spawn_count = demo_module.scene.get("spawn_count")
    print(f"\nLoading Geodude with {objects}...", flush=True)
    robot = setup_robot(objects, fixtures, spawn_count=spawn_count)

    from geodude.console import start_console
    start_console(
        robot,
        physics=args.physics,
        viewer=args.viewer,
        viser=args.viser,
        model_name=args.model,
        demo_module=demo_module,
        objects=objects,
        fixtures=fixtures,
    )
