# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for demo discovery, loading, and scene setup."""

import pytest

from geodude.demo_loader import (
    discover_demos,
    get_demo_functions,
    inject_robot,
    load_demo,
    resolve_scene,
    setup_robot,
)


class TestDiscoverDemos:
    def test_finds_recycling(self):
        demos = discover_demos()
        assert "recycling" in demos

    def test_returns_paths(self):
        demos = discover_demos()
        assert demos["recycling"].name == "recycling.py"


class TestLoadDemo:
    def test_load_by_name(self):
        mod = load_demo("recycling")
        assert hasattr(mod, "scene")
        assert "objects" in mod.scene

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="not found"):
            load_demo("nonexistent_demo")

    def test_scene_has_objects(self):
        mod = load_demo("recycling")
        assert "can" in mod.scene["objects"]


class TestGetDemoFunctions:
    def test_recycling_has_sort_all(self):
        mod = load_demo("recycling")
        funcs = get_demo_functions(mod)
        assert "sort_all" in funcs
        assert callable(funcs["sort_all"])


class TestInjectRobot:
    def test_inject(self):
        mod = load_demo("recycling")
        inject_robot(mod, "fake_robot")
        assert mod.robot == "fake_robot"


class TestResolveScene:
    def test_resolve_demo(self):
        objects, fixtures, mod = resolve_scene("recycling")
        assert "can" in objects
        assert mod is not None

    def test_resolve_json(self):
        objects, fixtures, mod = resolve_scene(objects_json='{"can": 2}')
        assert objects == {"can": 2}
        assert mod is None

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_scene("nonexistent")


class TestSetupRobot:
    def test_creates_robot_with_objects(self):
        robot = setup_robot(
            {"can": 2, "recycle_bin": 1},
            {"recycle_bin": [[0.5, 0, 0]]},
        )
        objects = robot.find_objects()
        assert len(objects) >= 3  # 2 cans + 1 bin
