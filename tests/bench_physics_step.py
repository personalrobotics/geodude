#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Headless benchmark: per-step timing with and without GraspVerifier.

Measures the wall-clock time of a single PhysicsController._step_physics
call across 1000 iterations, first with the verifier attached (current
state) and then with the verifier detached. Reports mean, p50, p99,
and max per-step times in microseconds.

Also benchmarks a full event-loop tick cycle to see the total overhead
from the verifier in context.

Run:
    cd /Users/siddh/code/robot-code/geodude
    uv run python tests/bench_physics_step.py
"""

import time

import numpy as np

from geodude.robot import Geodude


def bench_step(robot, ctx, label, n=1000):
    """Time n physics steps, return per-step times in microseconds."""
    controller = ctx._controller
    if controller is None:
        print(f"  {label}: no physics controller (kinematic mode), skipping")
        return None

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        controller.step()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds

    times = np.array(times)
    print(
        f"  {label}: mean={times.mean():.1f}µs  "
        f"p50={np.median(times):.1f}µs  "
        f"p99={np.percentile(times, 99):.1f}µs  "
        f"max={times.max():.1f}µs  "
        f"(n={n})"
    )
    return times


def bench_tick_overhead(robot, ctx, label, n=1000):
    """Time what the verifier tick itself costs, isolated."""
    times = []
    for side in ("left", "right"):
        arm = robot.arms[side]
        v = arm.gripper.grasp_verifier
        if v is None:
            continue
        for _ in range(n):
            t0 = time.perf_counter()
            v.tick()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)

    if not times:
        print(f"  {label}: no verifiers configured")
        return None

    times = np.array(times)
    print(
        f"  {label}: mean={times.mean():.3f}µs  "
        f"p50={np.median(times):.3f}µs  "
        f"p99={np.percentile(times, 99):.3f}µs  "
        f"max={times.max():.3f}µs  "
        f"(n={len(times)}, both arms)"
    )
    return times


def main():
    robot = Geodude()
    print(f"Geodude loaded: {robot.model.nbody} bodies, {robot.model.njnt} joints\n")

    with robot.sim(headless=True, physics=True) as ctx:
        # Warm up — let physics settle
        controller = ctx._controller
        for _ in range(100):
            controller.step()

        # ---- Benchmark 1: verifier tick overhead in isolation ----
        print("=== Verifier tick overhead (isolated) ===")
        bench_tick_overhead(robot, ctx, "IDLE state (no grasp)")

        # Mark a fake grasp so the verifier is in HOLDING
        for side in ("left", "right"):
            v = robot.arms[side].gripper.grasp_verifier
            if v is not None:
                v.mark_grasped("fake_object")
        bench_tick_overhead(robot, ctx, "HOLDING state (settling window)")

        # Burn through the settling window
        for _ in range(10):
            for side in ("left", "right"):
                v = robot.arms[side].gripper.grasp_verifier
                if v is not None:
                    v.tick()
        bench_tick_overhead(robot, ctx, "HOLDING state (post-settling, real verification)")

        # Release
        for side in ("left", "right"):
            v = robot.arms[side].gripper.grasp_verifier
            if v is not None:
                v.mark_released()

        # ---- Benchmark 2: full physics step with verifier ----
        print("\n=== Full physics step (controller.step) ===")
        bench_step(robot, ctx, "WITH verifier (IDLE)")

        # ---- Benchmark 3: detach verifiers and re-measure ----
        saved_verifiers = {}
        for side in ("left", "right"):
            gripper = robot.arms[side].gripper
            saved_verifiers[side] = gripper.grasp_verifier
            gripper.grasp_verifier = None

        bench_step(robot, ctx, "WITHOUT verifier")

        # Restore
        for side, v in saved_verifiers.items():
            robot.arms[side].gripper.grasp_verifier = v

        # ---- Benchmark 4: with verifier in HOLDING + real verification ----
        for side in ("left", "right"):
            v = robot.arms[side].gripper.grasp_verifier
            if v is not None:
                v.mark_grasped("fake_object")
        # Burn through settling
        for _ in range(10):
            controller.step()

        bench_step(robot, ctx, "WITH verifier (HOLDING, verifying)")

        for side in ("left", "right"):
            v = robot.arms[side].gripper.grasp_verifier
            if v is not None:
                v.mark_released()

        # ---- Summary ----
        print("\n=== Comparison ===")
        print("If the WITH/WITHOUT difference is < 10µs per step, the")
        print("verifier tick is not the jitter source and we need to")
        print("look elsewhere (event loop, viewer sync, teleop IK, etc.)")


if __name__ == "__main__":
    main()
