# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Headless PD gain tuning harness for the geodude UR5e arms.

This is a one-shot tool for ``personalrobotics/geodude_assets#12`` — not a
demo that ships with the package's normal surface area. It lives in
``geodude/demos/`` so it's discoverable and reproducible alongside other
demos, but it's only meant to be run when someone wants to re-derive the
``gainprm``/``biasprm`` values in
``geodude_assets/src/geodude_assets/models/universal_robots_ur5e/ur5e.xml``.

Usage::

    uv run python -m geodude.demos.tune_ur5e_gains

Produces a CSV with per-run metrics (``/tmp/ur5e_gain_sweep.csv`` by
default) and a terminal summary picking the best gain set under the
tracking/force/settling constraints.

============================================================================
 Why this exists, and why the naive "sweep kp, keep kv proportional" plan
 wasn't right
============================================================================

The geodude_assets UR5e XML defines two actuator classes that hardcode a
``kv = 0.1 * kp`` relationship::

    <general gainprm="8000" biasprm="0 -8000 -800"   />  # size3
    <general gainprm="2000" biasprm="0 -2000 -200"   />  # size1

These gains were cranked 4x above menagerie defaults (2000 / 500) as
compensation for gravity sag in a pre-gravcomp era. #86 landed gravcomp
on the source XMLs, so the stiffness is no longer load-bearing — the PD
loop doesn't have to fight gravity via steady-state error anymore.

Before sweeping gains, this tool measures the **actual damping ratio**
per joint by reading the mass matrix diagonal at home pose::

    Joint          kp    kv    J_eff    ζ       kv_critical
    shoulder_pan   8000  800   4.36     2.14    374
    shoulder_lift  8000  800   4.07     2.22    361
    elbow          8000  800   0.83     4.91    163
    wrist_1        2000  200   0.13     6.15    33
    wrist_2        2000  200   0.13     6.11    33
    wrist_3        2000  200   0.10     7.02    29

Every joint is at 2-7x critical damping. The wrist joints are the worst,
sitting at ζ ≈ 6-7. Critical is ζ = 1. Over-damping manifests as
tracking lag in every physics-mode demo we've run since #86 landed.

The source of the over-damping is the fixed ``kv/kp = 0.1`` relationship.
For a second-order PD loop::

    J*q̈ + kv*q̇ + kp*(q - q_target) = 0

critical damping (ζ = 1) requires::

    kv_crit = 2 * sqrt(kp * J_eff)

J_eff varies by ~40x across the arm (shoulder_pan at ~4.4 kg·m² to wrist_3
at ~0.1 kg·m²), so a single ``kv/kp`` ratio cannot be right for every
joint. ``kv = 0.1 * kp`` happens to be at-or-above critical for every joint
in the current geometry, which is why nothing oscillates — but it's also
why tracking lags.

The correct retune is **not** "sweep kp, keep kv proportional." It is
"pick a damping target, sweep kp, and derive kv per-joint from J_eff at
every kp value." That's a 1D line through correct-damping space instead
of a 2D grid through wrong-damping space.

============================================================================
 Sweep design
============================================================================

**Fixed choices**:

- Gravcomp on (already in source XML as of #90)
- J_eff per joint = max of ``mj_fullM`` diagonal over a small set of
  representative poses (home, reach-forward, stretched). Using the max
  makes kv slightly-overdamped at the easier poses and at-or-above
  critical at the hardest poses — the fallback is "too damped" rather
  than "unstable."
- Fixed damping ratio across the sweep (varied as one of the sweep axes)

**Grid**:

- ζ ∈ {0.7, 1.0, 1.2}
    - 0.7: classic control "responsive with small overshoot"
    - 1.0: critical, no overshoot, fastest unambiguous convergence
    - 1.2: slightly over-damped, for sanity comparison with current behavior
- kp_size3 ∈ {500, 1000, 2000, 3000, 4000, 6000, 8000}
- kp_size1 ∈ {125, 250, 500, 750, 1000, 1500, 2000}

At each ``(ζ, kp_size3, kp_size1)`` point, kv per joint is computed as
``2 * ζ * sqrt(kp_joint * J_eff_joint)``.

**Trajectories** (all from home via ``arm.plan_to_configuration`` +
``arm.retime``):

1. **Slow reach**: shoulder/elbow deltas at 10% velocity limits — baseline
   sanity check, should be trivial at any reasonable gains
2. **Fast reach**: same target, 100% velocity limits — stress test
3. **Fast swing**: shoulder_pan ±1.2 rad — exercises the heaviest joint
   under max gravity torque at mid-swing
4. **Fast stretched**: elbow ~= 0, arm nearly straight — worst-case
   inertia / edge-of-workspace

**Per-run metrics**:

- ``max_pos_err_deg``: max abs position error per joint during motion
- ``rms_pos_err_deg``: RMS position error per joint during motion
- ``max_vel_err_dps``: max abs velocity error per joint
- ``max_force_frac``: max abs actuator force / forcerange per joint
- ``settling_time_s``: time from last commanded waypoint until
  ``|pos_err| < 2°`` AND ``|vel| < 5 deg/s``
- ``max_overshoot_pct``: max |pos_err| after last waypoint / trajectory range.
  Tracks whether under-damped configurations oscillate around the target.

**Winner selection**:

1. Drop any config where any trajectory fails ``max_pos_err < 2°``.
2. Drop any config where any trajectory uses more than 70% force headroom.
3. Drop any config where any trajectory doesn't settle within 0.5 s.
4. Drop any config with ``max_overshoot_pct > 5%`` on any trajectory.
5. Among survivors, pick lowest ``kp_size3``; break ties on ``kp_size1``.
6. Prefer ``ζ = 1.0`` for the final tiebreaker.

============================================================================
 Known risks
============================================================================

- **Over-damping hides instabilities**. Dropping kv to critical will
  expose behaviors that the current ζ ≈ 6 was masking. The overshoot
  metric catches symptomatic oscillation. If some configs oscillate but
  pass the max_pos_err check because they do it around the target, the
  overshoot filter handles it.
- **J_eff at home pose is an approximation**. If the winner jitters at
  poses far from the sampling set, we may need to add more sample poses
  or switch to adaptive kv (update from current qpos every N steps).
  Deferring this until the basic sweep reveals whether it's actually a
  problem.

============================================================================
 Not in scope
============================================================================

- Gripper actuator retune (Robotiq ``fingers_actuator`` uses
  ``gaintype=fixed`` with non-standard scaling — separate analysis)
- Vention base gains
- Franka (different repo, different hardware, different issue)
- Pose-adaptive kv schemes

"""

from __future__ import annotations

import csv
import dataclasses
import logging
import math
import time
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
from mj_manipulator.arm import Arm
from mj_manipulator.config import PhysicsConfig, PhysicsExecutionConfig
from mj_manipulator.trajectory import Trajectory

from geodude.robot import Geodude

logger = logging.getLogger("tune_ur5e_gains")

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GainSet:
    """A complete actuator gain specification for the UR5e arm.

    kp is passed as a 6-tuple (one per joint), NOT per-class, because the
    optimal kp may differ even between the two wrist joints once kv is
    derived from per-joint J_eff. kv is then computed at use time from
    (kp, J_eff, zeta) so that the controller stays at the desired damping.
    """

    zeta: float  # Target damping ratio
    kp_per_joint: tuple[float, ...]  # Length-6 tuple of kp per joint

    def kv_per_joint(self, j_eff: np.ndarray) -> np.ndarray:
        """kv derived per joint from J_eff and the target damping ratio."""
        return np.array([2 * self.zeta * math.sqrt(kp * j) for kp, j in zip(self.kp_per_joint, j_eff)])


@dataclasses.dataclass
class TrajectoryResult:
    """Per-trajectory tracking metrics under one gain set."""

    trajectory_name: str
    max_pos_err_deg: np.ndarray  # shape (6,), degrees
    rms_pos_err_deg: np.ndarray  # shape (6,), degrees
    max_vel_err_dps: np.ndarray  # shape (6,), degrees per second
    max_force_frac: np.ndarray  # shape (6,), [0, 1] abs force / forcerange
    settling_time_s: float
    max_overshoot_pct: float  # percent of trajectory range


@dataclasses.dataclass
class SweepResult:
    """Aggregated metrics for one gain set across all trajectories."""

    gain_set: GainSet
    per_traj: list[TrajectoryResult]

    def worst_max_pos_err_deg(self) -> float:
        return float(max(r.max_pos_err_deg.max() for r in self.per_traj))

    def worst_rms_pos_err_deg(self) -> float:
        return float(max(r.rms_pos_err_deg.max() for r in self.per_traj))

    def worst_max_force_frac(self) -> float:
        return float(max(r.max_force_frac.max() for r in self.per_traj))

    def worst_settling_time_s(self) -> float:
        return float(max(r.settling_time_s for r in self.per_traj))

    def worst_overshoot_pct(self) -> float:
        return float(max(r.max_overshoot_pct for r in self.per_traj))


# ---------------------------------------------------------------------------
# Damping math: compute J_eff per joint at representative poses
# ---------------------------------------------------------------------------


def compute_j_eff_conservative(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: Arm,
    sample_poses: list[np.ndarray],
) -> np.ndarray:
    """Compute effective inertia per arm joint, taking the max over poses.

    For each sample pose, set qpos, forward, read the mass matrix
    diagonal at the arm's dof indices. Return the element-wise max across
    poses.

    Using the max rather than the mean makes the kv derivation
    pessimistic: at "easier" poses the controller ends up slightly
    over-damped (safe), at the hardest sampled pose it's at critical.

    Args:
        model: MuJoCo model with left arm actuators.
        data: Scratch MjData, mutated then restored.
        arm: The arm whose joints we measure.
        sample_poses: List of arm joint configurations (6-vectors).

    Returns:
        Per-joint J_eff as shape (6,) numpy array.
    """
    # Save and restore qpos/qvel so this is non-destructive
    orig_qpos = data.qpos.copy()
    orig_qvel = data.qvel.copy()

    n = len(arm.joint_qvel_indices)
    j_eff_max = np.zeros(n)
    M = np.zeros((model.nv, model.nv))

    try:
        for pose in sample_poses:
            # Zero everything, then set arm joints to the sample pose
            data.qpos[:] = orig_qpos
            data.qvel[:] = 0.0
            for i, idx in enumerate(arm.joint_qpos_indices):
                data.qpos[idx] = pose[i]
            mujoco.mj_forward(model, data)
            mujoco.mj_fullM(model, M, data.qM)
            for i, dof in enumerate(arm.joint_qvel_indices):
                j_eff_max[i] = max(j_eff_max[i], M[dof, dof])
    finally:
        data.qpos[:] = orig_qpos
        data.qvel[:] = orig_qvel
        mujoco.mj_forward(model, data)

    return j_eff_max


# ---------------------------------------------------------------------------
# Actuator gain overrides (runtime, non-destructive)
# ---------------------------------------------------------------------------


def apply_gains(
    model: mujoco.MjModel,
    actuator_ids: list[int],
    kp_per_joint: np.ndarray,
    kv_per_joint: np.ndarray,
) -> None:
    """Write kp/kv into the model's actuator parameters.

    For MuJoCo's ``general`` actuator with ``biastype="affine"``::

        force = gainprm[0] * ctrl + biasprm[1] * qpos + biasprm[2] * qvel

    To match ``force = kp*(ctrl - qpos) - kv*qvel``:

        gainprm[0] = kp
        biasprm[1] = -kp
        biasprm[2] = -kv

    This is the same form the UR5e class XML already uses, so we're
    just overwriting the numeric values. The actuator's ``gaintype``,
    ``biastype``, and other fields stay untouched.
    """
    for aid, kp, kv in zip(actuator_ids, kp_per_joint, kv_per_joint):
        model.actuator_gainprm[aid, 0] = kp
        model.actuator_biasprm[aid, 1] = -kp
        model.actuator_biasprm[aid, 2] = -kv


def read_baseline_gains(
    model: mujoco.MjModel,
    actuator_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot current (kp, kv) per joint so we can restore after the sweep."""
    kp = np.array([model.actuator_gainprm[aid, 0] for aid in actuator_ids])
    kv = np.array([-model.actuator_biasprm[aid, 2] for aid in actuator_ids])
    return kp, kv


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


# Representative poses for the **left** arm, matching the geodude ``ready``
# keyframe. Note the sign on wrist_2 (+π/2, not −π/2) — this is what
# geodude's own named_poses dict returns.
HOME_POSE = np.array([-1.5708, -1.5708, 1.5708, -1.5708, 1.5708, 0.0])

# Target poses. Small joint-space deltas from home that stay well inside
# the arm's reachable workspace and don't induce self-collision or
# inter-arm collision (confirmed empirically by loading the bimanual
# model and checking each target against the baseline gains).
#
# We deliberately skip plan_to_configuration (CBiRRT + collision
# checking) because:
#  1. The tuning harness doesn't need path planning — it needs
#     REFERENCE TRAJECTORIES that the PD loop has to track
#  2. plan_to_configuration introduces planner-dependent randomness
#     and slow iteration
#  3. It can fail on collision boundaries between the two arms, which
#     has nothing to do with the controller we're trying to tune
#
# Instead we hand the start and goal joint configurations directly to
# ``Trajectory.from_path``, which uses TOPP-RA to produce a smooth
# time-parameterization at fixed control_dt. That's a straight-line
# joint-space motion, which is exactly what a tuning rig wants.

_REACH_DELTA = np.array([0.3, -0.2, 0.1, -0.1, 0.2, 0.0])
_SWING_DELTA = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
_STRETCH_DELTA = np.array([0.0, -0.4, -0.2, 0.0, 0.0, 0.0])  # elbow toward straight
_WRIST_DELTA = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5])  # wrist-only

REACH_GOAL = HOME_POSE + _REACH_DELTA
SWING_GOAL = HOME_POSE + _SWING_DELTA
STRETCH_GOAL = HOME_POSE + _STRETCH_DELTA
WRIST_GOAL = HOME_POSE + _WRIST_DELTA

SAMPLE_POSES_FOR_J_EFF = [HOME_POSE, REACH_GOAL, SWING_GOAL, STRETCH_GOAL]


def make_trajectories(arm: Arm) -> list[tuple[str, Trajectory]]:
    """Generate the four representative tracking trajectories.

    Each trajectory is a straight-line joint-space motion from
    HOME_POSE to a target, retimed via TOPP-RA at the arm's kinematic
    limits (or a scaled version for the slow variant).

    Deliberately skips ``arm.plan_to_configuration``: tuning reference
    trajectories just need to be smooth, not collision-free, because
    the PD loop's job is to track whatever was commanded. The poses
    below are hand-picked to avoid self-collision and inter-arm
    collision in the bimanual setup.

    Returns:
        List of (name, trajectory) tuples.
    """
    limits = arm.config.kinematic_limits
    vel_full = limits.velocity.copy()
    acc_full = limits.acceleration.copy()

    trajs: list[tuple[str, Trajectory]] = []

    def add(name: str, goal: np.ndarray, vel_scale: float = 1.0) -> None:
        path = [HOME_POSE.copy(), goal.copy()]
        traj = Trajectory.from_path(
            path=path,
            vel_limits=vel_full * vel_scale,
            acc_limits=acc_full * vel_scale,
            control_dt=0.008,
            entity=arm.config.name,
            joint_names=arm.config.joint_names,
        )
        trajs.append((name, traj))

    # 1. Slow reach: 30% of vel limits, baseline sanity
    add("slow_reach", REACH_GOAL, vel_scale=0.3)
    # 2. Fast reach: 100% of vel limits, stress test
    add("fast_reach", REACH_GOAL, vel_scale=1.0)
    # 3. Shoulder_pan swing: heaviest joint, max gravity moment at midswing
    add("fast_swing", SWING_GOAL, vel_scale=1.0)
    # 4. Wrist-only: exercises the low-inertia joints in isolation
    add("fast_wrist", WRIST_GOAL, vel_scale=1.0)

    return trajs


# ---------------------------------------------------------------------------
# Measurement: run a trajectory headless and collect metrics
# ---------------------------------------------------------------------------


def run_trajectory_and_measure(
    arm: Arm,
    trajectory: Trajectory,
    actuator_ids: list[int],
    sim_step: Callable[[], None],
    control_dt: float,
    apply_target: Callable[[np.ndarray], None],
) -> TrajectoryResult:
    """Execute a trajectory step-by-step, recording tracking metrics.

    Deliberately avoids the full trajectory runner so this harness is
    self-contained and independent of mj_manipulator's execution
    plumbing. Instead it just advances through waypoints one per control
    cycle, calls ``apply_target`` to set the PD target, calls ``sim_step``
    to advance physics by one control period, and records the state.

    After the last waypoint, it keeps simulating (with the final target
    held) until settling conditions are met or a timeout fires.

    Args:
        arm: Arm under test (provides joint_qpos_indices, joint_qvel_indices).
        trajectory: Time-parameterized trajectory at ``control_dt`` rate.
        actuator_ids: Actuator ids for the arm's joints (length 6).
        sim_step: Callable that advances physics by one control period.
        control_dt: Control timestep in seconds.
        apply_target: Callable ``apply_target(q_cmd)`` that writes a
            length-6 target into ``data.ctrl`` at the arm actuators.

    Returns:
        TrajectoryResult with per-joint tracking metrics.
    """
    model = arm.env.model
    data = arm.env.data

    n_joints = len(arm.joint_qpos_indices)
    n_waypoints = trajectory.num_waypoints

    # Preallocate logs for the tracked portion
    actual_qpos = np.zeros((n_waypoints, n_joints))
    actual_qvel = np.zeros((n_waypoints, n_joints))
    actual_force = np.zeros((n_waypoints, n_joints))

    # Execute tracked portion
    for i in range(n_waypoints):
        apply_target(trajectory.positions[i])
        sim_step()
        for j, qidx in enumerate(arm.joint_qpos_indices):
            actual_qpos[i, j] = data.qpos[qidx]
        for j, vidx in enumerate(arm.joint_qvel_indices):
            actual_qvel[i, j] = data.qvel[vidx]
        for j, aid in enumerate(actuator_ids):
            actual_force[i, j] = data.actuator_force[aid]

    # --- Tracking metrics over the commanded portion ---
    pos_err = trajectory.positions - actual_qpos  # (N, 6)
    vel_err = trajectory.velocities - actual_qvel  # (N, 6)

    max_pos_err = np.rad2deg(np.max(np.abs(pos_err), axis=0))
    rms_pos_err = np.rad2deg(np.sqrt(np.mean(pos_err**2, axis=0)))
    max_vel_err = np.rad2deg(np.max(np.abs(vel_err), axis=0))

    # Force headroom (per-joint; forcerange is [low, high], take max abs)
    forcerange = np.array([model.actuator_forcerange[aid] for aid in actuator_ids])
    force_limit = np.abs(forcerange).max(axis=1)
    max_force_frac = np.max(np.abs(actual_force), axis=0) / force_limit

    # --- Settling phase: hold final target, wait for convergence ---
    #
    # Overshoot definition: how far past the target did the arm go in
    # the direction of motion? If start[j] = 1.0 and target[j] = 2.0
    # (positive motion), any q[j] > 2.0 is overshoot. Same logic for
    # negative motion (target < start: any q < target is overshoot).
    # Joints that barely moved (|target - start| < 0.01 rad) are
    # excluded from the overshoot calculation entirely because their
    # percentage would be numerically unstable — their max_pos_err
    # already catches meaningful error on them.
    final_target = trajectory.positions[-1]
    start_pose = trajectory.positions[0]
    motion = final_target - start_pose  # per-joint signed motion
    motion_sign = np.sign(motion)

    # For each joint, track the max excess past target in the motion direction,
    # across BOTH the tracked phase and the settling phase.
    def excess_past_target(q_log: np.ndarray) -> np.ndarray:
        """For each joint, max amount that q exceeded final_target in the
        direction of motion. Returns shape (n_joints,) in radians.
        """
        # positions past the target in the direction of motion
        signed_err = (q_log - final_target) * motion_sign  # (T, J)
        # Excess = max(signed_err) if positive, else 0
        max_excess = np.max(signed_err, axis=0)
        return np.maximum(max_excess, 0.0)

    settle_max_steps = int(1.0 / control_dt)  # 1 second budget
    pos_tol = np.deg2rad(2.0)
    vel_tol = np.deg2rad(5.0)

    settling_qpos_log: list[np.ndarray] = []
    settling_steps = 0
    for _ in range(settle_max_steps):
        apply_target(final_target)
        sim_step()
        q_now = np.array([data.qpos[qidx] for qidx in arm.joint_qpos_indices])
        v_now = np.array([data.qvel[vidx] for vidx in arm.joint_qvel_indices])
        settling_qpos_log.append(q_now.copy())
        settling_steps += 1
        if np.all(np.abs(final_target - q_now) < pos_tol) and np.all(np.abs(v_now) < vel_tol):
            break

    settling_time_s = settling_steps * control_dt

    # Compute overshoot across full q trajectory (tracked + settling)
    if settling_qpos_log:
        full_qpos = np.vstack([actual_qpos, np.array(settling_qpos_log)])
    else:
        full_qpos = actual_qpos
    excess = excess_past_target(full_qpos)  # (J,) rad of overshoot

    # Normalize excess by motion magnitude per joint, only for joints
    # that actually moved. Joints with |motion| < 1° are excluded.
    min_motion = np.deg2rad(1.0)
    moving_joints = np.abs(motion) > min_motion
    if np.any(moving_joints):
        overshoot_pct = excess[moving_joints] / np.abs(motion[moving_joints])
        max_overshoot_pct = float(np.max(overshoot_pct) * 100.0)
    else:
        max_overshoot_pct = 0.0

    return TrajectoryResult(
        trajectory_name="",  # filled in by caller
        max_pos_err_deg=max_pos_err,
        rms_pos_err_deg=rms_pos_err,
        max_vel_err_dps=max_vel_err,
        max_force_frac=max_force_frac,
        settling_time_s=settling_time_s,
        max_overshoot_pct=max_overshoot_pct,
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def iter_gain_grid() -> list[GainSet]:
    """Yield all (zeta, kp_size3, kp_size1) combinations to evaluate."""
    zetas = [0.7, 1.0, 1.2]
    kp_size3_values = [500, 1000, 2000, 3000, 4000, 6000, 8000]
    kp_size1_values = [125, 250, 500, 750, 1000, 1500, 2000]

    out: list[GainSet] = []
    for zeta in zetas:
        for kp3 in kp_size3_values:
            for kp1 in kp_size1_values:
                # Joints 0..2 (shoulder_pan, shoulder_lift, elbow) at kp_size3
                # Joints 3..5 (wrist_1, wrist_2, wrist_3) at kp_size1
                kp_per_joint = (kp3, kp3, kp3, kp1, kp1, kp1)
                out.append(GainSet(zeta=zeta, kp_per_joint=kp_per_joint))
    return out


def print_baseline_diagnostic(model: mujoco.MjModel, actuator_ids: list[int], j_eff: np.ndarray) -> None:
    """Print the baseline over-damping analysis as a diagnostic header."""
    kp_now, kv_now = read_baseline_gains(model, actuator_ids)

    print("=" * 78)
    print("BASELINE DIAGNOSTIC: current gains vs critical damping")
    print("=" * 78)
    print(f"{'joint':<15s} {'kp':>7s} {'kv':>7s} {'J_eff':>8s} {'ζ':>6s} {'kv_crit':>9s}")
    for i, aid in enumerate(actuator_ids):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or f"act{aid}"
        short = name.split("/")[-1]
        kv_crit = 2 * math.sqrt(kp_now[i] * j_eff[i])
        zeta = kv_now[i] / kv_crit if kv_crit > 0 else float("inf")
        print(f"{short:<15s} {kp_now[i]:7.0f} {kv_now[i]:7.1f} {j_eff[i]:8.4f} {zeta:6.2f} {kv_crit:9.1f}")
    print()
    print(
        "Over-damping means the PD loop crawls back to its target rather than snapping. "
        "Target: ζ = 1.0 (critical) or 0.7 (slightly under, responsive)."
    )
    print()


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AcceptanceCriteria:
    """Pass/fail thresholds for the winning gain set.

    Thresholds were anchored to the baseline kp=8000/2000 configuration's
    measured performance on the fast_swing trajectory (the hardest of
    the four), which tops out at:

        max_pos_err ≈ 1.15°, RMS ≈ 0.67°, force ≈ 12%, settle ≈ 8 ms

    A winner must do at least as well as the baseline on every metric.
    Max pos err and RMS are slightly above baseline worst-case so the
    sweep can find "equal or better" solutions. Force and settle give
    headroom for configurations with lower kp that use less force and
    take slightly longer to settle.
    """

    max_pos_err_deg: float = 2.0  # 1.75x baseline worst
    max_rms_pos_err_deg: float = 1.0  # 1.5x baseline worst
    max_force_frac: float = 0.7  # plenty of headroom vs 12% baseline
    max_settling_time_s: float = 0.1  # baseline is 8ms; generous headroom
    max_overshoot_pct: float = 10.0  # critical damping gives <1%; 10% catches oscillation


def passes(result: SweepResult, crit: AcceptanceCriteria) -> bool:
    return (
        result.worst_max_pos_err_deg() <= crit.max_pos_err_deg
        and result.worst_rms_pos_err_deg() <= crit.max_rms_pos_err_deg
        and result.worst_max_force_frac() <= crit.max_force_frac
        and result.worst_settling_time_s() <= crit.max_settling_time_s
        and result.worst_overshoot_pct() <= crit.max_overshoot_pct
    )


def pick_winner(
    results: list[SweepResult],
    crit: AcceptanceCriteria,
) -> SweepResult | None:
    """Pick the lowest-gain config that meets all acceptance criteria.

    Sort order: (kp_size3, kp_size1, |ζ - 1.0|). Lowest kp first (soft
    contact response), ties broken by kp on the wrist joints, further
    ties broken by preferring ζ closest to critical damping.
    """
    surviving = [r for r in results if passes(r, crit)]
    if not surviving:
        return None

    def sort_key(r: SweepResult) -> tuple[float, float, float]:
        kp3 = r.gain_set.kp_per_joint[0]
        kp1 = r.gain_set.kp_per_joint[3]
        zeta_dist = abs(r.gain_set.zeta - 1.0)
        return (kp3, kp1, zeta_dist)

    surviving.sort(key=sort_key)
    return surviving[0]


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv(results: list[SweepResult], path: Path) -> None:
    """Dump per-trajectory metrics to CSV for later analysis."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "zeta",
                "kp_size3",
                "kp_size1",
                "trajectory",
                "worst_max_pos_err_deg",
                "worst_rms_pos_err_deg",
                "worst_max_vel_err_dps",
                "worst_max_force_frac",
                "settling_time_s",
                "max_overshoot_pct",
            ]
        )
        for r in results:
            for traj in r.per_traj:
                writer.writerow(
                    [
                        f"{r.gain_set.zeta:.2f}",
                        f"{r.gain_set.kp_per_joint[0]:.0f}",
                        f"{r.gain_set.kp_per_joint[3]:.0f}",
                        traj.trajectory_name,
                        f"{float(traj.max_pos_err_deg.max()):.4f}",
                        f"{float(traj.rms_pos_err_deg.max()):.4f}",
                        f"{float(traj.max_vel_err_dps.max()):.4f}",
                        f"{float(traj.max_force_frac.max()):.4f}",
                        f"{traj.settling_time_s:.4f}",
                        f"{traj.max_overshoot_pct:.4f}",
                    ]
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    print("Loading geodude (headless)...")
    robot = Geodude()
    model, data = robot.model, robot.data
    left_arm = robot.arms["left"]

    # Source of truth for the "ready" pose: geodude's own named_poses
    # dict. left has wrist_2=+π/2, right has shoulder_pan=+π/2 (mirrored).
    # Using the real keyframe rather than a hand-coded constant avoids
    # sign errors.
    ready = robot._named_poses["ready"]
    left_home = np.array(ready["left"], dtype=float)
    right_home = np.array(ready["right"], dtype=float)
    # Sanity: our HOME_POSE constant must match left_home for J_eff and
    # trajectory generation to be consistent.
    assert np.allclose(HOME_POSE, left_home), (
        f"HOME_POSE constant drifted from geodude named_poses['ready']['left']. "
        f"constant={HOME_POSE}, ready_left={left_home}"
    )

    # Find left arm actuator ids, in joint order
    actuator_ids: list[int] = []
    for joint_name in (
        "left_ur5e/shoulder_pan_joint",
        "left_ur5e/shoulder_lift_joint",
        "left_ur5e/elbow_joint",
        "left_ur5e/wrist_1_joint",
        "left_ur5e/wrist_2_joint",
        "left_ur5e/wrist_3_joint",
    ):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        # Find the actuator whose trnid points at this joint
        aid = -1
        for a in range(model.nu):
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a, 0] == jid:
                aid = a
                break
        assert aid >= 0, f"no actuator for {joint_name}"
        actuator_ids.append(aid)
    print(f"actuator_ids: {actuator_ids}")

    # ------- J_eff measurement -------
    print("Measuring effective joint inertia across representative poses...")
    j_eff = compute_j_eff_conservative(model, data, left_arm, SAMPLE_POSES_FOR_J_EFF)
    print(f"J_eff (max over poses): {j_eff}")

    # ------- Baseline diagnostic -------
    print_baseline_diagnostic(model, actuator_ids, j_eff)

    # Snapshot baseline for restore after sweep
    baseline_kp, baseline_kv = read_baseline_gains(model, actuator_ids)

    # ------- Build trajectories -------
    print("Planning representative trajectories (home -> 4 goals)...")
    # Seed arm at home so plan_to_configuration has a sensible start
    for i, idx in enumerate(left_arm.joint_qpos_indices):
        data.qpos[idx] = HOME_POSE[i]
    data.qvel[left_arm.joint_qvel_indices] = 0.0
    mujoco.mj_forward(model, data)

    trajectories = make_trajectories(left_arm)
    for name, traj in trajectories:
        print(f"  {name:<20s} {traj.num_waypoints:4d} waypoints, {traj.duration:.3f} s")
    if not trajectories:
        print("No trajectories planned; aborting.")
        return

    # ------- Set up simulation executor (headless, no mj_manipulator ctx) -------
    control_dt = 0.008  # 125 Hz — matches mj_manipulator default
    physics_config = PhysicsConfig(execution=PhysicsExecutionConfig(control_dt=control_dt))  # noqa: F841
    steps_per_control = max(1, int(control_dt / model.opt.timestep))

    # Build the write-ctrl helper once per trajectory run
    right_arm = robot.arms["right"]
    right_actuator_ids: list[int] = []
    for joint_name in (
        "right_ur5e/shoulder_pan_joint",
        "right_ur5e/shoulder_lift_joint",
        "right_ur5e/elbow_joint",
        "right_ur5e/wrist_1_joint",
        "right_ur5e/wrist_2_joint",
        "right_ur5e/wrist_3_joint",
    ):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        for a in range(model.nu):
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a, 0] == jid:
                right_actuator_ids.append(a)
                break

    # The right arm holds at its (mirrored) home pose the whole time so
    # it doesn't sag and so inter-arm contacts stay zero-effort.
    def reset_state_to_home() -> None:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        for i, idx in enumerate(left_arm.joint_qpos_indices):
            data.qpos[idx] = left_home[i]
        for i, idx in enumerate(right_arm.joint_qpos_indices):
            data.qpos[idx] = right_home[i]
        mujoco.mj_forward(model, data)
        # Pre-seat ctrl targets at the home pose so the first mj_step
        # doesn't see a huge error.
        for i, aid in enumerate(actuator_ids):
            data.ctrl[aid] = left_home[i]
        for i, aid in enumerate(right_actuator_ids):
            data.ctrl[aid] = right_home[i]
        # Settle for 200 ms
        for _ in range(int(0.2 / control_dt)):
            for _ in range(steps_per_control):
                mujoco.mj_step(model, data)

    def sim_step_once() -> None:
        """Advance physics by one control period."""
        # Right arm ctrl stays at HOME_POSE (set in reset_state_to_home)
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)

    def apply_target_left(q_cmd: np.ndarray) -> None:
        for i, aid in enumerate(actuator_ids):
            data.ctrl[aid] = q_cmd[i]

    # ------- TRUE baseline measurement (BEFORE overriding anything) -------
    #
    # This runs the 4 trajectories against the gains as loaded from
    # geodude.xml, without any apply_gains() override. The number is
    # the "before" anchor for the writeup: any winner must improve on
    # these or at least match them. The sweep below will call
    # apply_gains() every iteration, so we capture the true baseline
    # here once, up front.
    print("\nMeasuring TRUE baseline tracking (gains as loaded from XML)...")
    baseline_per_traj: list[TrajectoryResult] = []
    for name, traj in trajectories:
        reset_state_to_home()
        try:
            result = run_trajectory_and_measure(
                left_arm,
                traj,
                actuator_ids,
                sim_step_once,
                control_dt,
                apply_target_left,
            )
        except Exception as exc:
            logger.warning("baseline run failed on %s: %s", name, exc)
            continue
        result.trajectory_name = name
        baseline_per_traj.append(result)

    print("TRUE baseline (kp=8000/2000, kv=800/200 as-loaded from geodude_assets ur5e.xml):")
    print(f"  {'trajectory':<15s} {'max_pos':>9s} {'rms_pos':>9s} {'force%':>7s} {'settle':>8s} {'oshoot%':>8s}")
    for t in baseline_per_traj:
        print(
            f"  {t.trajectory_name:<15s} "
            f"{float(t.max_pos_err_deg.max()):>8.3f}° "
            f"{float(t.rms_pos_err_deg.max()):>8.3f}° "
            f"{float(t.max_force_frac.max()) * 100:>6.1f}% "
            f"{t.settling_time_s * 1000:>6.0f}ms "
            f"{t.max_overshoot_pct:>7.2f}%"
        )
    print()

    # ------- Sweep -------
    gain_grid = iter_gain_grid()
    results: list[SweepResult] = []
    t0 = time.time()
    print(f"Sweeping {len(gain_grid)} gain configurations × {len(trajectories)} trajectories...")

    for gi, gain_set in enumerate(gain_grid):
        kp = np.array(gain_set.kp_per_joint)
        kv = gain_set.kv_per_joint(j_eff)
        apply_gains(model, actuator_ids, kp, kv)

        per_traj: list[TrajectoryResult] = []
        skipped = False
        for name, traj in trajectories:
            reset_state_to_home()
            try:
                result = run_trajectory_and_measure(
                    left_arm,
                    traj,
                    actuator_ids,
                    sim_step_once,
                    control_dt,
                    apply_target_left,
                )
            except Exception as exc:  # defensive: catch divergence/instability
                logger.warning("run failed on gain_set=%s traj=%s: %s", gain_set, name, exc)
                skipped = True
                break
            result.trajectory_name = name
            per_traj.append(result)

        if not skipped:
            results.append(SweepResult(gain_set=gain_set, per_traj=per_traj))

        if (gi + 1) % 20 == 0 or gi == len(gain_grid) - 1:
            elapsed = time.time() - t0
            print(f"  {gi + 1}/{len(gain_grid)} configs done ({elapsed:.1f}s elapsed)")

    # Restore baseline gains so the model isn't left in a weird state
    apply_gains(model, actuator_ids, baseline_kp, baseline_kv)

    # ------- CSV output -------
    csv_path = Path("/tmp/ur5e_gain_sweep.csv")
    write_csv(results, csv_path)
    print(f"\nPer-run metrics written to {csv_path}")

    # ------- Winner selection -------
    criteria = AcceptanceCriteria()
    winner = pick_winner(results, criteria)

    print("\n" + "=" * 78)
    print("WINNER SELECTION")
    print("=" * 78)
    print(
        f"Criteria: max_pos_err < {criteria.max_pos_err_deg}°, "
        f"RMS < {criteria.max_rms_pos_err_deg}°, "
        f"force < {criteria.max_force_frac * 100:.0f}%, "
        f"settle < {criteria.max_settling_time_s * 1000:.0f}ms, "
        f"overshoot < {criteria.max_overshoot_pct}%"
    )

    if winner is None:
        print()
        print("NO GAIN SET PASSED ALL CRITERIA.")
        print("Top 5 by worst_max_pos_err_deg (diagnostic):")
        results.sort(key=lambda r: r.worst_max_pos_err_deg())
        for r in results[:5]:
            gs = r.gain_set
            print(
                f"  ζ={gs.zeta:.2f} kp3={gs.kp_per_joint[0]:5.0f} "
                f"kp1={gs.kp_per_joint[3]:5.0f}  "
                f"pos_err={r.worst_max_pos_err_deg():5.2f}°  "
                f"rms={r.worst_rms_pos_err_deg():5.2f}°  "
                f"force={r.worst_max_force_frac() * 100:5.1f}%  "
                f"settle={r.worst_settling_time_s() * 1000:5.0f}ms  "
                f"overshoot={r.worst_overshoot_pct():5.2f}%"
            )
        return

    gs = winner.gain_set
    kp_w = np.array(gs.kp_per_joint)
    kv_w = gs.kv_per_joint(j_eff)
    print()
    print(f"Winning config: ζ={gs.zeta}, kp_size3={gs.kp_per_joint[0]:.0f}, kp_size1={gs.kp_per_joint[3]:.0f}")
    print(f"{'joint':<15s} {'kp':>7s} {'kv':>7s}")
    joint_names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    )
    for name, kp_i, kv_i in zip(joint_names, kp_w, kv_w):
        print(f"{name:<15s} {kp_i:7.0f} {kv_i:7.1f}")

    print()
    print("Winner worst-case across all trajectories:")
    print(f"  max_pos_err       = {winner.worst_max_pos_err_deg():.3f} °")
    print(f"  worst RMS pos err = {winner.worst_rms_pos_err_deg():.3f} °")
    print(f"  max force         = {winner.worst_max_force_frac() * 100:.1f}% of limit")
    print(f"  settling time     = {winner.worst_settling_time_s() * 1000:.0f} ms")
    print(f"  max overshoot     = {winner.worst_overshoot_pct():.2f}%")

    # Direct before/after comparison per trajectory — the core writeup table
    print()
    print("Before/after comparison (baseline → winner), per trajectory:")
    print(f"  {'trajectory':<15s} {'max_pos':>20s} {'rms':>20s} {'force%':>20s}")
    baseline_by_name = {t.trajectory_name: t for t in baseline_per_traj}
    for winner_t in winner.per_traj:
        base_t = baseline_by_name.get(winner_t.trajectory_name)
        if base_t is None:
            continue
        b_max = float(base_t.max_pos_err_deg.max())
        w_max = float(winner_t.max_pos_err_deg.max())
        b_rms = float(base_t.rms_pos_err_deg.max())
        w_rms = float(winner_t.rms_pos_err_deg.max())
        b_frc = float(base_t.max_force_frac.max()) * 100
        w_frc = float(winner_t.max_force_frac.max()) * 100
        print(
            f"  {winner_t.trajectory_name:<15s} "
            f"{b_max:7.2f}° → {w_max:6.2f}°  "
            f"{b_rms:7.2f}° → {w_rms:6.2f}°  "
            f"{b_frc:7.1f}% → {w_frc:6.1f}%"
        )


if __name__ == "__main__":
    main()
