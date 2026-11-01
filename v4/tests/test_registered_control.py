"""Recycling's control runs on registered object poses, never on simulator truth
(personalrobotics/geodude#248).

One seeded ``sort_all`` round of ``OBJECTS`` objects, run three ways in contexts with
registered object poses (``recycling.OBJECT_POSES``):
- ``A``: as the demo runs, with evaluation: the registration stub reads each object's
  privileged pose channel, and landings are scored against privileged poses.
- ``B``, twice: without evaluation, through a runtime that declares no privileged
  channel, so the context refuses any read of one. Registration reads MuJoCo directly,
  outside the context, standing in for a perception source.

The claims:
- ``B`` runs to the end, and its control record is exactly ``A``'s: per attempt, the
  object, the arm, the race's selection, every grasp and drop stage, each failure and
  recovery, the release time, and every transition estimated. Evaluation steers
  nothing.
- ``A`` and ``B`` register the same objects, as often (``registrations``).
- In ``B``'s trace no observation reads a privileged channel, and every snapshot holds
  each object as registered, carried by its attachment, unknown, or removed, never
  measured; each registered pose is exactly its latest registration's. So no object pose
  enters control except through ``recycling.register``.

``B``'s final placement is read from MuJoCo after its round, for the report. The
evidence is ``registered_control.json``; ``B`` writes byte-identical records twice.

A second test, ``moved_and_retried.json``, moves one can ``MOVE_M`` along the worktop
in MuJoCo after it was registered, as a bump would, with every privileged channel
withheld. The grasp, planned on the registered pose, closes on nothing (``missed``,
portable evidence). The table is registered again, at the can's new pose, and the one
retry, raced from a fresh snapshot, picks it up and drops it in a bin. Both attempts are
recorded, each with the registration its race planned on; the run is repeated
identically.

Reproduce with ``uv run pytest tests/test_registered_control.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from ssrobot import (
    ObjectStatus,
    Observation,
    ObservationRequest,
    Pose,
    RobotContext,
    RuntimeInfo,
    SceneSnapshot,
    SceneState,
    Timestamp,
    TraceRecord,
    ValidationError,
    load_installed_package,
)

from geodude import recycling
from geodude.recycling import Arm
from geodude.scene import stage_scene

SEED = 0
OBJECTS = 3


class Unprivileged:
    """Geodude in MuJoCo, declaring no privileged channel: its scene channels are
    withheld from ``RuntimeInfo``, and a request for one is refused here as well."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._withheld: frozenset[str] = frozenset()

    def open(self, description: Any) -> RuntimeInfo:
        info: RuntimeInfo = self._inner.open(description)
        scene = {c.name for c in info.scene_channels}
        self._withheld = frozenset(scene) | frozenset(info.privileged)
        return dataclasses.replace(
            info,
            channels=tuple(c for c in info.channels if c not in self._withheld),
            scene_channels=(),
            privileged=(),
            rest=tuple(r for r in info.rest if r.channel not in self._withheld),
        )

    @property
    def withheld(self) -> frozenset[str]:
        """The channels it withholds: every privileged channel the runtime declares."""
        return self._withheld

    def observe(self, request: ObservationRequest) -> Observation:
        refused = sorted(set(request.channels) & self._withheld)
        if refused:
            raise ValidationError("privileged_channel", f"refused: {', '.join(refused)}")
        observation: Observation = self._inner.observe(request)
        return observation

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _perceived(inner: Any) -> recycling.Read:
    """A registration source outside the context: the object's pose, read from MuJoCo."""

    def read(ctx: RobotContext, name: str) -> tuple[Pose, Timestamp]:
        observation = inner.observe(ObservationRequest(channels=(f"{name}.pose",)))
        x, y, z, w, i, j, k = observation.readings[0].value
        return Pose(position=(x, y, z), quat_wxyz=(w, i, j, k)), observation.stamp

    return read


def _counted(read: recycling.Read, counts: Counter[str]) -> recycling.Read:
    def counting(ctx: RobotContext, name: str) -> tuple[Pose, Timestamp]:
        counts[name] += 1
        return read(ctx, name)

    return counting


def _control(result: dict[str, Any]) -> list[dict[str, Any]]:
    """What control decided, per attempt: nothing evaluation contributes."""
    return [
        {
            "object": a["object"],
            "attempt": a["attempt"],
            "arm": a["arm"],
            "selected": (a.get("race") or {}).get("selected"),
            "grasp": a.get("grasp"),
            "drop": a.get("drop"),
            "failure": a.get("failure"),
            "recovery": a.get("recovery"),
            "grasp_ns": (a.get("times_ns") or {}).get("grasp"),
            "released_ns": (a.get("times_ns") or {}).get("released"),
            "transitions": [
                [t["skill"], t["stage"], t["transition"], t["outcome"], t["reason"], t["assumed"]]
                for t in a.get("transitions", [])
            ],
        }
        for a in result["attempts"]
    ]


def _provenance(records: list[TraceRecord], privileged: set[str]) -> dict[str, Any]:
    """Whether the trace shows object poses entering only through registration."""
    scenes: dict[int, SceneState] = {}
    statuses: Counter[str] = Counter()
    mismatched = observed_privileged = snapshots = 0
    for r in records:
        if isinstance(r.payload, SceneState):
            scenes[r.payload.revision] = r.payload
        elif isinstance(r.payload, Observation):
            observed_privileged += sum(x.channel in privileged for x in r.payload.readings)
        elif isinstance(r.payload, SceneSnapshot):
            snapshots += 1
            scene = scenes[r.payload.revision]
            for o in r.payload.objects:
                statuses[o.status.value] += 1
                if o.status is ObjectStatus.REGISTERED:
                    registration = scene.registration(o.name)
                    mismatched += registration is None or registration.pose != o.pose
                elif o.status is ObjectStatus.ATTACHED:
                    mismatched += scene.attachment(o.name) is None
    sources = Counter(
        r.source for r in records if isinstance(r.payload, SceneState) and r.source != "context"
    )
    return {
        "snapshots": snapshots,
        "object_states": dict(sorted(statuses.items())),
        "registered_poses_not_their_registration": mismatched,
        "privileged_readings": observed_privileged,
        "scene_changes_by_source": dict(sorted(sources.items())),
    }


def _round(work: Path, unprivileged: bool) -> dict[str, Any]:
    package = load_installed_package("geodude_assets")
    scene = stage_scene(
        work / "scene",
        package,
        objects=recycling.mix(SEED, OBJECTS),
        fixtures=recycling.BINS,
        seed=SEED,
    )
    inner = recycling.runtime(package, scene)
    runtime: Any = Unprivileged(inner) if unprivileged else inner
    records: list[TraceRecord] = []
    counts: Counter[str] = Counter()
    read = _perceived(inner) if unprivileged else recycling.pose_channel
    with RobotContext(
        package.description,
        runtime,
        sinks=[records.append],
        object_poses=recycling.OBJECT_POSES,
    ) as ctx:
        privileged = set(runtime.withheld if unprivileged else ctx.info.privileged)
        result = recycling.sort_all(
            ctx, package, read=_counted(read, counts), evaluate=not unprivileged
        )
        final = {}
        if unprivileged:  # read only now, for the report
            for obj in sorted(ctx.info.objects):
                pose, _ = _perceived(inner)(ctx, obj)
                bins = [
                    Arm(side).bin[0]
                    for side in recycling.BIN_OF
                    if recycling._in_bin(pose, Arm(side))
                ]
                final[obj] = bins[0] if bins else None
    return {
        "control": _control(result),
        "registrations": dict(sorted(counts.items())),
        "provenance": _provenance(records, privileged),
        "final_bins": final or {o: f["bin"] for o, f in result["objects"].items()},
    }


def _b() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as work:
        return _round(Path(work), unprivileged=True)


def test_control_runs_on_registered_poses_alone(artifacts: Path, tmp_path: Path) -> None:
    a = _round(tmp_path / "a", unprivileged=False)
    first = json.dumps(_b(), indent=2) + "\n"
    again = json.dumps(_b(), indent=2) + "\n"
    b = json.loads(first)
    (artifacts / "registered_control.json").write_text(
        json.dumps({"a": a, "b": b, "b_repeat_identical": first == again}, indent=2) + "\n"
    )

    assert first == again
    assert b["control"] == a["control"]
    assert b["registrations"] == a["registrations"] and b["registrations"]
    assert b["provenance"]["privileged_readings"] == 0
    assert b["provenance"]["registered_poses_not_their_registration"] == 0
    assert a["provenance"]["registered_poses_not_their_registration"] == 0
    assert "measured" not in b["provenance"]["object_states"]
    assert b["final_bins"] == a["final_bins"]


MOVED_SEED = 1
"""A one-can scene whose can, moved ``MOVE_M`` along y, is missed cleanly: the fingers
close beside it without touching it, and either arm can still reach it."""
MOVE_M = 0.10
"""How far the can is moved, along the worktop's y, after it was registered."""


def _moved_and_retried(work: Path) -> dict[str, Any]:
    package = load_installed_package("geodude_assets")
    scene = stage_scene(
        work / "scene", package, objects={"can": 1}, fixtures=recycling.BINS, seed=MOVED_SEED
    )
    inner = recycling.runtime(package, scene)
    runtime = Unprivileged(inner)
    records: list[TraceRecord] = []
    counts: Counter[str] = Counter()
    moved: list[Pose] = []

    def move_once() -> None:
        """After the first step, while the grasp is under way: the can is pushed along the
        worktop, behind registration's back."""
        if moved:
            return
        model, data = inner._model, inner._data
        at = int(model.jnt_qposadr[int(model.body_jntadr[model.body("can_0").id])])
        data.qpos[at + 1] += MOVE_M
        data.qvel[int(model.jnt_dofadr[int(model.body_jntadr[model.body("can_0").id])]) :][:6] = 0
        moved.append(_perceived(inner)(ctx, "can_0")[0])

    with RobotContext(
        package.description,
        runtime,
        sinks=[records.append],
        object_poses=recycling.OBJECT_POSES,
    ) as ctx:
        attempts = recycling.recycle(
            ctx,
            package,
            ["can_0"],
            move_once,
            retries=1,
            read=_counted(_perceived(inner), counts),
            evaluate=False,
        )
        pose, _ = _perceived(inner)(ctx, "can_0")
        final = [
            Arm(side).bin[0] for side in recycling.BIN_OF if recycling._in_bin(pose, Arm(side))
        ]
    registered = [
        r.payload.registration("can_0")
        for r in records
        if isinstance(r.payload, SceneState) and r.source == recycling.REGISTRATION
    ]
    return {
        "attempts": [
            {
                "attempt": a["attempt"],
                "registration": a["registration"],
                "race": [
                    [c["label"], c["outcome"]] for c in (a.get("race") or {}).get("candidates", [])
                ],
                "grasp": a.get("grasp"),
                "failure": None if a.get("failure") is None else a["failure"]["reason"],
                "released": (a.get("checks") or {}).get("released"),
            }
            for a in attempts
        ],
        "registrations": counts["can_0"],
        "registered_at_move_m": [
            round(math.dist(r.pose.position, moved[0].position), 6) for r in registered if r
        ],
        "privileged_readings": _provenance(records, set(runtime.withheld))["privileged_readings"],
        "final_bin": final[0] if final else None,
    }


def test_a_missed_grasp_registers_the_table_again_and_retries(
    artifacts: Path, tmp_path: Path
) -> None:
    first = _moved_and_retried(tmp_path / "first")
    again = _moved_and_retried(tmp_path / "again")
    (artifacts / "moved_and_retried.json").write_text(json.dumps(first, indent=2) + "\n")

    assert first == again
    missed, retried = first["attempts"]
    assert missed["failure"] == "missed" and missed["grasp"][-1] == ["obstructed", False]
    assert retried["failure"] is None and retried["released"]
    # The retry planned on a later registration, which saw the can where it was moved.
    assert retried["registration"] > missed["registration"]
    assert first["registered_at_move_m"][-1] < 0.01 < first["registered_at_move_m"][0]
    assert first["registrations"] == 2
    assert first["privileged_readings"] == 0
    assert first["final_bin"] is not None
