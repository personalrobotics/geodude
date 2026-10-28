"""Stage a recycling scene from prl_assets: a closed, seeded scene directory (#199).

``stage_scene`` composes prl_assets objects and fixtures into one MuJoCo scene that
``MujocoRuntime(package, scene=...)`` composes with Geodude. Objects rest upright on the
worktop at poses sampled, with a seeded generator, from sstsr's ``StablePlacer`` regions,
as the legacy demo spawns them; fixtures (the recycling bins) stand where they are given.
No object may start within ``CLEARANCE_M`` of the robot at its start configuration, by
each hand's ssrobot planning scene: the legacy demo's worktop reaches under the arms' ready
pose, and an object staged against a finger is pushed as the run starts (#201). A scene
that breaks this is sampled again, with the next generator.

Each prl_assets object file is a standalone model with fixed names (body ``can``, joint
``can_joint``, mesh ``can_visual``), so every instance is attached under its own name
(``can_0``, ``can_1``, ...), with its mesh and texture files copied beside the scene. A
fixture's free joint is removed: in ssrobot a top-level body with one is an object.

The scene directory is written to a temporary sibling and renamed into place, with a
``manifest.json`` recording the prl_assets distribution, the seed, every placement, and
each staged file's SHA-256. The same inputs give the same bytes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from prl_assets import OBJECTS_DIR
from ssrobot import RobotContext, RobotPackage
from ssrobot.mujoco import MujocoRuntime
from tsr.placement.stable_placer import StablePlacer

WORKTOP_INSET_M = 0.05
"""How far inside the worktop's edges objects are placed, as the legacy demo does."""
CLEARANCE_M = 0.01
"""The least gap between two placed objects' footprints, and between an object and the
robot at its start configuration."""
ATTEMPTS = 200
"""Samples tried per object before staging fails."""
SCENE_ATTEMPTS = 20
"""Scenes sampled, each with its own generator, before staging fails because every one
puts an object within ``CLEARANCE_M`` of the robot. Over #201's development corpus, 12 of
40 seeds' first scenes do, 8 of them touching a finger; every seed stages by its sixth."""


@dataclass(frozen=True)
class Worktop:
    """Where objects rest: the worktop's centre on the table's collision surface, and
    its half extents in x and y."""

    centre: tuple[float, float, float]
    half_extents: tuple[float, float]


@dataclass(frozen=True)
class Placement:
    """One staged instance: its name, prl_assets type, and pose (centre, wxyz)."""

    name: str
    kind: str
    position: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


def worktop(package: RobotPackage) -> Worktop:
    """Geodude's worktop: the ``worktop`` site's footprint, at the top of the table's
    collision box beneath it (the site itself is a thin visual marker)."""
    entry = next(m for m in package.manifest.models if m.name == package.manifest.canonical_model)
    model = mujoco.MjModel.from_xml_path(str(package.root / entry.path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    x, y, _ = (float(v) for v in data.site_xpos[site])
    tops = [
        float(data.geom_xpos[g][2] + model.geom_size[g][2])
        for g in range(model.ngeom)
        if model.geom_contype[g]
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        and abs(x - data.geom_xpos[g][0]) <= model.geom_size[g][0]
        and abs(y - data.geom_xpos[g][1]) <= model.geom_size[g][1]
        and data.geom_xpos[g][2] + model.geom_size[g][2] <= data.site_xpos[site][2] + 0.01
    ]
    hx, hy = (float(v) for v in model.site_size[site][:2])
    return Worktop(centre=(x, y, max(tops)), half_extents=(hx, hy))


def meta(kind: str) -> dict[str, Any]:
    """A prl_assets object's ``meta.yaml``."""
    data = yaml.safe_load((Path(OBJECTS_DIR) / kind / "meta.yaml").read_text())
    assert isinstance(data, dict)
    return data


def _footprint(geometry: Mapping[str, Any]) -> float:
    """The radius of a circle covering an upright object's footprint."""
    if geometry["type"] == "cylinder":
        return float(geometry["radius"])
    x, y, _ = geometry["size"]
    return math.hypot(x, y) / 2


def _upright(placer: StablePlacer, geometry: Mapping[str, Any]) -> Any:
    """sstsr's region for the object resting on its -z face, upright as modelled."""
    if geometry["type"] == "cylinder":
        templates = placer.place_cylinder(geometry["radius"], geometry["height"])
    elif geometry["type"] == "box":
        templates = placer.place_box(*geometry["size"])
    else:
        raise ValueError(f"cannot place a {geometry['type']}")
    return next(t for t in templates if t.variant == "-z")


def _quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(matrix).flatten())
    if quat[0] < 0:
        quat = -quat
    w, x, y, z = (float(v) for v in quat)
    return w, x, y, z


def place(
    table: Worktop, objects: Mapping[str, int], seed: int, attempt: int = 0
) -> tuple[Placement, ...]:
    """Upright poses on the worktop for ``objects`` (type: count), sampled from sstsr's
    regions with ``numpy.random.default_rng(seed)``, or ``default_rng((seed, attempt))``
    for a later ``attempt``, no two footprints within ``CLEARANCE_M``. Types are placed
    in name order."""
    rng = np.random.default_rng(seed if attempt == 0 else (seed, attempt))
    hx, hy = table.half_extents
    placer = StablePlacer(hx - WORKTOP_INSET_M, hy - WORKTOP_INSET_M)
    surface = np.eye(4)
    surface[:3, 3] = table.centre
    placed: list[tuple[Placement, float]] = []
    for kind in sorted(objects):
        geometry = meta(kind)["geometric_properties"]
        region = _upright(placer, geometry).instantiate(surface)
        radius = _footprint(geometry)
        for index in range(objects[kind]):
            for _ in range(ATTEMPTS):
                pose = region.sample(rng=rng)
                xy = pose[:2, 3]
                if all(
                    np.hypot(*(xy - np.array(p.position[:2]))) >= radius + r + CLEARANCE_M
                    for p, r in placed
                ):
                    break
            else:
                raise ValueError(f"no room on the worktop for {kind}_{index}")
            x, y, z = (float(v) for v in pose[:3, 3])
            placed.append(
                (
                    Placement(
                        name=f"{kind}_{index}",
                        kind=kind,
                        position=(x, y, z + 0.0005),  # resting, not touching, at start
                        quat_wxyz=_quaternion(pose[:3, :3]),
                    ),
                    radius,
                )
            )
    return tuple(p for p, _ in placed)


def _attach(
    scene: Any, kind: str, name: str, pose: Placement | None, at: Sequence[float], out: Path
) -> dict[str, str]:
    """Attach prl_assets ``kind`` to ``scene`` as body ``name``, at ``pose`` (an object)
    or standing at ``at`` without its free joint (a fixture). Its asset files are copied
    under ``out/assets/<kind>/``. Returns each staged file's path and SHA-256."""
    source = Path(OBJECTS_DIR) / kind
    xml = source / meta(kind)["mujoco"]["xml_path"]
    child = mujoco.MjSpec.from_file(str(xml))
    if child.compiler.degree:
        raise ValueError(f"{xml} uses degrees; prl_assets objects are in radians")
    staged: dict[str, str] = {}
    for asset in (*child.meshes, *child.textures, *child.hfields, *child.skins):
        if not asset.file:
            continue
        relative = f"assets/{kind}/{asset.file}"
        if relative not in staged:
            data = (source / asset.file).read_bytes()
            (out / relative).parent.mkdir(parents=True, exist_ok=True)
            (out / relative).write_bytes(data)
            staged[relative] = hashlib.sha256(data).hexdigest()
        # Absolute while composing (an attached child resolves its files from its own
        # directory); made relative to the scene when it is written.
        asset.file = str(out / relative)
    (body,) = child.worldbody.bodies
    if pose is None:
        for joint in list(body.joints):
            child.delete(joint)
        body.pos, body.quat = list(at), [1.0, 0.0, 0.0, 0.0]
    else:
        body.pos, body.quat = list(pose.position), list(pose.quat_wxyz)
    original = body.name  # attaching renames the child's elements in place
    scene.attach(child, frame=scene.worldbody.add_frame(), prefix=f"{name}/")
    attached = next(b for b in scene.bodies if b.name == f"{name}/{original}")
    attached.name = name
    return staged


def stage_scene(
    out: str | os.PathLike[str],
    package: RobotPackage,
    *,
    objects: Mapping[str, int],
    fixtures: Mapping[str, Sequence[Sequence[float]]],
    seed: int,
) -> Path:
    """Stage the scene into the new directory ``out`` and return its ``scene.xml``.

    ``objects`` maps a prl_assets type to how many instances to place on ``package``'s
    worktop; ``fixtures`` maps a type to where each instance stands, its base at each
    position. Fails if ``out`` exists; leaves nothing behind on failure."""
    target = Path(out)
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    table = worktop(package)
    target.parent.mkdir(parents=True, exist_ok=True)
    rejected: list[dict[str, Any]] = []
    for attempt in range(SCENE_ATTEMPTS):
        placements = place(table, objects, seed, attempt)
        staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))
        try:
            _write(staging, placements, fixtures, seed)
            near = near_robot(package, staging / "scene.xml")
            if not near:
                manifest = json.loads((staging / "manifest.json").read_text())
                manifest["attempt"] = attempt
                manifest["rejected"] = rejected
                (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                os.rename(staging, target)
                return target / "scene.xml"
            rejected.append({"attempt": attempt, "near_robot": near})
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    raise ValueError(f"every one of {SCENE_ATTEMPTS} scenes puts an object against the robot")


def near_robot(package: RobotPackage, scene: Path) -> list[str]:
    """The objects of ``scene`` within ``CLEARANCE_M`` of the robot at its start
    configuration, by each hand's planning scene (one manipulator per end effector, the
    one with the fewest joints: a planning scene checks only its group's links)."""
    description = package.description
    groups: dict[str, str] = {}
    for m in description.manipulators:
        if m.end_effector is None:
            continue
        held = groups.get(m.end_effector)
        if held is None or len(description.group(m.group).joints) < len(
            description.group(held).joints
        ):
            groups[m.end_effector] = m.group
    near: set[str] = set()
    with RobotContext(description, MujocoRuntime(package, scene=scene)) as ctx:
        snapshot = ctx.snapshot()
        names = {o.name for o in snapshot.objects}
        for group in sorted(groups.values()):
            q = tuple(snapshot.position(j) for j in description.group(group).joints)
            planning = ctx.planning_scene(snapshot, group, clearance_m=CLEARANCE_M)
            for contact in planning.contacts(q):
                near |= {contact.first, contact.second} & names
    return sorted(near)


def _write(
    staging: Path,
    placements: Sequence[Placement],
    fixtures: Mapping[str, Sequence[Sequence[float]]],
    seed: int,
) -> None:
    """Write the scene of ``placements`` and ``fixtures``, and its manifest, into
    ``staging``."""
    scene = mujoco.MjSpec()
    scene.compiler.degree = False
    staged: dict[str, str] = {}
    for p in placements:
        staged |= _attach(scene, p.kind, p.name, p, (), staging)
    stands = []
    for kind in sorted(fixtures):
        for index, at in enumerate(fixtures[kind]):
            name = f"{kind}_{index}"
            staged |= _attach(scene, kind, name, None, at, staging)
            stands.append({"name": name, "kind": kind, "base": [float(v) for v in at]})
    xml = scene.to_xml().replace(f"{staging}{os.sep}", "")
    (staging / "scene.xml").write_text(xml)
    distribution = importlib.metadata.distribution("prl-assets")
    direct = distribution.read_text("direct_url.json")
    manifest = {
        "prl_assets": {
            "version": distribution.version,
            "source": None if direct is None else json.loads(direct),
        },
        "seed": seed,
        "objects": [
            {"name": p.name, "kind": p.kind, "position": p.position, "quat_wxyz": p.quat_wxyz}
            for p in placements
        ],
        "fixtures": stands,
        "files": {
            "scene.xml": hashlib.sha256(xml.encode()).hexdigest(),
            **dict(sorted(staged.items())),
        },
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
