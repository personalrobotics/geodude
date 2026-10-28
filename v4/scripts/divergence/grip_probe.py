"""Record how the 2F-140 holds each object through a development round (#201,
geodude_assets#36), for diagnosis only.

``grip_probe.py SEED OUT.json`` runs the round as ``evaluate_recycling.py`` does
(``recycling.sort_all``) and samples the simulator, without changing it, every
``SAMPLE_STEPS`` physics steps from each grasp's ``close`` until its object is detached:
- the stage (skill and name) and the object;
- each gripper link touching the object, with its summed normal force;
- the gripping hand's driver, coupler, spring-link and follower joints, and its actuator's
  force;
- the object's displacement from where it was held at ``attach``, relative to the
  gripper's base: position (mm) and rotation (rad);
- whether the attachment monitor still reports it held.

Sampling stops when the drop begins to release. Per attempt it also reports the first
sample whose displacement exceeds ``SLIP_MM``, the first the monitor reports lost, and
the sample at ``attach``.
"""

from __future__ import annotations

import collections
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

SAMPLE_STEPS = 5  # 10 ms at the 2 ms timestep
SLIP_MM = 1.0  # a displacement past this, from the grasp, is a slip
FINGER = ("driver", "coupler", "spring_link", "follower")


def main() -> int:
    import ssgrasping._skill as sk
    from ssrobot import RobotContext, load_installed_package

    from geodude import recycling
    from geodude.scene import stage_scene

    seed, out = int(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f"{out} exists")
    where: dict[str, Any] = {"skill": None, "name": None, "obj": None}
    for method in ("begin", "record"):
        original = getattr(sk.Run, method)

        def tracked(self: Any, name: str, *a: Any, _original: Any = original, **k: Any) -> Any:
            where.update(skill=self.skill, name=name, obj=self.obj)
            return _original(self, name, *a, **k)

        setattr(sk.Run, method, tracked)

    package = load_installed_package("geodude_assets")
    work = Path(tempfile.mkdtemp(prefix="geodude-grip-"))
    attempts: list[dict[str, Any]] = []
    try:
        scene = stage_scene(
            work / "scene",
            package,
            objects=recycling.mix(seed, 5),
            fixtures=recycling.BINS,
            seed=seed,
        )
        with RobotContext(package.description, recycling.runtime(package, scene)) as ctx:
            m, d = ctx._runtime._model, ctx._runtime._data  # measuring only
            name = lambda kind, i: mujoco.mj_id2name(m, kind, i) or ""  # noqa: E731
            body = {b: name(mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)}
            current: dict[str, Any] = {}
            count = [0]
            step = ctx.step

            def hand_of(obj: str) -> str | None:
                attachment = ctx.scene.attachment(obj)
                if attachment is None:
                    return None
                return "right_ur5e" if attachment.end_effector.startswith("right") else "left_ur5e"

            def sample() -> None:
                obj = where["obj"]
                grasping = where["skill"] == "grasp" and where["name"] in ("close", "obstructed")
                attached = obj is not None and ctx.scene.attachment(obj) is not None
                released = where["skill"] == "drop" and where["name"] not in (
                    "preflight",
                    "transfer",
                )
                if obj is None or released or not (grasping or attached):
                    if current:
                        attempts.append(dict(current))
                        current.clear()
                    return
                if current and current["object"] != obj:
                    attempts.append(dict(current))
                    current.clear()
                if not current:
                    current.update(object=obj, start_s=d.time, samples=[], reference=None)
                hand = hand_of(obj) or current.get("hand")
                if hand is None:  # closing before attach: the hand is the arm whose fingers touch
                    for i in range(d.ncon):
                        c = d.contact[i]
                        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                            touching = body[m.geom_bodyid[b]]
                            if body[m.geom_bodyid[a]] == obj and "/gripper/" in touching:
                                hand = touching.split("/")[0]
                if hand is None:
                    return
                current["hand"] = hand
                ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
                base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{hand}/gripper/base")
                forces: dict[str, float] = collections.defaultdict(float)
                f = np.zeros(6)
                for i in range(d.ncon):
                    c = d.contact[i]
                    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                    if ob in (b1, b2):
                        other = body[b2 if b1 == ob else b1]
                        mujoco.mj_contactForce(m, d, i, f)
                        key = other.split("/gripper/")[-1] if "/gripper/" in other else other
                        forces[key] += float(f[0])
                rel_p = d.xmat[base].reshape(3, 3).T @ (d.xpos[ob] - d.xpos[base])
                rel_r = d.xmat[base].reshape(3, 3).T @ d.xmat[ob].reshape(3, 3)
                if attached and current["reference"] is None:
                    current["reference"] = (rel_p.tolist(), rel_r.tolist())
                moved_mm = turned = None
                if current["reference"] is not None:
                    p0, r0 = (np.array(v) for v in current["reference"])
                    moved_mm = float(1000 * np.linalg.norm(rel_p - p0))
                    cos = (np.trace(r0.T @ rel_r) - 1) / 2
                    turned = float(np.arccos(np.clip(cos, -1, 1)))
                joints = {}
                for side in ("left", "right"):
                    for part in FINGER:
                        j = mujoco.mj_name2id(
                            m, mujoco.mjtObj.mjOBJ_JOINT, f"{hand}/gripper/{side}_{part}_joint"
                        )
                        joints[f"{side}_{part}"] = round(float(d.qpos[m.jnt_qposadr[j]]), 4)
                act = mujoco.mj_name2id(
                    m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{hand}/gripper/fingers_actuator"
                )
                attachment = ctx.scene.attachment(obj)
                current["samples"].append(
                    {
                        "t": round(d.time, 3),
                        "stage": f"{where['skill']}.{where['name']}",
                        "contacts_n": {k: round(v, 2) for k, v in sorted(forces.items())},
                        "joints": joints,
                        "actuator_n": round(float(d.actuator_force[act]), 3),
                        "ctrl": round(float(d.ctrl[act]), 3),
                        "moved_mm": None if moved_mm is None else round(moved_mm, 3),
                        "turned_rad": None if turned is None else round(turned, 4),
                        "held": None if attachment is None else attachment.held,
                    }
                )

            def measured_step() -> None:
                step()
                count[0] += 1
                if count[0] % SAMPLE_STEPS == 0:
                    sample()

            ctx.step = measured_step  # type: ignore[method-assign]
            result = recycling.sort_all(ctx, package)
            if current:
                attempts.append(dict(current))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for a in attempts:
        samples = a["samples"]
        slip = next((s for s in samples if (s["moved_mm"] or 0) > SLIP_MM), None)
        lost = next((s for s in samples if s["held"] is False), None)
        a["first_slip"] = slip
        a["monitor_lost"] = lost
        a["at_attach"] = next((s for s in samples if s["moved_mm"] is not None), None)
    verdicts = {o: f["bin"] for o, f in result["objects"].items()}
    failures = [
        (at["object"], *((at.get("failure") or {}).get(k) for k in ("skill", "stage", "detail")))
        for at in result["attempts"]
        if not at["succeeded"]
    ]
    out.write_text(
        json.dumps(
            {
                "seed": seed,
                "sources": recycling.sources(),
                "bins": verdicts,
                "failures": failures,
                "attempts": attempts,
            },
            indent=1,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
