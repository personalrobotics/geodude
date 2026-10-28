"""``regions.pose`` recovers every rotation it is given (#201).

Every grasp region Geodude offers is an sstsr template converted by ``regions.pose``.
Half turns are where a matrix-to-quaternion conversion can lose a sign: the roll-180°
cylinder side grasps are half turns about an axis with two nonzero components, and the
old conversion turned them into hands facing away from the cylinder: half of every
cylinder's side grasps. No end-to-end run reported it.

Reproduce with ``uv run pytest tests/test_regions.py``.
"""

from __future__ import annotations

import itertools

import numpy as np
from ssrobot import Pose

from geodude import regions


def _half_turn(axis: np.ndarray) -> np.ndarray:
    u = axis / np.linalg.norm(axis)
    return 2 * np.outer(u, u) - np.eye(3)


def test_pose_recovers_every_rotation_including_half_turns() -> None:
    rng = np.random.default_rng(0)
    rotations = [
        regions.matrix(Pose(position=(0.0, 0.0, 0.0), quat_wxyz=tuple(q / np.linalg.norm(q))))[
            :3, :3
        ]
        for q in rng.normal(size=(500, 4))
    ]
    axes = {a for a in itertools.product((-1, 0, 1), repeat=3) if any(a)}
    rotations += [_half_turn(np.array(a, dtype=float)) for a in sorted(axes)]
    for r in rotations:
        transform = np.eye(4)
        transform[:3, :3] = r
        transform[:3, 3] = (0.1, -0.2, 0.3)
        assert np.allclose(regions.matrix(regions.pose(transform)), transform, atol=1e-9), r
