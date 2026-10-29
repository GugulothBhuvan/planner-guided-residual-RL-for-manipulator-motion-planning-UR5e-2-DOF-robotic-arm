"""Frozen test / validation sets (SPEC §9.2), stored by scripts/make_testsets.py in testsets/<name>.npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np

TESTSETS = Path(__file__).resolve().parents[3] / "testsets"
EPISODE_KEYS = ("q_start", "q_goal", "centers", "radii", "mask")


def load(name, root=None):
    """Returns (episodes, metadata). Each episode is a dict accepted by UR5ePoseEnv.reset(options={"episode": ep})
    (or KinematicArmEnv's, with root = 2-dof/testsets)."""
    z = np.load(Path(root or TESTSETS) / f"{name}.npz")
    n = len(z["seeds"])
    episodes = [{k: z[k][i] for k in EPISODE_KEYS} for i in range(n)]
    meta = {k: z[k] for k in z.files if k not in EPISODE_KEYS}
    return episodes, meta


def exists(name, root=None):
    return (Path(root or TESTSETS) / f"{name}.npz").exists()
