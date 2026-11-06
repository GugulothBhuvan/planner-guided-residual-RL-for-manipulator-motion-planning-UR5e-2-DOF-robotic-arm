"""Frozen reach episodes (SPEC M10): testsets/reach_<stage>_{val,test}.npz with q_start (N,6), goal (N,3), centers (N,K,3),
radii (N,K), mask (N,K). Built by scripts/make_reach_sets.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save_set(path, episodes):
    np.savez(path, **{k: np.stack([e[k] for e in episodes]) for k in ("q_start", "goal", "centers", "radii", "mask")})


def load_set(stage, which):
    z = np.load(ROOT / "testsets" / f"reach_{stage}_{which}.npz")
    return [dict(q_start=z["q_start"][i], goal=z["goal"][i], centers=z["centers"][i], radii=z["radii"][i], mask=z["mask"][i])
            for i in range(len(z["goal"]))]
