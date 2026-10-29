"""Exact C-space occupancy grid at 1 degree (SPEC §7.2 B3, §7.3).

Rows index q1 (360 cells, cylinder: row 0 and row 359 are neighbours), columns index q2 (324 cells within
+-0.9 pi). Cell values are evaluated at cell centres with the analytic collision check.
"""

from __future__ import annotations

import numpy as np

from arm2dof.envs.geometry import in_collision

RES = np.deg2rad(1.0)
Q1 = -np.pi + RES * (np.arange(360) + 0.5)
Q2 = -0.9 * np.pi + RES * (np.arange(324) + 0.5)


def configs():
    """All cell-centre configurations, (360, 324, 2)."""
    return np.stack(np.meshgrid(Q1, Q2, indexing="ij"), -1)


def grid(centers, radii):
    """Boolean occupancy (True = collision), (360, 324)."""
    return in_collision(configs(), np.asarray(centers, float).reshape(-1, 2), np.asarray(radii, float))


def cell_of(q):
    """Nearest cell (i, j) for configuration q; q1 wraps."""
    i = int(np.floor((((q[0] + np.pi) % (2 * np.pi)) / RES))) % 360
    j = int(np.clip(np.floor((q[1] + 0.9 * np.pi) / RES), 0, 323))
    return i, j


def config_of(i, j):
    return np.array([Q1[i % 360], Q2[j]])
