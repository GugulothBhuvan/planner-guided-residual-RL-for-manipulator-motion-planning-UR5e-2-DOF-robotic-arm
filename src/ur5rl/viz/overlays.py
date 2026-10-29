"""Viewer overlays drawn into viewer.user_scn (SPEC §10): coordinate frames, TCP trail, safety shells."""

from __future__ import annotations

import mujoco
import numpy as np

AXIS_RGBA = (np.array([1, 0, 0, 1.0]), np.array([0, 0.8, 0, 1.0]), np.array([0, 0.3, 1, 1.0]))


def _add(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def _segment(scn, a, b, width, rgba):
    g = _add(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3), rgba)
    if g is not None:
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))


def frame(scn, pos, R, length=0.07, width=0.004, alpha=1.0):
    """RGB = xyz axes."""
    for k in range(3):
        _segment(scn, pos, pos + length * R[:, k], width, AXIS_RGBA[k] * [1, 1, 1, alpha])


def trail(scn, points, rgba=(1.0, 0.6, 0.1, 0.8), width=0.003):
    for a, b in zip(points[:-1], points[1:]):
        _segment(scn, a, b, width, rgba)


def safety_shells(scn, centers, radii, mask, d_safe, rgba=(1.0, 0.3, 0.3, 0.15)):
    for c, r, m in zip(centers, radii, mask):
        if m:
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [r + d_safe] * 3, c, np.eye(3), rgba)


def path(scn, points, rgba=(0.3, 0.8, 1.0, 0.7), width=0.004):
    """Base reference path (residual mode), as TCP points."""
    trail(scn, points, rgba, width)
