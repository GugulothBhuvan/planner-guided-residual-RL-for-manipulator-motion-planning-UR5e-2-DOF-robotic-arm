"""Non-spherical obstacles (SPEC M4c): boxes and cylinders in the four obstacle slots.

An episode keeps its sphere fields (`centers`, `radii`, `mask`). With shapes it also stores, per slot,
  shape  0 sphere, 1 box, 2 cylinder (axis = local z)
  size   box: half extents (x, y, z); cylinder: (radius, half height, 0); sphere: unused (radii is used)
  quat   orientation (w, x, y, z)
`radii` is then the radius of the sphere that encloses the shape. That sphere is all the policy's observation and the
analytic clearance know (centre + radius per slot, so a trained policy takes them unchanged). Collisions (MuJoCo
contacts) and the planners use the exact shape; the cheap planner's inflation grows every half size by `inflate`.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

SPHERE, BOX, CYLINDER = 0, 1, 2
SHAPE_KEYS = ("shape", "size", "quat")
_TYPE = {SPHERE: mujoco.mjtGeom.mjGEOM_SPHERE, BOX: mujoco.mjtGeom.mjGEOM_BOX, CYLINDER: mujoco.mjtGeom.mjGEOM_CYLINDER}


def bounding_radius(kind, size, radius=0.0):
    if kind == BOX:
        return float(np.linalg.norm(size))
    if kind == CYLINDER:
        return float(np.hypot(size[0], size[1]))
    return float(radius)


def default_arrays(K):
    return np.zeros(K, np.int64), np.zeros((K, 3)), np.tile([1.0, 0, 0, 0], (K, 1))


def apply(model, data, geom, mocap, center, active, kind, size, quat, radius, inflate=0.0):
    """Place one obstacle slot: pose on the mocap body, geometry type and size on the geom."""
    data.mocap_pos[mocap] = center if active else (0.0, 0.0, -5.0)
    data.mocap_quat[mocap] = quat if kind != SPHERE else (1.0, 0.0, 0.0, 0.0)
    model.geom_type[geom] = _TYPE[kind]
    if kind == BOX:
        s = np.asarray(size, float) + inflate
        model.geom_size[geom] = s
        rb = float(np.linalg.norm(s))
    elif kind == CYLINDER:
        s = np.array([size[0] + inflate, size[1] + inflate, 0.0])
        model.geom_size[geom] = s
        rb = float(np.hypot(s[0], s[1]))
    else:
        rb = float(radius) + inflate
        model.geom_size[geom] = (rb, 0.0, 0.0)
    model.geom_rbound[geom] = rb
    model.geom_aabb[geom, :3] = 0.0
    model.geom_aabb[geom, 3:] = rb  # conservative broad phase for any orientation


def load_testset(name, root):
    """ur5rl's loader plus the shape fields when the set stores them."""
    from ur5rl.eval import testsets
    episodes, meta = testsets.load(name, root)
    z = np.load(Path(root) / f"{name}.npz")
    if "shape" in z.files:
        for i, ep in enumerate(episodes):
            for k in SHAPE_KEYS:
                ep[k] = z[k][i]
        meta = {k: v for k, v in meta.items() if k not in SHAPE_KEYS}
    return episodes, meta
