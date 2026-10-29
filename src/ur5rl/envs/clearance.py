"""Analytic robot-obstacle clearance (SPEC §6.5).

Every robot collision geom is a capsule (9 on the arm, 2 on the rigid gripper) and Week-1 obstacles are
spheres, so each distance is an exact segment-point distance minus both radii. These are the model's
actual collision geoms, so the result equals `mj_geomDistance` (checked in tests/test_clearance.py).
"""

from __future__ import annotations

import mujoco
import numpy as np

# The 7 "bodies" of the observation and margin reward: 6 links + gripper.
ROBOT_BODIES = ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
                "wrist_3_link", "gripper")


class Clearance:
    def __init__(self, model: mujoco.MjModel):
        ids, groups = [], []
        for b, name in enumerate(ROBOT_BODIES):
            bid = model.body(name).id
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == bid and model.geom_contype[g] != 0:
                    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                        raise ValueError(f"robot collision geom {model.geom(g).name} is not a capsule")
                    ids.append(g)
                    groups.append(b)
        self.geom_ids = np.array(ids)
        self.radius = model.geom_size[self.geom_ids, 0].copy()
        self.half_len = model.geom_size[self.geom_ids, 1].copy()
        groups = np.array(groups)
        assert (np.diff(groups) >= 0).all()
        self.body_start = np.searchsorted(groups, np.arange(len(ROBOT_BODIES)))  # for np.minimum.reduceat

    def segments(self, data: mujoco.MjData):
        """Capsule axis endpoints in world frame, each (n_caps, 3). Needs up-to-date kinematics."""
        c = data.geom_xpos[self.geom_ids]
        z = data.geom_xmat[self.geom_ids][:, 2::3]  # local z axis = 3rd column of the row-major matrix
        h = (self.half_len[:, None] * z)
        return c - h, c + h

    def capsule_distances(self, data, centers, radii):
        """Signed distance (n_caps, K) between every robot capsule and every sphere (negative = penetration)."""
        p0, p1 = self.segments(data)
        seg = p1 - p0
        rel = centers[None, :, :] - p0[:, None, :]  # (n_caps, K, 3)
        seg_len2 = np.maximum((seg * seg).sum(-1), 1e-12)[:, None]
        t = np.clip((rel * seg[:, None, :]).sum(-1) / seg_len2, 0.0, 1.0)
        closest = p0[:, None, :] + t[..., None] * seg[:, None, :]
        dist = np.linalg.norm(centers[None, :, :] - closest, axis=-1)
        return dist - self.radius[:, None] - radii[None, :]

    def body_min(self, data, centers, radii, mask, clip=0.3):
        """Min distance per robot body (7,) to the nearest active obstacle, clipped at `clip`."""
        if not mask.any():
            return np.full(len(ROBOT_BODIES), clip)
        d = self.capsule_distances(data, centers[mask], radii[mask]).min(axis=1)
        return np.minimum(np.minimum.reduceat(d, self.body_start), clip)
