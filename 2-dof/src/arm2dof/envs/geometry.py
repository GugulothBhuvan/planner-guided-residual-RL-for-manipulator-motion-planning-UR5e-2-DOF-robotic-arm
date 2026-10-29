"""Exact planar collision geometry (SPEC §6.1): links are capsules (segments with radius), obstacles circles."""

from __future__ import annotations

import numpy as np

from arm2dof.planners.ik import joints

LINK_RADIUS = 0.03


def segment_point_distance(a, b, p):
    """Distance from points p (..., 2) to segments a-b (..., 2); broadcasts."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-18), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def link_clearance(q, centers, radii):
    """Signed clearance (..., 2 links, K) of each link capsule to each circle (negative = penetration).

    q: (..., 2); centers (K, 2); radii (K,).
    """
    elbow, ee = joints(q)
    base = np.zeros_like(elbow)
    starts = np.stack([base, elbow], -2)[..., None, :]  # (..., 2, 1, 2)
    ends = np.stack([elbow, ee], -2)[..., None, :]
    d = segment_point_distance(starts, ends, centers)  # (..., 2, K)
    return d - LINK_RADIUS - radii


def in_collision(q, centers, radii):
    if len(radii) == 0:
        return np.zeros(np.shape(q)[:-1], bool)
    return (link_clearance(q, centers, radii) < 0).any(axis=(-1, -2))
