"""Planar 2-link kinematics (SPEC §5.2): FK, closed-form IK (elbow-up / elbow-down), Jacobian."""

from __future__ import annotations

import numpy as np

L1, L2 = 0.5, 0.4
Q_LO = np.array([-np.pi, -0.9 * np.pi])
Q_HI = np.array([np.pi, 0.9 * np.pi])


def joints(q):
    """Positions of the elbow and the end-effector, each (..., 2)."""
    q = np.asarray(q, float)
    q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
    elbow = np.stack([L1 * np.cos(q1), L1 * np.sin(q1)], -1)
    ee = elbow + np.stack([L2 * np.cos(q12), L2 * np.sin(q12)], -1)
    return elbow, ee


def fk(q):
    return joints(q)[1]


def ik(p):
    """All IK solutions within joint limits for EE position p: list of q (0, 1 or 2 entries).

    Elbow-down (q2 > 0) first. Unreachable or limit-violating solutions are dropped.
    """
    x, y = float(p[0]), float(p[1])
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1.0 + 1e-12:
        return []
    c2 = np.clip(c2, -1.0, 1.0)
    sols = []
    for s in (1.0, -1.0):
        q2 = s * np.arccos(c2)
        q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
        q = np.array([q1, q2])
        if np.all(q >= Q_LO - 1e-12) and np.all(q <= Q_HI + 1e-12):
            if not any(np.allclose(q, o) for o in sols):  # q2 = 0 gives one solution
                sols.append(q)
    return sols


def ik_nearest(p, q_ref):
    """IK branch nearest to q_ref (wrap-aware in q1), or None if unreachable."""
    sols = ik(p)
    if not sols:
        return None
    d = [np.abs(np.r_[(s[0] - q_ref[0] + np.pi) % (2 * np.pi) - np.pi, s[1] - q_ref[1]]).sum() for s in sols]
    return sols[int(np.argmin(d))]


def jacobian(q):
    q1, q12 = q[0], q[0] + q[1]
    return np.array([
        [-L1 * np.sin(q1) - L2 * np.sin(q12), -L2 * np.sin(q12)],
        [L1 * np.cos(q1) + L2 * np.cos(q12), L2 * np.cos(q12)],
    ])
