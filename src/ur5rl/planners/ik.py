"""UR5e kinematics (SPEC §7.2): analytic FK and closed-form IK (up to 8 solutions).

DH parameters use the *menagerie* link lengths, so this FK matches the simulator exactly. With these
conventions the MuJoCo world frame is the DH base frame and `attachment_site` is DH tool0 (flange).
The official UR5e values differ by up to 0.7 mm (d1 0.1625, a3 -0.3922, d4 0.1333, d5 0.0997,
d6 0.0996); that gap matters for S6 (sim-to-real), not for simulation.
"""

from __future__ import annotations

import numpy as np

DH_D = np.array([0.163, 0.0, 0.0, 0.134, 0.1, 0.1])
DH_A = np.array([0.0, -0.425, -0.392, 0.0, 0.0, 0.0])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])


def dh(i, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(DH_ALPHA[i]), np.sin(DH_ALPHA[i])
    return np.array([
        [ct, -st * ca, st * sa, DH_A[i] * ct],
        [st, ct * ca, -ct * sa, DH_A[i] * st],
        [0.0, sa, ca, DH_D[i]],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk_flange(q):
    """4x4 pose of the flange (tool0) in the world frame."""
    T = np.eye(4)
    for i in range(6):
        T = T @ dh(i, q[i])
    return T


def fk_tcp(q, T_flange_tcp):
    """TCP pose = flange pose composed with the gripper's fixed flange->TCP transform."""
    return fk_flange(q) @ T_flange_tcp


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ik_flange(T, tol=1e-6):
    """All closed-form IK solutions (angles in (-pi, pi]) for flange pose T, each verified against FK.

    Standard UR decomposition (Hawkins 2013): theta1 from the wrist centre, theta5 and theta6 from the flange
    orientation, then a planar 2-link problem for theta2, theta3 and theta4. Wrist singularity (sin theta5 ~ 0):
    theta6 is free and set to 0.
    """
    d1, d4, d5, d6 = DH_D[0], DH_D[3], DH_D[4], DH_D[5]
    a2, a3 = DH_A[1], DH_A[2]
    sols = []
    p05 = T @ np.array([0.0, 0.0, -d6, 1.0])
    r = np.hypot(p05[0], p05[1])
    if r < d4:
        return []
    psi, phi = np.arctan2(p05[1], p05[0]), np.arccos(d4 / r)
    for th1 in (psi + phi + np.pi / 2, psi - phi + np.pi / 2):
        s1, c1 = np.sin(th1), np.cos(th1)
        c5 = (T[0, 3] * s1 - T[1, 3] * c1 - d4) / d6
        if abs(c5) > 1 + 1e-12:
            continue
        for th5 in (np.arccos(np.clip(c5, -1, 1)), -np.arccos(np.clip(c5, -1, 1))):
            s5 = np.sin(th5)
            if abs(s5) < 1e-9:
                th6 = 0.0
            else:
                th6 = np.arctan2((-T[0, 1] * s1 + T[1, 1] * c1) / s5, (T[0, 0] * s1 - T[1, 0] * c1) / s5)
            T14 = np.linalg.inv(dh(0, th1)) @ T @ np.linalg.inv(dh(4, th5) @ dh(5, th6))
            p13 = (T14 @ np.array([0.0, -d4, 0.0, 1.0]))[:3]
            L = np.linalg.norm(p13)
            c3 = (L * L - a2 * a2 - a3 * a3) / (2 * a2 * a3)
            if abs(c3) > 1 + 1e-12:
                continue
            for th3 in (np.arccos(np.clip(c3, -1, 1)), -np.arccos(np.clip(c3, -1, 1))):
                th2 = -np.arctan2(p13[1], -p13[0]) + np.arcsin(np.clip(a3 * np.sin(th3) / L, -1, 1))
                T34 = np.linalg.inv(dh(1, th2) @ dh(2, th3)) @ T14
                th4 = np.arctan2(T34[1, 0], T34[0, 0])
                q = _wrap(np.array([th1, th2, th3, th4, th5, th6]))
                if np.abs(fk_flange(q) - T).max() < tol and not any(np.allclose(q, s) for s in sols):
                    sols.append(q)
    return sols


def ik_tcp(T_tcp, T_flange_tcp, **kw):
    """IK for a TCP pose: solve for the flange pose T_tcp @ inv(T_flange_tcp)."""
    return ik_flange(T_tcp @ np.linalg.inv(T_flange_tcp), **kw)


def nearest(solutions, q_ref, lo=None, hi=None):
    """Solution nearest to q_ref (max-norm) among those within [lo, hi]; None if there is none."""
    if lo is not None:
        solutions = [s for s in solutions if np.all(s >= lo - 1e-9) and np.all(s <= hi + 1e-9)]
    if not solutions:
        return None
    return min(solutions, key=lambda s: np.abs(s - q_ref).max())
