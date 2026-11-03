"""Numerical IK for the MyCobot 280 (no closed form for this arm): batched damped least squares from many seeds.

ik_tcp(T_tcp, T_flange_tcp, lo, hi, q_start) returns the distinct solutions (within limits, position < 0.1 mm,
rotation < 0.1 deg) found from q_start and a deterministic set of random seeds (the seed set depends only on the goal,
so a plan is reproducible). Never uses the episode's q_goal. All seeds are iterated together (vectorised).
"""
from __future__ import annotations

import numpy as np

from mc280.kinematics import FIXED

N_SEEDS = 48
ITERS = 60
POS_TOL, ROT_TOL = 1e-4, np.deg2rad(0.1)


def _chain_batch(Q):
    """Joint frames for a batch (S, 6): origins (S,6,3), z axes (S,6,3) and the flange pose (S,4,4)."""
    S = len(Q)
    T = np.broadcast_to(np.eye(4), (S, 4, 4)).copy()
    O, Z = np.empty((S, 6, 3)), np.empty((S, 6, 3))
    for i in range(6):
        T = T @ FIXED[i]
        O[:, i], Z[:, i] = T[:, :3, 3], T[:, :3, 2]
        c, s = np.cos(Q[:, i]), np.sin(Q[:, i])
        Rz = np.zeros((S, 4, 4))
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2], Rz[:, 3, 3] = c, -s, s, c, 1, 1
        T = T @ Rz
    return O, Z, T


def fk_tcp(q, T_flange_tcp):
    _, _, T = _chain_batch(np.asarray(q, float)[None])
    return T[0] @ T_flange_tcp


def _rot_err(R, Rg):
    """Batched axis-angle vector of Rg R^T, (S,3)."""
    Rd = Rg[None] @ np.swapaxes(R, 1, 2)
    tr = np.trace(Rd, axis1=1, axis2=2)
    th = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    v = np.stack([Rd[:, 2, 1] - Rd[:, 1, 2], Rd[:, 0, 2] - Rd[:, 2, 0], Rd[:, 1, 0] - Rd[:, 0, 1]], axis=1)
    sin = np.sin(th)
    k = np.where(sin > 1e-6, th / (2 * np.maximum(sin, 1e-6)), 0.5)
    return k[:, None] * v


def ik_tcp(T_tcp, T_flange_tcp, lo, hi, q_start, n_seeds=N_SEEDS):
    T_goal = np.asarray(T_tcp, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    key = int(abs(np.sum(T_goal[:3, 3] * np.array([1e3, 1e5, 1e7]))) * 1e3) % (2**31)
    rng = np.random.default_rng(key)
    Q = np.vstack([np.asarray(q_start, float)[None], rng.uniform(lo, hi, (n_seeds - 1, 6))])
    pg, Rg = T_goal[:3, 3], T_goal[:3, :3]
    for _ in range(ITERS):
        O, Z, Tf = _chain_batch(Q)
        T = Tf @ T_flange_tcp
        ep = pg[None] - T[:, :3, 3]
        er = _rot_err(T[:, :3, :3], Rg)
        p = T[:, :3, 3]
        J = np.empty((len(Q), 6, 6))
        J[:, :3] = np.cross(Z, p[:, None, :] - O).transpose(0, 2, 1)
        J[:, 3:] = 0.5 * Z.transpose(0, 2, 1)  # rotation rows weighted by 0.5 (1 rad ~ 0.5 m)
        e = np.concatenate([ep, 0.5 * er], axis=1)
        lam = 1e-3 + 0.05 * (e * e).sum(1)
        A = J @ J.transpose(0, 2, 1) + lam[:, None, None] * np.eye(6)
        dq = (J.transpose(0, 2, 1) @ np.linalg.solve(A, e[..., None]))[..., 0]
        n = np.abs(dq).max(1, keepdims=True)
        dq *= np.minimum(1.0, 0.3 / np.maximum(n, 1e-12))
        Q = np.clip(Q + dq, lo, hi)
    O, Z, Tf = _chain_batch(Q)
    T = Tf @ T_flange_tcp
    ok = (np.linalg.norm(pg[None] - T[:, :3, 3], axis=1) < POS_TOL) & \
         (np.linalg.norm(_rot_err(T[:, :3, :3], Rg), axis=1) < ROT_TOL)
    sols = []
    for q in Q[ok]:
        if not any(np.abs(q - x).max() < 0.05 for x in sols):
            sols.append(q)
    return sols
