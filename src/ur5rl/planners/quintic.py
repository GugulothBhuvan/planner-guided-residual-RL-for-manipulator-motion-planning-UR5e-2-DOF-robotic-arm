"""Quintic time scaling (SPEC §7.2 B1/B2): zero velocity and acceleration at both ends of every segment.

For q(s) = q0 + d (10 s^3 - 15 s^4 + 6 s^5), s = t / T: peak |velocity| = 1.875 |d| / T and peak
|acceleration| = 5.7735 |d| / T^2, so the shortest T within per-joint limits is closed-form.
"""

from __future__ import annotations

import numpy as np

V_PEAK, A_PEAK = 1.875, 10 / np.sqrt(3)  # 10/sqrt(3) = 5.7735
V_MAX, A_MAX = 0.9, 3.0  # rad/s, rad/s^2: B2 time scaling (the S4 horizon was derived with these, SPEC §4)
V_CAP = 0.98  # rad/s, joint speed the arm realises at |a| = 1 (SPEC §6.2): B1's shortest quintic


def duration(q0, q1, v_max=V_MAX, a_max=A_MAX):
    """Shortest quintic duration within the limits; a_max=None = velocity cap only (T = 1.875 max|dq| / v_max)."""
    d = np.abs(np.asarray(q1, float) - np.asarray(q0, float)).max()
    T = V_PEAK * d / v_max
    return T if a_max is None else max(T, np.sqrt(A_PEAK * d / a_max))


def blend(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def segment(q0, q1, T, dt):
    """Samples at t = dt, 2 dt, ..., ending exactly at q1."""
    n = max(1, int(np.ceil(T / dt)))
    s = np.arange(1, n + 1) / n
    return np.asarray(q0, float) + blend(s)[:, None] * (np.asarray(q1, float) - np.asarray(q0, float))


def time_scale(waypoints, dt=0.04, v_max=V_MAX, a_max=A_MAX):
    """Stop-and-go quintic through waypoints. Returns (trajectory (N, dof) incl. the start, total duration s)."""
    waypoints = np.asarray(waypoints, float)
    parts, total = [waypoints[:1]], 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        T = duration(a, b, v_max, a_max)
        total += np.ceil(T / dt) * dt if T > 0 else 0.0
        if T > 0:
            parts.append(segment(a, b, T, dt))
    return np.vstack(parts), total
