"""P5 model-based baselines (companion SPEC §7.2): B5 computed torque and B5' PD + gravity compensation, both tracking
a B1-style reference (nearest IK branch + quintic) built for the torque arm.

Reference limits V_REF / A_REF are chosen so the nominal feedforward torque stays within 90% of tau_max on every
reference (tested), leaving headroom for feedback. Controllers use the *nominal* model (Params()); the perturbed
plant (mass_factor 0.8 / 1.2) is what makes B5 and B5' imperfect (§9.2 robustness gap).
"""

from __future__ import annotations

import numpy as np

from arm2dof.dynamics.model import TAU_MAX, Params, gravity, inverse, mass_matrix
from arm2dof.planners.ik import ik
from ur5rl.planners.quintic import time_scale

V_REF, A_REF = 3.0, 8.0  # rad/s, rad/s^2
KP, KD = 100.0, 20.0  # B5 (acceleration units, critically damped)
M_DIAG = np.diag(mass_matrix(np.zeros(2)))
KP_TAU, KD_TAU = KP * M_DIAG, KD * M_DIAG  # B5': same nominal stiffness per joint, in torque units


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def reference(env):
    """(q_ref, qd_ref, qdd_ref) at the control rate: nearest IK branch (q1 the short way round), one quintic."""
    q0 = env.q_start
    sols = ik(env.goal) or [env.q_goal]
    q1 = min(sols, key=lambda s: abs(wrap(s[0] - q0[0])) + abs(s[1] - q0[1])).copy()
    q1[0] = q0[0] + wrap(q1[0] - q0[0])  # unwrapped target
    q_ref, _ = time_scale(np.array([q0, q1]), dt=env.dt, v_max=V_REF, a_max=A_REF)
    qd_ref = np.gradient(q_ref, env.dt, axis=0)
    qdd_ref = np.gradient(qd_ref, env.dt, axis=0)
    qd_ref[0] = qd_ref[-1] = 0.0
    return q_ref, qd_ref, qdd_ref


class Tracker:
    """B5 ('ct') or B5' ('pdg') on the nominal model; returns a = tau / tau_max."""

    def __init__(self, env, kind="ct", params=Params()):
        self.env, self.kind, self.p = env, kind, params
        self.ref = reference(env)
        self.t = 0

    def __call__(self, obs=None):
        q_ref, qd_ref, qdd_ref = self.ref
        k = min(self.t + 1, len(q_ref) - 1)
        self.t += 1
        q, qd = self.env.q, self.env.qd
        e = q_ref[k] - q
        e[0] = wrap(e[0])
        ed = qd_ref[k] - qd
        if self.kind == "ct":
            tau = inverse(q, qd, qdd_ref[k] + KD * ed + KP * e, self.p)
        else:
            tau = KP_TAU * e + KD_TAU * ed + gravity(q, self.p)
        return np.clip(tau / TAU_MAX, -1.0, 1.0)
