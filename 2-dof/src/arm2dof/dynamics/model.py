"""Analytic planar 2-link dynamics (companion SPEC §5.3): M(q) q'' + C(q, q') q' + g(q) + B q' = tau.

q is measured from the horizontal, gravity acts in the plane (P5 is vertical). Used by the computed-torque (B5)
and PD + gravity (B5') baselines and checked against MuJoCo (assets/arm2dof_torque.xml) in the tests. A Params
with scaled masses gives the perturbed model (§9.2: +-20% link masses).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

XML = Path(__file__).resolve().parents[3] / "assets" / "arm2dof_torque.xml"
TAU_MAX = np.array([20.0, 10.0])


@dataclass(frozen=True)
class Params:
    l1: float = 0.5
    l2: float = 0.4
    m1: float = 1.0
    m2: float = 1.0
    b1: float = 0.05
    b2: float = 0.05
    g: float = 9.81

    @property
    def lc1(self):
        return self.l1 / 2

    @property
    def lc2(self):
        return self.l2 / 2

    @property
    def I1(self):
        return self.m1 * self.l1 ** 2 / 12

    @property
    def I2(self):
        return self.m2 * self.l2 ** 2 / 12

    def scaled(self, mass_factor):
        """Perturbed model: both link masses (and so their inertias) scaled by mass_factor."""
        return replace(self, m1=self.m1 * mass_factor, m2=self.m2 * mass_factor)


def mass_matrix(q, p=Params()):
    c2 = np.cos(q[1])
    m11 = p.m1 * p.lc1 ** 2 + p.I1 + p.m2 * (p.l1 ** 2 + p.lc2 ** 2 + 2 * p.l1 * p.lc2 * c2) + p.I2
    m12 = p.m2 * (p.lc2 ** 2 + p.l1 * p.lc2 * c2) + p.I2
    m22 = p.m2 * p.lc2 ** 2 + p.I2
    return np.array([[m11, m12], [m12, m22]])


def coriolis(q, qd, p=Params()):
    h = p.m2 * p.l1 * p.lc2 * np.sin(q[1])
    return np.array([[-h * qd[1], -h * (qd[0] + qd[1])], [h * qd[0], 0.0]])


def gravity(q, p=Params()):
    c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
    return np.array([(p.m1 * p.lc1 + p.m2 * p.l1) * p.g * c1 + p.m2 * p.lc2 * p.g * c12, p.m2 * p.lc2 * p.g * c12])


def damping(qd, p=Params()):
    return np.array([p.b1, p.b2]) * qd


def forward(q, qd, tau, p=Params()):
    """q'' = M^-1 (tau - C q' - g - B q')."""
    return np.linalg.solve(mass_matrix(q, p), tau - coriolis(q, qd, p) @ qd - gravity(q, p) - damping(qd, p))


def inverse(q, qd, qdd, p=Params()):
    """tau = M q'' + C q' + g + B q' (computed-torque feedforward)."""
    return mass_matrix(q, p) @ qdd + coriolis(q, qd, p) @ qd + gravity(q, p) + damping(qd, p)


def energy(q, qd, p=Params()):
    """Kinetic + potential energy (zero potential at height 0)."""
    kin = 0.5 * qd @ mass_matrix(q, p) @ qd
    pot = p.g * (p.m1 * p.lc1 * np.sin(q[0]) + p.m2 * (p.l1 * np.sin(q[0]) + p.lc2 * np.sin(q[0] + q[1])))
    return kin + pot
