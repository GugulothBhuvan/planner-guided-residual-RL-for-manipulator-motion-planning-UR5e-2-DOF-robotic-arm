"""Companion SPEC §13 dynamics checks: analytic model properties, MuJoCo agreement, gravity acts on P5."""

import mujoco
import numpy as np

from arm2dof.dynamics.model import XML, Params, coriolis, energy, forward, gravity, mass_matrix

RNG = np.random.default_rng(0)


def load():
    m = mujoco.MjModel.from_xml_path(str(XML))
    return m, mujoco.MjData(m)


def random_state():
    return RNG.uniform([-np.pi, -2.5], [np.pi, 2.5]), RNG.uniform(-3, 3, 2)


def test_mass_matrix_symmetric_positive_definite():
    for _ in range(200):
        M = mass_matrix(random_state()[0])
        np.testing.assert_allclose(M, M.T)
        assert np.linalg.eigvalsh(M).min() > 0


def test_mdot_minus_2c_is_skew_symmetric():
    eps = 1e-6
    for _ in range(100):
        q, qd = random_state()
        Mdot = (mass_matrix(q + eps * qd) - mass_matrix(q - eps * qd)) / (2 * eps)
        N = Mdot - 2 * coriolis(q, qd)
        np.testing.assert_allclose(N, -N.T, atol=1e-7)


def test_gravity_acts_on_the_p5_model():
    m, d = load()
    assert m.ngravcomp == 0
    assert not m.jnt_actgravcomp.any()
    assert np.allclose(m.opt.gravity, [0, 0, -9.81])
    d.qpos[:] = [0.0, 0.0]  # horizontal, at rest, zero torque
    mujoco.mj_forward(m, d)
    expected = -np.linalg.solve(mass_matrix(d.qpos), gravity(d.qpos))
    np.testing.assert_allclose(d.qacc, expected, atol=1e-6)
    assert d.qacc[0] < 0  # the arm falls


def test_analytic_model_matches_mujoco():
    m, d = load()
    for _ in range(50):
        q, qd = random_state()
        q[1] = np.clip(q[1], -2.5, 2.5)  # away from the q2 limit (its constraint force is not modelled)
        tau = RNG.uniform(-5, 5, 2)
        d.qpos[:], d.qvel[:], d.ctrl[:] = q, qd, tau
        mujoco.mj_forward(m, d)
        M = np.zeros((2, 2))
        mujoco.mj_fullM(m, d, M)
        np.testing.assert_allclose(M, mass_matrix(q), atol=1e-9)
        np.testing.assert_allclose(d.qacc, forward(q, qd, tau), atol=1e-6)


def test_energy_conserved_without_torque_and_damping():
    m, d = load()
    m.dof_damping[:] = 0.0
    m.jnt_limited[:] = False  # a free double pendulum reaches the q2 limit; links do not collide, so drop it
    p = Params(b1=0.0, b2=0.0)
    d.qpos[:] = [0.3, 0.8]
    d.qvel[:] = [1.0, -0.5]
    mujoco.mj_forward(m, d)
    e0 = energy(d.qpos.copy(), d.qvel.copy(), p)
    for _ in range(int(10 / m.opt.timestep)):  # 10 s
        mujoco.mj_step(m, d)
    e1 = energy(d.qpos.copy(), d.qvel.copy(), p)
    assert abs(e1 - e0) < 1e-3 * max(1.0, abs(e0))
