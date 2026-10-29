"""P5a torque env and baselines (companion SPEC §13): API, HER-safe reward, perturbed plant, B5 / B5' tracking."""

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from arm2dof.control.controllers import Tracker, reference
from arm2dof.dynamics.model import TAU_MAX, Params, inverse
from arm2dof.envs.torque_env import TorqueArmEnv, TorqueConfig


def test_env_checkers():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gym_check_env(TorqueArmEnv(), skip_render_check=True)
        sb3_check_env(TorqueArmEnv())


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_step_reward_matches_compute_reward(mode):
    env = TorqueArmEnv(TorqueConfig(reward=mode))
    env.reset(seed=0)
    for _ in range(60):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert r == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        assert r == env.compute_reward(obs["achieved_goal"][None], obs["desired_goal"][None],
                                       np.array([info["r_indep"]]))[0]


def test_success_needs_the_arm_to_stop():
    env = TorqueArmEnv()
    ag_fast, ag_slow, dg = np.array([0.5, 0.0, 0.05]), np.array([0.5, 0.0, 0.005]), np.array([0.5, 0.0, 0.0])
    assert env.compute_reward(ag_fast, dg, {"r_indep": 0.0}) == -1.0  # at the goal but moving 0.5 rad/s
    assert env.compute_reward(ag_slow, dg, {"r_indep": 0.0}) == 0.0


def test_perturbed_plant_scales_masses():
    nominal, heavy = TorqueArmEnv(), TorqueArmEnv(TorqueConfig(mass_factor=1.2))
    np.testing.assert_allclose(heavy.model.body_mass[1:], 1.2 * nominal.model.body_mass[1:])
    np.testing.assert_allclose(heavy.model.body_inertia[1:], 1.2 * nominal.model.body_inertia[1:])


def test_reference_feedforward_within_torque_limits():
    env = TorqueArmEnv()
    for s in range(100):
        env.reset(seed=s)
        q, qd, qdd = reference(env)
        tau = np.array([inverse(a, b, c, Params()) for a, b, c in zip(q, qd, qdd)])
        assert np.all(np.abs(tau) <= 0.9 * TAU_MAX)


@pytest.mark.parametrize("kind", ["ct", "pdg"])
def test_model_based_baselines_reach_and_stop(kind):
    env = TorqueArmEnv(TorqueConfig(episode_length=600))
    ok = 0
    for s in range(20):
        env.reset(seed=100 + s)
        ctrl = Tracker(env, kind)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(ctrl())
            done = term or trunc
        ok += info["is_success"]
    assert ok >= 18


def test_zero_residual_reproduces_b5_prime():
    from arm2dof.envs.residual_torque import ResidualTorqueEnv
    res, plain = ResidualTorqueEnv(TorqueArmEnv(), alpha=0.3), TorqueArmEnv()
    res.reset(seed=5)
    plain.reset(seed=5)
    ctrl = Tracker(plain, "pdg")
    assert res.observation_space["observation"].shape[0] == plain.observation_space["observation"].shape[0] + 6
    for _ in range(100):
        _, r1, *_ , info = res.step(np.zeros(2))
        a = ctrl()
        _, r2, *_ = plain.step(a)
        np.testing.assert_array_equal(info["a_base"], a)
        assert r1 == r2
