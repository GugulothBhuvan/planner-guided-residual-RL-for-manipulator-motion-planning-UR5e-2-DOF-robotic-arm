"""SPEC §6.7 absorbing collision penalty: crashing is always worse than timing out, in both reward modes."""

import numpy as np
import pytest

from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv

GAMMA = {"sparse": 0.98, "dense": 0.99}  # gamma_her (sparse + HER) and gamma (dense), configs/agent/*.yaml


def make(mode):
    return UR5ePoseEnv(EnvConfig(n_obstacles=(1, 4), episode_length=407, reward=mode, gamma=GAMMA[mode]))


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_per_step_min_is_a_lower_bound(mode):
    env = make(mode)
    r_min, rng = env.per_step_min(), np.random.default_rng(0)
    for ep in range(6):
        env.reset(seed=ep)
        for t in range(120):  # bang-bang actions maximise the smoothness and effort terms
            a = np.where(rng.random(6) < 0.5, -1.0, 1.0) * (1 if t % 2 else -1)
            _, r, term, trunc, info = env.step(a)
            if not info["collision"]:
                assert r >= r_min - 1e-9
            if term or trunc:
                break


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_crashing_is_always_worse_than_timing_out(mode):
    env = make(mode)
    g, r_min, r_coll = GAMMA[mode], env.per_step_min(), env.collision_reward()
    assert r_coll == pytest.approx(r_min / (1 - g) - env.cfg.w_c)
    # a crash now vs surviving any number of further steps, each at the worst possible reward
    for k in range(1, 2000):
        assert r_coll < r_min * (1 - g ** k) / (1 - g)
    # and the env really pays R_coll on a collision step: drop an obstacle onto the gripper
    env.reset(seed=3)
    tcp = env.data.site_xpos[env.tcp_id].copy()
    env._set_obstacles(np.r_[[tcp], np.zeros((env.K - 1, 3))], np.full(env.K, 0.06), np.r_[True, np.zeros(env.K - 1, bool)])
    _, r, term, _, info = env.step(np.zeros(6))
    assert info["collision"] and term
    assert r == pytest.approx(r_coll)


def test_legacy_one_off_penalty_without_gamma():
    """gamma=None keeps the v1 reward (archived runs): the collision costs only w_c on top of the step."""
    env = UR5ePoseEnv(EnvConfig(n_obstacles=(1, 4), episode_length=407, reward="sparse"))
    env.reset(seed=3)
    tcp = env.data.site_xpos[env.tcp_id].copy()
    env._set_obstacles(np.r_[[tcp], np.zeros((env.K - 1, 3))], np.full(env.K, 0.06), np.r_[True, np.zeros(env.K - 1, bool)])
    _, r, _, _, info = env.step(np.zeros(6))
    assert info["collision"] and r > -15
