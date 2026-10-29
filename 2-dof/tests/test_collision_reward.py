"""Absorbing collision penalty (UR5e SPEC §6.7, companion SPEC §6.5): crashing is always worse than timing out."""

import numpy as np
import pytest

from arm2dof.envs.kinematic_env import EnvConfig, KinematicArmEnv
from arm2dof.planners.ik import fk

GAMMA = {"sparse": 0.98, "dense": 0.99}


def make(mode):
    return KinematicArmEnv(EnvConfig(n_obstacles=(1, 4), episode_length=303, reward=mode, wrap_q1=True,
                                     gamma=GAMMA[mode]))


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_per_step_min_is_a_lower_bound(mode):
    env = make(mode)
    r_min, rng = env.per_step_min(), np.random.default_rng(0)
    for ep in range(20):
        env.reset(seed=ep)
        for t in range(200):
            a = np.where(rng.random(2) < 0.5, -1.0, 1.0) * (1 if t % 2 else -1)
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
    for k in range(1, 2000):
        assert r_coll < r_min * (1 - g ** k) / (1 - g)
    env.reset(seed=3)
    env.centers[0], env.radii[0], env.mask[0] = fk(env.q), 0.05, True  # obstacle on the end-effector
    _, r, term, _, info = env.step(np.zeros(2))
    assert info["collision"] and term
    assert r == pytest.approx(r_coll)
