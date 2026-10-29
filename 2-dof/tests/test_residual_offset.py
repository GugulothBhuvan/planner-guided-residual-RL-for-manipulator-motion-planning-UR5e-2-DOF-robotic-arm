"""Companion SPEC §8 PE4b: integrated residual offset on the planar arm (ResidualArmEnv mode="offset")."""

import numpy as np

from arm2dof.envs.kinematic_env import EnvConfig, KinematicArmEnv
from arm2dof.envs.residual import ResidualArmEnv
from arm2dof.planners.base import Tracker
from arm2dof.planners.ik import fk

P3 = EnvConfig(n_obstacles=(1, 3), episode_length=150, reward="sparse", wrap_q1=True)


def make():
    return ResidualArmEnv(KinematicArmEnv(P3), alpha=0.3, mode="offset")


def test_offset_persists_and_is_bounded_per_step():
    env = make()
    env.reset(seed=20)
    bound = 0.3 * env.unwrapped.cfg.dq_max
    before = env.offset.copy()
    env.step(np.array([1.0, -0.5]))
    np.testing.assert_allclose(env.offset, bound * np.array([1.0, -0.5]))
    assert np.abs(env.offset - before).max() <= bound + 1e-12
    o = env.offset.copy()
    for _ in range(20):
        _, _, term, trunc, _ = env.step(np.zeros(2))
        np.testing.assert_array_equal(env.offset, o)  # kept, not cancelled by the base
        if term or trunc:
            break
    rng = np.random.default_rng(0)
    env.reset(seed=21)
    for _ in range(40):
        before = env.offset.copy()
        _, _, term, trunc, _ = env.step(rng.uniform(-3, 3, 2))
        assert np.abs(env.offset - before).max() <= bound + 1e-12
        if term or trunc:
            break


def test_offset_reaches_zero_at_the_goal():
    env = make()
    env.reset(seed=22)
    base = env.unwrapped
    base.q = env.reference[-1].copy()  # at the goal configuration
    np.testing.assert_allclose(fk(base.q), base.goal, atol=1e-9)
    assert env.offset_scale() < 1e-6
    env.offset = np.full(2, 0.2)
    env.tracker.t = len(env.reference)
    env.step(np.zeros(2))
    np.testing.assert_allclose(base.q, env.reference[-1], atol=1e-9)  # the offset had no effect at the goal


def test_zero_residual_is_base_only_exactly():
    res = make()
    plain = KinematicArmEnv(P3)
    res.reset(seed=23)
    plain.reset(seed=23)
    tracker = Tracker(plain, res.reference)
    for _ in range(150):
        _, r1, t1, tr1, _ = res.step(np.zeros(2))
        _, r2, t2, tr2, _ = plain.step(tracker())
        np.testing.assert_array_equal(res.unwrapped.q, plain.q)
        assert r1 == r2 and t1 == t2 and tr1 == tr2
        if t1 or tr1:
            break


def make_v2(w_res=0.0):
    cfg = EnvConfig(n_obstacles=(0, 0), episode_length=300, reward="sparse", wrap_q1=True)
    return ResidualArmEnv(KinematicArmEnv(cfg), alpha=0.3, mode="offset", offset_fade_ref=0.1, penalty="offset",
                          w_res=w_res)


def test_v2_fade_reaches_zero_by_the_end_of_the_reference_far_from_the_goal():
    env = make_v2()
    for seed in range(30, 60):  # an episode whose reference is long enough to saturate the offset first
        env.reset(seed=seed)
        if len(env.reference) > 40:
            break
    n = len(env.reference)
    env.offset_fade = 1e-9  # the distance fade never engages (as with a tool held far from the goal)
    for _ in range(25):  # 0.3 rad at 0.015 rad per step
        env.step(np.ones(2))
    np.testing.assert_allclose(env.offset, env.offset_max)
    while env.tracker.t < n - 1:
        env.step(np.zeros(2))
    base = env.unwrapped
    assert env.offset_scale() == 0.0  # the reference fade alone brought it to 0
    for _ in range(30):
        env.step(np.zeros(2))
    np.testing.assert_allclose(base.qdiff(base.q, env.reference[-1]), 0.0, atol=1e-9)


def test_v2_penalty_scales_with_the_offset():
    rewards = {}
    for c in (0.0, 0.15, 0.3):
        env = make_v2(w_res=0.1)
        env.reset(seed=31)
        env.offset = np.full(2, c)
        _, r, *_ = env.step(np.zeros(2))
        rewards[c] = r
    for c in (0.15, 0.3):
        assert abs(rewards[c] - (rewards[0.0] - 0.1 * 2 * (c / 0.3) ** 2)) < 1e-9
