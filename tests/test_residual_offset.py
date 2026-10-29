"""SPEC §7.3 E5b: integrated residual offset (ResidualEnv mode="offset")."""

import mujoco
import pytest
import numpy as np

from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv
from ur5rl.envs.residual import ResidualEnv
from ur5rl.planners.baselines import ReferenceTracker

CFG = EnvConfig(n_obstacles=(1, 4), episode_length=300, reward="sparse")


def make(alpha=0.3):
    return ResidualEnv(UR5ePoseEnv(CFG), alpha=alpha, mode="offset")


def test_observation_carries_the_offset():
    env = make()
    obs, _ = env.reset(seed=93_000)
    plain = ResidualEnv(UR5ePoseEnv(CFG))
    assert env.observation_space["observation"].shape[0] == plain.observation_space["observation"].shape[0] + 6
    np.testing.assert_array_equal(obs["observation"][-6:], 0.0)
    obs, *_ = env.step(np.full(6, 0.5))
    np.testing.assert_allclose(obs["observation"][-6:], env.offset / env.offset_max, atol=1e-6)


def test_offset_persists_across_steps():
    env = make()
    env.reset(seed=93_001)
    env.step(np.array([1.0, -1.0, 0.5, 0, 0, 0]))  # one push: o = alpha * dq_max * a_res
    o = env.offset.copy()
    np.testing.assert_allclose(o, 0.3 * 0.04 * np.array([1.0, -1.0, 0.5, 0, 0, 0]))
    for _ in range(40):  # zero residual afterwards: the offset is kept, not cancelled by the base
        env.step(np.zeros(6))
        np.testing.assert_array_equal(env.offset, o)
    # and the arm follows the shifted reference once the target has caught up (far from the goal, s = 1)
    base = env.unwrapped
    assert env.offset_scale() == 1.0
    ref = env.reference[min(env.tracker.t, len(env.reference) - 1)]
    np.testing.assert_allclose(base.q_target - ref, o, atol=1e-6)


def test_offset_is_clipped():
    env = make()
    env.reset(seed=93_002)
    for _ in range(30):  # 0.3 rad at 0.012 rad per step: 25 steps
        env.step(np.ones(6))
    np.testing.assert_allclose(env.offset, env.offset_max)


def test_offset_changes_by_at_most_alpha_dq_max_per_step():
    env = make()
    env.reset(seed=93_005)
    rng = np.random.default_rng(0)
    bound = env.alpha * env.unwrapped.cfg.dq_max  # 0.012 rad
    for _ in range(60):
        before = env.offset.copy()
        env.step(rng.uniform(-3, 3, 6))  # out-of-range actions are clipped to [-1, 1] first
        assert np.abs(env.offset - before).max() <= bound + 1e-12


def test_offset_reaches_zero_at_the_goal():
    env = make()
    env.reset(seed=93_003)
    base = env.unwrapped
    q_goal = env.reference[-1]  # the reference ends at the goal pose (test_residual.py)
    base.data.qpos[:6] = q_goal
    base.data.qvel[:6] = 0
    base.q_target = q_goal.copy()
    mujoco.mj_forward(base.model, base.data)
    env.tracker.t = len(env.reference)  # past the end: the tracker holds the last point
    env.offset = np.full(6, 0.2)
    assert env.offset_scale() < 1e-3
    env.step(np.zeros(6))
    # at the goal the shifted target is the reference itself: the offset has no effect
    np.testing.assert_allclose(base.q_target, q_goal, atol=0.2 * env.offset_scale() + 1e-6)
    assert abs(env.offset_scale()) < 0.2


def test_zero_residual_is_base_only_exactly():
    res = make()
    plain = UR5ePoseEnv(CFG)
    res.reset(seed=93_004)
    plain.reset(seed=93_004)
    tracker = ReferenceTracker(plain, res.reference)
    for _ in range(200):
        _, r1, t1, tr1, info = res.step(np.zeros(6))
        _, r2, t2, tr2, _ = plain.step(tracker())
        np.testing.assert_array_equal(res.unwrapped.q_target, plain.q_target)
        np.testing.assert_array_equal(res.unwrapped.data.qpos, plain.data.qpos)
        assert r1 == r2 and t1 == t2 and tr1 == tr2
        if t1 or tr1:
            break


def make_v2(w_res=0.0):
    return ResidualEnv(UR5ePoseEnv(EnvConfig(n_obstacles=(0, 0), episode_length=400, reward="sparse")), alpha=0.3,
                       mode="offset", offset_fade_ref=0.1, penalty="offset", w_res=w_res)


def test_v2_fade_reaches_zero_by_the_end_of_the_reference_far_from_the_goal():
    env = make_v2()
    env.reset(seed=93_010)
    base, n = env.unwrapped, len(env.reference)
    for _ in range(30):  # saturate the offset (0.3 rad on every joint)
        env.step(np.ones(6))
    np.testing.assert_allclose(env.offset, env.offset_max)
    scales = []
    while env.tracker.t < n - 1:
        env.step(np.zeros(6))
        scales.append(env.offset_scale())
    d = float(np.linalg.norm(base.data.site_xpos[base.tcp_id] - base.goal[:3]))
    assert d > env.offset_fade  # far from the goal: the distance fade alone would still be 1
    assert env.offset_scale() == 0.0  # the reference fade brought it to 0 at the end of the reference
    assert all(a >= b for a, b in zip(scales, scales[1:]))  # monotone, linear over the last 10%
    for _ in range(15):  # the target then returns to the true goal configuration at the arm's speed
        env.step(np.zeros(6))
    np.testing.assert_allclose(base.q_target, env.reference[-1], atol=1e-9)


def test_v2_penalty_scales_with_the_offset():
    rewards = {}
    for c in (0.0, 0.15, 0.3):
        env = make_v2(w_res=0.1)
        env.reset(seed=93_011)
        env.offset = np.full(6, c)
        _, r, *_ = env.step(np.zeros(6))  # zero residual: the offset is unchanged this step
        rewards[c] = r
    for c in (0.15, 0.3):  # sparse reward -1 per step, minus w_res * |o / 0.3|^2
        assert rewards[c] == pytest.approx(rewards[0.0] - 0.1 * 6 * (c / 0.3) ** 2)
