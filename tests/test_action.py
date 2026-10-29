"""SPEC §6.2 action rule: integrated targets, lead cap, realised speed, gravity compensation; §6.5 goal filter."""

import mujoco
import numpy as np
import pytest

from ur5rl.envs.pose_env import HOME_Q, EnvConfig, UR5ePoseEnv


@pytest.fixture
def env():
    e = UR5ePoseEnv(EnvConfig())
    e.reset(seed=0)
    return e


def put_at(env, q):
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    d.qpos[:6] = d.ctrl[:6] = env.q_target = np.array(q, float)
    mujoco.mj_forward(m, d)


def test_reset_target_equals_q(env):
    np.testing.assert_array_equal(env.q_target, env.data.qpos[:6])


def test_target_integrates_and_is_capped(env):
    rng = np.random.default_rng(0)
    for _ in range(300):
        prev = env.q_target.copy()
        a = rng.uniform(-1, 1, 6)
        q_before = env.data.qpos[:6].copy()
        env.step(a)
        expected = np.clip(prev + a * env.cfg.dq_max, env.q_lo, env.q_hi)
        expected = q_before + np.clip(expected - q_before, -env.cfg.target_lead_max, env.cfg.target_lead_max)
        np.testing.assert_allclose(env.q_target, expected, atol=1e-12)
        assert np.all(np.abs(env.q_target - q_before) <= env.cfg.target_lead_max + 1e-12)


def test_observation_contains_target_lead(env):
    obs, *_ = env.step(np.ones(6))
    lead = (env.q_target - env.data.qpos[:6]) / env.cfg.target_lead_max
    np.testing.assert_allclose(obs["observation"][39:45], lead, atol=1e-6)


@pytest.mark.parametrize("joint, sign", [(0, 1), (1, -1), (2, -1), (3, 1), (4, 1), (5, 1)])
def test_realised_speed_close_to_dq_max(env, joint, sign):
    put_at(env, HOME_Q)
    a = np.zeros(6)
    a[joint] = sign
    for _ in range(10):
        env.step(a)
    q0 = env.data.qpos[joint]
    contact = False
    for _ in range(25):  # 1 s
        contact |= env.step(a)[4]["collision"]
    assert not contact
    assert abs(env.data.qpos[joint] - q0) > 0.95 * env.cfg.dq_max / 0.04


def test_gravity_compensated_hold(env):
    """Holding a target, the arm settles on it (no G/kp sag), like the UR's own controller."""
    for _ in range(5):
        q = env._sample_goal_q(HOME_Q)
        put_at(env, q)
        for _ in range(50):
            env.step(np.zeros(6))
        assert np.abs(env.data.qpos[:6] - q).max() < 1e-3


def test_goal_bound():
    e = UR5ePoseEnv(EnvConfig(goal_max_joint_dist=3.2, episode_length=150))
    for ep in range(50):
        e.reset(seed=ep)
        assert np.abs(e.q_goal - e.data.qpos[:6]).max() <= 3.2 + 1e-12
        assert np.all(e.q_goal >= -np.pi) and np.all(e.q_goal <= np.pi)


def test_goal_curriculum():
    """Bound starts at curriculum_start; ~20% of goals still use the full bound; level is settable."""
    e = UR5ePoseEnv(EnvConfig(goal_max_joint_dist=3.2, curriculum=True, curriculum_start=0.6))
    dists = []
    for ep in range(300):
        e.reset(seed=ep)
        dists.append(np.abs(e.q_goal - e.data.qpos[:6]).max())
    dists = np.array(dists)
    assert np.all(dists <= 3.2 + 1e-12)
    far = np.mean(dists > 0.6 + 1e-12)
    assert 0.08 < far < 0.25  # only the full-range 20% can exceed the level (some of those fall below 0.6)
    e.set_curriculum_level(1.5)
    e.reset(seed=1000)
    assert e.curriculum_level == 1.5
    for ep in range(100):
        e.reset(seed=2000 + ep)
        d = np.abs(e.q_goal - e.data.qpos[:6]).max()
        assert d <= 3.2 + 1e-12
