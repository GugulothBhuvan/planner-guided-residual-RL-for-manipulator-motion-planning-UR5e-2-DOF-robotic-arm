"""Planar residual env (base, fallback, zero-residual) and the gap-to-best-known-path metric."""

import numpy as np

from arm2dof.envs.kinematic_env import EnvConfig, KinematicArmEnv
from arm2dof.envs.residual import ResidualArmEnv
from arm2dof.eval.metrics import best_known_length, gap_to_best_known, trajectory_length
from arm2dof.planners.base import Tracker
from arm2dof.planners.ik import fk

P3 = EnvConfig(n_obstacles=(1, 3), episode_length=150, reward="dense", wrap_q1=True)


def test_wrap_q1_wraps_and_default_clips():
    wrap = KinematicArmEnv(EnvConfig(wrap_q1=True))
    clip = KinematicArmEnv(EnvConfig())
    for env in (wrap, clip):
        env.reset(seed=0)
        env.q = np.array([np.pi - 0.01, 0.0])
        env.step(np.array([1.0, 0.0]))
    assert wrap.q[0] < -np.pi + 0.05  # crossed the seam
    assert clip.q[0] == np.pi  # clipped (PE1 behaviour)


def test_base_reference_ends_at_goal_and_fallback_is_straight():
    env = ResidualArmEnv(KinematicArmEnv(P3))
    for seed in range(5):
        env.reset(seed=seed)
        np.testing.assert_allclose(fk(env.reference[-1]), env.unwrapped.goal, atol=1e-6)
    env.max_checks = 1  # always fails -> straight line
    _, info = env.reset(seed=10)
    assert not info["base_ok"]
    ref = env.reference
    d = ref[-1] - ref[0]
    for q in ref[1:-1]:
        t = (q - ref[0]) @ d / (d @ d)
        np.testing.assert_allclose(q, ref[0] + t * d, atol=1e-9)


def test_zero_residual_reproduces_base():
    res = ResidualArmEnv(KinematicArmEnv(P3), alpha=0.3)
    plain = KinematicArmEnv(P3)
    res.reset(seed=3)
    plain.reset(seed=3)
    tracker = Tracker(plain, res.reference)
    for _ in range(100):
        _, r1, t1, tr1, info = res.step(np.zeros(2))
        a = tracker()
        _, r2, t2, tr2, _ = plain.step(a)
        np.testing.assert_array_equal(info["a_base"], a)
        assert r1 == r2 and t1 == t2 and tr1 == tr2
        if t1 or tr1:
            break


def test_planned_base_reaches_goals():
    env = ResidualArmEnv(KinematicArmEnv(P3), alpha=0.0, max_checks=20_000)
    ok = 0
    for seed in range(10):
        _, info = env.reset(seed=100 + seed)
        if not info["base_ok"]:
            continue
        done = False
        while not done:
            _, _, t, tr, i = env.step(np.zeros(2))
            done = t or tr
        ok += i["is_success"] and not i["collision"]
    assert ok >= 7


def test_gap_to_best_known_path():
    rng = np.random.default_rng(0)
    centers, radii = np.array([[0.45, 0.25]]), np.array([0.1])
    q0, q1 = np.array([-0.5, 0.4]), np.array([1.6, 0.3])
    l_best, parts = best_known_length(centers, radii, q0, q1, rng)
    assert set(parts) == {"astar", "rrt"} and l_best == min(parts.values())
    assert l_best >= np.linalg.norm(q1 - q0) - 1e-9  # never shorter than the straight line
    straight = np.linspace(q0, q1, 50)
    assert abs(trajectory_length(straight) - np.linalg.norm(q1 - q0)) < 1e-9
    assert gap_to_best_known(straight, np.linalg.norm(q1 - q0)) < 1e-9
