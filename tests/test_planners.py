"""SPEC §13 planners: IK satisfies FK, quintic boundary conditions and limits, B2 paths collision-free at a dense
re-check, and B1 tracked through the env reaches its goal."""

import numpy as np
import pytest

from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv
from ur5rl.planners.baselines import ReferenceTracker, flange_to_tcp, plan_b1, plan_b2, rollout
from ur5rl.planners.collision import CollisionChecker
from ur5rl.planners.ik import fk_flange, fk_tcp, ik_flange, ik_tcp, nearest
from ur5rl.planners.quintic import A_MAX, V_MAX, duration, segment, time_scale


def test_ik_solutions_satisfy_fk_and_include_generator():
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = rng.uniform(-np.pi, np.pi, 6)
        T = fk_flange(q)
        sols = ik_flange(T)
        assert any(np.allclose(s, q, atol=1e-6) for s in sols)
        for s in sols:
            assert np.abs(fk_flange(s) - T).max() < 1e-4


def test_ik_tcp_and_nearest():
    env = UR5ePoseEnv(EnvConfig())
    env.reset(seed=0)
    T_ft = flange_to_tcp(env)
    q = env.q_goal
    sols = ik_tcp(fk_tcp(q, T_ft), T_ft)
    best = nearest(sols, q, env.q_lo, env.q_hi)
    np.testing.assert_allclose(best, q, atol=1e-6)


def test_quintic_boundaries_and_limits():
    q0, q1 = np.zeros(6), np.array([2.0, -1.0, 0.5, 0.0, 3.0, -0.2])
    T = duration(q0, q1)
    dt = 1e-3
    traj = np.vstack([q0, segment(q0, q1, T, dt)])
    np.testing.assert_allclose(traj[-1], q1)
    v = np.diff(traj, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    assert np.abs(v).max() <= V_MAX * 1.01 and np.abs(a).max() <= A_MAX * 1.01
    assert np.abs(v[0]).max() < 1e-2 and np.abs(v[-1]).max() < 1e-2  # starts and ends at rest


def test_time_scale_through_waypoints():
    wp = np.array([np.zeros(6), np.full(6, 0.5), np.full(6, -0.3)])
    ref, total = time_scale(wp, dt=0.04)
    np.testing.assert_allclose(ref[0], wp[0])
    np.testing.assert_allclose(ref[-1], wp[-1])
    assert abs(total - (len(ref) - 1) * 0.04) < 1e-9


@pytest.fixture(scope="module")
def s4_env():
    return UR5ePoseEnv(EnvConfig(n_obstacles=(1, 4), episode_length=500))


def test_b2_paths_are_collision_free_at_dense_recheck(s4_env):
    checker, rng = CollisionChecker(), np.random.default_rng(0)
    planned = 0
    for seed in range(8):
        s4_env.reset(seed=70_000 + seed)
        p = plan_b2(s4_env, rng, checker)
        if not p.ok:
            continue
        planned += 1
        checker.set_obstacles(s4_env.ob_center, s4_env.ob_radius, s4_env.ob_mask)
        for a, b in zip(p.waypoints[:-1], p.waypoints[1:]):
            n = int(np.ceil(np.abs(b - a).max() / 0.001))  # 10x finer than the planner's 0.01 rad
            for t in np.linspace(0, 1, n + 1):
                assert not checker(a + t * (b - a))
        np.testing.assert_allclose(p.waypoints[0], s4_env.q_start)
    assert planned >= 6


def test_b1_tracked_through_env_reaches_goal():
    env = UR5ePoseEnv(EnvConfig(reward="sparse", goal_max_joint_dist=1.5, episode_length=150))
    reached = 0
    for seed in range(5):
        env.reset(seed=80_000 + seed)
        p = plan_b1(env)
        assert p.ok
        info, _ = rollout(env, ReferenceTracker(env, p.reference))
        reached += info["is_success"]
    assert reached == 5
