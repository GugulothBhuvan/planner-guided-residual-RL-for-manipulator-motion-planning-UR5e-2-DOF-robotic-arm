"""Reach env tests (SPEC M10): IK direction, reward consistency (also on relabelled batches), collision handling, and that the
pose env's defaults are unchanged."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcrl import robot  # noqa: E402
from mcrl.envs.pose_env import EnvConfig, MyCobotPoseEnv  # noqa: E402
from mcrl.envs.reach_env import MyCobotReachEnv, ReachConfig  # noqa: E402
from mcrl.reach_baselines import Detour, rollout, straight  # noqa: E402

GAMMA = 0.98


@pytest.fixture(scope="module")
def env():
    return MyCobotReachEnv(ReachConfig(episode_length=60))


def test_ee_action_moves_the_tool_in_the_commanded_direction(env):
    env.reset(seed=1)
    for direction in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([-1.0, 0, 0]), np.array([0, 0, -1.0])):
        env.reset(seed=1)
        p0 = env.tcp()
        for _ in range(8):
            env.step(direction)
        dp = env.tcp() - p0
        cos = float(dp @ direction / (np.linalg.norm(dp) + 1e-12))
        assert cos > 0.9, (direction, dp)
        assert np.linalg.norm(dp) > 0.0015  # the home pose is near a singularity going straight up


def test_joint_increment_never_exceeds_the_cap(env):
    env.reset(seed=2)
    for a in (np.array([1.0, 1, 1]), np.array([-1.0, 1, -1])):
        aj = env.ee_to_joint_action(a * env.cfg.ee_step)
        assert np.abs(aj).max() <= 1.0 + 1e-9


def test_reward_from_step_equals_compute_reward_and_relabelling(env):
    obs, _ = env.reset(seed=3)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isclose(r, env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
    assert r in (-1.0, 0.0)
    # relabelled batch: goal replaced by the achieved position -> success, reward 0 (+ the stored goal-independent part)
    ag = np.tile(obs["achieved_goal"], (4, 1))
    dg = ag.copy()
    r_rel = env.compute_reward(ag, dg, np.full(4, info["r_indep"]))
    assert np.allclose(r_rel, 0.0 + info["r_indep"])
    dg_far = ag.copy()
    dg_far[:, 0] += 0.5
    assert np.allclose(env.compute_reward(ag, dg_far, np.full(4, info["r_indep"])), -1.0 + info["r_indep"])


def test_contact_ends_the_episode_with_the_absorbing_penalty_and_relabelling_keeps_it():
    e = MyCobotReachEnv(ReachConfig(obstacle="mid", episode_length=60))
    ep = e.sample_episode()
    p0, _ = e._tcp_pose(ep["q_start"])
    ep["centers"][0], ep["radii"][0], ep["mask"][0] = p0, 0.03, True  # a sphere on the tool at the start: contact on the first step
    obs, _ = e.reset(seed=0, options={"episode": ep})
    obs, r, term, trunc, info = e.step(np.zeros(3))
    assert info["collision"] and term
    assert np.isclose(r, -1.0 / (1.0 - GAMMA)), r
    ag = obs["achieved_goal"][None]
    # relabelled to a goal that was "reached": the crash penalty is still there (-1/(1-gamma) + 1 at worst)
    r_rel = e.compute_reward(ag, ag.copy(), np.array([info["r_indep"]]))
    assert r_rel[0] <= -1.0 / (1.0 - GAMMA) + 1.0 + 1e-9


def test_scripted_straight_line_reaches_and_the_sphere_blocks_it():
    e = MyCobotReachEnv(ReachConfig(episode_length=250, cube_center=(0.12, -0.05, 0.35)))
    rows = rollout(e, lambda env: straight, n=5, seed0=100)
    assert all(r["success"] and not r["collision"] for r in rows), rows
    e2 = MyCobotReachEnv(ReachConfig(episode_length=250, obstacle="mid", min_goal_dist=0.10, cube_center=(0.12, -0.05, 0.35)))
    rows2 = rollout(e2, lambda env: straight, n=5, seed0=100)
    assert all(r["collision"] for r in rows2), rows2  # the sphere is on the straight line by construction
    rows3 = rollout(e2, lambda env: Detour(env), n=3, seed0=100)  # runs, ends (no assertion on the outcome: see SPEC M10)
    assert len(rows3) == 3


def test_defaults_leave_the_pose_env_unchanged():
    cfg = EnvConfig()
    assert (cfg.n_obstacles, cfg.eps_pos, cfg.reward, cfg.orientation, cfg.episode_length, cfg.gamma) == ((0, 0), 0.01, "dense", True, 100, None)
    pe = MyCobotPoseEnv(EnvConfig(n_obstacles=(1, 4), episode_length=900, goal_max_joint_dist=0.9))
    assert pe.action_space.shape == (6,)
    assert MyCobotReachEnv(ReachConfig(mode="joints")).action_space.shape == (6,)
    assert MyCobotReachEnv(ReachConfig(mode="ee")).action_space.shape == (3,)
