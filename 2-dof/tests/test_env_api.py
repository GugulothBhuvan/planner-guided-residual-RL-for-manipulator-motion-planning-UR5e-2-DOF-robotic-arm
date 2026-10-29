import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from arm2dof.envs.kinematic_env import EnvConfig, KinematicArmEnv

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "env"


def stage(name, **overrides):
    """EnvConfig from the stage YAML the experiments use, so the tests cover the real settings."""
    cfg = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
    return EnvConfig(**{k: tuple(map(tuple, v)) if k == "fixed_layout" else v for k, v in cfg.items()} | overrides)


STAGES = {"p1": stage("p1"), "p2": stage("p2"), "p3": stage("p3", reward="dense"), "p1_clip": EnvConfig()}


@pytest.mark.parametrize("stage", STAGES)
def test_env_checkers(stage):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gym_check_env(KinematicArmEnv(STAGES[stage]), skip_render_check=True)
        sb3_check_env(KinematicArmEnv(STAGES[stage]))


@pytest.mark.parametrize("stage", STAGES)
def test_step_reward_matches_compute_reward(stage):
    env = KinematicArmEnv(STAGES[stage])
    env.reset(seed=0)
    for _ in range(80):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert r == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        assert r == env.compute_reward(obs["achieved_goal"][None], obs["desired_goal"][None],
                                       np.array([info["r_indep"]]))[0]  # SafetyHerReplayBuffer path
        if term or trunc:
            env.reset()


@pytest.mark.parametrize("stage", STAGES)
def test_observation_is_goal_independent(stage):
    env = KinematicArmEnv(STAGES[stage])
    env.reset(seed=1)
    env.step(env.action_space.sample())
    before = env._obs()["observation"]
    env.goal = np.array([0.1, -0.7])
    np.testing.assert_array_equal(env._obs()["observation"], before)


def test_starts_and_goals_collision_free():
    env = KinematicArmEnv(STAGES["p3"])
    for s in range(100):
        env.reset(seed=s)
        assert not env._collides(env.q)


def test_tolerance_is_settable():
    env = KinematicArmEnv(EnvConfig(eps=0.01))
    env.reset(seed=0)
    env.set_tolerance(0.05)
    ag, dg = np.zeros((1, 2)), np.array([[0.03, 0.0]])
    assert env.compute_reward(ag, dg, np.zeros(1))[0] == 0.0


def test_stage_configs_wrap_q1():
    for name in ("p1", "p2", "p3"):
        assert STAGES[name].wrap_q1
    assert STAGES["p1"].curriculum  # as UR5e E1
    assert not STAGES["p2"].curriculum and not STAGES["p3"].curriculum  # as UR5e S4 / E5
    assert STAGES["p1"].episode_length == 80


def test_observation_continuous_across_q1_seam():
    """With wrap on, q1 / q_limit is dropped: crossing +-pi changes the observation only slightly."""
    wrap, clip = KinematicArmEnv(STAGES["p1"]), KinematicArmEnv(STAGES["p1_clip"])
    assert clip.observation_space["observation"].shape[0] == wrap.observation_space["observation"].shape[0] + 1
    wrap.reset(seed=0)
    wrap.q = np.array([np.pi - 0.02, 0.3])
    wrap.step(np.array([0.2, 0.0]))  # +0.01 rad, before the seam
    a = wrap._obs()["observation"]
    wrap.step(np.array([0.4, 0.0]))  # +0.02 rad, across it
    assert wrap.q[0] < 0
    b = wrap._obs()["observation"]
    assert np.abs(a - b).max() < 0.25  # q-dot changes by 0.2 (units of max speed); nothing jumps


def test_goal_curriculum_bounds_joint_distance():
    env = KinematicArmEnv(STAGES["p1"])
    env.set_curriculum_level(0.6)
    n_bounded = 0
    for s in range(300):
        env.reset(seed=s)
        _, _, _, _, info = env.step(np.zeros(2))
        assert info["curriculum_goal"] == env.curriculum_goal
        if env.curriculum_goal:
            n_bounded += 1
            assert np.abs(env.qdiff(env.q_start, env.q_goal)).max() <= 0.6 + 1e-9
    assert 0.7 < n_bounded / 300 < 0.9  # 20% full-range goals


def test_goal_curriculum_crosses_seam():
    env = KinematicArmEnv(STAGES["p1"])
    env.set_curriculum_level(0.6)
    crossed = False
    for s in range(2000):
        env.reset(seed=s)
        if env.curriculum_goal and abs(env.q_goal[0] - env.q_start[0]) > np.pi:
            crossed = True
            break
    assert crossed


def test_curriculum_off_goals_full_range():
    env = KinematicArmEnv(EnvConfig(wrap_q1=True))
    d = []
    for s in range(300):
        env.reset(seed=s)
        assert not env.curriculum_goal
        d.append(np.abs(env.qdiff(env.q_start, env.q_goal)).max())
    assert max(d) > 2.5


@pytest.mark.parametrize("name", ["p2", "p3"])
def test_obstacle_scenes_are_connected(name):
    """§6.4: every sampled episode has an IK branch of the goal in the start's free C-space component."""
    from arm2dof.cspace.components import components
    from arm2dof.cspace.grid import cell_of, grid
    from arm2dof.planners.ik import ik
    env = KinematicArmEnv(STAGES[name])
    for s in range(15):
        env.reset(seed=s)
        labels = components(grid(env.centers[env.mask], env.radii[env.mask]))
        a = labels[cell_of(env.q_start)]
        assert a and any(labels[cell_of(q)] == a for q in ik(env.goal))


@pytest.mark.parametrize("name", ["p1", "p3"])
def test_episode_replay_reproduces_the_episode(name):
    env, other = KinematicArmEnv(STAGES[name]), KinematicArmEnv(STAGES[name])
    obs, _ = env.reset(seed=11)
    ep = env.episode()
    obs2, _ = other.reset(seed=999, options={"episode": ep})
    for k in obs:
        np.testing.assert_array_equal(obs[k], obs2[k])
    np.testing.assert_array_equal(other.q_goal, env.q_goal)
    np.testing.assert_array_equal(other.mask, env.mask)
