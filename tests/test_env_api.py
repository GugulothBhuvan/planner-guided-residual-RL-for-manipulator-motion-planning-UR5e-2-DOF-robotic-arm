import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv

STAGES = {
    "s1": EnvConfig(orientation=False, eps_pos=0.02, reward="sparse"),
    "s2": EnvConfig(reward="sparse"),
    "s4": EnvConfig(n_obstacles=(1, 4), episode_length=200, reward="dense"),
}


@pytest.mark.parametrize("stage", STAGES)
def test_env_checkers(stage):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gym_check_env(UR5ePoseEnv(STAGES[stage]), skip_render_check=True)
        sb3_check_env(UR5ePoseEnv(STAGES[stage]))


def test_s1_contact_neither_terminates_nor_penalises():
    """Collision termination and penalty apply from S3 on (SPEC §6.7, §6.8)."""
    env = UR5ePoseEnv(STAGES["s1"])
    env.reset(seed=0)
    hit = False
    for _ in range(60):  # shoulder lift +1 from near home drives the forearm into the table
        _, r, term, _, info = env.step(np.array([0, 1.0, 0, 0, 0, 0]))
        hit |= info["collision"]
        assert not term
        assert info["r_indep"] == 0.0
    assert hit


def test_step_reward_matches_compute_reward():
    env = UR5ePoseEnv(STAGES["s4"])
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert r == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        if term or trunc:
            obs, _ = env.reset()
