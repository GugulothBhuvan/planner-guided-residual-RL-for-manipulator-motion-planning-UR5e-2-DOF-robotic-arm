"""SPEC §6.7 / §13: compute_reward vectorisation and the HER reward decomposition."""

import pickle

import numpy as np
import pytest
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env

from ur5rl.agents.her import SafetyHerReplayBuffer
from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv

# Sparse S4: goal-independent terms (smoothness, margin, collision) are non-zero and must survive relabeling.
CFG = EnvConfig(n_obstacles=(1, 4), episode_length=50, reward="sparse")


def rollout_buffer(buffer_cls, n_steps=400, **kw):
    venv = make_vec_env(UR5ePoseEnv, n_envs=2, seed=0, env_kwargs=dict(config=CFG))
    buf = buffer_cls(10_000, venv.observation_space, venv.action_space, env=venv, n_envs=2, **kw)
    obs = venv.reset()
    infos_at = {}
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        a = rng.uniform(-1, 1, (2, 6)).astype(np.float32)
        next_obs, r, done, infos = venv.step(a)
        real_next = {k: v.copy() for k, v in next_obs.items()}
        for i, d in enumerate(done):  # VecEnv auto-resets; the buffer must get the true final observation
            if d:
                for k in real_next:
                    real_next[k][i] = infos[i]["terminal_observation"][k]
        infos_at[buf.pos] = [dict(i) for i in infos]
        buf.add(obs, real_next, a, r, done, infos)
        obs = next_obs
    return venv, buf, infos_at


def test_batch_equals_per_sample():
    env = UR5ePoseEnv(CFG)
    env.reset(seed=1)
    ag, dg, infos = [], [], []
    for _ in range(30):
        obs, _, term, trunc, info = env.step(env.action_space.sample())
        ag.append(obs["achieved_goal"])
        dg.append(obs["desired_goal"])
        infos.append(info)
        if term or trunc:
            env.reset()
    batch = env.compute_reward(np.array(ag), np.array(dg), infos)
    single = [env.compute_reward(a, d, i) for a, d, i in zip(ag, dg, infos)]
    np.testing.assert_array_equal(batch, single)


def test_relabeled_rewards_keep_safety_terms():
    venv, buf, infos_at = rollout_buffer(SafetyHerReplayBuffer, n_sampled_goal=4, goal_selection_strategy="future")
    env = venv.envs[0].unwrapped
    is_valid = buf.ep_length > 0
    valid = np.flatnonzero(is_valid)
    rng = np.random.default_rng(1)
    pick = rng.choice(valid, 256)
    b_idx, e_idx = np.unravel_index(pick, is_valid.shape)
    samples = buf._get_virtual_samples(b_idx, e_idx)
    next_ag = samples.next_observations["achieved_goal"].numpy()
    new_dg = samples.observations["desired_goal"].numpy()
    infos = [infos_at[b][e] for b, e in zip(b_idx, e_idx)]
    expected = env.compute_reward(next_ag, new_dg, infos).astype(np.float32)
    np.testing.assert_array_equal(samples.rewards.numpy().ravel(), expected)
    r_indep = np.array([i["r_indep"] for i in infos])
    assert (r_indep < 0).mean() > 0.5  # the safety terms are really present in these transitions
    without = env.compute_reward(next_ag, new_dg, np.zeros(len(infos))).astype(np.float32)
    assert not np.allclose(samples.rewards.numpy().ravel(), without)


@pytest.mark.parametrize("n_obstacles", [(0, 0), (1, 4)])
def test_observation_is_goal_independent(n_obstacles):
    """HER relabels desired_goal only, so `observation` must not depend on the goal (it would go stale)."""
    env = UR5ePoseEnv(EnvConfig(n_obstacles=n_obstacles))
    env.reset(seed=3)
    env.step(env.action_space.sample())
    before = env._obs()["observation"]
    rng = np.random.default_rng(0)
    env.goal = rng.normal(size=9)
    env.goal_R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    np.testing.assert_array_equal(env._obs()["observation"], before)


def test_policy_goal_error_follows_relabeled_goal():
    import torch
    from ur5rl.agents.policies import GoalErrorExtractor

    env = UR5ePoseEnv(EnvConfig())
    obs, _ = env.reset(seed=4)
    ext = GoalErrorExtractor(env.observation_space)
    batch = {k: torch.as_tensor(v)[None] for k, v in obs.items()}
    batch["desired_goal"] = torch.randn(1, 9)  # a relabeled goal
    feats = ext(batch)
    assert feats.shape[1] == ext.features_dim
    np.testing.assert_allclose(feats[0, -9:].numpy(), (batch["desired_goal"] - batch["achieved_goal"])[0].numpy())


def test_buffer_pickles_for_checkpoints():
    _, buf, _ = rollout_buffer(SafetyHerReplayBuffer, n_steps=60)
    buf.env = None
    clone = pickle.loads(pickle.dumps(buf))
    np.testing.assert_array_equal(clone.infos.values, buf.infos.values)


def test_stock_sb3_buffer_fails_loudly_instead_of_dropping_terms():
    """SB3's default (copy_info_dict=False) passes empty infos; compute_reward must refuse, not silently drop."""
    _, buf, _ = rollout_buffer(HerReplayBuffer, n_steps=120)
    valid = np.flatnonzero(buf.ep_length > 0)
    b_idx, e_idx = np.unravel_index(valid[:8], buf.ep_length.shape)
    with pytest.raises(KeyError):
        buf._get_virtual_samples(b_idx, e_idx)
