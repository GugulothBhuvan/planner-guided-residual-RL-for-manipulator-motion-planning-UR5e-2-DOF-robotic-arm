"""SPEC §13 residual mode: no HER, per-episode reference, zero residual = base, near-zero init, critic warm-up."""

import numpy as np
import pytest
import torch as th
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv

from ur5rl.agents.factory import make_agent
from ur5rl.agents.residual import residual_magnitude
from ur5rl.envs import rotations as rot
from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv
from ur5rl.envs.residual import ResidualEnv
from ur5rl.planners.baselines import ReferenceTracker, flange_to_tcp
from ur5rl.planners.ik import fk_tcp

CFG = EnvConfig(n_obstacles=(1, 4), episode_length=300, reward="dense")
AGENT = yaml.safe_load(open("configs/agent/residual.yaml"))


def make_venv():
    return DummyVecEnv([lambda: ResidualEnv(UR5ePoseEnv(CFG))] * 4)  # 0.25 grad/transition needs 4 envs


def test_residual_with_her_is_rejected():
    with pytest.raises(ValueError, match="without HER"):
        make_agent(AGENT, make_venv(), her=True, seed=0)


def test_reference_is_planned_for_each_episodes_goal():
    env = ResidualEnv(UR5ePoseEnv(CFG))
    T_ft = flange_to_tcp(env.unwrapped)
    budgets = (6000, 1)  # 1 check forces the straight-line fallback: it must end at the goal too
    for max_checks in budgets:
        env.max_checks = max_checks
        for seed in range(4):
            _, info = env.reset(seed=90_000 + seed)
            if max_checks == 1:
                assert not info["base_ok"]
            T = fk_tcp(env.reference[-1], T_ft)
            np.testing.assert_allclose(T[:3, 3], env.unwrapped.goal[:3], atol=1e-6)
            assert rot.geodesic_angle(T[:3, :3], env.unwrapped.goal_R) < 1e-6


def test_fallback_is_straight_line_to_ik_goal():
    env = ResidualEnv(UR5ePoseEnv(CFG), max_checks=1)  # a 1-check budget always fails -> fallback
    _, info = env.reset(seed=92_000)
    assert not info["base_ok"]
    ref = env.reference
    direction = ref[-1] - ref[0]
    for q in ref[1:-1]:  # every point lies on the segment start -> goal
        t = (q - ref[0]) @ direction / (direction @ direction)
        np.testing.assert_allclose(q, ref[0] + t * direction, atol=1e-9)


def test_zero_residual_reproduces_base_exactly():
    res = ResidualEnv(UR5ePoseEnv(CFG), alpha=0.3)
    plain = UR5ePoseEnv(CFG)
    res.reset(seed=91_000)
    plain.reset(seed=91_000)
    tracker = ReferenceTracker(plain, res.reference)
    for _ in range(120):
        _, r1, t1, tr1, info = res.step(np.zeros(6))
        a_base = tracker()
        _, r2, t2, tr2, _ = plain.step(a_base)
        np.testing.assert_array_equal(info["a_base"], a_base)
        assert r1 == r2 and t1 == t2 and tr1 == tr2
        if t1 or tr1:
            break


def test_initial_residual_is_near_zero():
    venv = make_venv()
    model = make_agent(AGENT, venv, her=False, seed=0)
    obs = venv.reset()
    batch = {k: np.repeat(v, 1000, axis=0) for k, v in obs.items()}
    assert residual_magnitude(model, batch, AGENT["alpha_max"]) < 0.02


def test_warmup_freezes_actor_not_critic():
    agent = dict(AGENT, learning_starts=100, critic_warmup_steps=40)
    model = make_agent(agent, make_venv(), her=False, seed=0)
    model.learn(300)  # fills the buffer past learning_starts
    actor0 = [p.detach().clone() for p in model.actor.parameters()]
    critic0 = [p.detach().clone() for p in model.critic.parameters()]
    model._n_updates = 0
    model.train(gradient_steps=10, batch_size=64)  # inside warm-up
    assert all(th.equal(a, b) for a, b in zip(actor0, model.actor.parameters()))
    assert not all(th.equal(a, b) for a, b in zip(critic0, model.critic.parameters()))
    model._n_updates = 40
    model.train(gradient_steps=5, batch_size=64)  # after warm-up
    assert not all(th.equal(a, b) for a, b in zip(actor0, model.actor.parameters()))


def test_training_budget_is_drawn_per_episode():
    """§7.3: training draws the base budget uniformly from {1500, 3000, 6000}; same seed -> same budget."""
    assert AGENT["base"]["max_checks"] == [1500, 3000, 6000]
    env = ResidualEnv(UR5ePoseEnv(CFG), **AGENT["base"])
    drawn = []
    for seed in range(12):
        _, info = env.reset(seed=95_000 + seed)
        assert info["base_budget"] == env.budget
        drawn.append(env.budget)
    assert set(drawn) == {1500, 3000, 6000}
    again = [env.reset(seed=95_000 + s)[1]["base_budget"] for s in range(12)]
    assert again == drawn


def test_fixed_budget_draws_nothing():
    """Evaluation passes one int budget: it is used as is and the env RNG stream is untouched."""
    a, b = ResidualEnv(UR5ePoseEnv(CFG), max_checks=3000), UR5ePoseEnv(CFG)
    _, info = a.reset(seed=96_000)
    assert info["base_budget"] == 3000
    b.reset(seed=96_000)
    b.np_random.integers(2**31)  # the base-plan seed is the only extra draw a fixed-budget reset makes
    assert a.unwrapped.np_random.integers(2**31) == b.np_random.integers(2**31)


def test_residual_l2_penalty_and_absorbing_bound():
    """§7.3 rule: r -= w_res |a_res|^2; on a collision the worst-case penalty is charged forever (absorbing)."""
    cfg = EnvConfig(n_obstacles=(1, 4), episode_length=300, reward="dense", gamma=0.99)
    a, b = ResidualEnv(UR5ePoseEnv(cfg), alpha=0.3, w_res=0.1), ResidualEnv(UR5ePoseEnv(cfg), alpha=0.3)
    a.reset(seed=91_000)
    b.reset(seed=91_000)
    act = np.full(6, 0.5)
    _, ra, *_ , ia = a.step(act)
    _, rb, *_ = b.step(act)
    assert not ia["collision"] and ra == pytest.approx(rb - 0.1 * 6 * 0.25)
    env = a.unwrapped
    tcp = env.data.site_xpos[env.tcp_id].copy()
    env._set_obstacles(np.r_[[tcp], np.zeros((env.K - 1, 3))], np.full(env.K, 0.06), np.r_[True, np.zeros(env.K - 1, bool)])
    _, r, _, _, info = a.step(np.zeros(6))
    assert info["collision"] and r == pytest.approx(env.collision_reward() - 0.1 * 6 / (1 - 0.99))
