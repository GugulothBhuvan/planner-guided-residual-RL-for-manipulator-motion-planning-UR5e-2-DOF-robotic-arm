"""Build SB3 agents from configs (SPEC §7.1). Shared by the UR5e and the planar 2-DOF arm."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

from ur5rl.agents.her import SafetyHerReplayBuffer
from ur5rl.agents.policies import GoalErrorExtractor
from ur5rl.agents.residual import WarmupSAC, init_near_zero

ALGOS = {"sac": SAC, "td3": TD3, "ddpg": DDPG, "ppo": PPO, "residual_sac": WarmupSAC}


def gradient_steps(agent_cfg, n_envs):
    """SB3 counts train_freq in vec-env steps (n_envs transitions each). Configs give the ratio per transition."""
    ratio = agent_cfg["grad_steps_per_transition"]
    steps = ratio * n_envs
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"grad_steps_per_transition={ratio} x n_envs={n_envs} is not an integer")
    return int(round(steps))


def make_agent(agent_cfg, env, her: bool, seed: int):
    cfg = dict(agent_cfg)
    algo = cfg["algo"]
    policy_kwargs = dict(net_arch=list(cfg["net_arch"]), features_extractor_class=GoalErrorExtractor)
    if algo == "ppo":
        if her:
            raise ValueError("PPO is on-policy; HER does not apply")
        return PPO("MultiInputPolicy", env, learning_rate=cfg["learning_rate"], n_steps=cfg["n_steps"],
                   batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"], gamma=cfg["gamma"],
                   gae_lambda=cfg["gae_lambda"], policy_kwargs=policy_kwargs, device="cpu", seed=seed, verbose=0)

    if algo == "residual_sac" and her:
        raise ValueError("residual agents use the dense reward without HER (SPEC §7.3)")
    n_act = env.action_space.shape[0]
    kw = dict(
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        tau=cfg["tau"],
        gamma=cfg["gamma_her"] if her else cfg["gamma"],
        train_freq=1,
        gradient_steps=gradient_steps(cfg, env.num_envs),
        learning_starts=cfg["learning_starts"],
        policy_kwargs=policy_kwargs,
        device="cpu",
        seed=seed,
        verbose=0,
    )
    if algo in ("td3", "ddpg"):
        kw["action_noise"] = NormalActionNoise(np.zeros(n_act), cfg["action_noise"] * np.ones(n_act))
    if algo == "td3":
        kw["target_policy_noise"] = cfg["target_policy_noise"]
    if algo == "residual_sac":
        # §7.3: dense reward, no HER (the reference is planned for the original goal; rejected above).
        kw.update(ent_coef=cfg["ent_coef"], target_entropy=cfg["target_entropy_per_dim"] * n_act,
                  critic_warmup_steps=cfg["critic_warmup_steps"])
        model = WarmupSAC("MultiInputPolicy", env, **kw)
        init_near_zero(model, log_std_bias=cfg["log_std_bias"], seed=seed)
        return model
    if her:
        # Keeps the goal-independent safety terms through relabeling (§6.7) at 8 B per transition.
        kw.update(replay_buffer_class=SafetyHerReplayBuffer,
                  replay_buffer_kwargs=dict(n_sampled_goal=cfg["her"]["n_sampled_goal"],
                                            goal_selection_strategy=cfg["her"]["goal_selection_strategy"]))
    return ALGOS[algo]("MultiInputPolicy", env, **kw)
