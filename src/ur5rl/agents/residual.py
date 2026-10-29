"""Residual SAC (SPEC §7.3): critic warm-up, near-zero start in mean and noise, alpha ramp.

The residual policy acts through ResidualEnv (envs/residual.py), which executes clip(a_base + alpha * a_res).
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


class WarmupSAC(SAC):
    """SAC whose actor and entropy coefficient are frozen for the first `critic_warmup_steps` gradient steps.

    Implemented by zeroing their learning rates after SB3 sets the schedule each train() call: with lr = 0 Adam
    leaves the parameters exactly unchanged, while the critic and its targets train normally.
    """

    def __init__(self, *args, critic_warmup_steps: int = 20_000, **kwargs):
        self.critic_warmup_steps = critic_warmup_steps
        super().__init__(*args, **kwargs)

    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate(optimizers)
        if self._n_updates < self.critic_warmup_steps:
            frozen = [self.actor.optimizer] + ([self.ent_coef_optimizer] if self.ent_coef_optimizer else [])
            for opt in frozen:
                for group in opt.param_groups:
                    group["lr"] = 0.0
        self.logger.record("train/critic_warmup", float(self._n_updates < self.critic_warmup_steps))


def init_near_zero(model, log_std_bias=-3.0, mean_std=1e-3, seed=0):
    """Mean head ~ 0 and log-std = log_std_bias (sigma ~ 0.05), so sampled residuals start near zero (§7.3)."""
    gen = th.Generator().manual_seed(seed)
    with th.no_grad():
        mu, log_std = model.actor.mu, model.actor.log_std
        mu.weight.copy_(th.randn(mu.weight.shape, generator=gen) * mean_std)
        mu.bias.zero_()
        log_std.weight.zero_()
        log_std.bias.fill_(log_std_bias)


class AlphaRamp(BaseCallback):
    """alpha = alpha_max * min(1, t / ramp_steps), pushed to every env; also stored on the model for evaluation."""

    def __init__(self, alpha_max=0.3, ramp_steps=50_000):
        super().__init__()
        self.alpha_max, self.ramp_steps = alpha_max, ramp_steps

    def alpha(self):
        return self.alpha_max * min(1.0, self.num_timesteps / self.ramp_steps)

    def _on_training_start(self):
        self._push()

    def _on_step(self):
        if self.n_calls % 50 == 0:
            self._push()
        return True

    def _push(self):
        a = self.alpha()
        self.training_env.env_method("set_alpha", a)
        self.model.residual_alpha = a
        self.logger.record("residual/alpha", a)


def residual_magnitude(model, obs_batch, alpha):
    """Mean |alpha * a_res| of stochastic policy samples on a batch of observations (SPEC §13 init check)."""
    actions, _ = model.predict(obs_batch, deterministic=False)
    return float(np.abs(alpha * actions).mean())
