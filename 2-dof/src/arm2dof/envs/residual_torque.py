"""P5 residual (companion SPEC §7.4): tau = tau_B5'(nominal model) + alpha * tau_max * pi_res(s).

Same formulation as the kinematic residual (ResidualArmEnv) with the PD + gravity controller as the base: the base
is built on the nominal model, so on a perturbed plant (mass_factor != 1) its gravity term is wrong and the residual
has something to correct. Observation adds q_ref(t), q_ref(t+5) as (sin q1, cos q1, q2 / pi) (6 values).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from arm2dof.control.controllers import Tracker


class ResidualTorqueEnv(gym.Wrapper):
    def __init__(self, env, alpha=0.0, lookahead=5, w_res=0.0, kind="pdg"):
        super().__init__(env)
        self.alpha, self.lookahead, self.w_res, self.kind = alpha, lookahead, w_res, kind
        spaces = dict(env.observation_space.spaces)
        spaces["observation"] = gym.spaces.Box(-np.inf, np.inf, (spaces["observation"].shape[0] + 6,), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.tracker = None

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)

    @property
    def reference(self):
        return self.tracker.ref[0]

    def _augment(self, obs):
        ref, t = self.tracker.ref[0], self.tracker.t
        feats = []
        for q in (ref[min(t, len(ref) - 1)], ref[min(t + self.lookahead, len(ref) - 1)]):
            feats += [np.sin(q[0]), np.cos(q[0]), q[1] / np.pi]
        obs = dict(obs)
        obs["observation"] = np.concatenate([obs["observation"], feats]).astype(np.float32)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.tracker = Tracker(self.env.unwrapped, self.kind)
        return self._augment(obs), info

    def step(self, action):
        a_res = np.clip(np.asarray(action, float), -1.0, 1.0)
        a_base = self.tracker()
        obs, reward, terminated, truncated, info = self.env.step(np.clip(a_base + self.alpha * a_res, -1.0, 1.0))
        reward -= self.w_res * float((a_res ** 2).sum())
        info = dict(info, a_base=a_base, residual_norm=float(np.linalg.norm(self.alpha * a_res)))
        return self._augment(obs), reward, terminated, truncated, info
