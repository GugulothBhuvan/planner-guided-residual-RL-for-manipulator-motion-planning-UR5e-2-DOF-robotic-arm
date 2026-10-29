"""Residual planar arm (companion SPEC §7.4): a = clip(a_base + alpha * a_res), base = degraded B2 tracker.

Same formulation as the UR5e ResidualEnv (UR5e SPEC §7.3). Observation adds q_ref(t), q_ref(t+5) as
(sin, cos) of q1 and q2 / pi (6 values): q1 wraps, so it is encoded on the circle.

mode="offset" (PE4b, companion SPEC §8; the UR5e E5b design): the residual integrates a persistent joint offset,
o <- clip(o + alpha * dq_max * a_res, +-offset_max), and the base tracks q_ref(t+1) + s * o with
s = clip(d_ee / offset_fade, 0, 1), so the offset fades to 0 at the goal. The observation adds o / offset_max (2).
With a_res = 0 the env is the base exactly. v2 (as UR5e E5b v2): offset_fade_ref > 0 also fades the offset
linearly over the last offset_fade_ref of the reference (s = min of the two fades), and penalty="offset" charges
w_res * |o / offset_max|^2 per step instead of w_res * |a_res|^2.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from arm2dof.planners.base import Tracker, plan
from arm2dof.planners.ik import fk


class ResidualArmEnv(gym.Wrapper):
    def __init__(self, env, inflate=0.05, max_checks=1500, shortcut_iters=20, alpha=0.0, lookahead=5, w_res=0.0,
                 mode="additive", offset_max=0.3, offset_fade=0.05, offset_fade_ref=0.0, penalty="action"):
        super().__init__(env)
        if mode not in ("additive", "offset") or penalty not in ("action", "offset"):
            raise ValueError((mode, penalty))
        self.offset_fade_ref, self.penalty = offset_fade_ref, penalty
        self.mode, self.offset_max, self.offset_fade = mode, offset_max, offset_fade
        self.offset = np.zeros(2)
        self.w_res = w_res  # L2 penalty on the residual, r -= w_res * |a_res|^2 (as the UR5e)
        self.inflate, self.max_checks, self.shortcut_iters = inflate, max_checks, shortcut_iters
        self.alpha, self.lookahead = alpha, lookahead
        spaces = dict(env.observation_space.spaces)
        n = spaces["observation"].shape[0] + 6 + (2 if mode == "offset" else 0)
        spaces["observation"] = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.reference, self.tracker, self.base_ok, self.budget = None, None, False, None

    def _draw_budget(self, rng):
        """max_checks: int (one fixed budget) or list (one drawn uniformly per episode), as on the UR5e."""
        budgets = np.atleast_1d(self.max_checks)
        return int(budgets[0]) if len(budgets) == 1 else int(budgets[rng.integers(len(budgets))])

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)

    def _augment(self, obs):
        ref, t = self.reference, self.tracker.t
        feats = []
        for q in (ref[min(t, len(ref) - 1)], ref[min(t + self.lookahead, len(ref) - 1)]):
            feats += [np.sin(q[0]), np.cos(q[0]), q[1] / np.pi]
        if self.mode == "offset":
            feats += list(self.offset / self.offset_max)
        obs = dict(obs)
        obs["observation"] = np.concatenate([obs["observation"], feats]).astype(np.float32)
        return obs

    def _plan(self):
        base = self.env.unwrapped
        rng = np.random.default_rng(base.np_random.integers(2**31))
        p = plan(base, rng, inflate=self.inflate, max_checks=self.budget, shortcut_iters=self.shortcut_iters)
        self.base_ok, self.reference = p.ok, p.reference
        self.tracker = Tracker(base, self.reference)
        self.offset = np.zeros(2)

    def offset_scale(self):
        """s in [0, 1]: 1 farther than offset_fade from the goal, 0 at the goal."""
        base = self.env.unwrapped
        s = min(1.0, float(np.linalg.norm(base.goal - fk(base.q))) / self.offset_fade)
        if self.offset_fade_ref > 0:
            n = len(self.reference)
            s = min(s, max(0.0, (n - 1 - min(self.tracker.t, n - 1)) / (self.offset_fade_ref * n)))
        return s

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.budget = self._draw_budget(self.env.unwrapped.np_random)
        self._plan()
        return self._augment(obs), dict(info, base_ok=self.base_ok, base_budget=self.budget)

    def replan(self):
        """Re-plan the base from the current q after the goal or obstacles changed (interactive viewer)."""
        base = self.env.unwrapped
        base.q_start = base.q.copy()
        self._plan()
        return self._augment(base._obs())

    def step(self, action):
        a_res = np.clip(np.asarray(action, float), -1.0, 1.0)
        a_base = self.tracker()
        if self.mode == "offset":
            base = self.env.unwrapped
            self.offset = np.clip(self.offset + self.alpha * base.cfg.dq_max * a_res, -self.offset_max, self.offset_max)
            k = min(self.tracker.t, len(self.reference) - 1)  # the point the tracker just aimed at
            target = self.reference[k] + self.offset_scale() * self.offset
            a = np.clip(base.qdiff(base.q, target) / base.cfg.dq_max, -1.0, 1.0)
        else:
            a = np.clip(a_base + self.alpha * a_res, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(a)
        info = dict(info, base_ok=self.base_ok, a_base=a_base,
                    residual_norm=float(np.linalg.norm(self.alpha * a_res)), offset_norm=float(np.abs(self.offset).max()))
        if self.w_res:
            gamma = self.env.unwrapped.cfg.gamma
            if info["collision"] and gamma is not None:
                # keep the absorbing bound: the worst per-step penalty (2 x w_res) forever, not this step's
                reward -= self.w_res * 2 / (1.0 - gamma)
            elif self.penalty == "offset":
                reward -= self.w_res * float(((self.offset / self.offset_max) ** 2).sum())
            else:
                reward -= self.w_res * float((a_res ** 2).sum())
        return self._augment(obs), reward, terminated, truncated, info
