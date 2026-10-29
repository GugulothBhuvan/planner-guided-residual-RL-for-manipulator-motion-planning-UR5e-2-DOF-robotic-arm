"""Residual env (SPEC §7.3): the policy outputs a residual on top of a degraded-B2 tracking controller.

Executed action: a = clip(a_base + alpha * a_res, -1, 1), with a_base = clip((q_ref(t+1) - q_target) / dq_max).
The base reference is planned at every reset for that episode's goal, with obstacles inflated by `inflate`, a
budget of `max_checks` collision checks and few shortcut iterations (deliberately degraded). `max_checks` is an
int (evaluation: one fixed budget) or a list (training: one budget drawn uniformly per episode from the env's RNG,
§7.3; info["base_budget"] records it). If the base cannot
plan within its budget, the reference is a straight joint-space line to the nearest IK goal that ignores
collisions; the residual has to handle it (info["base_ok"] = False).
Observation = the wrapped env's observation + q_ref(t) and q_ref(t+5) (12, scaled by 1/pi). Residual agents use
the dense reward without HER (§7.3), so these goal-specific features are never relabeled.

mode="offset" (E5b, SPEC §7.3): the base re-aims q_target at q_ref every step and so cancels an additive residual,
bounding its authority to ~alpha * dq_max around q_ref (E5 diagnostic). Here the residual instead integrates a
joint offset that persists, o <- clip(o + alpha * dq_max * a_res, +-offset_max) (offset_step="dq_max": at most
alpha * dq_max = 0.012 rad per step, +-0.3 rad in ~25 steps; "rad", alpha * a_res per step, was E5b v0, archived),
and the base tracks the shifted reference
q_ref(t+1) + s * o, with s = clip(d_pos / offset_fade, 0, 1) fading the offset to 0 at the goal. The observation
adds o / offset_max (6). With a_res = 0 the offset stays 0 and the env is the base exactly.
E5b v2 (2026-10-29): offset_fade_ref > 0 also fades the offset linearly over the last offset_fade_ref of the
reference, s = min(s_dist, s_ref), so the target returns to the true reference by its end even when the tool is far
from the goal (v1's distance-only fade never engaged: a saturated offset kept the tool > 5 cm away). penalty="offset"
charges w_res * |o / offset_max|^2 per step instead of w_res * |a_res|^2.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from ur5rl.planners.baselines import ReferenceTracker, goal_configs, plan_b2
from ur5rl.planners.collision import CollisionChecker
from ur5rl.planners.quintic import time_scale


class ResidualEnv(gym.Wrapper):
    def __init__(self, env, inflate=0.05, max_checks=6000, shortcut_iters=20, alpha=0.0, lookahead=5, w_res=0.0,
                 mode="additive", offset_max=0.3, offset_fade=0.05, offset_step="dq_max", offset_fade_ref=0.0,
                 penalty="action"):
        super().__init__(env)
        if mode not in ("additive", "offset") or offset_step not in ("dq_max", "rad") or penalty not in ("action", "offset"):
            raise ValueError((mode, offset_step, penalty))
        self.offset_step, self.offset_fade_ref, self.penalty = offset_step, offset_fade_ref, penalty
        self.mode, self.offset_max, self.offset_fade = mode, offset_max, offset_fade
        self.offset = np.zeros(6)
        self.w_res = w_res  # L2 penalty on the residual, r -= w_res * |a_res|^2 (§7.3 rule)
        self.inflate, self.max_checks, self.shortcut_iters = inflate, max_checks, shortcut_iters
        self.alpha, self.lookahead = alpha, lookahead
        self.checker, self.inflated_checker = CollisionChecker(), CollisionChecker()
        spaces = dict(env.observation_space.spaces)
        n = spaces["observation"].shape[0] + 12 + (6 if mode == "offset" else 0)
        spaces["observation"] = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.reference = None
        self.tracker = None
        self.base_ok = False
        self.budget = None

    def _draw_budget(self, rng):
        budgets = np.atleast_1d(self.max_checks)
        return int(budgets[0]) if len(budgets) == 1 else int(budgets[rng.integers(len(budgets))])

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)

    def _augment(self, obs):
        t = self.tracker.t
        ref = self.reference
        q_now, q_ahead = ref[min(t, len(ref) - 1)], ref[min(t + self.lookahead, len(ref) - 1)]
        obs = dict(obs)
        parts = [obs["observation"], q_now / np.pi, q_ahead / np.pi]
        if self.mode == "offset":
            parts.append(self.offset / self.offset_max)
        obs["observation"] = np.concatenate(parts).astype(np.float32)
        return obs

    def offset_scale(self):
        """s in [0, 1]: 1 farther than offset_fade from the goal position, 0 at the goal; with offset_fade_ref also
        falling linearly to 0 over the last offset_fade_ref of the reference (s = min of the two)."""
        base = self.env.unwrapped
        d = float(np.linalg.norm(base.data.site_xpos[base.tcp_id] - base.goal[:3]))
        s = min(1.0, d / self.offset_fade)
        if self.offset_fade_ref > 0:
            n = len(self.reference)
            left = n - 1 - min(self.tracker.t, n - 1)  # reference steps after the point the tracker aims at
            s = min(s, max(0.0, left / (self.offset_fade_ref * n)))
        return s

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        self.budget = self._draw_budget(base.np_random)  # a fixed int budget draws nothing
        rng = np.random.default_rng(base.np_random.integers(2**31))  # base plan reproducible per env seed
        plan = plan_b2(base, rng, self.checker, inflate=self.inflate, max_checks=self.budget,
                       shortcut_iters=self.shortcut_iters, inflated_checker=self.inflated_checker)
        self.base_ok = plan.ok
        if plan.ok:
            self.reference = plan.reference
        else:  # SPEC §7.3 fallback: straight joint-space line to the IK goal, collisions ignored
            goals = goal_configs(base, self.checker)
            q_goal = goals[0] if goals else base.q_goal
            self.reference, _ = time_scale(np.array([base.q_start, q_goal]),
                                           dt=base.cfg.frame_skip * base.cfg.timestep)
        self.tracker = ReferenceTracker(base, self.reference)
        self.offset = np.zeros(6)
        info = dict(info, base_ok=self.base_ok, base_plan_time=plan.plan_time, base_budget=self.budget)
        return self._augment(obs), info

    def step(self, action):
        a_res = np.clip(np.asarray(action, float), -1.0, 1.0)
        a_base = self.tracker()
        if self.mode == "offset":
            base = self.env.unwrapped
            step = self.alpha * a_res * (base.cfg.dq_max if self.offset_step == "dq_max" else 1.0)
            self.offset = np.clip(self.offset + step, -self.offset_max, self.offset_max)
            k = min(self.tracker.t, len(self.reference) - 1)  # the point the tracker just aimed at
            target = self.reference[k] + self.offset_scale() * self.offset
            a = np.clip((target - base.q_target) / base.cfg.dq_max, -1.0, 1.0)
        else:
            a = np.clip(a_base + self.alpha * a_res, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(a)
        info = dict(info, base_ok=self.base_ok, residual_norm=float(np.linalg.norm(self.alpha * a_res)),
                    a_base=a_base, offset_norm=float(np.abs(self.offset).max()))
        if self.w_res:
            gamma = self.env.unwrapped.cfg.gamma
            if info["collision"] and gamma is not None:
                # keep the absorbing bound: the worst per-step penalty (6 x w_res) forever, not this step's
                reward -= self.w_res * 6 / (1.0 - gamma)
            elif self.penalty == "offset":
                reward -= self.w_res * float(((self.offset / self.offset_max) ** 2).sum())
            else:
                reward -= self.w_res * float((a_res ** 2).sum())
        return self._augment(obs), reward, terminated, truncated, info
