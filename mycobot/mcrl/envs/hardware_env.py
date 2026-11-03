"""MyCobot env whose physics step is the real (or fake) arm: the sim only provides kinematics, virtual obstacles and
observations. The target update is the sim env's (integrated target, lead cap); the state comes from the arm's reading.
"""
from __future__ import annotations

import mujoco
import numpy as np

from mcrl.envs.pose_env import MyCobotPoseEnv


class HardwareEnv(MyCobotPoseEnv):
    def __init__(self, config, arm):
        super().__init__(config)
        self.arm = arm
        self.q_prev = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._sync(self.arm.q, np.zeros(6))
        self.q_target = self.arm.q.copy()
        self.q_prev = self.arm.q.copy()
        return self._obs(), info

    def _sync(self, q, qd):
        d = self.data
        d.qpos[:6], d.qvel[:6] = q, qd
        mujoco.mj_forward(self.model, d)  # kinematics + contacts against the virtual obstacles and the table

    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        q = self.data.qpos[:6].copy()
        target = np.clip(self.q_target + a * cfg.dq_max, self.q_lo, self.q_hi)
        self.q_target = q + np.clip(target - q, -cfg.target_lead_max, cfg.target_lead_max)
        q_new = self.arm.tick(self.q_target)
        dt = cfg.frame_skip * cfg.timestep
        self._sync(q_new, (q_new - q) / dt)
        collided = bool(self.data.ncon > 0)
        self.t += 1
        d_pos, theta = self._errors(self.achieved_goal, self.goal[None])
        success = bool(self._success(d_pos, theta)[0])
        self.hold = self.hold + 1 if success else 0
        obs = self._obs()
        info = dict(r_indep=0.0, is_success=success, collision=collided, curriculum_goal=False, d_pos=float(d_pos[0]),
                    theta=float(theta[0]),
                    min_clearance=float(self._body_min.min()) if self.use_obstacles else np.inf,
                    q_measured=q_new.copy(), q_target=self.q_target.copy())
        self.prev_action = a
        terminated = collided or self.hold >= 5  # stop on a virtual collision, or after 5 successful ticks
        truncated = self.t >= cfg.episode_length
        return obs, 0.0, terminated, truncated, info
