"""Basic reach on the MyCobot 280 (SPEC M10): a goal-conditioned, position-only reach env on top of MyCobotPoseEnv.

A subclass: the S4 env and its defaults are untouched. Differences from the pose env:
  - goal: TCP position uniform in a cube (`cube_center`, `cube_half`), orientation not part of the task; the goal's rotation
    entries (desired_goal[3:]) hold the home orientation so the 9-D goal layout of the shared agent code is unchanged;
  - start: the fixed home pose (`start="home"`) or home plus uniform noise within `start_noise` rad (`start="noise"`);
  - action: `mode="ee"` -> 3 numbers in [-1, 1] x `ee_step` (m), converted to joint increments by damped least squares on the
    tool-position Jacobian plus a weak null-space pull to the home joints, scaled so that no joint moves more than dq_max
    (0.5 deg) per tick; `mode="joints"` -> 6 joint increments as in the pose env;
  - reward: sparse -1 / 0; the episode ends on success (`eps_pos`); contact (table, self or obstacle) ends it with the
    absorbing reward -1/(1-gamma) (w_c = w_s = w_m = 0), which is kept through HER (shared SafetyHerReplayBuffer);
  - obstacles: `obstacle="mid"` places one sphere (radius `obstacle_radius`) near the start-goal midpoint (+- `mid_jitter` m
    per axis), clear of the start and goal.
Frozen test episodes: dicts with q_start (6), goal (3), centers (K,3), radii (K), mask (K), stored by scripts/make_reach_sets.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from ur5rl.envs import rotations as rot
from mcrl import robot
from mcrl.envs.pose_env import EnvConfig, MyCobotPoseEnv

HOME_Q = robot.HOME_Q


@dataclass
class ReachConfig(EnvConfig):
    mode: str = "ee"  # "ee" | "joints"
    ee_step: float = 0.004  # m per tick per axis at |a| = 1 (the joint cap usually binds first)
    cube_center: tuple = (0.10, -0.06, 0.30)
    cube_half: float = 0.05  # 10 cm cube
    start: str = "home"  # "home" | "noise"
    start_noise: float = 0.3  # rad, as s4_test
    obstacle: str = "none"  # "none" | "mid"
    mid_jitter: float = 0.01
    obstacle_margin: float = 0.01  # sphere clearance from the arm at the start pose (m)
    obstacle_frac: float = 0.5  # sphere centre at this fraction of the start->goal line (0.5 = midpoint)
    min_goal_dist: float = 0.0  # reject goals closer than this to the start TCP (m)
    nullspace_gain: float = 0.05
    ik_damping: float = 0.02
    # position-only sparse reach; the pose env's reward weights that are not wanted here
    orientation: bool = False
    reward: str = "sparse"
    eps_pos: float = 0.02
    n_obstacles: tuple = (1, 1)  # slots exist in every stage so that the observation layout is the same; `obstacle` fills them
    episode_length: int = 100
    w_s: float = 0.0
    w_m: float = 0.0
    w_c: float = 0.0
    gamma: float | None = 0.98


class MyCobotReachEnv(MyCobotPoseEnv):
    def __init__(self, config: ReachConfig | dict | None = None, **kw):
        if isinstance(config, dict):
            config = ReachConfig(**{**config, **({"cube_center": tuple(config["cube_center"])} if "cube_center" in config else {})})
        config = config or ReachConfig()
        super().__init__(config, **kw)
        self.use_obstacles = True  # contact with the table, the arm itself or a sphere always ends the episode
        if config.mode == "ee":
            self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.q_home = HOME_Q.copy()
        _, self.R_hold = self._tcp_pose(HOME_Q)
        self.p_start = np.zeros(3)
        self._jacp = np.zeros((3, self.model.nv))

    # ------------------------------------------------------------------ sampling

    def _cube_goal(self):
        c, h = np.asarray(self.cfg.cube_center), self.cfg.cube_half
        return c + self.np_random.uniform(-h, h, 3)

    def _start(self):
        if self.cfg.start == "home":
            return HOME_Q.copy()
        return self._sample_start()

    def sample_episode(self):
        """One episode dict from the env's RNG (also used to build frozen test sets)."""
        cfg = self.cfg
        for _ in range(500):
            q_start = self._start()
            p0, _ = self._tcp_pose(q_start)
            goal = self._cube_goal()
            if np.linalg.norm(goal - p0) < cfg.min_goal_dist:
                continue
            centers, radii, mask = np.zeros((self.K, 3)), np.full(self.K, 0.05), np.zeros(self.K, bool)
            if cfg.obstacle == "mid":
                ok = False
                for _ in range(100):
                    c = p0 + cfg.obstacle_frac * (goal - p0) + self.np_random.uniform(-cfg.mid_jitter, cfg.mid_jitter, 3)
                    r = self.np_random.uniform(*cfg.obstacle_radius)
                    # clear of the start and of the goal (tool tip) and of the arm at the start pose
                    if np.linalg.norm(c - p0) < r + 0.03 or np.linalg.norm(c - goal) < r + 0.03:
                        continue
                    if not (r < c[2]):
                        continue
                    if not self._clear_at(q_start, c[None], np.array([r]), margin=cfg.obstacle_margin):
                        continue
                    centers[0], radii[0], mask[0] = c, r, True
                    ok = True
                    break
                if not ok:
                    continue
            return dict(q_start=q_start, goal=goal, centers=centers, radii=radii, mask=mask)
        raise RuntimeError("could not sample a reach episode")

    # ------------------------------------------------------------------ gym API

    def episode(self):
        return dict(q_start=self.q_start.copy(), goal=self.goal[:3].copy(), centers=self.ob_center.copy(),
                    radii=self.ob_radius.copy(), mask=self.ob_mask.copy())

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        m, d = self.model, self.data
        ep = (options or {}).get("episode")
        if ep is None:
            ep = self.sample_episode()
        self._set_obstacles(np.asarray(ep["centers"], float), np.asarray(ep["radii"], float), np.asarray(ep["mask"], bool))
        self.q_start = np.asarray(ep["q_start"], float)
        self.q_goal = self.q_start.copy()  # no joint-space goal for a position task
        p_goal = np.asarray(ep["goal"], float)
        self.goal = np.r_[p_goal, rot.mat_to_r6d(self.R_hold)]
        self.goal_R = self.R_hold
        self.p_start, _ = self._tcp_pose(self.q_start)
        mujoco.mj_resetData(m, d)
        d.mocap_pos[self.goal_mocap] = p_goal
        d.mocap_quat[self.goal_mocap] = rot.mat_to_quat(self.R_hold)
        for k in range(self.K):
            d.mocap_pos[self.obs_mocap[k]] = self.ob_center[k] if self.ob_mask[k] else (0.0, 0.0, -5.0)
            d.mocap_quat[self.obs_mocap[k]] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[:6] = self.q_start
        d.ctrl[:6] = self.q_start
        mujoco.mj_forward(m, d)
        self.q_target = d.qpos[:6].copy()
        self.prev_action[:] = 0
        self.t = 0
        self.hold = 0
        return self._obs(), {}

    def ee_to_joint_action(self, dx):
        """Tool displacement dx (3, m) -> joint action in [-1, 1]^6 (one DLS step, null-space pull to home)."""
        m, d = self.model, self.data
        mujoco.mj_jacSite(m, d, self._jacp, None, self.tcp_id)
        J = self._jacp[:, :6]
        lam = self.cfg.ik_damping
        Jp = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(3))
        dq = Jp @ dx + (np.eye(6) - Jp @ J) @ (self.cfg.nullspace_gain * (self.q_home - d.qpos[:6]))
        n = np.abs(dq).max()
        if n > self.cfg.dq_max:
            dq *= self.cfg.dq_max / n
        return dq / self.cfg.dq_max

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        if self.cfg.mode == "ee":
            a = self.ee_to_joint_action(a * self.cfg.ee_step)
        obs, reward, terminated, truncated, info = super().step(a)
        if info["is_success"]:
            terminated = True  # the episode ends on success
        return obs, reward, terminated, truncated, info

    def tcp(self):
        return self.data.site_xpos[self.tcp_id].copy()
