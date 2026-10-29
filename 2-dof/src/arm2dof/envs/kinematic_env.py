"""Planar 2-DOF kinematic goal env, P1-P3 (SPEC §6). Pure NumPy, no physics.

Same contracts as the UR5e env so the shared agent code (factory, SafetyHerReplayBuffer, GoalErrorExtractor)
works unchanged: `observation` is goal-independent, the goal-independent reward term travels in info["r_indep"],
and compute_reward accepts a dict, a list of dicts or a float array of r_indep.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from arm2dof.cspace.components import components
from arm2dof.cspace.grid import cell_of, grid
from arm2dof.envs.geometry import in_collision, link_clearance
from arm2dof.planners.ik import Q_HI, Q_LO, fk, ik


@dataclass
class EnvConfig:
    n_obstacles: tuple[int, int] = (0, 0)  # (min, max) circles per episode; P1 = (0, 0)
    fixed_layout: tuple = ()  # P2: ((x, y, r), ...) used every episode instead of random circles
    obstacle_radius: tuple[float, float] = (0.05, 0.15)
    episode_length: int = 50
    reward: str = "sparse"  # "sparse" | "dense"
    eps: float = 0.01  # m, final tolerance; set_tolerance() changes the current one (curriculum)
    # q1 wraps at +-pi (C-space is a cylinder, §7.2). Off by default only for the archived PE1 (q1 clipped);
    # every stage config (P1-P3) turns it on.
    wrap_q1: bool = False
    tol_curriculum: bool = False  # §6.5: eps 5 cm -> eps in 1 cm steps on rolling success > 80% (training only)
    # Goal-distance curriculum, same rule as the UR5e (UR5e SPEC §6.5): goal configs within a per-joint bound of the
    # start (wrap-aware in q1), raised by the GoalCurriculum callback; curriculum_full_frac of goals are full-range.
    curriculum: bool = False
    curriculum_start: float = 0.6
    curriculum_full_frac: float = 0.2
    curriculum_step: float = 0.3
    goal_max_joint_dist: float = 2 * np.pi  # curriculum cap = full range (no joint distance exceeds 2 pi)
    # §6.4: with obstacles, reject scenes where no IK branch of the goal lies in the start's free C-space component
    # (exact 1-degree grid + flood fill, the same check as the test-set filter), so training never sees impossible goals
    require_connected: bool = True
    tol_start: float = 0.05
    tol_step: float = 0.01
    curriculum_threshold: float = 0.8
    curriculum_window: int = 100
    success_hold: int = 5
    dq_max: float = 0.05  # rad / step
    dt: float = 0.04
    K: int = 4
    w_p: float = 1.0
    w_a: float = 0.01
    w_s: float = 0.05
    w_m: float = 5.0
    d_safe: float = 0.05
    w_c: float = 10.0
    r_succ: float = 5.0
    # Absorbing collision penalty, as on the UR5e (UR5e SPEC §6.7): with gamma set, a collision step's reward is
    # r_min / (1 - gamma) - w_c, so crashing never beats timing out. None = the v1 one-off -w_c
    # (archived pe3_v1_crash_exploit). train.py sets gamma from the agent on obstacle stages.
    gamma: float | None = None


class KinematicArmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | dict | None = None):
        if isinstance(config, dict):
            config = EnvConfig(**config)
        self.cfg = cfg = config or EnvConfig()
        self.use_obstacles = cfg.n_obstacles[1] > 0 or len(cfg.fixed_layout) > 0
        # q / q_limit only for joints with a limit: a wrapping q1 is continuous on the circle but q1 / pi jumps
        # from +1 to -1 at the seam, so it is left out and sin/cos q1 carry it (§6.3)
        self.limited = np.array([not cfg.wrap_q1, True])
        n_obs = 4 + int(self.limited.sum()) + 2 + 2 + 2 + 2 + ((4 * cfg.K + 2) if self.use_obstacles else 0)
        box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.observation_space = gym.spaces.Dict(observation=box(n_obs), achieved_goal=box(2), desired_goal=box(2))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.eps = cfg.eps
        self.centers = np.zeros((cfg.K, 2))
        self.radii = np.zeros(cfg.K)
        self.mask = np.zeros(cfg.K, bool)
        self.q = np.zeros(2)
        self.q_prev = np.zeros(2)
        self.goal = np.zeros(2)
        self.prev_action = np.zeros(2)
        self.t = self.hold = 0
        self._labels_key, self._labels = None, None
        self.curriculum_level = cfg.curriculum_start
        self.curriculum_goal = False  # this episode's goal came from the curriculum level (not full range)

    def set_curriculum_level(self, level):
        """Current per-joint goal bound (rad) while the curriculum is on; called through VecEnv.env_method."""
        self.curriculum_level = float(level)

    def set_tolerance(self, eps):
        self.eps = float(eps)

    @property
    def q_target(self):
        """The kinematic arm is its own target (q_target = q), so UR5e-style trackers work unchanged."""
        return self.q

    def qdiff(self, a, b):
        """b - a, the short way round in q1 when it wraps."""
        d = np.asarray(b, float) - np.asarray(a, float)
        if self.cfg.wrap_q1:
            d[..., 0] = (d[..., 0] + np.pi) % (2 * np.pi) - np.pi
        return d

    # ---------------------------------------------------------------- sampling (§6.4)

    def _collides(self, q):
        return bool(in_collision(q, self.centers[self.mask], self.radii[self.mask]))

    def _sample_q(self, tries=1000):
        """Collision-free q, or None if `tries` samples all collide (the scene is then resampled)."""
        for _ in range(tries):
            q = self.np_random.uniform(Q_LO, Q_HI)
            if not self._collides(q):
                return q
        return None

    def _sample_goal_q(self, q_start, tries=1000):
        """Goal config: full range (as _sample_q) unless the curriculum bounds max_j |q_goal - q_start| (wrap-aware
        in q1). The bounded set is a box (an arc in q1), sampled directly as on the UR5e."""
        cfg = self.cfg
        self.curriculum_goal = cfg.curriculum and self.np_random.random() >= cfg.curriculum_full_frac
        if not self.curriculum_goal:
            return self._sample_q(tries)
        b = min(self.curriculum_level, cfg.goal_max_joint_dist)
        lo, hi = q_start - b, q_start + b
        if not cfg.wrap_q1:
            lo[0], hi[0] = max(lo[0], Q_LO[0]), min(hi[0], Q_HI[0])
        lo[1], hi[1] = max(lo[1], Q_LO[1]), min(hi[1], Q_HI[1])
        for _ in range(tries):
            q = self.np_random.uniform(lo, hi)
            if cfg.wrap_q1:
                q[0] = (q[0] + np.pi) % (2 * np.pi) - np.pi
            if not self._collides(q):
                return q
        return None

    def _component_labels(self):
        key = (self.centers[self.mask].tobytes(), self.radii[self.mask].tobytes())
        if key != self._labels_key:  # fixed layouts build the grid once
            self._labels_key = key
            self._labels = components(grid(self.centers[self.mask], self.radii[self.mask]))
        return self._labels

    def _connected(self, q_start):
        """Some IK branch of self.goal shares q_start's free component (wrap-aware grid)."""
        labels = self._component_labels()
        a = labels[cell_of(q_start)]
        return bool(a) and any(labels[cell_of(q)] == a for q in ik(self.goal) if not self._collides(q))

    def _sample_obstacles(self):
        cfg = self.cfg
        self.mask[:] = False
        if cfg.fixed_layout:
            for k, (x, y, r) in enumerate(cfg.fixed_layout):
                self.centers[k], self.radii[k], self.mask[k] = (x, y), r, True
            return
        n = self.np_random.integers(cfg.n_obstacles[0], cfg.n_obstacles[1] + 1)
        for k in range(n):
            r = self.np_random.uniform(*cfg.obstacle_radius)
            # keep the base clear: a circle over the origin would make every configuration collide
            rho = self.np_random.uniform(r + 0.1, 0.9)
            phi = self.np_random.uniform(-np.pi, np.pi)
            self.centers[k] = rho * np.cos(phi), rho * np.sin(phi)
            self.radii[k] = r
            self.mask[k] = True

    # ---------------------------------------------------------------- gym API

    def episode(self):
        """The current episode as a dict that reset(options={"episode": ...}) reproduces (test sets, §9.2)."""
        return dict(q_start=self.q_start.copy(), q_goal=self.q_goal.copy(), centers=self.centers.copy(),
                    radii=self.radii.copy(), mask=self.mask.copy())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.cfg
        ep = (options or {}).get("episode")
        if ep is not None:  # replay a stored episode (frozen test sets); the seeded RNG still drives the rest
            self.centers[:], self.radii[:] = np.asarray(ep["centers"], float), np.asarray(ep["radii"], float)
            self.mask[:] = np.asarray(ep["mask"], bool)
            q = np.asarray(ep["q_start"], float)
            self.q, self.q_start, self.q_goal = q.copy(), q.copy(), np.asarray(ep["q_goal"], float).copy()
            self.goal = fk(self.q_goal)
            self.curriculum_goal = False
            self.q_prev = self.q.copy()
            self.prev_action[:] = 0
            self.t = self.hold = 0
            return self._obs(), {}
        # §6.4 scene validity: with obstacles, start and goal must share a free C-space component (require_connected)
        while True:
            self._sample_obstacles()
            q = self._sample_q()
            q_goal = self._sample_goal_q(q) if q is not None else None
            if q_goal is None:
                continue
            self.goal = fk(q_goal)
            if not (self.use_obstacles and cfg.require_connected and self.mask.any()) or self._connected(q):
                break
        self.q, self.q_start, self.q_goal = q, q.copy(), q_goal
        self.q_prev = self.q.copy()
        self.prev_action[:] = 0
        self.t = self.hold = 0
        return self._obs(), {}

    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        self.q_prev = self.q.copy()
        q = self.q + a * cfg.dq_max
        if cfg.wrap_q1:
            q[0] = (q[0] + np.pi) % (2 * np.pi) - np.pi
        self.q = np.clip(q, Q_LO, Q_HI)
        self.t += 1
        obs = self._obs()
        d = float(np.linalg.norm(self.goal - fk(self.q)))
        success = d < self.eps
        self.hold = self.hold + 1 if success else 0

        collided = self.use_obstacles and self._collides(self.q)
        margin = float(np.maximum(0.0, cfg.d_safe - np.maximum(self._link_min, 0.0)).sum()) if self.use_obstacles else 0.0
        smooth = float(((a - self.prev_action) ** 2).sum())
        safety = (cfg.w_m * margin + cfg.w_c * collided) if self.use_obstacles else 0.0
        if cfg.reward == "dense":
            r_indep = -cfg.w_a * float((a ** 2).sum()) - cfg.w_s * smooth - safety
        else:
            r_indep = (-cfg.w_s * smooth - safety) if self.use_obstacles else 0.0
        if collided and cfg.gamma is not None:  # absorbing penalty: the total is exactly R_coll for this goal
            goal_term = (-cfg.w_p * d + cfg.r_succ * success) if cfg.reward == "dense" else float(success) - 1.0
            r_indep = self.collision_reward() - goal_term
        info = dict(r_indep=r_indep, is_success=success, collision=bool(collided), d_pos=d,
                    curriculum_goal=self.curriculum_goal)
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        self.prev_action = a
        terminated = bool(collided) or (cfg.reward == "dense" and self.hold >= cfg.success_hold)
        truncated = self.t >= cfg.episode_length
        return obs, reward, terminated, truncated, info

    def per_step_min(self):
        """Analytic lower bound of the per-step (non-collision) reward in this mode."""
        cfg = self.cfg
        r = -cfg.w_s * 2 * 2.0 ** 2 - (cfg.w_m * 2 * cfg.d_safe if self.use_obstacles else 0.0)  # a in [-1, 1]^2
        if cfg.reward == "dense":
            return r - cfg.w_p * 2 * 0.9 - cfg.w_a * 2.0  # goal and EE both within the 0.9 m reach
        return r - 1.0

    def collision_reward(self):
        return self.per_step_min() / (1.0 - self.cfg.gamma) - self.cfg.w_c

    def compute_reward(self, achieved_goal, desired_goal, info):
        single = np.ndim(achieved_goal) == 1
        d = np.linalg.norm(np.atleast_2d(desired_goal) - np.atleast_2d(achieved_goal), axis=-1)
        success = d < self.eps
        if isinstance(info, dict):
            r_indep = np.array([info["r_indep"]])
        elif isinstance(info, np.ndarray) and info.dtype != object:
            r_indep = info.astype(float)
        else:
            r_indep = np.array([i["r_indep"] for i in info])
        if self.cfg.reward == "dense":
            r = -self.cfg.w_p * d + self.cfg.r_succ * success
        else:
            r = success.astype(float) - 1.0
        r = r + r_indep
        return r[0] if single else r

    def _obs(self):
        cfg = self.cfg
        qd = self.qdiff(self.q_prev, self.q) / cfg.dq_max  # in units of max joint speed; wrap-aware in q1
        ee = fk(self.q)
        q_lim = (self.q / Q_HI)[self.limited]
        parts = [np.sin(self.q), np.cos(self.q), q_lim, qd, ee, self.prev_action, np.zeros(2)]  # q_target - q = 0
        if self.use_obstacles:
            if self.mask.any():
                self._link_min = np.minimum(link_clearance(self.q, self.centers[self.mask],
                                                           self.radii[self.mask]).min(-1), 0.3)
            else:
                self._link_min = np.full(2, 0.3)
            slots = np.c_[self.centers, 10 * self.radii, self.mask] * self.mask[:, None]
            parts += [slots.ravel(), self._link_min / 0.3]
        return dict(observation=np.concatenate(parts).astype(np.float32),
                    achieved_goal=ee.astype(np.float32), desired_goal=self.goal.astype(np.float32))
