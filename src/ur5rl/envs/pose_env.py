"""Goal-conditioned UR5e + rigid 2F-85 pose-reaching env, with optional sphere obstacles (SPEC §6).

One class covers S1-S4; the stage config decides orientation, obstacles and episode length.
Not yet implemented (later days): tolerance curriculum (§6.6), residual observation (§7.3),
sim-to-real hooks (§6.9), S5 moving obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from ur5rl.envs import rotations as rot
from ur5rl.envs.clearance import ROBOT_BODIES, Clearance
from ur5rl.planners.ik import DH_A, DH_D, fk_flange

SCENE_XML = Path(__file__).resolve().parents[3] / "assets" / "scene_ur5e.xml"
HOME_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
SHOULDER = np.array([0.0, 0.0, 0.163])  # shoulder-lift axis height (DH d1)


@dataclass
class EnvConfig:
    orientation: bool = True  # False = S1 (position-only success)
    n_obstacles: tuple[int, int] = (0, 0)  # (min, max) active sphere slots per episode
    episode_length: int = 100
    reward: str = "dense"  # "dense" | "sparse"
    eps_pos: float = 0.01  # m
    eps_rot_deg: float = 5.0
    success_hold: int = 5  # dense mode: terminate after this many consecutive successful steps
    dq_max: float = 0.04  # rad / control step
    target_lead_max: float = 0.25  # rad, cap on |q_target - q| (§6.2)
    joint_limit: float = np.pi  # Week 1: +-pi on every joint (narrows the model's +-2pi)
    # S1/S2 goal bound: max_j |q_goal - q_start| <= goal_max_joint_dist (rad). None = whole joint box (S4).
    # 3.2 rad = 0.8 * v_max * 4 s; with the 6 s S1/S2 horizon this leaves a 1.5x time margin (§4, §6.5).
    goal_max_joint_dist: float | None = None
    # Goal-distance curriculum (§6.5): the bound starts at curriculum_start and is raised by a training callback;
    # curriculum_full_frac of goals always use the full bound. Evaluation envs keep curriculum off.
    curriculum: bool = False
    curriculum_start: float = 0.6
    curriculum_full_frac: float = 0.2
    curriculum_step: float = 0.3  # raise by this when rolling success (window episodes) exceeds threshold
    curriculum_threshold: float = 0.8
    curriculum_window: int = 100
    visual: bool = False  # load visual meshes (rendering / viewer only)
    goal_min_reach: float | None = None  # OOD test sets: TCP goal at least this far (m) from the shoulder
    goal_min_reach: float | None = None  # OOD test set: goals at least this far (m) from the shoulder joint
    frame_skip: int = 20
    timestep: float = 0.002
    home_noise: float = 0.3
    obstacle_radius: tuple[float, float] = (0.04, 0.10)
    # reward weights (§6.7)
    w_p: float = 1.0
    w_r: float = 0.3
    w_a: float = 0.01
    w_s: float = 0.05
    w_m: float = 5.0
    d_safe: float = 0.05
    w_c: float = 10.0
    r_succ: float = 5.0
    # Absorbing collision penalty (§6.7, decision 2026-10-28): with gamma set, a collision step's reward is
    # R_coll = r_min / (1 - gamma) - w_c (r_min = analytic per-step minimum, per_step_min()), the value of staying
    # in the worst state forever, so crashing can never beat timing out. None = the v1 one-off -w_c, which made
    # crashing early pay (archived e5_v1_crash_exploit). train.py sets gamma from the agent on obstacle stages.
    gamma: float | None = None
    extra: dict = field(default_factory=dict)


class UR5ePoseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | dict | None = None, scene_xml: str | Path = SCENE_XML):
        if isinstance(config, dict):
            config = EnvConfig(**config)
        self.cfg = cfg = config or EnvConfig()
        spec = mujoco.MjSpec.from_file(str(scene_xml))
        # Visual meshes cost ~98 MB per model and are never needed for training (identical physics without them).
        spec.compiler.discardvisual = not cfg.visual
        self.model = spec.compile()
        self.model.opt.timestep = cfg.timestep
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.q_lo = np.maximum(m.jnt_range[:6, 0], -cfg.joint_limit)
        self.q_hi = np.minimum(m.jnt_range[:6, 1], cfg.joint_limit)
        self.q_lim = np.maximum(np.abs(self.q_lo), np.abs(self.q_hi))
        self.tcp_id = m.site("tcp").id
        self.goal_mocap = m.body_mocapid[m.body("goal").id]
        self.K = sum(1 for b in range(m.nbody) if m.body(b).name.startswith("obstacle"))
        self.obs_geom = np.array([m.geom(f"obstacle{k}").id for k in range(self.K)])
        self.obs_mocap = np.array([m.body_mocapid[m.body(f"obstacle{k}").id] for k in range(self.K)])
        self.clearance = Clearance(m)
        self._body_min_bodies = ROBOT_BODIES
        # Reach bound (shoulder -> TCP) for the analytic per-step reward minimum: arm lengths after the shoulder
        # plus the TCP's offset from the flange (rigid gripper, so the offset is a constant).
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        tcp_offset = float(np.linalg.norm(d0.site_xpos[self.tcp_id] - fk_flange(d0.qpos[:6])[:3, 3]))
        self.reach = float(np.abs(DH_A).sum() + DH_D[3:].sum() + tcp_offset)
        self.use_obstacles = cfg.n_obstacles[1] > 0

        n_obs = 12 + 6 + 6 + 9 + 6 + 6
        if self.use_obstacles:
            n_obs += 5 * self.K + len(ROBOT_BODIES)
        box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.observation_space = gym.spaces.Dict(
            observation=box(n_obs), achieved_goal=box(9), desired_goal=box(9))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)

        self.ob_center = np.zeros((self.K, 3))
        self.ob_radius = np.full(self.K, 0.05)
        self.ob_mask = np.zeros(self.K, bool)
        self.goal = np.zeros(9)
        self.goal_R = np.eye(3)
        self.prev_action = np.zeros(6)
        self.q_target = HOME_Q.copy()
        self.curriculum_level = cfg.curriculum_start
        self.curriculum_goal = False  # this episode's goal came from the curriculum level (not full range)
        self.q_start = self.q_goal = HOME_Q.copy()
        self.t = 0
        self.hold = 0
        self.stats = dict(reset_samples=0)

    # ------------------------------------------------------------------ sampling (§6.5)

    def _in_collision(self, q):
        d = self.data
        d.qpos[:6] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        self.stats["reset_samples"] += 1
        return d.ncon > 0

    def _tcp_pose(self, q):
        d = self.data
        d.qpos[:6] = q
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.tcp_id].copy(), d.site_xmat[self.tcp_id].reshape(3, 3).copy()

    def _sample_start(self):
        while True:
            q = np.clip(HOME_Q + self.np_random.uniform(-1, 1, 6) * self.cfg.home_noise, self.q_lo, self.q_hi)
            if not self._in_collision(q):
                return q

    def set_curriculum_level(self, level):
        """Current per-joint goal bound (rad) while the curriculum is on; called through VecEnv.env_method."""
        self.curriculum_level = float(level)

    def _goal_bound(self):
        full = self.cfg.goal_max_joint_dist
        self.curriculum_goal = self.cfg.curriculum and self.np_random.random() >= self.cfg.curriculum_full_frac
        if self.curriculum_goal:
            return min(self.curriculum_level, full) if full is not None else self.curriculum_level
        return full

    def _sample_goal_q(self, q_start):
        bound = self._goal_bound()
        # Uniform over {q in limits : max_j |q_j - q_start_j| <= bound}. That set is a box, so it is sampled
        # directly (rejection from the full joint box would accept ~1 in 20,000 draws at 0.6 rad).
        lo, hi = self.q_lo, self.q_hi
        if bound is not None:
            lo, hi = np.maximum(lo, q_start - bound), np.minimum(hi, q_start + bound)
        while True:
            q = self.np_random.uniform(lo, hi)
            if self._in_collision(q):
                continue
            p = self._tcp_pose(q)[0]
            if p[2] <= 0.0:
                continue
            # OOD "boundary goals" (§9.2): TCP at least goal_min_reach from the shoulder joint
            if self.cfg.goal_min_reach is not None and np.linalg.norm(p - SHOULDER) < self.cfg.goal_min_reach:
                continue
            return q

    def _set_obstacles(self, centers, radii, mask):
        m, d = self.model, self.data
        self.ob_center[:], self.ob_radius[:], self.ob_mask[:] = centers, radii, mask
        for k in range(self.K):
            d.mocap_pos[self.obs_mocap[k]] = centers[k] if mask[k] else (0.0, 0.0, -5.0)
            g = self.obs_geom[k]
            m.geom_size[g, 0] = m.geom_rbound[g] = radii[k]
            m.geom_aabb[g, 3:] = radii[k]

    def _clear_at(self, q, centers, radii, margin=0.02):
        d = self.data
        d.qpos[:6] = q
        mujoco.mj_kinematics(self.model, d)
        return self.clearance.capsule_distances(d, centers, radii).min() > margin

    def _sample_obstacles(self, q_start, q_goal):
        lo, hi = self.cfg.n_obstacles
        n = self.np_random.integers(lo, hi + 1)
        p0, _ = self._tcp_pose(q_start)
        p1, _ = self._tcp_pose(q_goal)
        centers, radii = np.zeros((self.K, 3)), np.full(self.K, 0.05)
        for k in range(n):
            for _ in range(200):
                c = p0 + self.np_random.uniform(0.2, 0.8) * (p1 - p0) + self.np_random.uniform(-0.15, 0.15, 3)
                r = self.np_random.uniform(*self.cfg.obstacle_radius)
                radial = np.linalg.norm(c[:2])
                if not (0.2 < radial < 0.85 and r < c[2] < 1.0):
                    continue
                if self._clear_at(q_start, c[None], np.array([r])) and \
                        self._clear_at(q_goal, c[None], np.array([r])):
                    centers[k], radii[k] = c, r
                    break
            else:  # could not place this one; leave the slot inactive
                n = k
                break
        mask = np.arange(self.K) < n
        self._set_obstacles(centers, radii, mask)

    # ------------------------------------------------------------------ gym API

    def episode(self):
        """The current episode as a dict that reset(options={"episode": ...}) reproduces (test sets, §9.2)."""
        return dict(q_start=self.q_start.copy(), q_goal=self.q_goal.copy(), centers=self.ob_center.copy(),
                    radii=self.ob_radius.copy(), mask=self.ob_mask.copy())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        ep = (options or {}).get("episode")
        if ep is not None:  # replay a stored episode (frozen test sets)
            q_start, q_goal = np.asarray(ep["q_start"], float), np.asarray(ep["q_goal"], float)
            self._set_obstacles(np.asarray(ep["centers"]), np.asarray(ep["radii"]), np.asarray(ep["mask"], bool))
        else:
            self._set_obstacles(np.zeros((self.K, 3)), np.full(self.K, 0.05), np.zeros(self.K, bool))
            q_start = self._sample_start()
            q_goal = self._sample_goal_q(q_start)
            if self.use_obstacles:
                self._sample_obstacles(q_start, q_goal)
        self.q_start = q_start
        self.q_goal = q_goal  # for oracles and analysis only; never observed
        p_goal, R_goal = self._tcp_pose(q_goal)
        self.goal = np.r_[p_goal, rot.mat_to_r6d(R_goal)]
        self.goal_R = R_goal
        d.mocap_pos[self.goal_mocap] = p_goal
        d.mocap_quat[self.goal_mocap] = rot.mat_to_quat(R_goal)

        mujoco.mj_resetData(m, d)
        d.mocap_pos[self.goal_mocap] = p_goal
        d.mocap_quat[self.goal_mocap] = rot.mat_to_quat(R_goal)
        for k in range(self.K):
            d.mocap_pos[self.obs_mocap[k]] = self.ob_center[k] if self.ob_mask[k] else (0.0, 0.0, -5.0)
        d.qpos[:6] = q_start
        d.ctrl[:6] = q_start
        mujoco.mj_forward(m, d)
        self.q_target = d.qpos[:6].copy()
        self.prev_action[:] = 0
        self.t = 0
        self.hold = 0
        obs = self._obs()
        return obs, {}

    def per_step_min(self):
        """Analytic lower bound of the per-step (non-collision) reward in this mode (§6.7)."""
        cfg = self.cfg
        smooth_max, effort_max = 6 * 2.0 ** 2, 6.0  # a in [-1, 1]^6
        margin_max = len(self._body_min_bodies) * cfg.d_safe if self.use_obstacles else 0.0  # clearance clipped at 0
        r = -cfg.w_s * smooth_max - cfg.w_m * margin_max
        if cfg.reward == "dense":
            w_r = cfg.w_r if cfg.orientation else 0.0
            return r - cfg.w_p * 2 * self.reach - w_r * np.pi - cfg.w_a * effort_max
        return r - 1.0

    def collision_reward(self):
        return self.per_step_min() / (1.0 - self.cfg.gamma) - self.cfg.w_c

    def step(self, action):
        cfg, m, d = self.cfg, self.model, self.data
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        # Integrated targets (§6.2): the increment is added to the previous target, not to the measured q,
        # and the target may lead the arm by at most target_lead_max (no wind-up when the arm is blocked).
        q = d.qpos[:6]
        target = np.clip(self.q_target + a * cfg.dq_max, self.q_lo, self.q_hi)
        self.q_target = q + np.clip(target - q, -cfg.target_lead_max, cfg.target_lead_max)
        d.ctrl[:6] = self.q_target
        collided = False
        for _ in range(cfg.frame_skip):
            mujoco.mj_step(m, d)
            collided |= d.ncon > 0  # env-env contacts are filtered, so any contact involves the robot
        self.t += 1

        obs = self._obs()
        d_pos, theta = self._errors(self.achieved_goal, self.goal[None])
        success = bool(self._success(d_pos, theta)[0])
        self.hold = self.hold + 1 if success else 0

        smooth = float(((a - self.prev_action) ** 2).sum())
        effort = float((a ** 2).sum())
        # clearance clipped at 0: penetration is the collision term's job, and it keeps the margin term bounded
        margin = float(np.maximum(0.0, cfg.d_safe - np.maximum(self._body_min, 0.0)).sum())             if self.use_obstacles else 0.0
        # Collision and margin terms (and collision termination) apply from S3 on (§6.7, §6.8).
        safety = (cfg.w_m * margin + cfg.w_c * collided) if self.use_obstacles else 0.0
        if cfg.reward == "dense":
            r_indep = -cfg.w_a * effort - cfg.w_s * smooth - safety
        else:
            r_indep = (-cfg.w_s * smooth - safety) if self.use_obstacles else 0.0
        if self.use_obstacles and collided and cfg.gamma is not None:
            # absorbing penalty: the step's total reward is exactly R_coll for the episode's goal. Under HER
            # relabeling it can differ by the goal term's range (sparse: 1), still far below any time-out.
            if cfg.reward == "dense":
                w_r = cfg.w_r if cfg.orientation else 0.0
                goal_term = -cfg.w_p * float(d_pos[0]) - w_r * float(theta[0]) + cfg.r_succ * success
            else:
                goal_term = float(success) - 1.0
            r_indep = self.collision_reward() - goal_term
        info = dict(r_indep=r_indep, is_success=success, collision=collided, curriculum_goal=self.curriculum_goal,
                    d_pos=float(d_pos[0]),
                    theta=float(theta[0]), min_clearance=float(self._body_min.min()) if self.use_obstacles else np.inf)
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        self.prev_action = a

        terminated = (self.use_obstacles and collided) or (cfg.reward == "dense" and self.hold >= cfg.success_hold)
        truncated = self.t >= cfg.episode_length
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ reward (§6.7)

    def _errors(self, achieved, desired):
        achieved, desired = np.atleast_2d(achieved), np.atleast_2d(desired)
        d_pos = np.linalg.norm(desired[:, :3] - achieved[:, :3], axis=-1)
        theta = rot.geodesic_angle(rot.r6d_to_mat(desired[:, 3:]), rot.r6d_to_mat(achieved[:, 3:]))
        return d_pos, theta

    def _success(self, d_pos, theta):
        ok = d_pos < self.cfg.eps_pos
        if self.cfg.orientation:
            ok &= theta < np.deg2rad(self.cfg.eps_rot_deg)
        return ok

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Vectorised over a batch; goal-independent terms come back from info (HER decomposition)."""
        cfg = self.cfg
        single = np.ndim(achieved_goal) == 1
        d_pos, theta = self._errors(achieved_goal, desired_goal)
        success = self._success(d_pos, theta)
        if isinstance(info, dict):
            r_indep = np.array([info["r_indep"]])
        elif isinstance(info, np.ndarray) and info.dtype != object:
            r_indep = info.astype(float)  # SafetyHerReplayBuffer passes the stored r_indep array directly
        else:
            r_indep = np.array([i["r_indep"] for i in info])
        if cfg.reward == "dense":
            w_r = cfg.w_r if cfg.orientation else 0.0
            r = -cfg.w_p * d_pos - w_r * theta + cfg.r_succ * success
        else:
            r = success.astype(float) - 1.0
        r = r + r_indep
        return r[0] if single else r

    # ------------------------------------------------------------------ observation (§6.4)

    @property
    def achieved_goal(self):
        d = self.data
        return np.r_[d.site_xpos[self.tcp_id], rot.mat_to_r6d(d.site_xmat[self.tcp_id].reshape(3, 3))]

    def _obs(self):
        d = self.data
        q, qd = d.qpos[:6], d.qvel[:6]
        p = d.site_xpos[self.tcp_id]
        R = d.site_xmat[self.tcp_id].reshape(3, 3)
        ag = np.r_[p, rot.mat_to_r6d(R)]
        # No goal-dependent features here: HER relabels desired_goal only, so anything computed from the goal
        # would go stale on relabeled samples. Goal errors are computed by the policy (agents/policies.py).
        parts = [np.sin(q), np.cos(q), q / self.q_lim, qd / np.pi, ag, self.prev_action,
                 (self.q_target - q) / self.cfg.target_lead_max]
        if self.use_obstacles:
            self._body_min = self.clearance.body_min(d, self.ob_center, self.ob_radius, self.ob_mask)
            slots = np.c_[self.ob_center, 10 * self.ob_radius, self.ob_mask] * self.ob_mask[:, None]
            parts += [slots.ravel(), self._body_min / 0.3]
        return dict(observation=np.concatenate(parts).astype(np.float32),
                    achieved_goal=ag.astype(np.float32), desired_goal=self.goal.astype(np.float32))
