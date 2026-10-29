"""P5a: planar 2-DOF reaching with torque control and gravity (companion SPEC §4, §5.3, §6).

MuJoCo integrates the vertical-plane model (assets/arm2dof_torque.xml; gravity acts, no gravcomp). Control at
125 Hz (timestep 0.002 s, frame skip 4); a in [-1, 1]^2 maps to tau = a * tau_max. Same GoalEnv contracts as the
kinematic env: goal-independent `observation`, goal-independent reward terms in info["r_indep"], HER-safe
compute_reward. Success = EE within eps of the goal AND |q'| < 0.1 rad/s (the arm must stop there). Starts at rest.
The speed condition lives in the goal space so HER stays exact: achieved_goal = [x, z, |q'| / 10],
desired_goal = [x_g, z_g, 0]; success compares positions and thresholds the achieved speed only.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from arm2dof.dynamics.model import TAU_MAX, XML
from arm2dof.planners.ik import Q_HI, Q_LO, fk

QD_SCALE = 10.0  # rad/s, observation scale for joint velocities


@dataclass
class TorqueConfig:
    episode_length: int = 100  # steps at 125 Hz; sized by the B5-duration rule before PE8 (§4)
    reward: str = "sparse"
    eps: float = 0.01
    qd_success: float = 0.1  # rad/s
    success_hold: int = 5
    frame_skip: int = 4
    mass_factor: float = 1.0  # perturbed plant (§9.2: 0.8 / 1.2); the controller models stay nominal
    w_p: float = 1.0
    w_a: float = 0.01  # normalised torque (energy) penalty, |a|^2
    w_s: float = 0.05
    r_succ: float = 5.0


class TorqueArmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: TorqueConfig | dict | None = None):
        if isinstance(config, dict):
            config = TorqueConfig(**config)
        self.cfg = cfg = config or TorqueConfig()
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        if cfg.mass_factor != 1.0:
            self.model.body_mass[1:] *= cfg.mass_factor
            self.model.body_inertia[1:] *= cfg.mass_factor
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * cfg.frame_skip
        box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        # sin q, cos q (4), q2 / q2_limit (1; q1 is continuous), q' / 10 (2), EE (2), previous action (2)
        self.observation_space = gym.spaces.Dict(observation=box(11), achieved_goal=box(3), desired_goal=box(3))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.goal = np.zeros(2)
        self.prev_action = np.zeros(2)
        self.t = self.hold = 0

    @property
    def q(self):
        return self.data.qpos.copy()

    @property
    def qd(self):
        return self.data.qvel.copy()

    def episode(self):
        return dict(q_start=self.q_start.copy(), q_goal=self.q_goal.copy())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ep = (options or {}).get("episode")
        if ep is not None:
            q, q_goal = np.asarray(ep["q_start"], float), np.asarray(ep["q_goal"], float)
        else:
            q, q_goal = self.np_random.uniform(Q_LO, Q_HI), self.np_random.uniform(Q_LO, Q_HI)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        self.q_start, self.q_goal, self.goal = q.copy(), q_goal, fk(q_goal)
        self.prev_action[:] = 0
        self.t = self.hold = 0
        return self._obs(), {}

    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        self.data.ctrl[:] = a * TAU_MAX
        mujoco.mj_step(self.model, self.data, nstep=cfg.frame_skip)
        self.t += 1
        obs = self._obs()
        d = float(np.linalg.norm(self.goal - fk(self.q)))
        success = bool(self._success(obs["achieved_goal"], obs["desired_goal"])[0])  # same test as compute_reward
        self.hold = self.hold + 1 if success else 0
        smooth = float(((a - self.prev_action) ** 2).sum())
        r_indep = -cfg.w_a * float((a ** 2).sum()) - cfg.w_s * smooth if cfg.reward == "dense" else 0.0
        info = dict(r_indep=r_indep, is_success=success, collision=False, d_pos=d, qd_norm=float(np.linalg.norm(self.qd)))
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        self.prev_action = a
        terminated = cfg.reward == "dense" and self.hold >= cfg.success_hold
        return obs, reward, terminated, self.t >= cfg.episode_length, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        single = np.ndim(achieved_goal) == 1
        ag, dg = np.atleast_2d(achieved_goal), np.atleast_2d(desired_goal)
        d = np.linalg.norm(dg[:, :2] - ag[:, :2], axis=-1)
        if isinstance(info, dict):
            r_indep = np.array([info["r_indep"]])
        elif isinstance(info, np.ndarray) and info.dtype != object:
            r_indep = info.astype(float)  # SafetyHerReplayBuffer path
        else:
            r_indep = np.array([i["r_indep"] for i in info])
        success = self._success(ag, dg)
        if self.cfg.reward == "dense":
            r = -self.cfg.w_p * d + self.cfg.r_succ * success
        else:
            r = success.astype(float) - 1.0
        r = r + r_indep
        return r[0] if single else r

    def _success(self, achieved, desired):
        ag, dg = np.atleast_2d(achieved), np.atleast_2d(desired)
        return (np.linalg.norm(dg[:, :2] - ag[:, :2], axis=-1) < self.cfg.eps) & (ag[:, 2] * QD_SCALE < self.cfg.qd_success)

    def _obs(self):
        q, qd = self.q, self.qd
        ee = fk(q)
        parts = [np.sin(q), np.cos(q), [q[1] / Q_HI[1]], qd / QD_SCALE, ee, self.prev_action]
        ag = np.r_[ee, np.linalg.norm(qd) / QD_SCALE]
        return dict(observation=np.concatenate(parts).astype(np.float32), achieved_goal=ag.astype(np.float32),
                    desired_goal=np.r_[self.goal, 0.0].astype(np.float32))
