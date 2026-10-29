"""Classical baselines on the UR5e (SPEC §7.2) and the reference tracker shared with the residual base (§7.3).

B1 = IK + quintic (S1/S2). B2 = RRT-Connect + shortcut smoothing + quintic time scaling (S3+). Both produce a
joint reference sampled at the control rate; it is executed through the env's own action interface by
`ReferenceTracker`, so baselines and policies are measured identically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ur5rl.envs import rotations as rot
from ur5rl.planners.collision import CollisionChecker
from ur5rl.planners.ik import ik_tcp
from ur5rl.planners.quintic import V_CAP, time_scale
from ur5rl.planners.rrt_connect import JointSpace, rrt_connect, shortcut


@dataclass
class Plan:
    ok: bool
    reference: np.ndarray | None = None  # (N, 6) joint targets at the control rate, starting at q_start
    waypoints: np.ndarray | None = None
    duration: float = np.inf  # s, time-scaled trajectory duration (planning time excluded)
    plan_time: float = 0.0  # s, wall time to the first valid plan (latency metric)
    path_length: float = np.inf  # rad, sum over segments of the joint-space length
    reason: str = ""


def goal_pose(env):
    T = np.eye(4)
    T[:3, :3] = rot.r6d_to_mat(env.goal[3:])
    T[:3, 3] = env.goal[:3]
    return T


def flange_to_tcp(env):
    """Fixed flange->TCP transform read from the model (as in tests/test_fk_tcp.py)."""
    import mujoco
    d = env.data.__class__(env.model)
    d.qpos[:6] = 0
    mujoco.mj_kinematics(env.model, d)
    T_f, T_t = np.eye(4), np.eye(4)
    T_f[:3, :3], T_f[:3, 3] = d.site("attachment_site").xmat.reshape(3, 3), d.site("attachment_site").xpos
    T_t[:3, :3], T_t[:3, 3] = d.site("tcp").xmat.reshape(3, 3), d.site("tcp").xpos
    return np.linalg.inv(T_f) @ T_t


def goal_configs(env, checker):
    """Collision-free IK solutions for the env's goal pose, nearest to the start first."""
    sols = ik_tcp(goal_pose(env), flange_to_tcp(env))
    sols = [s for s in sols if np.all(s >= env.q_lo - 1e-9) and np.all(s <= env.q_hi + 1e-9) and not checker(s)]
    return sorted(sols, key=lambda s: np.abs(s - env.q_start).max())


def plan_b1(env, checker=None):
    """IK (nearest collision-free branch) + quintic. No obstacle avoidance: S1/S2 only."""
    checker = checker or CollisionChecker()
    checker.set_obstacles(env.ob_center, env.ob_radius, env.ob_mask)
    t0 = time.perf_counter()
    goals = goal_configs(env, checker)
    if not goals:
        return Plan(False, reason="no collision-free IK solution")
    wp = np.array([env.q_start, goals[0]])
    # shortest quintic under the arm's 0.98 rad/s cap, T = 1.875 max|dq| / v_max (SPEC §7.2 B1)
    ref, dur = time_scale(wp, dt=env.cfg.frame_skip * env.cfg.timestep, v_max=V_CAP, a_max=None)
    return Plan(True, ref, wp, dur, time.perf_counter() - t0, float(np.abs(wp[1] - wp[0]).sum()))


def plan_b2(env, rng, checker=None, budget=2.0, inflate=0.0, shortcut_iters=200, step=0.3, edge_res=0.01,
            inflated_checker=None, exempt=0.3, max_checks=None):
    """RRT-Connect to the nearest reachable collision-free IK goal, then shortcut + quintic (SPEC §7.2 B2).

    `inflate` and a small `budget` / `shortcut_iters` give the deliberately degraded base of §7.3. Inflation
    applies everywhere except within `exempt` rad (max-norm) of the start and the goal, where the exact check is
    used: episodes only guarantee 2 cm clearance at the endpoints, so inflating there would make the endpoints
    themselves "collide" and the base would fail for a reason unrelated to path quality.
    """
    exact = checker or CollisionChecker()
    exact.set_obstacles(env.ob_center, env.ob_radius, env.ob_mask)
    infl = None
    if inflate > 0:
        infl = inflated_checker or CollisionChecker()
        infl.set_obstacles(env.ob_center, env.ob_radius, env.ob_mask, inflate=inflate)
    space = JointSpace(env.q_lo, env.q_hi)
    t0 = time.perf_counter()
    goals = goal_configs(env, exact)
    if not goals:
        return Plan(False, reason="no collision-free IK solution", plan_time=time.perf_counter() - t0)

    def checker_for(q_goal):
        if infl is None:
            return exact
        def collides(q):
            near = min(np.abs(q - env.q_start).max(), np.abs(q - q_goal).max()) < exempt
            return exact(q) if near else infl(q)
        return collides

    path, collides = None, exact
    for q_goal in goals[:1] if max_checks is not None else goals:  # nearest IK branch first
        left = budget - (time.perf_counter() - t0)
        if left <= 0:
            break
        collides = checker_for(q_goal)
        path = rrt_connect(space, collides, env.q_start, q_goal, rng, step=step, edge_res=edge_res,
                           time_budget=left if max_checks is None else np.inf, max_checks=max_checks)
        if path is not None:
            break
    plan_time = time.perf_counter() - t0
    if path is None:
        return Plan(False, reason="planning budget exceeded", plan_time=plan_time)
    path = shortcut(space, collides, path, rng, iters=shortcut_iters, edge_res=edge_res)
    ref, dur = time_scale(path, dt=env.cfg.frame_skip * env.cfg.timestep)
    return Plan(True, ref, path, dur, plan_time, float(np.abs(np.diff(path, axis=0)).sum()))  # L1, as §9.1


class ReferenceTracker:
    """a = clip((q_ref(t+1) - q_target) / dq_max, -1, 1) (SPEC §7.3 base controller); holds the last point."""

    def __init__(self, env, reference):
        self.env, self.ref, self.t = env, np.asarray(reference), 0

    def __call__(self, obs=None):
        k = min(self.t + 1, len(self.ref) - 1)
        self.t += 1
        return np.clip((self.ref[k] - self.env.q_target) / self.env.cfg.dq_max, -1.0, 1.0)


def rollout(env, policy, max_steps=None):
    """Run policy to termination/truncation; returns the episode's final info and per-step joint positions."""
    qs = [env.data.qpos[:6].copy()]
    collided = False
    for _ in range(max_steps or env.cfg.episode_length):
        _, _, term, trunc, info = env.step(policy(None))
        qs.append(env.data.qpos[:6].copy())
        collided |= info["collision"]
        if term or trunc:
            break
    info = dict(info, any_collision=collided)
    return info, np.array(qs)
