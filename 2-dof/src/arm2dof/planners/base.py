"""Planar-arm planners on the shared RRT-Connect (companion SPEC §7.2 B2, §7.4 residual base).

Same degradation as the UR5e base (UR5e SPEC §7.3): obstacles inflated except within `exempt` rad of the start
and goal, a budget counted in collision checks, few shortcuts, and a straight-line fallback that ignores
collisions when the budget runs out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm2dof.envs.geometry import in_collision
from arm2dof.planners.ik import Q_HI, Q_LO, ik
from ur5rl.planners.quintic import time_scale
from ur5rl.planners.rrt_connect import JointSpace, rrt_connect, shortcut

V_MAX, A_MAX = 1.1, 4.0  # rad/s, rad/s^2: below the kinematic env's 1.25 rad/s (0.05 rad / 0.04 s)


@dataclass
class Plan:
    ok: bool
    reference: np.ndarray  # (N, 2) at the control rate, q1 unwrapped (continuous)
    waypoints: np.ndarray
    duration: float


def space_for(env):
    return JointSpace(Q_LO, Q_HI, wrap=[env.cfg.wrap_q1, False])


def goal_configs(env):
    """Collision-free IK branches of the env's goal, nearest to the start first (wrap-aware); the sampled goal
    configuration if none is free."""
    exact = lambda q: bool(in_collision(q, env.centers[env.mask], env.radii[env.mask]))
    space = space_for(env)
    sols = sorted((q for q in ik(env.goal) if not exact(q)), key=lambda q: np.abs(space.diff(env.q_start, q)).sum())
    return sols or [env.q_goal]


def goal_config(env):
    """Nearest collision-free IK branch (the degraded base's target and the fallback line's end)."""
    return goal_configs(env)[0]


B2_BUDGET = 5.0  # s, full B2 (companion SPEC §7.2): at 0.5 s the Python RRT missed ~25% of feasible P2 episodes


def plan(env, rng, inflate=0.0, max_checks=None, time_budget=B2_BUDGET, shortcut_iters=200, exempt=0.3,
         step=0.1, edge_res=0.005):
    """Full B2 (max_checks None): every collision-free IK branch, nearest first, each with an equal share of
    `time_budget` (a branch in another C-space component would otherwise use up the whole budget). Degraded base
    (max_checks set): nearest branch only, as on the UR5e (UR5e SPEC §7.3)."""
    space = space_for(env)
    c, r = env.centers[env.mask], env.radii[env.mask]
    q0 = env.q_start
    exact = lambda q: bool(in_collision(q, c, r))
    goals = goal_configs(env)
    path, collides = None, exact
    for q1 in goals[:1] if max_checks is not None else goals:
        if inflate > 0:
            def collides(q, q1=q1):
                near = min(np.abs(space.diff(q, q0)).max(), np.abs(space.diff(q, q1)).max()) < exempt
                return exact(q) if near else bool(in_collision(q, c, r + inflate))
        else:
            collides = exact
        path = rrt_connect(space, collides, q0, q1, rng, step=step, edge_res=edge_res,
                           time_budget=time_budget / len(goals) if max_checks is None else np.inf,
                           max_checks=max_checks)
        if path is not None:
            break
    q1 = goals[0]
    dt = env.cfg.dt
    if path is None:  # fallback: straight joint-space line to the IK goal, collisions ignored
        wp = space.unwrap(np.array([q0, q1]))
        ref, dur = time_scale(wp, dt=dt, v_max=V_MAX, a_max=A_MAX)
        return Plan(False, ref, wp, dur)
    path = shortcut(space, collides, path, rng, iters=shortcut_iters, edge_res=edge_res)
    wp = space.unwrap(path)
    ref, dur = time_scale(wp, dt=dt, v_max=V_MAX, a_max=A_MAX)
    return Plan(True, ref, wp, dur)


class Tracker:
    """a = clip(diff(q, q_ref(t+1)) / dq_max, -1, 1), wrap-aware in q1; holds the last reference point."""

    def __init__(self, env, reference):
        self.env, self.ref, self.t = env, np.asarray(reference), 0

    def __call__(self, obs=None):
        k = min(self.t + 1, len(self.ref) - 1)
        self.t += 1
        return np.clip(self.env.qdiff(self.env.q, self.ref[k]) / self.env.cfg.dq_max, -1.0, 1.0)
