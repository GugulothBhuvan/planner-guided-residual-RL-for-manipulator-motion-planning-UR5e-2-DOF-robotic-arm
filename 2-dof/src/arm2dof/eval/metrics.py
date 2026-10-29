"""Planar-arm metrics (companion SPEC §9.1): gap to the best-known path."""

from __future__ import annotations

import numpy as np

from arm2dof.cspace.astar import astar
from arm2dof.cspace.grid import grid
from arm2dof.envs.geometry import in_collision
from arm2dof.planners.ik import Q_HI, Q_LO
from ur5rl.planners.rrt_connect import JointSpace, path_length, rrt_connect, shortcut

SPACE = JointSpace(Q_LO, Q_HI, wrap=[True, False])


def best_known_length(centers, radii, q_start, q_goal, rng, occ=None, shortcut_iters=2000, rrt_budget=2.0):
    """L_best = min(L(A* + shortcut), L(RRT-Connect + shortcut)) in joint space (rad), or inf if neither found
    a path. Returns (L_best, details)."""
    collides = lambda q: bool(in_collision(q, centers, radii))
    occ = grid(centers, radii) if occ is None else occ
    lengths = {}
    q_a, _ = astar(occ, q_start, q_goal)
    if q_a is not None:
        # the grid path ends at cell centres: pin the true endpoints before smoothing
        q_a = np.vstack([q_start, q_a[1:-1], q_goal]) if len(q_a) > 2 else np.array([q_start, q_goal])
        lengths["astar"] = path_length(SPACE, shortcut(SPACE, collides, q_a, rng, iters=shortcut_iters, edge_res=0.005))
    q_r = rrt_connect(SPACE, collides, q_start, q_goal, rng, step=0.1, edge_res=0.005, time_budget=rrt_budget)
    if q_r is not None:
        lengths["rrt"] = path_length(SPACE, shortcut(SPACE, collides, q_r, rng, iters=shortcut_iters, edge_res=0.005))
    return (min(lengths.values()) if lengths else np.inf), lengths


def trajectory_length(qs):
    """Joint-space length of an executed trajectory (q1 wrap-aware), same metric as L_best."""
    qs = np.asarray(qs, float)
    return path_length(SPACE, qs)


def gap_to_best_known(policy_qs, l_best):
    """L(policy) / L_best - 1 on a successful episode (can be slightly negative if the policy beats both)."""
    return trajectory_length(policy_qs) / l_best - 1.0
