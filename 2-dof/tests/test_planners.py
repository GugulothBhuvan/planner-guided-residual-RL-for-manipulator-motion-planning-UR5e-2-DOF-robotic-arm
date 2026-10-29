"""RRT-Connect + shortcut on the planar arm (q1 wraps), checked against grid A* (companion SPEC §13)."""

import numpy as np
import pytest

from arm2dof.cspace.astar import astar
from arm2dof.cspace.components import components
from arm2dof.cspace.grid import config_of, grid
from arm2dof.envs.geometry import in_collision
from arm2dof.planners.ik import Q_HI, Q_LO
from ur5rl.planners.rrt_connect import JointSpace, path_length, rrt_connect, shortcut

SPACE = JointSpace(Q_LO, Q_HI, wrap=[True, False])


def scene(seed):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.05, 0.15, 3)
    rho = rng.uniform(r + 0.1, 0.9)
    phi = rng.uniform(-np.pi, np.pi, 3)
    return np.c_[rho * np.cos(phi), rho * np.sin(phi)], r


def free_pair(occ, labels, rng):
    free = np.argwhere(~occ)
    while True:
        a, b = (tuple(free[k]) for k in rng.integers(0, len(free), 2))
        if labels[a] == labels[b] and a != b:
            return config_of(*a), config_of(*b)


@pytest.mark.parametrize("seed", range(5))
def test_rrt_path_is_collision_free_and_astar_is_shorter(seed):
    centers, radii = scene(seed)
    collides = lambda q: bool(in_collision(q, centers, radii))
    occ = grid(centers, radii)
    rng = np.random.default_rng(100 + seed)
    q0, q1 = free_pair(occ, components(occ), rng)
    path = rrt_connect(SPACE, collides, q0, q1, rng, step=0.1, edge_res=0.005, time_budget=5.0)
    assert path is not None
    np.testing.assert_allclose(path[0], q0)
    np.testing.assert_allclose(path[-1], q1)
    dense = SPACE.unwrap(path)
    for a, b in zip(dense[:-1], dense[1:]):  # re-check at 10x finer resolution
        for t in np.linspace(0, 1, 50):
            assert not collides(SPACE.normalize(a + t * (b - a)))

    # Optimal reference = grid A* + any-angle shortcut smoothing. Raw grid A* is optimal only for the
    # 8-connected metric (up to ~8% longer than the true joint-space optimum), so it is not compared directly.
    q_astar, _ = astar(occ, q0, q1)
    ref = shortcut(SPACE, collides, q_astar, np.random.default_rng(seed), iters=2000, edge_res=0.005)
    assert path_length(SPACE, ref) <= path_length(SPACE, path) + 2 * np.deg2rad(1.0)
    smooth = shortcut(SPACE, collides, path, rng, iters=200, edge_res=0.005)
    assert path_length(SPACE, smooth) <= path_length(SPACE, path) + 1e-12
    np.testing.assert_allclose(smooth[-1], q1)


def test_wrap_path_crosses_seam():
    collides = lambda q: bool(in_collision(q, np.array([[0.45, 0.0]]), np.array([0.12])))  # blocks q1 ~ 0
    q0, q1 = np.array([2.8, 0.3]), np.array([-2.8, 0.3])
    path = rrt_connect(SPACE, collides, q0, q1, np.random.default_rng(0), step=0.1, time_budget=5.0)
    smooth = shortcut(SPACE, collides, path, np.random.default_rng(1), iters=300)
    # the short way is across +-pi (0.68 rad), not the long way through 0 (5.6 rad, also blocked)
    assert path_length(SPACE, smooth) < 1.0


def test_infeasible_start_returns_none():
    collides = lambda q: True
    assert rrt_connect(SPACE, collides, np.zeros(2), np.ones(2), np.random.default_rng(0)) is None
