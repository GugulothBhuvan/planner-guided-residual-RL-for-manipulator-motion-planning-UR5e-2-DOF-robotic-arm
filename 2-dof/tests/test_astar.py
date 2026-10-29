import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from arm2dof.cspace.astar import STEPS, astar, astar_cells
from arm2dof.cspace.components import components
from arm2dof.cspace.grid import RES, cell_of, config_of, grid


def dijkstra_cost(occ, start, goal):
    """Reference optimum on the same 8-connected, q1-wrapped grid graph."""
    n_r, n_c = occ.shape
    idx = lambda i, j: i * n_c + j
    rows, cols, w = [], [], []
    free = np.argwhere(~occ)
    for i, j in free:
        for di, dj, c in STEPS:
            ni, nj = (i + di) % n_r, j + dj
            if 0 <= nj < n_c and not occ[ni, nj]:
                rows.append(idx(i, j))
                cols.append(idx(ni, nj))
                w.append(c)
    g = coo_matrix((w, (rows, cols)), shape=(n_r * n_c, n_r * n_c)).tocsr()
    return dijkstra(g, indices=idx(*start))[idx(*goal)]


def test_astar_is_optimal_and_collision_free():
    rng = np.random.default_rng(0)
    for scene in range(3):
        centers = rng.uniform(-0.8, 0.8, (3, 2))
        radii = rng.uniform(0.05, 0.15, 3)
        occ = grid(centers, radii)
        labels = components(occ)
        free = np.argwhere(~occ)
        for _ in range(3):
            a, b = (tuple(free[k]) for k in rng.integers(0, len(free), 2))
            res = astar_cells(occ, a, b)
            ref = dijkstra_cost(occ, a, b)
            if labels[a] != labels[b]:
                assert res is None and np.isinf(ref)
                continue
            path, cost = res
            assert abs(cost - ref) < 1e-9
            assert not any(occ[c] for c in path)
            for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
                di = min(abs(i1 - i0), 360 - abs(i1 - i0))
                assert di <= 1 and abs(j1 - j0) <= 1


def test_astar_uses_the_q1_seam():
    occ = np.zeros((360, 324), bool)
    occ[170:191, :] = True  # full-height wall around q1 = 0: the only route is across q1 = +-pi
    start, goal = (100, 160), (260, 160)
    path, cost = astar_cells(occ, start, goal)
    assert not any(170 <= i <= 190 for i, _ in path)
    assert any(i in (0, 359) for i, _ in path)
    assert abs(cost - 200) < 1e-9  # 100 -> 0 -> 359 -> 260 = 100 + 1 + 99 rows, straight in q2
    assert abs(cost - dijkstra_cost(occ, start, goal)) < 1e-9


def test_astar_configs_unwrapped():
    occ = np.zeros((360, 324), bool)
    q_start, q_goal = config_of(2, 160), config_of(357, 160)
    q, length = astar(occ, q_start, q_goal)
    assert np.abs(np.diff(q[:, 0])).max() <= RES + 1e-12  # continuous across the seam
    assert abs(length - 5 * RES) < 1e-9  # 2 -> 1 -> 0 -> 359 -> 358 -> 357
    assert cell_of(q[-1]) == (357, 160)
