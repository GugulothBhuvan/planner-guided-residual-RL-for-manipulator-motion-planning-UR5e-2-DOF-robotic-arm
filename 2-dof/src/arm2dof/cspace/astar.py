"""Grid A* in C-space (SPEC §7.2 B3): 8-connected, minimises joint-space path length, q1 wraps at +-pi."""

from __future__ import annotations

import heapq

import numpy as np

from arm2dof.cspace.grid import RES, cell_of, config_of

STEPS = [(di, dj, np.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]


def _h(a, b, n_rows):
    di = abs(a[0] - b[0])
    di = min(di, n_rows - di)
    dj = abs(a[1] - b[1])
    return np.hypot(di, dj)  # admissible: straight line is the shortest 8-connected cost lower bound


def astar_cells(occ, start, goal):
    """Optimal 8-connected cell path from start to goal (cells), or None. Cost in cells (x RES = radians)."""
    n_rows, n_cols = occ.shape
    if occ[start] or occ[goal]:
        return None
    g = {start: 0.0}
    parent = {start: None}
    heap = [(_h(start, goal, n_rows), 0.0, start)]
    closed = set()
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1], gc
        closed.add(cur)
        i, j = cur
        for di, dj, c in STEPS:
            ni, nj = (i + di) % n_rows, j + dj
            if not 0 <= nj < n_cols or occ[ni, nj]:
                continue
            nxt, ng = (ni, nj), gc + c
            if ng < g.get(nxt, np.inf):
                g[nxt], parent[nxt] = ng, cur
                heapq.heappush(heap, (ng + _h(nxt, goal, n_rows), ng, nxt))
    return None


def astar(occ, q_start, q_goal):
    """Joint-space path (N, 2) between configurations and its length (rad), or (None, inf).

    q1 values in the path are unwrapped (continuous), so consecutive points differ by at most one cell.
    """
    res = astar_cells(occ, cell_of(q_start), cell_of(q_goal))
    if res is None:
        return None, np.inf
    cells, cost = res
    q = np.array([config_of(i, j) for i, j in cells])
    q[:, 0] = np.unwrap(q[:, 0])
    return q, cost * RES
