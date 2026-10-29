"""Joint-space RRT-Connect with optional wrap-around joints (SPEC §7.2 B2; companion spec B2).

Generic over the dimension: the UR5e (6 joints, bounded) and the planar 2-DOF arm (q1 wraps at +-pi) share it.
`collides(q)` takes one configuration; edges are checked at `edge_res` (rad, max-norm) intermediate points.
"""

from __future__ import annotations

import time

import numpy as np


class JointSpace:
    def __init__(self, lo, hi, wrap=None):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.wrap = np.zeros(len(self.lo), bool) if wrap is None else np.asarray(wrap, bool)

    def diff(self, a, b):
        """Shortest displacement from a to b (wrapping joints take the short way round)."""
        d = np.asarray(b, float) - np.asarray(a, float)
        d[..., self.wrap] = (d[..., self.wrap] + np.pi) % (2 * np.pi) - np.pi
        return d

    def normalize(self, q):
        q = np.array(q, float)
        q[..., self.wrap] = (q[..., self.wrap] + np.pi) % (2 * np.pi) - np.pi
        return q

    def dist(self, a, b):
        return np.linalg.norm(self.diff(a, b), axis=-1)

    def sample(self, rng):
        return rng.uniform(self.lo, self.hi)

    def unwrap(self, path):
        """Make wrapping joints continuous along a path (for execution and length)."""
        path = np.array(path, float)
        steps = self.diff(path[:-1], path[1:])
        return np.vstack([path[:1], path[:1] + np.cumsum(steps, 0)])


def edge_free(space, collides, a, b, edge_res):
    d = space.diff(a, b)
    n = max(1, int(np.ceil(np.abs(d).max() / edge_res)))
    for t in np.arange(1, n + 1) / n:
        if collides(space.normalize(a + t * d)):
            return False
    return True


def path_length(space, path):
    return float(space.dist(np.asarray(path)[:-1], np.asarray(path)[1:]).sum()) if len(path) > 1 else 0.0


class _Tree:
    def __init__(self, root):
        self.nodes = [np.asarray(root, float)]
        self.parent = [-1]

    def nearest(self, space, q):
        return int(np.argmin(space.dist(np.array(self.nodes), q)))

    def add(self, q, parent):
        self.nodes.append(q)
        self.parent.append(parent)
        return len(self.nodes) - 1

    def branch(self, i):
        out = []
        while i >= 0:
            out.append(self.nodes[i])
            i = self.parent[i]
        return out  # node i back to the root


def rrt_connect(space, collides, q_start, q_goal, rng, step=0.05, edge_res=0.01, time_budget=2.0, max_iters=50_000,
                max_checks=None):
    """Collision-free path [q_start, ..., q_goal] (wrapping joints normalised) or None (timeout / failure).

    `max_checks` caps the number of collision checks: a load-independent, reproducible budget (used for the
    degraded base inside training envs); `time_budget` is wall-clock (used where latency is the metric).
    """
    if max_checks is not None:
        count, inner = [0], collides

        def collides(q):
            count[0] += 1
            return inner(q)
    if collides(q_start) or collides(q_goal):
        return None
    t_end = time.perf_counter() + time_budget
    start_tree = _Tree(q_start)
    trees = [start_tree, _Tree(q_goal)]  # swapped every iteration

    def extend(tree, target):
        i = tree.nearest(space, target)
        d = space.diff(tree.nodes[i], target)
        n = np.linalg.norm(d)
        reached = n <= step
        q_new = space.normalize(tree.nodes[i] + (d if reached else d * step / n))
        if not edge_free(space, collides, tree.nodes[i], q_new, edge_res):
            return None, False
        return tree.add(q_new, i), reached

    for _ in range(max_iters):
        if time.perf_counter() > t_end or (max_checks is not None and count[0] > max_checks):
            return None
        a, b = trees
        k, _ = extend(a, space.sample(rng))
        if k is not None:
            target = a.nodes[k]
            while True:  # connect: greedily extend b toward the new node
                j, reached = extend(b, target)
                if j is None:
                    break
                if reached:
                    pa, pb = a.branch(k)[::-1], b.branch(j)
                    path = pa + pb[1:]  # root(a) -> root(b)
                    if a is not start_tree:
                        path = path[::-1]
                    return np.array(path)
        trees.reverse()
    return None


def shortcut(space, collides, path, rng, iters=200, edge_res=0.01):
    """Random shortcut smoothing: replace path[i..j] by a straight edge when it is collision-free."""
    path = [np.asarray(q, float) for q in path]
    for _ in range(iters):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if edge_free(space, collides, path[i], path[j], edge_res):
            path = path[:i + 1] + path[j:]
    return np.array(path)
