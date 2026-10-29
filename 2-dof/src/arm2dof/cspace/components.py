"""Connected free-space components with q1 wrap-around (SPEC §7.3), used for scene validity."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def components(occ):
    """Labels (360, 324) of 8-connected free components; 0 = occupied. q1 rows 0 and 359 are adjacent."""
    labels, n = ndimage.label(~occ, structure=np.ones((3, 3), int))
    # union-find over labels that touch across the q1 = +-pi seam (8-connectivity: j-1, j, j+1)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    top, bottom = labels[0], labels[-1]
    for dj in (-1, 0, 1):
        a = top[max(0, dj):len(top) + min(0, dj)]
        b = bottom[max(0, -dj):len(bottom) + min(0, -dj)]
        for x, y in zip(a, b):
            if x and y:
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[max(rx, ry)] = min(rx, ry)
    roots = np.array([find(k) for k in range(n + 1)])
    return roots[labels]


def connected(occ, cell_a, cell_b, labels=None):
    labels = components(occ) if labels is None else labels
    a, b = labels[cell_a], labels[cell_b]
    return bool(a and a == b)
