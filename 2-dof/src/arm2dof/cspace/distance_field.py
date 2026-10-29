"""Joint-space clearance field (SPEC §7.3): distance (rad) from each free cell to the nearest occupied cell."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from arm2dof.cspace.grid import RES


def distance_field(occ):
    """(360, 324) clearance in radians; q1 wraps (the grid is tiled three times along q1, then cropped).

    q2 limits are not treated as obstacles: the field measures distance to C-obstacles only.
    """
    if not occ.any():
        return np.full(occ.shape, np.inf)
    tiled = np.concatenate([occ, occ, occ], axis=0)
    d = ndimage.distance_transform_edt(~tiled) * RES
    return d[occ.shape[0]:2 * occ.shape[0]]
