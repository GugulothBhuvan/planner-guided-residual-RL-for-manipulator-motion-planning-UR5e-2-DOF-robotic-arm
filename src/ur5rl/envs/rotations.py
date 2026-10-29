"""Rotation utilities (SPEC §6.3, §6.6). Quaternions are MuJoCo order (w, x, y, z).

Every function is vectorised over leading dimensions.
"""

from __future__ import annotations

import numpy as np


def quat_to_mat(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(R):
    """Rotation matrix -> unit quaternion, canonicalised so that w >= 0."""
    R = np.asarray(R, dtype=float)
    batch = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    q = np.empty((len(R), 4))
    tr = np.trace(R, axis1=1, axis2=2)
    # Shepperd's method: pick the largest of (w, x, y, z) as the pivot for stability.
    cand = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], axis=1)
    k = cand.argmax(1)
    for i in range(4):
        s = k == i
        if not s.any():
            continue
        r = R[s]
        if i == 0:
            t = np.sqrt(1 + tr[s]) * 2
            q[s] = np.stack([0.25 * t, (r[:, 2, 1] - r[:, 1, 2]) / t, (r[:, 0, 2] - r[:, 2, 0]) / t,
                             (r[:, 1, 0] - r[:, 0, 1]) / t], 1)
        elif i == 1:
            t = np.sqrt(1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]) * 2
            q[s] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / t, 0.25 * t, (r[:, 0, 1] + r[:, 1, 0]) / t,
                             (r[:, 0, 2] + r[:, 2, 0]) / t], 1)
        elif i == 2:
            t = np.sqrt(1 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]) * 2
            q[s] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / t, (r[:, 0, 1] + r[:, 1, 0]) / t, 0.25 * t,
                             (r[:, 1, 2] + r[:, 2, 1]) / t], 1)
        else:
            t = np.sqrt(1 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]) * 2
            q[s] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / t, (r[:, 0, 2] + r[:, 2, 0]) / t,
                             (r[:, 1, 2] + r[:, 2, 1]) / t, 0.25 * t], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q.reshape(batch + (4,))


def mat_to_r6d(R):
    """First two columns of R, concatenated (Zhou et al., 2019)."""
    R = np.asarray(R, dtype=float)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def r6d_to_mat(x):
    """Gram-Schmidt on the two columns, then cross product for the third."""
    x = np.asarray(x, dtype=float)
    a, b = x[..., :3], x[..., 3:6]
    c1 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (c1 * b).sum(-1, keepdims=True) * c1
    c2 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    c3 = np.cross(c1, c2)
    return np.stack([c1, c2, c3], axis=-1)


def geodesic_angle(R1, R2):
    """theta = arccos((tr(R1^T R2) - 1) / 2), clipped for numerical safety."""
    tr = np.einsum("...ji,...ji->...", np.asarray(R1), np.asarray(R2))
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def mat_to_rotvec(R):
    """Log map: rotation matrix -> axis * angle, angle in [0, pi]. Stable near 0 and pi."""
    q = mat_to_quat(R)
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(n, q[..., :1])
    scale = np.where(n > 1e-12, angle / np.maximum(n, 1e-12), 2.0)  # small-angle limit: 2 * v
    return v * scale


def rotation_error(R_goal, R_cur):
    """Axis-angle vector (world frame) rotating R_cur onto R_goal."""
    return mat_to_rotvec(np.asarray(R_goal) @ np.swapaxes(np.asarray(R_cur), -1, -2))
