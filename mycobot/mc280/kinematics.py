"""MyCobot 280 M5 forward kinematics from the official mycobot_ros URDF (mycobot_280_m5 / with_pump).

Joint chain (parent->child origin xyz, rpy; all revolute about local z):
  base -> J1:  (0, 0, 0.13156)
  J2 : (0, 0, 0)               rpy (0, pi/2, -pi/2)
  J3 : (-0.1104, 0, 0)         rpy (0, 0, 0)
  J4 : (-0.096, 0, 0.06462)    rpy (0, 0, -pi/2)
  J5 : (0, -0.07318, 0)        rpy (pi/2, -pi/2, 0)
  J6 : (0, 0.0456, 0)          rpy (-pi/2, 0, 0)
Units: metres and radians internally; pymycobot reports degrees and millimetres.
"""
import numpy as np

HALF = np.pi / 2
# (xyz, rpy) for the fixed transform that precedes each joint rotation, J1..J6
JOINTS = [
    ((0.0, 0.0, 0.13156), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, HALF, -HALF)),
    ((-0.1104, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.096, 0.0, 0.06462), (0.0, 0.0, -HALF)),
    ((0.0, -0.07318, 0.0), (HALF, -HALF, 0.0)),
    ((0.0, 0.0456, 0.0), (-HALF, 0.0, 0.0)),
]
# Joint limits (rad) used by the sim: the guide's conservative GUI limits (deg), see SPEC.
LIMITS_DEG = np.array([[-170, 170], [-130, 85], [-155, 155], [-165, 165], [-165, 165], [-180, 180]], float)


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def fixed_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


def rotz(q):
    c, s = np.cos(q), np.sin(q)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


FIXED = [fixed_T(x, r) for x, r in JOINTS]


def fk_flange(q_rad, zero_offset=None):
    """4x4 pose of the J6 output frame (flange) in the base frame."""
    q = np.asarray(q_rad, float)
    if zero_offset is not None:
        q = q + np.asarray(zero_offset, float)
    T = np.eye(4)
    for F, qi in zip(FIXED, q):
        T = T @ F @ rotz(qi)
    return T


def fk_tool(q_rad, tool_xyz=(0.0, 0.0, 0.0), zero_offset=None):
    """Tool point (metres) = flange origin + tool_xyz expressed in the flange frame."""
    T = fk_flange(q_rad, zero_offset)
    return T[:3, 3] + T[:3, :3] @ np.asarray(tool_xyz, float), T
