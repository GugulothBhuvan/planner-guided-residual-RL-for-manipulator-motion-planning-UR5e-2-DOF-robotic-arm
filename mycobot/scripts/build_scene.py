"""Generate assets/scene_mc280.xml: MyCobot 280 M5 (+ suction head) in MuJoCo from the official URDF kinematics.

Capsule collision geoms only (the env's clearance code requires capsules); no visual meshes. The kinematic chain is the
URDF's (mc280/kinematics.py, validated against 15 real poses: 2.1 mm RMS). Servo model: first-order position
actuators, time constant kv/kp = 0.15 s (hardware: ~1.5 deg lag at 10 deg/s, results/track_test.json).

    python mycobot/scripts/build_scene.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc280.kinematics import JOINTS, LIMITS_DEG, rpy_to_R  # noqa: E402

TCP_OFFSET = (0.0072, -0.0053, 0.0008)  # m, in the J6 frame: fit to the arm's reported coords (results/poses.json)
KP, KV, ARMATURE, FORCE = 20.0, 3.0, 0.01, 3.0
# (name, fromto in the body frame, radius) per joint body; capsules approximate the links between joint frames
CAPS = {
    1: ("link1_col0", (0, 0, -0.07, 0, 0, 0.0), 0.035),
    2: ("link2_col0", (0, 0, 0, -0.1104, 0, 0), 0.028),
    3: ("link3_col0", (0, 0, 0, -0.096, 0, 0.06462), 0.025),
    4: ("link4_col0", (0, 0, 0, 0, -0.07318, 0), 0.022),
    5: ("link5_col0", (0, 0, 0, 0, 0.0456, 0), 0.022),
    6: ("link6_col0", (0, 0, -0.02, 0, 0, 0), 0.020),
}
TOOL_CAP = ("tool_col0", (0, 0, 0, 0, 0, 0.040), 0.015)  # suction head, along the flange axis
# non-adjacent pairs that overlap by construction (capsules are fatter than the links); verified collision-free
# on all real poses by tests
EXCLUDE = [("base", "link1"), ("link1", "link2"), ("link2", "link3"), ("link3", "link4"), ("link4", "link5"),
           ("link5", "link6"), ("link6", "tool"), ("link1", "link3"), ("link2", "link4"), ("link3", "link5"),
           ("link4", "link6"), ("link5", "tool"), ("link4", "tool")]


def quat(R):
    """Rotation matrix -> (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) / 2
    q = np.zeros(4)
    q[1 + i] = s
    q[1 + j] = (R[j, i] + R[i, j]) / (4 * s)
    q[1 + k] = (R[k, i] + R[i, k]) / (4 * s)
    q[0] = (R[k, j] - R[j, k]) / (4 * s)
    return q


f = lambda v: " ".join(f"{x:.6g}" for x in v)


def body(i, indent):
    xyz, rpy = JOINTS[i - 1]
    lo, hi = np.deg2rad(LIMITS_DEG[i - 1])
    name, ft, r = CAPS[i]
    pad = "  " * indent
    s = [f'{pad}<body name="link{i}" pos="{f(xyz)}" quat="{f(quat(rpy_to_R(*rpy)))}" gravcomp="1">',
         f'{pad}  <inertial pos="0 0 0" mass="0.15" diaginertia="0.0002 0.0002 0.0002"/>',
         f'{pad}  <joint name="j{i}" axis="0 0 1" range="{lo:.6f} {hi:.6f}" limited="true" armature="{ARMATURE}"/>',
         f'{pad}  <geom name="{name}" class="col" fromto="{f(ft)}" size="{r}"/>']
    if i < 6:
        s += body(i + 1, indent + 1)
    else:
        s += [f'{pad}  <site name="attachment_site" size="0.004" group="1"/>',
              f'{pad}  <body name="tool" gravcomp="1">',
              f'{pad}    <inertial pos="0 0 0.02" mass="0.05" diaginertia="0.00002 0.00002 0.00002"/>',
              f'{pad}    <geom name="{TOOL_CAP[0]}" class="col" fromto="{f(TOOL_CAP[1])}" size="{TOOL_CAP[2]}"/>',
              f'{pad}    <site name="tcp" size="0.005" pos="{f(TCP_OFFSET)}" group="1" rgba="1 0 0 1"/>',
              f'{pad}  </body>']
    s.append(f"{pad}</body>")
    return s


def main():
    lines = ['<mujoco model="mycobot280_m5">', '  <compiler angle="radian"/>', '  <option integrator="implicitfast"/>',
             '  <default>', '    <default class="col"><geom type="capsule" group="3" rgba="0.8 0.8 0.85 0.4"/></default>',
             '  </default>', '  <worldbody>',
             '    <geom name="floor" size="3 3 0.05" pos="0 0 -0.75" type="plane" contype="2"/>',
             '    <geom name="table" size="1.2 1.2 0.02" pos="0 0 -0.02" type="box" contype="2" rgba="0.55 0.45 0.35 1"/>',
             '    <light name="top" pos="0 0 3" type="directional"/>',
             '    <body name="base">', '      <inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>']
    lines += body(1, 3)
    lines += ['    </body>',
              '    <body name="goal" pos="0.2 0 0.2" mocap="true">',
              '      <geom name="goal_axis0" size="0.002 0.015" pos="0.015 0 0" quat="0.707107 0 -0.707107 0" type="capsule" contype="0" conaffinity="0" group="1" rgba="1 0 0 0.6"/>',
              '      <geom name="goal_axis1" size="0.002 0.015" pos="0 0.015 0" quat="0.707107 0.707107 0 0" type="capsule" contype="0" conaffinity="0" group="1" rgba="0 1 0 0.6"/>',
              '      <geom name="goal_axis2" size="0.002 0.015" pos="0 0 0.015" quat="0 1 0 0" type="capsule" contype="0" conaffinity="0" group="1" rgba="0 0 1 0.6"/>',
              '      <site name="goal" size="0.005" group="1" rgba="0 0.8 0 0.8"/>', '    </body>']
    for k in range(4):
        lines += [f'    <body name="obstacle{k}" pos="0 0 -5" mocap="true">',
                  f'      <geom name="obstacle{k}" size="0.02" contype="2" rgba="0.85 0.2 0.2 0.6"/>', '    </body>']
    lines += ['  </worldbody>', '  <contact>']
    lines += [f'    <exclude body1="{a}" body2="{b}"/>' for a, b in EXCLUDE]
    lines += ['  </contact>', '  <actuator>']
    for i in range(1, 7):
        lo, hi = np.deg2rad(LIMITS_DEG[i - 1])
        lines.append(f'    <general name="a{i}" joint="j{i}" ctrlrange="{lo:.6f} {hi:.6f}" forcerange="-{FORCE} {FORCE}" '
                     f'gainprm="{KP}" biasprm="0 -{KP} -{KV}" biastype="affine"/>')
    lines += ['  </actuator>', '</mujoco>']
    out = ROOT / "assets" / "scene_mc280.xml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
