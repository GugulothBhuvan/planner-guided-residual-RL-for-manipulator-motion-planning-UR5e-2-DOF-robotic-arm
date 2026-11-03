"""Collect (joint angles, reported coords) pairs at varied safe poses, for FK calibration.

Poses are drawn from a fixed seed inside moderate joint ranges and filtered with the URDF FK (flange z >= 0.20 m,
joint changes <= 40 deg between consecutive poses). Speed 20, wait for is_moving == 0, settle, then read.
Results -> results/poses.json. Usage: python collect_poses.py [COM13] [n_poses]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mc280.kinematics import LIMITS_DEG, fk_tool  # noqa: E402
from pymycobot import MyCobot280  # noqa: E402

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM13"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
SPEED = 20
RANGES = np.array([[-35, 35], [-30, 30], [-40, 30], [-40, 40], [-40, 40], [-90, -30]], float)
Z_MIN, MAX_STEP = 0.20, 40.0


def read_retry(fn, tries=6):
    for _ in range(tries):
        v = fn()
        if isinstance(v, list) and len(v) >= 6:
            return [float(x) for x in v[:6]]
        time.sleep(0.15)
    raise RuntimeError("read failed")


mc = MyCobot280(PORT, 115200)
time.sleep(1.0)
cur = np.array(read_retry(mc.get_angles))
print("start", cur.round(2).tolist())

rng = np.random.default_rng(7)
plan, last = [], cur
while len(plan) < N:
    q = rng.uniform(RANGES[:, 0], RANGES[:, 1])
    if np.abs(q - last).max() > MAX_STEP:
        continue
    assert np.all(q >= LIMITS_DEG[:, 0]) and np.all(q <= LIMITS_DEG[:, 1])
    p, _ = fk_tool(np.deg2rad(q))
    if p[2] < Z_MIN:
        continue
    plan.append(q)
    last = q

records = []
try:
    for i, q in enumerate(plan):
        mc.send_angles(q.tolist(), SPEED)
        time.sleep(0.2)
        t0, still = time.time(), 0
        while time.time() - t0 < 10 and still < 3:
            m = mc.is_moving()
            still = still + 1 if m == 0 else 0
            time.sleep(0.12)
        time.sleep(0.6)
        a = read_retry(mc.get_angles)
        time.sleep(0.05)
        c = read_retry(mc.get_coords)
        records.append(dict(cmd=q.tolist(), angles=a, coords=c))
        print(f"{i:2d} cmd {np.round(q,1).tolist()} -> read {np.round(a,1).tolist()} coords {np.round(c[:3],1).tolist()}")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "poses.json"), "w") as f:
        json.dump(records, f, indent=2)
    # return to the starting pose
    mc.send_angles(cur.tolist(), SPEED)
    time.sleep(3.0)
    print("returned; final", np.round(read_retry(mc.get_angles), 2).tolist())
except BaseException as exc:
    print("ERROR:", repr(exc), "-> stop()")
    try:
        mc.stop()
    except Exception as e2:
        print("stop failed:", e2)
    raise
