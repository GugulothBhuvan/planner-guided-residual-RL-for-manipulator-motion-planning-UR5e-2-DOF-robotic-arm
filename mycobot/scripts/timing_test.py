"""Per-call timing of the MyCobot 280 serial API, at rest and in motion (J1 only, 4 deg, speed 20).

1. rest:    get_angles / get_radians / get_coords / is_moving / send_angles(current pose) call durations
2. send:    wall-clock ramp +DELTA on J1 streaming send_angles at 25 Hz, NO reads (call durations, misses)
3. read:    one single send_angles back to start, then poll get_angles for the whole move (durations)
Safety: excursion capped at 5 deg, guide's J1 limit, stop() on any error. Results -> results/timing.json.
Usage: python timing_test.py [COM13] [delta_deg]
"""
import json
import os
import sys
import time

import numpy as np
from pymycobot import MyCobot280

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM13"
DELTA = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
SPEED, HZ, RAMP = 20, 25, 2.0
J1_LIMIT = 170.0
assert abs(DELTA) <= 5


def stats(x):
    x = 1e3 * np.asarray(x)
    return dict(n=len(x), mean=float(x.mean()), p50=float(np.median(x)), p95=float(np.percentile(x, 95)),
                max=float(x.max()))


def timed(fn, n):
    d = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        d.append(time.perf_counter() - t)
        time.sleep(0.005)
    return d


mc = MyCobot280(PORT, 115200)
time.sleep(0.5)
start = np.array(mc.get_angles()[:6], float)
assert len(start) == 6
print("start", start.round(2).tolist())
res = {"delta": DELTA, "speed": SPEED}

try:
    # 1. at rest
    rest = {}
    pose = start.tolist()
    for name in ("get_angles", "get_radians", "get_coords", "is_moving"):
        fn = getattr(mc, name, None)
        if callable(fn):
            rest[name] = stats(timed(fn, 30))
    rest["send_angles_same_pose"] = stats(timed(lambda: mc.send_angles(pose, SPEED), 30))
    res["rest"] = rest
    for k, v in rest.items():
        print(f"rest  {k:24s} mean {v['mean']:6.1f} p50 {v['p50']:6.1f} p95 {v['p95']:6.1f} max {v['max']:6.1f} ms")

    # 2. send-only streaming during motion, 25 Hz, wall-clock ramp
    n = int(RAMP * HZ)
    t0 = time.perf_counter()
    calls, late = [], 0
    for k in range(1, n + 1):
        tgt = start.copy()
        tgt[0] = start[0] + DELTA * k / n
        assert abs(tgt[0]) <= J1_LIMIT
        t = time.perf_counter()
        mc.send_angles(tgt.tolist(), SPEED)
        calls.append(time.perf_counter() - t)
        wait = t0 + k / HZ - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            late += 1
    wall = time.perf_counter() - t0
    res["send_only_motion"] = dict(stats(calls), ticks=n, late_ticks=late, wall_s=wall, planned_s=RAMP)
    print(f"send  motion  send_angles mean {1e3*np.mean(calls):.1f} p95 {1e3*np.percentile(calls,95):.1f} "
          f"max {1e3*max(calls):.1f} ms | late ticks {late}/{n} | wall {wall:.2f}s (planned {RAMP}s)")
    time.sleep(1.0)
    print("      J1 after send-only ramp:", round(mc.get_angles()[0] - start[0], 2), "deg from start")

    # 3. read-only polling during one single move back
    back = start.tolist()
    mc.send_angles(back, SPEED)
    t0, rd, vals = time.perf_counter(), [], []
    while time.perf_counter() - t0 < 3.0:
        t = time.perf_counter()
        a = mc.get_angles()
        rd.append(time.perf_counter() - t)
        vals.append((time.perf_counter() - t0, a[0]))
    res["read_only_motion"] = stats(rd)
    s = res["read_only_motion"]
    print(f"read  motion  get_angles mean {s['mean']:.1f} p50 {s['p50']:.1f} p95 {s['p95']:.1f} max {s['max']:.1f} ms "
          f"({s['n']} reads in 3 s)")
    res["read_only_motion"]["trace"] = [(round(a, 3), round(b, 2)) for a, b in vals[:60]]

    final = np.array(mc.get_angles()[:6], float)
    res["final_dev_deg"] = (final - start).round(2).tolist()
    print("final deviation from start (deg):", res["final_dev_deg"])
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "timing.json"), "w") as f:
        json.dump(res, f, indent=2)
except BaseException as exc:
    print("ERROR:", repr(exc), "-> stop()")
    try:
        mc.stop()
    except Exception as e2:
        print("stop failed:", e2)
    raise
