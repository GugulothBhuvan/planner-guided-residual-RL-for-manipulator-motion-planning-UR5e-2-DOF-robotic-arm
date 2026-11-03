"""Map the send_angles speed setting to deg/s: J1, +/-10 deg moves at several speeds, polled with a clean gap.

Safety: J1 only, excursion 10 deg, guide's J1 limit, stop() on any error. Results -> results/speed_map.json.
Usage: python speed_map.py [COM13] [speeds comma-separated]
"""
import json
import os
import sys
import time

import numpy as np
from pymycobot import MyCobot280

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM13"
SPEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "10,20,30,40,50").split(",")]
DELTA = 10.0
J1_LIMIT = 170.0
assert all(1 <= s <= 50 for s in SPEEDS)

mc = MyCobot280(PORT, 115200)
time.sleep(0.5)
start = np.array(mc.get_angles()[:6], float)
print("start", start.round(2).tolist())
res = {}


def move(target, speed):
    assert abs(target[0]) <= J1_LIMIT and abs(target[0] - start[0]) <= DELTA + 3
    t0 = time.perf_counter()
    mc.send_angles(target.tolist(), speed)
    time.sleep(0.03)  # firmware needs a quiet gap after a send
    trace = []
    while time.perf_counter() - t0 < 8.0:
        a = mc.get_angles()
        trace.append((time.perf_counter() - t0, a[0]))
        if len(trace) > 6 and abs(trace[-1][1] - trace[-4][1]) < 0.05 and abs(trace[-1][1] - target[0]) < 2.0:
            break
    return np.array(trace)


try:
    for sp in SPEEDS:
        out = {}
        for name, sign in (("fwd", 1), ("back", -1)):
            cur0 = float(mc.get_angles()[0])
            tgt = start.copy()
            tgt[0] = cur0 + sign * DELTA
            tr = move(tgt, sp)
            a0, a1 = tr[0, 1], tr[-1, 1]
            moved = a1 - cur0
            t, a = tr[:, 0], tr[:, 1]
            # time between 10% and 90% of the actual travel
            frac = (a - cur0) / (moved if abs(moved) > 1e-6 else 1)
            i10 = int(np.argmax(frac >= 0.1))
            i90 = int(np.argmax(frac >= 0.9))
            span = t[i90] - t[i10]
            rate = 0.8 * abs(moved) / span if span > 1e-3 else float("nan")
            out[name] = dict(moved=float(moved), t_total=float(t[-1]), rate_10_90=float(rate), n=len(t))
            print(f"speed {sp:2d} {name:4s}: moved {moved:+6.2f} deg, reached in {t[-1]:.2f} s, "
                  f"10-90% rate {rate:6.1f} deg/s ({len(t)} reads)")
            time.sleep(0.5)
        res[sp] = out
    final = np.array(mc.get_angles()[:6], float)
    print("final deviation from start (deg):", (final - start).round(2).tolist())
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "speed_map.json"), "w") as f:
        json.dump(res, f, indent=2)
except BaseException as exc:
    print("ERROR:", repr(exc), "-> stop()")
    try:
        mc.stop()
    except Exception as e2:
        print("stop failed:", e2)
    raise
