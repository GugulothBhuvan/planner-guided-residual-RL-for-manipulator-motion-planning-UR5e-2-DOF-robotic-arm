"""25 Hz streaming + settling test on J1, J2, J3 (small excursions, conservative speed).

For each joint: (a) ramp +DELTA deg over RAMP s streaming send_angles at 25 Hz, logging tracking lag;
hold HOLD s and read the settled error; (b) ramp back and settle again.
Safety: excursion capped, guide's joint limits, abort + stop() on any anomaly. Results -> results/stream_settle.json.
Usage: python stream_settle_test.py [COM13] [delta_deg] [speed]
"""
import json
import os
import sys
import time

import numpy as np
from pymycobot import MyCobot280

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM13"
DELTA = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
SPEED = int(sys.argv[3]) if len(sys.argv) > 3 else 20
LIMITS = [(-170, 170), (-130, 85), (-155, 155), (-165, 165), (-165, 165), (-180, 180)]
HZ, RAMP, HOLD = 25, 1.5, 1.0
MAX_EXC, MAX_OTHER = 8.0, 3.0

assert abs(DELTA) <= 5 and 1 <= SPEED <= 20, "capped at 5 deg and speed 20"
DT = 1.0 / HZ


def read(mc):
    a = mc.get_angles()
    if not a or len(a) < 6 or any(not isinstance(v, (int, float)) for v in a):
        raise RuntimeError(f"bad get_angles: {a}")
    return np.array(a[:6], float)


def stream(mc, start, j, a0, a1):
    """Stream a linear ramp of joint j from a0 to a1; return per-tick (target, actual, tick_time)."""
    rows, t0, n = [], time.perf_counter(), int(RAMP * HZ)
    for k in range(1, n + 1):
        tick = t0 + k * DT
        tgt = start.copy()
        tgt[j] = a0 + (a1 - a0) * k / n
        lo, hi = LIMITS[j]
        assert lo <= tgt[j] <= hi
        mc.send_angles(tgt.tolist(), SPEED)
        cur = read(mc)
        guard(start, cur, j)
        rows.append((tgt[j], cur[j], time.perf_counter() - t0))
        wait = tick - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
    return np.array(rows)


def guard(start, cur, j):
    d = np.abs(cur - start)
    if d[j] > MAX_EXC:
        raise RuntimeError(f"joint {j+1} excursion {d[j]:.1f} deg > {MAX_EXC}")
    others = np.delete(d, j)
    if others.max() > MAX_OTHER:
        raise RuntimeError(f"non-commanded joint moved {others.max():.1f} deg > {MAX_OTHER}")


def settle(mc, target_vec, j):
    time.sleep(HOLD)
    cur = read(mc)
    return float(cur[j] - target_vec[j])


mc = MyCobot280(PORT, 115200)
time.sleep(0.5)
start = read(mc)
print("start", start.round(2).tolist())

try:
    fresh = False
    try:
        mc.set_fresh_mode(1)
        fresh = True
    except Exception as exc:
        print("set_fresh_mode unavailable:", exc)

    results = {"speed": SPEED, "hz": HZ, "delta": DELTA, "fresh_mode": fresh, "joints": {}}
    for j in (0, 1, 2):
        a0 = start[j]
        out = {}
        for name, (x0, x1) in (("forward", (a0, a0 + DELTA)), ("back", (a0 + DELTA, a0))):
            rows = stream(mc, start, j, x0, x1)
            tgt_vec = start.copy()
            tgt_vec[j] = x1
            err = settle(mc, tgt_vec, j)
            lag = rows[:, 0] - rows[:, 1] if x1 > x0 else rows[:, 1] - rows[:, 0]
            dts = np.diff(rows[:, 2])
            out[name] = dict(lag_mean=float(lag.mean()), lag_max=float(lag.max()), settled_err=err,
                             tick_mean_ms=float(1e3 * dts.mean()), tick_max_ms=float(1e3 * dts.max()))
            print(f"J{j+1} {name:7s} lag mean {lag.mean():5.2f} max {lag.max():5.2f} deg | "
                  f"settled err {err:+.2f} deg | tick {1e3*dts.mean():.0f} ms (max {1e3*dts.max():.0f})")
        results["joints"][f"J{j+1}"] = out

    final = read(mc)
    results["final_dev_deg"] = (final - start).round(2).tolist()
    print("final deviation from start (deg):", results["final_dev_deg"])
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "stream_settle.json"), "w") as f:
        json.dump(results, f, indent=2)
except BaseException as exc:
    print("ERROR:", repr(exc), "-> stop()")
    try:
        mc.stop()
    except Exception as e2:
        print("stop failed:", e2)
    raise
finally:
    try:
        mc.set_fresh_mode(0)  # restore default
    except Exception:
        pass
