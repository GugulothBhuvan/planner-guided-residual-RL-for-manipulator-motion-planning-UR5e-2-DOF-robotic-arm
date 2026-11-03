"""Tracking test at ~20 Hz (send, 30 ms gap, read): for each (speed setting, step per tick) stream a J1 ramp and
report tracking lag, to choose the speed cap and step size for the real loop.

Safety: J1 only, total excursion <= 10 deg per ramp, guide's J1 limit, bad reads skipped, stop() on error.
Results -> results/track_test.json. Usage: python track_test.py [COM13]
"""
import json
import os
import sys
import time

import numpy as np
from pymycobot import MyCobot280

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM13"
GAP, CYCLE = 0.030, 0.050
CASES = [(20, 0.5), (20, 1.0), (30, 1.0), (30, 1.5), (40, 1.0), (40, 1.5), (40, 2.0), (50, 2.0)]  # (speed, deg/tick)
MAX_EXC, J1_LIMIT = 10.0, 170.0


def read_all(mc, tries=6):
    for _ in range(tries):
        a = mc.get_angles()
        if isinstance(a, list) and len(a) >= 6:
            return np.array(a[:6], float)
        time.sleep(0.1)
    raise RuntimeError(f"get_angles failed {tries}x")


def read1(mc):
    return float(read_all(mc)[0])


mc = MyCobot280(PORT, 115200)
time.sleep(1.0)
start = read_all(mc)
print("start", start.round(2).tolist())
res = {}
try:
    for speed, step in CASES:
        n = int(MAX_EXC / step)
        out = {}
        for name, sign in (("fwd", 1), ("back", -1)):
            a_begin = read1(mc)
            lags, ticks = [], []
            t_prev = time.perf_counter()
            for k in range(1, n + 1):
                tgt = start.copy()
                tgt[0] = a_begin + sign * step * k
                assert abs(tgt[0]) <= J1_LIMIT and abs(tgt[0] - a_begin) <= MAX_EXC + 1e-6
                t = time.perf_counter()
                mc.send_angles(tgt.tolist(), speed)
                time.sleep(GAP)
                cur = read1(mc)
                lags.append(sign * (tgt[0] - cur))
                now = time.perf_counter()
                ticks.append(now - t)
                wait = t + CYCLE - now
                if wait > 0:
                    time.sleep(wait)
            time.sleep(1.0)
            settled = read1(mc)
            out[name] = dict(lag_mean=float(np.mean(lags)), lag_max=float(np.max(lags)), lag_last=float(lags[-1]),
                             tick_mean_ms=1e3 * float(np.mean(ticks)), tick_max_ms=1e3 * float(np.max(ticks)),
                             final_err=float(settled - (a_begin + sign * step * n)))
            o = out[name]
            print(f"speed {speed:2d} step {step:.1f} deg/tick ({step/CYCLE:4.0f} deg/s) {name:4s}: "
                  f"lag mean {o['lag_mean']:5.2f} max {o['lag_max']:5.2f} | tick {o['tick_mean_ms']:.0f}"
                  f" (max {o['tick_max_ms']:.0f}) ms | settled err {o['final_err']:+.2f}")
        res[f"{speed}_{step}"] = out
    final = read_all(mc)
    print("final deviation from start (deg):", (final - start).round(2).tolist())
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "track_test.json"), "w") as f:
        json.dump(res, f, indent=2)
except BaseException as exc:
    print("ERROR:", repr(exc), "-> stop()")
    try:
        mc.stop()
    except Exception as e2:
        print("stop failed:", e2)
    raise
