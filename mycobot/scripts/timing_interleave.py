"""Isolate the slow-tick cause: interleaved send_angles + get_angles at 25 Hz on J1 (4 deg), fresh_mode 0 vs 1.

Order per mode: ramp +DELTA then ramp back (wall-clock), each tick = send then read. Restores fresh_mode 0.
Usage: python timing_interleave.py [COM13] [delta_deg]
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
GAP = float(sys.argv[3]) / 1000 if len(sys.argv) > 3 else 0.0  # sleep after each send (firmware needs >= ~10 ms)
MODES = (0,) if len(sys.argv) > 3 else (0, 1)
assert abs(DELTA) <= 5

mc = MyCobot280(PORT, 115200)
time.sleep(0.5)
start = np.array(mc.get_angles()[:6], float)
print("start", start.round(2).tolist())
res = {}


def ramp(a0, a1, read):
    n = int(RAMP * HZ)
    t0, snd, rd, late = time.perf_counter(), [], [], 0
    for k in range(1, n + 1):
        tgt = start.copy()
        tgt[0] = a0 + (a1 - a0) * k / n
        t = time.perf_counter()
        mc.send_angles(tgt.tolist(), SPEED)
        snd.append(time.perf_counter() - t)
        time.sleep(GAP)
        if read:
            t = time.perf_counter()
            mc.get_angles()
            rd.append(time.perf_counter() - t)
        wait = t0 + k / HZ - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            late += 1
    return snd, rd, late, time.perf_counter() - t0


try:
    for mode in MODES:
        mc.set_fresh_mode(mode)
        time.sleep(0.3)
        out = {}
        for name, (a0, a1) in (("forward", (start[0], start[0] + DELTA)), ("back", (start[0] + DELTA, start[0]))):
            snd, rd, late, wall = ramp(a0, a1, read=True)
            out[name] = dict(send_ms_mean=1e3 * np.mean(snd), send_ms_max=1e3 * max(snd),
                             read_ms_mean=1e3 * np.mean(rd), read_ms_max=1e3 * max(rd), late=late, wall_s=wall)
            print(f"fresh={mode} {name:7s} send mean {1e3*np.mean(snd):6.1f} max {1e3*max(snd):6.1f} | "
                  f"read mean {1e3*np.mean(rd):6.1f} max {1e3*max(rd):6.1f} | late {late}/50 | wall {wall:.2f}s")
            time.sleep(1.0)
        res[f"fresh_{mode}"] = out
    final = np.array(mc.get_angles()[:6], float)
    print("final deviation from start (deg):", (final - start).round(2).tolist())
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "timing_interleave.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
except BaseException as exc:
    print("ERROR:", repr(exc), "-> stop()")
    try:
        mc.stop()
    except Exception as e2:
        print("stop failed:", e2)
    raise
finally:
    try:
        mc.set_fresh_mode(0)
    except Exception:
        pass
