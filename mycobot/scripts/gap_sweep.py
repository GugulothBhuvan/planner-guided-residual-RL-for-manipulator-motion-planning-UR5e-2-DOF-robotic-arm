"""Find the minimum gap the arm needs between commands. NO MOTION: send_angles re-sends the current pose.

For each gap g: repeat (A) then sleep g then (B) and time B, for pairs send->read, read->send, send->send.
Slow B (>~100 ms) means a dropped reply + library retry. Results -> results/gap_sweep.json.
Usage: python gap_sweep.py [COM13]
"""
import json
import os
import sys
import time

import numpy as np
from pymycobot import MyCobot280

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM13"
GAPS_MS = [0, 5, 10, 15, 20, 30, 40, 60]
N = 15

mc = MyCobot280(PORT, 115200)
time.sleep(0.5)
pose = [float(v) for v in mc.get_angles()[:6]]
print("pose", pose)

ops = {
    "send": lambda: mc.send_angles(pose, 20),
    "read": lambda: mc.get_angles(),
}
res = {}
try:
    for a, b in (("send", "read"), ("read", "send"), ("send", "send")):
        res[f"{a}->{b}"] = {}
        for g in GAPS_MS:
            tb = []
            for _ in range(N):
                ops[a]()
                time.sleep(g / 1000)
                t = time.perf_counter()
                ops[b]()
                tb.append(1e3 * (time.perf_counter() - t))
                time.sleep(0.3)  # let everything settle between trials
            slow = sum(x > 80 for x in tb)
            res[f"{a}->{b}"][g] = dict(mean=float(np.mean(tb)), max=float(max(tb)), slow=slow, n=N)
            print(f"{a}->{b} gap {g:2d} ms: {b} mean {np.mean(tb):6.1f} max {max(tb):6.1f} ms | slow {slow}/{N}")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "gap_sweep.json"), "w") as f:
        json.dump(res, f, indent=2)
except BaseException as exc:
    print("ERROR:", repr(exc), "-> stop()")
    try:
        mc.stop()
    except Exception as e2:
        print("stop failed:", e2)
    raise
