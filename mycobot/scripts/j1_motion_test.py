"""Small slow J1 motion test: +delta deg and back. Logs read latency and move time.

Safety: refuses outside the conservative limits, speed capped, stop() on any error.
Usage: python j1_motion_test.py [COM13] [delta_deg] [speed]
"""
import sys
import time

from pymycobot import MyCobot280

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM13"
DELTA = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
SPEED = int(sys.argv[3]) if len(sys.argv) > 3 else 10
LIMITS = [(-170, 170), (-130, 85), (-155, 155), (-165, 165), (-165, 165), (-180, 180)]  # guide's GUI limits
TOL = 1.0
TIMEOUT = 15.0

assert abs(DELTA) <= 10 and 1 <= SPEED <= 20, "test is capped at 10 deg and speed 20"


def read(mc):
    t = time.perf_counter()
    a = mc.get_angles()
    dt = time.perf_counter() - t
    if not a or len(a) < 6 or any(not isinstance(v, (int, float)) for v in a):
        raise RuntimeError(f"bad get_angles: {a}")
    return [float(v) for v in a[:6]], dt


def move_and_wait(mc, target):
    t0 = time.perf_counter()
    mc.send_angles(target, SPEED)
    t_send = time.perf_counter() - t0
    while True:
        cur, _ = read(mc)
        err = max(abs(c - g) for c, g in zip(cur, target))
        if err <= TOL:
            return time.perf_counter() - t0, t_send, cur
        if time.perf_counter() - t0 > TIMEOUT:
            raise TimeoutError(f"did not reach target in {TIMEOUT}s, err={err:.2f} deg, cur={cur}")
        time.sleep(0.05)


mc = MyCobot280(PORT, 115200)
time.sleep(0.5)
start, _ = read(mc)
print("start", start)

try:
    lat = [read(mc)[1] for _ in range(20)]
    print(f"get_angles latency: mean {1e3*sum(lat)/len(lat):.1f} ms, max {1e3*max(lat):.1f} ms")

    goal = list(start)
    goal[0] = start[0] + DELTA
    lo, hi = LIMITS[0]
    assert lo <= goal[0] <= hi, f"J1 target {goal[0]} outside limits"

    dur, t_send, cur = move_and_wait(mc, goal)
    print(f"forward : {dur:.2f} s (send_angles call {1e3*t_send:.0f} ms), reached J1={cur[0]:.2f}")
    time.sleep(0.5)
    dur, t_send, cur = move_and_wait(mc, start)
    print(f"back    : {dur:.2f} s (send_angles call {1e3*t_send:.0f} ms), reached J1={cur[0]:.2f}")
    final, _ = read(mc)
    print("final", final, "max deviation from start:", round(max(abs(a - b) for a, b in zip(final, start)), 2), "deg")
except BaseException as exc:
    print("ERROR:", repr(exc), "-> stop()")
    try:
        mc.stop()
    except Exception as e2:
        print("stop failed:", e2)
    raise
