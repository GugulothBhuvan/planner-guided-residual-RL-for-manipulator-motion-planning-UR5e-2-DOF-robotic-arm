"""Motion diagnostic (SPEC M7f): how fast and how smoothly does the arm move under different command styles?

J1 only (base rotation), 20 deg excursions, each move out and back; all other joints are never commanded away from their
current values. Styles:
  incr   today's streaming: +0.5 deg every 50 ms (send, 30 ms, read), speed 20       [baseline]
  p2p    ONE send_angles to the far end, the firmware plans the whole move; speeds 20..70
  chain  the same 20 deg as 4 waypoints of 5 deg; the next is sent when < 2 deg remain (firmware blends the segments)
  burst  streaming with bigger steps (1 and 2 deg) every 40 ms, SEND ONLY (no reads while moving), to see whether the
         library/firmware stalls on its own or only when reads are interleaved
Metrics per move (polled joint reading, ~60 Hz): total time, 10-90% rate, peak rate over 0.1 s, fraction of polls in the
10-90% window with no advance (stop-and-go), residual of a straight-line fit of the window (deg rms), settle error.

Safety: J1 limit +-170, excursion <= 23 deg, stop() on any error or if a move ends > 3 deg from its target.
Usage: python scripts/motion_diag.py [COM13] [--only incr,p2p,chain,burst] [--speeds 20,30,50]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pymycobot import MyCobot280

ROOT = Path(__file__).resolve().parents[1]
D = 20.0  # overridden by --d
J1_LIMIT = 170.0

p = argparse.ArgumentParser()
p.add_argument("port", nargs="?", default="COM13")
p.add_argument("--only", default="incr,p2p,chain,burst")
p.add_argument("--speeds", default="20,30,40,50,70")
p.add_argument("--out", default="results/motion_diag.json")
p.add_argument("--d", type=float, default=20.0, help="excursion in deg (J1)")
args = p.parse_args()
ONLY = args.only.split(",")
D = args.d
assert D <= 70
SPEEDS = [int(s) for s in args.speeds.split(",")]

mc = MyCobot280(args.port, 115200)
time.sleep(1.0)


def read(tries=6):
    for _ in range(tries):
        a = mc.get_angles()
        if isinstance(a, list) and len(a) >= 6:
            return np.array(a[:6], float)
        time.sleep(0.05)
    raise RuntimeError("get_angles failed")


START = read()
print("start", START.round(2).tolist(), flush=True)
results = {}
TRACES = {}


def tgt_for(j1):
    t = START.copy()
    t[0] = j1
    assert abs(t[0]) <= J1_LIMIT and abs(t[0] - START[0]) <= D + 3, "J1 target out of range"
    return t


def settle(target_j1, timeout=8.0):
    """Poll until J1 stops changing; returns final reading."""
    t0 = time.perf_counter()
    last, still = None, 0
    while time.perf_counter() - t0 < timeout:
        a = read()[0]
        still = still + 1 if last is not None and abs(a - last) < 0.06 else 0
        last = a
        if still >= 8:
            break
        time.sleep(0.02)
    return read()[0]


def summarize(tr, a_begin, a_target):
    """tr: array (t, angle) polled during the move."""
    t, a = tr[:, 0], tr[:, 1]
    sgn = np.sign(a_target - a_begin)
    prog = sgn * (a - a_begin) / abs(a_target - a_begin)
    out = dict(t_total=float(t[-1]), n_polls=len(t), poll_ms=float(1e3 * np.mean(np.diff(t))) if len(t) > 1 else None)
    i10 = int(np.argmax(prog >= 0.1)) if (prog >= 0.1).any() else None
    i90 = int(np.argmax(prog >= 0.9)) if (prog >= 0.9).any() else None
    if i10 is None or i90 is None or i90 - i10 < 3:
        out["note"] = "too few polls in the 10-90% window"
        return out
    w = slice(i10, i90 + 1)
    tw, aw = t[w], a[w]
    span = tw[-1] - tw[0]
    out["t_10_90"] = float(span)
    out["rate_10_90"] = float(0.8 * abs(a_target - a_begin) / span)
    # peak over a 0.1 s window
    vmax = 0.0
    for i in range(len(tw)):
        j = np.searchsorted(tw, tw[i] + 0.1)
        if j < len(tw):
            vmax = max(vmax, abs(aw[j] - aw[i]) / (tw[j] - tw[i]))
    out["peak_rate_0.1s"] = float(vmax)
    adv = np.abs(np.diff(aw))
    out["stall_frac"] = float((adv < 0.1).mean())
    fit = np.polyfit(tw, aw, 1)
    out["line_resid_rms"] = float(np.sqrt(np.mean((aw - np.polyval(fit, tw)) ** 2)))
    return out


def poll_until_settled(t0, a_begin, a_target, timeout=10.0, on_poll=None):
    tr = []
    while time.perf_counter() - t0 < timeout:
        a = read()[0]
        tr.append((time.perf_counter() - t0, a))
        if on_poll:
            on_poll(a)
        if len(tr) > 8 and abs(tr[-1][1] - tr[-6][1]) < 0.06 and abs(tr[-1][1] - a_target) < 3.0:
            break
    return np.array(tr)


def finish(name, tr, a_begin, a_target):
    final = settle(a_target)
    err = final - a_target
    if abs(err) > 3.0:
        raise RuntimeError(f"{name}: ended {err:+.2f} deg from target")
    m = summarize(tr, a_begin, a_target)
    m["settle_err"] = float(err)
    return m


def show(name, m):
    keys = ["t_total", "rate_10_90", "peak_rate_0.1s", "stall_frac", "line_resid_rms", "settle_err", "poll_ms"]
    print(f"{name:26s} " + " ".join(f"{k}={m[k]:.2f}" if k in m and m[k] is not None else f"{k}=-" for k in keys)
          + (f"  [{m['note']}]" if "note" in m else ""), flush=True)


def go_back(speed=20):
    """Return to START at a gentle speed (also resets between experiments)."""
    mc.send_angles(START.tolist(), speed)
    time.sleep(0.03)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 10.0 and abs(read()[0] - START[0]) > 1.0:
        time.sleep(0.05)
    settle(START[0])


def exp_incr(step=0.5, speed=20, label=None):
    label = label or f"incr_{step}deg_s{speed}"
    for name, sgn in (("out", 1), ("back", -1)):
        a0 = read()[0]
        tgt_end = a0 + sgn * D
        n = int(round(D / step))
        tr, t0 = [], time.perf_counter()
        for k in range(1, n + 1):
            t = time.perf_counter()
            mc.send_angles(tgt_for(a0 + sgn * step * k).tolist(), speed)
            time.sleep(0.030)
            a = read()[0]
            tr.append((time.perf_counter() - t0, a))
            wait = t + 0.050 - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        tr = np.array(tr)
        m = finish(f"{label}_{name}", tr, a0, tgt_end)
        results[f"{label}_{name}"] = m
        TRACES[f"{label}_{name}"] = tr.tolist()
        show(f"{label}_{name}", m)
        time.sleep(0.5)


def exp_p2p(speed):
    label = f"p2p_s{speed}"
    for name, sgn in (("out", 1), ("back", -1)):
        a0 = read()[0]
        tgt_end = a0 + sgn * D
        t0 = time.perf_counter()
        mc.send_angles(tgt_for(tgt_end).tolist(), speed)
        time.sleep(0.030)
        tr = poll_until_settled(t0, a0, tgt_end)
        m = finish(f"{label}_{name}", tr, a0, tgt_end)
        results[f"{label}_{name}"] = m
        TRACES[f"{label}_{name}"] = tr.tolist()
        show(f"{label}_{name}", m)
        time.sleep(0.5)


def exp_chain(speed, nwp=4, near=2.0):
    label = f"chain{nwp}_s{speed}"
    for name, sgn in (("out", 1), ("back", -1)):
        a0 = read()[0]
        wps = [a0 + sgn * D * (i + 1) / nwp for i in range(nwp)]
        t0 = time.perf_counter()
        idx = 0
        mc.send_angles(tgt_for(wps[0]).tolist(), speed)
        time.sleep(0.030)
        state = {"idx": 0}

        def on_poll(a):
            i = state["idx"]
            if i < nwp - 1 and abs(a - wps[i]) < near:
                state["idx"] = i + 1
                mc.send_angles(tgt_for(wps[i + 1]).tolist(), speed)
                time.sleep(0.030)

        tr = poll_until_settled(t0, a0, wps[-1], on_poll=on_poll)
        m = finish(f"{label}_{name}", tr, a0, wps[-1])
        results[f"{label}_{name}"] = m
        TRACES[f"{label}_{name}"] = tr.tolist()
        show(f"{label}_{name}", m)
        time.sleep(0.5)


def exp_burst(step, speed=50, period=0.040):
    label = f"burst_{step}deg_s{speed}"
    for name, sgn in (("out", 1), ("back", -1)):
        a0 = read()[0]
        n = int(round(D / step))
        tgt_end = a0 + sgn * D
        send_ms = []
        t0 = time.perf_counter()
        for k in range(1, n + 1):
            t = time.perf_counter()
            mc.send_angles(tgt_for(a0 + sgn * step * k).tolist(), speed)
            send_ms.append(1e3 * (time.perf_counter() - t))
            wait = t + period - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        t_sent = time.perf_counter() - t0
        time.sleep(0.03)
        while abs(read()[0] - tgt_end) > 1.0 and time.perf_counter() - t0 < 10.0:
            time.sleep(0.02)
        t_arr = time.perf_counter() - t0
        final = settle(tgt_end)
        err = final - tgt_end
        if abs(err) > 3.0:
            raise RuntimeError(f"{label}_{name}: ended {err:+.2f} deg from target")
        m = dict(t_sent=t_sent, t_arrive=t_arr, send_ms_mean=float(np.mean(send_ms)), send_ms_max=float(np.max(send_ms)),
                 n_sends=n, settle_err=float(err))
        results[f"{label}_{name}"] = m
        print(f"{label}_{name:5s} sent {n} cmds in {t_sent:.2f} s (send call mean {m['send_ms_mean']:.0f} ms, max {m['send_ms_max']:.0f} ms), "
              f"arrived at {t_arr:.2f} s, settle err {err:+.2f}", flush=True)
        time.sleep(0.5)


try:
    if "incr" in ONLY:
        exp_incr(0.5, 20)
        go_back()
    if "p2p" in ONLY:
        for sp in SPEEDS:
            exp_p2p(sp)
            go_back()
    if "chain" in ONLY:
        for sp in [s for s in SPEEDS if s in (30, 50, 70)] or SPEEDS[:2]:
            exp_chain(sp)
            go_back()
    if "burst" in ONLY:
        exp_burst(1.0)
        go_back()
        exp_burst(2.0)
    go_back()
    final = read()
    print("final deviation from start (deg):", (final - START).round(2).tolist(), flush=True)
except BaseException as exc:
    print("ERROR:", repr(exc), "-> stop()", flush=True)
    try:
        mc.stop()
    except Exception as e2:
        print("stop failed:", e2)
    raise
finally:
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(start=START.tolist(), results=results, traces=TRACES), indent=1))
    print("wrote", out)
