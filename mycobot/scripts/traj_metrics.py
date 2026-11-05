"""Trajectory-quality metrics from the real-arm logs (results/real/real_<set>_ep<N>_<method>.json).

Per run, from the measured joint readings q (20 Hz) and the commanded target:
  duration_s     time to the end of the episode (success ends it after 5 held ticks; a failure runs to the horizon)
  path_deg       joint-space path length of the measured motion, sum over joints of |dq| (deg)
  tcp_mm         length of the tool path (model FK on the measured q), mm; straight_mm is start-goal distance
  vmax_dps       peak joint speed (deg/s), on q smoothed with a 5-tick moving average
  amax_dps2      peak joint acceleration (deg/s^2), same smoothing
  jerk_rms       RMS joint jerk over moving ticks (deg/s^3), same smoothing
  track_mean/max largest |target - q| over joints per tick (deg), mean and max
  clear_min_cm   closest approach of the model arm to a virtual obstacle (cm), from the measured q
  success, collision
Derivatives on raw readings are dominated by the 0.088 deg encoder step, hence the smoothing; compare methods, not absolute values.

    python scripts/traj_metrics.py [results/real] [--set s4_test] [--out results/real_traj_metrics.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate  # noqa: E402
from mcrl.envs.pose_env import EnvConfig, MyCobotPoseEnv  # noqa: E402
from mcrl.shapes import load_testset  # noqa: E402
from ur5rl.eval import testsets  # noqa: E402

DT = 0.05


def smooth(x, k=5):
    ker = np.ones(k) / k
    pad = np.pad(x, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(x.shape[1])], axis=1)


def run_metrics(env, ep, ticks):
    q = np.array([t["q"] for t in ticks])
    g = np.array([t["target"] for t in ticks])
    qd = np.rad2deg(q)
    qs = smooth(qd)
    v = np.diff(qs, axis=0) / DT
    a = np.diff(v, axis=0) / DT
    j = np.diff(a, axis=0) / DT
    moving = np.abs(v).max(1) > 1.0
    env.reset(seed=0, options={"episode": ep})
    tcp, clear = [], []
    for qq in q:
        env.data.qpos[:6] = qq
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        tcp.append(env._tcp_pose(qq)[0])
        clear.append(float(env.clearance.body_min(env.data, env.ob_center, env.ob_radius, env.ob_mask).min()))
    tcp = np.array(tcp)
    return dict(
        duration_s=len(ticks) * DT,
        path_deg=float(np.abs(np.diff(qd, axis=0)).sum()),
        tcp_mm=float(1000 * np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum()),
        straight_mm=float(1000 * np.linalg.norm(tcp[-1] - tcp[0])),
        vmax_dps=float(np.abs(v).max()),
        amax_dps2=float(np.abs(a).max()),
        jerk_rms=float(np.sqrt((j[moving[:len(j)]] ** 2).mean())) if moving[:len(j)].any() else float("nan"),
        track_mean=float(np.rad2deg(np.abs(g - q).max(1)).mean()),
        track_max=float(np.rad2deg(np.abs(g - q).max(1)).max()),
        clear_min_cm=100 * min(clear),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", nargs="?", default="results/real")
    p.add_argument("--out", default="results/real_traj_metrics.csv")
    args = p.parse_args()
    envs, sets = {}, {}
    rows = []
    for f in sorted((ROOT / args.dir).glob("real_*_ep*_*.json")):
        m = re.match(r"real_(.+)_ep(\d+)_(b2|base|hybrid)(?:@\d+)?\.json$", f.name)
        if not m:
            continue
        name, ep_i, method = m[1], int(m[2]), m[3]
        if name not in sets:
            eps, meta = load_testset(name, ROOT / "testsets")
            sets[name] = eps
            envs[name] = MyCobotPoseEnv(EnvConfig(**evaluate.stage_env(meta)))
        data = json.load(open(f))
        s = data["summary"]
        if s.get("status") != "ok":
            continue
        r = run_metrics(envs[name], sets[name][ep_i], data["ticks"])
        rows.append(dict(set=name, ep=ep_i, method=method, success=bool(s["success"]), collision=bool(s["collision"]), **r))
    keys = list(rows[0].keys())
    with open(ROOT / args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[(r["set"], r["method"])].append(r)
    cols = ["duration_s", "path_deg", "tcp_mm", "vmax_dps", "amax_dps2", "jerk_rms", "track_mean", "track_max", "clear_min_cm"]
    print("median over episodes" + " " * 14 + "  ".join(f"{c:>12s}" for c in cols))
    for (name, method), rs in sorted(by.items()):
        print(f"{name:10s} {method:7s} n={len(rs):3d} succ={sum(r['success'] for r in rs):3d}  " +
              "  ".join(f"{np.median([r[c] for r in rs]):12.2f}" for c in cols))
    print("tool path / straight line (median):",
          {k: round(float(np.median([r['tcp_mm'] / max(r['straight_mm'], 1) for r in rs])), 2) for k, rs in sorted(by.items())})
    print("wrote", ROOT / args.out)


if __name__ == "__main__":
    main()
