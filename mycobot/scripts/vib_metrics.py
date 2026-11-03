"""Smoothness metrics of the vibration experiment (results/vib/*.json, scripts/vib_batch.sh).

Per run (measured joint readings q at the 20 Hz control ticks, restricted to ticks where the commanded target moves):
  hf_rms   RMS of q minus its centred 9-tick moving average, all joints pooled (deg): high-frequency motion
  acc_rms  RMS of the second difference of q (deg/tick^2)
  stall    fraction of moving ticks where no joint advanced more than 0.1 deg
  cv       std / mean of the per-tick advance of the fastest joint
  lag      mean largest |target - q| (deg)
  dt_*     tick period mean / std / max (ms)
The arm's encoder step is 0.088 deg, so hf_rms has a floor of about 0.03 deg that every condition shares.

    python scripts/vib_metrics.py [results/vib]
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def metrics(ticks):
    q = np.rad2deg(np.array([t["q"] for t in ticks]))
    g = np.rad2deg(np.array([t["target"] for t in ticks]))
    wall = np.array([t.get("wall", np.nan) for t in ticks])
    dg = np.abs(np.diff(g, axis=0)).max(1)
    mov = np.r_[False, dg > 0.2]  # ticks where the target is moving
    # trim the ends: keep ticks whose 4 neighbours on both sides also move
    idx = [i for i in range(4, len(q) - 4) if mov[i - 4:i + 5].all()]
    if len(idx) < 20:
        return None
    k = np.ones(9) / 9
    trend = np.stack([np.convolve(q[:, j], k, mode="same") for j in range(6)], axis=1)
    hf = (q - trend)[idx]
    d2 = np.diff(q, 2, axis=0)  # d2[i] is centred on tick i + 1
    d2 = d2[[i - 1 for i in idx]]
    adv = np.abs(np.diff(q, axis=0)).max(1)[[i - 1 for i in idx]]
    dt = np.diff(wall) * 1000
    return dict(hf_rms=float(np.sqrt((hf ** 2).mean())), acc_rms=float(np.sqrt((d2 ** 2).mean())),
                stall=float((adv < 0.1).mean()), cv=float(adv.std() / adv.mean()),
                lag=float(np.abs(g - q).max(1)[idx].mean()), dt_mean=float(np.nanmean(dt)), dt_std=float(np.nanstd(dt)),
                dt_max=float(np.nanmax(dt)), n=len(idx))


def main():
    d = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/vib")
    rows = []
    for f in sorted(d.glob("real*_b2_*.json")):
        m = re.search(r"_b2_(.+)_r(\d+)\.json$", f.name)
        data = json.load(open(f))
        s = data["summary"]
        mt = metrics(data["ticks"]) if s.get("status") == "ok" else None
        rows.append(dict(cond=m[1], rep=int(m[2]), status=s["status"][:40], success=s.get("success"), **(mt or {})))
    out = d / "metrics.csv"
    keys = ["cond", "rep", "status", "success", "hf_rms", "acc_rms", "stall", "cv", "lag", "dt_mean", "dt_std", "dt_max", "n"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r.get(k) for k in keys} for r in rows)
    by = defaultdict(list)
    for r in rows:
        by[r["cond"]].append(r)
    print(f"{'condition':14s} {'runs':>4s} {'hf_rms':>7s} {'acc_rms':>8s} {'stall':>6s} {'cv':>5s} {'lag':>5s} {'dt_std':>7s} {'dt_max':>7s}  status")
    for c, rs in by.items():
        ok = [r for r in rs if "hf_rms" in r and r["hf_rms"] is not None]
        if not ok:
            print(f"{c:14s} {len(rs):4d}  no valid run: {[r['status'] for r in rs]}")
            continue
        mean = lambda k: np.mean([r[k] for r in ok])
        print(f"{c:14s} {len(rs):4d} {mean('hf_rms'):7.3f} {mean('acc_rms'):8.3f} {mean('stall'):6.2f} {mean('cv'):5.2f} "
              f"{mean('lag'):5.2f} {mean('dt_std'):7.1f} {max(r['dt_max'] for r in ok):7.0f}  "
              f"{sum(bool(r['success']) for r in rs)}/{len(rs)} success")
    print("wrote", out)


if __name__ == "__main__":
    main()
