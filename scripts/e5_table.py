"""E5 headline table (SPEC §7.3, §9.3): success and collision on the 200-episode S4 test set for full B2, base@N,
dense and sparse hybrid@N (primary and final checkpoints, SPEC §9.2) and pure SAC+HER.

Two intervals per trained arm, as §9.3 asks:
  seeds   mean of the per-seed rates, 95% bootstrap CI over seeds (3 seeds: close to uninformative, stated once)
  pooled  all episodes of all seeds pooled, Wilson 95% interval (episode-level variation only, clearly labeled)
Planner rows (B2, base@N) are one deterministic evaluation, so only the episode-level interval exists for them.

    python scripts/e5_table.py   ->  results/tables/e5.md and e5.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "eval" / "s4_test"
ARM_NAMES = {"residual": "dense hybrid", "residual_sparse": "sparse hybrid", "residual_offset": "E5b offset hybrid",
             "sac_her": "pure SAC+HER"}


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def seed_ci(means, n=10_000, seed=0):
    m = np.asarray(means, float)
    if len(m) < 2:
        return float("nan"), float("nan")
    boot = m[np.random.default_rng(seed).integers(0, len(m), (n, len(m)))].mean(1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def collect():
    """{(method, budget, checkpoint): {run or "-": rows}}"""
    out: dict = {}
    for f in sorted(EVAL.glob("*.jsonl")):
        if f.stem == "b2":
            out[("full B2", None, "-")] = {"-": load(f)}
        elif m := re.fullmatch(r"base_(\d+)", f.stem):
            out[("base-only", int(m[1]), "-")] = {"-": load(f)}
        elif m := re.fullmatch(r"hybrid_(\d+)__(.+?)_([0-9a-f]{8})_s(\d+)__(primary|final)", f.stem):
            key = (ARM_NAMES.get(m[2], m[2]), int(m[1]), m[5])
            out.setdefault(key, {})[f"s{m[4]}"] = load(f)
        elif m := re.fullmatch(r"policy__(.+?)_([0-9a-f]{8})_s(\d+)", f.stem):
            out.setdefault((ARM_NAMES.get(m[1], m[1]), None, "best"), {})[f"s{m[3]}"] = load(f)
    return out


def row(method, budget, ckpt, runs):
    r = dict(method=method, budget=budget or "-", checkpoint=ckpt, seeds=len(runs) if "-" not in runs else "-")
    for key in ("success", "collision"):
        per_seed = [float(np.mean([e[key] for e in rows])) for rows in runs.values()]
        pooled = [e[key] for rows in runs.values() for e in rows]
        lo, hi = wilson(sum(pooled), len(pooled))
        r[key] = float(np.mean(per_seed))
        r[f"{key}_seed_ci"] = seed_ci(per_seed) if "-" not in runs else None
        r[f"{key}_pooled_ci"] = (lo, hi)
        r[f"{key}_per_seed"] = per_seed if "-" not in runs else None
        r["n_episodes"] = len(pooled)
    return r


def fmt(r, key):
    pct = lambda x: f"{100 * x:.1f}"
    s = pct(r[key])
    if r[f"{key}_seed_ci"]:
        s += f" [{pct(r[f'{key}_seed_ci'][0])}, {pct(r[f'{key}_seed_ci'][1])}]"
    lo, hi = r[f"{key}_pooled_ci"]
    s += f" ({pct(lo)}–{pct(hi)})"
    return s


def main():
    data = collect()
    order = {"full B2": 0, "base-only": 1, "dense hybrid": 2, "sparse hybrid": 3, "E5b offset hybrid": 4, "pure SAC+HER": 5}
    keys = sorted(data, key=lambda k: (order.get(k[0], 9), k[1] or 0, k[2] == "final"))
    rows = [row(*k, data[k]) for k in keys]
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# E5 headline: S4 test set (200 episodes per seed)", "",
          "Rates in %. Trained arms: mean over seeds, [95% bootstrap CI over seeds], (pooled Wilson 95% over all "
          "episodes of all seeds, episode-level only, SPEC §9.3). Planner rows: one deterministic evaluation, "
          "(Wilson 95%). Hybrid checkpoints: primary = best validation from 300k on, final = 1M (SPEC §9.2).", "",
          "| Method | Budget | Checkpoint | Seeds | Success | Collision | Per-seed success |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        per = ", ".join(f"{100 * x:.1f}" for x in r["success_per_seed"]) if r["success_per_seed"] else "-"
        md.append(f"| {r['method']} | {r['budget']} | {r['checkpoint']} | {r['seeds']} | {fmt(r, 'success')} | "
                  f"{fmt(r, 'collision')} | {per} |")
    (out / "e5.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    csv = ["method,budget,checkpoint,seeds,n_episodes,success,success_seed_lo,success_seed_hi,success_pool_lo,"
           "success_pool_hi,collision,collision_seed_lo,collision_seed_hi,collision_pool_lo,collision_pool_hi"]
    for r in rows:
        vals = [r["method"], r["budget"], r["checkpoint"], r["seeds"], r["n_episodes"]]
        for key in ("success", "collision"):
            vals += [r[key], *(r[f"{key}_seed_ci"] or ("", "")), *r[f"{key}_pooled_ci"]]
        csv.append(",".join(map(str, vals)))
    (out / "e5.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
