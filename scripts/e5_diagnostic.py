"""E5 diagnostic (evaluation only; user hypothesis 2026-10-29): the base tracker re-aims q_target at q_ref every step,
a_base = clip((q_ref(t+1) - q_target) / dq_max), so it cancels the previous step's residual and the residual's
authority is bounded to about alpha * dq_max = 0.3 * 0.04 = 0.012 rad around q_ref: too small to detour around an
obstacle, enough to break the 1 cm / 5 deg goal tolerance.

Replays the S4 test episodes exactly as scripts/evaluate.py does (same seeds, budgets, deterministic actions) for
base@N and hybrid@N at the primary checkpoint of every residual run, recording per episode:
  dev_max      max over steps of max_j |q_target - q_ref(t)| (rad), q_ref(t) the point the base aimed at that step
  d_pos_cm, theta_deg, success, collision, and a failure class:
  collision    any contact;
  timeout      no contact, not successful, goal error still falling at the horizon (> 5% over the last 10 steps,
               the scripts/failure_breakdown.py rule);
  pose miss    no contact, not successful, otherwise (the arm stopped outside the tolerance).

    python scripts/e5_diagnostic.py [--runs <run dir> ...]   ->  results/eval/s4_test_diag/*.jsonl
    python scripts/e5_diagnostic.py --report                  ->  results/tables/e5_diagnostic.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv
from ur5rl.envs.residual import ResidualEnv
from ur5rl.eval import testsets

from evaluate import load_policy, stage_env

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "eval" / "s4_test_diag"
BUDGETS = (1500, 3000, 6000)


def classify(success, collision, d_hist):
    if success:
        return "success"
    if collision:
        return "collision"
    if len(d_hist) > 10 and d_hist[-1] < 0.95 * d_hist[-11]:
        return "timeout"
    return "pose miss"


def replay(run, budget, episodes, env_cfg):
    model, cfg = load_policy(run, "primary") if run else (None, None)
    base_kw = dict(cfg["agent"]["base"]) if cfg else dict(inflate=0.05, shortcut_iters=20)
    base_kw["max_checks"] = budget
    env = ResidualEnv(UR5ePoseEnv(EnvConfig(**env_cfg)), alpha=cfg["agent"]["alpha_max"] if cfg else 0.0, **base_kw)
    rows = []
    for i, ep in enumerate(episodes):
        obs, info0 = env.reset(seed=i, options={"episode": ep})
        base = env.unwrapped
        dev, d_hist, hit = 0.0, [], False
        while True:
            a = np.zeros(6) if model is None else model.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(a)
            ref = env.reference[min(env.tracker.t, len(env.reference) - 1)]
            dev = max(dev, float(np.abs(base.q_target - ref).max()))
            d_hist.append(info["d_pos"] / 0.01 + info["theta"] / np.deg2rad(5))  # goal error in tolerance units
            hit |= bool(info["collision"])
            if term or trunc:
                break
        success = bool(info["is_success"])
        rows.append(dict(success=success, collision=hit, d_pos_cm=100 * info["d_pos"],
                         theta_deg=float(np.rad2deg(info["theta"])), dev_max=dev, base_ok=bool(info0["base_ok"]),
                         outcome=classify(success, hit, d_hist)))
    return rows


def run_all(runs, with_base=True):
    episodes, meta = testsets.load("s4_test")
    env_cfg = stage_env(meta)
    OUT.mkdir(parents=True, exist_ok=True)
    for run in ([None] if with_base else []) + list(runs):
        for b in BUDGETS:
            name = f"base_{b}" if run is None else f"hybrid_{b}__{Path(run).name}"
            f = OUT / f"{name}.jsonl"
            if f.exists():
                continue
            rows = replay(run, b, episodes, env_cfg)
            f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            print(f"{name}: success {np.mean([r['success'] for r in rows]):.3f}, dev_max median "
                  f"{np.median([r['dev_max'] for r in rows]):.4f} rad", flush=True)


def report():
    load = lambda f: [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    base = {b: load(OUT / f"base_{b}.jsonl") for b in BUDGETS if (OUT / f"base_{b}.jsonl").exists()}
    hyb: dict = {}
    for f in sorted(OUT.glob("hybrid_*.jsonl")):
        m = re.fullmatch(r"hybrid_(\d+)__(residual(?:_sparse|_offset)?)_[0-9a-f]{8}_s(\d+)", f.stem)
        arm = {"residual": "dense", "residual_sparse": "sparse", "residual_offset": "E5b offset"}[m[2]]
        hyb.setdefault((arm, int(m[1])), {})[f"s{m[3]}"] = load(f)
    pct = lambda xs, q: float(np.percentile(xs, q))
    md = ["# E5 diagnostic: residual authority around the base reference (S4 test set, primary checkpoint)", "",
          "dev_max = max over steps of max_j |q_target - q_ref(t)| per episode (rad). Bound if the base cancels the "
          "residual every step: alpha * dq_max = 0.3 * 0.04 = 0.012 rad (the base action itself is clipped to "
          "+-dq_max, so the bound is exceeded only where the base saturates).", "",
          "## 1. Distribution of dev_max per episode", "",
          "| Arm | Budget | Seeds | Episodes | p5 | median | p95 | max | > 0.012 rad |", "|---|---|---|---|---|---|---|---|---|"]
    for b, rows in base.items():
        d = [r["dev_max"] for r in rows]
        md.append(f"| base-only | {b} | - | {len(d)} | {pct(d, 5):.4f} | {pct(d, 50):.4f} | {pct(d, 95):.4f} | "
                  f"{max(d):.4f} | {np.mean(np.array(d) > 0.012):.1%} |")
    for (arm, b), runs in sorted(hyb.items()):
        d = [r["dev_max"] for rows in runs.values() for r in rows]
        md.append(f"| {arm} hybrid | {b} | {len(runs)} | {len(d)} | {pct(d, 5):.4f} | {pct(d, 50):.4f} | "
                  f"{pct(d, 95):.4f} | {max(d):.4f} | {np.mean(np.array(d) > 0.012):.1%} |")
    md += ["", "## 2. Final pose error on episodes where both base-only and the hybrid avoided collisions", "",
           "Paired by episode (same test episode, same budget); hybrid seeds pooled. Mean and median.", "",
           "| Arm | Budget | Pairs | d_pos base (cm) | d_pos hybrid (cm) | theta base (deg) | theta hybrid (deg) | "
           "success base | success hybrid |", "|---|---|---|---|---|---|---|---|---|"]
    for (arm, b), runs in sorted(hyb.items()):
        pairs = [(base[b][i], r) for rows in runs.values() for i, r in enumerate(rows)
                 if not base[b][i]["collision"] and not r["collision"]]
        f = lambda key, k: f"{np.mean([p[k][key] for p in pairs]):.2f} / {np.median([p[k][key] for p in pairs]):.2f}"
        md.append(f"| {arm} | {b} | {len(pairs)} | {f('d_pos_cm', 0)} | {f('d_pos_cm', 1)} | {f('theta_deg', 0)} | "
                  f"{f('theta_deg', 1)} | {np.mean([p[0]['success'] for p in pairs]):.1%} | "
                  f"{np.mean([p[1]['success'] for p in pairs]):.1%} |")
    md += ["", "## 3. Outcome split per budget (% of episodes)", "",
           "| Arm | Budget | Episodes | success | collision | timeout | pose miss |", "|---|---|---|---|---|---|---|"]
    split = lambda rows: " | ".join(f"{np.mean([r['outcome'] == k for r in rows]):.1%}"
                                    for k in ("success", "collision", "timeout", "pose miss"))
    for b, rows in base.items():
        md.append(f"| base-only | {b} | {len(rows)} | {split(rows)} |")
        for arm in ("dense", "sparse", "E5b offset"):
            if (arm, b) in hyb:
                rows_h = [r for rs in hyb[(arm, b)].values() for r in rs]
                md.append(f"| {arm} hybrid | {b} | {len(rows_h)} | {split(rows_h)} |")
    out = ROOT / "results" / "tables" / "e5_diagnostic.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=None, help="residual run dirs (default: every finished residual run)")
    p.add_argument("--report", action="store_true")
    p.add_argument("--skip-base", action="store_true", help="another process replays base@N")
    args = p.parse_args()
    if args.report:
        return report()
    runs = args.runs if args.runs is not None else [
        str(d) for d in sorted((ROOT / "results" / "runs" / "e5").glob("residual*_s*")) if (d / "done.json").exists()]
    run_all(runs, with_base=not args.skip_base)


if __name__ == "__main__":
    main()
