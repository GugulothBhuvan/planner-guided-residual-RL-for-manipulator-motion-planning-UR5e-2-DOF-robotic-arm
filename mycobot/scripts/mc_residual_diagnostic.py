"""Task 0 of the reach plan (SPEC M10): is the MyCobot residual doing anything?  Evaluation only.

Replays the 200 s4_test episodes exactly as scripts/evaluate.py does (reset seed = episode index, deterministic actions,
final checkpoint) for base@N and hybrid@N, N in {1500, 3000, 6000}, both seeds, recording per episode
  - the per-joint mean and max of |alpha * pi_res| over the steps (rad),
  - dev_max = max over steps of max_j |q_target - q_ref(t)| (rad),
  - success, collision, final position error.
Bound of the additive residual around the reference: alpha * dq_max = 0.3 * 0.5 deg = 0.0026 rad per joint.

    python scripts/mc_residual_diagnostic.py run [--budgets 1500 3000 6000]    ->  results/eval/mc_diag/*.jsonl
    python scripts/mc_residual_diagnostic.py report                              ->  results/tables/mc_residual_diagnostic.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate  # noqa: E402
from mcrl.envs.pose_env import EnvConfig, MyCobotPoseEnv  # noqa: E402
from mcrl.envs.residual import ResidualEnv  # noqa: E402
from mcrl.shapes import load_testset  # noqa: E402

OUT = ROOT / "results" / "eval" / "mc_diag"
RUNS = {"s0": ROOT / "results/runs/m5/residual_b99f130c_s0", "s1": ROOT / "results/runs/m5/residual_b99f130c_s1"}
BUDGETS = (1500, 3000, 6000)


def replay(run, budget, episodes, env_cfg):
    model, cfg = evaluate.load_policy(run, "final") if run else (None, None)
    base_kw = dict(cfg["agent"]["base"]) if cfg else dict(inflate=0.017, shortcut_iters=20)
    if cfg:
        base_kw["inflate"] = 0.017
    base_kw["max_checks"] = budget
    alpha = cfg["agent"]["alpha_max"] if cfg else 0.0
    env = ResidualEnv(MyCobotPoseEnv(EnvConfig(**env_cfg)), alpha=alpha, **base_kw)
    base = env.unwrapped
    rows = []
    for i, ep in enumerate(episodes):
        obs, info0 = env.reset(seed=i, options={"episode": ep})
        mags, dev, hit = [], 0.0, False
        while True:
            a = np.zeros(6) if model is None else np.clip(model.predict(obs, deterministic=True)[0], -1, 1)
            mags.append(np.abs(alpha * a))
            obs, _, term, trunc, info = env.step(a)
            ref = env.reference[min(env.tracker.t, len(env.reference) - 1)]
            dev = max(dev, float(np.abs(base.q_target - ref).max()))
            hit |= bool(info["collision"])
            if term or trunc:
                break
        m = np.array(mags)
        rows.append(dict(success=bool(info["is_success"]), collision=hit, d_pos_cm=100 * info["d_pos"], dev_max=dev,
                         ares_mean=m.mean(0).tolist(), ares_max=m.max(0).tolist(), base_ok=bool(info0["base_ok"]),
                         steps=len(m)))
    return rows


def run(budgets):
    episodes, meta = load_testset("s4_test", ROOT / "testsets")
    env_cfg = evaluate.stage_env(meta)
    OUT.mkdir(parents=True, exist_ok=True)
    for b in budgets:
        for name, rd in (("base", None), ("s0", RUNS["s0"]), ("s1", RUNS["s1"])):
            f = OUT / f"{name}_{b}.jsonl"
            if f.exists():
                continue
            rows = replay(rd, b, episodes, env_cfg)
            f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            print(f"{name}@{b}: success {np.mean([r['success'] for r in rows]):.3f} collision "
                  f"{np.mean([r['collision'] for r in rows]):.3f} dev_max median {np.median([r['dev_max'] for r in rows]):.5f}", flush=True)


def report():
    load = lambda f: [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    ALPHA, DQ = 0.3, np.deg2rad(0.5)
    BOUND = ALPHA * DQ  # rad per joint per step
    pct = lambda x, q: float(np.percentile(x, q))
    md = ["# MyCobot residual diagnostic (s4_test, 200 episodes, horizon 900, final checkpoints)", "",
          f"Additive residual a = clip(a_base + alpha * pi_res); alpha = 0.3, dq_max = 0.5 deg per tick, so the residual moves the "
          f"target by at most alpha * dq_max = **{BOUND:.5f} rad** per joint per step (the bound, 1.0 below).", "",
          "## 1. Size of the residual action alpha * |pi_res| per step", "",
          "Mean over steps (then over episodes) and maximum over all steps, per joint J1-J6, in units of the bound (alpha * |pi_res| / alpha = |pi_res|; 1.0 = the residual output saturated).", "",
          "| Seed | Budget | mean J1 | J2 | J3 | J4 | J5 | J6 | max over all steps and joints | mean (rad), all joints |", "|---|---|---|---|---|---|---|---|---|---|"]
    data = {}
    for b in BUDGETS:
        for s in ("base", "s0", "s1"):
            f = OUT / f"{s}_{b}.jsonl"
            if f.exists():
                data[(s, b)] = load(f)
    for b in BUDGETS:
        for s in ("s0", "s1"):
            if (s, b) not in data:
                continue
            rows = data[(s, b)]
            mean = np.mean([r["ares_mean"] for r in rows], 0)
            mx = np.max([r["ares_max"] for r in rows])
            md.append(f"| {s} | {b} | " + " | ".join(f"{v / ALPHA:.3f}" for v in mean) + f" | {mx / ALPHA:.3f} | {mean.mean() * DQ:.5f} |")
    md += ["", "## 2. Deviation of the target from the reference, dev_max per episode (rad)", "",
           "dev_max = max over steps of max_j |q_target - q_ref(t)|. In units of the bound in the last column.", "",
           "| Arm | Budget | Episodes | p5 | median | p95 | max | median / bound |", "|---|---|---|---|---|---|---|---|"]
    for b in BUDGETS:
        for s in ("base", "s0", "s1"):
            if (s, b) not in data:
                continue
            d = [r["dev_max"] for r in data[(s, b)]]
            md.append(f"| {'base-only' if s == 'base' else 'hybrid ' + s} | {b} | {len(d)} | {pct(d, 5):.4f} | {pct(d, 50):.4f} | "
                      f"{pct(d, 95):.4f} | {max(d):.4f} | {pct(d, 50) / BOUND:.2f} |")
    md += ["", "## 3. Episodes whose outcome differs from the planner alone", "",
           "Same episode, same budget. Outcome = success, collision, or other (no success, no contact).", "",
           "| Seed | Budget | Success base | Success hybrid | Outcome differs | hybrid better (fail -> success) | hybrid worse (success -> fail) | success identical in the rest |", "|---|---|---|---|---|---|---|---|"]
    changed_total, n_total = 0, 0
    outcome = lambda r: "success" if r["success"] else ("collision" if r["collision"] else "other")
    for b in BUDGETS:
        for s in ("s0", "s1"):
            if (s, b) not in data or ("base", b) not in data:
                continue
            bs, hs = data[("base", b)], data[(s, b)]
            diff = [i for i in range(len(bs)) if outcome(bs[i]) != outcome(hs[i])]
            better = sum(1 for i in diff if not bs[i]["success"] and hs[i]["success"])
            worse = sum(1 for i in diff if bs[i]["success"] and not hs[i]["success"])
            changed_total += len(diff)
            n_total += len(bs)
            md.append(f"| {s} | {b} | {np.mean([r['success'] for r in bs]):.3f} | {np.mean([r['success'] for r in hs]):.3f} | "
                      f"{len(diff)} / {len(bs)} | {better} | {worse} | {len(bs) - len(diff)} |")
    # verdict
    allm = [np.mean(r["ares_mean"]) for k, rr in data.items() if k[0] != "base" for r in rr]
    mean_units = float(np.mean(allm) / ALPHA) if allm else float("nan")
    devs = [r["dev_max"] for k, rr in data.items() if k[0] != "base" for r in rr]
    devb = [r["dev_max"] for k, rr in data.items() if k[0] == "base" for r in rr]
    md += ["", "## Verdict", ""]
    if not allm:
        md.append("(no data yet)")
    else:
        if mean_units < 0.05:
            kind = "the residual collapsed to about zero"
        elif changed_total < 0.02 * n_total:
            kind = "the residual acts but cancels out"
        else:
            kind = f"the residual acts and changes the outcome in {changed_total} of {n_total} episode-runs"
        md.append(f"**{kind[0].upper() + kind[1:]}.** The mean residual action is {mean_units:.2f} of its bound "
                  f"({mean_units * BOUND:.5f} rad per joint per step); the median dev_max is {np.median(devs):.4f} rad for the hybrids "
                  f"and {np.median(devb):.4f} rad for the planner alone, against a bound of {BOUND:.4f} rad. "
                  f"{changed_total} of {n_total} episode-runs ({100 * changed_total / max(n_total, 1):.1f}%) end differently from the planner alone.")
    out = ROOT / "results" / "tables" / "mc_residual_diagnostic.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "report"])
    p.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    a = p.parse_args()
    run(a.budgets) if a.mode == "run" else report()
