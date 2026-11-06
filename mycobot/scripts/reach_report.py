"""Collect the reach results (SPEC M10) into results/tables/mc_reach.md and learning curves results/figures/mc_reach_curves.png.

    python scripts/reach_report.py            (after scripts/eval_reach.py scripted ... and policy ... for the runs)
Per stage: scripted straight / detour on the 100 test episodes, RL per seed (best-by-validation and last checkpoint),
steps and wall clock from done.json; 'matches' unless the two-proportion z of RL (3 seeds pooled) against the scripted rate > 2.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "results" / "eval" / "reach"
RUNS = ROOT / "results" / "runs" / "reach"
STAGES = [("r1", "R1 fixed start, no obstacle, ee"), ("r2", "R2 + sphere at the midpoint"), ("r3", "R3 + random start"), ("r4", "R4 R1 in joints mode")]


def z2(p1, n1, p2, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(max(p * (1 - p), 1e-12) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    md = ["# MyCobot reach (SPEC M10): learned controller against no-learning rules", "",
          "100 frozen test episodes per stage (`testsets/reach_<stage>_test.npz`), deterministic policies. Success = TCP within 2 cm. "
          "Collision = any contact (table, arm, sphere). RL = DDPG + HER (two critics) unless stated; 'best' = checkpoint with the best "
          "validation success (100 validation episodes, every 10k steps), 'last' = final.", ""]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, (st, title) in zip(axes, STAGES):
        sc = load(EV / f"scripted_{st}.json")
        md += [f"## {title}", "", "| Controller | Success | Collision | Final error median / max (cm) | Steps median | Training |", "|---|---|---|---|---|---|"]
        if sc:
            for k, lab in (("straight", "scripted straight line"), ("detour", "scripted detour (up over, across, down)")):
                r = sc[k]
                md.append(f"| {lab} | {r['success']:.0%} | {r['collision']:.0%} | {r['d_cm_median']:.2f} / {r['d_cm_max']:.1f} | {r['steps_median']:.0f} | none |")
        orc = load(EV / f"oracle_detour_{st}.json")
        if orc:
            md.append(f"| oracle: some via-point detour found by dry runs | {orc['solved']:.0%} | 0% | | | simulation look-ahead |")
        pooled = []
        for algo in ("ddpg", "sac"):
            for run in sorted(RUNS.glob(f"{st}_{algo}_s*")):
                done = load(run / "done.json")
                for which in ("best", "last"):
                    r = load(EV / f"policy_{run.name}_{which}.json")
                    if not r:
                        continue
                    wall = f"{done['steps'] // 1000}k steps, {done['wall_s'] / 60:.0f} min" if done else ""
                    md.append(f"| {algo.upper()}+HER seed {run.name[-1]} ({which}) | {r['success']:.0%} | {r['collision']:.0%} | "
                              f"{r['d_cm_median']:.2f} / {r['d_cm_max']:.1f} | {r['steps_median']:.0f} | {wall} |")
                    if algo == "ddpg" and which == "best":
                        pooled.append(r["success"])
                v = [json.loads(line) for line in (run / "val.jsonl").read_text().splitlines()] if (run / "val.jsonl").exists() else []
                if v:
                    ax.plot([x["steps"] for x in v], [x["success"] for x in v], lw=1.6, ls="-" if algo == "ddpg" else ":", label=f"{algo} s{run.name[-1]}")
        if pooled and sc:
            ref = max(sc["straight"]["success"], sc["detour"]["success"])
            z = z2(float(np.mean(pooled)), 100 * len(pooled), ref, 100)
            verdict = "beats" if z > 2 and len(pooled) >= 3 else ("below" if z < -2 else "matches")
            md += ["", f"RL (best checkpoints, {len(pooled)} seeds pooled) {np.mean(pooled):.0%} against the best scripted rule {ref:.0%}: z = {z:.1f}, so RL **{verdict}** it."]
        md.append("")
        if sc:
            ax.axhline(sc["straight"]["success"], color="#6b7385", ls="--", lw=1, label="scripted straight")
            if sc["detour"]["success"] > 0:
                ax.axhline(sc["detour"]["success"], color="#d9822b", ls="--", lw=1, label="scripted detour")
        ax.set_title(title.split(" ")[0] + " " + " ".join(title.split(" ")[1:4]), fontsize=9)
        ax.set_xlabel("training steps")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color="#d9dde6", lw=0.5)
    axes[0].set_ylabel("validation success")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    (ROOT / "results" / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "results" / "figures" / "mc_reach_curves.png", dpi=150)
    (ROOT / "results" / "tables" / "mc_reach.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
