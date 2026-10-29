"""E1 / E2 results tables (final validation, 100 episodes per seed): per seed and mean, from each run's done.json.
Means of rates keep one decimal (2026-10-30: the E2 sparse mean of 3% and 6% printed as 4%; the text says 4.5%).

    python scripts/e2_table.py              ->  results/tables/e2.md
    python scripts/e2_table.py --exp e1     ->  results/tables/e1.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPS = {
    "e1": ("# E1: S1 (position, no obstacles, success within 2 cm), 500k steps, final validation (100 episodes)",
           {"sac": "SAC + HER", "td3": "TD3 + HER"}, "Table/self contact"),
    "e2": ("# E2: S2 (orientation, no obstacles), 500k steps, final validation (100 episodes)",
           {"sac_sparse_her": "SAC sparse + HER", "sac_dense": "SAC dense"}, "Table/self contact"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=sorted(EXPS), default="e2")
    args = p.parse_args()
    title, labels, contact = EXPS[args.exp]
    cols = [("success", "Success", "{:.0%}", "{:.1%}"), ("d_pos_cm", "Pos. error (cm)", "{:.1f}", "{:.1f}"),
            ("theta_deg", "Orient. error (°)", "{:.1f}", "{:.1f}"), ("collision", contact, "{:.0%}", "{:.1%}")]
    runs = ROOT / "results" / "runs" / args.exp
    rows = []
    for run, label in labels.items():
        vals = []
        for d in sorted(runs.glob(f"{run}_*_s*")):
            done = d / "done.json"
            if not done.exists() or not d.name.rsplit("_s", 1)[0].rsplit("_", 1)[0] == run:
                continue
            v = json.loads(done.read_text())["final_val"]
            vals.append(v)
            rows.append([label, d.name.rsplit("_s", 1)[1]] + [f.format(v[k]) for k, _, f, _ in cols])
        if vals:
            rows.append([f"**{label}**", f"**mean ({len(vals)})**"]
                        + [f"**{f.format(sum(v[k] for v in vals) / len(vals))}**" for k, _, _, f in cols])
    head = ["Method", "Seed"] + [c[1] for c in cols]
    lines = [title, "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
    out = ROOT / "results" / "tables" / f"{args.exp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
