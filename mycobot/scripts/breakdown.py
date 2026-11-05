"""Per-type success / collision of an evaluated challenge set (results/eval/<set>/*.jsonl, types from testsets/<set>.npz).

    python scripts/breakdown.py s4_chal [s4_shape ...]        # prints a table and writes results/eval/<set>/by_type.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = {"b2": 0, "base": 1, "hybrid": 2}


def main():
    for name in sys.argv[1:] or ["s4_chal"]:
        z = np.load(ROOT / "testsets" / f"{name}.npz")
        cats = z["cat"]
        names = json.loads((ROOT / "testsets" / "summary.json").read_text())[name]["categories"]
        files = sorted((ROOT / "results" / "eval" / name).glob("*.jsonl"),
                       key=lambda f: (ORDER.get(f.stem.split("_")[0], 9), f.stem))
        rows = []
        for f in files:
            ep = [json.loads(line) for line in open(f)]
            if len(ep) != len(cats):
                continue
            label = f.stem.split("__")[0]
            for ci, cname in enumerate(names):
                sel = [e for e, c in zip(ep, cats) if c == ci]
                rows.append(dict(method=label, type=cname, n=len(sel),
                                 success=np.mean([e["success"] for e in sel]), collision=np.mean([e["collision"] for e in sel])))
            rows.append(dict(method=label, type="all", n=len(ep), success=np.mean([e["success"] for e in ep]),
                             collision=np.mean([e["collision"] for e in ep])))
        if not rows:
            print(name, "no finished methods yet")
            continue
        with open(ROOT / "results" / "eval" / name / "by_type.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["method", "type", "n", "success", "collision"])
            w.writeheader()
            w.writerows(rows)
        types = names + ["all"]
        print(f"\n{name}: success / collision per type")
        print(f"{'method':22s}" + "".join(f"{t:>16s}" for t in types))
        for m in dict.fromkeys(r["method"] for r in rows):
            cell = lambda t: next((f"{r['success']:.2f} / {r['collision']:.2f}" for r in rows if r["method"] == m and r["type"] == t), "-")
            print(f"{m:22s}" + "".join(f"{cell(t):>16s}" for t in types))


if __name__ == "__main__":
    main()
