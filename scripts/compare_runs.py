"""Compare validation curves of two runs at equal env steps (SPEC §7.1 update-ratio decision).

"Clearly faster" (decision rule, fixed before the result is seen):
  candidate success at the final step >= baseline + 0.10, OR
  candidate mean final position error over the last 2 evaluations <= 0.75 x the baseline's.

    python scripts/compare_runs.py <baseline_run_dir> <candidate_run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def curve(run):
    rows = [json.loads(l) for l in (Path(run) / "val.jsonl").read_text().splitlines() if l.strip()]
    by_step = {}
    for r in rows:  # a resumed run may re-evaluate a step; keep the latest
        by_step[r["num_timesteps"]] = r
    return dict(sorted(by_step.items()))


def curriculum(run):
    p = Path(run) / "curriculum.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    base, cand = sys.argv[1], sys.argv[2]
    cb, cc = curve(base), curve(cand)
    print(f"{'steps':>8} | {'base succ':>9} {'d_pos cm':>8} | {'cand succ':>9} {'d_pos cm':>8}")
    for s in sorted(set(cb) | set(cc)):
        b, c = cb.get(s), cc.get(s)
        fmt = lambda r: f"{r['success']:9.2f} {r['d_pos_cm']:8.1f}" if r else f"{'-':>9} {'-':>8}"
        print(f"{s:>8} | {fmt(b)} | {fmt(c)}")
    print("curriculum (last raise):", "base", curriculum(base), "| cand", curriculum(cand))

    common = sorted(set(cb) & set(cc))
    last2 = common[-2:]
    succ_gain = cc[common[-1]]["success"] - cb[common[-1]]["success"]
    d_ratio = sum(cc[s]["d_pos_cm"] for s in last2) / sum(cb[s]["d_pos_cm"] for s in last2)
    faster = succ_gain >= 0.10 or d_ratio <= 0.75
    print(f"\nat {common[-1]} steps: success gain {succ_gain:+.2f}, final-distance ratio {d_ratio:.2f} -> "
          f"{'CLEARLY FASTER' if faster else 'not clearly faster'}")


if __name__ == "__main__":
    main()
