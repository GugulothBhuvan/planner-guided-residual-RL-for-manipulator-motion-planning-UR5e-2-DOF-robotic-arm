"""PE3 / PE4 tables (companion SPEC §8, §9): success, collision and gap to the best-known path on the frozen P2 / P3
test sets for B2, B3, base@N, hybrid@N (primary and final checkpoint, UR5e SPEC §9.2) and pure SAC+HER.
Intervals as in the E5 headline table (UR5e SPEC §9.3): trained arms show the mean over seeds, [95% bootstrap CI
over seeds] and (pooled Wilson 95%, episode-level only); planner rows only the Wilson interval.

    python 2-dof/scripts/pe_tables.py   ->  results/tables/pe3.md, pe4.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from e5_table import seed_ci, wilson  # noqa: E402

SETS = {"pe3": ("p2_test", "P2 (static obstacles)"), "pe4": ("p3_test", "P3 (narrow passage); PE4b rows included")}
NAMES = {"residual": "hybrid", "residual_offset": "offset hybrid (PE4b)", "sac_her": "pure SAC+HER"}


def load(f):
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect(d):
    out: dict = {}
    for f in sorted(d.glob("*.jsonl")):
        if f.stem in ("b2", "b3"):
            out[(f.stem.upper() + (" (full)" if f.stem == "b2" else " (A*)"), None, "-")] = {"-": load(f)}
        elif m := re.fullmatch(r"base_(\d+)", f.stem):
            out[("base-only", int(m[1]), "-")] = {"-": load(f)}
        elif m := re.fullmatch(r"hybrid_(\d+)__(.+?)_[0-9a-f]{8}_s(\d+)__(primary|final)", f.stem):
            out.setdefault((NAMES.get(m[2], m[2]), int(m[1]), m[4]), {})[f"s{m[3]}"] = load(f)
        elif m := re.fullmatch(r"policy__(.+?)_[0-9a-f]{8}_s(\d+)", f.stem):
            out.setdefault((NAMES.get(m[1], m[1]), None, "best"), {})[f"s{m[2]}"] = load(f)
    return out


def cell(runs, key):
    per = [float(np.mean([e[key] for e in rows])) for rows in runs.values()]
    pooled = [e[key] for rows in runs.values() for e in rows]
    lo, hi = wilson(sum(pooled), len(pooled))
    s = f"{100 * np.mean(per):.1f}"
    if "-" not in runs and len(per) > 1:
        slo, shi = seed_ci(per)
        s += f" [{100 * slo:.1f}, {100 * shi:.1f}]"
    return s + f" ({100 * lo:.1f}–{100 * hi:.1f})"


def gap(runs):
    vals = [e["gap"] for rows in runs.values() for e in rows if e.get("gap") is not None]
    return f"{np.median(vals):.3f}" if vals else "-"


def main():
    order = {"B2 (full)": 0, "B3 (A*)": 1, "base-only": 2, "hybrid": 3, "offset hybrid (PE4b)": 4, "pure SAC+HER": 5}
    for exp, (test_set, title) in SETS.items():
        d = ROOT / "results" / "eval" / "2dof" / test_set
        if not d.exists():
            continue
        data = collect(d)
        n_eps = len(next(iter(next(iter(data.values())).values()))) if data else 0
        md = [f"# {exp.upper()}: {title}, {test_set} ({n_eps} episodes per seed)", "",
              "Success and collision in %: trained arms mean over seeds [95% bootstrap CI over seeds] (pooled Wilson "
              "95%, episode-level only); planner rows one deterministic evaluation (Wilson 95%). Gap = median "
              "L(executed) / L(best known) - 1 over successful episodes.", "",
              "| Method | Budget | Checkpoint | Seeds | Success | Collision | Gap |", "|---|---|---|---|---|---|---|"]
        for k in sorted(data, key=lambda k: (order.get(k[0], 9), k[1] or 0, k[2] == "final")):
            runs = data[k]
            md.append(f"| {k[0]} | {k[1] or '-'} | {k[2]} | {len(runs) if '-' not in runs else '-'} | "
                      f"{cell(runs, 'success')} | {cell(runs, 'collision')} | {gap(runs)} |")
        out = ROOT / "results" / "tables" / f"{exp}.md"
        out.write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))


if __name__ == "__main__":
    main()
