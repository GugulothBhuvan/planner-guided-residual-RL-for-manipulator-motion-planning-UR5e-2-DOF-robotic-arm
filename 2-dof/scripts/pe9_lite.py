"""PE9-lite (companion SPEC §8, P-RQ5): do the design rankings found on the planar arm hold on the UR5e?

Each design choice is compared on both arms with the same metric (final validation success for the training-side
choices, S4 / P2-P3 test success for the planner + residual question). A pairwise ranking "agrees" when both arms put
the same option first. Each side also says whether its own difference is clear: the 95% bootstrap CI over seeds of
the difference excludes 0 (with 1-3 UR5e seeds this is rarely possible, so "agrees" is about direction only).

  1. HER agent (PE1 vs E1): SAC vs TD3 (DDPG ran on the planar arm only).
  2. Reward (PE2 vs E2): sparse + HER vs dense. The UR5e E2 runs use the goal-distance curriculum (the S1 probe
     winner), so the planar pair is the curriculum one too (PE1 sac_her vs PE2 sac_dense_curr).
  3. Goal-distance curriculum (PE2 vs the S1 probe): on vs off with SAC+HER. The UR5e side is the single-seed
     120k-step probe (a: gamma 0.98 + curriculum vs c: gamma 0.98 alone); stated as weak evidence.
  4. Planner + residual (PE3 / PE4 vs E5): hybrid@N vs base-only@N at the middle budget, and hybrid vs pure SAC+HER
     (primary checkpoint).
  5. Does the redesign transfer (PE4b vs PE4, E5b vs E5; added 2026-10-29): offset hybrid vs additive hybrid at the
     middle budget (primary checkpoint), and offset hybrid vs base-only.

    python 2-dof/scripts/pe9_lite.py   ->  results/tables/pe9_lite.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"
EVAL = ROOT / "results" / "eval"


def final_success(exp, prefix):
    """Per-seed final validation success of the runs of `exp` whose name starts with `prefix` (no hash)."""
    out = []
    for d in sorted((RUNS / exp).glob(f"{prefix}_*_s*")):
        if re.fullmatch(rf"{prefix}_[0-9a-f]{{8}}_s\d+", d.name) and (d / "done.json").exists():
            out.append(json.loads((d / "done.json").read_text())["final_val"]["success"])
    return out


def test_success(folder, stem_re):
    """Per-seed (or single) test success from evaluation rows whose file stem matches `stem_re`."""
    out = []
    for f in sorted(folder.glob("*.jsonl")) if folder.exists() else []:
        if re.fullmatch(stem_re, f.stem):
            rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            out.append(float(np.mean([r["success"] for r in rows])))
    return out


def diff_ci(a, b, n=10_000, seed=0):
    """Mean(a) - mean(b) and a 95% bootstrap CI over seeds (None with a single value on either side)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(a.mean() - b.mean())
    if len(a) < 2 and len(b) < 2:
        return d, None
    rng = np.random.default_rng(seed)
    boot = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return d, (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def side(a, b, b_fixed=False):
    """b_fixed: b is a deterministic planner row (one evaluation, no seed variance), so the seed count applies to a
    only (companion SPEC v0.12)."""
    if not a or not b:
        return None
    d, ci = diff_ci(a, b)
    # a bootstrap over 1-2 seeds cannot support "clear" (UR5e SPEC §9.3 seed-count caveat): needs >= 3 per trained side
    clear = ci is not None and len(a) >= 3 and (b_fixed or len(b) >= 3) and (ci[0] > 0 or ci[1] < 0)
    txt = f"{100 * np.mean(a):.1f} vs {100 * np.mean(b):.1f} (Δ {100 * d:+.1f}"
    txt += f", CI [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}])" if ci else ", no CI)"
    return d, clear, txt, f"{len(a)}/{len(b)}"


def main():
    p3 = EVAL / "2dof"
    comparisons = [
        ("HER agent: SAC vs TD3", "PE1 (P1)", side(final_success("pe1_v2", "sac_her"), final_success("pe1_v2", "td3_her")),
         "E1 (S1)", side(final_success("e1", "sac"), final_success("e1", "td3"))),
        ("Reward: sparse+HER vs dense (curriculum on)", "PE1/PE2 (P1)",
         side(final_success("pe1_v2", "sac_her"), final_success("pe2", "sac_dense_curr")),
         "E2 (S2)", side(final_success("e2", "sac_sparse_her"), final_success("e2", "sac_dense"))),
        ("Curriculum on vs off (SAC+HER)", "PE1/PE2 (P1)",
         side(final_success("pe1_v2", "sac_her"), final_success("pe2", "sac_her_nocurr")),
         "S1 probe (1 seed, 120k)", side(final_success("probe_s1", "a_g098_curr"), final_success("probe_s1", "c_g098"))),
    ]
    for exp, test_set, mid in (("PE3", "p2_test", 6400), ("PE4", "p3_test", 2300)):
        f = p3 / test_set
        hyb = test_success(f, rf"hybrid_{mid}__residual_[0-9a-f]{{8}}_s\d+__primary")
        comparisons += [
            (f"Hybrid vs base-only, middle budget", f"{exp} ({test_set}, {mid})", side(hyb, test_success(f, f"base_{mid}"), b_fixed=True),
             "E5 (S4, 3000)", side(test_success(EVAL / "s4_test", r"hybrid_3000__residual_[0-9a-f]{8}_s\d+__primary"),
                                   test_success(EVAL / "s4_test", "base_3000"), b_fixed=True)),
            (f"Hybrid vs pure SAC+HER", f"{exp} ({test_set}, {mid})",
             side(hyb, test_success(f, r"policy__sac_her_[0-9a-f]{8}_s\d+")),
             "E5 (S4, 3000)", side(test_success(EVAL / "s4_test", r"hybrid_3000__residual_[0-9a-f]{8}_s\d+__primary"),
                                   test_success(EVAL / "s4_test", r"policy__sac_her_[0-9a-f]{8}_s\d+"))),
        ]
    f3, s4 = p3 / "p3_test", EVAL / "s4_test"
    off_p = test_success(f3, r"hybrid_2300__residual_offset_[0-9a-f]{8}_s\d+__primary")
    off_u = test_success(s4, r"hybrid_3000__residual_offset_[0-9a-f]{8}_s\d+__primary")
    comparisons += [
        ("Redesign: offset vs additive residual", "PE4b vs PE4 (p3_test, 2300)",
         side(off_p, test_success(f3, r"hybrid_2300__residual_[0-9a-f]{8}_s\d+__primary")),
         "E5b vs E5 (S4, 3000)", side(off_u, test_success(s4, r"hybrid_3000__residual_[0-9a-f]{8}_s\d+__primary"))),
        ("Redesign: offset hybrid vs base-only", "PE4b (p3_test, 2300)", side(off_p, test_success(f3, "base_2300"), b_fixed=True),
         "E5b (S4, 3000)", side(off_u, test_success(s4, "base_3000"), b_fixed=True)),
    ]
    md = ["# PE9-lite: do the planar-arm design rankings transfer to the UR5e?", "",
          "Success in % (first option vs second), Δ = first − second with a 95% bootstrap CI over seeds. "
          "\"Clear\" = that CI excludes 0 with >= 3 seeds on each trained side (a planner row is one deterministic evaluation, a fixed value). \"Agree\" = both arms rank the same option first (direction only).", "",
          "| Design choice | 2-DOF | 2-DOF result | Clear | UR5e | UR5e result | Clear | Agree |",
          "|---|---|---|---|---|---|---|---|"]
    agree_n = total = 0
    for name, p_name, p, u_name, u in comparisons:
        if p is None or u is None:
            md.append(f"| {name} | {p_name} | {'-' if p is None else p[2]} | | {u_name} | "
                      f"{'-' if u is None else u[2]} | | pending |")
            continue
        same = np.sign(p[0]) == np.sign(u[0]) and p[0] != 0 and u[0] != 0
        agree_n, total = agree_n + same, total + 1
        md.append(f"| {name} | {p_name} | {p[2]} | {'yes' if p[1] else 'no'} | {u_name} | {u[2]} | "
                  f"{'yes' if u[1] else 'no'} | {'yes' if same else 'no'} |")
    md += ["", f"Rankings agree on {agree_n} of {total} comparisons with results on both arms."]
    out = ROOT / "results" / "tables" / "pe9_lite.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
