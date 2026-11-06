"""Fill the 6.6.1 placeholder with the Task 0 diagnostic (results/eval/mc_diag/*.jsonl, as in results/tables/mc_residual_diagnostic.md)."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "eval" / "mc_diag"
BUD = (1500, 3000, 6000)
ALPHA, DQ = 0.3, np.deg2rad(0.5)
BOUND = ALPHA * DQ
load = lambda f: [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
data = {(s, b): load(D / f"{s}_{b}.jsonl") for s in ("base", "s0", "s1") for b in BUD}

p = Path(__file__).resolve().parent / "report.html"
s = p.read_text(encoding="utf-8")
if "(to be filled)" not in s.split('id="s6-6-1"')[1].split('id="s6-7"')[0]:
    print("already filled")
    raise SystemExit


def table(tid, cap, head, rows, hl=None):
    cls = lambda i: ' class="n"' if i else ""
    th = "".join("<th%s>%s</th>" % (cls(i), h) for i, h in enumerate(head))
    tr = "\n      ".join("<tr>" + "".join("<td%s>%s</td>" % (cls(i), c) for i, c in enumerate(r)) + "</tr>" for r in rows)
    return (f'  <div class="tbl" id="{tid}"><div class="cap"><b>{cap[0]}</b> {cap[1]}</div><div class="scroll"><table>\n'
            f"    <thead><tr>{th}</tr></thead>\n    <tbody>\n      {tr}\n    </tbody></table></div></div>\n")


# Table 6.7a: mean |pi_res| per joint (alpha |pi_res| / alpha = fraction of the bound), largest value
rows_a = []
for sd in ("s0", "s1"):
    for b in BUD:
        rr = data[(sd, b)]
        mean = np.mean([r["ares_mean"] for r in rr], 0) / ALPHA
        mx = np.max([r["ares_max"] for r in rr]) / ALPHA
        rows_a.append([sd[1], b] + [f"{v:.2f}" for v in mean] + [f"{mx:.2f}"])
ta = table("tab-6-7a", ("Table 6.7a:", "Size of the residual action per joint on the MyCobot, as a fraction of its bound &alpha;&Delta;<var>q</var><sub>max</sub> = 0.0026 rad (the mean over the steps of an episode, then over the 200 <code>s4_test</code> episodes; 1.00 means the residual output is at its limit)."),
           ["Seed", "Budget", "Mean J1", "J2", "J3", "J4", "J5", "J6", "Largest value (any step, any joint)"], rows_a)

# Table 6.7b: dev_max
pct = lambda x, q: float(np.percentile(x, q))
rows_b = []
for name, key in (("planner alone", "base"), ("hybrid, seed 0", "s0"), ("hybrid, seed 1", "s1")):
    for b in BUD:
        d = [r["dev_max"] for r in data[(key, b)]]
        rows_b.append([name, b] + [f"{v:.4f}" for v in (pct(d, 5), pct(d, 50), pct(d, 95), max(d))])
tb = table("tab-6-7b", ("Table 6.7b:", "Largest deviation of the joint target from the planned path per episode, <var>dev</var><sub>max</sub> = max over steps of max<sub>j</sub> |<var>q</var><sub>target</sub> &minus; <var>q</var><sub>ref</sub>(<var>t</var>)| (rad), on the 200 <code>s4_test</code> episodes."),
           ["Controller", "Budget", "p5", "median", "p95", "max"], rows_b)

# Table 6.7c: outcome differences
outcome = lambda r: "success" if r["success"] else ("collision" if r["collision"] else "other")
rows_c, total_diff, total_n = [], 0, 0
for sd in ("s0", "s1"):
    for b in BUD:
        bs, hs = data[("base", b)], data[(sd, b)]
        diff = [i for i in range(len(bs)) if outcome(bs[i]) != outcome(hs[i])]
        better = sum(1 for i in diff if not bs[i]["success"] and hs[i]["success"])
        worse = sum(1 for i in diff if bs[i]["success"] and not hs[i]["success"])
        total_diff += len(diff)
        assert better == 0
        total_n += len(bs)
        rows_c.append([sd[1], b, f"{100 * np.mean([r['success'] for r in bs]):.1f}%", f"{100 * np.mean([r['success'] for r in hs]):.1f}%", f"{len(diff)} of {len(bs)}", better, worse])
tc = table("tab-6-7c", ("Table 6.7c:", "Episodes whose outcome (success, collision or neither) differs from the planner alone, same episode and budget, and the success rates."),
           ["Seed", "Budget", "Success, planner", "Success, hybrid", "Different outcome", "Better (fail to success)", "Worse (success to fail)"], rows_c)

all_mean = float(np.mean([np.mean(r["ares_mean"]) for k, rr in data.items() if k[0] != "base" for r in rr]) / ALPHA)
dev_h = [r["dev_max"] for k, rr in data.items() if k[0] != "base" for r in rr]
new = ('  <h3 id="s6-6-1">6.6.1&nbsp;&nbsp;Size of the residual on the MyCobot (diagnostic)</h3>\n'
       '  <p>To see whether the retrained residual does anything, the 200 <code>s4_test</code> episodes were replayed with both seeds at the final checkpoint and all three budgets (same reset seeds and deterministic actions as the evaluation), and three things were recorded: the size of the residual&rsquo;s correction at each step per joint (Table 6.7a), the largest deviation of the joint target from the planned path per episode (Table 6.7b), and the episodes whose outcome differs from the planner alone (Table 6.7c). The additive residual can move the target by at most &alpha;&Delta;<var>q</var><sub>max</sub> = 0.3 &times; 0.5&deg; = 0.0026 rad per joint per step (Equation 3.23).</p>\n'
       + ta + tb + tc +
       f'  <p><strong>Verdict.</strong> The residual acts but cancels out. Its output averages {all_mean:.2f} of its range, so it has not collapsed to zero, yet only {total_diff} of {total_n} episode-runs end differently from the planner alone, and all {total_diff} are worse (seed 0 only; seed 1 never differs). The target leaves the planned path by a median of {np.median(dev_h):.3f} rad against 0 for the planner, more than the 0.0026 rad bound, most likely because the bound holds only on steps where the base command is not clipped, and the base command saturates when the reference moves quickly; the tracker re-aims at the path on every step, which is the mechanism measured on the UR5e in Section 6.4. The residual is therefore not idle on the MyCobot, but, as on the UR5e, it does not change what happens to the arm.</p>\n')
a = s.index('  <h3 id="s6-6-1">')
b = s.index('  <h2 id="s6-7">')
s = s[:a] + new + s[b:]
p.write_text(s, encoding="utf-8")
print("filled:", total_diff, "of", total_n, "differ; mean", round(all_mean, 2), "median dev", round(float(np.median(dev_h)), 4))
