"""Section 4 of the report rewrite: yellow placeholder subsections with empty table layouts (no numbers, no conclusions).
6.6.1 residual size on the MyCobot (diagnostic); 6.9.8 basic reach with one obstacle, simulation and real arm. Idempotent."""
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "report.html"
s = open(p, encoding="utf-8").read()
if 'id="s6-6-1"' in s:
    print("already patched")
    sys.exit(0)

D = "&mdash;"


def table(tid, cap, head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "\n      ".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'  <div class="tbl" id="{tid}"><div class="cap"><b>{cap[0]}</b> {cap[1]}</div><div class="scroll"><table>\n'
            f"    <thead><tr>{th}</tr></thead>\n    <tbody>\n      {tr}\n    </tbody></table></div></div>\n")


# ---------------------------------------------------------------- 6.6.1
t1 = table("tab-6-7a", ("Table 6.7a:", "Size of the residual action per joint on the MyCobot, as a fraction of its bound &alpha;&Delta;<var>q</var><sub>max</sub> = 0.0026 rad (to be filled)."),
           ["Seed", "Budget", "Mean J1", "J2", "J3", "J4", "J5", "J6", "Largest value (any step, any joint)"],
           [[sd, b] + [D] * 7 for sd in ("0", "1") for b in ("1500", "3000", "6000")])
t2 = table("tab-6-7b", ("Table 6.7b:", "Largest deviation of the joint target from the planned path per episode, <var>dev</var><sub>max</sub> (rad), on the 200 <code>s4_test</code> episodes (to be filled)."),
           ["Controller", "Budget", "p5", "median", "p95", "max"],
           [[c, b] + [D] * 4 for c in ("planner alone", "hybrid, seed 0", "hybrid, seed 1") for b in ("1500", "3000", "6000")])
t3 = table("tab-6-7c", ("Table 6.7c:", "Episodes whose outcome differs from the planner alone (to be filled)."),
           ["Seed", "Budget", "Different outcome", "Better (fail to success)", "Worse (success to fail)"],
           [[sd, b] + [D] * 3 for sd in ("0", "1") for b in ("1500", "3000", "6000")])
sec1 = ('  <h3 id="s6-6-1">6.6.1&nbsp;&nbsp;Size of the residual on the MyCobot (diagnostic)</h3>\n'
        '  <div class="todo-block"><strong class="tag">Pending</strong>Placeholder for the residual-size diagnostic on the MyCobot: how large the residual\'s correction is per joint, how far the joint target leaves the planned path, and in how many episodes the outcome differs from the planner alone. The measurement uses the 200 <code>s4_test</code> episodes, both seeds at the final checkpoint and all three budgets. No result is stated here; the tables below show the layout to be filled and a verdict of two or three sentences goes under them.</div>\n'
        + t1 + t2 + t3 +
        '  <p><em>Verdict (to be written once the tables are filled).</em></p>\n')

# ---------------------------------------------------------------- 6.9.8
t4 = table("tab-6-12c", ("Table 6.12c:", "Basic reach with one virtual obstacle in simulation, on 100 frozen test episodes per stage (to be filled)."),
           ["Stage", "Controller", "Success", "Collision", "Final error median / max (cm)", "Steps median"],
           [[st, c] + [D] * 4 for st in ("Reach, no obstacle", "Reach, one sphere") for c in ("scripted straight line", "scripted detour", "learned controller, seed 0", "learned controller, seed 1", "learned controller, seed 2")])
t5 = table("tab-6-12d", ("Table 6.12d:", "Basic reach on the real arm, streamed closed-loop at 20 Hz and 0.5&deg; per tick, with virtual obstacles, first 20 test episodes (to be filled)."),
           ["Controller", "Runs", "Success", "Collisions", "Final error median / max (cm)", "Wall time median (s)"],
           [[c] + [D] * 5 for c in ("learned controller (best seed by validation)", "scripted straight line", "scripted detour")])
sec2 = ('  <h3 id="s6-9-8">6.9.8&nbsp;&nbsp;Basic reach with one obstacle: simulation and real arm</h3>\n'
        '  <div class="todo-block"><strong class="tag">Pending</strong>Placeholder for a basic reach experiment on the MyCobot: a learned position-reaching controller against scripted rules (a straight line and a hand-coded detour) on the same test episodes, first in simulation and then on the real arm with a virtual obstacle. No result is stated here. Section 6.9.7 describes the attempt that was made so far and why it is not a result. The layouts below are to be filled, and the controllers would run in streaming mode so that the learned policy is the one tested on hardware.</div>\n'
        + t4 + t5)

marker = '  <h2 id="s6-7">'
assert marker in s
s = s.replace(marker, sec1 + marker, 1)
marker = '  <h2 id="s6-10">'
assert marker in s
s = s.replace(marker, sec2 + marker, 1)
open(p, "w", encoding="utf-8").write(s)
print("placeholders added")
