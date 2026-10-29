"""E5b (SPEC §7.3, §8; pre-registered 2026-10-29): integrated-offset residual, 3 seeds x 1M on S4, then the same
S4 test-set evaluation as E5 (primary and final checkpoint, alpha sweeps at both; SPEC §9.2), the budget and
alpha-sweep figures and the headline table redrawn with E5b as its own line. Reported whatever it shows.

Laptop: seed 0 (with results/runs/e5b_offloaded_s1s2, seeds 1-2 run on Kaggle, notebooks/colab_e5b_s1s2.ipynb).
At most 2 UR5e jobs on the laptop (E5b s0 may start beside sparse s0), each start gated on 1.4 GB free RAM (the
measured E5 peak, so it cannot page). Writes results/runs/e5b/PIPELINE_DONE.

    python scripts/pipeline_e5b.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_e5_v2 import ROOT, RUNS, evaluate_runs, log, run  # noqa: E402


def main():
    local = [0] if (RUNS / "e5b_offloaded_s1s2").exists() else [0, 1, 2]
    log(f"E5b: launch seeds {local} on the laptop (at most 2 UR5e jobs, RAM gate 1.4 GB)")
    run([ROOT / "scripts" / "launch.py", "--exp", ROOT / "configs" / "exp" / "e5b.yaml", "--seeds", *local,
         "--workers", 1, "--threads", 2, "--min-free-gb", 1.4, "--cap-ur5e", 2])
    offset = lambda n: n.startswith("residual_offset_")
    log("E5b: waiting for all 3 seeds (seeds 1-2 from Kaggle)")
    while len([d for d in (RUNS / "e5b").glob("*_s*") if offset(d.name) and (d / "done.json").exists()]) < 3:
        time.sleep(300)
    log("E5b finished; evaluating on s4_test")
    evaluate_runs(offset, exp="e5b")
    for script in ("plot_budget.py", "plot_alpha_sweep.py", "e5_table.py"):
        run([ROOT / "scripts" / script] + (["--set", "s4_test"] if script == "plot_budget.py" else []))
    (RUNS / "e5b" / "PIPELINE_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("E5b evaluated: results/tables/e5.md and the E5 figures now include the offset residual")


if __name__ == "__main__":
    main()
