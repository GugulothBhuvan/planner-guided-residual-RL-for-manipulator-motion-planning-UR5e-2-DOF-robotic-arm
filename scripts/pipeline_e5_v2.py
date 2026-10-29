"""E5 rerun with the absorbing collision penalty (SPEC §6.7, §7.3, §11; decisions 2026-10-28).

  1. Wait for the 100k smoke test of pure SAC+HER on S4 with the fix (results/runs/e5_smoke_fix). Pass = validation
     collision rate at 100k <= 0.80 ("clearly below 100%") and mean validation episode length >= 50 steps.
     Writes results/runs/e5_smoke_fix/PASS or FAIL; on FAIL nothing else runs (the 2-DOF lane waits on the same
     marker before PE3/PE4).
  2. E5 from step 0, all 6 runs (pure SAC+HER and residual SAC with w_res 0.1, 50k warm-up, 150k alpha ramp),
     at most 2 UR5e jobs on the laptop (RAM), each start gated on 1.7 GB free.
  3. Evaluation on the 200-episode S4 test set: base@N and hybrid@N (N = 1500, 3000, 6000) for every residual seed,
     the pure SAC policy for every seed, then the budget figure (scripts/plot_budget.py); writes results/runs/e5/EVAL_DONE
     (the 2-DOF lane then runs PE3/PE4).
  4. Sparse-residual rule (SPEC §7.3): if residual s0's validation success at 500k is below 0.88, the extra arm
     configs/exp/e5_residual_sparse.yaml (3 seeds) runs, is evaluated the same way and the figure is redrawn with it
     as its own line.
  5. (E2 seed 1 ran on Colab; results/runs/e2_offloaded.)
Offloading (decision 2026-10-28 06:50; Chrome stays open, so RAM is short): with results/runs/e5_offloaded_s1s2 the
laptop runs only seed 0, one job at a time (residual first, then SAC+HER; --min-free-gb from measured peak RSS),
and seeds 1 and 2 run on Colab (notebooks/colab_e5_s1s2.ipynb); evaluation waits until all six runs are in
results/runs/e5. The sparse-residual rule is decided as soon as residual s0 has its 500k validation; if it
triggers, notebooks/colab_e5_sparse.ipynb is written for sparse s1 and s2 (Colab or Kaggle, marker
results/runs/e5_sparse_offloaded_s1s2) and sparse s0 runs on the laptop right after the dense seed-0 runs
(decision 2026-10-28 07:20). Laptop order: residual s0 -> SAC+HER s0 -> sparse residual s0 -> 2-DOF lane (PE8);
results/runs/UR5E_LANE_BUSY keeps the 2-DOF lane paused across these launches. Residual arms are evaluated at the
primary and the final checkpoint, with the alpha sweep at both (SPEC §9.2 checkpoint rule; results/figures/e5_alpha_sweep.*).
--decision-a: the smoke test failed by its pre-set threshold, and the user chose to launch the rerun anyway
(2026-10-28, option A). Residual validation at 200k and 500k is logged against the base-only line (0.88 success,
0.12 collision; results/runs/e5_v1_crash_exploit/base_reference.json) in results/runs/e5/residual_milestones.log.
No interventions: these runs are final.
Decisions go to results/runs/pipeline.log and results/runs/pipeline_decisions.json.

    python scripts/pipeline_e5_v2.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import HOLD  # noqa: E402
from pipeline_e1_e2 import RUNS, ROOT, log, record  # noqa: E402

SMOKE = RUNS / "e5_smoke_fix"
# own process group and no console: a console close or Ctrl+C (IDE terminal closed) reaches no job
DETACH = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
PY = sys.executable


def run(args):
    subprocess.run([PY, *map(str, args)], check=True, creationflags=DETACH)


def smoke_verdict():
    while not list(SMOKE.glob("e5/*/done.json")):
        time.sleep(60)
    done = json.loads(next(SMOKE.glob("e5/*/done.json")).read_text())
    v = done["final_val"]
    ok = v["collision"] <= 0.80 and v["ep_len"] >= 50
    log(f"E5 smoke (pure SAC+HER, fixed reward, {done['num_timesteps']} steps): collision {v['collision']:.2f}, "
        f"ep_len {v['ep_len']:.1f}, success {v['success']:.2f} -> {'PASS' if ok else 'FAIL'}")
    record("e5_smoke_fix", dict(final_val=v, passed=ok))
    (SMOKE / ("PASS" if ok else "FAIL")).write_text(json.dumps(v, indent=2))
    return ok


BASE_LINE = dict(success=0.88, collision=0.12)


MAIN_RESIDUAL = re.compile(r"^residual_[0-9a-f]{8}_s\d+$")  # the dense residual (not residual_sparse_...)


def milestones(stop):
    """Log each residual seed's validation at 200k and 500k against the base-only line (report only)."""
    out = RUNS / "e5" / "residual_milestones.log"
    seen = set()
    if out.exists():  # a restarted pipeline does not log the same milestone twice
        for line in out.read_text(encoding="utf-8").splitlines():
            name, _, rest = line.partition(" @ ")
            seen.add((name, int(rest.split("k:")[0]) * 1000))
    while not stop.is_set():
        for d in sorted((RUNS / "e5").glob("residual_*_s*")):  # dense and (if queued) sparse residuals
            f = d / "val.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                key = (d.name, r["num_timesteps"])
                if r["num_timesteps"] in (200_000, 500_000) and key not in seen:
                    seen.add(key)
                    msg = (f"{d.name} @ {r['num_timesteps'] // 1000}k: success {r['success']:.2f} "
                           f"(base {BASE_LINE['success']:.2f}), collision {r['collision']:.2f} "
                           f"(base {BASE_LINE['collision']:.2f})")
                    with open(out, "a", encoding="utf-8") as fh:
                        fh.write(msg + "\n")
                    log(msg)
        stop.wait(120)


EVAL = ROOT / "results" / "eval" / "s4_test"
BUDGETS = (1500, 3000, 6000)


def eval_tasks(d):
    """The evaluation commands for one finished run that have no output yet (so a restarted pipeline resumes)."""
    tasks = []
    if d.name.startswith("residual"):
        for ckpt in ("primary", "final"):
            if not all((EVAL / f"hybrid_{b}__{d.name}__{ckpt}.jsonl").exists() for b in BUDGETS):
                tasks.append([ROOT / "scripts" / "evaluate.py", "--set", "s4_test", "--budgets", *BUDGETS,
                              "--skip-base", "--checkpoint", ckpt, "--run", d])
            if not (d / f"alpha_sweep_{ckpt}.json").exists():
                tasks.append([ROOT / "scripts" / "alpha_sweep.py", "--run", d, "--checkpoint", ckpt])
    elif not (EVAL / f"policy__{d.name}.jsonl").exists():
        tasks.append([ROOT / "scripts" / "evaluate.py", "--set", "s4_test", "--methods", "policy", "--run", d])
    return tasks


def evaluate_runs(select, workers=2, exp="e5"):
    """S4 test set. Residual arms: hybrid@N at the primary and the final checkpoint (SPEC §9.2 checkpoint rule; base@N
    and b2 are run-independent and already evaluated), plus the validation alpha sweep at both checkpoints. Runs are
    spread over `workers` parallel streams (evaluate.py locks summary.json); finished outputs are skipped."""
    from concurrent.futures import ThreadPoolExecutor
    runs = [d for d in sorted((RUNS / exp).glob("*_s*")) if (d / "done.json").exists() and select(d.name)]
    todo = [t for d in runs for t in eval_tasks(d)]
    log(f"evaluating {len(runs)} runs: {len(todo)} evaluation commands left, {workers} in parallel")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, todo))  # re-raises the first failure


def finished(select):
    return [d for d in (RUNS / "e5").glob("*_s*") if select(d.name) and (d / "done.json").exists()]


def wait_finished(select, n, what):
    logged = -1
    while (k := len(finished(select))) < n:
        if k != logged:
            log(f"waiting for {what}: {k}/{n} finished in results/runs/e5")
            logged = k
        time.sleep(300)


def s0_success_at_500k():
    s0 = next((d for d in (RUNS / "e5").glob("residual_*_s0") if MAIN_RESIDUAL.match(d.name)), None)
    rows = [json.loads(l) for l in (s0 / "val.jsonl").read_text().splitlines()] if s0 and (s0 / "val.jsonl").exists() else []
    return next((r["success"] for r in rows if r["num_timesteps"] == 500_000), None)


def sparse_rule(result):
    """SPEC §7.3: decided as soon as residual s0 has its 500k validation (not after all six runs)."""
    while (at500 := s0_success_at_500k()) is None:
        time.sleep(120)
    triggered = at500 < BASE_LINE["success"]
    log(f"sparse-residual rule: residual s0 at 500k = {at500} vs {BASE_LINE['success']} -> "
        f"{'queue the sparse-residual arm (Colab, after the dense seeds)' if triggered else 'no extra arm'}")
    record("e5_sparse_residual_rule", dict(s0_success_500k=at500, threshold=BASE_LINE["success"], triggered=triggered))
    if triggered:
        run([ROOT / "scripts" / "make_colab_notebook.py", "e5_sparse"])
        log("wrote notebooks/colab_e5_sparse.ipynb (sparse s1, s2; Colab or Kaggle). Sparse s0 runs on the laptop")
    result["triggered"] = triggered


def main():
    import argparse
    import threading
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-a", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=1.7, help="RAM gate for starting a laptop E5 job")
    args = ap.parse_args()
    if args.decision_a:
        log("smoke test FAIL overridden by the user (option A, 2026-10-28): launching the E5 rerun")
        record("e5_rerun", dict(smoke="FAIL", decision="A: launch anyway"))
        (SMOKE / "DECISION_A").write_text("launch the rerun despite the smoke FAIL (user, 2026-10-28)")
    elif not smoke_verdict():
        log("smoke test failed: E5 rerun NOT launched; waiting for a decision")
        return
    (RUNS / "e5").mkdir(exist_ok=True)
    stop, rule = threading.Event(), {}
    threading.Thread(target=milestones, args=(stop,), daemon=True).start()
    rule_thread = threading.Thread(target=sparse_rule, args=(rule,), daemon=True)
    rule_thread.start()
    HOLD.write_text(str(os.getpid()))  # the 2-DOF lane stays paused across this pipeline's laptop launches
    launch = [ROOT / "scripts" / "launch.py", "--exp", ROOT / "configs" / "exp" / "e5.yaml", "--threads", 2,
              "--min-free-gb", args.min_free_gb]
    if (RUNS / "e5_offloaded_s1s2").exists():
        log(f"launch e5.yaml seed 0 only (s1/s2 on Colab), 1 job at a time, residual first; RAM gate {args.min_free_gb} GB")
        run(launch + ["--seeds", 0, "--runs", "residual", "sac_her", "--workers", 1, "--cap-ur5e", 1])
    else:
        log("launch e5.yaml (rerun, fixed reward; cap 2 UR5e jobs)")
        run(launch + ["--workers", 2, "--cap-ur5e", 2])
    rule_thread.join()  # decided at residual s0's 500k validation, long before its seed-0 runs finish
    # Order from 2026-10-29 (user: evaluate as soon as the cloud runs are in): the dense evaluation goes before the
    # rest of sparse s0, which needs 1.4 GB the evaluation would otherwise wait for. The hold is released meanwhile,
    # so the 2-DOF lane (PE3/PE4) runs beside the evaluation.
    HOLD.unlink(missing_ok=True)
    main_runs = lambda n: n.startswith("sac_her_") or MAIN_RESIDUAL.match(n)
    wait_finished(main_runs, 6, "the six dense E5 runs (seeds 1 and 2 come from Colab)")
    if not (RUNS / "e5" / "EVAL_DONE").exists():
        log("E5 rerun finished; evaluating on s4_test")
        evaluate_runs(main_runs)
        run([ROOT / "scripts" / "plot_budget.py", "--set", "s4_test"])
        run([ROOT / "scripts" / "plot_alpha_sweep.py"])
        run([ROOT / "scripts" / "e5_table.py"])
        (RUNS / "e5" / "EVAL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        log("E5 evaluation and budget figure done: results/figures/e5_budget_s4_test.*, results/tables/e5.md")

    if rule["triggered"]:
        local = [0] if (RUNS / "e5_sparse_offloaded_s1s2").exists() else [0, 1, 2]
        sparse = lambda n: n.startswith("residual_sparse_")
        if len(finished(sparse)) < 3:  # a launcher waiting for RAM signals the 2-DOF lane itself (launch.py WAIT)
            log(f"launch e5_residual_sparse.yaml seeds {local} on the laptop (others on Colab/Kaggle)")
            run([ROOT / "scripts" / "launch.py", "--exp", ROOT / "configs" / "exp" / "e5_residual_sparse.yaml",
                 "--seeds", *local, "--workers", 1, "--threads", 2, "--min-free-gb", args.min_free_gb, "--cap-ur5e", 1])
            HOLD.unlink(missing_ok=True)
        wait_finished(sparse, 3, "the sparse-residual arm (s1, s2 from Colab/Kaggle)")
        log("sparse-residual arm finished; evaluating on s4_test")
        evaluate_runs(sparse)
        run([ROOT / "scripts" / "plot_budget.py", "--set", "s4_test"])
        run([ROOT / "scripts" / "plot_alpha_sweep.py"])
        run([ROOT / "scripts" / "e5_table.py"])
        log("figures redrawn with the sparse residual as its own line")
    HOLD.unlink(missing_ok=True)
    (RUNS / "e5" / "PIPELINE_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("E5 pipeline finished; the laptop UR5e lane is free")
    stop.set()


if __name__ == "__main__":
    main()
