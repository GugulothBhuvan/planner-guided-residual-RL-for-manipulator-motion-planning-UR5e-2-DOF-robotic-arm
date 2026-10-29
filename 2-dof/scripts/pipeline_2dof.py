"""2-DOF lane chain (companion SPEC §8): PE1 -> PE2 (PE1 winner) -> PE3 -> PE4 -> PE8.

Decisions are logged to results/runs/pipeline_2dof.log and results/runs/pipeline_decisions.json:
  1. Run PE1 (configs/exp/pe1_v2.yaml; jobs already running under an earlier launcher are skipped) and wait for
     all its runs.
  2. PE2 agent = the HER agent (DDPG/TD3/SAC) with the higher mean final validation success over PE1 seeds; ties
     within 0.03 go to the lower final position error (the E1 -> E2 rule). PPO-dense is excluded: PE2 needs
     the same agent with and without HER. configs/exp/pe2.yaml is rewritten for that agent.
  3. PE2, then PE3 and PE4 (SAC, mirroring E5; 5 seeds, the spec minimum), 1 torch thread each. Up to 2 workers,
     but none while any UR5e job runs or waits to start (--yield-to ur5e --yield-workers 0; decision 2026-10-28), and each start waits for 1.0 GB free RAM (MIN_FREE_GB), so the
     2-DOF lane never takes the room an E5 run needs (1.7 GB): the UR5e lane has priority (SPEC §11).
  4. (2026-10-29) PE3, PE4 and PE8 start right away, beside the E5 evaluation; with the offload markers the laptop
     runs seeds 0-1 and Kaggle seeds 2-4 (local_seeds). Once every run of all three is in results/runs: PE3/PE4
     test-set evaluation (B2, B3, base@N, hybrid@N at the primary and final checkpoint, pure SAC+HER), tables,
     the PE8 evaluation on the exact and +-20% mass plants (eval_pe8.py) and PE9-lite (pe9_lite.py); then
     results/runs/PIPELINE_2DOF_DONE.

    python 2-dof/scripts/pipeline_2dof.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_e1_e2 import RUNS, finished_runs, pick_algorithm, record  # noqa: E402

EXP = ROOT / "2-dof" / "configs" / "exp"
LOG = RUNS / "pipeline_2dof.log"
MIN_FREE_GB = 1.0  # 2 E5 runs (~1.7 GB each at full buffer) leave room for one ~0.35 GB 2-DOF job (2026-10-27)
# 2026-10-29: the lane runs beside UR5e jobs again (user: PE3/PE4 in parallel with the E5 evaluation); the RAM gate
# still keeps it out of memory a UR5e job needs. It was 0 (fully paused) while E5 training starts were pending.
YIELD_WORKERS = 2


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def n_jobs(exp_path):
    exp = yaml.safe_load(exp_path.read_text(encoding="utf-8"))
    return exp["name"], len(exp["runs"]) * len(exp["seeds"])


def complete(exp_path):
    name, n = n_jobs(exp_path)
    return len(finished_runs(name)) >= n


def local_seeds(exp_path):
    """Seeds this laptop runs: with results/runs/<exp>_offloaded_s234 (PE3/PE4: pe34_offloaded_s234), seeds 2-4 run on
    Kaggle (notebooks/colab_pe_s234.ipynb) and only 0-1 here (decision 2026-10-29)."""
    name = exp_path.stem
    marker = RUNS / ("pe34_offloaded_s234" if name in ("pe3", "pe4") else f"{name}_offloaded_s234")
    return [0, 1] if marker.exists() else None


def launch(exp_path, stop_file=None, seeds=None):
    if complete(exp_path):
        return
    log(f"launch {exp_path.name}" + (f" seeds {seeds}" if seeds else "") + (f" (until {stop_file.name} exists)" if stop_file else ""))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "launch.py"), "--exp", str(exp_path), "--workers", "2",
                    "--threads", "1", "--min-free-gb", str(MIN_FREE_GB), "--yield-to", "ur5e",
                    "--yield-workers", str(YIELD_WORKERS)] + (["--stop-file", str(stop_file)] if stop_file else [])
                   + (["--seeds", *map(str, seeds)] if seeds else []), check=True, creationflags=DETACH)
    log(f"{exp_path.name} finished" + (f" for seeds {seeds}" if seeds else ""))


DETACH = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
EVAL = ROOT / "results" / "eval" / "2dof"
PE34 = {"pe3": ("p2_test", (4000, 6400, 9000)), "pe4": ("p3_test", (700, 2300, 3500)),
        "pe4b": ("p3_test", (700, 2300, 3500))}  # budgets as in pe3/pe4/pe4b.yaml


def run(args):
    subprocess.run([sys.executable, *map(str, args)], check=True, creationflags=DETACH)


def wait_complete(exp_path):
    name, n = n_jobs(exp_path)
    logged = -1
    while (k := len(finished_runs(name))) < n:
        if k != logged:
            log(f"waiting for {name}: {k}/{n} runs finished (offloaded seeds arrive from Kaggle)")
            logged = k
        time.sleep(300)


def evaluate_pe34(exp, workers=4):
    """Frozen test set: B2, B3 and base@N once; every residual seed at the primary and final checkpoint (UR5e SPEC
    §9.2 rule), every SAC+HER seed's best-validation policy. Finished outputs are skipped (restartable); the
    commands run in `workers` parallel streams (evaluate.py locks summary.json; 2026-10-29: serially the 1000-episode
    sets took hours)."""
    from concurrent.futures import ThreadPoolExecutor
    test_set, budgets = PE34[exp]
    ev, out = ROOT / "2-dof" / "scripts" / "evaluate.py", EVAL / test_set
    tasks = []
    if not all((out / f"{m}.jsonl").exists() for m in ["b2", "b3"] + [f"base_{b}" for b in budgets]):
        tasks.append([ev, "--set", test_set, "--methods", "b2", "b3", "--budgets", *budgets])
    for d in sorted((RUNS / exp).glob("*_s*")):
        if not (d / "done.json").exists():
            continue
        if d.name.startswith("residual"):
            for ckpt in ("primary", "final"):
                if not all((out / f"hybrid_{b}__{d.name}__{ckpt}.jsonl").exists() for b in budgets):
                    tasks.append([ev, "--set", test_set, "--budgets", *budgets, "--skip-base", "--checkpoint", ckpt,
                                  "--run", d])
        elif not (out / f"policy__{d.name}.jsonl").exists():
            tasks.append([ev, "--set", test_set, "--methods", "policy", "--run", d])
    log(f"{exp}: {len(tasks)} evaluation commands, {workers} in parallel")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, tasks))


def main():
    name, n = n_jobs(EXP / "pe1_v2.yaml")
    launch(EXP / "pe1_v2.yaml")
    while len(finished_runs(name)) < n:  # jobs skipped as already running finish under their old launcher
        time.sleep(60)

    per_algo = {}
    for r in finished_runs(name).values():
        c = r["config"]["config"]
        if c["her"]:
            per_algo.setdefault(c["agent"]["algo"], []).append((r["final_val"]["success"], r["final_val"]["d_pos_cm"]))
    algo, means = pick_algorithm(per_algo)
    log(f"PE1 means (success, d_pos cm) {means}; PE2 agent {algo}")
    record("pe1", dict(means=means, pe2_agent=algo))

    pe2 = yaml.safe_load((EXP / "pe2.yaml").read_text(encoding="utf-8"))
    for run in pe2["runs"]:
        run["name"] = run["name"].replace("sac", algo, 1)
        run["agent"] = algo
        if not run.get("her"):
            run.pop("overrides", None)
    (EXP / "pe2.yaml").write_text(f"# Written by 2-dof/scripts/pipeline_2dof.py for the PE1 winner ({algo}); see the\n"
                                  "# companion SPEC §8. The sparse+HER / curriculum-on cell is PE1's run.\n"
                                  + yaml.safe_dump(pe2, sort_keys=False), encoding="utf-8")
    launch(EXP / "pe2.yaml")
    # PE3/PE4 (absorbing collision penalty, UR5e SPEC §6.7) start now, beside the E5 evaluation (user, 2026-10-29;
    # until then they waited for EVAL_DONE), then the laptop's PE8 seeds. Seeds 2-4 of all three run on Kaggle.
    for exp in ("pe3.yaml", "pe4.yaml", "pe8.yaml"):
        launch(EXP / exp, seeds=local_seeds(EXP / exp))
    for exp in ("pe3.yaml", "pe4.yaml", "pe8.yaml"):
        wait_complete(EXP / exp)
    # PE4b (recorded 2026-10-29, user; E5b v2 design): all 5 seeds on the laptop, training while the finished PE runs
    # are evaluated (the report waits on those, not on PE4b)
    import threading
    pe4b = threading.Thread(target=launch, args=(EXP / "pe4b.yaml",))
    pe4b.start()
    for exp in ("pe3", "pe4"):
        log(f"{exp}: test-set evaluation")
        evaluate_pe34(exp)
    run([ROOT / "2-dof" / "scripts" / "pe_tables.py"])
    log("PE8: evaluation on the exact and +-20% mass plants")
    run([ROOT / "2-dof" / "scripts" / "eval_pe8.py"])
    run([ROOT / "2-dof" / "scripts" / "pe9_lite.py"])
    (RUNS / "PE_EVAL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("PE3/PE4/PE8 evaluated: results/tables/pe3.md, pe4.md, pe8.md, pe9_lite.md")
    pe4b.join()
    wait_complete(EXP / "pe4b.yaml")
    log("pe4b: test-set evaluation")
    evaluate_pe34("pe4b")
    run([ROOT / "2-dof" / "scripts" / "pe_tables.py"])
    log("PE9-lite")
    run([ROOT / "2-dof" / "scripts" / "pe9_lite.py"])
    (RUNS / "PIPELINE_2DOF_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("2-DOF lane finished: results/tables/pe3.md, pe4.md, pe8.md, pe9_lite.md")


if __name__ == "__main__":
    main()
