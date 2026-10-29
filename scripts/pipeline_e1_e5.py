"""UR5e lane schedule (SPEC §11, decisions 2026-10-27): E5 first on the laptop; E1's second wave on Colab/Kaggle.

E5 does not depend on E1 (it trains SAC either way), so it takes the laptop slots E1 frees. Two threads:
  A. E5 (configs/exp/e5.yaml, 6 runs) with at most 2 UR5e training jobs alive on the machine (--cap-ur5e 2, RAM),
     so each E5 run starts as another UR5e run finishes; every start waits for 1.7 GB free RAM. After E5, any E1 run still
     missing is run here, unless results/runs/e1_offloaded exists (E1's second wave runs on Colab/Kaggle:
     notebooks/colab_e1_wave2.ipynb; copy its run dirs into results/runs/e1).
  B. Once all six E1 runs have done.json (wherever they ran): failure breakdown, E2 algorithm pick (the
     E1 -> E2 rule, pick_algorithm), configs/exp/e2.yaml, rebuilt dist/ur5rl_bundle.zip. E2 then runs on the
     laptop after E5, unless results/runs/e2_offloaded exists (notebooks/colab_e2.ipynb).
Decisions go to results/runs/pipeline.log and results/runs/pipeline_decisions.json.

    python scripts/pipeline_e1_e5.py
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_e1_e2 import RUNS, ROOT, finished_runs, log, pick_algorithm, record  # noqa: E402

EXP = ROOT / "configs" / "exp"
RUN_GB = 1.7  # per-run RAM budget for a 1M-step UR5e run
# At most 2 UR5e training jobs on the laptop (decision 2026-10-27: RAM cannot be freed; 3 would page as the 1M
# buffers fill). Queue order puts E5 SAC+HER s1 next, so it resumes when either running E5 job finishes.
UR5E_CAP = 2
# own process group and no console: a console close or Ctrl+C (IDE terminal closed) reaches no job
DETACH = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
E1_PATH = EXP / "e1.yaml"


def launch(exp_path, workers=UR5E_CAP):
    log(f"launch {exp_path.name} (workers {workers}, cap {UR5E_CAP} UR5e jobs, {RUN_GB} GB free per start)")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "launch.py"), "--exp", str(exp_path), "--workers",
                    str(workers), "--threads", "2", "--min-free-gb", str(RUN_GB), "--cap-ur5e", str(UR5E_CAP)],
                   check=True, creationflags=DETACH)
    log(f"{exp_path.name} launcher finished")


def e1_total():
    e1 = yaml.safe_load(E1_PATH.read_text(encoding="utf-8"))
    return len(e1["runs"]) * len(e1["seeds"])


def after_e1(e5_done: threading.Event):
    while len(finished_runs("e1")) < e1_total():
        time.sleep(60)
    log("all E1 runs present; failure breakdown")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "failure_breakdown.py"), "--exp", str(E1_PATH),
                          "--episodes", "200"], capture_output=True, text=True)
    (RUNS / "e1" / "failure_breakdown.txt").write_text(out.stdout + out.stderr[-2000:], encoding="utf-8")
    log("breakdown:\n" + out.stdout)

    e1 = yaml.safe_load(E1_PATH.read_text(encoding="utf-8"))
    per_algo = {}
    for r in finished_runs("e1").values():
        algo = r["config"]["config"]["agent"]["algo"]
        per_algo.setdefault(algo, []).append((r["final_val"]["success"], r["final_val"]["d_pos_cm"]))
    algo, means = pick_algorithm(per_algo)
    log(f"E1 means (success, d_pos cm) {means}; E2 algorithm {algo}")
    record("e1", dict(means=means, e2_algorithm=algo))
    gamma_her = e1["runs"][0]["overrides"]["gamma_her"]
    env_over = dict(e1["runs"][0].get("env_overrides") or {})
    e2 = dict(name="e2", env="s2", steps=500_000, n_envs=4, seeds=[0, 1],
              runs=[dict(name=f"{algo}_sparse_her", agent=algo, her=True, overrides={"gamma_her": gamma_her},
                         env_overrides=dict(env_over, reward="sparse")),
                    dict(name=f"{algo}_dense", agent=algo, her=False, env_overrides=dict(env_over, reward="dense"))])
    e2_path = EXP / "e2.yaml"
    e2_path.write_text("# Written by scripts/pipeline_e1_e5.py (SPEC §8 E2).\n" + yaml.safe_dump(e2, sort_keys=False),
                       encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_bundle.py")], check=False)
    log("wrote configs/exp/e2.yaml and rebuilt dist/ur5rl_bundle.zip: E2 can start on Colab/Kaggle")

    e5_done.wait()
    if (RUNS / "e2_offloaded").exists():
        log("E2 offloaded (results/runs/e2_offloaded); not launching locally")
        return
    record("e2", dict(algorithm=algo, steps=e2["steps"], where="laptop"))
    launch(e2_path)


def main():
    e5_done = threading.Event()
    b = threading.Thread(target=after_e1, args=(e5_done,), daemon=False)
    b.start()
    launch(EXP / "e5.yaml")
    if len(finished_runs("e1")) < e1_total() and not (RUNS / "e1_offloaded").exists():
        log("E5 done; running the missing E1 runs on the laptop")
        launch(E1_PATH)
    e5_done.set()
    b.join()
    log("UR5e lane finished")


if __name__ == "__main__":
    main()
