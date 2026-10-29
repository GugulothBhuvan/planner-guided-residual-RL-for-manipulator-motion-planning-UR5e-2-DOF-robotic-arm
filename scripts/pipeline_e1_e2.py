"""Overnight chain after the S1 probe (SPEC §4, §8): probe -> E1 -> failure breakdown -> E2.

Decisions are mechanical and logged to results/runs/pipeline.log and results/runs/pipeline_decisions.json:
  1. Probe winner = highest full-distribution validation success at 120k. Configs within 0.03 of the best
     count as tied; ties go to the simplest config (c_g098 < a_g098_curr < b_g099_curr).
     If every probe is below 0.20, E1 and E2 use 500k steps per run (shortfall reported); otherwise E1 uses
     300k and E2 1M (SPEC §11).
  2. E1 = TD3+HER vs SAC+HER on S1, 3 seeds, both with the winner's gamma and curriculum setting.
  3. After E1: failure breakdown (scripts/failure_breakdown.py).
  4. E2 algorithm = higher mean final validation success over E1 seeds (tie within 0.03: lower mean final
     position error). E2 = that algorithm x {sparse+HER, dense} on S2, 2 seeds, with the winner's env settings.
  5. E2 workers = how many ~1.7 GB runs fit in the RAM free at launch, leaving 1 GB (between 1 and 3).

    python scripts/pipeline_e1_e2.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
LOG = RUNS / "pipeline.log"
DECISIONS = RUNS / "pipeline_decisions.json"
SIMPLICITY = ["c_g098", "a_g098_curr", "b_g099_curr"]
PROBE = yaml.safe_load((ROOT / "configs" / "exp" / "probe_s1.yaml").read_text())
TIE = 0.03


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def record(key, value):
    d = json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}
    d[key] = value
    DECISIONS.write_text(json.dumps(d, indent=2))


def finished_runs(exp_name):
    d = RUNS / exp_name
    return {p.name: json.loads((p / "done.json").read_text()) | {"config": json.loads((p / "config.json").read_text())}
            for p in d.iterdir() if (p / "done.json").exists()} if d.exists() else {}


def wait_for(exp_name, n):
    while len(finished_runs(exp_name)) < n:
        time.sleep(60)


def launch(exp_path, workers):
    log(f"launch {exp_path.name} with {workers} workers")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "launch.py"), "--exp", str(exp_path), "--workers",
                    str(workers), "--threads", "2"], check=True)


def pick_probe(scores):
    """scores: run name -> success. Returns (winner, tied names, all below 0.20)."""
    best = max(scores.values())
    tied = [name for name in SIMPLICITY if name in scores and scores[name] >= best - TIE]
    return tied[0], tied, best < 0.20


def pick_algorithm(per_algo):
    """per_algo: algo -> [(success, d_pos_cm) per seed]. Returns (algo, means)."""
    means = {a: (sum(s for s, _ in v) / len(v), sum(d for _, d in v) / len(v)) for a, v in per_algo.items()}
    ranked = sorted(means, key=lambda a: (-means[a][0], means[a][1]))
    algo = ranked[0]
    if len(ranked) > 1 and abs(means[ranked[0]][0] - means[ranked[1]][0]) <= TIE:
        algo = min(ranked[:2], key=lambda a: means[a][1])
    return algo, means


def main():
    # 1. probe winner
    wait_for("probe_s1", len(PROBE["runs"]))
    scores = {r["config"]["config"]["run"]: r["final_val"]["success"] for r in finished_runs("probe_s1").values()}
    winner, tied, weak = pick_probe(scores)
    e1_steps, e2_steps = (500_000, 500_000) if weak else (300_000, 1_000_000)
    win_run = next(r for r in PROBE["runs"] if r["name"] == winner)
    gamma_her = win_run["overrides"]["gamma_her"]
    curriculum = win_run.get("env_overrides", {}).get("curriculum", False)
    log(f"probe scores {scores}; winner {winner} (gamma_her={gamma_her}, curriculum={curriculum}); "
        f"weak={weak}; E1 steps {e1_steps}, E2 steps {e2_steps}")
    record("probe", dict(scores=scores, winner=winner, tied=tied, all_below_20pct=weak,
                         gamma_her=gamma_her, curriculum=curriculum, e1_steps=e1_steps, e2_steps=e2_steps))

    # 2. E1
    env_over = {"curriculum": True} if curriculum else {}
    e1 = dict(name="e1", env="s1", steps=e1_steps, n_envs=4, seeds=[0, 1, 2],
              runs=[dict(name=f"{a}", agent=a, her=True, overrides={"gamma_her": gamma_her}, env_overrides=env_over)
                    for a in ("td3", "sac")])
    e1_path = ROOT / "configs" / "exp" / "e1.yaml"
    e1_path.write_text("# Written by scripts/pipeline_e1_e2.py from the S1 probe winner (SPEC §4).\n"
                       + yaml.safe_dump(e1, sort_keys=False), encoding="utf-8")
    launch(e1_path, workers=3)
    wait_for("e1", 6)

    # 3. failure breakdown
    log("E1 finished; failure breakdown")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "failure_breakdown.py"), "--exp", str(e1_path),
                          "--episodes", "200"], capture_output=True, text=True)
    (RUNS / "e1" / "failure_breakdown.txt").write_text(out.stdout + out.stderr[-2000:])
    log("breakdown:\n" + out.stdout)

    # 4. E2 algorithm
    per_algo = {}
    for r in finished_runs("e1").values():
        algo = r["config"]["config"]["agent"]["algo"]
        per_algo.setdefault(algo, []).append((r["final_val"]["success"], r["final_val"]["d_pos_cm"]))
    algo, means = pick_algorithm(per_algo)
    log(f"E1 means (success, d_pos cm) {means}; E2 algorithm {algo}")
    record("e1", dict(means=means, e2_algorithm=algo))

    # 5. E2
    e2 = dict(name="e2", env="s2", steps=e2_steps, n_envs=4, seeds=[0, 1],
              runs=[dict(name=f"{algo}_sparse_her", agent=algo, her=True, overrides={"gamma_her": gamma_her},
                         env_overrides=dict(env_over, reward="sparse")),
                    dict(name=f"{algo}_dense", agent=algo, her=False, env_overrides=dict(env_over, reward="dense"))])
    e2_path = ROOT / "configs" / "exp" / "e2.yaml"
    e2_path.write_text("# Written by scripts/pipeline_e1_e2.py (SPEC §8 E2).\n" + yaml.safe_dump(e2, sort_keys=False), encoding="utf-8")
    free_gb = psutil.virtual_memory().available / 2**30
    workers = max(1, min(3, int((free_gb - 1.0) // 1.7)))
    log(f"free RAM {free_gb:.1f} GB -> E2 workers {workers}")
    record("e2", dict(algorithm=algo, steps=e2_steps, workers=workers, free_ram_gb=round(free_gb, 1)))
    launch(e2_path, workers=workers)
    log("E2 finished")


if __name__ == "__main__":
    main()
