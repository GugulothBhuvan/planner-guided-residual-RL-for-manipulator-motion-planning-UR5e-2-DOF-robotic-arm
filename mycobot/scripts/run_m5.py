"""Supervisor for the M5 runs: one detached train.py per seed, restarted from its checkpoint if it dies without done.json.

Started via WMI (Win32_Process.Create) or by scripts/watchdog_m5.py so it outlives the Claude session. Children get no
console and their own process group (see memory note on detached launches). A
trainer that is already alive for a seed is adopted, never duplicated.
Usage: python scripts/run_m5.py [--seeds 0 1] [--threads 2]
"""
import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS


def done(seed):
    return any(Path(p).exists() for p in glob.glob(str(ROOT / "results" / "runs" / "m5" / f"residual_*_s{seed}" / "done.json")))


def trainer_alive(seed):
    for p in psutil.process_iter(["cmdline"]):
        cmd = p.info["cmdline"] or []
        if any(c.endswith("train.py") for c in cmd) and "m5.yaml" in " ".join(cmd) and "--seed" in cmd \
                and cmd[cmd.index("--seed") + 1] == str(seed):
            return True
    return False


def spawn(seed, threads):
    log = open(ROOT / "results" / f"m5_s{seed}.log", "a")
    return subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--exp", "configs/exp/m5.yaml", "--run", "residual",
         "--seed", str(seed), "--threads", str(threads), "--ckpt-every", "10000"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    procs, tries = {}, {s: 0 for s in args.seeds}
    while True:
        for s in args.seeds:
            if done(s) or trainer_alive(s):
                continue
            pr = procs.get(s)
            if pr is not None:
                print(time.strftime("%F %T"), f"seed {s} exited with {pr.poll()}, restarting", flush=True)
            if tries[s] >= 20:
                continue
            tries[s] += 1
            procs[s] = spawn(s, args.threads)
        if all(done(s) or tries[s] >= 20 for s in args.seeds):
            print(time.strftime("%F %T"), "all runs finished", flush=True)
            if all(done(s) for s in args.seeds):
                (ROOT / "results" / "runs" / "m5" / "PIPELINE_DONE").write_text("done")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
