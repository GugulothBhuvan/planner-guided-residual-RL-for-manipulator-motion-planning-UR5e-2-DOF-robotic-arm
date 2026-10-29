"""Run every (run, seed) job of an experiment with at most --workers at a time (SPEC §11).

    python scripts/launch.py --exp configs/exp/e1.yaml --workers 4 --threads 2

Each job's stdout/stderr goes to results/runs/<exp>/_logs/<agent>_s<seed>.log. Finished runs (done.json) are
skipped by train.py, so re-launching resumes an interrupted experiment. A job already running on this machine
(started by another launcher) is skipped, so restarting a launcher never runs a job twice.

Machine-wide scheduling (SPEC §11):
  --cap-ur5e N              start a job only while fewer than N UR5e training jobs run on the machine (any launcher)
  --stop-file PATH          once PATH exists, start no further jobs (running ones finish); a later launch resumes
  --seeds 0 --runs a b      only these seeds / runs, runs in the given order (the rest is offloaded, e.g. Colab)
  --yield-to e5.yaml        while any job of that experiment runs, use at most --yield-workers workers;
                            --yield-to ur5e yields to any UR5e training job, or UR5e launcher still waiting to
                            start one (the 2-DOF lane, SPEC §11)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Own process group and no console (Windows): a console interrupt or close killed every job on 2026-10-27 04:51 and
# again on 2026-10-28 06:17, the second time despite the process group
DETACH = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)


def live_jobs():
    """(exp path, run, seed) of every train.py job alive on the machine (the venv stub and the interpreter it
    spawns share a command line, so the set dedupes them)."""
    jobs = set()
    for proc in psutil.process_iter(["cmdline"]):
        cmd = proc.info["cmdline"] or []
        if not any(c.endswith("train.py") for c in cmd):
            continue
        arg = lambda k: cmd[cmd.index(k) + 1] if k in cmd and cmd.index(k) + 1 < len(cmd) else None
        if arg("--exp") and arg("--run") and arg("--seed"):
            jobs.add((Path(arg("--exp")).as_posix().replace("\\", "/"), arg("--run"), int(arg("--seed"))))
    return jobs


HOLD = ROOT / "results" / "runs" / "UR5E_LANE_BUSY"  # holds the pid of a pipeline with more laptop UR5e jobs to start
WAIT = ROOT / "results" / "runs" / "UR5E_WAITING_{}"  # a UR5e launcher (pid) whose next job waits only for RAM


def ur5e_launcher_waiting():
    """A UR5e launcher has a job that waits only for RAM (it wrote UR5E_WAITING_<pid>), or a UR5e pipeline holds the
    lane between two of its launches: the 2-DOF lane then starts nothing, so it cannot take the memory a pending UR5e
    start is waiting for. (Until 2026-10-29 any live UR5e launcher counted, which also paused the lane while that
    launcher's job simply ran.)"""
    try:
        if HOLD.exists() and psutil.pid_exists(int(HOLD.read_text().strip())):
            return True
    except ValueError:
        pass
    for f in WAIT.parent.glob(WAIT.name.format("*")):
        try:
            if psutil.pid_exists(int(f.name.rsplit("_", 1)[1])):
                return True
            f.unlink(missing_ok=True)  # its launcher died
        except ValueError:
            pass
    return False


def finished(exp_path, run, seed):
    """The run already has done.json. Dropped before any gating: a finished job must not wait for RAM (on 2026-10-29
    a relaunched pipeline's launcher waited for 1.4 GB to start two finished runs, and the 2-DOF lane yielded to it)."""
    from train import resolve, run_id  # same id as train.py: buffer cap included
    cfg = resolve(exp_path, run)
    if "buffer_size" in cfg["agent"]:
        cfg["agent"]["buffer_size"] = min(int(cfg["agent"]["buffer_size"]), cfg["steps"])
    return (ROOT / "results" / "runs" / cfg["exp"] / run_id(cfg, seed) / "done.json").exists()


def same_exp(a, b):
    return Path(a).name == Path(b).name and ("2-dof" in Path(a).as_posix()) == ("2-dof" in Path(b).as_posix())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--min-free-gb", type=float, default=0.0,
                   help="start a job only when this much RAM is available (keeps lower-priority lanes from paging)")
    p.add_argument("--cap-ur5e", type=int, default=None, help="max UR5e training jobs alive on the machine")
    p.add_argument("--yield-to", default=None, help="experiment file name (e.g. e5.yaml) that has priority")
    p.add_argument("--yield-workers", type=int, default=1)
    p.add_argument("--stop-file", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="only these seeds (others run elsewhere)")
    p.add_argument("--runs", nargs="+", default=None, help="only these runs, in this order (per seed)")
    args = p.parse_args()

    exp = yaml.safe_load(Path(args.exp).read_text())
    names = [r.get("name", r["agent"]) for r in exp["runs"]]
    if args.runs:
        assert set(args.runs) <= set(names), f"unknown runs {set(args.runs) - set(names)}"
        names = args.runs
    seeds = [s for s in exp["seeds"] if args.seeds is None or s in args.seeds]
    jobs = [(name, s) for s in seeds for name in names if not finished(args.exp, name, s)]  # interleave runs across seeds
    log_dir = ROOT / "results" / "runs" / exp["name"] / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def may_start():
        live = live_jobs()
        workers = args.workers
        if args.yield_to == "ur5e":
            if ur5e_launcher_waiting():
                return False  # a pending UR5e start has priority on RAM
            yielding = any("2-dof" not in e for e, _, _ in live)
        else:
            yielding = bool(args.yield_to) and any(Path(e).name == args.yield_to for e, _, _ in live)
        if yielding:
            workers = min(workers, args.yield_workers)
        # jobs of this experiment started by an earlier launcher count toward the worker limit too
        active = max(len(running), sum(same_exp(e, args.exp) for e, _, _ in live))
        if active >= workers or psutil.virtual_memory().available / 2**30 < args.min_free_gb:
            return False
        if args.cap_ur5e is not None and sum("2-dof" not in e for e, _, _ in live) >= args.cap_ur5e:
            return False
        return True

    running = []
    while jobs or running:
        running = [(j, p) for j, p in running if p.poll() is None]
        if args.stop_file and Path(args.stop_file).exists() and jobs:
            print(f"{time.strftime('%H:%M:%S')} stop file {args.stop_file}: {len(jobs)} jobs left for a later launch", flush=True)
            jobs = []
        # a UR5e launcher blocked only by the RAM gate tells the 2-DOF lane to leave the memory free
        wait_file = Path(str(WAIT).format(os.getpid()))
        ram_short = psutil.virtual_memory().available / 2**30 < args.min_free_gb
        if jobs and ram_short and "2-dof" not in Path(args.exp).as_posix():
            wait_file.touch()
        else:
            wait_file.unlink(missing_ok=True)
        while jobs and may_start():
            agent, seed = jobs.pop(0)
            if any(same_exp(e, args.exp) and r == agent and s == seed for e, r, s in live_jobs()):
                print(f"{time.strftime('%H:%M:%S')} skip {agent} s{seed}: already running", flush=True)
                continue
            log = open(log_dir / f"{agent}_s{seed}.log", "a")
            cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--exp", args.exp, "--run", agent,
                   "--seed", str(seed), "--threads", str(args.threads)]
            running.append(((agent, seed), subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                            creationflags=DETACH)))
            print(f"{time.strftime('%H:%M:%S')} started {agent} s{seed}", flush=True)
        time.sleep(10)
    Path(str(WAIT).format(os.getpid())).unlink(missing_ok=True)
    print(f"{time.strftime('%H:%M:%S')} all jobs finished", flush=True)


if __name__ == "__main__":
    main()
