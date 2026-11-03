"""Watchdog for the M5 training lane: one check per call, called every 10 minutes by the existing scheduled task
"ur5rl-watchdog" (scripts/watchdog.py imports this) or in a loop with --loop (second layer, started via WMI).

Each check appends one heartbeat line to results/watchdog_m5.log:  time | supervisor | per-seed trainer pid, checkpoint
steps, minutes since the last checkpoint write. Actions:
  * supervisor (scripts/run_m5.py) missing and no PIPELINE_DONE -> restart it (detached, breakaway from the job object);
  * a trainer alive but its checkpoint is older than STALE_MIN minutes (checkpoints are written every 10k steps, ~2-3 min)
    -> hung: kill it, the supervisor respawns it from the checkpoint.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "m5"
LOG = ROOT / "results" / "watchdog_m5.log"
PY = ROOT.parent / ".venv" / "Scripts" / "python.exe"
SEEDS = (0, 1)
STALE_MIN = 25
DETACH = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
BREAKAWAY = 0x01000000


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def procs(name, seed=None):
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = p.info["cmdline"] or []
        if any(c.endswith(name) for c in cmd) and (seed is None or ("--seed" in cmd and cmd[cmd.index("--seed") + 1] == str(seed))):
            out.append(p)
    return out


def ckpt_state(seed):
    """(steps, minutes since the newest slot was written) over both slots, or (None, None)."""
    best = None
    for d in glob.glob(str(RUNS / f"residual_*_s{seed}")):
        for slot in ("checkpoint", "checkpoint_b"):
            f = Path(d) / slot / "state.json"
            if f.exists():
                try:
                    steps = json.loads(f.read_text())["num_timesteps"]
                except Exception:
                    continue
                age = (time.time() - f.stat().st_mtime) / 60
                if best is None or steps > best[0]:
                    best = (steps, age)
    return best or (None, None)


def done(seed):
    return bool(glob.glob(str(RUNS / f"residual_*_s{seed}" / "done.json")))


def check():
    RUNS.mkdir(parents=True, exist_ok=True)
    finished = (RUNS / "PIPELINE_DONE").exists()
    sup = procs("run_m5.py")
    parts = [f"supervisor={'up' if sup else 'DOWN'}{' (finished)' if finished else ''}"]
    for s in SEEDS:
        tr = procs("train.py", s)
        steps, age = ckpt_state(s)
        parts.append(f"s{s}: {'done' if done(s) else ('pid ' + str(tr[0].info['pid']) if tr else 'NO TRAINER')}, "
                     f"ckpt {steps} ({'%.1f' % age if age is not None else '-'} min ago)")
        if tr and age is not None and age > STALE_MIN and not done(s):
            for p in tr:
                try:
                    p.kill()
                except psutil.Error:
                    pass
            log(f"seed {s} trainer stale ({age:.0f} min since the last checkpoint): killed, the supervisor respawns it")
    if not sup and not finished:
        with open(ROOT / "results" / "m5_launcher.out", "a") as fo, open(ROOT / "results" / "m5_launcher.err", "a") as fe:
            cmd = [str(PY), str(ROOT / "scripts" / "run_m5.py"), "--seeds", *map(str, SEEDS), "--threads", "2"]
            try:
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=fo, stderr=fe, creationflags=DETACH | BREAKAWAY)
            except OSError:
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=fo, stderr=fe, creationflags=DETACH)
        parts.append(f"-> restarted supervisor pid {p.pid}")
    log(" | ".join(parts))


if __name__ == "__main__":
    if "--loop" in sys.argv:
        while True:
            try:
                check()
            except Exception as exc:
                log(f"check failed: {exc!r}")
            time.sleep(600)
    else:
        check()
