"""Restart the laptop pipelines if they died (SPEC §11). Run every 10 minutes by Windows Task Scheduler (task
"ur5rl-watchdog", scripts/install_watchdog.ps1).

Every venv Python on this laptop runs inside a kill-on-close job object, and on 2026-10-27 04:51, 2026-10-28 06:17,
2026-10-28 10:02 and 2026-10-29 02:39 every training process and both pipelines died at once with no traceback, no
crash, sleep or reboot event. The cause is outside the code. The pipelines and train.py are restartable (done.json
skips, checkpoints resume), so this watchdog brings the lanes back within 10 minutes and logs each restart, which
also timestamps the next stop.

    python scripts/watchdog.py            # one check (what the scheduled task runs)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
LOG = RUNS / "watchdog.log"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
# (script, arguments, finished marker, stdout, stderr)
PIPELINES = [
    ("scripts/pipeline_e5_v2.py", ["--decision-a", "--min-free-gb", "1.4"], RUNS / "e5" / "PIPELINE_DONE",
     RUNS / "pipeline_e5_v2.out", RUNS / "pipeline.err"),
    ("2-dof/scripts/pipeline_2dof.py", [], RUNS / "PIPELINE_2DOF_DONE",
     RUNS / "pipeline_2dof.out", RUNS / "pipeline_2dof.err"),
    # E5b runs only once its pre-registration is recorded (configs/exp/e5b.yaml exists and results/runs/e5b was made)
    ("scripts/pipeline_e5b.py", [], RUNS / "e5b" / "PIPELINE_DONE", RUNS / "pipeline_e5b.out", RUNS / "pipeline_e5b.err"),
]
# no console, own process group, and out of the scheduler's job object (falls back if the job forbids breakaway)
DETACH = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
BREAKAWAY = 0x01000000  # CREATE_BREAKAWAY_FROM_JOB


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def alive(script):
    name = Path(script).name
    for proc in psutil.process_iter(["cmdline"]):
        if any(c.endswith(name) for c in proc.info["cmdline"] or []):
            return True
    return False


def start(script, args, out, err):
    cmd = [str(PY), str(ROOT / script), *args]
    with open(out, "a") as fo, open(err, "a") as fe:
        try:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=fo, stderr=fe, creationflags=DETACH | BREAKAWAY)
        except OSError:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=fo, stderr=fe, creationflags=DETACH)
    return p.pid


def main():
    trains = sum(any(c.endswith("train.py") for c in p.info["cmdline"] or []) for p in psutil.process_iter(["cmdline"]))
    for script, args, done, out, err in PIPELINES:
        if done.exists() or alive(script) or not done.parent.exists():  # e.g. results/runs/e5b before E5b starts
            continue
        pid = start(script, args, out, err)
        log(f"{script} was not running ({trains} train.py processes alive): restarted, pid {pid}")
        time.sleep(15)  # the E5 pipeline writes its lane hold before the 2-DOF lane looks for it


if __name__ == "__main__":
    main()
