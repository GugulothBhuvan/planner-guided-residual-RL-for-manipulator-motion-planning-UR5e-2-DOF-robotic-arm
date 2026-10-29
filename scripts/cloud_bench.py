"""Per-machine throughput gate (SPEC §11) for a cloud machine: machine info, one run, then N parallel runs.

N = cores // 2 (2 torch threads per run, as on the laptop). Prints one JSON summary line at the end and writes
results/day1/cloud_<platform>.json.

    python scripts/cloud_bench.py --platform colab
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
BENCH = [sys.executable, str(ROOT / "scripts" / "bench_throughput.py"), "--net", "256,256,256",
         "--grad-per-transition", "0.25", "--threads", "2"]


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def run_parallel(n, extra=()):
    procs = [subprocess.Popen(BENCH + list(extra), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
             for _ in range(n)]
    return [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--platform", required=True, help="colab | kaggle | other")
    args = p.parse_args()

    cores = os.cpu_count()
    vm = psutil.virtual_memory()
    info = dict(platform=args.platform, cpu=cpu_name(), cores=cores, python=platform.python_version(),
                ram_total_gb=round(vm.total / 2**30, 1),
                ram_available_gb=round(vm.available / 2**30, 1))
    print("machine:", json.dumps(info), flush=True)

    single = run_parallel(1)[0]
    print(f"1 run: {single['steps_per_s_training']} steps/s, peak {single['peak_rss_mb']} MB", flush=True)
    her = run_parallel(1, ["--mode", "her"])[0]
    print(f"1 run HER: {her['steps_per_s_training']} steps/s", flush=True)

    n = max(1, cores // 2)
    par = run_parallel(n) if n > 1 else [single]
    total = sum(r["steps_per_s_training"] for r in par)
    print(f"{n} parallel runs: {[r['steps_per_s_training'] for r in par]} -> total {total:.1f} steps/s", flush=True)

    # 1M-step runs peak at ~1.6 GB each (SPEC §11); how many fit in this machine's RAM with 1.5 GB headroom?
    fit_1m = int((vm.available / 2**30 - 1.5) // 1.6)
    summary = dict(info, single=single["steps_per_s_training"], single_her=her["steps_per_s_training"],
                   parallel_runs=n, parallel_each=[r["steps_per_s_training"] for r in par],
                   total_steps_per_s=round(total, 1), gate_100=total >= 100, peak_rss_mb=single["peak_rss_mb"],
                   runs_1m_that_fit_in_ram=fit_1m, recommended_parallel_1m_runs=max(1, min(n, fit_1m)))
    out = ROOT / "results" / "day1" / f"cloud_{args.platform}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
