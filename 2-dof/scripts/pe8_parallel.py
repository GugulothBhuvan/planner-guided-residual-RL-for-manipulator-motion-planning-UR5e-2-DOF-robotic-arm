"""Run the PE8 evaluations (eval_pe8.py) in parallel processes, one (controller, plant) pair per task. eval_pe8.py runs
them serially (~1.5 h for 51 x 300 episodes, measured 2026-10-30); it skips any output file that already exists, so
running this first only saves time, and the results are the same.

    python 2-dof/scripts/pe8_parallel.py --workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path


def pe8():
    if "eval_pe8" not in sys.modules:
        spec = importlib.util.spec_from_file_location("eval_pe8", Path(__file__).with_name("eval_pe8.py"))
        assert spec is not None and spec.loader is not None
        sys.modules["eval_pe8"] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules["eval_pe8"]


def one(task):
    name, run, ckpt, m = task
    m8 = pe8()
    f = m8.OUT / f"{name}__{ckpt}__m{m}.jsonl"
    if f.exists():
        return f"{f.name}: exists"
    part = f.with_suffix(".part")
    if part.exists() or (m8.OUT / f".{f.stem}.lock").exists():
        return f"{f.name}: in progress elsewhere"
    lock = m8.OUT / f".{f.stem}.lock"
    lock.write_text(str(time.time()))
    t0 = time.perf_counter()
    rows = m8.evaluate(name, m, run, ckpt) if run else m8.evaluate(name, m)
    tmp = f.with_suffix(".part")
    tmp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    tmp.replace(f)
    return f"{f.name}: {time.perf_counter() - t0:.0f} s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    m8 = pe8()
    m8.OUT.mkdir(parents=True, exist_ok=True)
    m8.episodes()  # create the frozen test episodes once, before the workers start
    tasks = [(name, run, ckpt, m) for name, run, ckpt in m8.jobs() for m in m8.PLANTS]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(one, tasks):
            print(time.strftime("%H:%M:%S"), msg, flush=True)


if __name__ == "__main__":
    main()
