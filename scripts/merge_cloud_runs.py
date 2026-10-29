"""Merge finished cloud runs (Colab / Kaggle output, already extracted to a temporary folder) into results/runs.

A run folder <exp>/<run id> anywhere under SRC is moved to results/runs/<exp>/<run id> only if it is complete and
verified: done.json with num_timesteps equal to the config's steps, the sha1 of config.json's config equal to the
hash in the run id, val.jsonl present, and a final model (checkpoints/model_<steps>.zip or last_model.zip). A run
that fails any check is not moved; its step count is reported. If the destination exists it is first moved to
results/runs/archive/<exp>_preexisting_<run id>/ (never merged or overwritten).

    python scripts/merge_cloud_runs.py <extracted folder>          # prints one line per run; exit 1 if any failed
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def check(d):
    """-> (ok, message)"""
    try:
        meta = json.loads((d / "config.json").read_text())
    except (OSError, ValueError) as e:
        return False, f"no readable config.json ({e})"
    cfg = meta["config"]
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    done = json.loads((d / "done.json").read_text()) if (d / "done.json").exists() else None
    steps = done["num_timesteps"] if done else None
    if not done:
        last = (d / "val.jsonl").read_text().splitlines()[-1] if (d / "val.jsonl").exists() else None
        return False, f"unfinished (no done.json; last validation at {json.loads(last)['num_timesteps'] if last else 0})"
    if steps != cfg["steps"]:
        return False, f"done.json at {steps} steps, config says {cfg['steps']}"
    if h not in d.name or meta.get("run_id") != d.name:
        return False, f"config hash {h} does not match the run id"
    if not (d / "val.jsonl").exists():
        return False, "no val.jsonl"
    if not ((d / "checkpoints" / f"model_{steps}.zip").exists() or (d / "last_model.zip").exists()):
        return False, "no final model"
    return True, f"{steps} steps, hash {h}"


def main():
    src = Path(sys.argv[1])
    runs = sorted(p.parent for p in src.rglob("config.json") if p.parent.parent.name and "_s" in p.parent.name)
    failed = 0
    for d in runs:
        exp = d.parent.name
        ok, msg = check(d)
        if not ok:
            failed += 1
            print(f"SKIP {exp}/{d.name}: {msg}")
            continue
        dst = RUNS / exp / d.name
        if dst.exists():
            arch = RUNS / "archive" / f"{exp}_preexisting_{d.name}"
            arch.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(arch))
            print(f"archived existing {dst.relative_to(ROOT)} -> {arch.relative_to(ROOT)}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dst))
        print(f"MOVED {exp}/{d.name}: {msg}")
    print(f"{len(runs) - failed} moved, {failed} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
