"""Evaluation checkpoint rule for residual runs (SPEC §9.2, decision 2026-10-28), used by evaluate.py and alpha_sweep.py.

  primary: the kept checkpoint (checkpoints/model_<N>.zip, every 100k) with the best validation result (success, then
           lower final position error) among those taken after the alpha ramp has completed AND the critic warm-up
           has ended (before that the actor is still its initial near-zero policy, so the run is the base alone);
  final:   the last checkpoint (model_<steps>.zip, else last_model.zip);
  best:    best_model.zip, the plain best-validation model (non-residual policies).
"""

from __future__ import annotations

import json
from pathlib import Path


def residual_start(cfg):
    """First env step at which the residual is both fully scaled and trained."""
    a = cfg["agent"]
    warmup_end = a["learning_starts"] + a.get("critic_warmup_steps", 0) / a["grad_steps_per_transition"]
    return max(a.get("alpha_ramp_steps", 0), warmup_end)


def select(run, which):
    """-> (model path, env steps or None, note)."""
    run = Path(run)
    if which == "best":
        path = run / "best_model.zip" if (run / "best_model.zip").exists() else run / "last_model.zip"
        return path, None, "best validation"
    cfg = json.loads((run / "config.json").read_text())["config"]
    kept = {int(p.stem.split("_")[1]): p for p in (run / "checkpoints").glob("model_*.zip")}
    if which == "final":
        steps = cfg["steps"]
        return (kept[steps], steps, "final") if steps in kept else (run / "last_model.zip", steps, "final (last_model)")
    if which == "primary":
        val = {r["num_timesteps"]: r for r in map(json.loads, (run / "val.jsonl").read_text().splitlines())}
        start = residual_start(cfg)
        cands = [n for n in kept if n >= start and n in val]
        if not cands:
            raise ValueError(f"{run.name}: no kept checkpoint after step {start:.0f} with a validation result")
        n = max(cands, key=lambda k: (val[k]["success"], -val[k]["d_pos_cm"]))
        return kept[n], n, f"best validation after step {start:.0f} (val success {val[n]['success']:.2f})"
    raise ValueError(which)
