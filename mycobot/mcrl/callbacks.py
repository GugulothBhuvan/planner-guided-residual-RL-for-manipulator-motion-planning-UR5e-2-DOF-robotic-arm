"""Crash-safe checkpoints for the long MyCobot runs (an unexplained kill stopped both 1M-step runs on 2026-11-03 00:16).

AtomicCheckpoint: every `every` env steps writes model + replay buffer into a temporary folder, then swaps it in as one
of two alternating slots (checkpoint, checkpoint_b), always overwriting the OLDER slot. A kill at any moment leaves the
newest complete slot untouched; a half-written folder is never visible under a slot name.
pick_checkpoint: resume from the newest slot that loads, falling back to the older one (and ignoring leftovers).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from ur5rl.agents.callbacks import CheckpointWithBuffer

SLOTS = ("checkpoint", "checkpoint_b")


def slot_steps(run_dir, slot):
    f = Path(run_dir) / slot / "state.json"
    try:
        return int(json.loads(f.read_text())["num_timesteps"]) if (Path(run_dir) / slot / "model.zip").exists() else -1
    except (OSError, ValueError, KeyError):
        return -1


class AtomicCheckpoint(CheckpointWithBuffer):
    def save(self):
        run = self.dir.parent
        older = min(SLOTS, key=lambda s: slot_steps(run, s))
        tmp, dst = run / "checkpoint_tmp", run / older
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        self.model.save(tmp / "model.zip")
        if hasattr(self.model, "replay_buffer"):
            self.model.save_replay_buffer(tmp / "replay_buffer.pkl")
        (tmp / "state.json").write_text(json.dumps(dict(num_timesteps=self.num_timesteps)))
        shutil.rmtree(dst, ignore_errors=True)  # the other slot still holds the newest complete checkpoint
        tmp.rename(dst)
        n = self.num_timesteps
        if self.keep_every and n % self.keep_every == 0:
            self.keep_dir.mkdir(exist_ok=True)
            shutil.copy2(dst / "model.zip", self.keep_dir / f"model_{n}.zip")
        if n in self.keep_full_at and not (self.keep_dir / f"full_{n}").exists():
            shutil.copytree(dst, self.keep_dir / f"full_{n}")


def pick_checkpoint(run_dir, algo_cls, venv):
    """Newest loadable slot -> model (buffer loaded), or None if there is none."""
    run = Path(run_dir)
    shutil.rmtree(run / "checkpoint_tmp", ignore_errors=True)  # a save that was cut off
    for slot in sorted(SLOTS, key=lambda s: -slot_steps(run, s)):
        d = run / slot
        if slot_steps(run, slot) < 0:
            continue
        try:
            model = algo_cls.load(d / "model.zip", env=venv, device="cpu")
            if (d / "replay_buffer.pkl").exists():
                model.load_replay_buffer(d / "replay_buffer.pkl")
            print(f"{run.name}: resumed from {slot} at {model.num_timesteps} steps", file=sys.stderr, flush=True)
            return model
        except Exception as exc:  # corrupt slot: fall back to the other one
            print(f"{run.name}: {slot} failed to load ({exc!r}); trying the other slot", file=sys.stderr, flush=True)
    return None
