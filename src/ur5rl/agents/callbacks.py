"""Training callbacks (SPEC §11, §14): checkpoints with replay buffer, validation eval, resource logging."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from stable_baselines3.common.callbacks import BaseCallback


def peak_rss_mb() -> float:
    """Peak resident memory of this process (MB). Windows: peak working set; Linux: ru_maxrss."""
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux


class ResourceLogger(BaseCallback):
    """Logs throughput, current and peak RAM every `every` env steps."""

    def __init__(self, every=10_000):
        super().__init__()
        self.every, self.last_t, self.last_n = every, None, 0

    def _on_step(self):
        if self.num_timesteps - self.last_n < self.every and self.last_t is not None:
            return True
        now = time.perf_counter()
        if self.last_t is not None:
            self.logger.record("sys/steps_per_s", (self.num_timesteps - self.last_n) / (now - self.last_t))
        self.last_t, self.last_n = now, self.num_timesteps
        self.logger.record("sys/rss_mb", psutil.Process().memory_info().rss / 2**20)
        self.logger.record("sys/peak_rss_mb", peak_rss_mb())
        self.logger.record("sys/free_ram_mb", psutil.virtual_memory().available / 2**20)
        return True


class GoalCurriculum(BaseCallback):
    """Goal-distance curriculum (SPEC §6.5): raise the per-joint goal bound by `step` whenever the success rate
    of the last `window` training episodes exceeds `threshold`, up to `cap`. The level is logged and persisted
    in run_dir/curriculum.json so a resumed run continues at the same level."""

    def __init__(self, run_dir: Path, start, step, cap, threshold=0.8, window=100):
        super().__init__()
        self.path = Path(run_dir) / "curriculum.json"
        self.level = json.loads(self.path.read_text())["level"] if self.path.exists() else start
        self.step, self.cap, self.threshold, self.window = step, cap, threshold, window
        self.recent = []

    def _on_training_start(self):
        self.training_env.env_method("set_curriculum_level", self.level)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            # Only episodes whose goal came from the current level count: the always-full-range goals would
            # otherwise cap the rolling rate near 0.8 * p_level + 0.2 * p_full and stall the curriculum.
            if done and info.get("curriculum_goal", True):
                self.recent.append(float(info["is_success"]))
        self.recent = self.recent[-self.window:]
        rate = float(np.mean(self.recent)) if self.recent else 0.0
        if len(self.recent) == self.window and rate > self.threshold and self.level < self.cap:
            self.level = min(self.level + self.step, self.cap)
            self.training_env.env_method("set_curriculum_level", self.level)
            self.recent = []
            self.path.write_text(json.dumps(dict(level=self.level, num_timesteps=self.num_timesteps)))
            print(f"[curriculum] {self.num_timesteps:>8} level -> {self.level:.2f} rad", file=sys.stderr, flush=True)
        self.logger.record("curriculum/level", self.level)
        self.logger.record("curriculum/rolling_success", rate)
        return True


class ToleranceCurriculum(BaseCallback):
    """Success-tolerance curriculum (companion SPEC §6.5): eps starts at `start` and drops by `step` (to `final`)
    whenever rolling success over `window` training episodes exceeds `threshold`. Persisted for resume."""

    def __init__(self, run_dir: Path, start, step, final, threshold=0.8, window=100):
        super().__init__()
        self.path = Path(run_dir) / "tolerance.json"
        self.eps = json.loads(self.path.read_text())["eps"] if self.path.exists() else start
        self.step, self.final, self.threshold, self.window = step, final, threshold, window
        self.recent = []

    def _on_training_start(self):
        self.training_env.env_method("set_tolerance", self.eps)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.recent.append(float(info["is_success"]))
        self.recent = self.recent[-self.window:]
        rate = float(np.mean(self.recent)) if self.recent else 0.0
        if len(self.recent) == self.window and rate > self.threshold and self.eps > self.final + 1e-12:
            self.eps = max(self.eps - self.step, self.final)
            self.training_env.env_method("set_tolerance", self.eps)
            self.recent = []
            self.path.write_text(json.dumps(dict(eps=self.eps, num_timesteps=self.num_timesteps)))
            print(f"[tolerance] {self.num_timesteps:>8} eps -> {self.eps * 100:.1f} cm", file=sys.stderr, flush=True)
        self.logger.record("curriculum/eps", self.eps)
        self.logger.record("curriculum/rolling_success", rate)
        return True


class CheckpointWithBuffer(BaseCallback):
    """Every `every` env steps: save model + replay buffer to run_dir/checkpoint (overwritten), for resume.

    Kept copies (never overwritten, SPEC §11): the model alone at every multiple of `keep_every`
    (run_dir/checkpoints/model_<steps>.zip, ~7 MB), and model + replay buffer at `keep_full_at`
    (run_dir/checkpoints/full_<steps>/), the restart points of the §7.3 residual rule. Full copies at every 100k
    would need ~51 GB for E5 (0.85 GB buffer each); the disk has ~24 GB.
    """

    def __init__(self, run_dir: Path, every=50_000, keep_every=100_000, keep_full_at=(100_000, 300_000)):
        super().__init__()
        self.dir, self.every, self.last = Path(run_dir) / "checkpoint", every, 0
        self.keep_dir, self.keep_every, self.keep_full_at = Path(run_dir) / "checkpoints", keep_every, set(keep_full_at)

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self.save()
        return True

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "model_tmp.zip"
        self.model.save(tmp)
        if hasattr(self.model, "replay_buffer"):  # off-policy only (PPO has none)
            self.model.save_replay_buffer(self.dir / "replay_buffer.pkl")
        tmp.replace(self.dir / "model.zip")  # model last, so a complete model.zip implies a complete buffer
        (self.dir / "state.json").write_text(json.dumps(dict(num_timesteps=self.num_timesteps)))
        n = self.num_timesteps
        if self.keep_every and n % self.keep_every == 0:
            self.keep_dir.mkdir(exist_ok=True)
            shutil.copy2(self.dir / "model.zip", self.keep_dir / f"model_{n}.zip")
        if n in self.keep_full_at and not (self.keep_dir / f"full_{n}").exists():
            shutil.copytree(self.dir, self.keep_dir / f"full_{n}")


class ValidationEval(BaseCallback):
    """Every `every` env steps: deterministic rollouts on the fixed validation episodes (§9.2, §14).

    Validation episodes are reset with seeds val_seed0 .. val_seed0 + n - 1, disjoint from training seeds and
    from the frozen test-set seeds. Saves best (by success, then final position error) and last checkpoints.
    """

    def __init__(self, make_env, run_dir: Path, every=25_000, n_episodes=100, val_seed0=1_000_000, episodes=None):
        """episodes: stored validation episodes (testsets/<stage>_val.npz) replayed via reset(options=...);
        without them, episodes are regenerated from seeds val_seed0 + i (identical where nothing was rejected)."""
        super().__init__()
        self.env = make_env()
        self.dir, self.every, self.n, self.seed0 = Path(run_dir), every, n_episodes, val_seed0
        self.episodes = episodes
        if episodes is not None:
            self.n = len(episodes)
        self.last, self.best = 0, None
        self.history = []

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self.evaluate()
        return True

    def _on_training_end(self):
        self.evaluate()

    def evaluate(self):
        if hasattr(self.env, "set_alpha") and hasattr(self.model, "residual_alpha"):
            self.env.set_alpha(self.model.residual_alpha)  # residual agents: evaluate at the current alpha
        succ, d_pos, theta, coll, length = [], [], [], [], []
        for i in range(self.n):
            if self.episodes is not None:
                obs, _ = self.env.reset(seed=self.seed0 + i, options={"episode": self.episodes[i]})
            else:
                obs, _ = self.env.reset(seed=self.seed0 + i)
            done, t, hit = False, 0, False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = self.env.step(action)
                hit |= info["collision"]
                done, t = term or trunc, t + 1
            succ.append(info["is_success"])
            d_pos.append(info["d_pos"])
            theta.append(info.get("theta", 0.0))  # position-only envs (2-DOF) have no orientation error
            coll.append(hit)
            length.append(t)
        r = dict(num_timesteps=self.num_timesteps, success=float(np.mean(succ)),
                 d_pos_cm=float(np.mean(d_pos) * 100), theta_deg=float(np.rad2deg(np.mean(theta))),
                 collision=float(np.mean(coll)), ep_len=float(np.mean(length)))
        for k, v in r.items():
            if k != "num_timesteps":
                self.logger.record(f"val/{k}", v)
        self.history.append(r)
        with open(self.dir / "val.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        key = (r["success"], -r["d_pos_cm"])
        if self.best is None or key > self.best:
            self.best = key
            self.model.save(self.dir / "best_model.zip")
        self.model.save(self.dir / "last_model.zip")
        print(f"[val] {self.num_timesteps:>8} success {r['success']:.2f} d_pos {r['d_pos_cm']:.2f} cm "
              f"theta {r['theta_deg']:.1f} deg", file=sys.stderr, flush=True)
