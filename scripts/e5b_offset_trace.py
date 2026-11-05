"""E5b offset over an episode (evaluation only): replays the final E5b checkpoints on the 100 stored S4 validation episodes
(same settings as scripts/e5b_diagnostic.py: reset seeds 1_000_000 + i, alpha 0.3, deterministic) and stores the per-step
max-joint |o| of every episode, then plots the median |o| against the step (failed and successful episodes separately,
with the 10-90% band) and the share of episodes at the +-0.30 rad limit.

    python scripts/e5b_offset_trace.py --seeds 0 1 2 --steps 1000000 --out results/figures/e5b_offset_trace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from ur5rl.agents.factory import ALGOS
from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv
from ur5rl.envs.residual import ResidualEnv
from ur5rl.eval import testsets

LIMIT = 0.30


def trace(run, steps):
    cfg = json.loads((run / "config.json").read_text())["config"]
    env_cfg = dict(cfg["env"], n_obstacles=tuple(cfg["env"]["n_obstacles"]), curriculum=False)
    make = lambda: ResidualEnv(UR5ePoseEnv(EnvConfig(**env_cfg)), **cfg["agent"]["base"])
    model = ALGOS["residual_sac"].load(run / "checkpoints" / f"model_{steps}.zip", env=DummyVecEnv([make]), device="cpu")
    env = make()
    env.set_alpha(cfg["agent"]["alpha_max"])
    episodes, _ = testsets.load("s4_val")
    rows = []
    for i, ep in enumerate(episodes):
        obs, _ = env.reset(seed=1_000_000 + i, options={"episode": ep})
        mags, hit = [], False
        while True:
            obs, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            mags.append(float(np.abs(env.offset).max()))
            hit = hit or bool(info["collision"])
            if term or trunc:
                break
        rows.append(dict(success=bool(info["is_success"]), collision=hit, o=mags))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs/e5b/residual_offset_4f49c707_s{seed}")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--out", default="results/figures/e5b_offset_trace")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for s in args.seeds:
        run = Path(args.runs.format(seed=s))
        data[s] = trace(run, args.steps)
        n = len(data[s])
        ok = sum(r["success"] for r in data[s])
        print(f"seed {s}: {ok}/{n} success", flush=True)
    out.with_suffix(".json").write_text(json.dumps({str(k): v for k, v in data.items()}), encoding="utf-8")


if __name__ == "__main__":
    main()
