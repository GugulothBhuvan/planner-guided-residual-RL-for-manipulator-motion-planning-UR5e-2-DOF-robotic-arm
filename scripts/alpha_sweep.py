"""Diagnostic (evaluation only): a residual checkpoint's policy on the stage's validation episodes at fixed alphas.

Replays the training-time validation exactly (ValidationEval: stored <stage>_val episodes, reset seeds 1_000_000 + i,
so the same per-episode base-budget draws; deterministic actions), but with alpha fixed instead of the run's ramped
value. alpha = 0 is the base alone and should reproduce base_reference.json.

    python scripts/alpha_sweep.py --run results/runs/e5/residual_<hash>_s0 --checkpoint primary
Writes <run>/alpha_sweep_<checkpoint>.json (or alpha_sweep.json for an explicit --model).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from ur5rl.agents.factory import ALGOS
from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv
from ur5rl.envs.residual import ResidualEnv
from ur5rl.eval import testsets

import checkpoints  # scripts/checkpoints.py: evaluation checkpoint rule

VAL_SEED0 = 1_000_000  # ValidationEval default, as train.py uses it


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", choices=["primary", "final"], default=None, help="SPEC §9.2 checkpoint rule")
    p.add_argument("--model", default="checkpoint/model.zip", help="relative to the run dir (without --checkpoint)")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    args = p.parse_args()
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())["config"]
    env_cfg = dict(cfg["env"], n_obstacles=tuple(cfg["env"]["n_obstacles"]), curriculum=False)
    make = lambda: ResidualEnv(UR5ePoseEnv(EnvConfig(**env_cfg)), **cfg["agent"]["base"],
                               w_res=cfg["agent"].get("w_res", 0.0))
    path = checkpoints.select(run, args.checkpoint)[0] if args.checkpoint else run / args.model
    model = ALGOS[cfg["agent"]["algo"]].load(path, env=DummyVecEnv([make]), device="cpu")
    env = make()
    episodes, _ = testsets.load(f"{cfg['stage']}_val")
    rows = []
    for alpha in args.alphas:
        env.set_alpha(alpha)
        t0 = time.time()
        succ, coll, d_pos = [], [], []
        for i, ep in enumerate(episodes):
            obs, _ = env.reset(seed=VAL_SEED0 + i, options={"episode": ep})
            done, hit = False, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                hit |= info["collision"]
                done = term or trunc
            succ.append(info["is_success"])
            coll.append(hit)
            d_pos.append(info["d_pos"])
        r = dict(alpha=alpha, n=len(succ), success=float(np.mean(succ)), collision=float(np.mean(coll)),
                 d_pos_cm=float(np.mean(d_pos) * 100), model_steps=int(model.num_timesteps), model=path.name,
                 wall_s=round(time.time() - t0, 1))
        rows.append(r)
        print(json.dumps(r), flush=True)
    (run / (f"alpha_sweep_{args.checkpoint}.json" if args.checkpoint else "alpha_sweep.json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
