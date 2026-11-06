"""Train one reach run (SPEC M10): DDPG or SAC + HER on MyCobotReachEnv, with periodic validation against the stage's
frozen validation episodes. Resumes from run_dir/last_model.zip is NOT implemented (runs are short); results are written to
results/runs/reach/<stage>_<algo>_s<seed>/{config.json, val.jsonl, last_model.zip, best_model.zip, done.json}.

    python scripts/train_reach.py --stage r1 --algo ddpg --seed 0 --steps 100000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcrl.envs.reach_env import MyCobotReachEnv, ReachConfig  # noqa: E402
from mcrl.reachsets import load_set  # noqa: E402
from ur5rl.agents.factory import make_agent  # noqa: E402


def stage_cfg(stage):
    cfg = yaml.safe_load((ROOT / "configs" / "env" / f"mc_reach_{stage}.yaml").read_text())
    cfg["cube_center"] = tuple(cfg["cube_center"])
    cfg["obstacle_radius"] = tuple(cfg.get("obstacle_radius", (0.015, 0.035)))
    return cfg


def evaluate_policy(model, env_cfg, episodes, seed0=0):
    env = MyCobotReachEnv(ReachConfig(**env_cfg))
    ok = hit = 0
    d = []
    for i, ep in enumerate(episodes):
        obs, _ = env.reset(seed=seed0 + i, options={"episode": ep})
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        ok += bool(info["is_success"])
        hit += bool(info["collision"])
        d.append(100 * info["d_pos"])
    n = len(episodes)
    return dict(success=ok / n, collision=hit / n, d_cm_median=float(np.median(d)))


class Validation(BaseCallback):
    def __init__(self, env_cfg, episodes, every, run_dir):
        super().__init__()
        self.env_cfg, self.episodes, self.every, self.run_dir = env_cfg, episodes, every, run_dir
        self.best = (-1.0, 1.0)
        self.t0 = time.time()
        self.next = every

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            r = evaluate_policy(self.model, self.env_cfg, self.episodes)
            r.update(steps=self.num_timesteps, wall_s=round(time.time() - self.t0, 1))
            with open(self.run_dir / "val.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            key = (r["success"], -r["collision"])
            if key > (self.best[0], -self.best[1]):
                self.best = (r["success"], r["collision"])
                self.model.save(self.run_dir / "best_model")
            print(json.dumps(r), flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--algo", default="ddpg", choices=["ddpg", "sac"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--val-every", type=int, default=10_000)
    p.add_argument("--out", default=str(ROOT / "results" / "runs" / "reach"))
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    env_cfg = stage_cfg(args.stage)
    agent_cfg = yaml.safe_load((ROOT / "configs" / "agent" / f"{args.algo}_reach.yaml").read_text())
    agent_cfg["buffer_size"] = min(int(agent_cfg["buffer_size"]), args.steps)
    run_dir = Path(args.out) / f"{args.stage}_{args.algo}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "done.json").exists():
        print("already done", run_dir)
        return
    (run_dir / "config.json").write_text(json.dumps(dict(stage=args.stage, algo=args.algo, seed=args.seed, steps=args.steps,
                                                         env=env_cfg, agent=agent_cfg), indent=1, default=str))
    make = lambda: MyCobotReachEnv(ReachConfig(**env_cfg))
    venv = DummyVecEnv([make for _ in range(args.n_envs)])
    venv.seed(args.seed)
    model = make_agent(agent_cfg, venv, her=True, seed=args.seed)
    val = load_set(args.stage, "val")
    cb = Validation(env_cfg, val, args.val_every, run_dir)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(run_dir / "last_model")
    wall = time.time() - t0
    (run_dir / "done.json").write_text(json.dumps(dict(steps=args.steps, wall_s=wall, steps_per_s=args.steps / wall,
                                                       best_val=cb.best)))
    print("done", run_dir, f"{args.steps / wall:.0f} steps/s")


if __name__ == "__main__":
    main()
