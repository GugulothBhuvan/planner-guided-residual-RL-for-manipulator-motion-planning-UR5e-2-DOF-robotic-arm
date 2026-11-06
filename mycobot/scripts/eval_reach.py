"""Evaluate reach controllers on a stage's frozen test episodes (SPEC M10).
    python scripts/eval_reach.py scripted r1 r2 r3 r4          -> results/eval/reach/scripted_<stage>.json (straight, detour)
    python scripts/eval_reach.py policy r1 --run results/runs/reach/r1_ddpg_s0 [--which best|last]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from mcrl.envs.reach_env import MyCobotReachEnv, ReachConfig  # noqa: E402
from mcrl.reach_baselines import Detour, rollout, straight  # noqa: E402
from mcrl.reachsets import load_set  # noqa: E402
from train_reach import stage_cfg  # noqa: E402

OUT = ROOT / "results" / "eval" / "reach"


def summ(rows):
    return dict(n=len(rows), success=float(np.mean([r["success"] for r in rows])), collision=float(np.mean([r["collision"] for r in rows])),
                d_cm_median=float(np.median([r["d_cm"] for r in rows])), d_cm_max=float(max(r["d_cm"] for r in rows)),
                steps_median=float(np.median([r["steps"] for r in rows])))


def policy_rows(model, env, episodes):
    rows = []
    for i, ep in enumerate(episodes):
        obs, _ = env.reset(seed=i, options={"episode": ep})
        hit, n = False, 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            hit |= bool(info["collision"])
            n += 1
            if term or trunc:
                break
        rows.append(dict(success=bool(info["is_success"]), collision=hit, d_cm=100 * info["d_pos"], steps=n))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["scripted", "policy"])
    p.add_argument("stages", nargs="+")
    p.add_argument("--run")
    p.add_argument("--which", default="best")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for st in a.stages:
        cfg = stage_cfg(st)
        env = MyCobotReachEnv(ReachConfig(**cfg))
        eps = load_set(st, "test")
        if a.mode == "scripted":
            res = dict(straight=summ(rollout(env, lambda e: straight, episodes=eps)), detour=summ(rollout(env, lambda e: Detour(e), episodes=eps)))
            (OUT / f"scripted_{st}.json").write_text(json.dumps(res, indent=1))
            print(st, json.dumps(res))
        else:
            from stable_baselines3 import DDPG, SAC
            run = Path(a.run)
            rc = json.loads((run / "config.json").read_text())
            cls = {"ddpg": DDPG, "sac": SAC}[rc["algo"]]
            from stable_baselines3.common.vec_env import DummyVecEnv
            model = cls.load(run / ("best_model.zip" if a.which == "best" else "last_model.zip"), device="cpu",
                             env=DummyVecEnv([lambda: MyCobotReachEnv(ReachConfig(**cfg))]))  # HER buffers need an env to load
            res = summ(policy_rows(model, env, eps))
            (OUT / f"policy_{run.name}_{a.which}.json").write_text(json.dumps(res, indent=1))
            print(run.name, a.which, json.dumps(res))
