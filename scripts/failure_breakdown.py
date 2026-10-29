"""Classify the failures of finished runs (SPEC §4 horizon rule) and replay them with a longer horizon.

For each run's final model (last_model.zip), deterministic rollouts on fresh episodes (seeds 2_000_000+i,
disjoint from training, validation and test seeds). Each failure is classified by the last steps:
  collision  any contact in the final 25 steps
  timeout    normalised goal error still decreasing (fell by > 5% over the last 10 steps)
  stuck      otherwise
Failures are then replayed on the same episodes with a 1.5x horizon (the goal bound is an explicit distance,
independent of the horizon, so the sampled episodes are identical). Curriculum is off: full goal distribution.

    python scripts/failure_breakdown.py --exp configs/exp/e1.yaml --episodes 200
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv

ROOT = Path(__file__).resolve().parents[1]
ALGOS = {"sac": SAC, "td3": TD3}
SEED0 = 2_000_000


def rollout(model, env, seed):
    obs, _ = env.reset(seed=seed)
    errs, contacts = [], []
    eps_r = np.deg2rad(env.cfg.eps_rot_deg)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        e = info["d_pos"] / env.cfg.eps_pos + (info["theta"] / eps_r if env.cfg.orientation else 0.0)
        errs.append(e)
        contacts.append(info["collision"])
        if term or trunc:
            return info["is_success"], np.array(errs), np.array(contacts)


def classify(errs, contacts):
    if contacts[-25:].any():
        return "collision"
    if len(errs) > 10 and errs[-11] - errs[-1] > 0.05 * errs[-11]:
        return "timeout"
    return "stuck"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--stretch", type=float, default=1.5)
    p.add_argument("--runs-dir", default=str(ROOT / "results" / "runs"))
    args = p.parse_args()

    exp = yaml.safe_load(Path(args.exp).read_text())
    exp_dir = Path(args.runs_dir) / exp["name"]
    runs = sorted(d for d in exp_dir.iterdir() if (d / "done.json").exists())
    out, totals = {}, {}
    for run in runs:
        meta = json.loads((run / "config.json").read_text())
        cfg = meta["config"]
        env_cfg = dict(cfg["env"], n_obstacles=tuple(cfg["env"]["n_obstacles"]), curriculum=False)
        algo = cfg["agent"]["algo"]
        env = UR5ePoseEnv(EnvConfig(**env_cfg))
        # HER models need an env at load time (the buffer is not loaded, only the policy is used)
        model = ALGOS[algo].load(run / "last_model.zip", env=DummyVecEnv([lambda: UR5ePoseEnv(EnvConfig(**env_cfg))]),
                                 device="cpu")
        long_cfg = EnvConfig(**env_cfg)
        long_cfg.episode_length = int(round(env_cfg.get("episode_length", 100) * args.stretch))
        long_env = UR5ePoseEnv(long_cfg)

        cats, rescued = Counter(), Counter()
        for i in range(args.episodes):
            ok, errs, contacts = rollout(model, env, SEED0 + i)
            if ok:
                cats["success"] += 1
                continue
            c = classify(errs, contacts)
            cats[c] += 1
            ok_long, _, _ = rollout(model, long_env, SEED0 + i)
            rescued[c] += int(ok_long)
        out[run.name] = dict(categories=dict(cats), rescued_by_longer_horizon=dict(rescued))
        key = algo + ("_her" if cfg["her"] else "")
        totals.setdefault(key, [Counter(), Counter()])
        totals[key][0].update(cats)
        totals[key][1].update(rescued)
        n_fail = args.episodes - cats["success"]
        print(f"{run.name}: success {cats['success'] / args.episodes:.1%} | failures {n_fail}: "
              + ", ".join(f"{k} {cats[k]} (rescued {rescued[k]})" for k in ("timeout", "stuck", "collision")))

    print(f"\nPer agent ({args.episodes} episodes per run; rescued = succeeds with a {args.stretch}x horizon):")
    for key, (cats, rescued) in totals.items():
        n = sum(cats.values())
        fails = n - cats["success"]
        print(f"  {key}: success {cats['success'] / n:.1%}; of {fails} failures: "
              + ", ".join(f"{k} {cats[k] / max(fails, 1):.0%} (rescued {rescued[k]}/{cats[k]})"
                          for k in ("timeout", "stuck", "collision")))
    (exp_dir / "failure_breakdown.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
