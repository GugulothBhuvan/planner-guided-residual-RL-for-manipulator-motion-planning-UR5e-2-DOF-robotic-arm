"""Base-only reference line for residual validation curves (SPEC §7.3 residual decision rule).

Replays the residual run's validation exactly (same stored S4-validation episodes, same reset seeds, hence the same
per-episode base-budget draws) with alpha = 0, i.e. the degraded base alone. Writes <exp dir>/base_reference.json.

    python scripts/residual_reference.py --run results/runs/e5/residual_<hash>_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv
from ur5rl.envs.residual import ResidualEnv
from ur5rl.eval import testsets

VAL_SEED0 = 1_000_000  # ValidationEval default, as train.py uses it


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    cfg = json.loads((Path(args.run) / "config.json").read_text())["config"]
    env_cfg = dict(cfg["env"], n_obstacles=tuple(cfg["env"]["n_obstacles"]), curriculum=False)
    env = ResidualEnv(UR5ePoseEnv(EnvConfig(**env_cfg)), alpha=0.0, **cfg["agent"]["base"])
    episodes, _ = testsets.load(f"{cfg['stage']}_val")
    succ, coll, d_pos, budgets, base_ok = [], [], [], [], []
    for i, ep in enumerate(episodes):
        obs, info0 = env.reset(seed=VAL_SEED0 + i, options={"episode": ep})
        done, hit = False, False
        while not done:
            obs, _, term, trunc, info = env.step(np.zeros(6))
            hit |= info["collision"]
            done = term or trunc
        succ.append(info["is_success"])
        coll.append(hit)
        d_pos.append(info["d_pos"])
        budgets.append(info0["base_budget"])
        base_ok.append(info0["base_ok"])
    out = dict(n=len(succ), success=float(np.mean(succ)), collision=float(np.mean(coll)),
               d_pos_cm=float(np.mean(d_pos) * 100), base_ok=float(np.mean(base_ok)),
               budget_counts={int(b): budgets.count(b) for b in sorted(set(budgets))})
    path = Path(args.run).parent / "base_reference.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
