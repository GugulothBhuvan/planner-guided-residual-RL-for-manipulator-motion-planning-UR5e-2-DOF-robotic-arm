"""Base-only reference for a 2-DOF residual run's validation (the UR5e scripts/residual_reference.py).

Replays the run's training-time validation episodes (ValidationEval without a stored set: reset seeds 1_000_000 + i,
100 episodes) with a zero residual: B5' (PD + gravity, nominal model) alone for P5a (PE8), the degraded B2 base for
the kinematic stages (PE3 / PE4; the per-episode budget draw is the training one, as in validation). Writes
<run>/base_reference.json.

    python 2-dof/scripts/base_reference.py --run results/runs/pe8/residual_<hash>_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arm2dof.envs.kinematic_env import EnvConfig, KinematicArmEnv
from arm2dof.envs.residual import ResidualArmEnv
from arm2dof.envs.residual_torque import ResidualTorqueEnv
from arm2dof.envs.torque_env import TorqueArmEnv, TorqueConfig

VAL_SEED0, N = 1_000_000, 100  # ValidationEval defaults, as train.py uses them


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    cfg = json.loads((Path(args.run) / "config.json").read_text())["config"]
    if str(cfg["stage"]).startswith("p5"):
        env = ResidualTorqueEnv(TorqueArmEnv(TorqueConfig(**cfg["env"])), alpha=0.0, kind=cfg["agent"]["base"]["kind"])
        dim = 2
    else:
        env_cfg = dict(cfg["env"], n_obstacles=tuple(cfg["env"]["n_obstacles"]), curriculum=False)
        if "fixed_layout" in env_cfg:
            env_cfg["fixed_layout"] = tuple(map(tuple, env_cfg["fixed_layout"]))
        env = ResidualArmEnv(KinematicArmEnv(EnvConfig(**env_cfg)), alpha=0.0, **cfg["agent"]["base"])
        dim = env.action_space.shape[0]
    succ, d_pos, length, coll = [], [], [], []
    for i in range(N):
        env.reset(seed=VAL_SEED0 + i)
        done, t, hit = False, 0, False
        while not done:
            _, _, term, trunc, info = env.step(np.zeros(dim))
            done, t = term or trunc, t + 1
            hit = hit or bool(info.get("collision", False))
        succ.append(info["is_success"])
        d_pos.append(info["d_pos"])
        length.append(t)
        coll.append(hit)
    out = dict(n=N, base=cfg["agent"]["base"].get("kind", "degraded B2"), success=float(np.mean(succ)),
               collision=float(np.mean(coll)),
               d_pos_cm=float(np.mean(d_pos) * 100), ep_len=float(np.mean(length)))
    (Path(args.run) / "base_reference.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
