"""Obstacle-stage horizons from B2 durations (companion SPEC §4; same rule as UR5e S4):
T = max(80, ceil(1.25 x p99 B2 duration in control steps)), 80 being the P1 horizon.

Samples episodes from the stage config (P2, P3), plans full B2 (5 s, exact obstacles, 200 shortcuts) and
counts the quintic reference's control steps. Episodes B2 cannot solve are excluded, as the test-set filter
will exclude them (§6.4).

    python 2-dof/scripts/horizon_check.py --stage p2 p3 --episodes 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from arm2dof.envs.kinematic_env import EnvConfig, KinematicArmEnv
from arm2dof.planners.base import plan

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 80


def stage_config(name):
    cfg = yaml.safe_load((ROOT / "configs" / "env" / f"{name}.yaml").read_text(encoding="utf-8"))
    if "fixed_layout" in cfg:
        cfg["fixed_layout"] = tuple(map(tuple, cfg["fixed_layout"]))
    return EnvConfig(**(cfg | dict(curriculum=False)))  # full goal distribution


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", nargs="+", default=["p2", "p3"])
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed0", type=int, default=7_000_000)  # outside every test/val seed range
    args = p.parse_args()
    out = {}
    for name in args.stage:
        env = KinematicArmEnv(stage_config(name))
        steps, failed = [], 0
        for i in range(args.episodes):
            env.reset(seed=args.seed0 + i)
            pl = plan(env, np.random.default_rng(i))
            if pl.ok:
                steps.append(len(pl.reference))
            else:
                failed += 1
        s = np.array(steps)
        p99 = float(np.percentile(s, 99))
        horizon = max(FLOOR, int(np.ceil(1.25 * p99)))
        out[name] = dict(episodes=args.episodes, b2_failed=failed, median=float(np.median(s)), p90=float(np.percentile(s, 90)),
                         p99=p99, max=int(s.max()), horizon=horizon, current=env.cfg.episode_length)
        print(name, json.dumps(out[name]), flush=True)
    path = ROOT / "configs" / "env" / "horizon_check.json"
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    path.write_text(json.dumps(prev | out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
