"""Frozen planar-arm test and validation sets (companion SPEC §9.2), as the UR5e's scripts/make_testsets.py.

Writes 2-dof/testsets/<name>.npz (q_start, q_goal, obstacle layout, seeds, horizon, B2 durations) and summary.json.
  test:       p1, p2, p3 (1000 episodes each, §9.2; seeds 5.0M / 5.1M / 5.2M) and p3_ood (300; all 4 circle slots,
              radius 0.10-0.20 m instead of 1-4 circles of 0.05-0.15 m; seeds 5.3M)
  validation: p1, p2, p3 (100 each; seeds 4.0M / 4.1M / 4.2M)
Seed ranges are disjoint from each other and from every UR5e range (1.0M-1.2M, 3.0M-3.3M). Episodes come from the
stage config with the curriculum off (full distribution). P2/P3 episodes must be solved by full B2 (5 s, every IK
branch); an episode B2 cannot solve is replaced by the next seed (the env already guarantees C-space connectivity,
so this is expected to reject nothing).

    python 2-dof/scripts/make_testsets.py [--only p3_test ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from horizon_check import stage_config  # noqa: E402

from arm2dof.envs.kinematic_env import KinematicArmEnv  # noqa: E402
from arm2dof.planners.base import plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testsets"
SETS = [  # name, stage, n, first seed, env overrides
    ("p1_test", "p1", 1000, 5_000_000, {}), ("p2_test", "p2", 1000, 5_100_000, {}),
    ("p3_test", "p3", 1000, 5_200_000, {}),
    ("p3_ood_test", "p3", 300, 5_300_000, dict(n_obstacles=(4, 4), obstacle_radius=(0.10, 0.20))),
    ("p1_val", "p1", 100, 4_000_000, {}), ("p2_val", "p2", 100, 4_100_000, {}), ("p3_val", "p3", 100, 4_200_000, {})]
KEYS = ("q_start", "q_goal", "centers", "radii", "mask")


def build(stage, n, seed0, overrides):
    import dataclasses
    env = KinematicArmEnv(dataclasses.replace(stage_config(stage), **overrides))
    eps, seeds, b2_steps, rejected, seed = [], [], [], 0, seed0
    while len(eps) < n:
        env.reset(seed=seed)
        if env.use_obstacles:
            p = plan(env, np.random.default_rng(seed))
            if not p.ok:
                rejected += 1
                seed += 1
                continue
            b2_steps.append(len(p.reference))
        eps.append(env.episode())
        seeds.append(seed)
        seed += 1
    return env, eps, seeds, b2_steps, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for name, stage, n, seed0, overrides in SETS:
        if args.only and name not in args.only:
            continue
        env, eps, seeds, b2, rejected = build(stage, n, seed0, overrides)
        arrays = {k: np.stack([e[k] for e in eps]) for k in KEYS}
        np.savez(OUT / f"{name}.npz", seeds=np.array(seeds), stage=stage, horizon=env.cfg.episode_length,
                 overrides=json.dumps(overrides),
                 b2_steps=np.array(b2), **arrays)
        summary[name] = dict(stage=stage, n=n, seeds=[seeds[0], seeds[-1]], rejected=rejected,
                             horizon=env.cfg.episode_length,
                             b2_steps=dict(median=float(np.median(b2)), p99=float(np.percentile(b2, 99)),
                                           max=int(max(b2))) if b2 else None)
        print(name, json.dumps(summary[name]), flush=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # disjointness check over every stored set (seed ranges)
    names = [s[0] for s in SETS if (OUT / f"{s[0]}.npz").exists()]
    seen = {}
    for nm in names:
        for sd in np.load(OUT / f"{nm}.npz")["seeds"]:
            assert sd not in seen, f"seed {sd} in both {seen[sd]} and {nm}"
            seen[sd] = nm
    print("all sets disjoint by seed:", names)


if __name__ == "__main__":
    main()
