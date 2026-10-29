"""Frozen test and validation sets (SPEC §9.2) and the S4 horizon from B2 durations (SPEC §4).

Writes testsets/<name>.npz (q_start, q_goal, obstacle layout, seeds, horizon) and testsets/summary.json.
  test:       s1, s2, s4 (200 episodes each), s4_ood (100: 4 larger spheres, boundary goals)
  validation: s1, s2, s4 (100 each; seeds 1_000_000+, the same episodes ValidationEval uses)
S4 / S4-OOD episodes must admit a collision-free path: B2 (2 s budget) is tried first, then a 10 s RRT budget;
an episode with neither is replaced by the next seed. B2 durations on the S4 test set set the S4 horizon:
H = max(200, ceil(1.25 * p99 duration / control_dt)); configs/env/s4.yaml is updated to H.

    python scripts/make_testsets.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv
from ur5rl.planners.baselines import plan_b2
from ur5rl.planners.collision import CollisionChecker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testsets"
SETS = [  # name, stage, n, first seed, env overrides
    ("s1_test", "s1", 200, 3_000_000, {}),
    ("s2_test", "s2", 200, 3_100_000, {}),
    ("s4_test", "s4", 200, 3_200_000, {}),
    ("s4_ood_test", "s4", 100, 3_300_000, dict(n_obstacles=(4, 4), obstacle_radius=(0.08, 0.14), goal_min_reach=0.75)),
    # validation seed ranges are distinct per stage (S1 and S2 sample identically, so equal seeds = equal episodes)
    ("s1_val", "s1", 100, 1_000_000, {}),
    ("s2_val", "s2", 100, 1_100_000, {}),
    ("s4_val", "s4", 100, 1_200_000, {}),
]


def stage_cfg(stage, overrides):
    cfg = yaml.safe_load((ROOT / "configs" / "env" / f"{stage}.yaml").read_text())
    cfg.update(overrides)
    cfg["n_obstacles"] = tuple(cfg["n_obstacles"])
    return cfg


def build(name, stage, n, seed0, overrides, checker, rng):
    env = UR5ePoseEnv(EnvConfig(**stage_cfg(stage, overrides)))
    eps, seeds, b2 = [], [], []
    seed, rejected = seed0, 0
    while len(eps) < n:
        env.reset(seed=seed)
        ep = env.episode()
        if env.use_obstacles:
            p = plan_b2(env, rng, checker)
            if not p.ok:  # B2 failed within 2 s: is there any path at all?
                q = plan_b2(env, rng, checker, budget=10.0, shortcut_iters=0)
                if not q.ok:
                    rejected += 1
                    seed += 1
                    continue
            b2.append((p.ok, p.duration, p.plan_time))
        eps.append(ep)
        seeds.append(seed)
        seed += 1
    arrays = {k: np.stack([e[k] for e in eps]) for k in eps[0]}
    arrays["seeds"] = np.array(seeds)
    if b2:
        arrays["b2_ok"] = np.array([b[0] for b in b2])
        arrays["b2_duration"] = np.array([b[1] for b in b2])
        arrays["b2_plan_time"] = np.array([b[2] for b in b2])
    return arrays, rejected, env.cfg.frame_skip * env.cfg.timestep


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="rebuild only these sets; others and the S4 horizon stay as they are")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    checker, rng = CollisionChecker(), np.random.default_rng(0)
    old = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    summary, built = dict(old), {}
    for name, stage, n, seed0, over in SETS:
        if args.only and name not in args.only:
            continue
        arrays, rejected, control_dt = build(name, stage, n, seed0, over, checker, rng)
        built[name] = (arrays, stage, over, control_dt)
        summary[name] = dict(stage=stage, n=n, first_seed=seed0, rejected_no_path=rejected, overrides=over)
        if "b2_ok" in arrays:
            d = arrays["b2_duration"][arrays["b2_ok"]]
            summary[name].update(b2_success=float(arrays["b2_ok"].mean()), b2_duration_p50=float(np.median(d)),
                                 b2_duration_p99=float(np.percentile(d, 99)), b2_duration_max=float(d.max()),
                                 b2_plan_time_p50=float(np.median(arrays["b2_plan_time"][arrays["b2_ok"]])))
        print(name, json.dumps(summary[name]), flush=True)

    if "s4_test" in built:  # S4 horizon rule (SPEC §4), from the S4 test set's B2 durations
        s4 = summary["s4_test"]
        control_dt = built["s4_test"][3]
        horizon = max(200, math.ceil(1.25 * s4["b2_duration_p99"] / control_dt))
        summary["s4_horizon"] = dict(rule="max(200, ceil(1.25 * p99 B2 duration / 0.04 s))",
                                     p99_s=s4["b2_duration_p99"], horizon_steps=horizon)
        s4_yaml = ROOT / "configs" / "env" / "s4.yaml"
        text = s4_yaml.read_text(encoding="utf-8")
        line = next(ln for ln in text.splitlines() if ln.startswith("episode_length:"))
        s4_yaml.write_text(text.replace(line, f"episode_length: {horizon}        # SPEC §4 rule from B2 durations "
                                              f"(p99 {s4['b2_duration_p99']:.2f} s)"), encoding="utf-8")
    horizon = summary["s4_horizon"]["horizon_steps"]

    for name, (arrays, stage, over, _) in built.items():
        h = horizon if stage == "s4" else stage_cfg(stage, {})["episode_length"]
        np.savez_compressed(OUT / f"{name}.npz", horizon=h, stage=stage, overrides=json.dumps(over), **arrays)
        summary[name]["horizon"] = h
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("S4 horizon:", json.dumps(summary["s4_horizon"]))


if __name__ == "__main__":
    main()
