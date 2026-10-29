"""Evaluate methods on a frozen planar-arm test set (companion SPEC §9), as the UR5e's scripts/evaluate.py.

Methods:
  b2          full B2: RRT-Connect (5 s, exact obstacles, every IK branch) + 200 shortcuts + quintic
  b3          A* on the exact 1-degree C-space grid (every IK branch) + shortcut + quintic, tracked
  base@N      degraded base only (inflated, N collision checks, straight-line fallback), residual = 0
  hybrid@N    degraded base + trained residual policy (--run), base budget N
  policy      a trained (non-residual) policy (--run)
Per-episode rows go to results/eval/2dof/<set>/<method>.jsonl, the summary (mean and 95% bootstrap CI) to
summary.json in the same folder. Methods that use a trained run are keyed "<method>:<run id>".
Gap to best-known path (§9.1) = L(executed) / L_best - 1 on successful episodes, L_best = min(A* + shortcut,
RRT + shortcut) per episode (eval/metrics.py), cached in l_best.json.

    python 2-dof/scripts/evaluate.py --set p3_test --methods b2 b3 --budgets 700 2300 3500 --run <residual run>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from arm2dof.cspace.astar import astar
from arm2dof.cspace.grid import grid
from arm2dof.envs.kinematic_env import EnvConfig, KinematicArmEnv
from arm2dof.envs.residual import ResidualArmEnv
from arm2dof.eval.metrics import best_known_length, trajectory_length
from arm2dof.planners.base import A_MAX, V_MAX, Tracker, goal_configs, plan, space_for
from ur5rl.eval import testsets
from ur5rl.eval.locking import SummaryLock
from ur5rl.planners.quintic import time_scale
from ur5rl.planners.rrt_connect import shortcut

ROOT_ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_ / "scripts"))
import checkpoints  # noqa: E402  UR5e scripts/checkpoints.py: evaluation checkpoint rule

ROOT = Path(__file__).resolve().parents[2]
SETS = ROOT / "2-dof" / "testsets"


def bootstrap_ci(x, n=2000, seed=0):
    """Mean and 95% bootstrap CI, identical to the UR5e scripts/evaluate.py."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    means = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def stage_env(meta):
    sys.path.insert(0, str(ROOT / "2-dof" / "scripts"))
    import dataclasses
    from horizon_check import stage_config
    overrides = json.loads(str(meta["overrides"])) if "overrides" in meta else {}
    overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
    return dataclasses.replace(stage_config(str(meta["stage"])), **overrides)


def run_episode(env, act):
    base = env.unwrapped
    qs, infos, done = [base.q.copy()], [], False
    obs = env._last_obs
    while not done:
        obs, _, term, trunc, info = env.step(act(obs))
        qs.append(base.q.copy())
        infos.append(info)
        done = term or trunc
    return infos, np.array(qs)


def metrics(infos, qs, l_best, latency):
    succ = bool(infos[-1]["is_success"])
    coll = any(i["collision"] for i in infos)
    first = next((k for k, i in enumerate(infos) if i["is_success"]), None)
    # path up to the first success: the reach itself, not the final hold
    length = trajectory_length(qs[: first + 2]) if first is not None else trajectory_length(qs)
    return dict(success=succ, collision=coll, d_pos_cm=100 * infos[-1]["d_pos"],
                time_to_success_s=(first + 1) * 0.04 if first is not None else None, joint_path_rad=length,
                gap=(length / l_best - 1.0) if succ and np.isfinite(l_best) and l_best > 0 else None, latency_s=latency)


def b3_reference(env):
    """A* to the first reachable IK branch, endpoints pinned, shortcut, quintic (companion SPEC §7.2 B3)."""
    space = space_for(env)
    c, r = env.centers[env.mask], env.radii[env.mask]
    occ = grid(c, r)
    from arm2dof.envs.geometry import in_collision
    collides = lambda q: bool(in_collision(q, c, r))
    for q_goal in goal_configs(env):
        q_a, _ = astar(occ, env.q_start, q_goal)
        if q_a is None:
            continue
        path = np.vstack([env.q_start, q_a[1:-1], q_goal]) if len(q_a) > 2 else np.array([env.q_start, q_goal])
        path = shortcut(space, collides, path, np.random.default_rng(0), iters=200, edge_res=0.005)
        ref, _ = time_scale(space.unwrap(path), dt=env.cfg.dt, v_max=V_MAX, a_max=A_MAX)
        return ref
    return None


def load_policy(run, which="best"):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ur5rl.agents.factory import ALGOS
    cfg = json.loads((Path(run) / "config.json").read_text())["config"]
    env_cfg = dict(cfg["env"], n_obstacles=tuple(cfg["env"]["n_obstacles"]), curriculum=False)
    if "fixed_layout" in env_cfg:
        env_cfg["fixed_layout"] = tuple(map(tuple, env_cfg["fixed_layout"]))
    residual = cfg["agent"]["algo"] == "residual_sac"
    make = (lambda: ResidualArmEnv(KinematicArmEnv(EnvConfig(**env_cfg)), **cfg["agent"]["base"])) if residual \
        else (lambda: KinematicArmEnv(EnvConfig(**env_cfg)))
    path, _, note = checkpoints.select(run, which)  # UR5e SPEC §9.2 checkpoint rule, shared
    print(f"{Path(run).name}: {which} checkpoint {path.name} ({note})", flush=True)
    return ALGOS[cfg["agent"]["algo"]].load(path, env=DummyVecEnv([make]), device="cpu"), cfg


def evaluate(method, episodes, env_cfg, l_best, run=None, which="best"):
    rows = []
    kind, _, budget = method.partition("@")
    if kind in ("b2", "b3"):
        env = KinematicArmEnv(env_cfg)
        for i, ep in enumerate(episodes):
            env._last_obs, _ = env.reset(seed=i, options={"episode": ep})
            t0 = time.perf_counter()
            ref = plan(env, np.random.default_rng(i)).reference if kind == "b2" else b3_reference(env)
            lat = time.perf_counter() - t0
            if ref is None:
                rows.append(dict(success=False, collision=False, plan_failed=True, latency_s=lat))
                continue
            tracker = Tracker(env, ref)
            infos, qs = run_episode(env, lambda o: tracker())
            rows.append(metrics(infos, qs, l_best[i], lat))
        return rows
    if kind in ("base", "hybrid", "policy"):
        model, cfg = load_policy(run, which) if kind != "base" else (None, None)
        if kind == "policy":
            env = KinematicArmEnv(env_cfg)
        else:
            base_kw = dict(cfg["agent"]["base"]) if cfg else dict(inflate=0.05, shortcut_iters=20)
            base_kw["max_checks"] = int(budget)
            env = ResidualArmEnv(KinematicArmEnv(env_cfg), alpha=cfg["agent"]["alpha_max"] if cfg else 0.0, **base_kw)
        for i, ep in enumerate(episodes):
            env._last_obs, info0 = env.reset(seed=i, options={"episode": ep})
            t_inf = []

            def act(obs):
                if model is None:
                    return np.zeros(2)
                t0 = time.perf_counter()
                a, _ = model.predict(obs, deterministic=True)
                t_inf.append(time.perf_counter() - t0)
                return a

            infos, qs = run_episode(env, act)
            row = metrics(infos, qs, l_best[i], float(np.mean(t_inf)) if t_inf else 0.0)
            if kind != "policy":
                row["base_ok"] = bool(info0["base_ok"])
            rows.append(row)
        return rows
    raise ValueError(method)


def summarize(rows):
    out = {k: bootstrap_ci([r[k] for r in rows]) for k in ("success", "collision")}
    for k in ("gap", "joint_path_rad", "time_to_success_s", "d_pos_cm", "latency_s"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = bootstrap_ci(vals) if vals else None
    if "base_ok" in rows[0]:
        out["base_ok"] = float(np.mean([r["base_ok"] for r in rows]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True)
    p.add_argument("--methods", nargs="*", default=[])
    p.add_argument("--budgets", nargs="*", type=int, default=[])
    p.add_argument("--run")
    p.add_argument("--limit", type=int, default=None, help="first N episodes only (smoke tests; separate folder)")
    p.add_argument("--checkpoint", default="best", choices=["best", "primary", "final"],
                   help="model of --run (scripts/checkpoints.py); primary/final are keyed '<method>:<run>#<checkpoint>'")
    p.add_argument("--skip-base", action="store_true", help="--budgets adds hybrid@N only")
    args = p.parse_args()
    episodes, meta = testsets.load(args.set, root=SETS)
    episodes = episodes[: args.limit] if args.limit else episodes
    env_cfg = stage_env(meta)
    out = ROOT / "results" / "eval" / "2dof" / (f"{args.set}_first{args.limit}" if args.limit else args.set)
    out.mkdir(parents=True, exist_ok=True)

    lb_path = out / "l_best.json"
    l_best = json.loads(lb_path.read_text()) if lb_path.exists() else []
    if len(l_best) < len(episodes):
        env = KinematicArmEnv(env_cfg)
        for i in range(len(l_best), len(episodes)):
            env.reset(seed=i, options={"episode": episodes[i]})
            q_goal = goal_configs(env)[0]
            l_best.append(best_known_length(env.centers[env.mask], env.radii[env.mask], env.q_start, q_goal,
                                            np.random.default_rng(i))[0])
        lb_path.write_text(json.dumps(l_best))

    methods = list(args.methods) + [m for b in args.budgets for m in
                                    ([] if args.skip_base else [f"base@{b}"]) + ([f"hybrid@{b}"] if args.run else [])]
    for m in methods:
        t0 = time.perf_counter()
        rows = evaluate(m, episodes, env_cfg, l_best, run=args.run, which=args.checkpoint)
        uses_run = m == "policy" or m.startswith("hybrid@")
        key = f"{m}:{Path(args.run).name}" if uses_run else m
        if uses_run and args.checkpoint != "best":
            key += f"#{args.checkpoint}"
        (out / f"{key.replace('@', '_').replace(':', '__').replace('#', '__')}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        entry = dict(summarize(rows), n=len(rows), run=args.run if uses_run else None)
        sp = out / "summary.json"
        with SummaryLock(sp):  # parallel evaluations merge into one summary
            summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
            summary[key] = entry
            sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        s, c, g = entry["success"], entry["collision"], entry["gap"]
        print(f"{key:14} success {s[0]:.3f}  collision {c[0]:.3f}  gap {g[0] if g else float('nan'):.3f}"
              f"  ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
