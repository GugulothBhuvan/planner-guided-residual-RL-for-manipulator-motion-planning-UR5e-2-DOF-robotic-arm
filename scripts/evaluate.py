"""Evaluate methods on a frozen test set (SPEC §9).

Methods:
  b1                 IK + quintic (S1/S2)
  b2                 full B2: RRT-Connect (2 s, exact obstacles) + 200 shortcuts + quintic (S4 reference)
  base@N             degraded base only (inflated, N collision checks, straight-line fallback), residual = 0
  hybrid@N           degraded base + trained residual policy (--run), base budget N
  policy             a trained (non-residual) policy (--run)
Per-episode rows go to results/eval/<set>/<method>.jsonl; the summary (mean and 95% bootstrap CI) to
results/eval/<set>/summary.json. Methods that use a trained run are keyed "<method>:<run id>", so seeds sit side by
side. --limit runs write to results/eval/<set>_first<N>/ and never mix with full-set results (SPEC §7.3: reported
numbers use the whole test set).

E5 primary figure (SPEC §3.1, §7.3):
    python scripts/evaluate.py --set s4_test --methods b2 --budgets 1500 3000 6000 --run results/runs/e5/residual_..._s0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv
from ur5rl.envs.residual import ResidualEnv
from ur5rl.eval import testsets
from ur5rl.eval.locking import SummaryLock
from ur5rl.planners.baselines import ReferenceTracker, plan_b1, plan_b2
from ur5rl.planners.collision import CollisionChecker

import checkpoints  # scripts/checkpoints.py: evaluation checkpoint rule

ROOT = Path(__file__).resolve().parents[1]
CONTROL_DT = 0.04


def jerk(qs, dt=CONTROL_DT):
    """Mean squared joint jerk (rad^2/s^6) by third finite differences at the control rate (§9.1)."""
    if len(qs) < 4:
        return 0.0
    return float((np.diff(qs, n=3, axis=0) / dt**3).__pow__(2).sum(1).mean())


def episode_metrics(infos, qs, tcps, latency):
    final = infos[-1]
    succ_steps = [i for i, inf in enumerate(infos) if inf["is_success"]]
    clear = [inf["min_clearance"] for inf in infos if np.isfinite(inf.get("min_clearance", np.inf))]
    return dict(
        success=bool(final["is_success"]),
        collision=bool(any(inf["collision"] for inf in infos)),
        d_pos_cm=100 * final["d_pos"],
        theta_deg=float(np.rad2deg(final["theta"])),
        time_to_success_s=(succ_steps[0] + 1) * CONTROL_DT if succ_steps else None,
        joint_path_rad=float(np.abs(np.diff(qs, axis=0)).sum()),
        cart_path_m=float(np.linalg.norm(np.diff(tcps, axis=0), axis=1).sum()),
        jerk=jerk(qs),
        min_clearance_m=float(min(clear)) if clear else None,
        latency_s=latency,
    )


def run_episode(env, act, base_env=None):
    """Steps `env` with act(obs) until done; returns per-step infos, joint and TCP trajectories."""
    base = base_env or env
    obs = env._last_obs
    infos, qs, tcps = [], [base.data.qpos[:6].copy()], [base.data.site_xpos[base.tcp_id].copy()]
    while True:
        obs, _, term, trunc, info = env.step(act(obs))
        infos.append(info)
        qs.append(base.data.qpos[:6].copy())
        tcps.append(base.data.site_xpos[base.tcp_id].copy())
        if term or trunc:
            return infos, np.array(qs), np.array(tcps)


def stage_env(meta):
    stage = str(meta["stage"])
    cfg = yaml.safe_load((ROOT / "configs" / "env" / f"{stage}.yaml").read_text())
    cfg.update(json.loads(str(meta["overrides"])))
    cfg.update(episode_length=int(meta["horizon"]), n_obstacles=tuple(cfg["n_obstacles"]), curriculum=False)
    if stage == "s4":
        cfg["reward"] = "dense"  # metrics do not depend on the reward; the residual policy was trained dense
    return cfg


def load_policy(run, which="best"):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ur5rl.agents.factory import ALGOS
    cfg = json.loads((Path(run) / "config.json").read_text())["config"]
    path, steps, note = checkpoints.select(run, which)
    print(f"{Path(run).name}: {which} checkpoint {path.name} ({note})", flush=True)
    env_cfg = dict(cfg["env"], n_obstacles=tuple(cfg["env"]["n_obstacles"]), curriculum=False)
    make = (lambda: ResidualEnv(UR5ePoseEnv(EnvConfig(**env_cfg)), **cfg["agent"]["base"])) \
        if cfg["agent"]["algo"] == "residual_sac" else (lambda: UR5ePoseEnv(EnvConfig(**env_cfg)))
    model = ALGOS[cfg["agent"]["algo"]].load(path, env=DummyVecEnv([make]), device="cpu")
    return model, cfg


def evaluate(method, episodes, env_cfg, run=None, seed=0, which="best"):
    rng = np.random.default_rng(seed)
    rows = []
    if method in ("b1", "b2"):
        env, checker = UR5ePoseEnv(EnvConfig(**env_cfg)), CollisionChecker()
        for i, ep in enumerate(episodes):
            env._last_obs, _ = env.reset(seed=i, options={"episode": ep})
            plan = plan_b1(env, checker) if method == "b1" else plan_b2(env, rng, checker)
            if not plan.ok:
                rows.append(dict(success=False, collision=False, plan_failed=True, latency_s=plan.plan_time))
                continue
            tracker = ReferenceTracker(env, plan.reference)
            infos, qs, tcps = run_episode(env, lambda o: tracker())
            rows.append(dict(episode_metrics(infos, qs, tcps, plan.plan_time), plan_failed=False))
        return rows

    kind, _, budget = method.partition("@")
    if kind in ("base", "hybrid"):
        model, cfg = load_policy(run, which) if kind == "hybrid" else (None, None)
        base_kw = dict(cfg["agent"]["base"]) if cfg else dict(inflate=0.05, shortcut_iters=20)
        base_kw["max_checks"] = int(budget)
        alpha = cfg["agent"]["alpha_max"] if cfg else 0.0
        env = ResidualEnv(UR5ePoseEnv(EnvConfig(**env_cfg)), alpha=alpha, **base_kw)
        for i, ep in enumerate(episodes):
            env._last_obs, info0 = env.reset(seed=i, options={"episode": ep})
            t_inf = []

            def act(obs):
                if model is None:
                    return np.zeros(6)
                t0 = time.perf_counter()
                a, _ = model.predict(obs, deterministic=True)
                t_inf.append(time.perf_counter() - t0)
                return a

            infos, qs, tcps = run_episode(env, act, base_env=env.unwrapped)
            lat = info0["base_plan_time"] + (float(np.mean(t_inf)) if t_inf else 0.0)
            rows.append(dict(episode_metrics(infos, qs, tcps, lat), base_ok=bool(info0["base_ok"])))
        return rows

    if kind == "policy":
        model, _ = load_policy(run, which)
        env = UR5ePoseEnv(EnvConfig(**env_cfg))
        for i, ep in enumerate(episodes):
            env._last_obs, _ = env.reset(seed=i, options={"episode": ep})
            t_inf = []

            def act(obs):
                t0 = time.perf_counter()
                a, _ = model.predict(obs, deterministic=True)
                t_inf.append(time.perf_counter() - t0)
                return a

            infos, qs, tcps = run_episode(env, act)
            rows.append(episode_metrics(infos, qs, tcps, float(np.mean(t_inf))))
        return rows
    raise ValueError(method)


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    means = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(rows):
    out = {}
    for key in ("success", "collision"):
        out[key] = bootstrap_ci([r[key] for r in rows])
    for key in ("jerk", "min_clearance_m", "time_to_success_s", "d_pos_cm", "joint_path_rad", "latency_s"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = bootstrap_ci(vals) if vals else None
    if "base_ok" in rows[0]:
        out["base_ok"] = float(np.mean([r["base_ok"] for r in rows]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True)
    p.add_argument("--methods", nargs="*", default=[])
    p.add_argument("--budgets", nargs="*", type=int, default=[], help="adds base@N (and hybrid@N with --run)")
    p.add_argument("--run", help="trained run dir (policy or residual)")
    p.add_argument("--limit", type=int, default=None, help="first N episodes only (smoke tests)")
    p.add_argument("--checkpoint", default="best", choices=["best", "primary", "final"],
                   help="model of --run (scripts/checkpoints.py); primary/final are keyed '<method>:<run>#<checkpoint>'")
    p.add_argument("--skip-base", action="store_true", help="--budgets adds hybrid@N only (base@N already evaluated)")
    args = p.parse_args()

    episodes, meta = testsets.load(args.set)
    episodes = episodes[:args.limit] if args.limit else episodes
    env_cfg = stage_env(meta)
    methods = list(args.methods)
    for b in args.budgets:
        if not args.skip_base:
            methods.append(f"base@{b}")
        if args.run:
            methods.append(f"hybrid@{b}")
    out = ROOT / "results" / "eval" / (f"{args.set}_first{args.limit}" if args.limit else args.set)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    for m in methods:
        t0 = time.perf_counter()
        rows = evaluate(m, episodes, env_cfg, run=args.run, which=args.checkpoint)
        uses_run = m == "policy" or m.startswith("hybrid@")
        key = f"{m}:{Path(args.run).name}" if uses_run else m
        if uses_run and args.checkpoint != "best":
            key += f"#{args.checkpoint}"
        (out / f"{key.replace('@', '_').replace(':', '__').replace('#', '__')}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        entry = dict(summarize(rows), n=len(rows), run=args.run if uses_run else None)
        # re-read before merging, so evaluations of other methods running in parallel are kept
        with SummaryLock(summary_path):
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            summary[key] = entry
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        s, c = entry["success"], entry["collision"]
        print(f"{key:14} success {s[0]:.3f} [{s[1]:.3f}, {s[2]:.3f}]  collision {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]"
              f"  ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
