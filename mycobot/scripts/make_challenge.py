"""Challenging-but-simple test set `s4_chal` (SPEC M4b): four scene types on top of the s4 sampler, no clutter beyond 4 spheres.

  gap     two spheres flank the straight tool path at its middle with a 5-8 cm clear gap between their surfaces
  clutter four spheres within a few cm of the straight tool path (s4 uses 1-4 spheres within +-5 cm, uniformly)
  limit   goal with the shoulder (J2) pushed 0.75-0.9 rad above the start, so a joint is within 16-30 deg of its limit
          (the closest the s4 protocol allows: starts are near the home pose and goals within 0.9 rad of the start),
          1-3 s4-style spheres
  edge    goal tool position >= 0.27 m from the shoulder axis (near full reach), 1-3 s4-style spheres

Common rules (as s4): start and goal collision-free and >= 1 cm clear of every sphere, goal <= 0.9 rad from the start in
every joint, B2 finds a path within 20 s (otherwise the episode is replaced). The training distribution never contained
the first two types by construction and only rarely the last two. 25 episodes per type (100 in all).

    python scripts/make_challenge.py            # writes testsets/s4_chal.npz and its entry in testsets/summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from mcrl.envs.pose_env import EnvConfig, MyCobotPoseEnv
from mcrl.planners.baselines import plan_b2
from mcrl.planners.collision import CollisionChecker
from mcrl.robot import LIMITS_RAD, SHOULDER

ROOT = Path(__file__).resolve().parents[1]
CATS = ["gap", "clutter", "limit", "edge"]
N_PER = 25
MAX_B2_DURATION = 20.0
HORIZON = 900


def cfg_s4(**over):
    cfg = yaml.safe_load((ROOT / "configs" / "env" / "s4.yaml").read_text())
    cfg.update(over)
    cfg["n_obstacles"] = tuple(cfg["n_obstacles"])
    return EnvConfig(**cfg)


def place(env, centers, radii, q_start, q_goal):
    """True if every sphere is inside the workspace band and clear (>= 1 cm) at the start and the goal."""
    for c, r in zip(centers, radii):
        radial = np.linalg.norm(c[:2])
        if not (0.04 < radial < 0.32 and r < c[2] < 0.45):
            return False
        if not (env._clear_at(q_start, c[None], np.array([r])) and env._clear_at(q_goal, c[None], np.array([r]))):
            return False
    return True


def sample(cat, env, rng):
    """One candidate episode of this type (not yet checked for B2 feasibility), or None."""
    q_start = env._sample_start()
    if cat == "limit":
        # Starts lie within ~17 deg of the home pose and goals within 0.9 rad of the start, so under the s4 protocol no
        # joint can get within ~16 deg of a limit except the shoulder (J2, upper limit 85 deg): push J2 up by 0.75-0.9 rad.
        for _ in range(400):
            q = env._sample_goal_q(q_start)
            q[1] = q_start[1] + rng.uniform(0.75, 0.9)
            margin = np.rad2deg(np.minimum(q - LIMITS_RAD[:, 0], LIMITS_RAD[:, 1] - q))
            if margin.min() <= 30.0 and not env._in_collision(q) and env._tcp_pose(q)[0][2] > 0.0:
                q_goal = q
                break
        else:
            return None
    else:  # edge uses goal_min_reach set on the env; gap/clutter use the plain s4 goal
        q_goal = env._sample_goal_q(q_start)
    p0, p1 = env._tcp_pose(q_start)[0], env._tcp_pose(q_goal)[0]
    K = env.K
    centers, radii = np.zeros((K, 3)), np.full(K, 0.05)
    if cat in ("limit", "edge"):
        env._sample_obstacles(q_start, q_goal)
        centers, radii, n = env.ob_center.copy(), env.ob_radius.copy(), int(env.ob_mask.sum())
        n = min(max(n, 1), 3)
        mask = np.arange(K) < n
        if env.ob_mask.sum() == 0:
            return None
        return dict(q_start=q_start, q_goal=q_goal, centers=centers, radii=radii, mask=mask)
    u = p1 - p0
    length = np.linalg.norm(u)
    if length < 0.08:
        return None
    u /= length
    if cat == "gap":
        r = rng.uniform(0.025, 0.035, 2)
        t = rng.uniform(0.4, 0.6)
        w = np.cross(u, rng.normal(size=3))
        w /= np.linalg.norm(w)
        mid = p0 + t * (p1 - p0)
        gap = rng.uniform(0.05, 0.08)
        cs = np.stack([mid + (r[0] + gap / 2) * w, mid - (r[1] + gap / 2) * w])
        centers[:2], radii[:2], n = cs, r, 2
    else:  # clutter: four spheres close to the path
        n = 4
        ts = np.array([0.25, 0.45, 0.65, 0.85]) + rng.uniform(-0.05, 0.05, 4)
        for k in range(4):
            r = rng.uniform(0.025, 0.035)
            off = rng.normal(size=3)
            off -= off.dot(u) * u
            off = off / np.linalg.norm(off) * rng.uniform(0.0, 0.035)
            centers[k], radii[k] = p0 + ts[k] * (p1 - p0) + off, r
    mask = np.arange(K) < n
    if not place(env, centers[:n], radii[:n], q_start, q_goal):
        return None
    return dict(q_start=q_start, q_goal=q_goal, centers=centers, radii=radii, mask=mask)


def main():
    checker, rng = CollisionChecker(), np.random.default_rng(20261103)
    eps, cats, b2d, tried = [], [], [], {c: 0 for c in CATS}
    for ci, cat in enumerate(CATS):
        env = MyCobotPoseEnv(cfg_s4(goal_min_reach=0.27 if cat == "edge" else None))
        env.reset(seed=int(rng.integers(1 << 30)))
        got = 0
        while got < N_PER:
            tried[cat] += 1
            ep = sample(cat, env, rng)
            if ep is None:
                continue
            env.reset(seed=0, options={"episode": ep})
            p = plan_b2(env, rng, checker)
            if not p.ok or p.duration > MAX_B2_DURATION:
                continue
            eps.append(ep)
            cats.append(ci)
            b2d.append(p.duration)
            got += 1
        print(f"{cat}: {N_PER} episodes from {tried[cat]} candidates", flush=True)
    arrays = {k: np.stack([e[k] for e in eps]) for k in eps[0]}
    arrays["seeds"] = np.arange(len(eps))
    arrays["cat"] = np.array(cats)
    arrays["b2_duration"] = np.array(b2d)
    arrays["b2_ok"] = np.ones(len(eps), bool)
    out = ROOT / "testsets"
    np.savez_compressed(out / "s4_chal.npz", horizon=HORIZON, stage="s4", overrides=json.dumps({}), **arrays)
    summ = json.loads((out / "summary.json").read_text())
    summ["s4_chal"] = dict(stage="s4", n=len(eps), categories=CATS, per_category=N_PER, candidates=tried, horizon=HORIZON,
                           b2_duration_p50=float(np.median(b2d)), b2_duration_max=float(np.max(b2d)),
                           note="SPEC M4b: gap / clutter / limit / edge; generator scripts/make_challenge.py (rng seed 20261103)")
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ["s4_chal"]))


if __name__ == "__main__":
    main()
