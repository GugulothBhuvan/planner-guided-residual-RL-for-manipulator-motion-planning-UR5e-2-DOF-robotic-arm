"""Cluttered scenes with non-spherical obstacles, test set `s4_shape` (SPEC M4c).

Same layout as the `clutter` type of s4_chal (four objects whose centres lie within 0-3.5 cm of the straight tool path at
t = 0.25 / 0.45 / 0.65 / 0.85), but the objects are
  box       half extents U(1.2, 2.5) cm per axis, uniformly random orientation
  cylinder  radius U(1.2, 2.2) cm, half height U(2.0, 3.5) cm, uniformly random orientation
  mixed     each object drawn from sphere (r 2.5-3.5 cm) / box / cylinder
Rules as s4: start and goal collision-free and >= 1 cm clear of the objects' enclosing spheres (so also of the exact
shapes, and every sphere-based consumer has a valid episode), goal <= 0.9 rad from the start in every joint, B2 (exact
shapes) finds a path within 20 s, horizon 900. 25 episodes per type. `radii` holds the enclosing-sphere radius, which is
all the policy observes (mcrl/shapes.py).

    python scripts/make_shapes.py          # writes testsets/s4_shape.npz and its entry in testsets/summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_challenge import HORIZON, MAX_B2_DURATION, N_PER, cfg_s4, place  # noqa: E402

from mcrl import shapes  # noqa: E402
from mcrl.envs.pose_env import MyCobotPoseEnv  # noqa: E402
from mcrl.planners.baselines import plan_b2  # noqa: E402
from mcrl.planners.collision import CollisionChecker  # noqa: E402

CATS = ["box", "cylinder", "mixed"]
TIME_CAP = 1800.0  # s per type: stop with an error instead of searching forever


def draw_object(kind, rng):
    """(shape, size, quat, enclosing radius) of one object."""
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    if kind == shapes.BOX:
        size = rng.uniform(0.012, 0.025, 3)
    elif kind == shapes.CYLINDER:
        size = np.array([rng.uniform(0.012, 0.022), rng.uniform(0.02, 0.035), 0.0])
    else:
        return shapes.SPHERE, np.zeros(3), np.array([1.0, 0, 0, 0]), rng.uniform(0.025, 0.035)
    return kind, size, quat, shapes.bounding_radius(kind, size)


def sample(cat, env, rng):
    q_start = env._sample_start()
    q_goal = env._sample_goal_q(q_start)
    p0, p1 = env._tcp_pose(q_start)[0], env._tcp_pose(q_goal)[0]
    u = p1 - p0
    length = np.linalg.norm(u)
    if length < 0.08:
        return None
    u /= length
    K = env.K
    centers, radii = np.zeros((K, 3)), np.full(K, 0.05)
    shape, size, quat = shapes.default_arrays(K)
    ts = np.array([0.25, 0.45, 0.65, 0.85]) + rng.uniform(-0.05, 0.05, 4)
    for k in range(4):
        kind = {"box": shapes.BOX, "cylinder": shapes.CYLINDER}.get(cat) or int(rng.integers(3))
        shape[k], size[k], quat[k], radii[k] = draw_object(kind, rng)
        off = rng.normal(size=3)
        off -= off.dot(u) * u
        off = off / np.linalg.norm(off) * rng.uniform(0.0, 0.035)
        centers[k] = p0 + ts[k] * (p1 - p0) + off
    if cat == "mixed" and not shape.any():
        return None
    mask = np.arange(K) < 4
    if not place(env, centers, radii[:4], q_start, q_goal):
        return None
    return dict(q_start=q_start, q_goal=q_goal, centers=centers, radii=radii, mask=mask, shape=shape, size=size, quat=quat)


def main():
    checker, rng = CollisionChecker(), np.random.default_rng(20261105)
    eps, cats, b2d, tried = [], [], [], {c: 0 for c in CATS}
    env = MyCobotPoseEnv(cfg_s4())
    env.reset(seed=int(rng.integers(1 << 30)))
    for ci, cat in enumerate(CATS):
        got, t0, last = 0, time.time(), time.time()
        while got < N_PER:
            if time.time() - t0 > TIME_CAP:
                raise SystemExit(f"{cat}: only {got} episodes after {TIME_CAP:.0f} s ({tried[cat]} candidates)")
            if time.time() - last > 60:
                print(f"  {cat}: {got}/{N_PER} after {tried[cat]} candidates", flush=True)
                last = time.time()
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
    np.savez_compressed(out / "s4_shape.npz", horizon=HORIZON, stage="s4", overrides=json.dumps({}), **arrays)
    summ = json.loads((out / "summary.json").read_text())
    summ["s4_shape"] = dict(stage="s4", n=len(eps), categories=CATS, per_category=N_PER, candidates=tried, horizon=HORIZON,
                            b2_duration_p50=float(np.median(b2d)), b2_duration_max=float(np.max(b2d)),
                            note="SPEC M4c: cluttered boxes / cylinders / mixed; generator scripts/make_shapes.py (rng seed 20261105)")
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ["s4_shape"]))


if __name__ == "__main__":
    main()
