"""Best-known path length per test episode (companion SPEC §9.1), computed in parallel and cached where
2-dof/scripts/evaluate.py reads it (results/eval/2dof/<set>/l_best.json). Same computation and per-episode seeds as
evaluate.py (reset seed i, rng(i)), so the values are identical; evaluate.py computes them serially, and a few
episodes take ~1 min each (A* + RRT on a hard scene), which made the 1000-episode sets take hours (2026-10-29).

    python 2-dof/scripts/precompute_lbest.py --sets p2_test p3_test --workers 6
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def ev():
    """The 2-DOF evaluate.py by path: it puts the UR5e scripts/ first on sys.path, so a plain `import evaluate` in a
    worker process finds the UR5e evaluate.py instead."""
    import importlib.util
    import sys
    if "evaluate_2dof" not in sys.modules:
        spec = importlib.util.spec_from_file_location("evaluate_2dof", Path(__file__).with_name("evaluate.py"))
        sys.modules["evaluate_2dof"] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules["evaluate_2dof"]


def one(args):
    set_name, i = args
    SETS, stage_env = ev().SETS, ev().stage_env
    from arm2dof.envs.kinematic_env import KinematicArmEnv
    from arm2dof.eval.metrics import best_known_length
    from arm2dof.planners.base import goal_configs
    from ur5rl.eval import testsets
    episodes, meta = testsets.load(set_name, root=SETS)
    env = KinematicArmEnv(stage_env(meta))
    env.reset(seed=i, options={"episode": episodes[i]})
    q_goal = goal_configs(env)[0]
    return i, best_known_length(env.centers[env.mask], env.radii[env.mask], env.q_start, q_goal,
                                np.random.default_rng(i))[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    SETS = ev().SETS
    from ur5rl.eval import testsets
    with Pool(args.workers) as pool:
        for set_name in args.sets:
            out = ROOT / "results" / "eval" / "2dof" / set_name / "l_best.json"
            n = len(testsets.load(set_name, root=SETS)[0])
            if out.exists() and len(json.loads(out.read_text())) >= n:
                continue
            t0 = time.time()
            vals = dict(pool.imap_unordered(one, [(set_name, i) for i in range(n)], chunksize=1))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps([vals[i] for i in range(n)]))
            print(f"{set_name}: {n} episodes in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
