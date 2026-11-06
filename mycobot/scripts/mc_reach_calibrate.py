"""Reach-cube calibration (SPEC M10): for candidate cube centres, run the scripted straight-line controller (R1: fixed home
start, no obstacle, `ee` mode) on sampled goals; report success, contact and the steps needed.

    python scripts/mc_reach_calibrate.py scan --n 60            # coarse grid of centres
    python scripts/mc_reach_calibrate.py check --center 0.12 -0.05 0.28 --n 500
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcrl.envs.reach_env import MyCobotReachEnv, ReachConfig  # noqa: E402
from mcrl.reach_baselines import rollout, straight  # noqa: E402


def run(center, n, horizon=300, half=0.05):
    env = MyCobotReachEnv(ReachConfig(cube_center=tuple(center), cube_half=half, episode_length=horizon))
    rows = rollout(env, lambda e: straight, n=n, seed0=10_000)
    return dict(center=list(center), n=n, success=float(np.mean([r["success"] for r in rows])),
                contact=float(np.mean([r["collision"] for r in rows])),
                steps_med=float(np.median([r["steps"] for r in rows if r["success"]] or [np.nan])),
                steps_max=int(max([r["steps"] for r in rows if r["success"]] or [0])), d_cm_max=float(max(r["d_cm"] for r in rows)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["scan", "check"])
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--center", type=float, nargs=3)
    a = p.parse_args()
    if a.mode == "check":
        print(json.dumps(run(a.center, a.n)))
    else:
        out = []
        for x, y, z in itertools.product((0.08, 0.12, 0.16, 0.20), (-0.10, -0.05, 0.0), (0.20, 0.25, 0.30, 0.35)):
            r = run((x, y, z), a.n)
            out.append(r)
            print(json.dumps(r), flush=True)
        (ROOT / "results" / "reach_scan.json").write_text(json.dumps(out, indent=1))
