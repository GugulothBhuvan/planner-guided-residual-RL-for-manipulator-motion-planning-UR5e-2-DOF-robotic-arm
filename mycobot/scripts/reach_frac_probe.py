"""Probe (SPEC M10): share of R2-like episodes solvable by some via-point detour, for a sphere at fraction f of the start-goal line."""
import sys
ARGS = list(sys.argv)
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from mcrl.envs.reach_env import MyCobotReachEnv, ReachConfig  # noqa: E402
from reach_oracle_detour import DIRS, Via  # noqa: E402
from train_reach import stage_cfg  # noqa: E402
f = float(ARGS[1]); n = int(ARGS[2])
cfg = stage_cfg("r2"); cfg["obstacle_frac"] = f
cfg["min_goal_dist"] = float(ARGS[3]) if len(ARGS) > 3 else cfg["min_goal_dist"]
if len(ARGS) > 4:
    cfg['obstacle_radius'] = (0.015, float(ARGS[4]))
if len(ARGS) > 7:
    cfg['cube_center'] = tuple(float(x) for x in ARGS[5:8]); cfg['obstacle_margin'] = float(ARGS[8]) if len(ARGS) > 8 else 0.04
env = MyCobotReachEnv(ReachConfig(**cfg))
solved = straight_hit = 0
for i in range(n):
    env.reset(seed=9_000_000 + i); ep = env.episode()
    won = None
    for name, u in DIRS.items():
        for clear in (0.045, 0.07, 0.09):
            env.reset(seed=0, options={"episode": ep}); ctrl, hit = Via(env, u, clear), False
            while True:
                _, _, term, trunc, info = env.step(ctrl(env)); hit |= bool(info["collision"])
                if term or trunc: break
            if info["is_success"] and not hit: won = (name, clear); break
        if won: break
    solved += won is not None
print("frac", f, "rmax", cfg["obstacle_radius"][1], "min_goal_dist", cfg["min_goal_dist"], "cube", cfg["cube_center"], "solvable", solved / n, flush=True)
