"""How many test episodes of a stage can be solved by SOME hand-coded via-point detour (SPEC M10)?  For every test episode try
via points c + u (r + clearance), u in {up, down, +-x, +-y}, clearance in {0.045, 0.07, 0.09} m, in a dry run of the env;
report the share solved without contact (a lower bound of the solvable share, and the ceiling for a scripted detour that
may look at the simulation).      python scripts/reach_oracle_detour.py r2 [r3]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from mcrl.envs.reach_env import MyCobotReachEnv, ReachConfig  # noqa: E402
from mcrl.reach_baselines import _toward  # noqa: E402
from mcrl.reachsets import load_set  # noqa: E402
from train_reach import stage_cfg  # noqa: E402

DIRS = {"up": (0, 0, 1), "down": (0, 0, -1), "+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}


class Via:
    def __init__(s, env, u, clear, tol=0.008):
        c, r = env.ob_center[0], env.ob_radius[0]
        s.wp = [c + np.array(u) * (r + clear), env.goal[:3].copy()]
        s.i, s.tol = 0, tol

    def __call__(s, env, obs=None):
        while s.i < len(s.wp) - 1 and np.linalg.norm(env.tcp() - s.wp[s.i]) < s.tol:
            s.i += 1
        return _toward(env, s.wp[s.i])


for st in sys.argv[1:]:
    env = MyCobotReachEnv(ReachConfig(**stage_cfg(st)))
    eps = load_set(st, "test")
    solved, which = 0, []
    for i, ep in enumerate(eps):
        won = None
        for name, u in DIRS.items():
            for clear in (0.045, 0.07, 0.09):
                env.reset(seed=i, options={"episode": ep})
                ctrl, hit = Via(env, u, clear), False
                while True:
                    _, _, term, trunc, info = env.step(ctrl(env))
                    hit |= bool(info["collision"])
                    if term or trunc:
                        break
                if info["is_success"] and not hit:
                    won = (name, clear)
                    break
            if won:
                break
        solved += won is not None
        which.append(won)
    res = dict(stage=st, n=len(eps), solved=solved / len(eps), which=[w[0] if w else None for w in which])
    (ROOT / "results" / "eval" / "reach" / f"oracle_detour_{st}.json").write_text(json.dumps(res))
    print(st, "solvable by some via-point detour:", res["solved"], flush=True)
