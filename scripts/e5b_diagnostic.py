"""E5b offset diagnostic (evaluation only; SPEC §7.3): the 100 S4 validation episodes (stored s4_val, reset seeds
1_000_000 + i, alpha 0.3, deterministic), per episode the fade value and |o| at the last step, mean |o| over the
episode, |o| at the first collision step, success, collision and final distance. The v1 run of this diagnostic decided
the v1 -> v2 change (results/runs/archive/e5b_v1_dist_fade/diagnostic_v1.json).

    python scripts/e5b_diagnostic.py --run results/runs/e5b/residual_offset_<hash>_s0 --checkpoints 300000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from ur5rl.agents.factory import ALGOS
from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv
from ur5rl.envs.residual import ResidualEnv
from ur5rl.eval import testsets

LARGE = 0.15  # rad: half the clip


def replay(run, steps):
    cfg = json.loads((run / "config.json").read_text())["config"]
    env_cfg = dict(cfg["env"], n_obstacles=tuple(cfg["env"]["n_obstacles"]), curriculum=False)
    make = lambda: ResidualEnv(UR5ePoseEnv(EnvConfig(**env_cfg)), **cfg["agent"]["base"])
    model = ALGOS["residual_sac"].load(run / "checkpoints" / f"model_{steps}.zip", env=DummyVecEnv([make]), device="cpu")
    env = make()
    env.set_alpha(cfg["agent"]["alpha_max"])
    episodes, _ = testsets.load("s4_val")
    rows = []
    for i, ep in enumerate(episodes):
        obs, _ = env.reset(seed=1_000_000 + i, options={"episode": ep})
        hit, mags, hit_mag = False, [], None
        while True:
            obs, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            mags.append(float(np.abs(env.offset).max()))
            if info["collision"] and not hit:
                hit, hit_mag = True, mags[-1]
            if term or trunc:
                break
        rows.append(dict(success=bool(info["is_success"]), collision=hit, fade_end=env.offset_scale(), o_end=mags[-1],
                         o_mean=float(np.mean(mags)), o_at_hit=hit_mag, d_pos_cm=100 * info["d_pos"]))
    return rows


def summary(rows):
    fail = [r for r in rows if not r["success"]]
    hits = [r["o_at_hit"] for r in rows if r["collision"]]
    med = lambda xs: float(np.median(xs)) if xs else float("nan")
    share = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return dict(
        success=share([r["success"] for r in rows]), collision=share([r["collision"] for r in rows]), failures=len(fail),
        fail_fade1=share([r["fade_end"] >= 0.999 for r in fail]), fail_large_o=share([r["o_end"] >= LARGE for r in fail]),
        fail_both=share([r["fade_end"] >= 0.999 and r["o_end"] >= LARGE for r in fail]),
        o_end_median_fail=med([r["o_end"] for r in fail]), o_end_median_succ=med([r["o_end"] for r in rows if r["success"]]),
        o_mean_median_collided=med([r["o_mean"] for r in rows if r["collision"]]),
        o_mean_median_clean=med([r["o_mean"] for r in rows if not r["collision"]]),
        o_at_hit_median=med(hits), o_at_hit_large=share([h >= LARGE for h in hits]),
        d_pos_cm_median_fail=med([r["d_pos_cm"] for r in fail]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[300_000])
    args = p.parse_args()
    run = Path(args.run)
    out = {}
    for steps in args.checkpoints:
        rows = replay(run, steps)
        out[steps] = dict(summary=summary(rows), rows=rows)
        print(json.dumps({"checkpoint": steps, **out[steps]["summary"]}), flush=True)
    (run / f"diagnostic_{'_'.join(map(str, args.checkpoints))}.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
