"""Reachability within the episode horizon under the §6.2 action rule.

(1) Realised joint speed with a = +1 on one joint (target: ~ dq_max / control_dt = 1 rad/s).
(2) Oracle rollouts: a = clip((q_goal - q_target) / dq_max, -1, 1), i.e. full speed straight to the sampled
    goal configuration. Reports the fraction of episodes the oracle finishes within the horizon (target 100%).
    S4 is checked for timing only (obstacles off, 200 steps, no goal filter).

    python scripts/check_reachability.py --episodes 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

from ur5rl.envs.pose_env import HOME_Q, EnvConfig, UR5ePoseEnv

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "env"


def stage_config(name):
    cfg = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    if name == "s4":
        cfg["n_obstacles"] = [0, 0]  # timing only
    cfg["n_obstacles"] = tuple(cfg["n_obstacles"])
    cfg["reward"] = "sparse"  # fixed horizon, no early termination
    return EnvConfig(**cfg)


def joint_speeds():
    env = UR5ePoseEnv(EnvConfig())
    env.reset(seed=0)
    m, d = env.model, env.data
    speeds = []
    direction = [1, -1, -1, 1, 1, 1]  # +1 on shoulder lift / elbow drives the forearm into the table
    for j in range(6):
        mujoco.mj_resetData(m, d)
        d.qpos[:6] = d.ctrl[:6] = env.q_target = HOME_Q.copy()
        mujoco.mj_forward(m, d)
        a = np.zeros(6)
        a[j] = direction[j]
        for _ in range(10):  # 0.4 s ramp-up
            env.step(a)
        q0 = d.qpos[j]
        for _ in range(25):  # 1 s steady state
            env.step(a)
        speeds.append(abs(d.qpos[j] - q0))
    return np.array(speeds)


def oracle(stage, episodes, seed=0):
    env = UR5ePoseEnv(stage_config(stage))
    final_ok, any_ok, collided, max_dist, t_success = [], [], [], [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        max_dist.append(np.abs(env.q_goal - env.data.qpos[:6]).max())
        hit, first = False, None
        for t in range(env.cfg.episode_length):
            a = np.clip((env.q_goal - env.q_target) / env.cfg.dq_max, -1, 1)
            _, _, _, _, info = env.step(a)
            hit |= info["collision"]
            if info["is_success"] and first is None:
                first = t + 1
        final_ok.append(info["is_success"])
        any_ok.append(first is not None)
        collided.append(hit)
        t_success.append(first if first is not None else np.nan)
    final_ok, collided = np.array(final_ok), np.array(collided)
    # The oracle drives a straight joint-space line; when that line touches the table or the arm itself the
    # goal is still reachable by another route, so time-reachability is judged on contact-free episodes.
    return dict(final=final_ok.mean(), free=final_ok[~collided].mean(), any=np.mean(any_ok), collided=collided.mean(),
                max_dist_p50=np.median(max_dist), max_dist_max=np.max(max_dist),
                t50=np.nanmedian(t_success) * 0.04, t_max=np.nanmax(t_success) * 0.04,
                horizon=env.cfg.episode_length * 0.04, bound=env.cfg.goal_max_joint_dist)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    args = p.parse_args()
    print("realised joint speed at |a|=1 (rad/s):", np.array2string(joint_speeds(), precision=3))
    print(f"\n{'stage':6} {'filter':>7} {'success@end':>11} {'contact-free':>12} {'contact':>8} "
          f"{'max|dq| p50/max':>16} {'t_succ p50/max':>15} {'horizon':>8}")
    for stage in ("s1", "s2", "s4"):
        r = oracle(stage, args.episodes)
        flt = f"{r['bound']:.1f}" if r["bound"] else "-"
        print(f"{stage.upper():6} {flt:>7} {r['final'] * 100:10.1f}% {r['free'] * 100:11.1f}% "
              f"{r['collided'] * 100:7.1f}% {r['max_dist_p50']:7.2f}/{r['max_dist_max']:.2f} rad "
              f"{r['t50']:6.2f}/{r['t_max']:.2f} s {r['horizon']:6.1f} s")


if __name__ == "__main__":
    main()
