"""Day-1 step-time profile (SPEC §6.5): physics vs clearance vs the rest, plus reset cost and an
actuator-stability check for the frame-skip fallback (§11). Diagnostics only; the hard gate is
scripts/bench_throughput.py.

    python scripts/profile_step.py
"""

from __future__ import annotations

import platform
import time

import mujoco
import numpy as np

from ur5rl.envs.pose_env import HOME_Q, EnvConfig, UR5ePoseEnv

SETTINGS = {"fs20_dt0.002": dict(frame_skip=20, timestep=0.002), "fs10_dt0.004": dict(frame_skip=10, timestep=0.004)}
STAGES = {"S1": dict(orientation=False, n_obstacles=(0, 0)), "S4": dict(n_obstacles=(1, 4), episode_length=200)}


def timeit(fn, n):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3  # ms


def profile(stage, setting, n_steps=5000):
    env = UR5ePoseEnv(EnvConfig(**STAGES[stage], **SETTINGS[setting]))
    m, d = env.model, env.data
    rng = np.random.default_rng(0)

    env.reset(seed=0)
    env.stats["reset_samples"] = 0
    n_resets = 200
    t_reset = timeit(lambda: env.reset(), n_resets)
    samples_per_reset = env.stats["reset_samples"] / (n_resets + 1)

    env.reset(seed=1)
    actions = rng.uniform(-1, 1, (n_steps, 6))
    t0 = time.perf_counter()
    for a in actions:
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()
    t_step = (time.perf_counter() - t0) / n_steps * 1e3  # includes the occasional reset

    env.reset(seed=2)

    def physics():
        for _ in range(env.cfg.frame_skip):
            mujoco.mj_step(m, d)
            _ = d.ncon > 0

    t_phys = timeit(physics, 2000)
    t_clear = timeit(lambda: env.clearance.body_min(d, env.ob_center, env.ob_radius, env.ob_mask), 5000) \
        if env.use_obstacles else 0.0
    t_obs = timeit(env._obs, 5000)  # includes clearance
    return dict(step=t_step, physics=t_phys, clearance=t_clear, obs=t_obs, reset=t_reset,
                samples_per_reset=samples_per_reset)


def _home(env):
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    d.qpos[:6] = HOME_Q
    d.ctrl[:6] = HOME_Q
    mujoco.mj_forward(m, d)


def actuator_check(setting):
    """(1) Joint speed actually realised by env.step with a = +1 on one joint for 2 s (SPEC §6.2 semantics).
    (2) Stability: hold home, step every target by +dq_max once, report overshoot and 1 s settling error."""
    env = UR5ePoseEnv(EnvConfig(**SETTINGS[setting]))
    env.reset(seed=0)
    m, d = env.model, env.data
    control_dt = env.cfg.frame_skip * env.cfg.timestep
    speeds = []
    for j in range(6):
        _home(env)
        a = np.zeros(6)
        a[j] = 1.0
        q0 = d.qpos[j]
        n = int(round(2.0 / control_dt))
        for _ in range(n):
            env.step(a)
        speeds.append((d.qpos[j] - q0) / 2.0)

    _home(env)
    target = HOME_Q + env.cfg.dq_max
    d.ctrl[:6] = target
    hist = []
    for _ in range(int(1.0 / env.cfg.timestep)):
        mujoco.mj_step(m, d)
        hist.append(d.qpos[:6] - target)
    hist = np.array(hist)
    return dict(speeds=np.array(speeds), intended=env.cfg.dq_max / control_dt,
                overshoot_mrad=hist.max() * 1e3, settle_err_mrad=np.abs(hist[-1]).max() * 1e3,
                finite=bool(np.isfinite(hist).all()))


def main():
    print(f"machine: {platform.processor()} | {platform.platform()} | python {platform.python_version()}")
    print(f"mujoco {mujoco.__version__}\n")
    print(f"{'stage':5} {'setting':13} {'step ms':>8} {'physics':>8} {'clearance':>9} {'obs':>7} "
          f"{'reset ms':>9} {'samples/reset':>13}")
    for stage in STAGES:
        for setting in SETTINGS:
            r = profile(stage, setting)
            print(f"{stage:5} {setting:13} {r['step']:8.3f} {r['physics']:8.3f} {r['clearance']:9.4f} "
                  f"{r['obs']:7.3f} {r['reset']:9.3f} {r['samples_per_reset']:13.1f}")
    print("\nactuator check:")
    for setting in SETTINGS:
        r = actuator_check(setting)
        print(f"  {setting}: realised joint speed at a=+1: {np.array2string(r['speeds'], precision=3)} rad/s "
              f"(intended {r['intended']:.2f}); step response overshoot {r['overshoot_mrad']:.2f} mrad, "
              f"1 s settling err {r['settle_err_mrad']:.2f} mrad, finite={r['finite']}")


if __name__ == "__main__":
    main()
