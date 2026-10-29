"""Train one run (SPEC §11, §14). Resumes automatically from run_dir/checkpoint if present.

    python scripts/train.py --exp configs/exp/e1.yaml --run sac --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from ur5rl.agents.callbacks import (CheckpointWithBuffer, GoalCurriculum, ResourceLogger, ToleranceCurriculum,
                                    ValidationEval, peak_rss_mb)
from ur5rl.agents.factory import ALGOS, make_agent
from ur5rl.agents.residual import AlphaRamp
from ur5rl.envs.residual import ResidualEnv
from ur5rl.eval import testsets
from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def run_name(run):
    return run.get("name", run["agent"])


def resolve(exp_path, name):
    """Experiment run `name` (defaults to the agent name) -> resolved config. Run-level `overrides` patch the
    agent config, `env_overrides` (experiment-level, then run-level) patch the stage env config."""
    exp = load_yaml(exp_path)
    run = next(r for r in exp["runs"] if run_name(r) == name)
    package = exp.get("package", "ur5rl")
    env_dir = ROOT / "configs" / "env" if package == "ur5rl" else ROOT / "2-dof" / "configs" / "env"
    env_cfg = load_yaml(env_dir / f"{exp['env']}.yaml")
    env_cfg.update(exp.get("env_overrides", {}))
    env_cfg.update(run.get("env_overrides", {}))
    if not str(exp["env"]).startswith("p5"):  # the torque env (P5) has no obstacle slots
        env_cfg["n_obstacles"] = tuple(env_cfg.get("n_obstacles", (0, 0)))
    agent_cfg = load_yaml(ROOT / "configs" / "agent" / f"{run['agent']}.yaml")
    agent_cfg.update(run.get("overrides", {}))
    # Obstacle stages: the absorbing collision penalty (SPEC §6.7) uses the agent's discount. Only added there,
    # so S1/S2 configs (and their run ids) are unchanged.
    if env_cfg.get("n_obstacles", (0, 0))[1] > 0 or env_cfg.get("fixed_layout"):
        env_cfg.setdefault("gamma", agent_cfg["gamma_her"] if run.get("her", False) else agent_cfg["gamma"])
    cfg = dict(exp=exp["name"], run=name, stage=exp["env"], steps=int(run.get("steps", exp["steps"])),
               n_envs=int(run.get("n_envs", exp.get("n_envs", 4))), her=run.get("her", False), env=env_cfg,
               agent=agent_cfg)
    if package != "ur5rl":  # only added when set, so UR5e configs (and run-id hashes) stay unchanged
        cfg["package"] = package
    return cfg


def env_classes(cfg):
    """(EnvConfig, Env, residual wrapper) for the experiment's package: the UR5e (default) or the planar 2-DOF arm."""
    if cfg.get("package", "ur5rl") == "arm2dof":
        from arm2dof.envs.residual import ResidualArmEnv
        if str(cfg.get("stage", "")).startswith("p5"):  # torque control + gravity (MuJoCo)
            from arm2dof.envs.residual_torque import ResidualTorqueEnv
            from arm2dof.envs.torque_env import TorqueArmEnv, TorqueConfig
            return TorqueConfig, TorqueArmEnv, ResidualTorqueEnv
        from arm2dof.envs.kinematic_env import EnvConfig as ArmConfig, KinematicArmEnv
        return ArmConfig, KinematicArmEnv, ResidualArmEnv
    return EnvConfig, UR5ePoseEnv, ResidualEnv


def run_id(cfg, seed):
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{cfg['run']}_{h}_s{seed}"


def no_power_throttling():
    """Windows: opt this process out of power throttling (EcoQoS). A console-less background job was otherwise moved
    to efficiency mode and ran ~3x slower (E5 residual s0: 36 vs 110-160 steps/s, 2026-10-28). Speed only."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    state = State(1, 0x1 | 0x4, 0)  # execution speed and timer resolution: controlled, never throttled
    if not k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)):
        print(f"power throttling not disabled (error {ctypes.get_last_error()})", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--run", required=True, help="run name in the experiment yaml (defaults to the agent name)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default=str(ROOT / "results" / "runs"))
    p.add_argument("--steps", type=int, default=None, help="override total steps (smoke tests)")
    p.add_argument("--ckpt-every", type=int, default=50_000, help="checkpoint (model + buffer) interval, env steps")
    p.add_argument("--keep-full-at", type=int, nargs="*", default=[100_000, 300_000],
                   help="steps at which a model + buffer copy is kept (none: --keep-full-at; saves Drive space)")
    args = p.parse_args()

    no_power_throttling()
    torch.set_num_threads(args.threads)
    cfg = resolve(args.exp, args.run)
    if args.steps:
        cfg["steps"] = args.steps
    # A buffer larger than the run never fills, so capping it changes nothing but RAM and checkpoint size
    # (SB3 buffers are fully resident on Windows and checkpoints pickle the whole buffer).
    if "buffer_size" in cfg["agent"]:  # off-policy only
        cfg["agent"]["buffer_size"] = min(int(cfg["agent"]["buffer_size"]), cfg["steps"])
    rid = run_id(cfg, args.seed)
    run_dir = Path(args.out) / cfg["exp"] / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "done.json").exists():
        print(f"{rid}: already done", file=sys.stderr)
        return

    meta = dict(run_id=rid, seed=args.seed, config=cfg, threads=args.threads, git_commit=None,
                python=platform.python_version(), machine=platform.processor(), platform=platform.platform(),
                packages={k: version(k) for k in ("mujoco", "gymnasium", "stable_baselines3", "torch", "numpy")})
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2, default=str))

    Config, Env, Residual = env_classes(cfg)
    env_config = Config(**cfg["env"])
    residual = cfg["agent"]["algo"] == "residual_sac"

    def make_env(config):
        env = Env(config)
        if not residual:
            return env
        return Residual(env, **cfg["agent"]["base"], w_res=cfg["agent"].get("w_res", 0.0))  # §7.3 degraded-B2 base

    venv = make_vec_env(lambda: make_env(env_config), n_envs=cfg["n_envs"], seed=args.seed)

    ckpt = run_dir / "checkpoint"
    if (ckpt / "model.zip").exists():
        model = ALGOS[cfg["agent"]["algo"]].load(ckpt / "model.zip", env=venv, device="cpu")
        if (ckpt / "replay_buffer.pkl").exists():
            model.load_replay_buffer(ckpt / "replay_buffer.pkl")
        print(f"{rid}: resumed at {model.num_timesteps} steps", file=sys.stderr, flush=True)
    else:
        model = make_agent(cfg["agent"], venv, her=cfg["her"], seed=args.seed)
    # SB3's CSV logger truncates progress.csv, so a resumed run logs to its own folder (val.jsonl appends).
    log_dir = run_dir / ("logs" if model.num_timesteps == 0 else f"logs_from_{model.num_timesteps}")
    model.set_logger(configure(str(log_dir), ["csv", "tensorboard"]))

    done_steps = model.num_timesteps
    ckpt_cb = CheckpointWithBuffer(run_dir, every=args.ckpt_every, keep_full_at=args.keep_full_at)
    # Validation always uses the full goal distribution and the final tolerance (curricula are training-only).
    val_config = dataclasses.replace(env_config, curriculum=False) if hasattr(env_config, "curriculum") \
        else env_config
    # UR5e stages validate on the frozen validation set when it exists (S4 excludes seeds with no path, §9.2)
    val_name = f"{cfg['stage']}_val"
    val_eps = testsets.load(val_name)[0] if cfg.get("package", "ur5rl") == "ur5rl" and testsets.exists(val_name) \
        else None
    val_cb = ValidationEval(lambda: make_env(val_config), run_dir, every=25_000, n_episodes=100, episodes=val_eps)
    ckpt_cb.last = val_cb.last = done_steps
    callbacks: list[BaseCallback] = [ResourceLogger(10_000), ckpt_cb, val_cb]
    if getattr(env_config, "curriculum", False):
        callbacks.append(GoalCurriculum(run_dir, start=env_config.curriculum_start, step=env_config.curriculum_step,
                                        cap=env_config.goal_max_joint_dist, threshold=env_config.curriculum_threshold,
                                        window=env_config.curriculum_window))
    if residual:
        callbacks.append(AlphaRamp(cfg["agent"]["alpha_max"], cfg["agent"]["alpha_ramp_steps"]))
    if getattr(env_config, "tol_curriculum", False):
        callbacks.append(ToleranceCurriculum(run_dir, start=env_config.tol_start, step=env_config.tol_step,
                                             final=env_config.eps, threshold=env_config.curriculum_threshold,
                                             window=env_config.curriculum_window))
    t0 = time.perf_counter()
    model.learn(total_timesteps=cfg["steps"] - done_steps, reset_num_timesteps=False,
                callback=CallbackList(callbacks))
    ckpt_cb.save()

    final = val_cb.history[-1] if val_cb.history else {}
    (run_dir / "done.json").write_text(json.dumps(dict(
        num_timesteps=model.num_timesteps, wall_s=time.perf_counter() - t0, peak_rss_mb=peak_rss_mb(),
        final_val=final), indent=2))
    venv.close()


if __name__ == "__main__":
    main()
