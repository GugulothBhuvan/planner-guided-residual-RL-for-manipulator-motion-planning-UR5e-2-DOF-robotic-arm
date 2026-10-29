"""Planar-arm viewer (companion SPEC §10): workspace + C-space, live policy, interactive editing.

    python 2-dof/scripts/play.py --run results/runs/pe1_v2/sac_her_<hash>_s0      # trained policy
    python 2-dof/scripts/play.py --stage p3 --policy base                         # base tracker, no model
    python 2-dof/scripts/play.py --run ... --save out.gif --episodes 3            # offline (Colab): GIF / MP4
    python 2-dof/scripts/play.py --stage p3 --headless --steps 200                # no window (tests)

Mouse (workspace panel): left-click moves the goal; right-click adds an obstacle, or removes the one under the
cursor (obstacle stages only; the scene keeps at most K circles).
Keys: R reset (same scene), N new scene, P pause, B base-only / residual (residual runs), A show A* path,
V value-function heatmap over C-space for the current goal (SAC/TD3/DDPG critic, PPO value head).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def env_config(stage=None, run_cfg=None):
    from arm2dof.envs.kinematic_env import EnvConfig
    cfg = dict(run_cfg["env"]) if run_cfg else yaml.safe_load(
        (ROOT / "2-dof" / "configs" / "env" / f"{stage}.yaml").read_text(encoding="utf-8"))
    cfg["n_obstacles"] = tuple(cfg.get("n_obstacles", (0, 0)))
    if "fixed_layout" in cfg:
        cfg["fixed_layout"] = tuple(map(tuple, cfg["fixed_layout"]))
    return EnvConfig(**(cfg | dict(curriculum=False)))


def build(args):
    from arm2dof.envs.kinematic_env import KinematicArmEnv
    from arm2dof.envs.residual import ResidualArmEnv
    model, residual, base_kw = None, False, {}
    if args.run:
        from stable_baselines3.common.vec_env import DummyVecEnv
        from ur5rl.agents.factory import ALGOS
        run_cfg = json.loads((Path(args.run) / "config.json").read_text())["config"]
        cfg = env_config(run_cfg=run_cfg)
        residual = run_cfg["agent"]["algo"] == "residual_sac"
        if residual:
            base_kw = dict(run_cfg["agent"]["base"])
            base_kw["max_checks"] = max(np.atleast_1d(base_kw["max_checks"]))  # viewer: most generous training budget
        make = (lambda: ResidualArmEnv(KinematicArmEnv(cfg), alpha=run_cfg["agent"].get("alpha_max", 0.3), **base_kw)) \
            if residual else (lambda: KinematicArmEnv(cfg))
        path = Path(args.run) / ("best_model.zip" if (Path(args.run) / "best_model.zip").exists() else "last_model.zip")
        model = ALGOS[run_cfg["agent"]["algo"]].load(path, env=DummyVecEnv([make]), device="cpu")
        env = make()
    else:
        cfg = env_config(stage=args.stage)
        residual = args.policy == "base"
        env = ResidualArmEnv(KinematicArmEnv(cfg), alpha=0.0) if residual else KinematicArmEnv(cfg)
    return env, model, residual


def value_map(model, env, n1=90, n2=81):
    """Value of every configuration (q-dot = 0, previous action 0) for the current goal and scene."""
    import torch as th
    base = env.unwrapped
    g1 = np.linspace(-np.pi, np.pi, n1)
    g2 = np.linspace(-0.9 * np.pi, 0.9 * np.pi, n2)
    saved = base.q.copy(), base.q_prev.copy(), base.prev_action.copy()
    obs = []
    for a in g1:
        for b in g2:
            base.q = base.q_prev = np.array([a, b])
            base.prev_action = np.zeros(2)
            o = base._obs()
            obs.append(env._augment(o) if hasattr(env, "_augment") else o)
    base.q, base.q_prev, base.prev_action = saved
    batch = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
    with th.no_grad():
        t, _ = model.policy.obs_to_tensor(batch)
        if hasattr(model, "critic"):
            act = model.policy._predict(t, deterministic=True)
            v = th.min(th.cat(model.critic(t, act), dim=1), dim=1).values
        else:
            v = model.policy.predict_values(t).squeeze(-1)
    return g1, g2, v.numpy().reshape(n1, n2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run")
    p.add_argument("--stage", default="p1")
    p.add_argument("--policy", choices=["random", "base", "zero"], default="zero", help="without --run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", help="write an animation (.gif or .mp4) instead of opening a window")
    p.add_argument("--episodes", type=int, default=2, help="with --save")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--steps", type=int, default=200, help="with --headless")
    args = p.parse_args()
    if args.save or args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from arm2dof.planners.ik import ik_nearest
    from arm2dof.viz.two_panel import TwoPanel

    env, model, residual = build(args)
    base = env.unwrapped
    view = TwoPanel(env)
    state = dict(obs=None, done=False, paused=False, base_only=False, episode=0, ret=0.0, seed=args.seed)

    def reset(new_scene=True):
        if new_scene:
            state["seed"] += 1
        state["obs"], _ = env.reset(seed=state["seed"])
        state["done"], state["ret"] = False, 0.0
        view.clear_trail()
        view.value = None
        view.record()

    def act():
        if model is not None and not state["base_only"]:
            a, _ = model.predict(state["obs"], deterministic=True)
            return a
        if args.policy == "random" and model is None:
            return env.action_space.sample()
        return np.zeros(env.action_space.shape)

    def step():
        if state["paused"] or state["done"]:
            return
        state["obs"], r, term, trunc, info = env.step(act())
        state["ret"] += r
        state["done"] = term or trunc
        view.record()
        mode = "base only" if (residual and (state["base_only"] or model is None)) else ("residual" if residual else "policy")
        view.status = (f"{mode}   t={base.t}   d={100 * info['d_pos']:.1f} cm   "
                       f"{'SUCCESS' if info['is_success'] else ''}{'  COLLISION' if info['collision'] else ''}"
                       f"{'   (paused)' if state['paused'] else ''}")

    def redraw():
        view.draw(reference=env.reference if residual else None)

    reset(new_scene=False)

    if args.headless:
        for _ in range(args.steps):
            step()
            if state["done"]:
                reset()
        redraw()
        print(f"headless ok: t={base.t} status='{view.status}'")
        return

    if args.save:
        frames = []

        def frame(_):
            if state["done"]:
                state["episode"] += 1
                reset()
            step()
            redraw()

        total = args.episodes * base.cfg.episode_length
        anim = FuncAnimation(view.fig, frame, frames=total, interval=40)
        writer = PillowWriter(fps=25) if args.save.endswith(".gif") else "ffmpeg"
        anim.save(args.save, writer=writer, dpi=90)
        print(f"wrote {args.save} ({total} frames)")
        return

    def on_click(ev):
        if ev.inaxes is not view.ws or ev.xdata is None:
            return
        pt = np.array([ev.xdata, ev.ydata])
        if ev.button == 1:  # move the goal (reachable annulus only)
            q = ik_nearest(pt, base.q)
            if q is None:
                return
            base.goal, base.q_goal = pt, q
        elif ev.button == 3 and base.use_obstacles:  # add / remove an obstacle
            inside = [k for k in np.flatnonzero(base.mask) if np.linalg.norm(base.centers[k] - pt) < base.radii[k]]
            if inside:
                base.mask[inside[0]] = False
            elif (~base.mask).any() and np.linalg.norm(pt) > 0.15:
                k = int(np.flatnonzero(~base.mask)[0])
                base.centers[k], base.radii[k], base.mask[k] = pt, 0.1, True
        if residual:
            state["obs"] = env.replan()
        else:
            state["obs"] = base._obs()
        state["done"] = False
        base.t = 0
        view.value = None

    def on_key(ev):
        k = (ev.key or "").lower()
        if k == "r":
            reset(new_scene=False)
        elif k == "n":
            reset()
        elif k == "p":
            state["paused"] = not state["paused"]
        elif k == "b" and residual and model is not None:
            state["base_only"] = not state["base_only"]
        elif k == "a":
            view.show_astar = not view.show_astar
        elif k == "v" and model is not None:
            view.value = None if view.value is not None else value_map(model, env)

    view.fig.canvas.mpl_connect("button_press_event", on_click)
    view.fig.canvas.mpl_connect("key_press_event", on_key)

    def loop(_):
        step()
        redraw()

    anim = FuncAnimation(view.fig, loop, interval=40, cache_frame_data=False)  # 25 Hz = the control rate
    plt.show()
    del anim


if __name__ == "__main__":
    main()
