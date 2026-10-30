"""Planar-arm viewer (companion SPEC §10): workspace + C-space, live policy, interactive editing.

    python 2-dof/scripts/play.py --run results/runs/pe1_v2/sac_her_<hash>_s0      # trained policy
    python 2-dof/scripts/play.py --stage p3 --policy base                         # base tracker, no model
    python 2-dof/scripts/play.py --run ... --save out.gif --episodes 3            # offline (Colab): GIF / MP4
    python 2-dof/scripts/play.py --stage p3 --headless --steps 200                # no window (tests)

Frozen test episodes, replayed exactly as evaluated (same env config, reset seed and episode; N = next episode):
    python 2-dof/scripts/play.py --run results/runs/pe4/residual_<hash>_s0 --checkpoint primary --testset p3_test --episode 9 --budget 2300
    python 2-dof/scripts/play.py --stage p3 --policy base --testset p3_test --episode 9 --budget 2300    # planner alone
    python 2-dof/scripts/play.py --run results/runs/pe8/residual_<hash>_s0 --checkpoint primary --testset p5a_test --episode 7 --mass 1.2
Torque (PE8) runs and --stage p5a use the torque arm (no obstacles; --policy base = B5', PD + gravity); --mass scales
the plant's link masses while the controllers keep the nominal model.

Mouse (workspace panel): left-click moves the goal; right-click adds an obstacle, or removes the one under the
cursor (obstacle stages only; the scene keeps at most K circles).
Keys: R reset (same scene), N new scene, P pause, B base-only / residual (residual runs), A show A* path,
V value-function heatmap over C-space for the current goal (SAC/TD3/DDPG critic, PPO value head).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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


def script(name, file):
    """A sibling script loaded by path (the UR5e scripts/ has modules of the same name, e.g. evaluate.py)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
        assert spec is not None and spec.loader is not None
        sys.modules[name] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


def test_episodes(args):
    """(episodes, kinematic env config or None) of --testset, loaded as the evaluation scripts load them."""
    if args.testset == "p5a_test":
        return script("eval_pe8", "eval_pe8.py").episodes(), None
    from ur5rl.eval import testsets
    ev = script("evaluate_2dof", "evaluate.py")
    episodes, meta = testsets.load(args.testset, root=ev.SETS)
    return episodes, ev.stage_env(meta)


def model_path(run, which):
    if which == "best":
        return run / ("best_model.zip" if (run / "best_model.zip").exists() else "last_model.zip")
    sys.path.insert(0, str(ROOT / "scripts"))
    import checkpoints  # UR5e scripts/checkpoints.py: the evaluation checkpoint rule (SPEC §9.2)
    path, _, note = checkpoints.select(run, which)
    print(f"{run.name}: {which} checkpoint {path.name} ({note})")
    return path


def build(args, test_cfg=None):
    from arm2dof.envs.kinematic_env import KinematicArmEnv
    from arm2dof.envs.residual import ResidualArmEnv
    from arm2dof.envs.residual_torque import ResidualTorqueEnv
    from arm2dof.envs.torque_env import TorqueArmEnv
    model, residual, base_kw = None, False, {}
    run_cfg = json.loads((Path(args.run) / "config.json").read_text())["config"] if args.run else None
    torque = ("n_obstacles" not in run_cfg["env"]) if run_cfg else args.stage == "p5a"
    plant = lambda: TorqueArmEnv(script("eval_pe8", "eval_pe8.py").stage_cfg(args.mass))  # PE8 evaluation config
    if run_cfg:
        from stable_baselines3.common.vec_env import DummyVecEnv
        from ur5rl.agents.factory import ALGOS
        residual = run_cfg["agent"]["algo"] == "residual_sac"
        alpha = run_cfg["agent"].get("alpha_max", 0.3)
        if torque:
            make = (lambda: ResidualTorqueEnv(plant(), alpha=alpha, **run_cfg["agent"]["base"])) if residual else plant
        else:
            cfg = test_cfg or env_config(run_cfg=run_cfg)
            if residual:
                base_kw = dict(run_cfg["agent"]["base"])
                # viewer: the given budget, else the most generous training budget
                base_kw["max_checks"] = args.budget or max(np.atleast_1d(base_kw["max_checks"]))
            make = (lambda: ResidualArmEnv(KinematicArmEnv(cfg), alpha=alpha, **base_kw)) if residual \
                else (lambda: KinematicArmEnv(cfg))
        model = ALGOS[run_cfg["agent"]["algo"]].load(model_path(Path(args.run), args.checkpoint),
                                                     env=DummyVecEnv([make]), device="cpu")
        env = make()
    elif torque:
        residual = args.policy == "base"
        env = ResidualTorqueEnv(plant(), alpha=0.0, kind="pdg") if residual else plant()
    else:
        cfg = test_cfg or env_config(stage=args.stage)
        residual = args.policy == "base"
        # the planner alone as the evaluation runs it (base@N); without --budget, the env's default budget
        kw = dict(inflate=0.05, shortcut_iters=20, max_checks=args.budget) if args.budget else {}
        env = ResidualArmEnv(KinematicArmEnv(cfg), alpha=0.0, **kw) if residual else KinematicArmEnv(cfg)
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
    p.add_argument("--checkpoint", choices=["best", "primary", "final"], default="best",
                   help="with --run: primary / final = the evaluation checkpoint rule (UR5e SPEC §9.2)")
    p.add_argument("--testset", help="replay frozen test episodes: p1_test, p2_test, p3_test, p5a_test")
    p.add_argument("--episode", type=int, default=0, help="with --testset: the first episode shown (N = next)")
    p.add_argument("--budget", type=int, help="planner budget in collision checks (residual / base, P2-P3)")
    p.add_argument("--mass", type=float, default=1.0, help="torque stage: plant mass factor (PE8: 0.8 / 1.2)")
    p.add_argument("--steps", type=int, default=200, help="with --headless")
    args = p.parse_args()
    if args.save or args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from arm2dof.planners.ik import ik_nearest
    from arm2dof.viz.two_panel import TwoPanel

    episodes, test_cfg = test_episodes(args) if args.testset else (None, None)
    env, model, residual = build(args, test_cfg)
    base = env.unwrapped
    torque = not hasattr(base, "centers")
    view = TwoPanel(env)
    state = dict(obs=None, done=False, paused=False, base_only=False, episode=0, ret=0.0, seed=args.seed,
                 test_i=args.episode)

    def reset(new_scene=True):
        if episodes is not None:  # test episode i: the evaluation's reset (seed i on P1-P3) and stored episode
            if new_scene:
                state["test_i"] = (state["test_i"] + 1) % len(episodes)
            i = state["test_i"]
            state["obs"], _ = env.reset(options={"episode": episodes[i]}) if torque \
                else env.reset(seed=i, options={"episode": episodes[i]})
        else:
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
        if episodes is not None:
            mode = f"{args.testset} #{state['test_i']}  {mode}"
        if torque:
            mode += f"  mass x{base.cfg.mass_factor:g}"
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
                if episodes is not None:  # a test episode: stop at its end, so the status is its outcome
                    break
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
        if residual and hasattr(env, "replan"):  # kinematic residual: replan the base for the new scene
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
