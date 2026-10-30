"""Thesis GIFs for the planar arm (companion SPEC §10), from frozen test episodes, replayed exactly as evaluated.

  PE4  two-panel view (workspace + C-space) on p3_test: each episode first with the planner alone (base@2300), then
       with the additive-residual hybrid (residual s0, primary checkpoint, same budget); base reference dashed.
  PE8  two workspace panels side by side, the same p5a_test episode on the exact and the x1.2 mass plant: the
       residual on B5' (s0, primary) drawn as the arm, B5' (PD + gravity) alone as a gray ghost.

    python 2-dof/scripts/make_gifs.py   ->  results/videos/pe4_p3_test.gif, pe8_exact_vs_mass1.2.gif
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "videos"
STRIDE, FPS, DPI = 2, 12.5, 72  # every 2nd 25 Hz control step at 12.5 fps: real time
PE4_EPISODES = (9, 0)  # 9: planner succeeds, hybrid stops 1.26 cm short; 0: both collide on the fallback line
PE8_EPISODE = 7  # residual succeeds on both plants; B5' alone misses by 3.35 cm on the x1.2 plant
PE4_RUN = ROOT / "results" / "runs" / "pe4" / "residual_dd445192_s0"
PE8_RUN = ROOT / "results" / "runs" / "pe8" / "residual_787d568a_s0"


def module(name, file):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
        assert spec is not None and spec.loader is not None
        sys.modules[name] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


def grab(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def outcome(info):
    if info["is_success"]:
        return "SUCCESS"
    if info["collision"]:
        return "COLLISION"
    return f"missed the goal by {100 * info['d_pos']:.2f} cm"


def pe4():
    from arm2dof.envs.kinematic_env import KinematicArmEnv
    from arm2dof.envs.residual import ResidualArmEnv
    from arm2dof.viz.two_panel import TwoPanel
    from ur5rl.eval import testsets
    ev = module("evaluate_2dof", "evaluate.py")
    episodes, meta = testsets.load("p3_test", root=ev.SETS)
    env_cfg = ev.stage_env(meta)
    model, cfg = ev.load_policy(PE4_RUN, "primary")
    hybrid_kw = dict(cfg["agent"]["base"], max_checks=2300)
    variants = [("planner alone (2300 checks)", None, ResidualArmEnv(KinematicArmEnv(env_cfg), alpha=0.0,
                                                                      inflate=0.05, shortcut_iters=20, max_checks=2300)),
                ("additive-residual hybrid (s0, primary)", model,
                 ResidualArmEnv(KinematicArmEnv(env_cfg), alpha=cfg["agent"]["alpha_max"], **hybrid_kw))]
    frames = []
    for i in PE4_EPISODES:
        for label, pol, env in variants:
            obs, _ = env.reset(seed=i, options={"episode": episodes[i]})  # the evaluation's seed and episode
            view = TwoPanel(env, figsize=(10, 5))
            view.record()
            done, k, info = False, 0, None
            while not done:
                obs, _, term, trunc, info = env.step(np.zeros(2) if pol is None else pol.predict(obs, deterministic=True)[0])
                done = term or trunc
                view.record()
                k += 1
                if k % STRIDE == 0 or done:
                    view.status = f"PE4, p3_test episode {i}: {label}   t = {k * 0.04:.2f} s   d = {100 * info['d_pos']:.1f} cm"
                    if done:
                        view.status += f"   {outcome(info)}"
                    view.draw(reference=env.reference)
                    frames.append(grab(view.fig))
            frames += [frames[-1]] * int(1.5 * FPS)  # hold the outcome
            plt.close(view.fig)
    path = OUT / "pe4_p3_test.gif"
    imageio.mimsave(path, frames, duration=1000 / FPS, loop=0)
    print(f"wrote {path} ({len(frames)} frames)")


def pe8():
    from arm2dof.envs.residual_torque import ResidualTorqueEnv
    from arm2dof.envs.torque_env import TorqueArmEnv
    from arm2dof.planners.ik import joints
    from arm2dof.viz.two_panel import ARM, GOAL, INK, INK_2, OBST
    e8 = module("eval_pe8", "eval_pe8.py")
    ep = e8.episodes()[PE8_EPISODE]
    model, cfg = e8.load(PE8_RUN, "primary")
    plants = (1.0, 1.2)
    res = [ResidualTorqueEnv(TorqueArmEnv(e8.stage_cfg(m)), alpha=cfg["agent"]["alpha_max"], **cfg["agent"]["base"])
           for m in plants]
    pdg = [ResidualTorqueEnv(TorqueArmEnv(e8.stage_cfg(m)), alpha=0.0, kind="pdg") for m in plants]
    obs = [e.reset(options={"episode": ep})[0] for e in res]
    for e in pdg:
        e.reset(options={"episode": ep})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    fig.patch.set_facecolor("white")
    trails = [[[], []] for _ in plants]  # [residual EE, B5' EE] per plant
    done = [[False, False] for _ in plants]
    info = [[None, None] for _ in plants]
    frames, k = [], 0
    while not all(all(d) for d in done):
        for j in range(len(plants)):
            for v, env in enumerate((res[j], pdg[j])):
                if done[j][v]:
                    continue
                a = model.predict(obs[j], deterministic=True)[0] if v == 0 else np.zeros(2)
                o, _, term, trunc, inf = env.step(a)
                if v == 0:
                    obs[j] = o
                info[j][v], done[j][v] = inf, term or trunc
                trails[j][v].append(joints(env.unwrapped.q)[1])
        k += 1
        if k % STRIDE and not all(all(d) for d in done):
            continue
        for j, ax in enumerate(axes):
            ax.clear()
            ax.set_aspect("equal")
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.tick_params(colors=INK_2, labelsize=8)
            ax.set_title(f"{'exact mass' if plants[j] == 1.0 else f'mass x{plants[j]}'} (controllers keep the nominal model)",
                         loc="left", fontsize=10, color=INK)
            base = pdg[j].unwrapped
            ghost = joints(base.q)
            ax.plot([0, ghost[0][0], ghost[1][0]], [0, ghost[0][1], ghost[1][1]], color=OBST, lw=9,
                    solid_capstyle="round", alpha=0.8)
            t = np.array(trails[j][1])
            ax.plot(t[:, 0], t[:, 1], color=OBST, lw=1.2, ls="--")
            arm = res[j].unwrapped
            elbow, ee = joints(arm.q)
            ax.plot([0, elbow[0], ee[0]], [0, elbow[1], ee[1]], color=ARM, lw=9, solid_capstyle="round", alpha=0.85)
            t = np.array(trails[j][0])
            ax.plot(t[:, 0], t[:, 1], color=ARM, lw=1.2, alpha=0.6)
            ax.plot(*arm.goal, marker="*", ms=14, color=GOAL, mec="white", mew=1.0, ls="none")
            ax.add_patch(Circle(arm.goal, 0.01, fill=False, color=GOAL, lw=0.8))
            lines = []
            for v, name in ((0, "residual on B5'"), (1, "B5' alone")):
                inf = info[j][v]
                state = ("SUCCESS" if inf["is_success"] else f"d = {100 * inf['d_pos']:.2f} cm") if done[j][v] \
                    else f"d = {100 * inf['d_pos']:.2f} cm"
                lines.append(f"{name}: {state}")
            ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=9, color=INK, va="bottom")
        handles = [plt.Line2D([], [], color=ARM, lw=4, label="residual on B5' (s0, primary)"),
                   plt.Line2D([], [], color=OBST, lw=4, label="B5' alone (PD + gravity)"),
                   plt.Line2D([], [], color=GOAL, marker="*", ls="none", label="goal (1 cm)")]
        axes[0].legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
        fig.suptitle(f"PE8, p5a_test episode {PE8_EPISODE}: torque control on the exact and the x1.2 mass plant   "
                     f"t = {k * 0.04:.2f} s", x=0.01, ha="left", fontsize=10, color=INK_2)
        frames.append(grab(fig))
    frames += [frames[-1]] * int(2 * FPS)
    plt.close(fig)
    path = OUT / "pe8_exact_vs_mass1.2.gif"
    imageio.mimsave(path, frames, duration=1000 / FPS, loop=0)
    print(f"wrote {path} ({len(frames)} frames)")


if __name__ == "__main__":
    pe4()
    pe8()
