"""Interactive local viewer (SPEC §10).

Free mode: drag the goal (Ctrl + right-drag on the green frame) and the policy follows it live.

    python scripts/play.py --run results/runs/e1/<run_dir>          # trained policy (best_model.zip)
    python scripts/play.py --stage s4 --policy random                # scene check without a model
    python scripts/play.py --run ... --headless --steps 500          # no window (tests / CI)

Keys (free mode): R reset (same goal), N new random goal / scene, P pause.

Replay mode (2026-10-30): one stored test episode, reproduced exactly as scripts/evaluate.py runs it (same env config,
reset seed = episode index, same base plan, deterministic policy), with the obstacles of that episode.

    python scripts/play.py --testset s4_test --episode 32 --method base --budget 3000
    python scripts/play.py --testset s4_test --episode 32 --method hybrid --budget 3000 --checkpoint primary --run <dir>
    python scripts/play.py --testset s4_test --episode 32 --method b2
    ... --record results/videos/<name>.mp4                          # offscreen MP4 instead of a window

Methods: base (degraded planner alone, residual 0), hybrid (degraded planner + residual of --run at --checkpoint),
b2 (full B2 reference). Keys (replay mode): R replay the episode, N next test episode, P pause, B toggle the
correction on/off (hybrid only; off = alpha 0 and, for the integrated offset, the offset cleared: the planner alone
from that step on, so the trajectory is no longer the evaluated one).
Overlays: goal and TCP frames, TCP trail (orange), the planner's reference path as TCP points (cyan), obstacle
safety-margin shells (translucent red); the obstacles themselves are scene geometry.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from ur5rl.envs import rotations as rot
from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv
from ur5rl.viz import overlays

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ALGOS = {"sac": SAC, "td3": TD3}


class Player:
    def __init__(self, env, policy):
        self.env, self.policy = env, policy
        self.paused = False
        self.trail = deque(maxlen=150)
        self.pending = None  # key action to apply on the main thread
        self.seed = 0

    # ---- keys (called from the viewer thread; only record the request)
    def on_key(self, keycode):
        key = chr(keycode) if 0 <= keycode < 256 else ""
        if key in "RNPB":
            self.pending = key

    def handle_pending(self):
        key, self.pending = self.pending, None
        if key == "P":
            self.paused = not self.paused
            print("paused" if self.paused else "running")
        elif key == "R":
            self.reset(new=False)
        elif key == "N":
            self.reset(new=True)
        elif key == "B":
            self.toggle()

    def toggle(self):
        print("B: correction toggle is for --method hybrid in replay mode")

    def reset(self, new):
        if new:
            self.seed += 1
        self.env.reset(seed=self.seed)
        self.trail.clear()

    # ---- control
    def goal_from_mocap(self):
        """The goal is wherever the mocap body is (it may have been dragged)."""
        env, d = self.env, self.env.data
        p = d.mocap_pos[env.goal_mocap].copy()
        R = rot.quat_to_mat(d.mocap_quat[env.goal_mocap])
        env.goal = np.r_[p, rot.mat_to_r6d(R)]
        env.goal_R = R

    def step(self):
        self.goal_from_mocap()
        obs = self.env._obs()
        action = self.policy(obs)
        self.env.step(action)
        self.trail.append(self.env.data.site_xpos[self.env.tcp_id].copy())
        return False

    def draw(self, scn, clear=True):
        base = self.env.unwrapped
        d = base.data
        if clear:  # viewer.user_scn holds only overlays; a Renderer's scene also holds the model (clear=False)
            scn.ngeom = 0
        overlays.frame(scn, d.site_xpos[base.tcp_id], d.site_xmat[base.tcp_id].reshape(3, 3), alpha=0.9)
        overlays.frame(scn, base.goal[:3], base.goal_R, length=0.09, width=0.005)
        overlays.trail(scn, list(self.trail))
        overlays.safety_shells(scn, base.ob_center, base.ob_radius, base.ob_mask, base.cfg.d_safe)


class Replay(Player):
    """One stored test episode, run exactly as scripts/evaluate.py runs it."""

    def __init__(self, args):
        from evaluate import load_policy, stage_env
        from ur5rl.envs.residual import ResidualEnv
        from ur5rl.eval import testsets
        self.episodes, meta = testsets.load(args.testset)
        env_cfg = dict(stage_env(meta), visual=True)
        self.method, self.index, self.model, self.alpha = args.method, args.episode, None, 0.0
        if args.method in ("base", "hybrid"):
            cfg = None
            if args.method == "hybrid":
                self.model, cfg = load_policy(args.run, args.checkpoint)
            base_kw = dict(cfg["agent"]["base"]) if cfg else dict(inflate=0.05, shortcut_iters=20)
            base_kw["max_checks"] = args.budget
            self.alpha = cfg["agent"]["alpha_max"] if cfg else 0.0
            env = ResidualEnv(UR5ePoseEnv(EnvConfig(**env_cfg)), alpha=self.alpha, **base_kw)
        else:
            env = UR5ePoseEnv(EnvConfig(**env_cfg))
        super().__init__(env, None)
        self.trail = deque(maxlen=2000)
        self.correction = True
        self.outcome = None

    def toggle(self):
        if self.method != "hybrid":
            return super().toggle()
        self.correction = not self.correction
        self.env.set_alpha(self.alpha if self.correction else 0.0)
        if not self.correction and getattr(self.env, "mode", "additive") == "offset":
            self.env.offset[:] = 0.0
        print(f"correction {'ON' if self.correction else 'OFF (planner alone from here)'}")

    def reset(self, new):
        from ur5rl.planners.baselines import ReferenceTracker, flange_to_tcp, plan_b2
        from ur5rl.planners.collision import CollisionChecker
        from ur5rl.planners.ik import fk_tcp
        if new:
            self.index = (self.index + 1) % len(self.episodes)
        i, ep = self.index, self.episodes[self.index]
        self.obs, info = self.env.reset(seed=i, options={"episode": ep})
        self.env._last_obs = self.obs
        base = self.env.unwrapped
        if self.method == "b2":
            # evaluate.py draws every B2 plan from one rng (seed 0) in episode order: replay the draws before i
            env0, checker = UR5ePoseEnv(EnvConfig(**dataclasses.asdict(base.cfg))), CollisionChecker()
            rng = np.random.default_rng(0)
            for k in range(i):
                env0.reset(seed=k, options={"episode": self.episodes[k]})
                plan_b2(env0, rng, checker)
            plan = plan_b2(base, rng, checker)
            self.reference = plan.reference if plan.ok else None
            self.tracker = ReferenceTracker(base, self.reference) if plan.ok else None
            plan_ok = plan.ok
        else:
            self.reference, plan_ok = self.env.reference, info["base_ok"]
        T = flange_to_tcp(base)
        self.ref_tcp = [] if self.reference is None else [fk_tcp(q, T)[:3, 3] for q in self.reference[::2]]
        self.correction, self.outcome, self.t = True, None, 0
        if self.method == "hybrid":
            self.env.set_alpha(self.alpha)
        self.trail.clear()
        print(f"episode {i}: method {self.method}, base plan {'ok' if plan_ok else 'FAILED (straight-line fallback)'}"
              f", {int(base.ob_mask.sum())} obstacles")

    def step(self):
        """One control step; returns True when the episode has ended."""
        if self.outcome is not None:
            return True
        if self.method == "b2":
            a = self.tracker() if self.tracker else np.zeros(6)
        elif self.method == "base":
            a = np.zeros(6)
        else:
            a = self.model.predict(self.obs, deterministic=True)[0]
        self.obs, _, term, trunc, info = self.env.step(a)
        self.t += 1
        base = self.env.unwrapped
        self.trail.append(base.data.site_xpos[base.tcp_id].copy())
        self.hit = getattr(self, "hit", False) or bool(info["collision"])
        if term or trunc:
            self.outcome = ("SUCCESS" if info["is_success"] else "COLLISION" if info["collision"] else
                            f"MISS ({100 * info['d_pos']:.1f} cm, {np.rad2deg(info['theta']):.1f} deg)")
            print(f"episode {self.index} ended at step {self.t} ({self.t * base.cfg.frame_skip * base.cfg.timestep:.1f} s):"
                  f" {self.outcome}")
            return True
        return False

    def draw(self, scn, clear=True):
        super().draw(scn, clear)
        overlays.path(scn, self.ref_tcp)


def load(args):
    if args.run:
        run = Path(args.run)
        cfg = json.loads((run / "config.json").read_text())["config"]
        env_cfg = dict(cfg["env"], n_obstacles=tuple(cfg["env"]["n_obstacles"]))
        algo = cfg["agent"]["algo"]
    else:
        env_cfg = yaml.safe_load((ROOT / "configs" / "env" / f"{args.stage}.yaml").read_text())
        env_cfg["n_obstacles"] = tuple(env_cfg["n_obstacles"])
        algo = None
    env_cfg.update(curriculum=False, visual=not args.headless)
    known = {f.name for f in dataclasses.fields(EnvConfig)}
    dropped = sorted(set(env_cfg) - known)  # runs saved before a config field was renamed
    if dropped:
        print(f"ignoring unknown env config keys from the run: {dropped}")
    env_cfg = {k: v for k, v in env_cfg.items() if k in known}
    env = UR5ePoseEnv(EnvConfig(**env_cfg))
    if args.run:
        model_path = Path(args.run) / ("best_model.zip" if (Path(args.run) / "best_model.zip").exists()
                                       else "last_model.zip")
        model = ALGOS[algo].load(model_path, env=DummyVecEnv([lambda: UR5ePoseEnv(EnvConfig(**env_cfg))]),
                                 device="cpu")
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
        print(f"policy: {model_path}")
    elif args.policy == "random":
        policy = lambda obs: env.action_space.sample()
    else:
        policy = lambda obs: np.zeros(6)
    return env, policy


def record(player, path, fps=25, size=(720, 960), hold_s=1.5):
    """Offscreen MP4 of one episode (control rate 25 Hz = real time), the final frame held for hold_s."""
    import imageio.v2 as imageio
    base = player.env.unwrapped
    base.model.vis.global_.offheight = max(base.model.vis.global_.offheight, size[0])  # scene default: 640 x 480
    base.model.vis.global_.offwidth = max(base.model.vis.global_.offwidth, size[1])
    renderer = mujoco.Renderer(base.model, *size)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.35, 0.0, 0.35]
    cam.distance, cam.azimuth, cam.elevation = 2.0, 135.0, -25.0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8) as w:
        done = False
        while not done:
            done = player.step()
            renderer.update_scene(base.data, cam)
            player.draw(renderer.scene, clear=False)
            frame = renderer.render()
            w.append_data(frame)
        for _ in range(int(hold_s * fps)):
            w.append_data(frame)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="run directory (free mode: config.json + best_model.zip; replay: residual run)")
    p.add_argument("--stage", default="s1", help="env config when no --run is given")
    p.add_argument("--policy", choices=["zero", "random"], default="zero", help="when no --run is given")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--steps", type=int, default=500, help="headless only")
    p.add_argument("--testset", help="replay mode: stored test set, e.g. s4_test")
    p.add_argument("--episode", type=int, default=0, help="replay mode: episode index (as in evaluate.py)")
    p.add_argument("--method", choices=["base", "hybrid", "b2"], default="base", help="replay mode")
    p.add_argument("--budget", type=int, default=3000, help="replay mode: base budget in collision checks")
    p.add_argument("--checkpoint", choices=["best", "primary", "final"], default="primary",
                   help="replay mode, hybrid: SPEC §9.2 checkpoint rule")
    p.add_argument("--record", help="replay mode: write an MP4 of the episode instead of opening a window")
    args = p.parse_args()

    if args.testset:
        player = Replay(args)
    else:
        env, policy = load(args)
        player = Player(env, policy)
    player.reset(new=False)
    base = player.env.unwrapped
    control_dt = base.cfg.frame_skip * base.cfg.timestep

    if args.record:
        return record(player, args.record)
    if args.headless:
        scn = mujoco.MjvScene(base.model, maxgeom=1000)
        for _ in range(args.steps):
            if player.step():
                break
            player.draw(scn)
        print(f"headless: overlay geoms {scn.ngeom}, "
              f"TCP-goal {np.linalg.norm(base.data.site_xpos[base.tcp_id] - base.goal[:3]) * 100:.1f} cm")
        return

    with mujoco.viewer.launch_passive(base.model, base.data, key_callback=player.on_key) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            with viewer.lock():
                if player.pending:
                    player.handle_pending()
                if not player.paused:
                    player.step()
                player.draw(viewer.user_scn)
            viewer.sync()
            time.sleep(max(0.0, control_dt - (time.perf_counter() - t0)))  # real time at the control rate


if __name__ == "__main__":
    main()
