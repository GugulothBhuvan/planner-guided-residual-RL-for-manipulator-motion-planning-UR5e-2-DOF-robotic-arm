"""Run one frozen S4 test episode on the arm (or a simulated arm with --dry-run), SPEC M7.

    python mycobot/scripts/run_real.py --episode 3 --method base   --budget 3000 --dry-run
    python mycobot/scripts/run_real.py --episode 3 --method hybrid --budget 3000 --run <run dir> --checkpoint primary --dry-run
    python mycobot/scripts/run_real.py --episode 3 --method b2 --execute      # real arm: needs --execute and pymycobot

Obstacles are virtual keep-out spheres (checked on the measured configuration; nothing real is placed). The arm first
moves to the episode's start through a collision-free path, then runs the controller on the real measurements at the
send -> 30 ms -> read cycle (~49 ms). Safety: joint limits, <= 1 deg commanded step, abort if the arm is > 5 deg from the
target, stop() on any exception or Ctrl-C. Per-tick log -> results/real/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402  (ported evaluate.py: stage_env, load_policy)
from mcrl import robot  # noqa: E402
from mcrl.arm import FakeArm, RealArm, SafetyAbort, TrimArm  # noqa: E402
from mcrl.envs.hardware_env import HardwareEnv  # noqa: E402
from mcrl.envs.pose_env import EnvConfig  # noqa: E402
from mcrl.envs.residual import ResidualEnv  # noqa: E402
from mcrl.planners.baselines import ReferenceTracker, plan_b2  # noqa: E402
from mcrl.planners.collision import CollisionChecker  # noqa: E402
from mcrl.shapes import load_testset  # noqa: E402
from ur5rl.planners.quintic import time_scale  # noqa: E402
from ur5rl.planners.rrt_connect import JointSpace, rrt_connect, shortcut  # noqa: E402


def go_to(arm, env, q_goal, rng):
    """Collision-free (table/self only; the obstacles are virtual) joint-space move of the arm to q_goal."""
    ck = CollisionChecker()
    ck.set_obstacles(np.zeros((ck.K, 3)), np.full(ck.K, 0.05), np.zeros(ck.K, bool))
    q0 = arm.q.copy()
    if np.abs(q0 - q_goal).max() < np.deg2rad(1.0):
        return
    if ck(q0):
        raise SafetyAbort("the arm's current pose collides with the table/itself in the model; move it by hand first")
    space = JointSpace(env.q_lo, env.q_hi)
    path = rrt_connect(space, ck, q0, q_goal, rng, step=0.3, edge_res=0.01, time_budget=10.0)
    if path is None:
        raise SafetyAbort("no collision-free path to the episode start")
    path = shortcut(space, ck, path, rng, iters=100, edge_res=0.01)
    ref, dur = time_scale(path, dt=robot.CONTROL_DT, v_max=robot.V_MAX, a_max=robot.A_MAX)
    print(f"moving to the episode start: {len(ref)} ticks, {dur:.1f} s", flush=True)
    for q in ref[1:]:
        arm.tick(q)
    for _ in range(10):  # settle
        arm.tick(q_goal)


def beep():
    """1 kHz, 300 ms, in a thread so the control loop is not delayed (the tone starts with tick 0)."""
    import threading
    import winsound
    threading.Thread(target=winsound.Beep, args=(1000, 300), daemon=True).start()


def hold(arm, q, seconds):
    if seconds > 0:
        print(f"holding the start pose for {seconds:.0f} s, then the episode begins", flush=True)
    for _ in range(int(seconds / robot.CONTROL_DT)):
        arm.tick(q)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="s4_test")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--method", choices=["b2", "base", "hybrid"], required=True)
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--run")
    p.add_argument("--checkpoint", default="primary")
    p.add_argument("--port", default="COM13")
    p.add_argument("--speed", type=int, default=20)
    p.add_argument("--dry-run", action="store_true", help="simulated arm (lag, deadband, noise); no hardware")
    p.add_argument("--send-every", type=int, default=1, help="send a new target every N control ticks (1 = 50 ms)")
    p.add_argument("--out-dir", default="results/real", help="log directory under mycobot/ (experiments use their own)")
    p.add_argument("--tag-suffix", default="", help="appended to the log name (experiments)")
    p.add_argument("--hold-before", type=float, default=0.0, help="seconds to hold the start pose before the episode (filming)")
    p.add_argument("--beep", action="store_true", help="sound a 1 kHz tone at the first control tick (sync cue for a phone video)")
    p.add_argument("--no-trim", action="store_true", help="disable the final-approach correction (TrimArm)")
    p.add_argument("--execute", action="store_true", help="required to move the real arm")
    args = p.parse_args()
    if not args.dry_run and not args.execute:
        sys.exit("refusing to move the real arm without --execute (use --dry-run to simulate)")

    episodes, meta = load_testset(args.set, ROOT / "testsets")
    ep = episodes[args.episode]
    env_cfg = evaluate.stage_env(meta)
    arm = FakeArm(np.clip(robot.HOME_Q, robot.LIMITS_RAD[:, 0], robot.LIMITS_RAD[:, 1]), seed=args.episode) \
        if args.dry_run else RealArm(args.port, args.speed, args.send_every)
    if not args.no_trim:
        arm = TrimArm(arm)
    rng = np.random.default_rng(args.episode)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{'dry' if args.dry_run else 'real'}{'_notrim' if args.no_trim else ''}_{args.set}_ep{args.episode}_{args.method}" + \
          (f"@{args.budget}" if args.method != "b2" else "") + args.tag_suffix
    ticks, status = [], "ok"
    try:
        if args.method == "b2":
            env = HardwareEnv(EnvConfig(**env_cfg), arm)
            env.reset(seed=args.episode, options={"episode": ep})
            go_to(arm, env, env.q_start, rng)
            hold(arm, env.q_start, args.hold_before)
            obs, _ = env.reset(seed=args.episode, options={"episode": ep})
            plan = plan_b2(env, rng, CollisionChecker())
            if not plan.ok:
                raise SafetyAbort(f"B2 found no plan: {plan.reason}")
            tracker = ReferenceTracker(env, plan.reference)
            act, step_env, base = (lambda o: tracker()), env, env
        else:
            model, cfg = (evaluate.load_policy(args.run, args.checkpoint) if args.method == "hybrid" else (None, None))
            base_kw = dict(cfg["agent"]["base"]) if cfg else dict(inflate=0.017, shortcut_iters=20)
            if cfg:
                base_kw["inflate"] = 0.017
            base_kw["max_checks"] = args.budget
            alpha = cfg["agent"]["alpha_max"] if cfg else 0.0
            inner = HardwareEnv(EnvConfig(**env_cfg), arm)
            renv = ResidualEnv(inner, alpha=alpha, **base_kw)
            renv.reset(seed=args.episode, options={"episode": ep})
            go_to(arm, inner, inner.q_start, rng)
            hold(arm, inner.q_start, args.hold_before)
            obs, info0 = renv.reset(seed=args.episode, options={"episode": ep})
            print("base plan ok:", info0["base_ok"], "plan time %.2f s" % info0["base_plan_time"], flush=True)
            act = (lambda o: model.predict(o, deterministic=True)[0]) if model else (lambda o: np.zeros(6))
            step_env, base = renv, inner
        print(f"start error {np.rad2deg(np.abs(arm.q - base.q_start).max()):.2f} deg; running up to "
              f"{base.cfg.episode_length} ticks", flush=True)
        t0 = time.perf_counter()
        epoch0 = time.time()
        if args.beep:
            beep()
        while True:
            obs, _, term, trunc, info = step_env.step(act(obs))
            ticks.append(dict(t=base.t, wall=time.perf_counter() - t0, success=info["is_success"], collision=info["collision"], d_pos=info["d_pos"],
                              theta=info["theta"], q=info["q_measured"].tolist(), target=info["q_target"].tolist()))
            if term or trunc:
                break
        wall = time.perf_counter() - t0
        final = ticks[-1]
        summary = dict(tag=tag, status=status, success=bool(final["success"]), collision=bool(any(x["collision"] for x in ticks)),
                       d_pos_cm=100 * final["d_pos"], theta_deg=float(np.rad2deg(final["theta"])), ticks=len(ticks),
                       wall_s=wall, dry_run=args.dry_run, trim=not args.no_trim,
                       epoch_start=epoch0, speed=args.speed, send_every=args.send_every,
                       coords_end=None if args.dry_run else arm.final_coords())
    except (SafetyAbort, KeyboardInterrupt, Exception) as exc:
        arm.stop()
        status = f"ABORT: {exc!r}"
        summary = dict(tag=tag, status=status, ticks=len(ticks), dry_run=args.dry_run)
        print(status, flush=True)
    finally:
        arm.stop()
    (out / f"{tag}.json").write_text(json.dumps(dict(summary=summary, ticks=ticks)))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
