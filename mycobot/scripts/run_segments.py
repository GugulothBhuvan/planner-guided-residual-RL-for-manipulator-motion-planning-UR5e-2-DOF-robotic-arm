"""Run one S4 episode on the arm as a few firmware-planned SEGMENTS instead of 50 ms streaming (SPEC M7g, option A).

    python scripts/run_segments.py --episode 3 --method hybrid --budget 3000 --run <dir> --checkpoint final --dry-run
    python scripts/run_segments.py --episode 3 --method b2 --execute --speed 30        # real arm
    python scripts/run_segments.py --probe --execute                                    # small multi-joint straight-line test

How it works
 1. The episode is first played in the model with the usual controller (cheap planner / B2 / hybrid residual) on the
    simulated arm, exactly as run_real.py does in --dry-run. That gives the nominal joint-target path.
 2. The path is thinned to a few waypoints (Douglas-Peucker, --tol deg), and every straight joint-space segment between
    them is re-checked against the table, the arm itself and the virtual obstacles in their exact shapes, with an extra
    --margin metres of clearance. A segment that fails is split again.
 3. On the arm each waypoint is one send_angles(...,speed): the firmware plans the ramp. The next waypoint is sent when
    the arm is within --near deg of the current one. Joint readings are polled continuously; the run is aborted
    (stop()) if the arm leaves the polyline by more than --dev deg, a limit would be violated, or the measured pose
    touches an obstacle in the model.
 4. At the end the same final-approach correction as TrimArm (shortfall nudged out of the servo deadband).

Everything is logged in the run_real.py format (per 20 Hz tick: q, target, d_pos, theta, collision) so that
make_comparison.py / traj_metrics.py work unchanged. Default log directory: results/seg/.
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
import evaluate  # noqa: E402
from mcrl import robot  # noqa: E402
from mcrl.arm import FakeArm, SafetyAbort, TrimArm  # noqa: E402
from mcrl.envs.hardware_env import HardwareEnv  # noqa: E402
from mcrl.envs.pose_env import EnvConfig  # noqa: E402
from mcrl.envs.residual import ResidualEnv  # noqa: E402
from mcrl.planners.baselines import ReferenceTracker, plan_b2, shape_kw  # noqa: E402
from mcrl.planners.collision import CollisionChecker  # noqa: E402
from mcrl.shapes import load_testset  # noqa: E402
from ur5rl.planners.quintic import time_scale  # noqa: E402
from ur5rl.planners.rrt_connect import JointSpace, rrt_connect, shortcut  # noqa: E402

LIMITS = robot.LIMITS_RAD
MAX_SPEED = 50  # firmware speed setting; the diagnostic (SPEC M7f) saturates around 40-60
DEG = np.pi / 180


# ----------------------------------------------------------------------------------------------------------------
# 1. nominal path from the model
# ----------------------------------------------------------------------------------------------------------------
def build_env(args, ep, env_cfg, arm):
    """Environment pieces as in run_real.py. Returns (step_env, inner, act, obs)."""
    rng = np.random.default_rng(args.episode)
    if args.method == "b2":
        env = HardwareEnv(EnvConfig(**env_cfg), arm)
        env.reset(seed=args.episode, options={"episode": ep})
        return rng, env, env, None
    model, cfg = (evaluate.load_policy(args.run, args.checkpoint) if args.method == "hybrid" else (None, None))
    base_kw = dict(cfg["agent"]["base"]) if cfg else dict(inflate=0.017, shortcut_iters=20)
    if cfg:
        base_kw["inflate"] = 0.017
    base_kw["max_checks"] = args.budget
    alpha = cfg["agent"]["alpha_max"] if cfg else 0.0
    inner = HardwareEnv(EnvConfig(**env_cfg), arm)
    renv = ResidualEnv(inner, alpha=alpha, **base_kw)
    renv.reset(seed=args.episode, options={"episode": ep})
    return rng, renv, inner, model


def simulate(args, ep, env_cfg):
    """Play the episode on the simulated arm; return nominal targets (N+1, 6) starting at q_start, the sim outcome."""
    arm = TrimArm(FakeArm(np.clip(robot.HOME_Q, LIMITS[:, 0], LIMITS[:, 1]), seed=args.episode))
    rng, step_env, inner, model = build_env(args, ep, env_cfg, arm)
    q_start = inner.q_start.copy()
    quiet(approach_sim(arm, inner, q_start, rng))
    if args.method == "b2":
        obs, _ = inner.reset(seed=args.episode, options={"episode": ep})
        plan = plan_b2(inner, rng, CollisionChecker())
        if not plan.ok:
            raise SafetyAbort(f"B2 found no plan: {plan.reason}")
        tracker = ReferenceTracker(inner, plan.reference)
        act = lambda o: tracker()  # noqa: E731
    else:
        obs, info0 = step_env.reset(seed=args.episode, options={"episode": ep})
        print("base plan ok:", info0["base_ok"], "plan time %.2f s" % info0["base_plan_time"], flush=True)
        act = (lambda o: model.predict(o, deterministic=True)[0]) if model else (lambda o: np.zeros(6))
    targets, last = [q_start], None
    while True:
        obs, _, term, trunc, info = step_env.step(act(obs))
        targets.append(info["q_target"].copy())
        last = info
        if term or trunc:
            break
    out = dict(success=bool(last["is_success"]), collision=bool(last["collision"]), ticks=len(targets) - 1,
               d_pos_cm=100 * last["d_pos"], theta_deg=float(np.rad2deg(last["theta"])))
    return np.array(targets), out, inner


def quiet(x):
    return x


def approach_sim(arm, env, q_goal, rng):
    for q in approach_ticks(arm.q.copy(), env, q_goal, rng):
        arm.tick(q)
    for _ in range(10):
        arm.tick(q_goal)


def approach_path(env, q0, q_goal, rng):
    """Collision-free (table / self; obstacles are virtual) polyline from q0 to q_goal, as run_real.go_to."""
    ck = CollisionChecker()
    ck.set_obstacles(np.zeros((ck.K, 3)), np.full(ck.K, 0.05), np.zeros(ck.K, bool))
    if np.abs(q0 - q_goal).max() < np.deg2rad(1.0):
        return np.array([q0, q_goal])
    if ck(q0):
        raise SafetyAbort("the arm's current pose collides with the table/itself in the model; move it by hand first")
    space = JointSpace(env.q_lo, env.q_hi)
    path = rrt_connect(space, ck, q0, q_goal, rng, step=0.3, edge_res=0.01, time_budget=10.0)
    if path is None:
        raise SafetyAbort("no collision-free path to the episode start")
    return np.asarray(shortcut(space, ck, path, rng, iters=100, edge_res=0.01))


def approach_ticks(q0, env, q_goal, rng):
    path = approach_path(env, q0, q_goal, rng)
    ref, _ = time_scale(path, dt=robot.CONTROL_DT, v_max=robot.V_MAX, a_max=robot.A_MAX)
    return ref[1:]


# ----------------------------------------------------------------------------------------------------------------
# 2. thinning + segment verification
# ----------------------------------------------------------------------------------------------------------------
def make_checker(inner, margin):
    ck = CollisionChecker()
    ck.set_obstacles(inner.ob_center, inner.ob_radius, inner.ob_mask, inflate=margin, **shape_kw(inner))
    return ck


def sheet(a, b, res):
    """All configurations with J1-J3 at progress s and J4-J6 at progress w along a -> b, (s, w) in [0, 1]^2: covers any lag or lead
    of the wrist against the base joints (measured: J4-J6 start 0.3-0.5 s late and their readings freeze, SPEC M7g)."""
    n = max(2, int(np.ceil(np.abs(b - a).max() / res)) + 1)
    u = np.linspace(0.0, 1.0, n)
    S, W = np.meshgrid(u, u, indexing="ij")
    S, W = S.ravel()[:, None], W.ravel()[:, None]
    return np.hstack([a[:3] + (b[:3] - a[:3]) * S, a[3:] + (b[3:] - a[3:]) * W])


_CORNERS = np.array([[(i >> k) & 1 for k in range(6)] for i in range(64)], float)


def box_samples(a, b, n=1500, seed=0):
    """Corners + random points of the box spanned by a and b: every joint at an independent progress (stale or late joints)."""
    U = np.vstack([_CORNERS, np.random.default_rng(seed).random((n, 6))])
    return a + (b - a) * U


def segment_free(ck, a, b, res=0.01):
    return not any(ck(q) for q in sheet(a, b, res)) and not any(ck(q) for q in box_samples(a, b))


def box_excess(q, a, b):
    """How far (rad) any joint is outside the span of its two end values."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return float(np.maximum(np.maximum(lo - q, q - hi), 0.0).max())


def dev_sheet(q, a, b):
    """Distance of a reading from the verified sheet: J1-J3 from the straight line, wrist outside the span of its end values."""
    ab = b[:3] - a[:3]
    n2 = float(ab @ ab)
    t = 0.0 if n2 < 1e-12 else float(np.clip((q[:3] - a[:3]) @ ab / n2, 0.0, 1.0))
    d123 = np.abs(q[:3] - (a[:3] + t * ab)).max()
    lo, hi = np.minimum(a[3:], b[3:]), np.maximum(a[3:], b[3:])
    dw = np.maximum(np.maximum(lo - q[3:], q[3:] - hi), 0.0).max()
    return max(d123, dw)


def thin(T, ck, tol_rad):
    """Waypoint indices into T: Douglas-Peucker (max-norm, per tick) with every kept segment collision-checked."""
    keep = {0, len(T) - 1}

    def rec(i, j):
        if j <= i + 1:
            return
        t = (np.arange(i + 1, j) - i) / (j - i)
        line = T[i] + (T[j] - T[i]) * t[:, None]
        dev = np.abs(T[i + 1:j] - line).max(1)
        k = int(np.argmax(dev)) + i + 1
        if dev.max() <= tol_rad and segment_free(ck, T[i], T[j]):
            return
        keep.add(k)
        rec(i, k)
        rec(k, j)

    rec(0, len(T) - 1)
    return sorted(keep)


def refine(ck, path, depth=5):
    """Split any segment whose lag sheet touches something (table / self) at its midpoint, recursively."""
    out = [path[0]]

    def rec(a, b, d):
        if segment_free(ck, a, b) or d == 0:
            out.append(b)
            return
        m = 0.5 * (a + b)
        rec(a, m, d - 1)
        rec(m, b, d - 1)

    for a, b in zip(path[:-1], path[1:]):
        rec(a, b, depth)
    return np.array(out)


def trim_tail(T, eps=0.05 * DEG):
    far = np.where(np.abs(T - T[-1]).max(1) > eps)[0]
    return T[: (far[-1] + 2 if len(far) else 1)]


# ----------------------------------------------------------------------------------------------------------------
# 3. arms (real / fake) with the same minimal mc API
# ----------------------------------------------------------------------------------------------------------------
class FakeMC:
    """Wall-clock stand-in for MyCobot280: joint-synchronised trapezoid toward the last goal, 0.8 deg deadband, noise."""

    def __init__(self, q0_deg, seed=0):
        self.q = np.array(q0_deg, float)
        self.seg = None
        self.rng = np.random.default_rng(seed)
        self.sent = 0

    def _now(self):
        return time.perf_counter()

    def send_angles(self, deg, speed):
        self._advance()
        self.sent += 1
        goal = np.array(deg, float)
        d = np.abs(goal - self.q).max()
        vmax, acc = min(1.75 * speed, 75.0), 250.0
        self.seg = dict(t0=self._now(), a=self.q.copy(), b=goal, d=d, vmax=vmax, acc=acc,
                        T=(d / vmax + vmax / acc) if d > 0 else 0.0)

    def _advance(self):
        s = self.seg
        if s is None:
            return
        t = self._now() - s["t0"]
        d, vmax, acc = s["d"], s["vmax"], s["acc"]
        if d <= 0:
            return
        ta = vmax / acc
        if d < vmax * ta:  # triangle
            ta = np.sqrt(d / acc)
            tt = 2 * ta
            sdist = 0.5 * acc * t * t if t < ta else (d - 0.5 * acc * max(tt - t, 0.0) ** 2)
        else:
            tt = d / vmax + ta
            sdist = 0.5 * acc * t * t if t < ta else (vmax * (t - ta / 2) if t < tt - ta else d - 0.5 * acc * max(tt - t, 0.0) ** 2)
        frac = min(sdist / d, 1.0) if t < tt else 1.0
        tgt = s["a"] + (s["b"] - s["a"]) * frac
        # deadband: the servo does not close the last 0.8 deg
        short = np.where(np.abs(s["b"] - tgt) < 0.8, np.sign(s["b"] - s["a"]) * -0.8, 0.0) if frac >= 1.0 else 0.0
        self.q = tgt + short * (np.abs(s["b"] - s["a"]) > 0.8)

    def get_angles(self):
        time.sleep(0.012)
        self._advance()
        return (self.q + self.rng.normal(0, 0.05, 6)).tolist()

    def get_coords(self):
        return None

    def stop(self):
        self.seg = None


def make_mc(args, q0_deg):
    if args.dry_run:
        return FakeMC(q0_deg, seed=args.episode)
    from pymycobot import MyCobot280
    mc = MyCobot280(args.port, 115200)
    time.sleep(1.0)
    return mc


def read_q(mc, tries=6):
    for _ in range(tries):
        a = mc.get_angles()
        if isinstance(a, list) and len(a) >= 6:
            return np.array(a[:6], float) * DEG
        time.sleep(0.03)
    raise SafetyAbort("get_angles failed 6x")


def dist_to_segment(q, a, b):
    ab = b - a
    n2 = float(ab @ ab)
    t = 0.0 if n2 < 1e-12 else float(np.clip((q - a) @ ab / n2, 0.0, 1.0))
    return np.abs(q - (a + t * ab)).max()


class Runner:
    """Executes a polyline with firmware segments; records every poll."""

    def __init__(self, mc, speed, near_deg, dev_deg, guard, stop_each=True):
        self.stop_each = stop_each
        self.mc, self.speed, self.near, self.dev = mc, speed, near_deg * DEG, dev_deg * DEG
        self.guard = guard  # callable(q) -> raises SafetyAbort when the measured pose is unsafe
        self.trace = []  # (t, q(6), seg_index)
        self.t0 = None

    def _poll(self, seg):
        try:
            q = read_q(self.mc, tries=3)
            self._last_ok = (time.perf_counter(), q)
        except SafetyAbort:
            last = getattr(self, "_last_ok", None)
            if last is None or time.perf_counter() - last[0] > 2.5:
                raise
            time.sleep(0.03)
            return last[1].copy()  # a short serial stall during fast motion: keep the last good reading, no new trace entry
        self.trace.append((time.perf_counter() - self.t0, q.copy(), seg))
        if np.any(q < LIMITS[:, 0] - 0.02) or np.any(q > LIMITS[:, 1] + 0.02):
            raise SafetyAbort(f"measured pose outside the joint limits: {np.rad2deg(q).round(1)}")
        self.guard(q)
        return q

    def run(self, V, final_wait=True, label="path"):
        if self.t0 is None:
            self.t0 = time.perf_counter()
        V = np.asarray(V, float)
        if np.any(V < LIMITS[:, 0] - 1e-9) or np.any(V > LIMITS[:, 1] + 1e-9):
            raise SafetyAbort("a waypoint is outside the joint limits")
        q = self._poll(-1)
        poly = np.vstack([q, V]) if np.abs(q - V[0]).max() > 0.5 * DEG else V
        sent_to = 0 if len(poly) == len(V) else 1
        # poly[0] is the pose we start from; segment k goes poly[k] -> poly[k+1]
        for k in range(1, len(poly)):
            last = k == len(poly) - 1
            dmax = np.abs(poly[k] - poly[k - 1]).max()
            self.mc.send_angles(np.rad2deg(poly[k]).tolist(), self.speed)
            time.sleep(0.03)
            ts = time.perf_counter()
            timeout = 3.0 + dmax / (8 * DEG)  # >= 8 deg/s overall, far below the measured >= 35 deg/s
            hist, viol = [], None
            while True:
                q = self._poll(k)
                near_segs = [(poly[j - 1], poly[j]) for j in (k - 1, k) if j >= 1]
                dev = min(dev_sheet(q, a, b) for a, b in near_segs)
                ex = min(box_excess(q, a, b) for a, b in near_segs)
                if ex > 3.0 * DEG:
                    raise SafetyAbort(f"{label}: a joint is {np.rad2deg(ex):.1f} deg outside the span of segment {k}")
                now = time.perf_counter()
                if dev > self.dev:  # readings can freeze for ~0.5 s in fast moves: only a persistent deviation is real
                    viol = viol or now
                    if now - viol > 0.8:
                        raise SafetyAbort(f"{label}: arm {np.rad2deg(dev):.1f} deg off the planned segment {k} for {now - viol:.1f} s")
                else:
                    viol = None
                hist.append(q)
                err = np.abs(q - poly[k]).max()
                if last or self.stop_each:
                    # settled: not moving, close enough (the servo stops 0.8-1.9 deg short, J2 under load most), and the command
                    # had time to start (the wrist starts 0.3-0.5 s late); the exact approach is left to final_trim
                    if err < 4.0 * DEG and len(hist) > 6 and np.abs(hist[-1] - hist[-6]).max() < 0.1 * DEG                             and time.perf_counter() - ts > 0.35:
                        break
                elif err < self.near:
                    break
                if time.perf_counter() - ts > timeout:
                    raise SafetyAbort(f"{label}: segment {k} not reached after {timeout:.1f} s ({np.rad2deg(err):.1f} deg left)")
        return q

    def settle(self, seconds=0.5):
        t = time.perf_counter()
        while time.perf_counter() - t < seconds:
            self._poll(-1)

    def final_trim(self, target, rounds=6, tol_deg=0.35, gain=0.6, max_deg=2.0):
        """TrimArm-like: nudge the command by the measured shortfall (servo deadband), at most max_deg from the target."""
        cmd = target.copy()
        for _ in range(rounds):
            q = self._poll(-2)
            err = target - q
            if np.abs(err).max() < tol_deg * DEG:
                break
            cmd = np.clip(cmd + gain * err, target - max_deg * DEG, target + max_deg * DEG)
            cmd = np.clip(cmd, LIMITS[:, 0], LIMITS[:, 1])
            self.mc.send_angles(np.rad2deg(cmd).tolist(), 15)
            time.sleep(0.03)
            t = time.perf_counter()
            hist = []
            while time.perf_counter() - t < 2.5:
                hist.append(self._poll(-2))
                if len(hist) > 6 and np.abs(hist[-1] - hist[-6]).max() < 0.08 * DEG:
                    break
        return self._poll(-2)


# ----------------------------------------------------------------------------------------------------------------
# 4. logging in the run_real format
# ----------------------------------------------------------------------------------------------------------------
def to_ticks(trace, inner, t_begin, waypoints_by_seg):
    """Resample the polled trace onto the 20 Hz control grid (wall = seconds since t_begin) and score every tick."""
    tr = [x for x in trace if x[0] >= t_begin]
    t = np.array([x[0] for x in tr]) - t_begin
    Q = np.array([x[1] for x in tr])
    S = np.array([x[2] for x in tr])
    grid = np.arange(1, int(t[-1] / robot.CONTROL_DT) + 1) * robot.CONTROL_DT
    ticks = []
    for g in grid:
        q = np.array([np.interp(g, t, Q[:, j]) for j in range(6)])
        seg = int(S[min(np.searchsorted(t, g), len(S) - 1)])
        inner._sync(q, np.zeros(6))
        d_pos, theta = inner._errors(inner.achieved_goal, inner.goal[None])
        ticks.append(dict(t=len(ticks) + 1, wall=float(g), success=bool(inner._success(d_pos, theta)[0]), collision=bool(inner.data.ncon > 0),
                          d_pos=float(d_pos[0]), theta=float(theta[0]), q=q.tolist(),
                          target=(waypoints_by_seg.get(seg, q)).tolist()))
    return ticks


def beep():
    import threading
    import winsound
    threading.Thread(target=winsound.Beep, args=(1000, 300), daemon=True).start()


# ----------------------------------------------------------------------------------------------------------------
def probe(args):
    """Small multi-joint move: is the firmware path a straight line in joint space (all joints finish together)?"""
    mc = make_mc(args, [0, 0, 0, 0, 0, 0])
    q0 = read_q(mc)
    d = np.array([float(x) for x in args.delta.split(",")]) * DEG
    target = np.clip(q0 + d, LIMITS[:, 0], LIMITS[:, 1])
    ck = CollisionChecker()
    ck.set_obstacles(np.zeros((ck.K, 3)), np.full(ck.K, 0.05), np.zeros(ck.K, bool))
    if ck(q0) or ck(target) or not segment_free(ck, q0, target):
        raise SystemExit("probe move is not collision-free in the model from this pose")
    print("probe: from", np.rad2deg(q0).round(1).tolist(), "by", np.rad2deg(d).tolist(), "speed", args.speed, flush=True)
    r = Runner(mc, args.speed, 1.5, 5.0, lambda q: None)
    try:
        for name, goal in (("out", target), ("back", q0)):
            r.t0 = None
            r.trace = []
            r.run(np.array([goal]), label="probe_" + name)
            r.settle(0.5)
            tr = r.trace
            a = q0 if name == "out" else target
            b = goal
            devs = [dist_to_segment(x[1], a, b) for x in tr]
            tt = [x[0] for x in tr]
            print(f"probe {name}: {len(tr)} polls in {tt[-1]:.2f} s, max deviation from the straight joint-space line "
                  f"{np.rad2deg(max(devs)):.2f} deg, final error {np.rad2deg(np.abs(tr[-1][1] - goal).max()):.2f} deg", flush=True)
        json.dump(dict(delta=args.delta, speed=args.speed, trace=[(t, q.tolist(), s) for t, q, s in tr]), open(ROOT / "results" / f"seg_probe_{args.tag_suffix or 'x'}.json", "w"))
    except BaseException as exc:
        print("ERROR", repr(exc), flush=True)
        mc.stop()
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="s4_test")
    p.add_argument("--episode", type=int)
    p.add_argument("--method", choices=["b2", "base", "hybrid"])
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--run")
    p.add_argument("--checkpoint", default="primary")
    p.add_argument("--port", default="COM13")
    p.add_argument("--speed", type=int, default=30)
    p.add_argument("--approach-speed", type=int, default=30)
    p.add_argument("--tol", type=float, default=2.0, help="waypoint thinning tolerance, deg (max-norm per tick)")
    p.add_argument("--margin", type=float, default=0.0, help="extra obstacle clearance when verifying segments, m")
    p.add_argument("--near", type=float, default=1.5, help="send the next waypoint when the arm is this close (deg) to the current one")
    p.add_argument("--dev", type=float, default=5.0, help="abort if the arm is further than this from the planned polyline, deg")
    p.add_argument("--out-dir", default="results/seg")
    p.add_argument("--tag-suffix", default="")
    p.add_argument("--hold-before", type=float, default=0.0)
    p.add_argument("--beep", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--blend", action="store_true", help="send the next waypoint while still moving (default: settle at every waypoint)")
    p.add_argument("--plan-only", action="store_true", help="simulate + thin, print the segment count, do not touch the arm")
    p.add_argument("--delta", default="20,6,-10,4,0,0", help="probe move, deg per joint")
    args = p.parse_args()
    assert 1 <= args.speed <= MAX_SPEED and 1 <= args.approach_speed <= MAX_SPEED
    if not args.dry_run and not args.execute:
        sys.exit("refusing to move the real arm without --execute (use --dry-run to simulate)")
    if args.probe:
        return probe(args)
    assert args.episode is not None and args.method

    episodes, meta = load_testset(args.set, ROOT / "testsets")
    ep = episodes[args.episode]
    env_cfg = evaluate.stage_env(meta)
    T, sim, inner = simulate(args, ep, env_cfg)
    print("simulation:", json.dumps(sim), flush=True)
    T = trim_tail(T)
    ck = make_checker(inner, args.margin)
    idx = thin(T, ck, args.tol * DEG)
    V = T[idx]
    travel = float(np.abs(np.diff(V, axis=0)).max(1).sum())
    print(f"{len(T) - 1} nominal ticks -> {len(V) - 1} segments (tol {args.tol} deg, margin {args.margin * 100:.1f} cm); "
          f"largest segment {np.rad2deg(np.abs(np.diff(V, axis=0)).max()):.1f} deg; sum of max-joint moves {np.rad2deg(travel):.0f} deg", flush=True)
    for k in range(len(V) - 1):
        assert segment_free(ck, V[k], V[k + 1], 0.005), f"segment {k} not free"

    if args.plan_only:
        return
    tag = f"{'dry' if args.dry_run else 'real'}_{args.set}_ep{args.episode}_{args.method}" + \
          (f"@{args.budget}" if args.method != "b2" else "") + args.tag_suffix
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    q_start = T[0]
    rng = np.random.default_rng(args.episode)
    mc = make_mc(args, np.rad2deg(np.clip(robot.HOME_Q, LIMITS[:, 0], LIMITS[:, 1])))
    guard_ck = CollisionChecker()
    guard_ck.set_obstacles(inner.ob_center, inner.ob_radius, inner.ob_mask, **shape_kw(inner))
    table_ck = CollisionChecker()
    table_ck.set_obstacles(np.zeros((table_ck.K, 3)), np.full(table_ck.K, 0.05), np.zeros(table_ck.K, bool))
    phase = {"obstacles": False}

    def guard(q):
        if (guard_ck if phase["obstacles"] else table_ck)(q):
            raise SafetyAbort("measured pose collides in the model (table / self / virtual obstacle)")

    runner = Runner(mc, args.approach_speed, args.near, args.dev, guard, stop_each=not args.blend)
    status, summary, ticks = "ok", {}, []
    try:
        q_now = read_q(mc)
        print("arm now:", np.rad2deg(q_now).round(1).tolist(), flush=True)
        path0 = refine(table_ck, approach_path(inner, q_now, q_start, rng))
        print(f"moving to the episode start: {len(path0) - 1} segments", flush=True)
        runner.run(path0[1:], label="approach")
        runner.final_trim(q_start, rounds=4)
        if args.hold_before > 0:
            print(f"holding the start pose for {args.hold_before:.0f} s", flush=True)
            runner.settle(args.hold_before)
        print(f"start error {np.rad2deg(np.abs(read_q(mc) - q_start).max()):.2f} deg; executing {len(V) - 1} segments", flush=True)
        runner.speed = args.speed
        phase["obstacles"] = True
        runner.t0 = runner.t0 or time.perf_counter()
        t_begin = time.perf_counter() - runner.t0
        epoch0 = time.time()
        if args.beep:
            beep()
        runner.run(V[1:], label="episode")
        runner.final_trim(V[-1])
        runner.settle(0.3)
        t_end = time.perf_counter() - runner.t0
        wp_by_seg = {k + 1: V[k + 1] for k in range(len(V) - 1)}
        wp_by_seg.update({-1: V[-1], -2: V[-1]})
        ticks = to_ticks(runner.trace, inner, t_begin, wp_by_seg)
        final = ticks[-1]
        coords = mc.get_coords() if not args.dry_run else None
        summary = dict(tag=tag, status=status, success=bool(final["success"]), collision=bool(any(x["collision"] for x in ticks)),
                       d_pos_cm=100 * final["d_pos"], theta_deg=float(np.rad2deg(final["theta"])), ticks=len(ticks),
                       wall_s=t_end - t_begin, dry_run=args.dry_run, trim=True, epoch_start=epoch0, speed=args.speed,
                       mode="segments", segments=len(V) - 1, tol_deg=args.tol, near_deg=args.near,
                       sim=sim, coords_end=coords if isinstance(coords, list) else None)
    except (SafetyAbort, KeyboardInterrupt, Exception) as exc:
        try:
            mc.stop()
        except Exception:
            pass
        status = f"ABORT: {exc!r}"
        summary = dict(tag=tag, status=status, ticks=len(ticks), dry_run=args.dry_run, mode="segments")
        print(status, flush=True)
    (out / f"{tag}.json").write_text(json.dumps(dict(summary=summary, ticks=ticks, waypoints=np.rad2deg(V).tolist(),
                                                      raw=[(t, q.tolist(), s) for t, q, s in runner.trace])))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
