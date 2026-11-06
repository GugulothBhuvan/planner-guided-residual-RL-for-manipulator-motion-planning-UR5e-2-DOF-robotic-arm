"""No-learning reference controllers for the reach task (SPEC M10), in the env's `ee` action space.

straight(env): tool moves on the straight Cartesian line to the goal (saturating, direction preserving).
Detour: hand-coded, up over the obstacle, across and down: waypoints (start xy at z_top) -> (goal xy at z_top) -> goal,
z_top = max(start z, goal z, sphere top + clearance); a waypoint counts as reached within `tol` metres.
Both use only the goal, the TCP and the sphere centre/radius (the information a scripted rule would have).
"""
from __future__ import annotations

import numpy as np


Z_CEILING = 0.425


def _toward(env, target, gain=1.0):
    d = np.asarray(target, float) - env.tcp()
    n = np.abs(d).max()
    step = env.cfg.ee_step
    if n > step:
        d = d * (step / n)
    return np.clip(d / step * gain, -1.0, 1.0)


def straight(env, obs=None):
    return _toward(env, env.goal[:3])


class Detour:
    def __init__(self, env, clearance=0.045, tol=0.006):
        p0, pg = env.tcp(), env.goal[:3].copy()
        z_top = max(p0[2], pg[2])
        if env.ob_mask.any():
            k = int(np.argmax(env.ob_mask))
            z_top = max(z_top, env.ob_center[k][2] + env.ob_radius[k] + clearance)
        z_top = min(z_top, Z_CEILING)  # the tool cannot go higher than this (workspace ceiling, measured: home TCP z = 0.42)
        self.wp = [np.array([p0[0], p0[1], z_top]), np.array([pg[0], pg[1], z_top]), pg]
        self.i, self.tol = 0, tol

    def __call__(self, env, obs=None):
        while self.i < len(self.wp) - 1 and np.linalg.norm(env.tcp() - self.wp[self.i]) < self.tol:
            self.i += 1
        return _toward(env, self.wp[self.i])


def rollout(env, controller_factory, episodes=None, n=None, seed0=0, max_steps=None):
    """Run a no-learning controller on frozen episodes (or n sampled ones); returns a list of per-episode dicts."""
    rows = []
    count = len(episodes) if episodes is not None else n
    for i in range(count):
        ep = None if episodes is None else episodes[i]
        obs, _ = env.reset(seed=seed0 + i, options=None if ep is None else {"episode": ep})
        ctrl = controller_factory(env)
        hit, steps = False, 0
        while True:
            a = ctrl(env, obs)
            if env.cfg.mode == "joints":  # the scripted rules are Cartesian: convert with the env's own DLS step
                a = env.ee_to_joint_action(a * env.cfg.ee_step)
            obs, _, term, trunc, info = env.step(a)
            hit |= bool(info["collision"])
            steps += 1
            if term or trunc or (max_steps and steps >= max_steps):
                break
        rows.append(dict(success=bool(info["is_success"]), collision=hit, d_cm=100 * info["d_pos"], steps=steps))
    return rows
