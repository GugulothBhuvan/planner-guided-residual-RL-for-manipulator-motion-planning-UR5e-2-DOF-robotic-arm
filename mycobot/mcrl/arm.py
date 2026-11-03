"""Arm interfaces for the hardware loop: RealArm (pymycobot, measured comms rules) and FakeArm (virtual-time simulation
of the measured behaviour: first-order lag, deadband, read noise) used for dry runs.

Both expose  q (rad, last reading),  tick(q_target) -> q_measured,  stop().  Hardware rules (SPEC M1):
a command must be followed by >= ~10 ms of quiet; send -> 30 ms -> read gives a clean ~49 ms cycle.
"""
from __future__ import annotations

import time

import numpy as np

from mcrl import robot

LIMITS = robot.LIMITS_RAD
MAX_LAG = np.deg2rad(5.0)  # abort if the arm is further than this from the commanded target
MAX_STEP = np.deg2rad(1.0)  # abort if one tick would command a larger joint step than this


class SafetyAbort(RuntimeError):
    pass


def check_command(q_prev_target, q_target, q_meas, max_step=MAX_STEP):
    if np.any(q_target < LIMITS[:, 0] - 1e-9) or np.any(q_target > LIMITS[:, 1] + 1e-9):
        raise SafetyAbort(f"target outside joint limits: {np.rad2deg(q_target).round(1)}")
    if q_prev_target is not None and np.abs(q_target - q_prev_target).max() > max_step + 1e-9:
        raise SafetyAbort(f"commanded step {np.rad2deg(np.abs(q_target - q_prev_target).max()):.2f} deg > {np.rad2deg(max_step):.2f} deg")
    if np.abs(q_target - q_meas).max() > MAX_LAG:
        raise SafetyAbort(f"target {np.rad2deg(np.abs(q_target - q_meas).max()):.1f} deg from the arm (limit 5)")


class FakeArm:
    """Virtual-time arm: each tick advances robot.CONTROL_DT seconds."""

    def __init__(self, q0, tau=0.15, deadband_deg=0.7, noise_deg=0.1, seed=0):
        self.q_true = np.asarray(q0, float).copy()
        self.q = self.q_true.copy()
        self.tau, self.db, self.noise = tau, np.deg2rad(deadband_deg), np.deg2rad(noise_deg)
        self.rng = np.random.default_rng(seed)
        self._prev = None
        self.n_ticks = 0

    def tick(self, q_target):
        q_target = np.asarray(q_target, float)
        check_command(self._prev, q_target, self.q)
        self._prev = q_target.copy()
        err = q_target - self.q_true
        move = np.where(np.abs(err) > self.db, err * (1 - np.exp(-robot.CONTROL_DT / self.tau)), 0.0)
        self.q_true = self.q_true + move
        self.q = self.q_true + self.rng.normal(0, self.noise, 6)
        self.n_ticks += 1
        return self.q.copy()

    def stop(self):
        pass


class RealArm:
    def __init__(self, port="COM13", speed=20, send_every=1):
        from pymycobot import MyCobot280  # installed in the hardware venv (mycobot/.venv) or the main venv

        self.mc = MyCobot280(port, 115200)
        time.sleep(1.0)
        self.speed = int(speed)
        assert 1 <= self.speed <= 30
        self.send_every = int(send_every)  # 1 = a new target every control tick (50 ms); 2 = every 100 ms, extrapolated
        assert self.send_every >= 1
        self._prev = None  # last commanded (sent) target
        self._ptarget = None  # nominal target of the previous tick
        self._n = 0
        self.q = self._read()

    def _read(self):
        for _ in range(6):
            a = self.mc.get_angles()
            if isinstance(a, list) and len(a) >= 6:
                return np.deg2rad(np.array(a[:6], float))
            time.sleep(0.1)
        raise SafetyAbort("get_angles failed 6x")

    def final_coords(self):
        """The arm's own tool pose [x, y, z (mm), rx, ry, rz (deg)] (get_coords), for cross-checking the model's FK."""
        for _ in range(6):
            c = self.mc.get_coords()
            if isinstance(c, list) and len(c) >= 6:
                return [float(v) for v in c[:6]]
            time.sleep(0.1)
        return None

    def tick(self, q_target):
        q_target = np.asarray(q_target, float)
        t0 = time.perf_counter()
        send = self._n % self.send_every == 0
        if send:
            ahead = 0.0 if self._ptarget is None else (q_target - self._ptarget) * (self.send_every - 1)
            cmd = np.clip(q_target + ahead, LIMITS[:, 0], LIMITS[:, 1])
            check_command(self._prev, cmd, self.q, MAX_STEP * self.send_every)
            self._prev = cmd.copy()
            self.mc.send_angles(np.rad2deg(cmd).tolist(), self.speed)
        self._ptarget = q_target.copy()
        self._n += 1
        time.sleep(0.030)  # the controller drops a command that follows a send within ~10 ms
        self.q = self._read()
        wait = robot.CONTROL_DT - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)
        return self.q.copy()

    def stop(self):
        try:
            self.mc.stop()
        except Exception:
            pass


class TrimArm:
    """Final-approach correction for a servo that stops short of a static target (SPEC M7b: J2 under load stopped 1.84 deg
    short, more than the unloaded deadband). While the nominal target holds still, the commanded target is nudged by the
    measured shortfall (integral action); while it moves, the nudge decays to zero so the path itself is untouched.
    Limits: |trim| <= MAX_DEG per joint, change <= RATE_DEG per tick (the 1 deg step check still applies to the sum)."""
    GAIN, MAX_DEG, RATE_DEG, STILL_TICKS, STILL_DEG = 0.2, 2.0, 0.1, 8, 0.15

    def __init__(self, arm):
        self.arm = arm
        self.trim = np.zeros(6)
        self._last, self._still = None, 0

    @property
    def q(self):
        return self.arm.q

    def final_coords(self):
        return self.arm.final_coords()

    def tick(self, q_target):
        q_target = np.asarray(q_target, float)
        moved = None if self._last is None else np.abs(q_target - self._last).max()
        self._still = self._still + 1 if moved is not None and moved < np.deg2rad(self.STILL_DEG) else 0
        self._last = q_target.copy()
        rate = np.deg2rad(self.RATE_DEG)
        if self._still >= self.STILL_TICKS:
            self.trim = np.clip(self.trim + np.clip(self.GAIN * (q_target - self.arm.q), -rate, rate),
                                -np.deg2rad(self.MAX_DEG), np.deg2rad(self.MAX_DEG))
        else:
            self.trim = self.trim - np.clip(self.trim, -rate, rate)
        return self.arm.tick(np.clip(q_target + self.trim, LIMITS[:, 0], LIMITS[:, 1]))

    def stop(self):
        self.arm.stop()
