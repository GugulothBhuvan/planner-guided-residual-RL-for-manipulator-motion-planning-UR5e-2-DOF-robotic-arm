"""MyCobot 280 constants for the sim lane (SPEC M3). Every value here is derived from a measurement in results/."""
import numpy as np

from mc280.kinematics import LIMITS_DEG

# Control: 20 Hz (send, 30 ms gap, read = ~49 ms cycle; results/timing.json, gap_sweep.json). 25 substeps of 2 ms.
FRAME_SKIP = 25
CONTROL_DT = FRAME_SKIP * 0.002
# Step size: reliable tracking on hardware up to ~0.5 deg per tick = 10 deg/s (results/track_test.json);
# 1 deg/tick already gave 270 ms comm stalls. 0.5 deg = 0.0087 rad.
DQ_MAX = np.deg2rad(0.5)
TARGET_LEAD_MAX = 0.055  # rad: UR5e's 0.25 rad x (DQ_MAX / 0.04)
V_CAP = DQ_MAX / CONTROL_DT  # rad/s the arm realises at |a| = 1  (0.1745)
V_MAX, A_MAX = 0.92 * V_CAP, 0.5  # B2 time scaling: same 92% of V_CAP as the UR5e (0.9 of 0.98)
HOME_Q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # upright, J6 about -57 deg like the arm's resting pose
SHOULDER = np.array([0.0, 0.0, 0.13156])  # J2 axis position
REACH = 0.36  # m, bound for the analytic per-step reward minimum (J2 -> tool at full stretch + margin)
LIMITS_RAD = np.deg2rad(LIMITS_DEG)
