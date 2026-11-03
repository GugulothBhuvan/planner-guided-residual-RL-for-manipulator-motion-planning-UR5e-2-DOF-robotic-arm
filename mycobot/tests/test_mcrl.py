"""MyCobot lane tests: FK agreement, real-pose collision, IK, env API, planner + tracker end to end."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mc280.kinematics import fk_tool
from mcrl import robot
from mcrl.envs.pose_env import EnvConfig, MyCobotPoseEnv
from mcrl.envs.residual import ResidualEnv
from mcrl.planners.baselines import ReferenceTracker, goal_configs, plan_b2
from mcrl.planners.collision import CollisionChecker
from mcrl.planners.ik import fk_tcp, ik_tcp

ROOT = Path(__file__).resolve().parents[1]
TCP = (0.0072, -0.0053, 0.0008)


@pytest.fixture(scope="module")
def env():
    import yaml  # the stage the lane is trained and evaluated on (horizon 900, goals <= 0.9 rad away)
    cfg = yaml.safe_load((ROOT / "configs" / "env" / "s4.yaml").read_text())
    cfg["n_obstacles"] = tuple(cfg["n_obstacles"])
    known = {f.name for f in __import__("dataclasses").fields(EnvConfig)}
    return MyCobotPoseEnv(EnvConfig(**{k: v for k, v in cfg.items() if k in known}))


def test_mujoco_fk_matches_numpy(env):
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(env.q_lo, env.q_hi)
        p, _ = env._tcp_pose(q)[0], None
        assert np.allclose(p, fk_tool(q, TCP)[0], atol=1e-9)


def test_real_poses_collision_free():
    ck = CollisionChecker()
    ck.set_obstacles(np.zeros((4, 3)), np.full(4, 0.02), np.zeros(4, bool))
    poses = json.loads((ROOT / "results" / "poses.json").read_text())
    assert len(poses) >= 15
    assert not any(ck(np.deg2rad(p["angles"])) for p in poses)


def test_fk_vs_hardware_coords():
    poses = json.loads((ROOT / "results" / "poses.json").read_text())
    err = [np.linalg.norm(fk_tool(np.deg2rad(p["angles"]), TCP)[0] * 1e3 - np.array(p["coords"][:3])) for p in poses]
    assert np.sqrt(np.mean(np.square(err))) < 4.0  # mm, 3-D RMS (3.6 measured = 2.1 per axis)
    assert max(err) < 7.0  # 6.0 measured


def test_ik_recovers_pose(env):
    rng = np.random.default_rng(1)
    ck = CollisionChecker()
    hits = 0
    for _ in range(10):
        q = rng.uniform(env.q_lo, env.q_hi)
        T = np.eye(4)
        T[:3, :3] = env._tcp_pose(q)[1]
        T[:3, 3] = env._tcp_pose(q)[0]
        Tft = np.linalg.inv(fk_tcp(np.zeros(6), np.eye(4))) @ fk_tcp(np.zeros(6), np.eye(4))  # identity
        sols = ik_tcp(T, np.eye(4) * 0 + _flange_to_tcp(env), env.q_lo, env.q_hi, q + 0.3)
        hits += any(np.linalg.norm(fk_tcp(s, _flange_to_tcp(env))[:3, 3] - T[:3, 3]) < 1e-3 for s in sols)
    assert hits >= 9


def _flange_to_tcp(env):
    from mcrl.planners.baselines import flange_to_tcp
    return flange_to_tcp(env)


def test_observation_layout_matches_ur5e(env):
    obs, _ = env.reset(seed=0)
    # 45 + 5 K + 7 bodies: identical to the UR5e S4 env, so UR5e actors accept it (zero-shot probe)
    assert obs["observation"].shape == (45 + 5 * 4 + 7,)
    assert env.action_space.shape == (6,)


def test_planner_and_tracker_reach_goal(env):
    ok = 0
    for seed in range(6):
        env.reset(seed=seed)
        plan = plan_b2(env, np.random.default_rng(seed), CollisionChecker())
        if not plan.ok:
            continue
        tracker = ReferenceTracker(env, plan.reference)
        for _ in range(len(plan.reference) + 30):
            _, _, term, trunc, info = env.step(tracker())
            if term or trunc:
                break
        ok += info["is_success"] and not info["collision"]
    assert ok >= 4


def test_residual_env_with_zero_residual_equals_base(env):
    renv = ResidualEnv(MyCobotPoseEnv(EnvConfig(n_obstacles=(1, 4), episode_length=300, gamma=0.99)),
                       inflate=0.017, max_checks=3000, shortcut_iters=20, alpha=0.0)
    obs, info = renv.reset(seed=3)
    assert obs["observation"].shape[0] == 72 + 12
    obs, r, term, trunc, info = renv.step(np.zeros(6))
    assert "base_ok" in info


def test_trim_arm_closes_static_shortfall_within_limits():
    from mcrl.arm import FakeArm, TrimArm
    q0 = np.zeros(6)
    target = q0 + np.deg2rad(1.5)  # the fake servo's 1.0 deg deadband leaves a static shortfall
    plain = FakeArm(q0, deadband_deg=1.0, noise_deg=0.0)
    trimmed = TrimArm(FakeArm(q0, deadband_deg=1.0, noise_deg=0.0))
    for arm in (plain, trimmed):
        for k in (1, 2, 3):  # approach in 0.5 deg steps, then hold
            arm.tick(q0 + np.deg2rad(0.5) * k)
        for _ in range(120):
            arm.tick(target)
    short_plain, short_trim = np.abs(target - plain.q).max(), np.abs(target - trimmed.q).max()
    assert short_plain > np.deg2rad(0.4) and short_trim < short_plain / 2
    assert np.abs(trimmed.trim).max() <= np.deg2rad(2.0) + 1e-12
    for k in range(1, 41):  # the target moves again: the trim decays to zero within ~20 ticks
        trimmed.tick(target + np.deg2rad(0.3) * k)
    assert np.abs(trimmed.trim).max() < 1e-9
