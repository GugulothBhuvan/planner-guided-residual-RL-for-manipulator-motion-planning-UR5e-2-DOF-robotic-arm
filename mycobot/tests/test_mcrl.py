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


def _dist_to_shape(p, kind, c, size, R):
    """Distance from points p (n,3) to a box / vertical cylinder at c with local rotation R (0 inside)."""
    local = (p - c) @ R
    if kind == 1:
        return np.linalg.norm(np.maximum(np.abs(local) - size, 0.0), axis=-1)
    radial = np.maximum(np.linalg.norm(local[:, :2], axis=-1) - size[0], 0.0)
    axial = np.maximum(np.abs(local[:, 2]) - size[1], 0.0)
    return np.hypot(radial, axial)


@pytest.mark.parametrize("kind", [1, 2])
def test_box_and_cylinder_collide_by_exact_shape(env, kind):
    """MuJoCo contacts against a rotated box / cylinder obstacle agree with an independent point-sampled distance
    between the arm's capsules and the shape (outside a 3 mm band), and differ from the enclosing sphere's verdict."""
    from mcrl import shapes
    from ur5rl.envs import rotations as rot
    rng = np.random.default_rng(7 + kind)
    env.reset(seed=0)
    agree = n_hit = n_miss = sphere_only = 0
    for _ in range(400):
        q = rng.uniform(env.q_lo, env.q_hi)
        env._set_obstacles(np.zeros((env.K, 3)), np.full(env.K, 0.05), np.zeros(env.K, bool))
        if env._in_collision(q) or env._tcp_pose(q)[0][2] <= 0.02:
            continue
        c = env._tcp_pose(q)[0] + rng.uniform(-0.08, 0.08, 3)
        size = np.array([0.012, 0.02, 0.03]) if kind == 1 else np.array([0.015, 0.035, 0.0])
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        R = rot.quat_to_mat(quat)
        rb = shapes.bounding_radius(kind, size)
        shape, sz, qt = (np.zeros(env.K, np.int64), np.zeros((env.K, 3)), np.tile([1.0, 0, 0, 0], (env.K, 1)))
        shape[0], sz[0], qt[0] = kind, size, quat
        centers, radii, mask = np.zeros((env.K, 3)), np.full(env.K, 0.05), np.zeros(env.K, bool)
        centers[0], radii[0], mask[0] = c, rb, True
        env._set_obstacles(centers, radii, mask, shape, sz, qt)
        hit = env._in_collision(q)
        p0, p1 = env.clearance.segments(env.data)
        t = np.linspace(0, 1, 25)[None, :, None]
        pts = (p0[:, None] + t * (p1 - p0)[:, None]).reshape(-1, 3)
        d = (_dist_to_shape(pts, kind, c, size, R).reshape(len(p0), -1) - env.clearance.radius[:, None]).min()
        if abs(d) < 0.003:
            continue
        agree += int(hit == (d < 0))
        n_hit += int(d < 0)
        n_miss += int(d >= 0)
        sphere_only += int((env.clearance.capsule_distances(env.data, c[None], np.array([rb])).min() < 0) and not hit)
    assert n_hit >= 10 and n_miss >= 10
    assert agree == n_hit + n_miss
    assert sphere_only > 0  # the enclosing sphere would have reported collisions the exact shape does not have
    env._set_obstacles(np.zeros((env.K, 3)), np.full(env.K, 0.05), np.zeros(env.K, bool))


def test_shape_episode_roundtrip_and_sphere_default(env):
    env.reset(seed=1)
    ep = env.episode()
    assert "shape" not in ep  # spheres only: the episode format is unchanged
    shape, sz, qt = (np.array([1, 2, 0, 0]), np.array([[.02, .02, .02], [.015, .03, 0], [0, 0, 0], [0, 0, 0]]),
                     np.tile([1.0, 0, 0, 0], (4, 1)))
    ep = dict(ep, centers=np.array([[.1, 0, .2], [.15, .05, .2], [0, 0, 0], [0, 0, 0]]), radii=np.array([.035, .033, .05, .05]),
              mask=np.array([True, True, False, False]), shape=shape, size=sz, quat=qt)
    env.reset(seed=1, options={"episode": ep})
    again = env.episode()
    assert (again["shape"] == shape).all() and np.allclose(again["size"], sz)
    env.reset(seed=1)  # a plain reset restores spheres
    assert not env.ob_shape.any()
