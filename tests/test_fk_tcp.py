import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from ur5rl.envs.pose_env import HOME_Q, SCENE_XML
from ur5rl.planners.ik import fk_flange, fk_tcp

META = json.loads(Path(SCENE_XML).with_name("scene_ur5e.meta.json").read_text())


@pytest.fixture(scope="module")
def sim():
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    return m, mujoco.MjData(m)


def site_pose(m, d, name, q):
    d.qpos[:6] = q
    mujoco.mj_kinematics(m, d)
    T = np.eye(4)
    T[:3, :3] = d.site(name).xmat.reshape(3, 3)
    T[:3, 3] = d.site(name).xpos
    return T


def flange_to_tcp(m, d):
    """Fixed flange->TCP transform, read from the model (never hard-coded)."""
    q0 = np.zeros(6)
    return np.linalg.inv(site_pose(m, d, "attachment_site", q0)) @ site_pose(m, d, "tcp", q0)


def test_tcp_site_is_pad_midpoint(sim):
    m, d = sim
    T = flange_to_tcp(m, d)
    np.testing.assert_allclose(T[:3, 3], META["tcp_in_gripper"], atol=1e-9)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-9)  # TCP axes = gripper frame (z = approach)
    assert 0.12 < T[2, 3] < 0.16  # sanity: ~14 cm from the flange


@pytest.mark.parametrize("q", [np.zeros(6), HOME_Q], ids=["zero", "home"])
def test_fk_known_poses(sim, q):
    m, d = sim
    np.testing.assert_allclose(fk_flange(q), site_pose(m, d, "attachment_site", q), atol=1e-6)
    np.testing.assert_allclose(fk_tcp(q, flange_to_tcp(m, d)), site_pose(m, d, "tcp", q), atol=1e-6)


def test_fk_zero_pose_value():
    # Independent hand computation for q = 0 (arm stretched along -x): x = a2 + a3, y = -(d4 + d6), z = d1 - d5
    T = fk_flange(np.zeros(6))
    np.testing.assert_allclose(T[:3, 3], [-0.425 - 0.392, -(0.134 + 0.1), 0.163 - 0.1], atol=1e-12)


def test_fk_random_configs(sim):
    m, d = sim
    T_ft = flange_to_tcp(m, d)
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = rng.uniform(m.jnt_range[:, 0], m.jnt_range[:, 1])
        np.testing.assert_allclose(fk_tcp(q, T_ft), site_pose(m, d, "tcp", q), atol=1e-6)
