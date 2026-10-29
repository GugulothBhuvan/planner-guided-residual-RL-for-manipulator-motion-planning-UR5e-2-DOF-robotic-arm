import mujoco
import numpy as np
import pytest

from ur5rl.envs.clearance import Clearance
from ur5rl.envs.pose_env import SCENE_XML


@pytest.fixture(scope="module")
def sim():
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    return m, mujoco.MjData(m)


def test_robot_collision_geoms(sim):
    m, _ = sim
    c = Clearance(m)
    names = [m.geom(g).name for g in c.geom_ids]
    assert len(names) == 11  # 9 arm capsules + 2 gripper capsules
    assert {"gripper_palm", "gripper_fingers"} <= set(names)


def test_analytic_matches_mj_geom_distance(sim):
    """1k random configurations x 1 random sphere: every capsule-sphere pair within 1e-6 (links + gripper)."""
    m, d = sim
    c = Clearance(m)
    g_ob = m.geom("obstacle0").id
    rng = np.random.default_rng(0)
    fromto = np.zeros(6)
    worst = 0.0
    for _ in range(1000):
        d.qpos[:6] = rng.uniform(m.jnt_range[:, 0], m.jnt_range[:, 1])
        center = rng.uniform([-0.9, -0.9, 0.0], [0.9, 0.9, 1.0])
        radius = rng.uniform(0.02, 0.12)
        m.geom_size[g_ob, 0] = m.geom_rbound[g_ob] = radius
        d.mocap_pos[m.body_mocapid[m.body("obstacle0").id]] = center
        mujoco.mj_kinematics(m, d)
        ours = c.capsule_distances(d, center[None], np.array([radius]))[:, 0]
        ref = np.array([mujoco.mj_geomDistance(m, d, int(g), g_ob, 10.0, fromto) for g in c.geom_ids])
        worst = max(worst, np.abs(ours - ref).max())
    assert worst < 1e-6, worst


def test_gripper_capsules_enclose_mesh(sim):
    """Every vertex of the 2F-85 visual mesh lies inside at least one gripper capsule."""
    m, d = sim
    d.qpos[:6] = 0
    mujoco.mj_kinematics(m, d)
    caps = [m.geom(n).id for n in ("gripper_palm", "gripper_fingers")]
    verts = []
    for g in range(m.ngeom):
        if m.geom(g).name.startswith("gripper_vis"):
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            verts.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
    verts = np.concatenate(verts)
    assert len(verts) > 10000
    inside = np.zeros(len(verts), bool)
    for g in caps:
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
        p0 = d.geom_xpos[g] - m.geom_size[g, 1] * axis
        seg = 2 * m.geom_size[g, 1] * axis
        t = np.clip((verts - p0) @ seg / max(seg @ seg, 1e-12), 0, 1)
        dist = np.linalg.norm(verts - (p0 + t[:, None] * seg), axis=1)
        inside |= dist <= m.geom_size[g, 0] + 1e-9
    assert inside.all(), f"{(~inside).sum()} vertices outside"


def test_body_min_reduces_per_body(sim):
    m, d = sim
    c = Clearance(m)
    d.qpos[:6] = 0
    mujoco.mj_kinematics(m, d)
    centers = np.array([[0.3, 0.3, 0.5], [-0.5, -0.2, 0.2], [0, 0, 0], [0, 0, 0]])
    radii = np.full(4, 0.05)
    mask = np.array([True, True, False, False])
    per_cap = c.capsule_distances(d, centers[mask], radii[mask]).min(1)
    per_body = c.body_min(d, centers, radii, mask, clip=10.0)
    assert per_body.shape == (7,)
    for b in range(7):
        sel = slice(c.body_start[b], c.body_start[b + 1] if b < 6 else None)
        assert per_body[b] == pytest.approx(per_cap[sel].min())
