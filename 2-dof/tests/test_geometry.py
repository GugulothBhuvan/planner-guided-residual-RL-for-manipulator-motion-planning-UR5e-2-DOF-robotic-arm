import numpy as np

from arm2dof.envs.geometry import LINK_RADIUS, in_collision, link_clearance, segment_point_distance
from arm2dof.planners.ik import Q_HI, Q_LO, joints


def test_segment_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 400_001)
    for _ in range(60):
        a, b, p = rng.uniform(-1, 1, (3, 2))
        pts = a + t[:, None] * (b - a)
        brute = np.linalg.norm(pts - p, axis=1).min()
        assert abs(segment_point_distance(a, b, p) - brute) < 1e-6


def test_link_clearance_matches_brute_force():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 100_001)[:, None]
    for _ in range(100):
        q = rng.uniform(Q_LO, Q_HI)
        centers = rng.uniform(-0.9, 0.9, (3, 2))
        radii = rng.uniform(0.05, 0.15, 3)
        elbow, ee = joints(q)
        ours = link_clearance(q, centers, radii)
        for li, (s, e) in enumerate(((np.zeros(2), elbow), (elbow, ee))):
            pts = s + t * (e - s)
            for k in range(3):
                brute = np.linalg.norm(pts - centers[k], axis=1).min() - LINK_RADIUS - radii[k]
                assert abs(ours[li, k] - brute) < 1e-6


def test_in_collision_batched_and_empty():
    q = np.zeros((5, 7, 2))
    assert in_collision(q, np.zeros((0, 2)), np.zeros(0)).shape == (5, 7)
    assert not in_collision(q, np.zeros((0, 2)), np.zeros(0)).any()
    assert in_collision(np.zeros(2), np.array([[0.3, 0.0]]), np.array([0.05]))  # circle on link 1
