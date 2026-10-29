import numpy as np
import pytest

from ur5rl.envs import rotations as rot


def random_rotations(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return rot.quat_to_mat(q)


def test_quat_mat_roundtrip():
    R = random_rotations(1000)
    q = rot.mat_to_quat(R)
    assert (q[:, 0] >= 0).all()
    np.testing.assert_allclose(rot.quat_to_mat(q), R, atol=1e-12)


def test_r6d_roundtrip():
    R = random_rotations(1000, seed=1)
    np.testing.assert_allclose(rot.r6d_to_mat(rot.mat_to_r6d(R)), R, atol=1e-12)


def test_r6d_gram_schmidt_projects_noisy_input():
    R = random_rotations(100, seed=2)
    x = rot.mat_to_r6d(R) * 3.0 + 1e-3 * np.random.default_rng(3).normal(size=(100, 6))
    M = rot.r6d_to_mat(x)
    np.testing.assert_allclose(M @ np.swapaxes(M, -1, -2), np.broadcast_to(np.eye(3), M.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(M), 1.0, atol=1e-12)


def test_geodesic_angle_properties():
    R1, R2 = random_rotations(500, 4), random_rotations(500, 5)
    np.testing.assert_allclose(rot.geodesic_angle(R1, R1), 0.0, atol=1e-6)
    np.testing.assert_allclose(rot.geodesic_angle(R1, R2), rot.geodesic_angle(R2, R1), atol=1e-12)
    flip = np.diag([1.0, -1.0, -1.0])  # 180 deg about x
    np.testing.assert_allclose(rot.geodesic_angle(R1, R1 @ flip), np.pi, atol=1e-6)


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.3, 2.0, np.pi - 1e-6, np.pi])
def test_rotvec_matches_axis_angle(angle):
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    v = rot.mat_to_rotvec(R)
    assert abs(np.linalg.norm(v) - angle) < 1e-6
    if angle > 1e-6:
        assert abs(abs(v @ axis) - angle) < 1e-6  # at pi the sign of the axis is ambiguous


def test_rotation_error_rotates_current_onto_goal():
    Rg, Rc = random_rotations(200, 6), random_rotations(200, 7)
    v = rot.rotation_error(Rg, Rc)
    np.testing.assert_allclose(np.linalg.norm(v, axis=-1), rot.geodesic_angle(Rg, Rc), atol=1e-6)
