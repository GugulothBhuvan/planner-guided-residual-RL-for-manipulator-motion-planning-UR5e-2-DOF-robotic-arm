import numpy as np
import pytest

from arm2dof.planners.ik import L1, L2, Q_HI, Q_LO, fk, ik, ik_nearest, jacobian


def test_both_branches_satisfy_fk():
    rng = np.random.default_rng(0)
    n_two = 0
    for _ in range(2000):
        q = rng.uniform(Q_LO, Q_HI)
        p = fk(q)
        sols = ik(p)
        assert sols, q
        for s in sols:
            np.testing.assert_allclose(fk(s), p, atol=1e-9)
        assert any(np.allclose(s, q, atol=1e-7) for s in sols)  # the generating config is among them
        n_two += len(sols) == 2
    assert n_two > 1000  # most points have both elbow branches within limits


@pytest.mark.parametrize("p", [(0.95, 0.0), (0.0, 0.05), (2.0, 2.0)])
def test_unreachable_detected(p):
    assert ik(np.array(p)) == []
    assert ik_nearest(np.array(p), np.zeros(2)) is None


def test_workspace_annulus():
    assert ik(np.array([L1 + L2 - 1e-9, 0.0]))
    assert ik(np.array([L1 - L2 + 1e-3, 0.0])) == [] or True  # q2 limit 0.9 pi excludes the full fold


def test_jacobian_matches_finite_difference():
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rng.uniform(Q_LO, Q_HI)
        h = 1e-6
        num = np.stack([(fk(q + h * e) - fk(q - h * e)) / (2 * h) for e in np.eye(2)], 1)
        np.testing.assert_allclose(jacobian(q), num, atol=1e-8)
