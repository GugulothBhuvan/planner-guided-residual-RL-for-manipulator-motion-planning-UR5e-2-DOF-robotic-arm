import numpy as np

from arm2dof.cspace.components import components, connected
from arm2dof.cspace.distance_field import distance_field
from arm2dof.cspace.grid import configs, grid
from arm2dof.envs.geometry import LINK_RADIUS, link_clearance
from arm2dof.planners.ik import joints

SCENE = (np.array([[0.45, 0.2], [-0.3, -0.5], [0.1, 0.7]]), np.array([0.1, 0.12, 0.08]))


def test_grid_shape():
    assert grid(*SCENE).shape == (360, 324)


def test_occupancy_agrees_with_direct_check():
    """Independent check: sample points along both links instead of the closed-form segment distance."""
    occ = grid(*SCENE)
    q = configs()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, [360, 324], (3000, 2))
    t = np.linspace(0, 1, 2001)[:, None]
    centers, radii = SCENE
    checked = 0
    for i, j in idx:
        qq = q[i, j]
        clear = link_clearance(qq, centers, radii).min()
        if abs(clear) < 1e-3:
            continue  # boundary cells: sampling resolution, not a disagreement
        elbow, ee = joints(qq)
        pts = np.concatenate([t * elbow, elbow + t * (ee - elbow)])
        d = np.linalg.norm(pts[:, None, :] - centers[None], axis=-1) - LINK_RADIUS - radii
        assert occ[i, j] == (d.min() < 0)
        checked += 1
    assert checked > 2900


def test_wraparound_connectivity():
    occ = np.ones((360, 324), bool)
    occ[:10, 100:110] = False  # free strip just above q1 = -pi
    occ[350:, 100:110] = False  # free strip just below q1 = +pi
    labels = components(occ)
    assert connected(occ, (0, 105), (359, 105), labels)  # joined across the seam
    occ2 = occ.copy()
    occ2[355:, :] = True  # cut the strip before the seam
    occ2[:5, :] = True
    assert not connected(occ2, (6, 105), (352, 105))


def test_distance_field_wraps():
    occ = np.zeros((360, 324), bool)
    occ[0, 150] = True
    d = distance_field(occ)
    assert d[0, 150] == 0
    np.testing.assert_allclose(d[359, 150], d[1, 150])  # neighbours across the seam are equally close
