"""[MyCobot 280 port of collision.py; see scripts/port_from_ur5.py]
UR5e configuration collision checker for planners (SPEC §7.2): its own lean MuJoCo model and data, so
planning never disturbs an environment's simulation state. Obstacles can be inflated (degraded base, §7.3)."""

from __future__ import annotations

import mujoco
import numpy as np

from mcrl import shapes
from mcrl.envs.pose_env import SCENE_XML


class CollisionChecker:
    def __init__(self, scene_xml=SCENE_XML):
        spec = mujoco.MjSpec.from_file(str(scene_xml))
        spec.compiler.discardvisual = True
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.K = sum(1 for b in range(m.nbody) if m.body(b).name.startswith("obstacle"))
        self.geom = np.array([m.geom(f"obstacle{k}").id for k in range(self.K)])
        self.mocap = np.array([m.body_mocapid[m.body(f"obstacle{k}").id] for k in range(self.K)])
        self.n_checks = 0

    def set_obstacles(self, centers, radii, mask, inflate=0.0, shape=None, size=None, quat=None):
        """Spheres by `radii`; with `shape` / `size` / `quat` (mcrl/shapes.py) boxes and cylinders in their exact form,
        every half size grown by `inflate`."""
        m, d = self.model, self.data
        if shape is None:
            shape, size, quat = shapes.default_arrays(self.K)
        for k in range(self.K):
            shapes.apply(m, d, self.geom[k], self.mocap[k], centers[k], mask[k], int(shape[k]), size[k], quat[k],
                         radii[k], inflate)

    def __call__(self, q):
        """True if configuration q touches the table, floor, an obstacle or itself."""
        d = self.data
        d.qpos[:6] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        self.n_checks += 1
        return d.ncon > 0
