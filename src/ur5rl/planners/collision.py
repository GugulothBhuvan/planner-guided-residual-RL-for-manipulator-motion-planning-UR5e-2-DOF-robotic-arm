"""UR5e configuration collision checker for planners (SPEC §7.2): its own lean MuJoCo model and data, so
planning never disturbs an environment's simulation state. Obstacles can be inflated (degraded base, §7.3)."""

from __future__ import annotations

import mujoco
import numpy as np

from ur5rl.envs.pose_env import SCENE_XML


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

    def set_obstacles(self, centers, radii, mask, inflate=0.0):
        m, d = self.model, self.data
        for k in range(self.K):
            r = float(radii[k]) + inflate
            d.mocap_pos[self.mocap[k]] = centers[k] if mask[k] else (0.0, 0.0, -5.0)
            m.geom_size[self.geom[k], 0] = m.geom_rbound[self.geom[k]] = r
            m.geom_aabb[self.geom[k], 3:] = r

    def __call__(self, q):
        """True if configuration q touches the table, floor, an obstacle or itself."""
        d = self.data
        d.qpos[:6] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        self.n_checks += 1
        return d.ncon > 0
