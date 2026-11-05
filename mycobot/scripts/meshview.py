"""Visual-only mesh overlay for the MyCobot model (same technique as render_mesh_compare.py, importable).

A second MuJoCo model with the official M5 shell meshes (assets/m5_meshes, from mycobot_ros) is built next to the physics
model; every frame the state (qpos, mocap, obstacle geoms) is copied across before drawing. The physics model is never touched.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

HP = np.pi / 2
WHITE = [0.93, 0.93, 0.95, 1]
DARK = [0.16, 0.16, 0.18, 1]
# (body, mesh, pos, rpy, rgba) from the official M5 URDF visual origins
PARTS = [
    ("base", "joint1", (0, 0, 0), (0, 0, -HP), WHITE),
    ("base", "G_base", (0, 0, -0.03), (0, 0, HP), DARK),
    ("link1", "joint2", (0, 0, -0.06096), (0, 0, -HP), WHITE),
    ("link2", "joint3", (0, 0, 0.03256), (0, -HP, 0), WHITE),
    ("link3", "joint4", (0, 0, 0.03056), (0, -HP, 0), WHITE),
    ("link4", "joint5", (0, 0, -0.03356), (-HP, 0, 0), WHITE),
    ("link5", "joint6", (0, 0, -0.038), (0, 0, 0), WHITE),
    ("link6", "joint7", (0, 0, -0.012), (0, 0, 0), DARK),
]


def quat_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def build_visual_model(scene_xml, mesh_dir):
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    spec.meshdir = str(mesh_dir)
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
    spec.visual.quality.shadowsize = 8192
    for name in sorted({p[1] for p in PARTS}):
        spec.add_mesh(name=name, file=f"{name}.stl")
    for body, mesh, pos, rpy, rgba in PARTS:
        g = spec.body(body).add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mesh
        g.pos = pos
        g.quat = quat_rpy(*rpy)
        g.contype = 0
        g.conaffinity = 0
        g.group = 2
        g.rgba = rgba
        g.mass = 0.0
    return spec.compile()


class VisRenderer:
    """Quacks like mujoco.Renderer for the comparison scripts: update_scene(env_data, cam, scene_option) draws the mesh model."""

    def __init__(self, base, h, w, mesh_dir):
        self.em, self.ed = base.model, base.data
        if not hasattr(base, "_vis_cache"):
            from mcrl.envs import pose_env
            base._vis_cache = build_visual_model(pose_env.SCENE_XML, mesh_dir)
        self.vm = base._vis_cache
        self.vm.vis.global_.offheight = max(self.vm.vis.global_.offheight, h)
        self.vm.vis.global_.offwidth = max(self.vm.vis.global_.offwidth, w)
        self.vd = mujoco.MjData(self.vm)
        self.r = mujoco.Renderer(self.vm, h, w)
        self.common = []
        for i in range(self.em.ngeom):
            nm = mujoco.mj_id2name(self.em, mujoco.mjtObj.mjOBJ_GEOM, i)
            if nm:
                j = mujoco.mj_name2id(self.vm, mujoco.mjtObj.mjOBJ_GEOM, nm)
                if j >= 0:
                    self.common.append((i, j))

    @property
    def scene(self):
        return self.r.scene

    def update_scene(self, data, cam, scene_option=None):
        vd, vm, em = self.vd, self.vm, self.em
        vd.qpos[:] = data.qpos
        if vm.nmocap:
            vd.mocap_pos[:] = data.mocap_pos
            vd.mocap_quat[:] = data.mocap_quat
        for i, j in self.common:  # obstacle geoms change type / size / colour at run time
            vm.geom_type[j] = em.geom_type[i]
            vm.geom_size[j] = em.geom_size[i]
            vm.geom_rgba[j] = em.geom_rgba[i]
        mujoco.mj_forward(vm, vd)
        self.r.update_scene(vd, cam, scene_option=scene_option)

    def render(self):
        return self.r.render()
