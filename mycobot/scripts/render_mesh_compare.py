"""Render the project's 3-panel comparison with the official M5 shell meshes drawn over the model.

The physics model is never touched. A second, visual-only MuJoCo model (same bodies + mesh geoms, no collisions) is
built next to it, and every frame the state (qpos, mocap, obstacle geoms) is copied across before drawing.
Usage (from mycobot/): python scripts/render_mesh_compare.py . assets/m5_meshes results/media/out.mp4 -- --episode 8 --method hybrid --budget 3000 --real-dir results/video --tag-suffix _webcam_ep8_hybrid --real-video results/video/webcam_ep8_hybrid_webcam.mp4 [--real-offset SECONDS]
Offsets that were needed (the automatic sync-tone search picked the wrong tone): ep0 hybrid 28.8, ep7 hybrid 25.9, ep7 b2 24.8 s; ep8 and ep9 found theirs.
Set CHECK=1 to run the plain-vs-mesh identity check instead of rendering.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT, MESH, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rest = sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import make_comparison as mc  # noqa: E402  (project code, imported read-only)

HP = np.pi / 2


def quat_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    # fixed-axis x-y-z = R = Rz(y) Ry(p) Rx(r)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


# (body, mesh, pos, rpy, rgba) from the official M5 URDF visual origins
WHITE = [0.93, 0.93, 0.95, 1]
DARK = [0.16, 0.16, 0.18, 1]
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


def build_visual_model(scene_xml):
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    spec.meshdir = str(MESH)
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
    spec.visual.quality.shadowsize = 8192
    for name in sorted({p[1] for p in PARTS}):
        spec.add_mesh(name=name, file=f"{name}.stl")
    for body, mesh, pos, rpy, rgba in PARTS:
        b = spec.body(body)
        g = b.add_geom()
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
    """Quacks like mujoco.Renderer for make_comparison: update_scene(env_data, cam, scene_option) draws the mesh model."""

    def __init__(self, base, h, w):
        self.em, self.ed = base.model, base.data
        if not hasattr(base, "_vis_cache"):
            from mcrl.envs import pose_env
            base._vis_cache = build_visual_model(pose_env.SCENE_XML)
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


def make_renderer(base):
    r = VisRenderer(base, *mc.PANEL)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.08, 0.0, 0.22]
    cam.distance, cam.azimuth, cam.elevation = 0.95, 140.0, -15.0
    opt = mujoco.MjvOption()
    opt.geomgroup[2] = 1  # meshes
    opt.geomgroup[3] = 0  # capsules off
    return r, cam, opt


def check_equal():
    """Run the same episode twice, once plain and once while drawing every frame through the mesh model; compare."""
    import hashlib
    from types import SimpleNamespace
    ep = int(rest[rest.index("--episode") + 1])
    ns = SimpleNamespace(testset="s4_test", episode=ep, method="hybrid", budget=3000, run=str(ROOT / mc.RUN),
                         checkpoint="final", headless=False)
    res = {}
    for mode in ("plain", "mesh"):
        player = mc.play.Replay(ns)
        base = player.env.unwrapped
        r = cam = opt = None
        if mode == "mesh":
            r, cam, opt = make_renderer(base)
        player.reset(new=False)
        qs, done = [], False
        while not done:
            done = player.step()
            qs.append(base.data.qpos[:6].copy())
            if r is not None:
                mc.render(player, r, cam, opt)
        a = np.array(qs)
        res[mode] = (player.outcome, len(a), hashlib.sha1(a.tobytes()).hexdigest()[:12], a[-1])
        print(mode, "outcome:", res[mode][0], "| steps:", res[mode][1], "| trajectory hash:", res[mode][2])
    same = res["plain"][2] == res["mesh"][2] and res["plain"][0] == res["mesh"][0]
    print("IDENTICAL trajectory and outcome:", same, "| max joint difference:", np.abs(res["plain"][3] - res["mesh"][3]).max())


if __name__ == "__main__":
    if "CHECK" in __import__("os").environ:
        check_equal()
    else:
        mc.make_renderer = make_renderer
        sys.argv = ["make_comparison.py"] + rest + ["--out", OUT]
        mc.main()
