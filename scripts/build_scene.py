"""Build assets/scene_ur5e.xml: menagerie UR5e + rigid Robotiq 2F-85 + table, goal, obstacle slots.

The 2F-85 linkage is settled at a fixed opening and frozen into ONE rigid body (SPEC §6.1):
no joints, actuators or equality constraints. Its visual meshes are kept for rendering and its
collision geometry is exactly 2 capsules fitted to enclose every visual-mesh vertex.

Run from the repo root:  python scripts/build_scene.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MENAGERIE = ROOT / "assets" / "menagerie"
UR5E_XML = MENAGERIE / "universal_robots_ur5e" / "ur5e.xml"
GRIPPER_XML = MENAGERIE / "robotiq_2f85" / "2f85.xml"
OUT_XML = ROOT / "assets" / "scene_ur5e.xml"
OUT_META = ROOT / "assets" / "scene_ur5e.meta.json"

GRIPPER_CTRL = 0.0  # fixed opening: 0 = fully open (0..255 in menagerie units)
CAPSULE_MARGIN = 0.001  # m, added to fitted radii
N_OBSTACLE_SLOTS = 4  # K (SPEC §6.4)
PARK_Z = -5.0  # inactive obstacle slots are parked here
GRAVITY_COMPENSATION = True  # actuator gravity compensation, as on the real UR controller

# Contact bits. robot<->robot and robot<->environment collide; environment<->environment never does.
ROBOT_CT, ROBOT_CA = 1, 1
ENV_CT, ENV_CA = 2, 1


# --------------------------------------------------------------------------- gripper


def settle_gripper():
    """Settle the menagerie 2F-85 at GRIPPER_CTRL without gravity; return model, data."""
    m = mujoco.MjModel.from_xml_path(str(GRIPPER_XML))
    m.opt.gravity[:] = 0
    d = mujoco.MjData(m)
    d.ctrl[:] = GRIPPER_CTRL
    for _ in range(3000):
        mujoco.mj_step(m, d)
    assert np.abs(d.qvel).max() < 1e-6, "gripper did not settle"
    return m, d


def gripper_visuals(m, d):
    """Visual mesh geoms with their pose in the gripper attachment frame, plus all their vertices."""
    geoms, verts = [], []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] != 0:
            continue
        mesh = m.geom_dataid[g]
        # The compiler re-centres meshes: compiled frame = user frame * (mesh_pos, mesh_quat).
        # Undo that offset so that re-adding the mesh at this pose does not apply it twice.
        R_mesh = np.zeros(9)
        mujoco.mju_quat2Mat(R_mesh, m.mesh_quat[mesh])
        R_user = d.geom_xmat[g].reshape(3, 3) @ R_mesh.reshape(3, 3).T
        p_user = d.geom_xpos[g] - R_user @ m.mesh_pos[mesh]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_user.ravel())
        geoms.append(dict(
            mesh=m.mesh(mesh).name,
            pos=p_user,
            quat=quat,
            rgba=m.mat_rgba[m.geom_matid[g]].copy() if m.geom_matid[g] >= 0 else m.geom_rgba[g].copy(),
        ))
        adr, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        v = m.mesh_vert[adr:adr + n]
        verts.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
    return geoms, np.concatenate(verts)


def composite_inertia(m, d):
    """Total mass, COM and inertia tensor (about COM, attachment frame) of all gripper bodies."""
    mass = m.body_mass[1:].sum()
    com = (m.body_mass[1:, None] * d.xipos[1:]).sum(0) / mass
    inertia = np.zeros((3, 3))
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3)
        r = d.xipos[b] - com
        inertia += R @ np.diag(m.body_inertia[b]) @ R.T
        inertia += m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
    return mass, com, inertia


def fit_capsule(pts, axis, center_grid):
    """Smallest-volume capsule with the given axis direction enclosing pts.

    The axis line passes through a point chosen from center_grid (offsets perpendicular to the axis).
    For a fixed line and radius r the shortest enclosing segment is exact, so enclosure holds by
    construction; r is then optimised by 1-D search.
    """
    axis = axis / np.linalg.norm(axis)
    best = None
    for c in center_grid:
        rel = pts - c
        t = rel @ axis
        rho = np.linalg.norm(rel - np.outer(t, axis), axis=1)
        for r in np.linspace(rho.max(), rho.max() + 0.06, 241):
            s = np.sqrt(np.maximum(r * r - rho * rho, 0.0))
            lo, hi = (t - s).max(), (t + s).min()  # segment must span [lo, hi]
            if lo < hi:  # a single sphere suffices
                lo = hi = 0.5 * (lo + hi)
            vol = np.pi * r * r * (lo - hi if lo > hi else 0.0) + 4 / 3 * np.pi * r**3
            a, b = min(lo, hi), max(lo, hi)
            if best is None or vol < best[0]:
                best = (vol, c + a * axis, c + b * axis, r)
    return best


def fit_gripper_capsules(verts):
    """Palm capsule along the approach axis (z) + finger capsule across the opening axis (x)."""
    best = None
    for z_split in np.linspace(0.0, 0.14, 57):
        palm, fing = verts[verts[:, 2] < z_split], verts[verts[:, 2] >= z_split]
        if len(palm) == 0 or len(fing) == 0:
            continue
        cp = 0.5 * (palm.min(0) + palm.max(0))
        cf = 0.5 * (fing.min(0) + fing.max(0))
        p = fit_capsule(palm, np.array([0, 0, 1.0]), [np.array([cp[0], cp[1], 0.0])])
        zs = np.linspace(fing[:, 2].min(), fing[:, 2].max(), 21)
        f = fit_capsule(fing, np.array([1.0, 0, 0]), [np.array([0.0, cf[1], z]) for z in zs])
        if best is None or p[0] + f[0] < best[0]:
            best = (p[0] + f[0], p, f, z_split)
    _, p, f, z_split = best
    caps = []
    for name, (_, a, b, r) in (("gripper_palm", p), ("gripper_fingers", f)):
        # Round to 1 um so the XML (6 significant digits) round-trips exactly; round the radius up.
        caps.append(dict(name=name, p0=np.round(a, 6), p1=np.round(b, 6),
                         radius=np.ceil((r + CAPSULE_MARGIN) * 1e6) / 1e6))
    return caps, z_split


# --------------------------------------------------------------------------- scene


def build():
    gm, gd = settle_gripper()
    visuals, verts = gripper_visuals(gm, gd)
    mass, com, inertia = composite_inertia(gm, gd)
    caps, z_split = fit_gripper_capsules(verts)
    pads = np.array([gd.geom(n).xpos for n in ("left_pad1", "left_pad2", "right_pad1", "right_pad2")])
    tcp_pos = np.round(pads.mean(0), 6)  # midpoint between the fingertip pads (SPEC §6.1), 1 um grid

    spec = mujoco.MjSpec.from_file(str(UR5E_XML))
    spec.modelname = "ur5e_2f85_rigid"
    spec.meshdir = str(MENAGERIE)  # rewritten to a relative path in the saved XML
    for mesh in spec.meshes:
        mesh.file = f"universal_robots_ur5e/assets/{mesh.file}"
    spec.option.timestep = 0.002

    # Robot collision geoms: name them, convert the wrist-3 cylinder to an enclosing capsule.
    robot_bodies = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
                    "wrist_3_link"]
    for bname in robot_bodies:
        body = spec.body(bname)
        cols = [g for g in body.geoms if g.contype != 0 or g.group == 3]
        cols = [g for g in cols if g.type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER)]
        for i, g in enumerate(cols):
            g.name = f"{bname}_col{i}"
            if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                # capsule with the same radius and half-length strictly contains the cylinder
                g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            g.contype, g.conaffinity = ROBOT_CT, ROBOT_CA

    # Rigid gripper, welded at the UR5e attachment site.
    site = spec.site("attachment_site")
    wrist3 = spec.body("wrist_3_link")
    for vis in visuals:
        src = gm.mesh(vis["mesh"])
        if spec.mesh(f"2f85_{vis['mesh']}") is None:
            spec.add_mesh(name=f"2f85_{vis['mesh']}", file=f"robotiq_2f85/assets/{vis['mesh']}.stl",
                          scale=[0.001, 0.001, 0.001])
    grip = wrist3.add_body(name="gripper", pos=site.pos, quat=site.quat)
    grip.mass = mass
    grip.ipos = com
    grip.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    grip.explicitinertial = True
    for i, vis in enumerate(visuals):
        grip.add_geom(name=f"gripper_vis{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"2f85_{vis['mesh']}",
                      pos=vis["pos"], quat=vis["quat"], rgba=vis["rgba"], contype=0, conaffinity=0, group=2,
                      mass=0)
    for c in caps:
        grip.add_geom(name=c["name"], type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=np.r_[c["p0"], c["p1"]],
                      size=[c["radius"], 0, 0], contype=ROBOT_CT, conaffinity=ROBOT_CA, group=3, mass=0,
                      rgba=[0.9, 0.6, 0.2, 0.3])
    grip.add_site(name="tcp", pos=tcp_pos, size=[0.005, 0, 0], rgba=[1, 0, 0, 1], group=1)

    # The gripper sits right against wrist 2 through the wrist-3 joint; exclude that structural contact.
    spec.add_exclude(bodyname1="wrist_2_link", bodyname2="gripper")

    # Gravity compensation through the actuators, like the UR controller's own: without it the position
    # servos sag G/kp (up to ~30 mrad, 2-3 cm at the TCP) below every streamed joint target.
    if GRAVITY_COMPENSATION:
        for bname in robot_bodies + ["gripper"]:
            spec.body(bname).gravcomp = 1.0
        for joint in spec.joints:
            joint.actgravcomp = True

    # Environment: table top at z = 0 (robot base on it), floor below, goal mocap, obstacle slots.
    wb = spec.worldbody
    spec.add_material(name="table", rgba=[0.55, 0.45, 0.35, 1])
    spec.add_material(name="floor", rgba=[0.3, 0.3, 0.32, 1])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], pos=[0, 0, -0.75],
                material="floor", contype=ENV_CT, conaffinity=ENV_CA)
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[1.2, 1.2, 0.02], pos=[0, 0, -0.02],
                material="table", contype=ENV_CT, conaffinity=ENV_CA)
    wb.add_light(name="top", pos=[0, 0, 3], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    goal = wb.add_body(name="goal", mocap=True, pos=[0.4, 0.0, 0.4])
    goal.add_site(name="goal", size=[0.01, 0, 0], rgba=[0, 0.8, 0, 0.8], group=1)
    for k, (axis, rgba) in enumerate(((0, [1, 0, 0, 0.6]), (1, [0, 1, 0, 0.6]), (2, [0, 0, 1, 0.6]))):
        end = np.zeros(3)
        end[axis] = 0.06
        goal.add_geom(name=f"goal_axis{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=np.r_[np.zeros(3), end],
                      size=[0.004, 0, 0], rgba=rgba, contype=0, conaffinity=0, group=1)

    for k in range(N_OBSTACLE_SLOTS):
        ob = wb.add_body(name=f"obstacle{k}", mocap=True, pos=[0, 0, PARK_Z])
        ob.add_geom(name=f"obstacle{k}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0],
                    rgba=[0.85, 0.2, 0.2, 0.6], contype=ENV_CT, conaffinity=ENV_CA)

    model = spec.compile()
    xml = spec.to_xml().replace(f'meshdir="{MENAGERIE.as_posix()}/"', 'meshdir="menagerie/"')
    xml = xml.replace(f'meshdir="{MENAGERIE}"', 'meshdir="menagerie/"').replace(
        f'meshdir="{MENAGERIE.as_posix()}"', 'meshdir="menagerie/"')
    OUT_XML.write_text(xml)

    meta = dict(
        gripper_ctrl=GRIPPER_CTRL,
        gripper_mass=float(mass),
        capsule_split_z=float(z_split),
        capsules=[dict(name=c["name"], p0=c["p0"].tolist(), p1=c["p1"].tolist(), radius=float(c["radius"]))
                  for c in caps],
        tcp_in_gripper=tcp_pos.tolist(),
        n_gripper_vertices=int(len(verts)),
    )
    OUT_META.write_text(json.dumps(meta, indent=2))
    return model, meta


if __name__ == "__main__":
    model, meta = build()
    print(json.dumps(meta, indent=2))
    print(f"nq={model.nq} nu={model.nu} ngeom={model.ngeom} nbody={model.nbody} neq={model.neq}")
    print(f"wrote {OUT_XML.relative_to(ROOT)}")
