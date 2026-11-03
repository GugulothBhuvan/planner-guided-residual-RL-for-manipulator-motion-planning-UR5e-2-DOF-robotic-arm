"""Open the reconstructed MyCobot 280 model in the MuJoCo viewer, with the collision capsules (geom group 3) shown.

    python mycobot/scripts/view_model.py            # zero pose
    python mycobot/scripts/view_model.py --pose 7   # one of the 15 real poses measured on the arm (results/poses.json)
Prints the model's tool position next to the arm's own reported coordinates for that pose.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--pose", type=int, default=None, help="index into results/poses.json (0-14)")
args = p.parse_args()

m = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "scene_mc280.xml"))
d = mujoco.MjData(m)
if args.pose is not None:
    rec = json.loads((ROOT / "results" / "poses.json").read_text())[args.pose]
    d.qpos[:6] = np.deg2rad(rec["angles"])
    mujoco.mj_forward(m, d)
    tcp = d.site_xpos[m.site("tcp").id] * 1000
    print("joint angles (deg):", rec["angles"])
    print("model tool position (mm):", tcp.round(1).tolist())
    print("arm-reported coords (mm):", rec["coords"][:3])
    print("difference (mm): %.1f" % np.linalg.norm(tcp - np.array(rec["coords"][:3])))
else:
    mujoco.mj_forward(m, d)

with mujoco.viewer.launch_passive(m, d) as v:
    v.opt.geomgroup[3] = 1  # capsules are in group 3 (hidden by default)
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.0, 135, -20
    v.cam.lookat[:] = (0, 0, 0.2)
    print("Viewer open. Drag the Control sliders (right panel) to move the joints. Close the window to exit.")
    while v.is_running():
        mujoco.mj_forward(m, d) if args.pose is not None else mujoco.mj_step(m, d)
        v.sync()
