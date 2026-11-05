"""Five-panel comparison video of one real-arm episode, for the report and the slides.

    +-------------------------+----------------------+----------------------+
    |                         | REAL ARM (model) iso | REAL ARM (model) side|
    |  camera (cropped)       +----------------------+----------------------+
    |                         | SIMULATION iso       | SIMULATION side      |
    +-------------------------+----------------------+----------------------+

REAL ARM (model) = the model arm driven by the real arm's measured joint angles (with the virtual obstacles, goal frame,
tool trail and planned path); SIMULATION = the same episode in the simulator. Both are drawn with the official M5 shell meshes
(visual only, scripts/meshview.py) from a top isometric camera and from the side, the direction the real camera looks from.
The simulation panels are time-compressed to the real run when the real run (firmware segments) was shorter; the label says so.

    python scripts/render_grid_compare.py --set s4_test --episode 9 --method hybrid --real-dir results/video --tag-suffix _seg1 \\
        --real-video results/video/seg1_webcam.mp4 --real-approx 313.2 --out results/media/grid_ep9_hybrid.mp4
    python scripts/render_grid_compare.py ... --still 6.0 --out frame.png      # one frame at t = 6 s, for checking the layout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import make_comparison as mc  # noqa: E402
import meshview  # noqa: E402
import play  # noqa: E402

W, H, HEAD = 1920, 1080, 60
CAM_W = 900  # camera panel width; the four model panels are 510 x 510
SIDE = 510
FPS = 30


def font(n):
    try:
        return ImageFont.truetype("arial.ttf", n)
    except OSError:
        return ImageFont.load_default()


def panel_label(img, title, sub):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 52], fill=(20, 20, 20))
    d.text((10, 3), title, fill=(255, 255, 255), font=font(22))
    d.text((10, 29), sub, fill=(190, 190, 190), font=font(17))
    return np.asarray(im)


def make_cams(iso_az, side_az):
    def cam(az, el, dist):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = [0.08, 0.0, 0.21]
        c.distance, c.azimuth, c.elevation = dist, az, el
        return c
    return [("top isometric", cam(iso_az, -35.0, 1.0)), ("side view", cam(side_az, -3.0, 1.0))]


def renderer_for(player, mesh_dir):
    base = player.env.unwrapped
    base.model.vis.global_.offheight = max(base.model.vis.global_.offheight, SIDE)
    base.model.vis.global_.offwidth = max(base.model.vis.global_.offwidth, SIDE)
    r = meshview.VisRenderer(base, SIDE, SIDE, mesh_dir) if mesh_dir else mujoco.Renderer(base.model, SIDE, SIDE)
    opt = mujoco.MjvOption()
    opt.geomgroup[2] = 1 if mesh_dir else 0  # meshes
    opt.geomgroup[3] = 0 if mesh_dir else 1  # capsules
    return r, opt


def views(player, r, opt, cams, title, sub):
    base = player.env.unwrapped
    out = []
    for name, cam in cams:
        r.update_scene(base.data, cam, scene_option=opt)
        player.draw(r.scene, clear=False)
        out.append(panel_label(r.render().copy(), title, f"{sub} | {name}"))
    return out


def camera_panel(rd, src_fps, t, crop):
    fr = rd.get_data(int(round(t * src_fps)))
    im = Image.fromarray(fr)
    x0, x1 = int(crop[0] * im.width), int(crop[1] * im.width)
    im = im.crop((x0, 0, x1, im.height))
    s = min(CAM_W / im.width, (H - HEAD) / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)))
    canvas = Image.new("RGB", (CAM_W, H - HEAD))
    canvas.paste(im, ((CAM_W - im.width) // 2, ((H - HEAD) - im.height) // 2))
    return panel_label(np.asarray(canvas), "REAL ARM (camera)", "camera recording")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="s4_test")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--method", choices=["b2", "base", "hybrid"], default="hybrid")
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--real-dir", default="results/video")
    p.add_argument("--tag-suffix", default="")
    p.add_argument("--real-video")
    p.add_argument("--real-offset", type=float, default=None)
    p.add_argument("--real-approx", type=float, default=None)
    p.add_argument("--cam-crop", default="0.25,0.75", help="fraction of the camera frame's width kept (x0,x1)")
    p.add_argument("--mesh-dir", default=str(ROOT / "assets" / "m5_meshes"), help="'' to draw the capsule model")
    p.add_argument("--iso-az", type=float, default=140.0)
    p.add_argument("--side-az", type=float, default=90.0)
    p.add_argument("--no-sim-fit", action="store_true")
    p.add_argument("--still", type=float, default=None, help="write one PNG at this time (s) instead of a video")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    crop = [float(x) for x in args.cam_crop.split(",")]
    mesh_dir = args.mesh_dir or None
    cams = make_cams(args.iso_az, args.side_az)

    tag = f"real_{args.set}_ep{args.episode}_{args.method}" + (f"@{args.budget}" if args.method != "b2" else "") + args.tag_suffix
    log = json.load(open(ROOT / args.real_dir / f"{tag}.json"))
    qs = np.array([t["q"] for t in log["ticks"]])
    s = log["summary"]
    wall = np.array([t["wall"] for t in log["ticks"]], float)
    ns = SimpleNamespace(testset=args.set, episode=args.episode, method=args.method, budget=args.budget,
                         run=str(ROOT / mc.RUN) if args.method == "hybrid" else None, checkpoint="final", headless=False)
    names = dict(b2="B2 (full planner)", base=f"cheap planner @{args.budget}", hybrid=f"hybrid @{args.budget}")

    # simulation: pass 1 counts the ticks and finds the outcome
    sim_p = play.Replay(ns)
    sim_p.reset(new=False)
    n_sim, done = 0, False
    while not done:
        done = sim_p.step()
        n_sim += 1
    outcome = sim_p.outcome
    sim_t = 0.05 * (np.arange(n_sim) + 1)
    twin_t = wall
    note = ""
    if not args.no_sim_fit and sim_t[-1] > twin_t[-1] + 1.0:
        k = sim_t[-1] / twin_t[-1]
        sim_t = sim_t / k
        note = f" (shown at {k:.1f}x speed)"

    # camera sync (same rules as make_comparison.py)
    rd = off = None
    real_video, approx = args.real_video, args.real_approx
    if real_video and approx is not None:
        start = max(0.0, approx - 8.0)
        real_video = mc.cut_clip(real_video, start, 8.0 + twin_t[-1] + 8.0)
        approx = approx - start
    if real_video:
        rd = imageio.get_reader(mc.to_cfr(real_video, FPS))
        off = args.real_offset
        if off is None:
            off = mc.beep_offset(real_video)
            if off is not None and approx is not None and abs(off - approx) > 3.0:
                print("tone at %.1f s is far from the clock estimate %.1f s: ignored" % (off, approx))
                off = None
            if off is None and approx is not None:
                off = mc.motion_offset(real_video, approx)
                print("arm motion onset found at %.2f s (estimate %.1f s)" % (off, approx) if off is not None else "no motion onset found")
            if off is None:
                raise SystemExit("no sync found: pass --real-offset SECONDS")

    twin_p = play.Replay(ns)
    twin_base = twin_p.env.unwrapped
    twin_p.reset(new=False)
    twin_p.trail.clear()
    twin_r, opt = renderer_for(twin_p, mesh_dir)
    sim_p = play.Replay(ns)
    sim_p.reset(new=False)
    sim_r, _ = renderer_for(sim_p, mesh_dir)

    state = {"j": -1, "i": 0, "last_cam": None}
    sub_real = f"{names[args.method]} | {'SUCCESS' if s.get('success') else 'MISS'} {s.get('d_pos_cm', 0):.2f} cm, {s.get('theta_deg', 0):.1f} deg"
    header = (f"{args.set} episode {args.episode}  |  {names[args.method]}  |  {'firmware-segment execution' if s.get('mode') == 'segments' else 'streaming execution (0.5 deg per 50 ms)'}  |  "
              f"real arm: {'SUCCESS' if s.get('success') else 'MISS'} {s.get('d_pos_cm', 0):.2f} cm, {s.get('theta_deg', 0):.1f} deg  |  "
              f"simulation: {outcome}{note}")

    def twin_at(j):
        while state["j"] < j:  # advance (the tool trail needs every tick)
            state["j"] += 1
            twin_base.data.qpos[:6] = qs[state["j"]]
            twin_base.data.qvel[:] = 0
            mujoco.mj_forward(twin_base.model, twin_base.data)
            twin_p.trail.append(twin_base.data.site_xpos[twin_base.tcp_id].copy())
        return views(twin_p, twin_r, opt, cams, "REAL ARM (model)", "measured joints, virtual obstacles")

    def sim_at(i):
        while state["i"] < i + 1:
            sim_p.step()
            state["i"] += 1
        return views(sim_p, sim_r, opt, cams, "SIMULATION", "virtual obstacles" + note)

    def compose(t):
        j = min(len(qs) - 1, int(np.searchsorted(twin_t, t, side="right")))
        i = min(n_sim - 1, int(np.searchsorted(sim_t, t, side="right")))
        tw, sm = twin_at(j), sim_at(i)
        canvas = np.zeros((H, W, 3), np.uint8)
        if rd is not None:
            try:
                state["last_cam"] = camera_panel(rd, FPS, max(0.0, off + t), crop)
            except (IndexError, ValueError):
                pass
            if state["last_cam"] is not None:
                canvas[HEAD:, :CAM_W] = state["last_cam"]
        canvas[HEAD:HEAD + SIDE, CAM_W:CAM_W + SIDE] = tw[0]
        canvas[HEAD:HEAD + SIDE, CAM_W + SIDE:] = tw[1]
        canvas[HEAD + SIDE:, CAM_W:CAM_W + SIDE] = sm[0]
        canvas[HEAD + SIDE:, CAM_W + SIDE:] = sm[1]
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, HEAD], fill=(12, 12, 14))
        d.text((16, 16), header, fill=(255, 255, 255), font=font(26))
        return np.asarray(im)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.still is not None:
        Image.fromarray(compose(args.still)).save(out)
        print("wrote", out)
        return
    T = max(twin_t[-1], sim_t[-1]) + 1.0
    nfr = int(T * FPS)
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=8) as w:
        for k in range(nfr):
            w.append_data(compose(k / FPS))
    print("wrote", out, f"({nfr} frames, {T:.1f} s)")


if __name__ == "__main__":
    main()
