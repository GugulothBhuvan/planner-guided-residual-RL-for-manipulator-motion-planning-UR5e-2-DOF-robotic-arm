"""Camera guide for filming: the simulator viewport (episode start pose, virtual obstacles) and a top-down / side schematic of
where the camera sits relative to the arm base, so a phone can be placed to match.

    python scripts/camera_guide.py [--episode 0] [--out results/media/camera_guide.png]
Robot frame: origin = centre of the arm base on the table, x = straight ahead (J1 = 0), y = left, z = up (metres).
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_comparison as mc  # noqa: E402
import play  # noqa: E402
from ur5rl.viz import overlays  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "results" / "media" / "camera_guide.png"))
    args = p.parse_args()
    player = play.Replay(SimpleNamespace(testset="s4_test", episode=args.episode, method="base", budget=3000, run=None,
                                         checkpoint="final", headless=False))
    player.reset(new=False)
    base = player.env.unwrapped
    base.data.qpos[:6] = base.q_start
    mujoco.mj_forward(base.model, base.data)
    r, cam, opt = mc.make_renderer(base)
    r.update_scene(base.data, cam, scene_option=opt)
    player.draw(r.scene, clear=False)
    overlays.frame(r.scene, np.zeros(3), np.eye(3), length=0.10, width=0.004, alpha=0.9)  # base axes: x red, y green, z blue
    view = Image.fromarray(r.render().copy())

    az, el, dist, look = np.deg2rad(cam.azimuth), np.deg2rad(cam.elevation), cam.distance, np.array(cam.lookat)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = look - dist * fwd
    try:
        f1, f2 = ImageFont.truetype("arial.ttf", 22), ImageFont.truetype("arial.ttf", 16)
    except OSError:
        f1 = f2 = ImageFont.load_default()

    W, H = view.width, view.height
    sch = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(sch)
    s = 330.0  # pixels per metre, top-down: x up, y left
    cx, cy = W * 0.40, H * 0.60
    to = lambda x, y: (cx - y * s, cy - x * s)
    d.text((14, 10), "TOP-DOWN (looking at the table from above)", fill=(0, 0, 0), font=f1)
    for rr, lab in ((0.28, "arm reach ~0.28 m"),):
        d.ellipse([cx - rr * s, cy - rr * s, cx + rr * s, cy + rr * s], outline=(150, 150, 150))
        d.text((cx + 6, cy - rr * s - 20), lab, fill=(110, 110, 110), font=f2)
    d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(60, 60, 60))
    d.text((cx + 12, cy + 10), "arm base", fill=(0, 0, 0), font=f2)
    d.line([to(0, 0), to(0.2, 0)], fill=(220, 0, 0), width=4)
    d.text((to(0.2, 0)[0] + 6, to(0.2, 0)[1] - 8), "x (ahead, J1 = 0)", fill=(220, 0, 0), font=f2)
    d.line([to(0, 0), to(0, 0.2)], fill=(0, 150, 0), width=4)
    d.text((to(0, 0.2)[0] - 20, to(0, 0.2)[1] - 24), "y (left)", fill=(0, 150, 0), font=f2)
    for k in range(base.K):
        if base.ob_mask[k]:
            c, rad = base.ob_center[k], base.ob_radius[k]
            px, py = to(c[0], c[1])
            d.ellipse([px - rad * s, py - rad * s, px + rad * s, py + rad * s], outline=(200, 40, 40), width=2)
    gx, gy = to(base.goal[0], base.goal[1])
    d.ellipse([gx - 5, gy - 5, gx + 5, gy + 5], fill=(30, 90, 200))
    d.text((gx + 8, gy - 8), "goal", fill=(30, 90, 200), font=f2)
    px, py = to(pos[0], pos[1])
    lx, ly = to(look[0], look[1])
    d.line([(px, py), (lx, ly)], fill=(0, 0, 0), width=2)
    d.polygon([(px - 14, py - 9), (px + 14, py - 9), (px + 14, py + 9), (px - 14, py + 9)], fill=(0, 0, 0))
    d.text((px - 60, py + 14), "CAMERA", fill=(0, 0, 0), font=f1)
    d.text((14, H - 130), f"camera position: x = {pos[0]:+.2f} m,  y = {pos[1]:+.2f} m,  height z = {pos[2]:.2f} m", fill=(0, 0, 0), font=f2)
    d.text((14, H - 104), f"looks at: x = {look[0]:+.2f} m, y = {look[1]:+.2f} m, z = {look[2]:.2f} m  (above the base, ahead)", fill=(0, 0, 0), font=f2)
    d.text((14, H - 78), f"distance to that point {dist:.2f} m, tilted {abs(cam.elevation):.0f} deg down, swung {cam.azimuth:.0f} deg", fill=(0, 0, 0), font=f2)
    d.text((14, H - 52), "field of view (vertical) ~45 deg: the whole arm and its tool path should fit with a margin", fill=(0, 0, 0), font=f2)

    out = Image.new("RGB", (W * 2, H), (255, 255, 255))
    out.paste(view, (0, 0))
    out.paste(sch, (W, 0))
    dd = ImageDraw.Draw(out)
    dd.rectangle([0, 0, W, 34], fill=(20, 20, 20))
    dd.text((10, 5), f"SIMULATOR VIEWPORT (episode {args.episode} start pose; axes: x red, y green, z blue)", fill=(255, 255, 255), font=f2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    view.save(Path(args.out).with_name("viewport_only.png"))
    print("camera position", pos.round(3), "look", look, "wrote", args.out)


if __name__ == "__main__":
    main()
