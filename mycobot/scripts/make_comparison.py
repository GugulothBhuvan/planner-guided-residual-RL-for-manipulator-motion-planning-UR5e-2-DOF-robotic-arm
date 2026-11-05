"""Side-by-side comparison videos of one test episode: simulation vs the real arm.

Panels (left to right, same camera, same virtual obstacles, goal frame, tool trail and planned path):
  [optional] REAL CAMERA   a phone / camera recording of the real arm (--real-video, --real-offset seconds to skip)
  REAL ARM (MODEL)         the model arm driven by the real arm's MEASURED joint angles from results/real/ (digital twin;
                           this is where the real arm is relative to the virtual obstacles and the goal)
  SIMULATION               the same episode run in the simulator (planner, or planner + residual)
The virtual obstacles do not exist in the real world, so they can only be drawn on the model panels; the camera panel
shows the same motion for comparison. All panels run at the 20 Hz control rate (real time); shorter ones hold their last frame.

    python scripts/make_comparison.py --episode 0 --method hybrid           # twin vs simulation
    python scripts/make_comparison.py --episode 0 --method hybrid --real-video clip.mp4 --real-offset 3.2
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
import play  # noqa: E402

PANEL = (540, 720)  # height, width
RUN = "results/runs/m5/residual_b99f130c_s1"


def make_renderer(base):
    base.model.vis.global_.offheight = max(base.model.vis.global_.offheight, PANEL[0])
    base.model.vis.global_.offwidth = max(base.model.vis.global_.offwidth, PANEL[1])
    r = mujoco.Renderer(base.model, *PANEL)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.08, 0.0, 0.22]
    cam.distance, cam.azimuth, cam.elevation = 0.95, 140.0, -15.0
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1  # the arm's capsules
    return r, cam, opt


def label(img, text, sub=""):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        f1, f2 = ImageFont.truetype("arial.ttf", 26), ImageFont.truetype("arial.ttf", 18)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 64], fill=(20, 20, 20))
    d.text((12, 6), text, fill=(255, 255, 255), font=f1)
    d.text((12, 38), sub, fill=(200, 200, 200), font=f2)
    return np.asarray(im)


def render(player, renderer, cam, opt):
    base = player.env.unwrapped
    renderer.update_scene(base.data, cam, scene_option=opt)
    player.draw(renderer.scene, clear=False)
    return renderer.render().copy()


def sim_frames(player):
    base = player.env.unwrapped
    r, cam, opt = make_renderer(base)
    player.reset(new=False)
    frames, done = [], False
    while not done:
        done = player.step()
        frames.append(render(player, r, cam, opt))
    return frames, player.outcome


def twin_frames(player, qs, title, sub):
    base = player.env.unwrapped
    r, cam, opt = make_renderer(base)
    player.reset(new=False)
    player.trail.clear()
    frames = []
    for q in qs:
        base.data.qpos[:6] = q
        base.data.qvel[:] = 0
        mujoco.mj_forward(base.model, base.data)
        player.trail.append(base.data.site_xpos[base.tcp_id].copy())
        frames.append(label(render(player, r, cam, opt), title, sub))
    return frames


def to_cfr(path, fps=30):
    """Re-encode a phone video to a constant frame rate (phones record variable rates) and return the new path."""
    import subprocess
    import tempfile

    import imageio_ffmpeg
    out = Path(tempfile.mkdtemp()) / "cfr.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-vf", "scale=-2:720", "-r", str(fps), "-vsync", "cfr",
                    "-c:v", "libx264", "-crf", "20", "-an", str(out)], check=True)
    return str(out)


def cut_clip(path, start, length, fps=30):
    """Cut [start, start + length] s from a long recording, re-encoded at 720p constant rate (batch recordings hold many episodes)."""
    import subprocess
    import tempfile

    import imageio_ffmpeg
    out = Path(tempfile.mkdtemp()) / "clip.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i",
                    str(path), "-vf", "scale=-2:720", "-r", str(fps), "-vsync", "cfr", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-c:a", "aac", str(out)], check=True)
    return str(out)


def beep_offset(path, freq=1000.0):
    """Seconds into the video where the 1 kHz sync tone (run_real.py --beep) starts, or None if it is not found."""
    import subprocess
    import tempfile
    import wave

    import imageio_ffmpeg
    wav = Path(tempfile.mkdtemp()) / "a.wav"
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar",
                        "16000", str(wav)])
    if r.returncode or not wav.exists():
        return None
    with wave.open(str(wav)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
    win = sr // 50  # 20 ms
    n = len(x) // win
    seg = x[:n * win].reshape(n, win) * np.hanning(win)
    spec = np.abs(np.fft.rfft(seg, axis=1))
    f = np.fft.rfftfreq(win, 1 / sr)
    score = spec[:, (f > freq - 100) & (f < freq + 100)].sum(1) / (spec.sum(1) + 1e-9)  # share of the 1 kHz band
    peak = int(np.argmax(score))
    if score[peak] < 2.5 * np.median(score):
        return None
    i = peak  # the tone starts at the first window of the run of windows above 40% of the peak that ends at the peak
    while i > 0 and score[i - 1] > 0.4 * score[peak]:
        i -= 1
    return float(i * 0.02)


def motion_offset(path, approx, half=3.0, fps=30):
    """Refine an approximate episode-start time (s into the video) by the first arm movement after the still hold:
    gray 160x90 frames of [approx - half, approx + half], frame-difference energy against the still period before approx - 0.5.
    Returns seconds into the video, or None. Accuracy about +-0.2 s (the arm starts slowly)."""
    import subprocess

    import imageio_ffmpeg
    t0 = max(0.0, approx - half)
    raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-ss", f"{t0}", "-t", f"{2 * half}", "-i", str(path),
                          "-vf", f"scale=160:90,fps={fps},format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 90, 160).astype(float)
    if len(fr) < 20:
        return None
    e = np.abs(np.diff(fr[:, :, 30:110], axis=0)).mean((1, 2))  # central columns, where the arm is
    still = e[: max(5, int((approx - 0.5 - t0) * fps))]
    base = np.median(still)
    thr = base + max(8 * np.median(np.abs(still - base)), 0.035)
    i0 = max(0, int((approx - 1.5 - t0) * fps))  # the first motion after the hold, searched from 1.5 s before the estimate
    for i in range(i0, len(e) - 3):
        if (e[i:i + 3] > thr).all():
            return float(t0 + (i + 1) / fps)
    return None


def camera_at(rd, src_fps, t):
    fr = rd.get_data(int(round(t * src_fps)))
    im = Image.fromarray(fr)
    s = min(PANEL[0] / im.height, PANEL[1] / im.width)  # fit inside the panel (no crop), black bars if the aspect differs
    im = im.resize((int(im.width * s), int(im.height * s)))
    canvas = Image.new("RGB", (PANEL[1], PANEL[0]))
    canvas.paste(im, ((PANEL[1] - im.width) // 2, (PANEL[0] - im.height) // 2))
    return label(np.asarray(canvas), "REAL ARM (camera)", "camera recording")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="s4_test")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--method", choices=["b2", "base", "hybrid"], default="hybrid")
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--real-dir", default="results/real")
    p.add_argument("--real-video")
    p.add_argument("--real-offset", type=float, default=None, help="seconds into the camera video where the episode starts; default: find the --beep tone in its audio")
    p.add_argument("--real-approx", type=float, default=None, help="approximate episode start in the video (s); used when no beep is found")
    p.add_argument("--tag-suffix", default="", help="log name suffix (e.g. _video1_s20e1)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tag = f"real_{args.set}_ep{args.episode}_{args.method}" + (f"@{args.budget}" if args.method != "b2" else "") + args.tag_suffix
    log = json.load(open(ROOT / args.real_dir / f"{tag}.json"))
    qs = [t["q"] for t in log["ticks"]]
    s = log["summary"]
    ns = SimpleNamespace(testset=args.set, episode=args.episode, method=args.method, budget=args.budget,
                         run=str(ROOT / RUN) if args.method == "hybrid" else None, checkpoint="final", headless=False)
    player = play.Replay(ns)
    names = dict(b2="B2 (full planner)", base=f"cheap planner @{args.budget}", hybrid=f"hybrid @{args.budget}")
    sub = f"{args.set} episode {args.episode}, {names[args.method]}"
    twin = twin_frames(player, qs, "REAL ARM, joints replayed in the model",
                       f"{sub} | measured | {'SUCCESS' if s.get('success') else 'MISS'} {s.get('d_pos_cm', 0):.2f} cm, "
                       f"{s.get('theta_deg', 0):.1f} deg")
    player2 = play.Replay(ns)
    raw, outcome = sim_frames(player2)
    sim = [label(f, "SIMULATION", f"{sub} | virtual obstacles | {outcome}") for f in raw]
    FPS = 30
    wall = [t.get("wall") for t in log["ticks"]]
    twin_t = np.array(wall, float) if all(w is not None for w in wall) else 0.05 * (np.arange(len(twin)) + 1)
    sim_t = 0.05 * (np.arange(len(sim)) + 1)
    rd = off = src = None
    if args.real_video and args.real_approx is not None:  # cut only the part of the recording around this episode
        start = max(0.0, args.real_approx - 8.0)
        args.real_video = cut_clip(args.real_video, start, 8.0 + twin_t[-1] + 8.0)
        args.real_approx = args.real_approx - start
    if args.real_video:
        cfr = to_cfr(args.real_video, FPS)
        rd = imageio.get_reader(cfr)
        src = FPS
        off = args.real_offset
        if off is None:
            off = beep_offset(args.real_video)
            if off is not None and args.real_approx is not None and abs(off - args.real_approx) > 3.0:
                print("tone at %.1f s is far from the clock estimate %.1f s: ignored" % (off, args.real_approx))
                off = None
            if off is None and args.real_approx is not None:
                off = motion_offset(args.real_video, args.real_approx)
                print("arm motion onset found at %.2f s in the video (estimate %.1f s)" % (off, args.real_approx) if off is not None else "no motion onset found")
            print("sync tone found at %.2f s in the video" % off if off is not None else "sync tone NOT found")
            if off is None:
                raise SystemExit("no sync tone in the video's audio: pass --real-offset SECONDS (episode start in the video)")
        n_cam = rd.count_frames() if hasattr(rd, "count_frames") else 10 ** 9
    T = max(twin_t[-1], sim_t[-1]) + 1.0
    out = Path(args.out or ROOT / "results" / "media" / f"compare_ep{args.episode}_{args.method}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    nfr = int(T * FPS)
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=8) as w:
        for k in range(nfr):
            t = k / FPS
            panels = []
            if rd is not None:
                try:
                    panels.append(camera_at(rd, src, max(0.0, off + t)))
                except (IndexError, ValueError):
                    panels.append(np.zeros((PANEL[0], PANEL[1], 3), np.uint8))
            panels.append(twin[min(len(twin) - 1, int(np.searchsorted(twin_t, t, side="right")))])
            panels.append(sim[min(len(sim) - 1, int(np.searchsorted(sim_t, t, side="right")))])
            w.append_data(np.concatenate(panels, axis=1))
    n = nfr
    panels = [0] * (3 if rd is not None else 2)
    print("wrote", out, f"({n} frames, {n / FPS:.1f} s, panels {len(panels)})")


if __name__ == "__main__":
    main()
