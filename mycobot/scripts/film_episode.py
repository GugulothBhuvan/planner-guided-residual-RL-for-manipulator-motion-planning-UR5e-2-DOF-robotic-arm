"""Run one real-arm episode while the laptop's USB webcam (and its microphone) records it, then build the 3-panel video.

The webcam is recorded by ffmpeg (DirectShow) only for the duration of this command and saved under results/video/. The same
1 kHz beep that starts the episode is recorded by the webcam's microphone and used to align the panels (make_comparison.py).

    python scripts/film_episode.py --episode 0 --method hybrid            # execute on the real arm, record, compose
    python scripts/film_episode.py --list                                 # show the cameras / microphones ffmpeg sees
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
FF = imageio_ffmpeg.get_ffmpeg_exe()
PY = sys.executable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--camera", default="w300")
    p.add_argument("--mic", default="Microphone (w300)")
    p.add_argument("--size", default="1280x720")
    p.add_argument("--episode", type=int)
    p.add_argument("--method", choices=["b2", "base", "hybrid"], default="hybrid")
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--hold", type=float, default=5.0)
    p.add_argument("--tag", default=None, help="name of the recording (default film_ep<N>_<method>_<time>)")
    args = p.parse_args()
    if args.list:
        subprocess.run([FF, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        return
    tag = args.tag or f"film_ep{args.episode}_{args.method}_{time.strftime('%H%M%S')}"
    out = ROOT / "results" / "video"
    out.mkdir(parents=True, exist_ok=True)
    video = out / f"{tag}_webcam.mp4"
    rec = subprocess.Popen(
        [FF, "-y", "-loglevel", "error", "-f", "dshow", "-rtbufsize", "200M", "-vcodec", "mjpeg", "-framerate", "30", "-video_size", args.size,
         "-i", f"video={args.camera}:audio={args.mic}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)],
        stdin=subprocess.PIPE)
    try:
        time.sleep(3.0)  # camera warm-up, so the start of the episode is not lost
        if rec.poll() is not None:
            raise SystemExit("ffmpeg could not open the webcam (is another program using it?)")
        cmd = [PY, str(ROOT / "scripts" / "run_real.py"), "--episode", str(args.episode), "--method", args.method,
               "--budget", str(args.budget), "--execute", "--hold-before", str(args.hold), "--beep",
               "--out-dir", "results/video", "--tag-suffix", f"_{tag}"]
        if args.method == "hybrid":
            cmd += ["--run", "results/runs/m5/residual_b99f130c_s1", "--checkpoint", "final"]
        subprocess.run(cmd, cwd=ROOT, check=False)
        time.sleep(2.0)
    finally:
        t_q = time.time()
        try:
            rec.stdin.write(b"q")  # ffmpeg finishes the file cleanly on 'q'
            rec.stdin.flush()
        except Exception:
            pass
        try:
            rec.wait(timeout=20)
        except subprocess.TimeoutExpired:
            rec.kill()
    print("webcam recording:", video)
    import json
    import imageio.v2 as imageio
    name = f"real_s4_test_ep{args.episode}_{args.method}" + (f"@{args.budget}" if args.method != "b2" else "") + f"_{tag}.json"
    approx = None
    try:
        e0 = json.load(open(out / name))["summary"]["epoch_start"]
        dur = imageio.get_reader(str(video)).get_meta_data()["duration"]
        approx = e0 - (t_q - dur)  # the video ends when 'q' was sent: its start is t_q - duration
        print(f"episode starts about {approx:.1f} s into the webcam video (clock estimate)")
    except Exception as exc:
        print("no clock estimate:", exc)
    cmd = [PY, str(ROOT / "scripts" / "make_comparison.py"), "--episode", str(args.episode), "--method", args.method,
           "--budget", str(args.budget), "--real-dir", "results/video", "--tag-suffix", f"_{tag}",
           "--real-video", str(video), "--out", str(out / f"{tag}_compare.mp4")]
    if approx is not None:
        cmd += ["--real-approx", f"{approx:.2f}"]
    subprocess.run(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
