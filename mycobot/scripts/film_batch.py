"""Film many real-arm episodes in ONE continuous webcam recording, then compose each episode's 3-panel video offline.

    python scripts/film_batch.py run  --name b1 --set s4_test --episodes 0 1 2 --methods b2 hybrid   # arm + recording (about 1 min per episode)
    python scripts/film_batch.py make --name b1 [--only s4_test:7:hybrid ...]                        # offline: one video per case

`run` records the webcam and microphone for the whole batch (lower bitrate), runs the cases one after another with the same
settings as film_episode.py (5 s hold, beep), and writes results/video/<name>_manifest.json (per case: log, epoch_start;
recording end time and length). `make` converts each case's clock estimate (epoch_start - video start) into --real-approx for
make_comparison.py, which cuts that episode out of the long recording and refines the start by the arm's first movement.
Close the Windows Camera app first.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
FF = imageio_ffmpeg.get_ffmpeg_exe()
PY = sys.executable
OUT = ROOT / "results" / "video"
RUN = "results/runs/m5/residual_b99f130c_s1"


def log_path(name, set_, ep, method):
    return OUT / (f"real_{set_}_ep{ep}_{method}" + ("@3000" if method != "b2" else "") + f"_{name}.json")


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / f"{args.name}_webcam.mp4"
    cases = [(args.set, e, m) for e in args.episodes for m in args.methods]
    rec = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "dshow", "-rtbufsize", "300M", "-vcodec", "mjpeg", "-framerate", "30",
                            "-video_size", "1280x720", "-i", f"video={args.camera}:audio={args.mic}", "-c:v", "libx264",
                            "-preset", "ultrafast", "-crf", "27", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)],
                           stdin=subprocess.PIPE)
    manifest = dict(name=args.name, cases=[])
    try:
        time.sleep(3.0)
        if rec.poll() is not None:
            raise SystemExit("ffmpeg could not open the webcam (is the Windows Camera app or a browser using it?)")
        for set_, ep, m in cases:
            cmd = [PY, str(ROOT / "scripts" / "run_real.py"), "--set", set_, "--episode", str(ep), "--method", m, "--budget", "3000",
                   "--execute", "--hold-before", "5", "--beep", "--out-dir", "results/video", "--tag-suffix", f"_{args.name}"]
            if m == "hybrid":
                cmd += ["--run", RUN, "--checkpoint", "final"]
            print(time.strftime("%H:%M:%S"), "case", set_, ep, m, flush=True)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            log = log_path(args.name, set_, ep, m)
            if not log.exists():
                print("  no log:", (r.stdout + r.stderr)[-300:], flush=True)
                manifest["cases"].append(dict(set=set_, episode=ep, method=m, ok=False))
                break
            sm = json.load(open(log))["summary"]
            print("  ", sm["status"][:40], "success" if sm.get("success") else "no success", flush=True)
            manifest["cases"].append(dict(set=set_, episode=ep, method=m, ok=sm["status"] == "ok", success=sm.get("success"),
                                          epoch_start=sm.get("epoch_start"), log=str(log.relative_to(ROOT))))
            if sm["status"] != "ok":
                print("  stopping the batch after an abort", flush=True)
                break
        time.sleep(2.0)
    finally:
        manifest["t_q"] = time.time()
        try:
            rec.stdin.write(b"q")
            rec.stdin.flush()
        except Exception:
            pass
        try:
            rec.wait(timeout=30)
        except subprocess.TimeoutExpired:
            rec.kill()
        manifest["video"] = str(video.relative_to(ROOT))
        try:
            manifest["duration"] = imageio.get_reader(str(video)).get_meta_data()["duration"]
        except Exception:
            manifest["duration"] = None
        (OUT / f"{args.name}_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("recording:", video, "manifest:", OUT / f"{args.name}_manifest.json")


def make(args):
    man = json.load(open(OUT / f"{args.name}_manifest.json"))
    start = man["t_q"] - man["duration"]  # the recording starts `duration` seconds before 'q' was sent
    for c in man["cases"]:
        key = f"{c['set']}:{c['episode']}:{c['method']}"
        if not c.get("ok") or (args.only and key not in args.only):
            continue
        approx = c["epoch_start"] - start
        out = OUT / f"{args.name}_{c['set']}_ep{c['episode']}_{c['method']}_compare.mp4"
        if out.exists() and not args.force:
            continue
        print("composing", key, "estimate %.1f s" % approx, flush=True)
        subprocess.run([PY, str(ROOT / "scripts" / "make_comparison.py"), "--set", c["set"], "--episode", str(c["episode"]),
                        "--method", c["method"], "--real-dir", "results/video", "--tag-suffix", f"_{args.name}",
                        "--real-video", str(ROOT / man["video"]), "--real-approx", f"{approx:.2f}", "--out", str(out)],
                       cwd=ROOT, stdout=subprocess.DEVNULL)
        print("  ->", out.name, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "make"])
    p.add_argument("--name", required=True)
    p.add_argument("--set", default="s4_test")
    p.add_argument("--episodes", type=int, nargs="*", default=[])
    p.add_argument("--methods", nargs="*", default=["hybrid"])
    p.add_argument("--camera", default="w300")
    p.add_argument("--mic", default="Microphone (w300)")
    p.add_argument("--only", nargs="*")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    (run if args.mode == "run" else make)(args)


if __name__ == "__main__":
    main()
