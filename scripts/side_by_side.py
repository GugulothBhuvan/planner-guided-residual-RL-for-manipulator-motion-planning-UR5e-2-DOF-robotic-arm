"""Side-by-side replay video: the four single-method replays of one S4 test episode (scripts/play.py --record) in a 2x2
grid, each tile labeled with its method and its evaluated outcome (read from results/eval/s4_test/*.jsonl). Shorter
clips hold their last frame until the longest ends.

    python scripts/side_by_side.py --episode 32   ->  results/videos/ep32_side_by_side.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "results" / "videos"
EVAL = ROOT / "results" / "eval" / "s4_test"
# (video suffix, tile label, evaluation rows) in grid order: top-left, top-right, bottom-left, bottom-right
TILES = [("1_base3000", "Planner alone (3000 checks)", "base_3000"),
         ("2_e5_hybrid_s0_primary", "E5 additive residual (s0, primary)", "hybrid_3000__residual_ff20115a_s0__primary"),
         ("3_e5b_hybrid_s0_final", "E5b offset residual (s0, final)", "hybrid_3000__residual_offset_4f49c707_s0__final"),
         ("4_b2", "Full B2 (reference)", "b2")]
TILE_W, TILE_H, BAR, TITLE = 640, 480, 34, 44  # height 1072: divisible by 8 for libx264
INK, MUTED, GOOD, BAD = (11, 11, 11), (82, 81, 78), (12, 128, 62), (196, 36, 48)


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def outcome(rows_name, episode):
    row = json.loads((EVAL / f"{rows_name}.jsonl").read_text(encoding="utf-8").splitlines()[episode])
    if row["success"]:
        return f"success at {row['time_to_success_s']:.1f} s", GOOD
    if row["collision"]:
        return "collision", BAD
    return f"pose miss ({row['d_pos_cm']:.1f} cm / {row['theta_deg']:.1f}°)", BAD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--title", default=None)
    args = p.parse_args()
    clips, labels = [], []
    for suffix, label, rows in TILES:
        reader = imageio.get_reader(VIDEOS / f"ep{args.episode}_{suffix}.mp4")
        fps = reader.get_meta_data()["fps"]
        clips.append([np.asarray(Image.fromarray(f).resize((TILE_W, TILE_H), Image.LANCZOS)) for f in reader])
        reader.close()
        labels.append((label, *outcome(rows, args.episode)))
    n = max(len(c) for c in clips)
    f_label, f_out, f_title = font(17), font(16), font(19)
    title = args.title or f"S4 test episode {args.episode}: same scene, same planner budget, four controllers"
    out = VIDEOS / f"ep{args.episode}_side_by_side.mp4"
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8) as w:
        for k in range(n):
            canvas = Image.new("RGB", (2 * TILE_W, TITLE + 2 * (TILE_H + BAR)), "white")
            d = ImageDraw.Draw(canvas)
            d.text((12, 10), title, font=f_title, fill=INK)
            for i, (clip, (label, text, color)) in enumerate(zip(clips, labels)):
                x, y = (i % 2) * TILE_W, TITLE + (i // 2) * (TILE_H + BAR)
                d.text((x + 10, y + 8), label, font=f_label, fill=INK)
                if k >= len(clip) - 1:  # the outcome appears when that clip has ended
                    tw = d.textlength(text, font=f_out)
                    d.text((x + TILE_W - tw - 10, y + 9), text, font=f_out, fill=color)
                canvas.paste(Image.fromarray(clip[min(k, len(clip) - 1)]), (x, y + BAR))
            w.append_data(np.asarray(canvas))
    print(f"wrote {out} ({n} frames)")


if __name__ == "__main__":
    main()
