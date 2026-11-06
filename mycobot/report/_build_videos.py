"""Rebuild the five-panel video block of Section 6.9.6 (and switch Videos 6.2-6.6 to five panels where rendered).
Copies finished videos from results/media/ into report/vid/. Run from mycobot/report with the project venv; idempotent.

    ../../.venv/Scripts/python.exe _build_videos.py
"""
import re
import shutil
from pathlib import Path

import imageio.v2 as imageio

HERE = Path(__file__).resolve().parent
MEDIA = HERE.parent / "results" / "media"
VID = HERE / "vid"
VID.mkdir(exist_ok=True)
p = HERE / "report.html"
s = p.read_text(encoding="utf-8")


def finished(path):
    if not path.exists() or path.stat().st_size < 100000:
        return False
    try:
        imageio.get_reader(str(path)).count_frames()
        return True
    except Exception:
        return False


def copy(path):
    shutil.copy(path, VID / path.name)
    return path.name


# ---- Videos 6.2-6.6 (streaming runs): five-panel where available ------------------------------------------------
stream = {"real_ep0_hybrid.mp4": ("grid_stream_s4_test_ep0_hybrid.mp4", "episode 0, hybrid @3000"),
          "real_ep7_b2.mp4": ("grid_stream_s4_test_ep7_b2.mp4", "episode 7, B2"),
          "real_ep7_hybrid.mp4": ("grid_stream_s4_test_ep7_hybrid.mp4", "episode 7, hybrid @3000"),
          "real_ep8_hybrid.mp4": ("grid_stream_s4_test_ep8_hybrid.mp4", "episode 8, hybrid @3000"),
          "real_ep9_hybrid.mp4": ("grid_stream_s4_test_ep9_hybrid.mp4", "episode 9, hybrid @3000")}
n_stream = 0
for old, (new, _) in stream.items():
    f = MEDIA / new
    if finished(f) and f"vid/{old}" in s:
        s = s.replace(f"vid/{old}", f"vid/{copy(f)}")
        n_stream += 1
if n_stream == 5:
    s = s.replace("(streaming, three panels)", "(streaming)")
    s = re.sub(r"Two layouts are used\. Videos 6\.2 to 6\.6 \(the streaming runs\) have three panels on one time axis:.*?\(Figure 5\.3\)\. In both, the obstacles",
               "Videos 6.2 to 6.16 use the same five-panel layout: the camera on the left and, on the right, the digital twin (the real arm&rsquo;s measured joint angles replayed in the model) and the simulation of the same episode, each from a top isometric camera and from the side (Figure 5.3). The obstacles", s, flags=re.S)

# ---- segment videos -------------------------------------------------------------------------------------------------
cases = [
    ("s4_test", 9, "hybrid", "Episode 9, hybrid @3000. The same case as Video 6.6, now driven by firmware segments."),
    ("s4_test", 8, "hybrid", "Episode 8, hybrid @3000."),
    ("s4_test", 7, "b2", "Episode 7, B2."),
    ("s4_test", 7, "hybrid", "Episode 7, hybrid @3000: the same scene with a different controller."),
    ("s4_chal", 2, "b2", "Challenge set, gap scene (episode 2), B2."),
    ("s4_chal", 2, "hybrid", "Challenge set, gap scene (episode 2), hybrid @3000."),
    ("s4_chal", 25, "b2", "Challenge set, clutter scene (episode 25), B2."),
    ("s4_chal", 25, "hybrid", "Challenge set, clutter scene (episode 25), hybrid @3000."),
    ("s4_chal", 50, "b2", "Challenge set, limit scene (episode 50), B2."),
    ("s4_chal", 50, "hybrid", "Challenge set, limit scene (episode 50), hybrid @3000."),
    ("s4_chal", 75, "b2", "Challenge set, edge scene (episode 75), B2: the run on which the guard stopped the arm because the measured pose touched a virtual obstacle."),
    ("s4_chal", 75, "hybrid", "Challenge set, edge scene (episode 75), hybrid @3000."),
    ("s4_shape", 0, "b2", "Shape set, box obstacles (episode 0), B2."),
    ("s4_shape", 0, "hybrid", "Shape set, box obstacles (episode 0), hybrid @3000."),
    ("s4_shape", 25, "b2", "Shape set, cylinder obstacles (episode 25), B2."),
    ("s4_shape", 25, "hybrid", "Shape set, cylinder obstacles (episode 25), hybrid @3000."),
    ("s4_shape", 50, "b2", "Shape set, mixed obstacles (episode 50), B2."),
    ("s4_shape", 50, "hybrid", "Shape set, mixed obstacles (episode 50), hybrid @3000."),
]
figs, n = [], 7
for st, ep, m, cap in cases:
    f = MEDIA / f"grid_{st}_ep{ep}_{m}.mp4"
    if not finished(f):
        continue
    name = copy(f)
    figs.append(f'''    <figure id="vid-6-{n}"><div class="frame"><video controls preload="metadata" playsinline src="vid/{name}"></video></div>
      <figcaption><b>Video 6.{n}:</b> {cap}</figcaption></figure>''')
    n += 1
intro = ('''  <p>Videos 6.7 onward show the firmware-segment runs in the five-panel layout: the camera on the left, and on the right the real arm&rsquo;s measured joint angles replayed in the model (top) and the simulation (bottom), each from a top isometric camera and from the side, the direction the real camera looks from. The simulation panels are time-compressed to the length of the real run and say so in their label. The red spheres, boxes and cylinders are the virtual obstacles. Each challenge and shape scene is shown for B2 and for the hybrid, except the edge scene, where the B2 run was stopped by the guard before completing and so has no video.</p>
  <div class="grid2">
''' + "\n".join(figs) + "\n  </div>\n")
a = s.index("<!--SEGVIDEOS-->")
b = s.rindex("<!--/SEGVIDEOS-->") + len("<!--/SEGVIDEOS-->")
s = s[:a] + "<!--SEGVIDEOS-->\n" + intro + "<!--/SEGVIDEOS-->" + s[b:]
p.write_text(s, encoding="utf-8")
print("streaming five-panel:", n_stream, "segment videos:", len(figs))
