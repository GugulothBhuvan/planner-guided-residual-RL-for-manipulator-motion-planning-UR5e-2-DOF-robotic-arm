"""Figure 1.1: deployment chain (development arms -> MyCobot 280). Run: python _make_chain_fig.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTE, TEAL, ORG, PALE = "#1d2433", "#6b7385", "#1f6f8b", "#d9822b", "#eef1f6"
fig, ax = plt.subplots(figsize=(8.2, 3.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
boxes = [
    (1, "Develop", "UR5e and planar\n2-DOF arm\n(simulation)", TEAL),
    (21, "Zero-shot probe", "UR5e residual on\nthe MyCobot model\n(simulation)", TEAL),
    (41, "Retrain", "residual retrained\non the MyCobot\n(simulation)", TEAL),
    (61, "Real arm: streaming", "policy on hardware,\n20 Hz, 0.5 deg\nper tick", ORG),
    (81, "Real arm: segments", "same paths as\nfirmware segments\n(replayed)", ORG),
]
for x, head, body, col in boxes:
    ax.add_patch(FancyBboxPatch((x, 9), 18, 22, boxstyle="round,pad=0.3,rounding_size=1.2", fc=PALE, ec=col, lw=1.6))
    ax.text(x + 9, 27.2, head, ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(x + 9, 17.5, body, ha="center", va="center", fontsize=7.6, color=INK, linespacing=1.35)
for x in (19.3, 39.3, 59.3, 79.3):
    ax.add_patch(FancyArrowPatch((x, 20), (x + 1.7, 20), arrowstyle="-|>", mutation_scale=10, color=INK, lw=1.4))
ax.text(1, 4.2, "development arms", fontsize=8, color=TEAL, fontweight="bold")
ax.text(21, 4.2, "target robot: MyCobot 280", fontsize=8, color=TEAL, fontweight="bold")
ax.plot([21, 79], [6.7, 6.7], color=MUTE, lw=0.8)
ax.text(61, 1.6, "hardware", fontsize=8, color=ORG, fontweight="bold")
fig.tight_layout(pad=0.3)
fig.savefig("img/deployment_chain.png", dpi=190)
