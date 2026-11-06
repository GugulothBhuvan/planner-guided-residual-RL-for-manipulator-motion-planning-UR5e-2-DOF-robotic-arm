"""Chapter 3 figures (clean vector-style diagrams, original): run from mycobot/report with the project venv.
  img/th_kinematics.png   MyCobot 280 joint axes (from the URDF chain in mc280/kinematics.py)
  img/th_cspace.png       workspace -> configuration space of a two-link planar arm with one disc obstacle
  img/th_rrtconnect.png   RRT-Connect (two trees, connect step) run on a 2-D configuration space
  img/th_sac.png          Soft Actor-Critic: actor, twin critics, target networks, replay buffer, temperature
  img/th_residual.png     planner + tracker + residual: block diagram of the controller
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc280.kinematics import FIXED, rotz  # noqa: E402

INK, MUTE, TEAL, ORG, PALE, RED = "#1d2433", "#6b7385", "#1f6f8b", "#d9822b", "#eef1f6", "#c0392b"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTE, "ytick.color": MUTE})


def box(ax, x, y, w, h, text, ec=TEAL, fc=PALE, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, p, q, color=INK, rad=0.0, lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------- 1. kinematics
def chain(q):
    T = np.eye(4)
    origins, axes = [T[:3, 3].copy()], []
    for F, qi in zip(FIXED, q):
        T = T @ F
        origins.append(T[:3, 3].copy())
        axes.append(T[:3, 2].copy())
        T = T @ rotz(qi)
    origins.append(T[:3, 3].copy())
    return np.array(origins), np.array(axes)


q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
q_bent = np.deg2rad([0, -40, 60, 30, 20, 0])
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.9))
for ax, qq, title in ((axs[0], q, "all joints at 0 (upright)"), (axs[1], q_bent, "J2 = -40, J3 = 60, J4 = 30, J5 = 20 deg")):
    O, A = chain(qq)
    pts = list(O)
    # draw links through the successive joint origins in the x-z plane
    xs = [p[0] for p in pts]
    zs = [p[2] for p in pts]
    ax.plot(xs, zs, color=MUTE, lw=5, solid_capstyle="round", alpha=0.35)
    for i in range(6):
        o, a = O[i + 1], A[i]
        ax.add_patch(Circle((o[0], o[2]), 0.0085, fc="white", ec=TEAL, lw=1.6, zorder=3))
        ax.annotate("", xy=(o[0] + 0.045 * a[0], o[2] + 0.045 * a[2]), xytext=(o[0], o[2]), arrowprops=dict(arrowstyle="-|>", color=ORG, lw=1.4), zorder=4)
        ax.text(o[0] + (0.03 if i == 1 else (0.0 if i == 5 else -0.03)), o[2] + (-0.02 if i == 1 else (0.04 if i == 5 else 0.012)), f"J{i + 1}", fontsize=8.5, color=TEAL, fontweight="bold", ha="left" if i == 1 else ("center" if i == 5 else "right"))
    ax.plot(O[-1][0], O[-1][2], marker="s", color=INK, ms=5, zorder=5)
    ax.text(O[-1][0] + 0.012, O[-1][2] - 0.012, "flange", fontsize=8, color=INK)
    ax.add_patch(Rectangle((-0.04, -0.012), 0.08, 0.012, fc=INK))
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.03, 0.46)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (m)")
    ax.grid(color="#d9dde6", lw=0.5)
axs[0].set_ylabel("z (m)")
axs[1].legend([plt.Line2D([0], [0], color=ORG, lw=1.4)], ["joint axis (z of the joint frame)"], fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("img/th_kinematics.png", dpi=190)
plt.close(fig)

# ---------------------------------------------------------------- 2. workspace -> configuration space
l1, l2 = 0.5, 0.4
obs_c, obs_r = np.array([0.45, 0.45]), 0.12


def seg_hits_disc(p, q_, c, r):
    d = q_ - p
    t = np.clip(((c - p) @ d) / (d @ d), 0, 1)
    return np.linalg.norm(p + t * d - c) < r


def collides(t1, t2):
    p1 = l1 * np.array([np.cos(t1), np.sin(t1)])
    p2 = p1 + l2 * np.array([np.cos(t1 + t2), np.sin(t1 + t2)])
    return seg_hits_disc(np.zeros(2), p1, obs_c, obs_r) or seg_hits_disc(p1, p2, obs_c, obs_r)


fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.6), gridspec_kw={"width_ratios": [1, 1.05]})
ax = axs[0]
ax.add_patch(Circle(obs_c, obs_r, fc="#f3c9c4", ec=RED, lw=1.5))
for (t1, t2, col) in ((0.15, 0.9, TEAL), (1.0, -0.2, RED)):
    p1 = l1 * np.array([np.cos(t1), np.sin(t1)])
    p2 = p1 + l2 * np.array([np.cos(t1 + t2), np.sin(t1 + t2)])
    ax.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], color=col, lw=3.2, marker="o", ms=5)
ax.set_aspect("equal")
ax.set_xlim(-0.2, 0.95)
ax.set_ylim(-0.2, 0.95)
ax.set_title("workspace: arm and obstacle", fontsize=9)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.text(0.58, 0.07, "free pose", color=TEAL, fontsize=8.5)
ax.text(0.02, 0.8, "pose in contact", color=RED, fontsize=8.5)
ax.grid(color="#d9dde6", lw=0.5)
ax = axs[1]
g = np.linspace(-np.pi, np.pi, 260)
M = np.array([[collides(a, b) for a in g] for b in g])
ax.imshow(M, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi], cmap=matplotlib.colors.ListedColormap(["white", "#f3c9c4"]), aspect="equal")
ax.plot(0.15, 0.9, marker="o", color=TEAL, ms=7)
ax.plot(1.0, -0.2, marker="o", color=RED, ms=7)
ax.set_title("configuration space (joint angles)", fontsize=9)
ax.set_xlabel(r"$\theta_1$ (rad)")
ax.set_ylabel(r"$\theta_2$ (rad)")
ax.text(-3.0, 2.7, "shaded: configurations in contact\n(one disc becomes a curved region)", fontsize=7.5, va="top", color=RED)
fig.tight_layout()
fig.savefig("img/th_cspace.png", dpi=190)
plt.close(fig)

# ---------------------------------------------------------------- 3. RRT-Connect on a 2-D C-space
rng = np.random.default_rng(7)
discs = [(np.array([0.5, y]), 0.075) for y in np.arange(0.0, 0.66, 0.1)] + [(np.array([0.5, 0.97]), 0.07)]


def free(p):
    return all(np.linalg.norm(p - c) > r for c, r in discs) and 0 <= p[0] <= 1 and 0 <= p[1] <= 1


def edge_free(a, b, res=0.01):
    n = max(2, int(np.linalg.norm(b - a) / res))
    return all(free(a + (b - a) * t) for t in np.linspace(0, 1, n))


def nearest(T, p):
    return int(np.argmin([np.linalg.norm(n - p) for n in T[0]]))


def extend(T, p, step=0.08):
    i = nearest(T, p)
    d = p - T[0][i]
    n = np.linalg.norm(d)
    new = p if n <= step else T[0][i] + d / n * step
    if edge_free(T[0][i], new):
        T[0].append(new)
        T[1].append(i)
        return ("reached" if n <= step else "advanced"), new
    return "trapped", None


start, goal = np.array([0.1, 0.12]), np.array([0.9, 0.12])
Ta, Tb = ([start], [-1]), ([goal], [-1])
swap = False
path = None
for it in range(3000):
    qr = rng.uniform(0, 1, 2)
    st, new = extend(Ta, qr)
    if st != "trapped":
        while True:
            st2, new2 = extend(Tb, new)
            if st2 != "advanced":
                break
        if st2 == "reached":
            def trace(T, idx):
                out = []
                while idx != -1:
                    out.append(T[0][idx])
                    idx = T[1][idx]
                return out
            pa, pb = trace(Ta, len(Ta[0]) - 1), trace(Tb, len(Tb[0]) - 1)
            path = (pa[::-1] + pb) if not swap else (pb[::-1] + pa)
            if swap:
                path = path[::-1]
            break
    Ta, Tb = Tb, Ta
    swap = not swap
if swap:
    Ta, Tb = Tb, Ta  # Ta is the tree grown from the start
fig, ax = plt.subplots(figsize=(5.2, 4.6))
for c, r in discs:
    ax.add_patch(Circle(c, r, fc="#f3c9c4", ec=RED, lw=1.2))
for T, col in ((Ta, TEAL), (Tb, ORG)):
    for k, par in enumerate(T[1]):
        if par >= 0:
            ax.plot([T[0][k][0], T[0][par][0]], [T[0][k][1], T[0][par][1]], color=col, lw=0.8, alpha=0.8)
if path is not None:
    P = np.array(path)
    ax.plot(P[:, 0], P[:, 1], color=INK, lw=2.4)
ax.plot(*start, marker="o", color=TEAL, ms=9)
ax.plot(*goal, marker="s", color=ORG, ms=9)
ax.text(0.04, 0.03, "start tree", color=TEAL, fontsize=8.5)
ax.text(0.74, 0.03, "goal tree", color=ORG, fontsize=8.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_aspect("equal")
ax.set_xlabel("joint 1 (normalised)")
ax.set_ylabel("joint 2 (normalised)")
fig.tight_layout()
fig.savefig("img/th_rrtconnect.png", dpi=190)
plt.close(fig)

# ---------------------------------------------------------------- 4. SAC diagram
fig, ax = plt.subplots(figsize=(8.2, 3.7))
ax.set_xlim(0, 100)
ax.set_ylim(0, 48)
ax.axis("off")
box(ax, 1, 24, 15, 10, "Environment\n(MuJoCo arm)", ec=MUTE)
box(ax, 24, 24, 17, 10, "Replay buffer\n(s, a, r, s', g)", ec=MUTE)
box(ax, 49, 24, 16, 10, "Actor\n$\\pi_\\phi(a\\,|\\,s,g)$", ec=TEAL, bold=True)
box(ax, 49, 4, 16, 10, "Twin critics\n$Q_{\\theta_1},\\ Q_{\\theta_2}$", ec=TEAL, bold=True)
box(ax, 74, 4, 24, 10, "Target critics (slow copies)\n$\\bar\\theta \\leftarrow \\tau\\theta + (1-\\tau)\\bar\\theta$", ec=ORG, fs=7.8)
box(ax, 74, 24, 24, 10, "Temperature $\\alpha$\n(tuned to a target entropy)", ec=ORG, fs=7.8)
arrow(ax, (16.6, 29), (23.6, 29))
ax.text(20, 31, "transitions", fontsize=7.5, ha="center", color=MUTE)
arrow(ax, (41.6, 29), (48.6, 29))
ax.text(45, 31, "batch", fontsize=7.5, ha="center", color=MUTE)
arrow(ax, (35, 23.4), (48.6, 10), rad=-0.2)
ax.text(36, 15, "batch", fontsize=7.5, color=MUTE)
arrow(ax, (57, 14.6), (57, 23.4), color=TEAL)
ax.text(58.5, 18.5, "gradient of $Q$ in $a$", fontsize=7.5, color=TEAL)
arrow(ax, (73.6, 9), (65.6, 9), color=ORG)
ax.text(68.5, 11, "$y$", fontsize=8, color=ORG, ha="center")
arrow(ax, (73.6, 29), (65.6, 29), color=ORG)
ax.plot([57, 57, 8.5], [34.6, 42, 42], color=TEAL, lw=1.3)
arrow(ax, (8.5, 42), (8.5, 34.6), color=TEAL)
ax.text(33, 43.6, "actions: the actor acts in the environment", fontsize=7.5, color=TEAL, ha="center")
ax.text(1, 0.4, "Critic target $y = r + \\gamma\\,[\\min_i \\bar Q_i(s',a') - \\alpha \\log \\pi(a'|s')]$.   Actor loss $\\alpha \\log \\pi(a|s) - \\min_i Q_i(s,a)$.", fontsize=8, color=INK)
fig.tight_layout(pad=0.3)
fig.savefig("img/th_sac.png", dpi=190)
plt.close(fig)

# ---------------------------------------------------------------- 5. residual controller block diagram
fig, ax = plt.subplots(figsize=(8.4, 3.8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)
ax.axis("off")
box(ax, 1, 28, 16, 11, "Cheap planner\n(RRT-Connect,\nbudget $N$)", ec=MUTE)
box(ax, 22, 28, 15, 11, "Reference\n$q_{ref}(t)$", ec=MUTE)
box(ax, 42, 28, 16, 11, "Tracker\n(Eq. 3.20)", ec=TEAL, bold=True)
ax.add_patch(Circle((68, 33.5), 2.4, fc="white", ec=INK, lw=1.4))
ax.text(68, 33.5, "+", ha="center", va="center", fontsize=12, fontweight="bold")
box(ax, 75, 28, 11, 11, "clip,\nintegrate", ec=TEAL, fs=8)
box(ax, 88, 28, 11, 11, "Arm\n(servo, lag)", ec=MUTE, fs=8)
box(ax, 42, 6, 16, 11, "Residual\n$\\pi_{res}(s)$ (SAC)", ec=ORG, bold=True)
arrow(ax, (17.6, 33.5), (21.6, 33.5))
arrow(ax, (37.6, 33.5), (41.6, 33.5))
arrow(ax, (58.6, 33.5), (65.5, 33.5))
ax.text(62, 35.6, "$a_{base}$", fontsize=8.5, ha="center")
arrow(ax, (58.6, 11.5), (68, 31), color=ORG, rad=-0.15)
ax.text(66.5, 19, "$\\alpha\\,\\pi_{res}$", fontsize=9, color=ORG)
arrow(ax, (70.5, 33.5), (74.6, 33.5))
ax.text(72.6, 36, "$a$", fontsize=8.5, ha="center")
arrow(ax, (86.6, 33.5), (87.6, 33.5))
ax.plot([93.5, 93.5, 58.6], [27.6, 11.5, 11.5], color=MUTE, lw=1.2)
arrow(ax, (59.5, 11.5), (58.6, 11.5), color=MUTE)
ax.text(76, 8.6, "state $s$: joints, tool pose, $q_{ref}(t), q_{ref}(t+5)$", fontsize=7.5, color=MUTE, ha="center")
ax.text(1, 3.4, "E5 (additive): the residual moves the target by at most $\\alpha\\Delta q_{max}$ around $q_{ref}$ (Eq. 3.23).", fontsize=7.8)
ax.text(1, 0.4, "E5b (offset): the residual integrates a joint offset $o_t$ that is added to $q_{ref}$ (Eq. 3.22).", fontsize=7.8)
fig.tight_layout(pad=0.3)
fig.savefig("img/th_residual.png", dpi=190)
plt.close(fig)
print("theory figures done; rrt path found:", path is not None)
