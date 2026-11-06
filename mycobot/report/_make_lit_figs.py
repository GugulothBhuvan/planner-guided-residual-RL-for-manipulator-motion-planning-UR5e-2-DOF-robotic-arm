"""Chapter 2 figures (original schematics). Run from mycobot/report with the project venv.
  img/lit_taxonomy.png   taxonomy of manipulator motion planning, with this thesis marked
  img/lit_sampling.png   PRM roadmap, RRT tree and the RRT-Connect connect step (schematic, 2-D)
  img/lit_optim.png      trajectory optimisation: the initial guess decides the local minimum
  img/lit_actorcritic.png  actor-critic loop and the lineage DQN -> DDPG -> TD3 -> SAC, PPO
  img/lit_her.png        hindsight relabelling of a failed episode
  img/lit_residual.png   residual on a reactive controller against a residual on a path-tracking base
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

INK, MUTE, TEAL, ORG, PALE, RED, GREEN = "#1d2433", "#6b7385", "#1f6f8b", "#d9822b", "#eef1f6", "#c0392b", "#3b7a57"
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.edgecolor": MUTE})


def box(ax, x, y, w, h, text, ec=TEAL, fc=PALE, fs=8.2, bold=False, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color=INK, rad=0.0, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


# ------------------------------------------------------------------ 1 taxonomy
fig, ax = plt.subplots(figsize=(8.4, 4.3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 54)
ax.axis("off")
box(ax, 33, 46, 34, 6, "Manipulator motion planning and control", ec=INK, bold=True, fs=9)
cols = [(2, "Classical", TEAL), (36, "Learned", TEAL), (70, "Hybrid", ORG)]
for x, t, c in cols:
    box(ax, x, 34, 28, 6, t, ec=c, bold=True, fs=9)
    arrow(ax, (50, 45.5), (x + 14, 40.8), color=MUTE)
items = {
    2: [(24, "Sampling-based\nPRM, RRT, RRT-Connect, RRT*"), (12, "Optimisation-based\nCHOMP, STOMP, TrajOpt"), (0.5, "GPU-parallel\ncuRobo")],
    36: [(24, "Imitation of a planner\nMPNet, Motion Policy Networks"), (12, "Pure deep RL\nreaching and avoidance policies"), (0.5, "Large-scale RL\non real robots")],
    70: [(24, "Learned sampling\nLearning sampling distributions"), (12, "RL as local planner\nPRM-RL (mobile robots)"), (0.5, "Residual on a planner\nthis thesis")],
}
for x, lst in items.items():
    for y, t in lst:
        this = "this thesis" in t
        box(ax, x, y, 28, 8, t, ec=ORG if this else MUTE, fc="#fbeede" if this else "white", fs=7.6, bold=this, lw=2.0 if this else 1.0)
    arrow(ax, (x + 14, 33.6), (x + 14, 32.4), color=MUTE)
ax.text(70, -2.6, "orange: where this thesis sits", fontsize=8, color=ORG, fontweight="bold")
fig.tight_layout(pad=0.3)
fig.savefig("img/lit_taxonomy.png", dpi=190)
plt.close(fig)

# ------------------------------------------------------------------ 2 sampling-based planning
rng = np.random.default_rng(3)
discs = [(np.array([0.36, 0.62]), 0.13), (np.array([0.66, 0.34]), 0.14), (np.array([0.62, 0.8]), 0.08)]


def free(p):
    return all(np.linalg.norm(p - c) > r for c, r in discs) and 0 <= p[0] <= 1 and 0 <= p[1] <= 1


def edge_free(a, b):
    return all(free(a + (b - a) * t) for t in np.linspace(0, 1, max(3, int(np.linalg.norm(b - a) / 0.01))))


def scene(ax, title):
    for c, r in discs:
        ax.add_patch(Circle(c, r, fc="#f3c9c4", ec=RED, lw=1.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.8)


S, G = np.array([0.08, 0.1]), np.array([0.92, 0.9])
fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.3))
# (a) PRM
ax = axs[0]
scene(ax, "(a) PRM: roadmap, then query")
pts = [p for p in rng.uniform(0, 1, (260, 2)) if free(p)][:70]
for i, p in enumerate(pts):
    d = np.linalg.norm(np.array(pts) - p, axis=1)
    for j in np.argsort(d)[1:4]:
        if d[j] < 0.2 and edge_free(p, pts[j]):
            ax.plot(*zip(p, pts[j]), color="#9aa3b5", lw=0.6)
ax.scatter(*zip(*pts), s=5, color=MUTE, zorder=3)
ax.plot(*S, "o", color=TEAL, ms=7, zorder=4)
ax.plot(*G, "s", color=ORG, ms=7, zorder=4)
# (b) RRT
ax = axs[1]
scene(ax, "(b) RRT: one tree from the start")
nodes, par = [S], [-1]
for _ in range(900):
    q = rng.uniform(0, 1, 2) if rng.random() > 0.08 else G
    i = int(np.argmin([np.linalg.norm(n - q) for n in nodes]))
    d = q - nodes[i]
    n = np.linalg.norm(d)
    if n < 1e-6:
        continue
    new = nodes[i] + d / n * min(0.06, n)
    if edge_free(nodes[i], new):
        nodes.append(new)
        par.append(i)
    if np.linalg.norm(nodes[-1] - G) < 0.06:
        break
for k, p in enumerate(par):
    if p >= 0:
        ax.plot(*zip(nodes[k], nodes[p]), color=TEAL, lw=0.7)
ax.plot(*S, "o", color=TEAL, ms=7, zorder=4)
ax.plot(*G, "s", color=ORG, ms=7, zorder=4)
# (c) RRT-Connect schematic
ax = axs[2]
scene(ax, "(c) RRT-Connect: two trees, greedy connect")
Ta = [(S, -1)]
Tb = [(G, -1)]


def grow(T, q, step=0.06):
    i = int(np.argmin([np.linalg.norm(n[0] - q) for n in T]))
    d = q - T[i][0]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    new = T[i][0] + d / n * min(step, n)
    if edge_free(T[i][0], new):
        T.append((new, i))
        return new
    return None


for _ in range(110):
    q = rng.uniform(0, 1, 2)
    grow(Ta, q)
    grow(Tb, q)
for T, col in ((Ta, TEAL), (Tb, ORG)):
    for k, (p, par_) in enumerate(T):
        if par_ >= 0:
            ax.plot(*zip(p, T[par_][0]), color=col, lw=0.7)
pairs = sorted(((np.linalg.norm(p[0] - q[0]), i, j) for i, p in enumerate(Ta) for j, q in enumerate(Tb)), key=lambda x: x[0])
for dist, i, j in pairs:
    if 0.12 < dist < 0.5 and edge_free(Ta[i][0], Tb[j][0]):
        na, cur = Ta[i][0], Tb[j][0]
        ax.plot(*na, marker="*", color=INK, ms=11, zorder=5)
        while np.linalg.norm(na - cur) > 1e-6:
            d = na - cur
            n = np.linalg.norm(d)
            new = cur + d / n * min(0.06, n)
            ax.plot(*zip(cur, new), color=INK, lw=1.8, ls="--")
            cur = new
        break
ax.plot(*S, "o", color=TEAL, ms=7, zorder=4)
ax.plot(*G, "s", color=ORG, ms=7, zorder=4)
ax.text(0.5, 0.02, "dashed: connect steps towards the star", fontsize=7.4, ha="center", color=INK)
fig.tight_layout()
fig.savefig("img/lit_sampling.png", dpi=190)
plt.close(fig)

# ------------------------------------------------------------------ 3 optimisation: initial guess decides the minimum
fig, ax = plt.subplots(figsize=(5.8, 3.6))
ax.add_patch(Circle((0.5, 0.5), 0.17, fc="#f3c9c4", ec=RED, lw=1.3))
t = np.linspace(0, 1, 60)
for off, col, lab in ((0.07, TEAL, "initial guess just above the centre"), (-0.07, ORG, "initial guess just below the centre")):
    x = 0.08 + 0.84 * t
    y0 = 0.1 + 0.8 * t + off * np.sin(np.pi * t)  # initial guess crossing the obstacle
    sign = np.sign(off)
    y1 = 0.1 + 0.8 * t + sign * 0.33 * np.sin(np.pi * t) ** 1.2  # optimised path, same side
    ax.plot(x, y0, color=col, lw=1.2, ls="--")
    ax.plot(x, y1, color=col, lw=2.4, label=lab)
ax.plot(0.08, 0.1, "o", color=INK, ms=7)
ax.plot(0.92, 0.9, "s", color=INK, ms=7)
ax.text(0.5, 0.5, "obstacle", ha="center", va="center", fontsize=8.5, color=RED)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(fontsize=7.6, frameon=False, loc="lower right")
ax.set_title("dashed: initial trajectory; solid: after optimisation", fontsize=8.6)
fig.tight_layout()
fig.savefig("img/lit_optim.png", dpi=190)
plt.close(fig)

# ------------------------------------------------------------------ 4 actor-critic loop and lineage
fig, axs = plt.subplots(2, 1, figsize=(8.4, 5.4), gridspec_kw={"height_ratios": [1.15, 1]})
ax = axs[0]
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")
box(ax, 2, 14, 16, 12, "Environment", ec=MUTE)
box(ax, 38, 26, 18, 10, "Actor\n(policy $\\pi$)", ec=TEAL, bold=True)
box(ax, 38, 3, 18, 10, "Critic\n(value $Q$)", ec=ORG, bold=True)
box(ax, 72, 14, 24, 12, "Update\nactor: follow the critic\ncritic: reduce the TD error", ec=INK, fs=7.8)
arrow(ax, (18.6, 22), (37.6, 30), color=TEAL)
ax.text(24, 29.5, "state $s$", fontsize=8)
arrow(ax, (37.6, 33), (10, 26.8), color=TEAL, rad=-0.3)
ax.text(20, 37, "action $a$", fontsize=8, color=TEAL)
arrow(ax, (18.6, 17), (37.6, 8), color=ORG)
ax.text(22, 9, "reward $r$, next state $s'$", fontsize=8, color=ORG)
arrow(ax, (56.6, 8), (71.6, 16), color=ORG)
ax.text(60, 8, "$Q(s,a)$", fontsize=8, color=ORG)
arrow(ax, (71.6, 24), (56.6, 30), color=INK)
ax.set_title("(a) one actor-critic step", fontsize=9, loc="left")
ax = axs[1]
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
ax.axis("off")
lines = [("DQN (2015)", TEAL, "deep Q-network: experience replay and a target network; discrete actions only"),
         ("DDPG (2015)", TEAL, "deterministic actor-critic for continuous actions; brittle to hyperparameters"),
         ("TD3 (2018)", TEAL, "twin critics, delayed actor updates, smoothed targets: less over-estimation"),
         ("SAC (2018)", TEAL, "maximum entropy, stochastic actor, automatic temperature"),
         ("PPO (2017)", ORG, "on-policy, clipped surrogate: stable, but cannot reuse old data")]
for k, (name, col, txt) in enumerate(lines):
    y = 8.6 - 2.05 * k
    ax.plot([1.5], [y], "o", color=col, ms=7)
    ax.text(4, y, name, fontsize=8.6, fontweight="bold", color=col, va="center")
    ax.text(19, y, txt, fontsize=8, va="center", color=INK)
ax.set_title("(b) lineage of the algorithms used in this thesis (blue: off-policy, orange: on-policy)", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig("img/lit_actorcritic.png", dpi=190)
plt.close(fig)

# ------------------------------------------------------------------ 5 HER
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
tt = np.linspace(0, 1, 40)
path = np.c_[0.1 + 0.55 * tt, 0.15 + 0.45 * np.sin(1.4 * tt) + 0.05 * np.sin(8 * tt)]
for ax, title in zip(axs, ("(a) as collected: the episode fails", "(b) after relabelling with a later achieved goal")):
    ax.plot(path[:, 0], path[:, 1], color=MUTE, lw=2)
    ax.plot(*path[0], "o", color=INK, ms=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.8)
axs[0].plot(0.85, 0.82, "*", color=RED, ms=14)
axs[0].text(0.85, 0.7, "goal $g$\n(not reached)", ha="center", fontsize=8, color=RED)
axs[0].plot(*path[-1], "s", color=MUTE, ms=7)
axs[0].text(path[-1][0] + 0.03, path[-1][1] - 0.06, "ends here", fontsize=8, color=MUTE)
axs[0].text(0.03, 0.9, "reward $-1$ on every step: no signal", fontsize=8, color=RED)
for k, col in ((22, TEAL), (33, GREEN), (39, ORG)):
    axs[1].plot(*path[k], "*", color=col, ms=13)
axs[1].text(path[22][0] - 0.02, path[22][1] + 0.09, "$g'$ (k = 4 later\nachieved goals)", fontsize=7.8, color=TEAL, ha="center")
axs[1].text(0.03, 0.9, "same steps, goal replaced: reward $0$ at the step\nwhere the new goal is achieved", fontsize=8, color=GREEN, va="top")
fig.tight_layout()
fig.savefig("img/lit_her.png", dpi=190)
plt.close(fig)

# ------------------------------------------------------------------ 6 residual: reactive base against path-tracking base
fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.9))
for ax in axs:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
ax = axs[0]
ax.set_title("(a) residual on a reactive controller (prior work)", fontsize=8.8)
box(ax, 4, 38, 30, 10, "Base controller\n$\\pi_{base}(s)$ reacts to the state", ec=MUTE, fs=7.6)
box(ax, 4, 6, 30, 10, "Residual\n$\\pi_{res}(s)$", ec=ORG, bold=True, fs=8)
ax.add_patch(Circle((50, 32), 3, fc="white", ec=INK, lw=1.3))
ax.text(50, 32, "+", ha="center", va="center", fontsize=12, fontweight="bold")
box(ax, 64, 26, 30, 12, "Robot, state $s$", ec=MUTE, fs=8)
arrow(ax, (34.6, 43), (49, 34.5))
arrow(ax, (34.6, 11), (49, 29.5), color=ORG)
arrow(ax, (53.2, 32), (63.6, 32))
ax.plot([79, 79, 19, 19], [25.6, 22, 22, 17], color=MUTE, lw=1.2)
arrow(ax, (19, 17), (19, 16.4), color=MUTE)
ax.plot([79, 79, 19], [38.6, 52, 52], color=MUTE, lw=1.2)
arrow(ax, (19, 52), (19, 48.6), color=MUTE)
ax.text(50, 53.5, "both see the new state", fontsize=7.4, color=MUTE, ha="center")
ax.text(50, 1.5, "a correction changes the state, the base reacts to the new\nstate, so the corrected behaviour persists", fontsize=7.8, ha="center")
ax = axs[1]
ax.set_title("(b) residual on a path-tracking base (this thesis)", fontsize=8.8)
box(ax, 4, 38, 30, 10, "Reference $q_{ref}(t)$ from the\nplanner, indexed by time", ec=MUTE, fs=7.6)
box(ax, 40, 38, 24, 10, "Tracker re-aims at\n$q_{ref}(t+1)$", ec=TEAL, bold=True, fs=7.6)
box(ax, 4, 6, 30, 10, "Residual\n$\\pi_{res}(s)$", ec=ORG, bold=True, fs=8)
ax.add_patch(Circle((78, 32), 3, fc="white", ec=INK, lw=1.3))
ax.text(78, 32, "+", ha="center", va="center", fontsize=12, fontweight="bold")
box(ax, 70, 6, 26, 10, "Robot, joint\ntarget", ec=MUTE, fs=8)
arrow(ax, (32.6, 43), (37.6, 43))
arrow(ax, (68.6, 43), (78, 35.2))
arrow(ax, (34.6, 11), (76, 29.5), color=ORG)
arrow(ax, (78, 28.6), (78, 16.6))
ax.text(52, 29, "next step: the tracker aims\nat the reference again", fontsize=7.4, color=TEAL, ha="center")
ax.text(50, 1.5, "the tracker erases the previous correction: its authority is\nbounded by $\\alpha\\Delta q_{max}$ (Eq. 3.23)", fontsize=7.8, ha="center")
fig.tight_layout()
fig.savefig("img/lit_residual.png", dpi=190)
plt.close(fig)
print("literature figures done")
