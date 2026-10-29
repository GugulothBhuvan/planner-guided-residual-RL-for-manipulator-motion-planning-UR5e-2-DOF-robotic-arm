"""Two-panel planar-arm view (companion SPEC §10): workspace on the left, C-space on the right.

Matplotlib only, so the same code drives the interactive window (2-dof/scripts/play.py) and offline GIF/MP4
export on Colab. Colors: reference palette (dataviz skill) - blue = policy/arm, orange = base reference,
aqua = A*, neutral grays for structure and obstacles.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from arm2dof.cspace.astar import astar
from arm2dof.cspace.grid import Q1, Q2, grid
from arm2dof.planners.base import goal_config
from arm2dof.planners.ik import ik, joints

ARM, REF, ASTAR, GOAL = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK_2, OBST, OCC = "#0b0b0b", "#52514e", "#b9b8b2", "#d9d8d3"
LINK_RADIUS = 0.03


def split_at_seam(qs):
    """Break a q-trajectory where q1 jumps across +-pi, so the C-space plot draws no false horizontal lines."""
    qs = np.asarray(qs, float).reshape(-1, 2)
    if len(qs) < 2:
        return [qs]
    cut = np.where(np.abs(np.diff(qs[:, 0])) > np.pi)[0] + 1
    return np.split(qs, cut)


def wrap(qs):
    qs = np.array(qs, float).reshape(-1, 2)
    qs[:, 0] = (qs[:, 0] + np.pi) % (2 * np.pi) - np.pi
    return qs


class TwoPanel:
    def __init__(self, env, figsize=(10, 5)):
        self.env = env.unwrapped
        self.fig, (self.ws, self.cs) = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1, 1]))
        self.fig.patch.set_facecolor("white")
        self.show_astar, self.value = False, None  # value: (q1 grid, q2 grid, values) or None
        self.trail_q, self.trail_ee = [], []
        self.occ_key, self.occ = None, None
        self.status = ""

    # ------------------------------------------------------------------ data
    def record(self):
        self.trail_q.append(self.env.q.copy())
        self.trail_ee.append(joints(self.env.q)[1])

    def clear_trail(self):
        self.trail_q, self.trail_ee = [], []

    def _occupancy(self):
        e = self.env
        key = (tuple(map(tuple, np.round(e.centers[e.mask], 6))), tuple(np.round(e.radii[e.mask], 6)))
        if key != self.occ_key:
            self.occ_key, self.occ = key, grid(e.centers[e.mask], e.radii[e.mask])
        return self.occ

    # ------------------------------------------------------------------ drawing
    def draw(self, reference=None):
        e = self.env
        ws, cs = self.ws, self.cs
        ws.clear()
        cs.clear()

        # workspace
        ws.set_aspect("equal")
        ws.set_xlim(-1.0, 1.0)
        ws.set_ylim(-1.0, 1.0)
        ws.set_title("Workspace", loc="left", fontsize=10, color=INK)
        ws.add_patch(Circle((0, 0), 0.9, fill=False, ls=":", lw=0.8, color=OBST))
        for c, r in zip(e.centers[e.mask], e.radii[e.mask]):
            ws.add_patch(Circle(c, r, color=OBST, lw=0))
            ws.add_patch(Circle(c, r + e.cfg.d_safe, fill=False, ls="--", lw=0.8, color=OBST))
        if reference is not None:
            ee_ref = joints(np.asarray(reference))[1]
            ws.plot(ee_ref[:, 0], ee_ref[:, 1], color=REF, lw=1.5, ls="--", label="base reference")
        if self.trail_ee:
            t = np.array(self.trail_ee)
            ws.plot(t[:, 0], t[:, 1], color=ARM, lw=1.2, alpha=0.6, label="EE trail")
        elbow, ee = joints(e.q)
        pts = np.array([[0, 0], elbow, ee])
        ws.plot(pts[:, 0], pts[:, 1], color=ARM, lw=9, solid_capstyle="round", alpha=0.85)
        ws.plot(pts[:2, 0], pts[:2, 1], "o", color="white", mec=ARM, ms=6, mew=1.5)
        ws.plot(*e.goal, marker="*", ms=14, color=GOAL, mec="white", mew=1.0, ls="none", label="goal")
        ws.add_patch(Circle(e.goal, e.eps, fill=False, color=GOAL, lw=0.8))
        ws.tick_params(colors=INK_2, labelsize=8)

        # C-space
        occ = self._occupancy()
        cs.imshow(occ.T, origin="lower", extent=[Q1[0], Q1[-1], Q2[0], Q2[-1]], aspect="auto",
                  cmap=plt.matplotlib.colors.ListedColormap(["white", OCC]), interpolation="nearest")
        if self.value is not None:
            g1, g2, v = self.value
            m = cs.pcolormesh(g1, g2, v.T, cmap="Blues", alpha=0.55, shading="auto")
            m.set_zorder(1)
        if reference is not None:
            for seg in split_at_seam(wrap(reference)):
                cs.plot(seg[:, 0], seg[:, 1], color=REF, lw=1.5, ls="--")
        if self.show_astar:
            q_a, _ = astar(occ, e.q_start, goal_config(e))  # the IK branch the base plans to
            if q_a is not None:
                for seg in split_at_seam(q_a):
                    cs.plot(seg[:, 0], seg[:, 1], color=ASTAR, lw=1.5, label="A*")
        if self.trail_q:
            for seg in split_at_seam(wrap(self.trail_q)):
                cs.plot(seg[:, 0], seg[:, 1], color=ARM, lw=1.5)
        cs.plot(*wrap(e.q_start)[0], "o", mfc="white", mec=INK_2, mew=1.2, ms=7)  # start
        cs.plot(*wrap(e.q)[0], "o", color=ARM, mec="white", mew=1.5, ms=8)
        for q in ik(e.goal):  # every IK branch of the goal: each is a valid C-space target
            cs.plot(*wrap(q)[0], marker="*", ms=12, color=GOAL, mec="white", mew=1.0, ls="none")
        cs.set_xlim(-np.pi, np.pi)
        cs.set_ylim(-0.9 * np.pi, 0.9 * np.pi)
        cs.set_xlabel("q1 (rad, wraps)", color=INK_2, fontsize=9)
        cs.set_ylabel("q2 (rad)", color=INK_2, fontsize=9)
        cs.set_title("C-space", loc="left", fontsize=10, color=INK)
        cs.tick_params(colors=INK_2, labelsize=8)
        self.fig.suptitle(self.status, x=0.99, ha="right", fontsize=9, color=INK_2)
        handles = [plt.Line2D([], [], color=ARM, lw=2, label="policy / arm"),
                   plt.Line2D([], [], color=REF, lw=1.5, ls="--", label="base reference"),
                   plt.Line2D([], [], color=ASTAR, lw=1.5, label="A*"),
                   plt.Line2D([], [], color=GOAL, marker="*", ls="none", label="goal")]
        ws.legend(handles=handles, loc="lower left", fontsize=7, frameon=False)
