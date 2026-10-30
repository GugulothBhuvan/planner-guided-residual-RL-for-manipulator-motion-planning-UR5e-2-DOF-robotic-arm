"""Thesis figures not produced elsewhere (plot_budget.py and plot_alpha_sweep.py make the E5 budget and alpha-sweep
figures). Same style and palette as plot_budget.py. All inputs are in the repository (val.jsonl, evaluation rows).

  learning_curves_e1_e2   validation success vs steps, E1 (SAC vs TD3 + HER) and E2 (sparse + HER vs dense)
  e5_validation_curves    S4 validation success and collision vs steps for the three residual arms, base line 0.88
  e5_outcome_split        S4 test outcome per arm and budget: success / collision / no contact but not at the goal
  pe34_budget             2-DOF success vs planning budget, PE3 (P2) and PE4 (P3) incl. the PE4b offset residual
  pe8_plants              2-DOF torque control: success on the exact and +-20% mass plants

    python scripts/thesis_figures.py   ->  results/figures/<name>.pdf / .png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS, EVAL, FIG = ROOT / "results" / "runs", ROOT / "results" / "eval", ROOT / "results" / "figures"
BLUE, ORANGE, GREEN, PURPLE, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#e34948"
INK, INK_2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#7a7975"
ACTOR_START = 202_000  # E5/E5b: critic warm-up 50k gradient steps at 0.25 per transition + 2k learning starts


def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                         "ytick.color": INK_2, "text.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False})


def axis(ax, title, ylabel=None, pct=True):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    if ylabel:
        ax.set_ylabel(ylabel)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote results/figures/{name}.pdf/.png")


def curves(exp, prefix, key="success"):
    """Per-seed validation curves of the runs of `exp` named <prefix>_<hash>_s<seed> -> (steps, seeds x steps)."""
    runs = sorted(d for d in (RUNS / exp).glob(f"{prefix}_*_s*") if re.fullmatch(rf"{prefix}_[0-9a-f]{{8}}_s\d+", d.name))
    per = []
    for d in runs:
        rows = {}
        for line in (d / "val.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["num_timesteps"]] = r[key]  # a resumed run repeats steps: keep the last value
        per.append(rows)
    steps = sorted(set.intersection(*(set(p) for p in per)))
    return np.array(steps), np.array([[p[s] for s in steps] for p in per])


def plot_curves(ax, steps, ys, color, label):
    for y in ys:
        ax.plot(steps, y, color=color, lw=0.8, alpha=0.35)
    ax.plot(steps, ys.mean(0), color=color, lw=2, label=f"{label} ({len(ys)} seeds)")


def learning_curves_e1_e2():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True, sharey=True)
    for ax, title, arms in ((axes[0], "E1, S1 position reaching", (("e1", "sac", "SAC + HER", BLUE), ("e1", "td3", "TD3 + HER", ORANGE))),
                            (axes[1], "E2, S2 pose reaching", (("e2", "sac_sparse_her", "sparse + HER", BLUE), ("e2", "sac_dense", "dense", ORANGE)))):
        for exp, prefix, label, color in arms:
            plot_curves(ax, *curves(exp, prefix), color, label)
        axis(ax, title, "validation success" if ax is axes[0] else None)
        ax.set_xlabel("environment steps")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "learning_curves_e1_e2")


def e5_validation_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    arms = (("e5", "residual", "E5 dense residual", BLUE), ("e5", "residual_sparse", "E5 sparse residual", GREEN),
            ("e5b", "residual_offset", "E5b offset residual", PURPLE))
    for ax, key, title in ((axes[0], "success", "S4 validation success"), (axes[1], "collision", "S4 validation collision")):
        ax.axvspan(0, ACTOR_START, color=REF, alpha=0.08, lw=0)
        ax.text(ACTOR_START / 2, 0.02 if key == "success" else 0.92, "actor frozen\n(critic warm-up)", ha="center",
                fontsize=7, color=INK_2, transform=ax.get_xaxis_transform())
        ax.axhline(0.88 if key == "success" else 0.12, color=ORANGE, lw=1.5, ls="--", label="base alone (validation)")
        for exp, prefix, label, color in arms:
            plot_curves(ax, *curves(exp, prefix, key), color, label)
        axis(ax, title)
        ax.set_xlabel("environment steps")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=2, frameon=False, fontsize=8)
    save(fig, "e5_validation_curves")


def rows(folder, pattern):
    out = []
    for f in sorted(folder.glob("*.jsonl")):
        if re.fullmatch(pattern, f.stem):
            out += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def e5_outcome_split():
    s4 = EVAL / "s4_test"
    arms = (("planner alone", r"base_{b}"), ("E5 dense", r"hybrid_{b}__residual_[0-9a-f]{{8}}_s\d__primary"),
            ("E5 sparse", r"hybrid_{b}__residual_sparse_[0-9a-f]{{8}}_s\d__primary"),
            ("E5b offset", r"hybrid_{b}__residual_offset_[0-9a-f]{{8}}_s\d__primary"))
    budgets = (1500, 3000, 6000)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True, sharey=True)
    for ax, b in zip(axes, budgets):
        for k, (label, pat) in enumerate(arms):
            r = rows(s4, pat.format(b=b))
            succ = np.mean([x["success"] for x in r])
            coll = np.mean([x["collision"] for x in r])
            other = 1 - succ - coll
            left = 0.0
            for v, color in ((succ, BLUE), (coll, RED), (other, ORANGE)):
                ax.barh(k, v, left=left, color=color, height=0.62, edgecolor="white", linewidth=1.5)
                if v >= 0.06:
                    ax.text(left + v / 2, k, f"{100 * v:.0f}", ha="center", va="center", fontsize=7, color="white")
                left += v
        ax.set_yticks(range(len(arms)), [a[0] for a in arms])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(f"{b} checks", loc="left", fontsize=10, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in
               ((BLUE, "success"), (RED, "collision"), (ORANGE, "no contact, not at the goal (pose miss or timeout)"))]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=8)
    save(fig, "e5_outcome_split")


def pe34_budget():
    p = EVAL / "2dof"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True, sharey=True)
    for ax, title, folder, budgets in ((axes[0], "PE3, P2 static clutter", p / "p2_test", (4000, 6400, 9000)),
                                       (axes[1], "PE4 / PE4b, P3 narrow passage", p / "p3_test", (700, 2300, 3500))):
        b2 = np.mean([x["success"] for x in rows(folder, "b2")])
        ax.axhline(b2, color=REF, lw=1.5, ls="--")
        ax.annotate("full B2", (budgets[-1], b2), xytext=(0, 3), textcoords="offset points", ha="right", fontsize=8, color=INK_2)
        series = [("planner alone", "base_{b}", ORANGE, "s"),
                  ("additive residual", r"hybrid_{b}__residual_[0-9a-f]{{8}}_s\d__primary", BLUE, "o")]
        if "p3" in folder.name:
            series.append(("offset residual (PE4b)", r"hybrid_{b}__residual_offset_[0-9a-f]{{8}}_s\d__primary", PURPLE, "D"))
        for label, pat, color, marker in series:
            means, seeds = [], []
            for b in budgets:
                files = [f for f in folder.glob("*.jsonl") if re.fullmatch(pat.format(b=b), f.stem)]
                per = [np.mean([json.loads(l)["success"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()])
                       for f in files]
                means.append(np.mean(per))
                seeds.append(per)
            ax.plot(budgets, means, color=color, lw=2, marker=marker, ms=6, mec="white", mew=1.5, label=label)
            for b, s in zip(budgets, seeds):
                if len(s) > 1:
                    ax.plot([b] * len(s), s, ls="none", marker="o", ms=3, color=color, alpha=0.5)
        axis(ax, title, "test success" if ax is axes[0] else None)
        ax.set_xticks(budgets)
        ax.set_xlabel("planning budget (collision checks)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylim(0.6, 1.02)
    save(fig, "pe34_budget")


def pe8_plants():
    folder = EVAL / "2dof" / "p5a_test"
    methods = (("B5 computed torque", r"b5__-__m{m}", REF), ("B5' PD + gravity", r"b5p__-__m{m}", ORANGE),
               ("residual on B5'", r"residual_[0-9a-f]{{8}}_s\d__primary__m{m}", BLUE),
               ("pure SAC+HER", r"sac_her_[0-9a-f]{{8}}_s\d__best__m{m}", GREEN))
    plants = (1.0, 0.8, 1.2)
    fig, ax = plt.subplots(figsize=(7.2, 2.7), constrained_layout=True)
    w = 0.2
    for k, (label, pat, color) in enumerate(methods):
        vals = []
        for m in plants:
            files = [f for f in folder.glob("*.jsonl") if re.fullmatch(pat.format(m=m), f.stem)]
            vals.append(np.mean([np.mean([json.loads(l)["success"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()])
                                 for f in files]))
        x = np.arange(len(plants)) + (k - 1.5) * w
        ax.bar(x, vals, width=w - 0.02, color=color, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.015, f"{100 * v:.0f}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(plants)), ["exact mass", "mass x0.8", "mass x1.2"])
    axis(ax, "PE8 torque reaching: every controller keeps the nominal model", "test success")
    ax.set_ylim(0, 1.1)
    fig.legend(loc="outside upper center", frameon=False, fontsize=8, ncol=4)
    save(fig, "pe8_plants")


def main():
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    learning_curves_e1_e2()
    e5_validation_curves()
    e5_outcome_split()
    pe34_budget()
    pe8_plants()


if __name__ == "__main__":
    main()
