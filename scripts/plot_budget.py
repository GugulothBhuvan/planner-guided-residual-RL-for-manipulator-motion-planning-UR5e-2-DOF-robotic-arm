"""E5 primary figure (SPEC §3.1, §7.3): success and collision rate vs planning budget.

Base-only and base + residual (hybrid) at each budget, full B2 as the reference line, 95% bootstrap CIs.
Reads the per-episode rows that scripts/evaluate.py wrote to results/eval/<set>/. Hybrid seeds are pooled per arm
(every episode of every seed, bootstrapped together); per-seed means are drawn as small dots. Arms are kept apart:
the dense residual and, if it ran, the sparse residual (SPEC §7.3) each get their own line. Each residual arm is drawn
at its primary checkpoint (solid) and its final 1M checkpoint (dashed, same colour); SPEC §9.2 checkpoint rule.

    python scripts/plot_budget.py --set s4_test
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# reference palette (dataviz skill): categorical slots 1-2, text and grid inks
HYBRID, BASE, HYBRID_SPARSE = "#2a78d6", "#eb6834", "#1baf7a"
ARMS = {"residual": ("base + residual (dense)", HYBRID, "o"), "residual_sparse": ("base + residual (sparse)", HYBRID_SPARSE, "^"),
        "residual_offset": ("base + offset residual (E5b)", "#8a5cd6", "D")}
INK, INK_2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#7a7975"


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    means = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def collect(d):
    """{"base": {N: rows}, "hybrid": {(arm, checkpoint): {N: {run: rows}}}, "b2": rows}"""
    out: dict = {"base": {}, "hybrid": {}, "b2": None}
    for f in sorted(d.glob("*.jsonl")):
        if f.stem == "b2":
            out["b2"] = load_rows(f)
        elif m := re.fullmatch(r"base_(\d+)", f.stem):
            out["base"][int(m[1])] = load_rows(f)
        elif m := re.fullmatch(r"hybrid_(\d+)__(.+?_[0-9a-f]{8}_s\d+)(?:__(primary|final))?", f.stem):
            arm = re.sub(r"_[0-9a-f]{8}_s\d+$", "", m[2])
            out["hybrid"].setdefault((arm, m[3] or "best"), {}).setdefault(int(m[1]), {})[m[2]] = load_rows(f)
    return out


def place_end_labels(ax, labels, key):
    """Direct labels at each line's right end, spread apart vertically so close lines stay readable."""
    if not labels:
        return
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)
    labels.sort(key=lambda t: t[0])
    ys = [t[0] for t in labels]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (y0, label, x), y in zip(labels, ys):
        ax.annotate(label, (x, y0), xytext=(x * 1.08, y), textcoords="data", va="center", fontsize=8, color=INK_2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="s4_test")
    p.add_argument("--out", default=None, help="output stem (default results/figures/e5_budget_<set>)")
    args = p.parse_args()
    data = collect(ROOT / "results" / "eval" / args.set)
    out = Path(args.out) if args.out else ROOT / "results" / "figures" / f"e5_budget_{args.set}"
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    table = []
    for ax, key, title in ((axes[0], "success", "Success rate"), (axes[1], "collision", "Collision rate")):
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        end_labels = []
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if data["b2"]:
            b2_mean, lo, hi = ci([r[key] for r in data["b2"]])
            ax.axhspan(lo, hi, color=REF, alpha=0.12, lw=0)
            ax.axhline(b2_mean, color=REF, lw=1.5, ls="--")
            ax.annotate("full B2", (1, b2_mean), xycoords=("axes fraction", "data"), xytext=(-2, 4),
                        textcoords="offset points", ha="right", fontsize=8, color=INK_2)
            table.append(("full B2", "-", key, b2_mean, lo, hi, len(data["b2"])))
        series = [("base-only", BASE, "s", "-", {n: rows for n, rows in data["base"].items()}, None)]
        for (arm, ckpt), by_n in sorted(data["hybrid"].items(), key=lambda kv: (kv[0][0], kv[0][1] == "final")):
            label, color, marker = ARMS.get(arm, (f"base + {arm}", INK_2, "D"))
            label = label[:-1] + ", final 1M)" if ckpt == "final" else label
            series.append((label, color, marker, "--" if ckpt == "final" else "-",
                           {n: [r for rows in runs.values() for r in rows] for n, runs in by_n.items()}, by_n))
        for label, color, marker, ls, by_budget, per_seed in series:
            if not by_budget:
                continue
            budgets = sorted(by_budget)
            stats = np.array([ci([r[key] for r in by_budget[n]]) for n in budgets])
            ax.fill_between(budgets, stats[:, 1], stats[:, 2], color=color, alpha=0.15, lw=0)
            ax.plot(budgets, stats[:, 0], color=color, lw=2, ls=ls, marker=marker, ms=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=label)
            table += [(label, n, key, *s, len(by_budget[n])) for n, s in zip(budgets, stats)]
            if per_seed is not None:
                for n in budgets:  # per-seed means
                    seeds = [np.mean([r[key] for r in rows]) for rows in per_seed[n].values()]
                    ax.plot([n] * len(seeds), seeds, ls="none", marker="o", ms=3, color=color, alpha=0.5)
            end_labels.append([stats[-1, 0], label, budgets[-1]])
        if key == "success":  # the collision lines overlap; the legend identifies them there
            place_end_labels(ax, end_labels, key)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1500, 3000, 6000], ["1500", "3000", "6000"])
        ax.set_xlim(1200, 11000)
        ax.set_xlabel("Planning budget (collision checks)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylim(top=1.02)
    axes[1].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([], [], color=REF, lw=1.5, ls="--"))
    labels.append("full B2 (reference)")
    fig.legend(handles, labels, loc="outside upper right", ncols=3, frameon=False, fontsize=8)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    # table view of the same numbers
    lines = ["method,budget,metric,mean,ci_lo,ci_hi,n"] + [",".join(map(str, row)) for row in table]
    out.with_suffix(".csv").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
