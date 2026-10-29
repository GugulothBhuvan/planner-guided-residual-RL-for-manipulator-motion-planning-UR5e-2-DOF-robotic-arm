"""E5 analysis figure (SPEC §9.2): validation success and collision vs the residual scale alpha, per residual arm, at
the primary and final checkpoints (scripts/alpha_sweep.py output). alpha = 0 is the base alone. Seeds are averaged;
per-seed values are drawn as small dots.

    python scripts/plot_alpha_sweep.py   ->  results/figures/e5_alpha_sweep.{pdf,png,csv}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from plot_budget import ARMS, GRID, INK, INK_2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def collect():
    """{(arm, checkpoint): {alpha: {run: row}}}"""
    out: dict = {}
    files = [f for exp in ("e5", "e5b") for f in (ROOT / "results" / "runs" / exp).glob("residual*_s*/alpha_sweep_*.json")]
    for f in sorted(files):
        ckpt = f.stem.removeprefix("alpha_sweep_")
        arm = re.sub(r"_[0-9a-f]{8}_s\d+$", "", f.parent.name)
        for row in json.loads(f.read_text(encoding="utf-8")):
            out.setdefault((arm, ckpt), {}).setdefault(row["alpha"], {})[f.parent.name] = row
    return out


def main():
    data = collect()
    out = ROOT / "results" / "figures" / "e5_alpha_sweep"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    table = []
    for ax, key, title in ((axes[0], "success", "Validation success"), (axes[1], "collision", "Validation collision")):
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for (arm, ckpt), by_alpha in sorted(data.items(), key=lambda kv: (kv[0][0], kv[0][1] == "final")):
            label, color, marker = ARMS.get(arm, (f"base + {arm}", INK_2, "D"))
            label = label[:-1] + (", final 1M)" if ckpt == "final" else ", primary)")
            alphas = sorted(by_alpha)
            means = [np.mean([r[key] for r in by_alpha[a].values()]) for a in alphas]
            ax.plot(alphas, means, color=color, lw=2, ls="--" if ckpt == "final" else "-", marker=marker, ms=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=label)
            for a in alphas:
                vals = [r[key] for r in by_alpha[a].values()]
                ax.plot([a] * len(vals), vals, ls="none", marker="o", ms=3, color=color, alpha=0.5)
                table.append((arm, ckpt, a, key, float(np.mean(vals)), len(vals)))
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.set_xlabel("Residual scale α (0 = base alone)")
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1 if key == "success" else 0.05))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylim(top=1.02)
    axes[1].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncols=2, frameon=False, fontsize=8)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    lines = ["arm,checkpoint,alpha,metric,mean_over_seeds,n_seeds"] + [",".join(map(str, row)) for row in table]
    out.with_suffix(".csv").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
