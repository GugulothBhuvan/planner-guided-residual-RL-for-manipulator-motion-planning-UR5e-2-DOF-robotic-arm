"""Figure for Section 6.4: the E5b offset |o| against the step, failures against successes (data from e5b_offset_trace.py).

    python scripts/plot_offset_trace.py results/figures/e5b_offset_trace.json results/figures/e5b_offset_trace.png
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTE = "#1d2433", "#6b7385"
OK, BAD = "#1f6f8b", "#d9822b"
LIMIT = 0.30


def curves(rows, min_n=8):
    n = max(len(r["o"]) for r in rows)
    m = np.full((len(rows), n), np.nan)
    for i, r in enumerate(rows):
        m[i, : len(r["o"])] = r["o"]
    keep = (~np.isnan(m)).sum(0) >= min_n
    last = int(np.nonzero(keep)[0].max()) + 1 if keep.any() else 0
    m = m[:, :last]
    return (np.nanmedian(m, 0), np.nanpercentile(m, 10, 0), np.nanpercentile(m, 90, 0))


def main(src, dst):
    data = json.load(open(src))
    rows = [r for seed in data.values() for r in seed]
    ok = [r for r in rows if r["success"]]
    bad = [r for r in rows if not r["success"]]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for grp, col, lab in ((ok, OK, f"successful episodes (n = {len(ok)})"), (bad, BAD, f"failed episodes (n = {len(bad)})")):
        med, lo, hi = curves(grp)
        t = np.arange(len(med))
        ax.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(t, med, color=col, lw=2, label=lab)
    ax.axhline(LIMIT, color=MUTE, lw=1, ls="--")
    ax.text(2, LIMIT + 0.006, "offset limit 0.30 rad", fontsize=8.5, color=MUTE, va="bottom")
    ax.set_xlabel("control step")
    ax.set_ylabel("largest joint offset |o| (rad)")
    ax.set_ylim(0, 0.34)
    ax.grid(axis="y", color="#d9dde6", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(dst, dpi=170)
    print("n episodes", len(rows), "ok", len(ok), "bad", len(bad))
    for name, grp in (("success", ok), ("failure", bad)):
        end = [r["o"][-1] for r in grp]
        print(name, "median |o| at the last step: %.3f; share at >= 0.27: %.2f" % (np.median(end), np.mean(np.array(end) >= 0.27)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
