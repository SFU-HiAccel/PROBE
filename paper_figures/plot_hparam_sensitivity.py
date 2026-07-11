"""Hyperparameter sensitivity figure.

Reads `plotdata.xlsx` -> sheet `hyper-param` and, for five representative
queries, draws a 5x3 grid of panels. Each row is one query; each column is one
hyperparameter (sampling rate, k-NN neighbor count k, explore rate rho). Within
a panel, F1 (left axis, solid blue) and end-to-end latency (right axis, dashed
red) are plotted against the swept hyperparameter value.

Run: python3 plot_hparam_sensitivity.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import openpyxl

from plot_style import apply_paper_style, short_q

HERE = Path(__file__).resolve().parent
PLOTDATA = HERE / "plotdata.xlsx"
OUT_PDF = HERE / "figures" / "fig_hparam.pdf"
OUT_PNG = HERE / "figures" / "fig_hparam.png"

QUERIES = ["mmqa_q7", "ecomm_q8", "ecomm_q9", "animals_q1", "match305"]

# Large-table queries run at a 0.1%-1% sampling budget; their Sampling-Rate
# x-axis is the 1%-10% swept index scaled by 0.1 (F1/latency datapoints
# unchanged, axis relabelled). animals_q1 and match305 stay on 1%-10%.
LARGE_HPARAM = {"mmqa_q7", "ecomm_q8", "ecomm_q9"}

# Per-query latency overrides for cells known to be corrupted in the
# `hyper-param` sheet. Empty: all cells are currently clean in the sheet.
LAT_FIX = {}

# (column title, x-values, sheet-column prefixes, x-ticks)
# Concise column headers; the bottom-row x-labels carry the units/symbols, so
# the headers stay short enough to fit one panel width without colliding.
# The sampling-rate sweep plots all ten 1%..10% points but ticks only a
# sparse subset so the narrow panels stay readable.
PARAMS = [
    ("Sampling Rate", list(range(1, 11)),
     [f"s{i}" for i in range(1, 11)], [1, 3, 5, 7, 9]),
    ("$k$",            [10, 30, 50],    ["k10", "k30", "k50"], [10, 30, 50]),
    ("Explore Rate",   [0.3, 0.5, 0.7], ["rho0.3", "rho0.5", "rho0.7"],
     [0.3, 0.5, 0.7]),
]

F1_COLOR = "#3B6EA5"   # blue
LAT_COLOR = "#B85C5C"  # red


def load():
    wb = openpyxl.load_workbook(PLOTDATA, data_only=True)
    ws = wb["hyper-param"]
    head = [c.value for c in ws[1]]
    idx = {h: i for i, h in enumerate(head)}
    rows = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is not None and not str(r[0]).startswith("AVERAGE"):
            rows[str(r[0])] = r
    return rows, idx


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row_latrange(rec, idx, fix=None):
    """Shared latency [lo, hi] for one query across all three hyperparameters."""
    fix = fix or {}
    vals = []
    for _, _, pref, _ in PARAMS:
        vals += [v for v in (fix.get(p, _num(rec[idx[p + "_lat"]])) for p in pref)
                 if v is not None]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.30 or max(hi * 0.1, 0.1)
    return lo - pad, hi + pad


def render():
    apply_paper_style(bold=True)
    plt.rcParams.update({"axes.spines.top": False})
    rows, idx = load()

    nr, nc = len(QUERIES), len(PARAMS)
    # Single-column ("half the paper width") compact grid. Share the x-axis per
    # column (only the bottom row is labeled) and the F1 axis across all panels
    # (only the left column is labeled) so the grid takes minimal space.
    # Wider, shorter canvas: wider figsize than tall so each panel is landscape
    # (not the old narrow portrait), and the whole figure renders to only the
    # top-left quarter of the page when placed at \columnwidth.
    # Fonts are sized so that after the figure is scaled to \columnwidth
    # (~3.5in, a 0.62x shrink from this 5.6in canvas), text renders at ~9pt,
    # matching the paper body size.
    # sharex=False so the Sampling-Rate column (col 0) can carry a different
    # x-range per row (0.1%-1% for large queries, 1%-10% otherwise); the k and
    # rho columns are kept visually shared by hiding their inner-row x labels.
    fig, axes = plt.subplots(nr, nc, figsize=(8.4, 6.8), squeeze=False,
                             sharex=False, sharey=True)

    for i, q in enumerate(QUERIES):
        rec = rows[q]
        fix = LAT_FIX.get(q, {})
        latrange = _row_latrange(rec, idx, fix)
        for j, (title, xs, pref, ticks) in enumerate(PARAMS):
            ax = axes[i][j]
            f1 = [_num(rec[idx[p + "_f1"]]) for p in pref]
            lat = [fix.get(p, _num(rec[idx[p + "_lat"]])) for p in pref]

            # column 0 = sampling rate; large queries are relabelled to 0.1%-1%.
            scale = 0.1 if (j == 0 and q in LARGE_HPARAM) else 1.0
            xs_p = [x * scale for x in xs]
            ticks_p = [t * scale for t in ticks]

            ax.plot(xs_p, f1, color=F1_COLOR, marker="o", markersize=4.5,
                    linewidth=1.5, zorder=3)
            ax.set_ylim(-0.08, 1.08)
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(axis="both", labelsize=12)
            ax.set_xticks(ticks_p)
            # sampling-rate column shows its own x per row; k/rho columns keep
            # one shared axis shown only on the bottom row.
            if j != 0 and i != nr - 1:
                ax.tick_params(labelbottom=False)

            if all(l is not None for l in lat) and latrange is not None:
                ax2 = ax.twinx()
                ax2.plot(xs_p, lat, color=LAT_COLOR, marker="s", markersize=4.2,
                         linewidth=1.4, linestyle=(0, (3, 2)), zorder=2)
                ax2.set_ylim(*latrange)
                ax2.spines["top"].set_visible(False)
                if j == nc - 1:
                    ax2.tick_params(axis="y", labelsize=11.5, colors=LAT_COLOR,
                                    pad=1)
                    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
                else:
                    ax2.set_yticklabels([])
                    ax2.tick_params(axis="y", length=0)

            if i == 0:
                ax.set_title(title, fontsize=13, fontweight="bold", pad=4)
            if i == nr - 1:
                xl = {0: "rate (%)", 1: "$k$", 2: r"$\rho$"}[j]
                ax.set_xlabel(xl, fontsize=13, labelpad=1)
            if j == 0:
                # Push the rotated query name left of the y-tick numbers so the
                # two never overlap; annotation_clip=False keeps it visible even
                # though it sits outside the axes box.
                ax.annotate(short_q(q), xy=(-0.42, 0.5), xycoords="axes fraction",
                            ha="center", va="center", rotation=90,
                            fontsize=12, fontweight="bold", annotation_clip=False)

    # Axis-quantity labels hug the plot rather than the page edges.
    fig.text(0.022, 0.5, r"F$_1$", rotation=90, va="center", ha="center",
             color=F1_COLOR, fontweight="bold", fontsize=14)
    fig.text(0.965, 0.5, "Latency (s)", rotation=270, va="center", ha="center",
             color=LAT_COLOR, fontweight="bold", fontsize=14)

    handles = [
        Line2D([0], [0], color=F1_COLOR, marker="o", markersize=6, linewidth=1.5,
               label=r"F$_1$ (left)"),
        Line2D([0], [0], color=LAT_COLOR, marker="s", markersize=5.5, linewidth=1.4,
               linestyle=(0, (3, 2)), label="Latency (right)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False, fontsize=14, handlelength=1.6,
               columnspacing=1.2)

    fig.subplots_adjust(hspace=0.30, wspace=0.12)
    # Tight left/right bands keep the F$_1$ and Latency labels close to the plot
    # while still leaving room for the rotated query names and the tick numbers.
    fig.tight_layout(rect=[0.075, 0, 0.93, 0.96], h_pad=0.8, w_pad=0.3)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=160)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    render()
