"""Paper Figure 6 — per-query end-to-end latency, log scale, grouped bars across systems.

Reads `plotdata.xlsx` -> sheet `ablation-latency-system` (columns: query,
system, latency_s) and renders `figures/fig_sys_latency.pdf` / `.png`.

Shares the palette and bar layout with `plot_system_f1.py`.

Run: python3 paper_figures/plot_system_latency.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib.transforms import blended_transform_factory

from plot_style import (
    PALETTE, SYSTEMS, SYS_LABELS,
    apply_paper_style, order_queries, short_q,
)


HERE = Path(__file__).resolve().parent  # paper_figures/: holds plotdata.xlsx, figures/, tables/
PLOTDATA = HERE / "plotdata.xlsx"
OUT_PDF = HERE / "figures" / "fig_sys_latency.pdf"
OUT_PNG = HERE / "figures" / "fig_sys_latency.png"
SHEET = "ablation-latency-system"


def load(sheet: str = SHEET) -> Dict[str, Dict[str, float]]:
    wb = openpyxl.load_workbook(PLOTDATA, data_only=True)
    ws = wb[sheet]
    out: Dict[str, Dict[str, float]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None or row[1] is None:
            continue
        q, s, v = str(row[0]).strip(), str(row[1]).strip(), row[2]
        try:
            val = float(v) if v is not None else None
        except (TypeError, ValueError):
            val = None
        out.setdefault(q, {})[s] = val
    return out


def render() -> None:
    apply_paper_style(bold=True)
    # The wide source PDF is scaled down to two-column text width in LaTeX.
    # Use larger source fonts so the rendered page matches body text size.
    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })
    data = load()
    queries = order_queries(data.keys())
    by_sys = {s: [data.get(q, {}).get(s) for q in queries] for s in SYSTEMS}

    n, k = len(queries), len(SYSTEMS)
    span, pitch = 1.08, 1.7
    w = span / k
    x = np.arange(n) * pitch

    # Wider canvas so the 7-system groups span the full two-column text width.
    fig, ax = plt.subplots(figsize=(20, 4.2))
    for i, s in enumerate(SYSTEMS):
        vals = by_sys[s]
        offs = (i - (k - 1) / 2.0) * w
        # log-scale: clamp non-positive / None to NaN so the bar disappears
        plot_vals = []
        for v in vals:
            if v is None or v <= 0:
                plot_vals.append(np.nan)
            else:
                plot_vals.append(v)
        bars = ax.bar(
            x + offs, plot_vals, width=w * 0.94,
            color=PALETTE[s], label=SYS_LABELS[s],
            edgecolor="#222", linewidth=0.6, zorder=3,
        )
        for b, v in zip(bars, vals):
            if v is None or (isinstance(v, float) and v <= 0):
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                ax.plot([b.get_x() + b.get_width() / 2], [0.0],
                        marker="x", color=PALETTE["Ours"], markersize=8,
                        markeredgewidth=2.0, clip_on=False, transform=trans, zorder=7)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([short_q(q) for q in queries], rotation=24,
                       ha="right", fontweight="bold", fontsize=18)
    ax.set_ylabel("Latency (s)", fontweight="bold")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(ncol=k, loc="upper center", bbox_to_anchor=(0.5, 1.13),
              frameon=False, fontsize=18, handletextpad=0.55,
              columnspacing=1.15)
    sb = sum(1 for q in queries
             if not str(q).lower().startswith(("match", "lro_")))
    if 0 < sb < n:
        ax.axvline(x[sb - 1] + pitch * 0.5,
                   color="#3A3A3A", linestyle=":", linewidth=0.7)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=160)
    print(f"wrote {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    render()
