"""Paper Figure 5 — per-query F1 grouped bars across systems.

Reads `plotdata.xlsx` -> sheet `ablation-f1-system` (columns: query, system,
F1) and renders `figures/fig_sys_f1.pdf` / `.png`.

Run from the repo root:
    python3 paper_figures/plot_system_f1.py
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
OUT_PDF = HERE / "figures" / "fig_sys_f1.pdf"
OUT_PNG = HERE / "figures" / "fig_sys_f1.png"
SHEET = "ablation-f1-system"


def _bar_grid(ax, queries: List[str], values_by_system: Dict[str, List[float]],
              ylabel: str, ymax: float, missing_color: str) -> None:
    n, k = len(queries), len(SYSTEMS)
    span, pitch = 1.08, 1.7
    w = span / k
    x = np.arange(n) * pitch
    for i, s in enumerate(SYSTEMS):
        vals = values_by_system[s]
        offs = (i - (k - 1) / 2.0) * w
        bars = ax.bar(
            x + offs,
            [v if (v is not None) else 0 for v in vals],
            width=w * 0.94, color=PALETTE[s], label=SYS_LABELS[s],
            edgecolor="#222", linewidth=0.6, zorder=3,
        )
        for b, v in zip(bars, vals):
            if v is None:
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                ax.plot([b.get_x() + b.get_width() / 2], [0.0],
                        marker="x", color=missing_color, markersize=8,
                        markeredgewidth=2.0, clip_on=False, transform=trans, zorder=7)
    ax.set_xticks(x)
    ax.set_xticklabels([short_q(q) for q in queries], rotation=24,
                       ha="right", fontweight="bold", fontsize=18)
    ax.set_ylabel(ylabel, fontweight="bold")
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
    ax.set_ylim(0, ymax)


def load(sheet: str = SHEET) -> Dict[str, Dict[str, float]]:
    wb = openpyxl.load_workbook(PLOTDATA, data_only=True)
    ws = wb[sheet]
    out: Dict[str, Dict[str, float]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None or row[1] is None:
            continue
        q, sys_n, val = str(row[0]).strip(), str(row[1]).strip(), row[2]
        try:
            v = float(val) if val is not None else None
        except (TypeError, ValueError):
            v = None
        out.setdefault(q, {})[sys_n] = v
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

    # Wider canvas so the 7-system groups span the full two-column text width.
    fig, ax = plt.subplots(figsize=(20, 4.2))
    _bar_grid(ax, queries, by_sys, ylabel=r"F$_1$", ymax=1.18,
              missing_color=PALETTE["Ours"])
    ax.axhline(1.0, color="#3A3A3A", linewidth=0.5, linestyle="--")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=160)
    print(f"wrote {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    render()
