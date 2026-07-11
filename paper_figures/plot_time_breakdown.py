"""Absolute-time variant of Figure 6 (fig:time-stack).

Unlike the normalized-% time variant (which normalizes everything to "% of full-MLLM
time"), this script renders ABSOLUTE wall-clock seconds. Per query we draw two
bars:

  LEFT  = the AI UDF no-opt absolute time (seconds), single solid grey bar.
  RIGHT = the LASO absolute time (seconds), with the six-stage breakdown drawn
          on it (Embedding, Sampling, Labeling, Label Propagation, Proxy Train,
          Proxy Predict; bottom -> top).

DATA-BASIS RULE (critical). Both bar heights come from the measured `latency_s`
column of `ablation-latency-system`; the `time-breakdown` sheet only supplies the
within-LASO proportions (its absolute per-query totals are from a different run
base and are never used as a height, e.g. mmqa_q7 breakdown Total=442s vs its
measured LASO latency=21.8s):
  * LEFT  height = noopt = latency sheet Baseline(SparkSQL) `latency_s`.
  * RIGHT total = laso   = latency sheet Ours `latency_s` (the same value in
    fig:sys-latency), so RIGHT/LEFT equals the reported speedup
    (mmqa_q7 = 4630/21.8 = 212x).
  * RIGHT stage segments = (stage_s / sum_stage_s from breakdown) * laso.
    We trust the breakdown sheet ONLY for the within-LASO PROPORTIONS and scale
    them to the absolute LASO latency.

Query-name map: breakdown uses `lro_113`; latency uses `match113`
(lro_<n> -> match<n>). Only queries with a Baseline(SparkSQL) row are plotted.

Because absolute times span ~5 orders of magnitude across queries (~1.4s for
match queries up to ~41600s for ecomm_q8), we produce BOTH a log-axis and a
linear-axis version so the tradeoff is visible:
  * log:    every query readable, but the stack segments compress (true linear
            stacking is not honest on a log axis), so we draw the stack from a
            small floor and annotate the dominant-stage / total seconds.
  * linear: the stack is honest, but the small queries vanish -- that is the
            point of showing the tradeoff.

Run: python3 paper_figures/plot_time_breakdown.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib.patches import Patch

from plot_style import PHASE_PALETTE, apply_paper_style, order_queries, short_q


def _normalize_query_name(q: str) -> str:
    s = str(q).strip()
    if s.startswith("lro_"):
        return "match" + s.split("_", 1)[1]
    return s


HERE = Path(__file__).resolve().parent  # paper_figures/: holds plotdata.xlsx, figures/, tables/
PLOTDATA = HERE / "plotdata.xlsx"
OUT_PDF = HERE / "figures" / "fig_time_abs.pdf"
OUT_PNG = HERE / "figures" / "fig_time_abs.png"
OUT_PNG_LOG = HERE / "figures" / "fig_time_abs_log.png"
OUT_PNG_LIN = HERE / "figures" / "fig_time_abs_linear.png"
SHEET = "time-breakdown"
LAT_SHEET = "ablation-latency-system"
NOOPT_SYSTEM = "Baseline(SparkSQL)"
NOOPT_COLOR = "#3A3A3A"

# Display order of the stack, bottom -> top. "LLM rollback" is the expert-MLLM
# overhead the online router incurs when it rolls back unconfident rows; it is
# read from the time-breakdown sheet's column H ("Fallback time", seconds).
PHASES = ["Embedding", "Sampling", "Labeling",
          "Label Propagation", "Proxy Train", "Proxy Predict", "LLM rollback"]
COLORS = [PHASE_PALETTE[p] for p in PHASES]
_DARK_FILLS = {"Embedding", "Labeling", "Label Propagation", "LLM rollback"}
# Legend display names (the palette/stage key stays "Labeling"; the displayed
# label says "LLM labeling" because this stage is the expert-MLLM oracle calls
# made during active sampling, the parallel of the "LLM rollback" stage).
LEGEND_NAMES = {"Labeling": "LLM labeling"}


def load_breakdown():
    """Return query list + the seven stage arrays in PHASES order.

    Breakdown sheet column layout (0-indexed), latest layout:
      0 query, 1 Embedding, 2 Sampling, 3 Labeling, 4 Proxy Train,
      5 Proxy Predict, 6 Label Propagation, 7 Fallback time (= the LLM-rollback
      overhead, seconds), 8 query, 9 |T|, 10 Fallback pCt, 11 Label PCt.
    """
    wb = openpyxl.load_workbook(PLOTDATA, data_only=True)
    ws = wb[SHEET]
    qs, emb, samp, lab, trn, pred, lp, fb = [], [], [], [], [], [], [], []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        try:
            e = float(row[1]); s = float(row[2]); l = float(row[3])
            t = float(row[4]); p = float(row[5]); g = float(row[6])
        except (TypeError, ValueError):
            continue
        try:
            f = float(row[7])            # column H: Fallback time (seconds)
        except (TypeError, ValueError, IndexError):
            f = 0.0
        qs.append(str(row[0]))
        emb.append(e); samp.append(s); lab.append(l)
        trn.append(t); pred.append(p); lp.append(g); fb.append(f)
    return (qs, np.array(emb), np.array(samp), np.array(lab),
            np.array(trn), np.array(pred), np.array(lp), np.array(fb))


def load_noopt_and_laso():
    """{normalized_query: (noopt_s, laso_s)} from the latency sheet's latency_s
    column. Both bar heights come straight from the measured per-system latency:
      * noopt = Baseline(SparkSQL) latency_s (full-MLLM, no LASO optimizations).
      * laso  = Ours latency_s (LASO end-to-end), the same number plotted in
        fig:sys-latency, so the no-opt/LASO gap equals the reported speedup
        (e.g. mmqa_q7 = 4630/21.8 = 212x).
    The time-breakdown sheet is used only to split the LASO bar into stages by
    proportion; it never sets a bar height.
    """
    wb = openpyxl.load_workbook(PLOTDATA, data_only=True)
    ws = wb[LAT_SHEET]
    noopt: dict[str, float] = {}
    ours: dict[str, float] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None or row[1] is None:
            continue
        q = str(row[0]).strip()
        sysn = str(row[1]).strip()
        try:
            lat = float(row[2])
        except (TypeError, ValueError):
            continue
        if sysn == NOOPT_SYSTEM:
            noopt[q] = lat
        elif sysn == "LASO (with router)":   # was "Ours"; right bar = LASO+router
            ours[q] = lat
    return {q: (noopt[q], ours[q]) for q in noopt if q in ours}


def _fmt_secs(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:.1f}k"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _prepare():
    qs_raw, emb, samp, lab, trn, pred, lp, fb = load_breakdown()
    noopt_map = load_noopt_and_laso()

    norm = [_normalize_query_name(q) for q in qs_raw]
    norm_to_idx = {n: i for i, n in enumerate(norm)}
    qs_ordered = [q for q in order_queries(norm) if q in noopt_map]
    perm = [norm_to_idx[q] for q in qs_ordered]
    qs = list(qs_ordered)
    emb = emb[perm]; samp = samp[perm]; lab = lab[perm]
    trn = trn[perm]; pred = pred[perm]; lp = lp[perm]; fb = fb[perm]

    # stack order = PHASES = Embedding, Sampling, Labeling, LP, Train, Predict,
    # LLM rollback (router fallback MLLM overhead, breakdown column H)
    stage_secs = np.vstack([emb, samp, lab, lp, trn, pred, fb])  # 7 x N
    breakdown_total = stage_secs.sum(axis=0)
    prop = stage_secs / breakdown_total                       # within-LASO mix

    noopt = np.array([noopt_map[q][0] for q in qs])
    laso_total = np.array([noopt_map[q][1] for q in qs])
    seg = prop * laso_total                                    # 6 x N abs seconds
    speedup = noopt / laso_total
    return qs, prop, seg, noopt, laso_total, speedup


def _draw_xaxis(ax, qs):
    x = np.arange(len(qs))
    ax.set_xticks(x)
    ax.set_xticklabels([short_q(q) for q in qs], rotation=30, ha="right",
                       fontweight="bold")
    sb = sum(1 for q in qs if not str(q).lower().startswith(("match", "lro_")))
    if 0 < sb < len(qs):
        ax.axvline(sb - 0.5, color="#3A3A3A", linestyle=":", linewidth=0.7)


def _legend(ax, fig):
    handles = [Patch(facecolor=NOOPT_COLOR, edgecolor="#222",
                     label="No-proxy baseline")]
    handles += [Patch(facecolor=c, edgecolor="#222", label=LEGEND_NAMES.get(p, p))
                for p, c in zip(PHASES, COLORS)]
    # Pick the LARGEST font (up to the 14pt axis-label size, i.e. paper text size)
    # that still fits all entries on ONE line spanning at most the plot width.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    target_w = fig.get_window_extent(rend).width * 0.99   # may span the full figure width
    leg = None
    chosen = 8.5
    for fs in (16, 15.5, 15, 14.5, 14, 13.5, 13, 12.5, 12, 11.5, 11, 10.5, 10, 9.5, 9, 8.5):
        if leg is not None:
            leg.remove()
        leg = ax.legend(handles=handles, ncol=len(handles), loc="upper center",
                        bbox_to_anchor=(0.5, 1.26), frameon=False, fontsize=fs,
                        handlelength=0.8, columnspacing=0.6, handletextpad=0.3)
        fig.canvas.draw()
        if leg.get_window_extent(rend).width <= target_w:
            chosen = fs
            break
    print(f"  legend fontsize chosen (one line): {chosen}")
    return leg


def plot_linear(qs, seg, noopt, laso_total, speedup):
    """Honest stacked bars on a linear axis. Small queries collapse to ~0 px."""
    apply_paper_style(bold=True)
    # Wider full-\textwidth banner; bump axis/tick fonts so rendered text stays
    # readable after the wider canvas is scaled to the two-column text width.
    plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 15,
                         "ytick.labelsize": 15})
    fig, ax = plt.subplots(figsize=(15.5, 5.6))
    x = np.arange(len(qs))
    bw = 0.40
    x_ref = x - 0.5 * bw - 0.01
    x_laso = x + 0.5 * bw + 0.01

    ax.bar(x_ref, noopt, width=bw, color=NOOPT_COLOR,
           edgecolor="#222", linewidth=0.4, zorder=2)
    # no-opt value label
    top = noopt.max()
    for j, v in enumerate(noopt):
        ax.text(x_ref[j], v + 0.01 * top, _fmt_secs(v), ha="center",
                va="bottom", fontsize=18, color=NOOPT_COLOR, rotation=90)

    bottom = np.zeros(len(qs))
    for i, (label, color) in enumerate(zip(PHASES, COLORS)):
        ax.bar(x_laso, seg[i], bottom=bottom, width=bw, color=color,
               edgecolor="#222", linewidth=0.3, label=label, zorder=3)
        bottom = bottom + seg[i]
    for j, h in enumerate(laso_total):
        ax.text(x_laso[j], h + 0.01 * top, _fmt_secs(h), ha="center",
                va="bottom", fontsize=18, color="#B85C5C", fontweight="bold",
                rotation=90)

    _draw_xaxis(ax, qs)
    ax.set_ylabel("Latency (s)", fontweight="bold")
    ax.set_ylim(0, top * 1.40)
    _legend(ax, fig)
    fig.tight_layout()
    fig.savefig(OUT_PNG_LIN, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return OUT_PNG_LIN


def plot_log(qs, prop, seg, noopt, laso_total, speedup):
    """Log axis: every query readable. Stack is drawn from a small floor; the
    segment heights are *visually compressed* (log space) so they are NOT a
    faithful area encoding -- they show the within-LASO ordering, and we
    annotate the right bar with its total seconds and the dominant stage.
    """
    apply_paper_style(bold=True)
    # Wider full-\textwidth banner; bump axis/tick fonts so rendered text stays
    # readable after the wider canvas is scaled to the two-column text width.
    plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 15,
                         "ytick.labelsize": 15})
    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    x = np.arange(len(qs))
    bw = 0.40
    x_ref = x - 0.5 * bw - 0.01
    x_laso = x + 0.5 * bw + 0.01

    floor = max(min(noopt.min(), laso_total.min()) * 0.5, 1e-2)
    ax.set_yscale("log")

    # LEFT no-opt: single solid bar from floor.
    ax.bar(x_ref, noopt, width=bw, bottom=floor - floor, color=NOOPT_COLOR,
           edgecolor="#222", linewidth=0.4, zorder=2)
    # matplotlib log bars start at the axis bottom; set bottom via clip.
    ax.set_ylim(floor, max(noopt.max(), laso_total.max()) * 4.0)

    for j, v in enumerate(noopt):
        ax.text(x_ref[j], v * 1.05, _fmt_secs(v), ha="center", va="bottom",
                fontsize=18, color=NOOPT_COLOR, rotation=90)

    # RIGHT LASO: stack each stage in LOG space from `floor` up to laso_total.
    # We map cumulative proportion -> log height so the full bar reaches the
    # true laso_total, with stage boundaries at log-interpolated positions.
    log_floor = np.log10(floor)
    for j in range(len(qs)):
        top_log = np.log10(laso_total[j])
        span = top_log - log_floor
        cum = 0.0
        prev_log = log_floor
        for i in range(len(PHASES)):
            cum += prop[i, j]
            cur_log = log_floor + span * cum
            lo = 10 ** prev_log
            hi = 10 ** cur_log
            ax.bar(x_laso[j], hi - lo, bottom=lo, width=bw, color=COLORS[i],
                   edgecolor="#222", linewidth=0.3, zorder=3)
            prev_log = cur_log
    for j, h in enumerate(laso_total):
        ax.text(x_laso[j], h * 1.05, _fmt_secs(h), ha="center",
                va="bottom", fontsize=18, color="#B85C5C", fontweight="bold",
                rotation=90)

    _draw_xaxis(ax, qs)
    ax.set_ylabel("Latency (s, log scale)", fontweight="bold")
    _legend(ax, fig)
    fig.tight_layout()
    fig.savefig(OUT_PNG_LOG, bbox_inches="tight", dpi=160)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return OUT_PNG_LOG


if __name__ == "__main__":
    qs, prop, seg, noopt, laso_total, speedup = _prepare()

    p_log = plot_log(qs, prop, seg, noopt, laso_total, speedup)
    p_lin = plot_linear(qs, seg, noopt, laso_total, speedup)

    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}  (= log version)")
    print(f"wrote {p_log}")
    print(f"wrote {p_lin}")
    print()
    print(f"{'query':12} {'noopt(s)':>10} {'laso(s)':>10} {'speedup':>8}")
    for q, n, l, s in zip(qs, noopt, laso_total, speedup):
        print(f"{q:12} {n:10.2f} {l:10.2f} {s:8.1f}x")
