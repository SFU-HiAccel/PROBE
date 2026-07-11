#!/usr/bin/env python3
"""delegation_curves.py - F1 vs delegation rate plots and summary tables.

Mirrors Figure 5 / Tables 2 / 3 of the GAMCAL paper for our two-policy
setup (proxy_only vs proxy+learned_defer) on the queries swept by
run.py.

Reads:
    ./results/direct_grid/{query}_results.csv

Writes (default ./results/delegation/):
    f1_vs_delegation.{png,pdf}              Figure 5 analogue
    table2_best_operating_points.{csv,md}   Table 2 analogue
    table3_budgeted_best.{csv,md}           Table 3 analogue
    delegation_long.csv                     Aggregated raw data

Axes:
    x = delegation rate = total LLM calls / N
        Ticks fixed at {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.
    y = F1 score

Strategies:
    proxy_only           -> ablation 'direct+cs+trust_proxy'
                            delegation rate = sample_size / N
    proxy+learned_defer  -> ablation 'direct+cs+learned_deferral'
                            delegation rate = (sample + probe + verify) / N
    proxy+dasp_topk      -> ablation 'direct+cs+dasp_topk'
                            delegation rate = (sample + verify) / N
                            (DASP splits sample into B_proxy + B_defer; both
                             count as LLM oracle calls)
    LLM_only             -> single anchor at (1.0, F1_anchor) per query.
                            By default F1_anchor is taken from the
                            measured Full-LLM F1 baked into
                            FULL_LLM_F1_PER_QUERY below; override with
                            --llm-anchor (numeric -> apply globally;
                            'skip' -> omit) or --llm-anchor-file
                            (CSV with columns query,full_llm_f1).

Curve sweep variable: BUDGET_PCTS (the existing sample-size sweep in the
grid). Each subplot point is the mean over repeats; error bars are 1 std.

Usage:
    python delegation_curves.py
    python delegation_curves.py ecomm_q7 cars_q4
    python delegation_curves.py --budgets 20,30,50
    python delegation_curves.py --llm-anchor 0.95
    python delegation_curves.py --llm-anchor skip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRUST_PROXY_LABEL = "direct+cs+trust_proxy"
LEARNED_DEFER_LABEL = "direct+cs+learned_deferral"
DASP_TOPK_LABEL = "direct+cs+dasp_topk"
DELEGATION_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

STRATEGY_DISPLAY = {
    "proxy_only": "proxy_only",
    "proxy+learned_defer": "proxy+learned_defer",
    "proxy+dasp_topk": "proxy+dasp_topk",
    "LLM_only": "LLM_only",
}
STRATEGY_COLORS = {
    "proxy_only": "#2a9d8f",
    "proxy+learned_defer": "#e76f51",
    "proxy+dasp_topk": "#9b59b6",
    "LLM_only": "#264653",
}
STRATEGY_MARKERS = {
    "proxy_only": "o",
    "proxy+learned_defer": "s",
    "proxy+dasp_topk": "D",
    "LLM_only": "*",
}

# Single source of truth: (ablation_label, strategy_key) for the three
# proxy-side curves. plot_grid, build_table2, build_table3, and
# write_long_csv all iterate over this list.
PROXY_STRATEGIES = (
    (TRUST_PROXY_LABEL,   "proxy_only"),
    (LEARNED_DEFER_LABEL, "proxy+learned_defer"),
    (DASP_TOPK_LABEL,     "proxy+dasp_topk"),
)
KEPT_LABELS = [label for label, _ in PROXY_STRATEGIES]

# Per-query F1 of the "Full LLM" baseline (calling the LLM oracle on every
# row and scoring against gold labels). Used as the y-coordinate of the
# LLM_only anchor at delegation rate 1.0. Queries missing here get no
# anchor unless --llm-anchor is set to a numeric global override.
FULL_LLM_F1_PER_QUERY: Dict[str, float] = {
    "mmqa_q2a":   0.434782609,
    "mmqa_q7":    0.769,
    "ecomm_q7":   0.771,
    "ecomm_q8":   0.5,
    "ecomm_q9":   0.419,
    "ecomm_q13":  0.545,
    "cars_q3":    0.9865,
    "cars_q4":    0.969101082,
    "cars_q8":    0.318,
    "animals_q1": 0.796,
    "match113":   0.8,
    "match121":   0.1138,
    "match202":   1.0,
    "match305":   0.6667,
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(out):
        return default
    return out


def total_llm_calls(row: pd.Series) -> Optional[float]:
    """Total LLM oracle invocations for one experiment row.

    Prefer vllm_requests_actual (HTTP-level truth from the vLLM server);
    fall back to llm_calls + probe + verify, which sums to the same
    number when the run completed successfully.
    """
    vllm_req = row.get("vllm_requests_actual")
    if pd.notna(vllm_req):
        v = _safe_float(vllm_req, default=-1.0)
        if v > 0:
            return v
    sampled = _safe_float(row.get("llm_calls"))
    probe = _safe_float(row.get("deferral_probe_n"))
    verify = _safe_float(row.get("policy_rollback_to_llm_count"))
    total = sampled + probe + verify
    return total if total > 0 else None


def _infer_n_total(df: pd.DataFrame) -> float:
    """Extract N from the CSV. Prefer policy_total_rows; otherwise back
    out from llm_calls / (budget_pct / 100) on trust_proxy rows."""
    if "policy_total_rows" in df.columns:
        n = _safe_float(df["policy_total_rows"].max())
        if n > 0:
            return n
    sub = df[df["ablation"] == TRUST_PROXY_LABEL]
    if sub.empty or "budget_pct" not in sub.columns or "llm_calls" not in sub.columns:
        return 0.0
    bps = sub["budget_pct"].astype(float)
    calls = sub["llm_calls"].astype(float)
    estimates = (calls / (bps / 100.0)).replace([np.inf, -np.inf], np.nan).dropna()
    return float(estimates.median()) if len(estimates) else 0.0


def load_query_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ablation" not in df.columns or "f1" not in df.columns:
        return pd.DataFrame()
    df = df[df["ablation"].isin(KEPT_LABELS)].copy()
    if df.empty:
        return df
    df["llm_calls_total"] = df.apply(total_llm_calls, axis=1)
    df = df[df["llm_calls_total"].notna()].copy()
    n_total = _infer_n_total(df)
    df["delegation_rate"] = (
        df["llm_calls_total"] / n_total if n_total > 0 else np.nan
    )
    df.attrs["n_total"] = n_total
    return df


def aggregate_curve(df_strategy: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std of F1 and delegation rate grouped by sample-size budget."""
    agg = (
        df_strategy.groupby("budget_pct", as_index=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            deleg_mean=("delegation_rate", "mean"),
            deleg_std=("delegation_rate", "std"),
            n_runs=("f1", "count"),
        )
        .sort_values("budget_pct")
        .reset_index(drop=True)
    )
    agg["f1_std"] = agg["f1_std"].fillna(0.0)
    agg["deleg_std"] = agg["deleg_std"].fillna(0.0)
    return agg


def strategy_curve(df: pd.DataFrame, ablation_label: str) -> pd.DataFrame:
    sub = df[df["ablation"] == ablation_label]
    if sub.empty:
        return pd.DataFrame()
    return aggregate_curve(sub)


def _compute_y_range(
    df: pd.DataFrame,
    anchor: Optional[float],
) -> Tuple[float, float]:
    """Tight y-range that covers both curves (with error bars) and the anchor.

    Adds 10% padding around the spanned interval and clamps to [0, 1].
    Falls back to (0.0, 1.0) if no data is present.
    """
    f1_vals: List[float] = []
    for ablation, _ in PROXY_STRATEGIES:
        agg = strategy_curve(df, ablation)
        if agg.empty:
            continue
        means = agg["f1_mean"].astype(float).to_numpy()
        stds = agg["f1_std"].astype(float).fillna(0.0).to_numpy()
        f1_vals.extend(means.tolist())
        f1_vals.extend((means + stds).tolist())
        f1_vals.extend((means - stds).tolist())
    if anchor is not None:
        f1_vals.append(float(anchor))
    if not f1_vals:
        return (0.0, 1.0)
    lo, hi = min(f1_vals), max(f1_vals)
    span = max(hi - lo, 0.05)
    pad = span * 0.10
    return (max(0.0, lo - pad), min(1.0, hi + pad))


def plot_grid(
    rows_by_query: Dict[str, pd.DataFrame],
    output_stem: Path,
    anchor_by_query: Dict[str, Optional[float]],
    ncols: int = 3,
    y_range: Optional[Tuple[float, float]] = None,
) -> None:
    queries = [(q, df) for q, df in rows_by_query.items() if not df.empty]
    if not queries:
        return
    n_q = len(queries)
    cols = max(1, min(ncols, n_q))
    rows_axes = (n_q + cols - 1) // cols
    fig, axes = plt.subplots(
        rows_axes, cols,
        figsize=(5.0 * cols, 3.6 * rows_axes),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    legend_keys_emitted: set = set()
    for ax_idx, (query, df) in enumerate(queries):
        ax = axes_flat[ax_idx]
        for ablation, key in PROXY_STRATEGIES:
            agg = strategy_curve(df, ablation)
            if agg.empty:
                continue
            label = STRATEGY_DISPLAY[key] if key not in legend_keys_emitted else None
            legend_keys_emitted.add(key)
            ax.errorbar(
                agg["deleg_mean"], agg["f1_mean"],
                xerr=agg["deleg_std"], yerr=agg["f1_std"],
                color=STRATEGY_COLORS[key],
                marker=STRATEGY_MARKERS[key],
                label=label,
                linewidth=1.5, markersize=6, capsize=3,
            )
        anchor = anchor_by_query.get(query)
        if anchor is not None:
            label = (STRATEGY_DISPLAY["LLM_only"]
                     if "LLM_only" not in legend_keys_emitted else None)
            legend_keys_emitted.add("LLM_only")
            ax.scatter(
                [1.0], [anchor],
                s=200, marker=STRATEGY_MARKERS["LLM_only"],
                color=STRATEGY_COLORS["LLM_only"],
                label=label,
                zorder=5,
            )
            ax.axhline(anchor, linestyle=":", color="#888", linewidth=0.8,
                       alpha=0.6, zorder=1)
        ax.set_title(query)
        ax.set_xlim(-0.02, 1.05)
        if y_range is None:
            y_lo, y_hi = _compute_y_range(df, anchor_by_query.get(query))
        else:
            y_lo, y_hi = y_range
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(DELEGATION_TICKS)
        ax.grid(True, alpha=0.3)
        if ax_idx >= n_q - cols:
            ax.set_xlabel("Delegation Rate")
        if ax_idx % cols == 0:
            ax.set_ylabel("F1 Score")
    for hidden in range(n_q, len(axes_flat)):
        axes_flat[hidden].axis("off")
    fig.legend(
        loc="lower center", ncol=3, fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("F1 vs Delegation Rate", y=1.01)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def best_operating_point(agg: pd.DataFrame) -> Optional[pd.Series]:
    """Highest-F1 budget; tie-break by smallest delegation rate."""
    if agg.empty:
        return None
    top = agg[agg["f1_mean"] == agg["f1_mean"].max()]
    return top.sort_values("deleg_mean").iloc[0]


def best_at_budget(agg: pd.DataFrame, budget_frac: float) -> Optional[pd.Series]:
    """Highest-F1 budget such that delegation rate <= budget_frac."""
    if agg.empty:
        return None
    eligible = agg[agg["deleg_mean"] <= budget_frac + 1e-9]
    if eligible.empty:
        return None
    top = eligible[eligible["f1_mean"] == eligible["f1_mean"].max()]
    return top.sort_values("deleg_mean").iloc[0]


_TABLE_KEY_PREFIX = {
    "proxy_only": "proxy_only",
    "proxy+learned_defer": "defer",
    "proxy+dasp_topk": "dasp",
}


def build_table2(
    rows_by_query: Dict[str, pd.DataFrame],
    anchor_by_query: Dict[str, Optional[float]],
) -> pd.DataFrame:
    rows: List[Dict] = []
    for q, df in rows_by_query.items():
        anchor = anchor_by_query.get(q)
        row: Dict[str, object] = {
            "query": q,
            "n_total": int(df.attrs.get("n_total", 0)),
            "LLM_only_F1": anchor if anchor is not None else np.nan,
        }
        for ablation, key in PROXY_STRATEGIES:
            prefix = _TABLE_KEY_PREFIX[key]
            best = best_operating_point(strategy_curve(df, ablation))
            row[f"{prefix}_F1"] = (
                float(best["f1_mean"]) if best is not None else np.nan
            )
            row[f"{prefix}_rate"] = (
                float(best["deleg_mean"]) if best is not None else np.nan
            )
            row[f"{prefix}_budget_pct"] = (
                float(best["budget_pct"]) if best is not None else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_table3(
    rows_by_query: Dict[str, pd.DataFrame],
    budgets: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict] = []
    for q, df in rows_by_query.items():
        row: Dict[str, object] = {"query": q}
        curves = {
            key: strategy_curve(df, ablation)
            for ablation, key in PROXY_STRATEGIES
        }
        for b in budgets:
            tag = int(round(b * 100))
            for ablation, key in PROXY_STRATEGIES:
                prefix = _TABLE_KEY_PREFIX[key]
                best = best_at_budget(curves[key], b)
                row[f"{prefix}_le_{tag}pct_F1"] = (
                    float(best["f1_mean"]) if best is not None else np.nan
                )
                row[f"{prefix}_le_{tag}pct_rate"] = (
                    float(best["deleg_mean"]) if best is not None else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_cell(f1: float, rate: Optional[float]) -> str:
    if pd.isna(f1):
        return "—"
    if rate is None or pd.isna(rate):
        return f"{f1:.3f}"
    return f"{f1:.3f} ({rate * 100:.0f}%)"


def _bold_best(cells: List[str], values: List[float]) -> List[str]:
    """Bold the cell(s) holding the highest non-NaN value among ``values``."""
    nan = [v for v in values if not pd.isna(v)]
    if not nan:
        return cells
    best = max(nan)
    return [
        f"**{c}**" if (not pd.isna(v) and v >= best) else c
        for c, v in zip(cells, values)
    ]


def format_table2_md(df_t2: pd.DataFrame) -> str:
    lines = [
        "| Query | LLM_only | proxy_only | proxy+learned_defer | proxy+dasp_topk |",
        "|---|---|---|---|---|",
    ]
    for _, r in df_t2.iterrows():
        llm = "—" if pd.isna(r["LLM_only_F1"]) else f"{r['LLM_only_F1']:.3f}"
        po = _fmt_cell(r["proxy_only_F1"], r.get("proxy_only_rate"))
        de = _fmt_cell(r.get("defer_F1"), r.get("defer_rate"))
        da = _fmt_cell(r.get("dasp_F1"), r.get("dasp_rate"))
        po, de, da = _bold_best(
            [po, de, da],
            [r["proxy_only_F1"], r.get("defer_F1"), r.get("dasp_F1")],
        )
        lines.append(f"| {r['query']} | {llm} | {po} | {de} | {da} |")
    return "\n".join(lines)


def format_table3_md(df_t3: pd.DataFrame, budgets: Sequence[float]) -> str:
    header1 = ["Query"]
    header2 = [""]
    for b in budgets:
        tag = int(round(b * 100))
        header1 += [f"d ≤ {tag}%", "", ""]
        header2 += ["proxy_only", "proxy+defer", "proxy+dasp_topk"]
    sep = ["---"] * len(header1)
    lines = [
        "| " + " | ".join(header1) + " |",
        "| " + " | ".join(header2) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for _, r in df_t3.iterrows():
        cells = [str(r["query"])]
        for b in budgets:
            tag = int(round(b * 100))
            po = r.get(f"proxy_only_le_{tag}pct_F1")
            po_r = r.get(f"proxy_only_le_{tag}pct_rate")
            de = r.get(f"defer_le_{tag}pct_F1")
            de_r = r.get(f"defer_le_{tag}pct_rate")
            da = r.get(f"dasp_le_{tag}pct_F1")
            da_r = r.get(f"dasp_le_{tag}pct_rate")
            po_s = _fmt_cell(po, po_r)
            de_s = _fmt_cell(de, de_r)
            da_s = _fmt_cell(da, da_r)
            po_s, de_s, da_s = _bold_best(
                [po_s, de_s, da_s], [po, de, da]
            )
            cells += [po_s, de_s, da_s]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_long_csv(rows_by_query: Dict[str, pd.DataFrame], output_csv: Path) -> None:
    long_rows: List[Dict] = []
    for q, df in rows_by_query.items():
        for ablation, key in PROXY_STRATEGIES:
            agg = strategy_curve(df, ablation)
            for _, r in agg.iterrows():
                long_rows.append({
                    "query": q,
                    "n_total": int(df.attrs.get("n_total", 0)),
                    "strategy": key,
                    "budget_pct": float(r["budget_pct"]),
                    "f1_mean": float(r["f1_mean"]),
                    "f1_std": float(r["f1_std"]),
                    "deleg_mean": float(r["deleg_mean"]),
                    "deleg_std": float(r["deleg_std"]),
                    "n_runs": int(r["n_runs"]),
                })
    if long_rows:
        pd.DataFrame(long_rows).to_csv(output_csv, index=False)


def _load_anchor_csv(path: Path) -> Dict[str, float]:
    """Load query -> Full LLM F1 mapping from a CSV.

    Expected columns: 'query' and one of {'full_llm_f1', 'f1', 'quality'}.
    Extra columns are ignored.
    """
    df = pd.read_csv(path)
    if "query" not in df.columns:
        raise ValueError(f"{path}: CSV must have a 'query' column")
    f1_col = next(
        (c for c in ("full_llm_f1", "f1", "quality") if c in df.columns),
        None,
    )
    if f1_col is None:
        raise ValueError(
            f"{path}: CSV must have one of full_llm_f1 / f1 / quality"
        )
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        q = str(r["query"]).strip()
        v = _safe_float(r[f1_col], default=float("nan"))
        if q and not pd.isna(v):
            out[q] = float(v)
    return out


def resolve_anchor_table(
    queries: Sequence[str],
    anchor_arg: str,
    anchor_file: Optional[Path],
) -> Dict[str, Optional[float]]:
    """Build the per-query anchor mapping from CLI flags.

    Precedence:
      1. --llm-anchor skip -> all anchors None
      2. --llm-anchor <number> -> all anchors set to that number
      3. --llm-anchor-file PATH -> per-query lookup from CSV
      4. --llm-anchor auto (default) -> embedded FULL_LLM_F1_PER_QUERY
    Queries missing from any per-query source get None (no anchor drawn).
    """
    if anchor_arg.lower() == "skip":
        return {q: None for q in queries}
    try:
        global_value: Optional[float] = float(anchor_arg)
    except ValueError:
        global_value = None
    if global_value is not None:
        return {q: global_value for q in queries}
    table: Dict[str, float] = dict(FULL_LLM_F1_PER_QUERY)
    if anchor_file is not None:
        loaded = _load_anchor_csv(anchor_file)
        table.update(loaded)
    return {q: table.get(q) for q in queries}


def discover_queries(results_dir: Path, requested: Sequence[str]) -> List[str]:
    csvs = sorted(results_dir.glob("*_results.csv"))
    available = [p.stem.removesuffix("_results") for p in csvs]
    if not requested:
        return available
    out: List[str] = []
    for q in requested:
        if q in available:
            out.append(q)
        else:
            print(f"[warn] no results CSV for {q!r} in {results_dir}",
                  file=sys.stderr)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("queries", nargs="*",
                   help="Query tags (default: every CSV in --results-dir).")
    p.add_argument("--results-dir", default="./results/direct_grid",
                   help="Where the per-query CSVs from run.py live.")
    p.add_argument("--output-dir", default="./results/delegation",
                   help="Where to write figures and tables.")
    p.add_argument("--llm-anchor", default="auto",
                   help="LLM_only anchor at delegation=1.0. "
                        "'auto' (default): use FULL_LLM_F1_PER_QUERY embedded in this "
                        "script. 'skip': no anchor. A number (e.g. '0.95'): apply that "
                        "F1 globally to all queries.")
    p.add_argument("--llm-anchor-file", default=None,
                   help="Optional CSV with columns 'query' and one of "
                        "'full_llm_f1' / 'f1' / 'quality'. Overrides matching "
                        "entries in FULL_LLM_F1_PER_QUERY (only used when "
                        "--llm-anchor is 'auto').")
    p.add_argument("--budgets", default="20,30",
                   help="Comma-separated delegation-rate budgets (in %%) for Table 3.")
    p.add_argument("--ncols", type=int, default=3,
                   help="Number of columns in the figure grid (default 3).")
    p.add_argument("--y-range", default="auto",
                   help="Per-subplot y-axis range: 'auto' (tight around the "
                        "data + anchor with 10%% padding, default), 'full' "
                        "(0..1.05), or 'lo,hi' (e.g. '0.5,1.0').")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"[error] results dir not found: {results_dir}", file=sys.stderr)
        return 1

    try:
        budgets = tuple(int(s.strip()) / 100.0 for s in args.budgets.split(","))
    except ValueError:
        print(f"[error] invalid --budgets={args.budgets!r}", file=sys.stderr)
        return 1

    queries = discover_queries(results_dir, args.queries)
    if not queries:
        print("[error] no queries to plot", file=sys.stderr)
        return 1

    print(f"Reading {len(queries)} queries from {results_dir} -> {output_dir}")
    rows_by_query: Dict[str, pd.DataFrame] = {}
    for q in queries:
        df = load_query_csv(results_dir / f"{q}_results.csv")
        if df.empty:
            print(f"  [skip] {q}: no rows for trust_proxy/learned_deferral")
            continue
        rows_by_query[q] = df

    if not rows_by_query:
        print("[error] no usable data", file=sys.stderr)
        return 1

    anchor_file = Path(args.llm_anchor_file) if args.llm_anchor_file else None
    try:
        anchor_by_query = resolve_anchor_table(
            list(rows_by_query.keys()), args.llm_anchor, anchor_file,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] anchor resolution failed: {exc}", file=sys.stderr)
        return 1
    missing = [q for q, v in anchor_by_query.items() if v is None]
    if missing:
        print(
            f"  [info] no LLM anchor for: {', '.join(missing)} "
            "(add to FULL_LLM_F1_PER_QUERY or --llm-anchor-file)"
        )

    y_range_arg = args.y_range.strip().lower()
    if y_range_arg == "auto":
        y_range: Optional[Tuple[float, float]] = None
    elif y_range_arg == "full":
        y_range = (-0.02, 1.05)
    else:
        try:
            lo, hi = [float(s) for s in args.y_range.split(",")]
            y_range = (lo, hi)
        except (ValueError, AttributeError):
            print(
                f"[error] invalid --y-range={args.y_range!r}; "
                "expected 'auto', 'full', or 'lo,hi'",
                file=sys.stderr,
            )
            return 1

    plot_grid(
        rows_by_query, output_dir / "f1_vs_delegation",
        anchor_by_query, ncols=args.ncols, y_range=y_range,
    )
    print(f"  [ok]   figure        -> {output_dir / 'f1_vs_delegation'}.png")

    df_t2 = build_table2(rows_by_query, anchor_by_query)
    df_t2.to_csv(output_dir / "table2_best_operating_points.csv", index=False)
    (output_dir / "table2_best_operating_points.md").write_text(
        "# Table 2 — Best F1 operating point per algorithm\n\n"
        "Each cascade cell shows F1 score and delegation rate d. "
        "Bold marks the better cascade between proxy_only and "
        "proxy+learned_defer for that query.\n\n"
        + format_table2_md(df_t2) + "\n"
    )
    print(f"  [ok]   table 2       -> {output_dir / 'table2_best_operating_points'}.{{csv,md}}")

    df_t3 = build_table3(rows_by_query, budgets=budgets)
    df_t3.to_csv(output_dir / "table3_budgeted_best.csv", index=False)
    (output_dir / "table3_budgeted_best.md").write_text(
        "# Table 3 — F1 at fixed delegation budgets\n\n"
        "Each cell shows the best F1 achievable with delegation rate "
        "d ≤ budget. Bold marks the better algorithm per (query, budget).\n\n"
        + format_table3_md(df_t3, budgets=budgets) + "\n"
    )
    print(f"  [ok]   table 3       -> {output_dir / 'table3_budgeted_best'}.{{csv,md}}")

    write_long_csv(rows_by_query, output_dir / "delegation_long.csv")
    print(f"  [ok]   long CSV      -> {output_dir / 'delegation_long.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
