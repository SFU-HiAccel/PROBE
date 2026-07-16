"""Visualize similarity-to-prompt distributions: GT positives vs negatives.

For every query in run_direct_galaxy_grid.py, this script computes the same
1D similarity axis that DIRECT sorts on and plots two overlapping
histograms (positives in green, negatives in red). The ROC-AUC of using
similarity as a binary classifier is also reported per query — it directly
quantifies how well DIRECT can locate the boundary on this axis.

A high AUC (>~0.8) means similarity is strongly informative of class
membership and DIRECT's version-space reduction will converge quickly to
the boundary. A near-0.5 AUC means the assumption breaks and DIRECT
degrades to random sampling.

Usage:
    python viz_similarity_distribution.py                  # all queries
    python viz_similarity_distribution.py movie_q5         # single query
    python viz_similarity_distribution.py movie            # all movie queries
    python viz_similarity_distribution.py grid             # 2x7 combined PDF

Outputs:
    ./results/similarity_dist/{query}_sim_dist.png
    ./results/similarity_dist/separability_summary.csv
    ./results/similarity_dist/similarity_dist_grid.pdf      # grid mode only
"""
import os
# Silence HF-tokenizers fork warning; must be set BEFORE transformers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if (_parent / "omnimcp" / "server").exists():
        sys.path.insert(0, str(_parent))
        break

import re
import numpy as np
if not hasattr(np, "object0"):
    np.object0 = np.object_
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit, length
from pyspark.sql.types import MapType, StructType, ArrayType

# --- Artifact path resolution (ported from the LLMSQL repo; identity-scrubbed) ---
# Repo root holds the local modules (proxy_model, sem_ops_register, cluster_proxy)
# and the Spark plugin JAR; this script lives one level down in paper_figures/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
# Data + model root: SemBench and the SigLIP snapshot live under here (default $HOME).
_HOME = os.environ.get("LASO_DATA_ROOT", os.path.expanduser("~"))

from proxy_model import embed_texts, embed_images_siglip
from sem_ops_register import register_semantic_operators  # noqa: F401

MODEL = os.environ.get("SEMBENCH_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
# Spark plugin JAR ships at the repo root; override with LASO_JAR if needed.
JAR_PATH = os.environ.get("LASO_JAR", str(_REPO_ROOT / "spark-semantic-plugin_2.12-0.1.0-SNAPSHOT.jar"))
EMBEDDING_MODEL = os.environ.get(
    "LLMSQL_EMBEDDING_MODEL",
    _HOME + "/siglip-base-patch16-224",
)

results_dir = "./results/similarity_dist/"
os.makedirs(results_dir, exist_ok=True)
SEPARABILITY_CSV = os.path.join(results_dir, "separability_summary.csv")

MAX_PAIRS_PER_QUERY = int(os.environ.get("VIZ_MAX_PAIRS", "1000000000"))
MAX_NEG_PER_PLOT = int(os.environ.get("VIZ_MAX_NEG", "1000000000"))

# Font scaling for the paper figures (scatter / sampling-baseline). The figures
# are drawn ~7in wide (≈ a 2-column paper's \textwidth), so at
# \includegraphics[width=\textwidth] there is little downscaling and the in-paper
# font ≈ the value below. Default ~11pt base ≈ body text; bump VIZ_FONT_SCALE if
# you include the figure at a smaller width (e.g. single \columnwidth -> ~2.0).
VIZ_FONT_SCALE = float(os.environ.get("VIZ_FONT_SCALE", "1.0"))
def _fs(pt: float) -> float:
    return float(pt) * VIZ_FONT_SCALE

# JoinML datasets use a blocked-matmul path that sweeps the FULL cross
# product — all positive pair sims are kept, negatives are Bernoulli-
# sampled to at most VIZ_JOINML_NEG_SAMPLE values (default 500k) for
# tractable histogramming. Set to 0 to keep every negative (uses more RAM).
VIZ_JOINML_NEG_SAMPLE = int(os.environ.get("VIZ_JOINML_NEG_SAMPLE", "500000"))
VIZ_JOINML_BLOCK = int(os.environ.get("VIZ_JOINML_BLOCK", "2048"))

spark = None


def get_spark():
    global spark
    if spark is not None:
        return spark
    spark = SparkSession.builder.appName("viz_sim_dist") \
        .master("local[*]") \
        .config("spark.driver.memory", "32g") \
        .config("spark.executor.memory", "32g") \
        .config("spark.jars", JAR_PATH) \
        .config("spark.sql.extensions",
                "com.probe.sparkai.SemanticTopKPlugin,"
                "com.probe.sparkai.SemanticMapPlugin,"
                "com.probe.sparkai.SemanticJoinPlugin,"
                "com.probe.sparkai.SemanticFilterPlugin,"
                "com.probe.sparkai.SemanticAggregatePlugin,"
                "com.probe.sparkai.SemanticGroupByPlugin,"
                "com.probe.sparkai.ApproximateFilterPlugin,"
                "com.probe.sparkai.ApproximateJoinPlugin") \
        .config("spark.driver.extraClassPath", JAR_PATH) \
        .config("spark.executor.extraClassPath", JAR_PATH) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.optimizer.columnPruning.enabled", "false") \
        .config("spark.probe.sparkai.disableFilterReorder", "true") \
        .config("spark.default.parallelism", "1") \
        .getOrCreate()
    return spark


# Side-channel stash for per-row embeddings. Populated by the cosine helpers
# below and consumed by `_capture_sims_labels` so the axis-refinement grid
# can fit an LR proxy on the same features the production pipeline uses.
_VIZ_EMB_STASH: dict = {"emb": None}


def _stash_per_row_embeddings(emb: np.ndarray) -> None:
    _VIZ_EMB_STASH["emb"] = np.asarray(emb, dtype=np.float32)


def _take_stashed_embeddings() -> np.ndarray | None:
    emb = _VIZ_EMB_STASH["emb"]
    _VIZ_EMB_STASH["emb"] = None
    return emb


def cosine_to_prompt(embeddings: np.ndarray, prompt: str) -> np.ndarray:
    """Cosine similarity from each row embedding to the prompt embedding."""
    prompt_emb = embed_texts([prompt], EMBEDDING_MODEL)
    norms_e = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    norms_p = np.linalg.norm(prompt_emb) + 1e-8
    cos = (embeddings @ prompt_emb.T) / (norms_e * norms_p)
    _stash_per_row_embeddings(embeddings)
    return cos.flatten().astype(np.float32)


def cosine_pair(left_emb: np.ndarray, right_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity between two equal-length stacks of embeddings."""
    dots = np.sum(left_emb * right_emb, axis=1)
    nl = np.linalg.norm(left_emb, axis=1) + 1e-8
    nr = np.linalg.norm(right_emb, axis=1) + 1e-8
    pair_emb = np.concatenate(
        [np.asarray(left_emb, dtype=np.float32),
         np.asarray(right_emb, dtype=np.float32)], axis=1,
    )
    _stash_per_row_embeddings(pair_emb)
    return (dots / (nl * nr)).astype(np.float32)


def plot_distribution(query_tag: str, similarities: np.ndarray, labels: np.ndarray,
                       summary_rows: list,
                       true_total_pairs: int | None = None,
                       true_n_pos: int | None = None,
                       boundaries: list | None = None,
                       sampled_indices: np.ndarray | None = None) -> None:
    """Save a histogram + KDE for pos vs neg sims and append the AUC to summary.

    When the caller oversampled positives (e.g. low-selectivity JoinML joins
    where we include every GT positive and only ~MAX_PAIRS random negatives),
    pass ``true_total_pairs`` (= |L|*|R|) and ``true_n_pos`` (= total GT
    positives) so the summary CSV reports the real selectivity instead of
    the biased sample ratio.

    ``boundaries`` accepts either:
      * a flat ``[(sim_lo, sim_hi), ...]`` list — rendered as a single
        series in the default band color (legacy form), OR
      * a list of ``{"label": str, "color": str, "intervals":
        [(lo, hi), ...]}`` dicts — rendered as multiple series (one per
        dict) each in its own color with a labeled legend. This form
        lets us overlay the legacy single-threshold DIRECT (k=1) on
        top of the Tier-A multi-interval DIRECT (k_auto) so the
        reader can see where multi-k adds decision boundaries that
        the single-threshold hypothesis misses.

    When ``sampled_indices`` (indices into ``similarities``) is provided,
    those positions are highlighted on the right recall curve — the rank
    of each sampled row is drawn as a small tick so we can see where
    DIRECT concentrated its oracle budget.
    """
    # Auto-overlay: when VIZ_OVERLAY_DIRECT=1 and the caller hasn't
    # already supplied boundaries, run the real sample_direct on this
    # (sims, labels) fixture and render its discovered intervals. When
    # VIZ_OVERLAY_COMPARE=1 is also set, TWO hypotheses are rendered
    # (single-threshold k=1 and multi-k k_auto) so the reader can see
    # where multi-k adds decision boundaries. Env-gated so legacy
    # plots stay pixel-identical when the flag is off.
    if boundaries is None and sampled_indices is None:
        _meta = _maybe_overlay_direct_boundaries(query_tag, similarities, labels)
        if _meta is not None:
            # Prefer the multi-series form when available so the
            # compare-mode two-color rendering lights up automatically;
            # otherwise fall back to the single-series flat list.
            boundaries = _meta.get("series") or _meta["sim_intervals"]
            sampled_indices = _meta["sampled_indices"]
    else:
        # Caller-provided overrides win; clear any stale overlay metadata.
        plot_distribution._last_overlay_meta = None  # type: ignore[attr-defined]

    # Normalize `boundaries` into a list of series dicts so the rest of
    # the plotting code handles one or many hypotheses uniformly. Empty
    # list / None → no series drawn. Flat list of tuples → single series
    # in the default Tier-A color.
    boundary_series: list = []
    if boundaries:
        if isinstance(boundaries[0], dict):
            for s in boundaries:
                ivs = [
                    (float(lo), float(hi))
                    for (lo, hi) in s.get("intervals", [])
                ]
                if not ivs:
                    continue
                boundary_series.append({
                    "label": str(s.get("label", "")),
                    "color": str(s.get("color", "#6a4c93")),
                    "intervals": ivs,
                    "linestyle": str(s.get("linestyle", ":")),
                })
        else:
            boundary_series.append({
                "label": "DIRECT k-interval",
                "color": "#6a4c93",
                "intervals": [(float(lo), float(hi)) for (lo, hi) in boundaries],
                "linestyle": ":",
            })

    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_sims = similarities[pos_mask]
    neg_sims = similarities[neg_mask]
    n_pos = len(pos_sims)
    n_neg = len(neg_sims)
    sample_selectivity = n_pos / max(n_pos + n_neg, 1)
    if true_total_pairs and true_n_pos is not None:
        selectivity = true_n_pos / max(true_total_pairs, 1)
        reported_n_pos = int(true_n_pos)
        reported_n_neg = int(true_total_pairs - true_n_pos)
    else:
        selectivity = sample_selectivity
        reported_n_pos = n_pos
        reported_n_neg = n_neg

    if n_pos == 0 or n_neg == 0:
        print(f"  [{query_tag}] degenerate (n_pos={n_pos}, n_neg={n_neg}), skipping plot")
        return

    if n_neg > MAX_NEG_PER_PLOT:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_neg, size=MAX_NEG_PER_PLOT, replace=False)
        neg_sims_plot = neg_sims[idx]
    else:
        neg_sims_plot = neg_sims

    try:
        auc = float(roc_auc_score(labels, similarities))
    except Exception:
        auc = float("nan")

    pos_q = np.quantile(pos_sims, [0.05, 0.5, 0.95])
    neg_q = np.quantile(neg_sims, [0.05, 0.5, 0.95])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = 60
    lo = float(min(similarities.min(), pos_sims.min()))
    hi = float(max(similarities.max(), pos_sims.max()))
    pad = (hi - lo) * 0.02
    edges = np.linspace(lo - pad, hi + pad, bins + 1)
    # Legend shows TRUE counts when the caller supplied them (JoinML datasets),
    # with the biased-sample counts in parens so it's clear how many rows
    # actually shaped the density estimate.
    if true_total_pairs and true_n_pos is not None:
        neg_label = f"Negative  (n={reported_n_neg:,}; sampled {n_neg:,})"
        pos_label = f"Positive  (n={reported_n_pos:,}; sampled {n_pos:,})"
    else:
        neg_label = f"Negative (n={n_neg:,})"
        pos_label = f"Positive (n={n_pos:,})"
    ax.hist(neg_sims_plot, bins=edges, alpha=0.55,
            label=neg_label, color="#e76f51", density=True)
    ax.hist(pos_sims, bins=edges, alpha=0.65,
            label=pos_label, color="#2a9d8f", density=True)
    ax.axvline(np.median(pos_sims), color="#2a9d8f", linestyle="--", linewidth=1)
    ax.axvline(np.median(neg_sims), color="#e76f51", linestyle="--", linewidth=1)

    # Overlay DIRECT-discovered decision boundaries on the histogram
    # panel as vertical dashed lines — one line per interval endpoint.
    # The 2K boundaries from K intervals are what the Tier-A DP
    # identified as the label-transition points on the similarity
    # axis, so drawing them as dashed verticals (instead of shaded
    # bands) makes their exact positions readable. Each series still
    # gets its own color so compare-mode runs distinguish the legacy
    # k=1 hypothesis from the multi-k hypothesis. The boundary value
    # is annotated at the top of each line; only the first boundary
    # of each series gets a legend entry so a series with many
    # intervals doesn't spam the legend.
    y_top = ax.get_ylim()[1]
    for s in boundary_series:
        color = s["color"]
        ls = s["linestyle"]
        # Flatten 2K endpoints per series so we loop over them in one
        # pass; endpoint_idx is used to decide which gets the legend
        # handle.
        endpoints: list = []
        for (lo, hi) in s["intervals"]:
            endpoints.append(lo)
            if hi != lo:
                endpoints.append(hi)
        for e_idx, boundary in enumerate(endpoints):
            line_label = s["label"] if e_idx == 0 else None
            ax.axvline(
                boundary, color=color, linestyle=ls,
                linewidth=1.6, label=line_label, zorder=5,
            )
            ax.annotate(
                f"{boundary:.3f}",
                xy=(boundary, y_top),
                xytext=(2, -4),
                textcoords="offset points",
                fontsize=7, color=color,
                rotation=90, ha="left", va="top",
            )

    ax.set_xlabel("Cosine similarity (1D DIRECT axis)")
    ax.set_ylabel("Density")
    if true_total_pairs and true_n_pos is not None:
        title = (f"{query_tag}: GT positives vs negatives  "
                 f"(true selectivity={selectivity:.2e}, "
                 f"cross-product={true_total_pairs:,})")
    else:
        title = f"{query_tag}: GT positives vs negatives (selectivity={selectivity:.1%})"
    ax.set_title(title)
    ax.legend()
    ax.text(0.02, 0.97,
            f"AUC = {auc:.3f}\n"
            f"pos median = {pos_q[1]:.3f}  [5%-95%: {pos_q[0]:.3f}, {pos_q[2]:.3f}]\n"
            f"neg median = {neg_q[1]:.3f}  [5%-95%: {neg_q[0]:.3f}, {neg_q[2]:.3f}]",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#fff8dc', edgecolor='gray'))

    ax = axes[1]
    sorted_idx = np.argsort(-similarities)
    sorted_labels = labels[sorted_idx]
    rank = np.arange(1, len(sorted_labels) + 1)
    cum_pos = np.cumsum(sorted_labels == 1)
    recall_curve = cum_pos / max(int(np.sum(labels == 1)), 1)
    ax.plot(rank, recall_curve, color="#264653", linewidth=2)
    ax.set_xlabel("Rank along similarity axis (highest sim first)")
    ax.set_ylabel("Cumulative recall of positives")
    ax.set_title("Sorted-by-similarity recall curve")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)

    for frac in (0.01, 0.05, 0.1, 0.25, 0.5):
        k = max(1, int(frac * len(rank)))
        recall_at_k = float(recall_curve[k - 1])
        ax.scatter([k], [recall_at_k], color="#e76f51", s=30, zorder=5)
        ax.annotate(f"top-{int(frac*100)}%: R={recall_at_k:.2f}",
                    xy=(k, recall_at_k), xytext=(8, -10), textcoords="offset points",
                    fontsize=8)

    # Convert boundaries from similarity-space back to rank-space so the
    # right panel visualizes where on the sorted axis DIRECT's intervals
    # live. sort_order is descending similarity; a similarity threshold
    # s_hi maps to the smallest rank whose similarity <= s_hi. Each
    # series keeps its own color so the two hypotheses are visually
    # distinguishable.
    if boundary_series:
        sorted_sims = similarities[sorted_idx]
        for s in boundary_series:
            color = s["color"]
            ls = s["linestyle"]
            for (lo, hi) in s["intervals"]:
                # Interval [lo, hi] on the sim axis maps to a contiguous
                # rank block: leftmost rank is the first position whose
                # sim <= hi; rightmost is the last position whose sim >= lo.
                rank_left = int(np.searchsorted(-sorted_sims, -float(hi), side="left")) + 1
                rank_right = int(np.searchsorted(-sorted_sims, -float(lo), side="right"))
                rank_left = max(1, min(rank_left, len(rank)))
                rank_right = max(rank_left, min(rank_right, len(rank)))
                ax.axvspan(rank_left, rank_right, alpha=0.18, color=color)
                ax.axvline(rank_left, color=color, linestyle=ls, linewidth=1.0)
                ax.axvline(rank_right, color=color, linestyle=ls, linewidth=1.0)

    # Tick marks for DIRECT-sampled rows (where the LLM budget actually
    # went) so reviewers can see whether sampling concentrated near the
    # discovered intervals.
    if sampled_indices is not None and len(sampled_indices) > 0:
        rank_of = np.empty(len(similarities), dtype=np.int64)
        rank_of[sorted_idx] = np.arange(1, len(similarities) + 1)
        sampled_ranks = rank_of[np.asarray(sampled_indices, dtype=np.int64)]
        ax.scatter(
            sampled_ranks,
            np.zeros_like(sampled_ranks, dtype=np.float32) + 0.02,
            marker="|", color="#6a4c93", s=40, alpha=0.6,
            label=f"DIRECT sampled (n={len(sampled_ranks):,})",
        )
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(results_dir, f"{query_tag}_sim_dist.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{query_tag}] AUC={auc:.3f}  pos_med={pos_q[1]:.3f}  neg_med={neg_q[1]:.3f}  -> {out_path}")

    row = {
        "query": query_tag,
        # True full-cross-product counts when the caller supplies them;
        # otherwise the sample counts (behavior unchanged for legacy queries).
        "n_pos": reported_n_pos,
        "n_neg": reported_n_neg,
        "selectivity": selectivity,
        "sample_n_pos": n_pos,
        "sample_n_neg": n_neg,
        "sample_selectivity": round(sample_selectivity, 4),
        "auc": round(auc, 4),
        "pos_p05": round(float(pos_q[0]), 4),
        "pos_p50": round(float(pos_q[1]), 4),
        "pos_p95": round(float(pos_q[2]), 4),
        "neg_p05": round(float(neg_q[0]), 4),
        "neg_p50": round(float(neg_q[1]), 4),
        "neg_p95": round(float(neg_q[2]), 4),
    }
    # Tier-A multi-interval DIRECT columns; populated when the overlay
    # runner attached metadata under the sentinel attribute. In compare
    # mode both the k=1 baseline and the k_auto multi-k series are
    # recorded under separate column prefixes so downstream analysis
    # can rank queries by absolute and relative F1 improvement.
    overlay_meta = getattr(plot_distribution, "_last_overlay_meta", None)
    if overlay_meta is not None and overlay_meta.get("query") == query_tag:
        import json as _json
        primary = overlay_meta.get("primary") or overlay_meta
        row["direct_k_chosen"] = int(primary.get("chosen_k", 0))
        row["direct_intervals"] = _json.dumps(primary.get("sim_intervals", []))
        row["direct_sampled_n"] = int(primary.get("sampled_n", 0))
        row["direct_f1_on_sample"] = round(
            float(primary.get("f1_on_sample", 0.0)), 4
        )
        baseline = overlay_meta.get("baseline")
        if baseline is not None:
            row["direct_k1_intervals"] = _json.dumps(
                baseline.get("sim_intervals", [])
            )
            row["direct_k1_sampled_n"] = int(baseline.get("sampled_n", 0))
            row["direct_k1_f1_on_sample"] = round(
                float(baseline.get("f1_on_sample", 0.0)), 4
            )
            row["direct_multi_vs_k1_delta_f1"] = round(
                float(primary.get("f1_on_sample", 0.0))
                - float(baseline.get("f1_on_sample", 0.0)),
                4,
            )
    summary_rows.append(row)


# ---------------------------------------------------------------------------
# Tier-A multi-interval DIRECT overlay
#
# When VIZ_OVERLAY_DIRECT=1 is set, every per-query viz function calls
# `_maybe_overlay_direct_boundaries(...)` before dispatching to
# `plot_distribution`. That helper runs the real `sample_direct` code path
# from cluster_proxy.py against a synthetic oracle backed by the ground-
# truth labels, extracts the discovered k-interval boundaries on the
# cosine-similarity axis, and hands them to `plot_distribution` for
# rendering. The point of threading through the real sampler (instead of
# re-implementing the DP here) is that the bands shown on the plot are
# exactly what DIRECT would pick in a live run at the same budget.
#
# Environment knobs (all optional):
#   VIZ_OVERLAY_DIRECT=1        — enable overlays (default off).
#   VIZ_OVERLAY_BUDGET=200      — oracle budget (integer) per query.
#   VIZ_OVERLAY_K=2             — k_intervals to request; ignored when
#                                 VIZ_OVERLAY_K_AUTO=1 (default).
#   VIZ_OVERLAY_K_AUTO=1        — run the elbow selector (DEFAULT=1) so
#                                 multi-interval DIRECT picks its own k
#                                 and the plot shows multiple decision
#                                 boundaries when they exist. Set to 0
#                                 to pin k to VIZ_OVERLAY_K.
#   VIZ_OVERLAY_K_MAX=4         — k_max for the elbow.
#   VIZ_OVERLAY_COMPARE=1       — also run the legacy single-threshold
#                                 DIRECT (k=1) and render both series
#                                 on the same plot in different colors,
#                                 so the reader can see exactly where
#                                 multi-k adds a new decision boundary.
#                                 Default 0 (multi-k only).
#   VIZ_OVERLAY_CLASS_BALANCE=1 — run the multi-interval DP with class-
#                                 balanced weights (positives scaled by
#                                 N_neg/N_pos) so secondary positive
#                                 modes that drown in negatives — e.g.
#                                 animals_q1's low-similarity zebra
#                                 cluster — still surface as net-
#                                 positive intervals. Default 1; set
#                                 to 0 to recover the unweighted
#                                 sum-of-w accuracy objective.
#   VIZ_OVERLAY_SEED=0          — RNG seed for reproducibility.
# ---------------------------------------------------------------------------


def _run_tier_a_direct_on_sims(similarities: np.ndarray,
                               labels: np.ndarray,
                               budget: int,
                               k_intervals: int = 2,
                               k_auto: bool = False,
                               k_max: int = 4,
                               seed: int = 0,
                               class_balance: bool = True) -> dict:
    """Run the real sample_direct on a (sims, labels) fixture.

    Builds a synthetic ``label_fn`` backed by the ground-truth labels —
    this hands every LLM call back to NumPy with zero cost, so the
    overlay runs in milliseconds even for the large JoinML cross
    products. Returns a dict with:

        sim_intervals:  list of (sim_lo, sim_hi) on the cosine-similarity
                        axis (what the reader sees on the histogram).
        rank_intervals: list of (rank_left, rank_right) in sorted-axis
                        positions (descending sim; 1-indexed inclusive).
        sampled_indices: np.ndarray of indices into `similarities` that
                        DIRECT queried.
        chosen_k, dp_acc, vs_spend, qa_spend: profiling scalars.
        f1_on_sample:   F1 score of the interval-union hypothesis against
                        ground truth on the full (sims, labels) array.
    """
    from cluster_proxy import sample_direct as _sample_direct  # local import; cluster_proxy is heavy

    similarities = np.asarray(similarities, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    n = int(len(similarities))
    # The sampler consumes `texts` opaquely; we encode the original index
    # into a short string so the oracle can look up the ground-truth label.
    texts = [f"row_{i}" for i in range(n)]

    def _label_fn(xs):
        out = np.empty(len(xs), dtype=np.int32)
        for k_, t in enumerate(xs):
            # texts are guaranteed "row_{i}" — robust to batch reordering.
            idx = int(str(t).split("_", 1)[1])
            out[k_] = int(labels[idx])
        return out

    # A dummy 8-dim embedding is enough for sample_direct's knn_k auto logic.
    rng = np.random.default_rng(int(seed))
    emb = rng.standard_normal((n, 8)).astype(np.float32)

    # Match the semantics of the opt-in multi-interval path.
    kwargs = dict(B1=max(1, budget // 20), vs_ratio=0.5,
                  class_balance=bool(class_balance))
    if k_auto:
        kwargs.update(k_auto=True, k_max=int(k_max))
    else:
        kwargs.update(k_intervals=int(k_intervals))

    np.random.seed(int(seed))
    sampled_idx, sampled_vals, _W, prof = _sample_direct(
        emb, similarities, texts, _label_fn, sample_size=int(budget),
        **kwargs,
    )

    # Convert sorted-axis intervals -> similarity-axis intervals.
    # sort_order inside sample_direct is argsort(-similarities) — so
    # position 0 on the sorted axis is the highest similarity.
    sort_order = np.argsort(-similarities, kind="stable")
    sorted_sims = similarities[sort_order]

    sim_intervals = []
    rank_intervals = []
    for (a, b) in prof.get("intervals_sorted_space", []):
        a = int(a); b = int(b)
        sim_hi = float(sorted_sims[a])
        sim_lo = float(sorted_sims[b])
        sim_intervals.append((sim_lo, sim_hi))
        rank_intervals.append((a + 1, b + 1))  # 1-indexed for plotting

    # F1 of the interval-union hypothesis (positive iff sim ∈ any interval)
    # evaluated against ground-truth labels. This is a cheap proxy for how
    # well the DIRECT-discovered boundaries separate the classes on this
    # query — independent of the Correct & Smooth propagation step.
    pred = np.zeros(n, dtype=np.int32)
    for (lo, hi) in sim_intervals:
        pred |= ((similarities >= lo) & (similarities <= hi)).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    f1 = (2 * tp) / max(1, 2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "sim_intervals": sim_intervals,
        "rank_intervals": rank_intervals,
        "sampled_indices": np.asarray(sampled_idx, dtype=np.int64),
        "sampled_n": int(len(sampled_idx)),
        "chosen_k": int(prof.get("chosen_k", 0)),
        "dp_acc": float(prof.get("dp_acc", 0.0)),
        "vs_spend": int(prof.get("vs_spend", 0)),
        "qa_spend": int(prof.get("qa_spend", 0)),
        "f1_on_sample": float(f1),
    }


def _maybe_overlay_direct_boundaries(query_tag: str,
                                     similarities: np.ndarray,
                                     labels: np.ndarray) -> dict | None:
    """Gate the overlay on VIZ_OVERLAY_DIRECT=1.

    Returns a metadata dict ready to hand to ``plot_distribution`` — or
    ``None`` when the overlay is disabled or the input has no variation.
    In compare mode (``VIZ_OVERLAY_COMPARE=1``) the result carries TWO
    series so the plotter renders both the legacy single-threshold
    DIRECT and the Tier-A multi-interval DIRECT on the same figure;
    otherwise only the multi-interval series is drawn.

    Side effect: stashes the overlay under
    ``plot_distribution._last_overlay_meta`` so the summary CSV writer
    can pick up the columns without an extra parameter threading
    through every viz_* function.
    """
    if os.environ.get("VIZ_OVERLAY_DIRECT", "0") != "1":
        plot_distribution._last_overlay_meta = None  # type: ignore[attr-defined]
        return None
    n = int(len(similarities))
    if n == 0:
        return None
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        print(f"  [{query_tag}] overlay skipped (degenerate labels)")
        plot_distribution._last_overlay_meta = None  # type: ignore[attr-defined]
        return None

    budget = int(os.environ.get("VIZ_OVERLAY_BUDGET", "200"))
    # Default to k_auto=True so every overlay uses the full Tier-A
    # pipeline and multiple decision boundaries get drawn whenever the
    # data supports them. Set VIZ_OVERLAY_K_AUTO=0 to pin to a fixed k.
    k_auto = os.environ.get("VIZ_OVERLAY_K_AUTO", "1") == "1"
    k_intervals = int(os.environ.get("VIZ_OVERLAY_K", "2"))
    k_max = int(os.environ.get("VIZ_OVERLAY_K_MAX", "4"))
    seed = int(os.environ.get("VIZ_OVERLAY_SEED", "0"))
    compare = os.environ.get("VIZ_OVERLAY_COMPARE", "0") == "1"
    # Default class_balance=True for the viz overlay so low-selectivity
    # multi-modal queries (animals_q1, ecomm_q13, etc.) surface their
    # secondary positive modes; set VIZ_OVERLAY_CLASS_BALANCE=0 to use
    # the unweighted accuracy objective.
    class_balance = os.environ.get("VIZ_OVERLAY_CLASS_BALANCE", "1") == "1"

    print(f"  [{query_tag}] running Tier-A DIRECT overlay "
          f"(budget={budget}, k_intervals={k_intervals}, "
          f"k_auto={k_auto}, k_max={k_max}, compare={compare}, "
          f"class_balance={class_balance})")

    # Primary series: Tier-A multi-interval DIRECT.
    primary = _run_tier_a_direct_on_sims(
        similarities, labels,
        budget=budget, k_intervals=k_intervals,
        k_auto=k_auto, k_max=k_max, seed=seed,
        class_balance=class_balance,
    )
    primary_label = (
        f"Tier-A multi-k (k_auto, found {primary['chosen_k']})"
        if k_auto else f"Tier-A multi-k (k={k_intervals})"
    )

    series: list = []
    baseline: dict | None = None
    if compare:
        # Baseline: legacy single-threshold DIRECT (k=1) with the
        # unweighted sum-of-w objective, so the comparison reflects
        # what BOTH multi-k and class-balanced rebalancing buy over
        # the original DIRECT formulation. Same budget + seed.
        baseline = _run_tier_a_direct_on_sims(
            similarities, labels,
            budget=budget, k_intervals=1, k_auto=False,
            k_max=1, seed=seed, class_balance=False,
        )
        series.append({
            "label": "Single-threshold DIRECT (k=1)",
            "color": "#e76f51",   # warm red, contrasts with multi-k purple
            "linestyle": "--",
            "intervals": baseline["sim_intervals"],
        })
    series.append({
        "label": primary_label,
        "color": "#6a4c93",       # Tier-A purple
        "linestyle": ":",
        "intervals": primary["sim_intervals"],
    })

    meta = {
        "query": query_tag,
        # Backwards-compat shallow fields — still populated from the
        # multi-k run so downstream consumers (summary CSV, any
        # external code that reads `.sim_intervals` / `.chosen_k`)
        # keep working.
        "sim_intervals": primary["sim_intervals"],
        "rank_intervals": primary["rank_intervals"],
        "sampled_indices": primary["sampled_indices"],
        "sampled_n": primary["sampled_n"],
        "chosen_k": primary["chosen_k"],
        "dp_acc": primary["dp_acc"],
        "f1_on_sample": primary["f1_on_sample"],
        # New compare-mode fields: both runs exposed so the summary
        # CSV can record absolute + relative improvement of multi-k
        # over the single-threshold baseline.
        "primary": primary,
        "baseline": baseline,
        "compare_mode": compare,
        "series": series,
    }
    plot_distribution._last_overlay_meta = meta  # type: ignore[attr-defined]

    print(f"  [{query_tag}] multi-k intervals (sim): "
          f"{[(round(lo,3), round(hi,3)) for lo,hi in primary['sim_intervals']]}  "
          f"chosen_k={primary['chosen_k']}  f1={primary['f1_on_sample']:.3f}")
    if baseline is not None:
        print(f"  [{query_tag}] k=1 baseline intervals (sim): "
              f"{[(round(lo,3), round(hi,3)) for lo,hi in baseline['sim_intervals']]}  "
              f"f1={baseline['f1_on_sample']:.3f}  "
              f"(multi-k Δf1 = {primary['f1_on_sample'] - baseline['f1_on_sample']:+.3f})")
    return meta


def _plot_with_optional_overlay(query_tag: str,
                                 similarities: np.ndarray,
                                 labels: np.ndarray,
                                 summary_rows: list,
                                 *,
                                 true_total_pairs: int | None = None,
                                 true_n_pos: int | None = None) -> None:
    """Shim that every viz_* caller can use in place of plot_distribution.

    Runs the Tier-A overlay when VIZ_OVERLAY_DIRECT=1, then dispatches to
    the core plotter with the resulting boundaries / sampled indices so
    the output .png matches the original layout when the flag is off.
    """
    meta = _maybe_overlay_direct_boundaries(query_tag, similarities, labels)
    plot_distribution(
        query_tag, similarities, labels, summary_rows,
        true_total_pairs=true_total_pairs, true_n_pos=true_n_pos,
        boundaries=meta["sim_intervals"] if meta else None,
        sampled_indices=meta["sampled_indices"] if meta else None,
    )


def subsample_pairs(n_left: int, n_right: int, max_pairs: int, exclude_self: bool = False):
    """Return (left_idx, right_idx) arrays of at most max_pairs random pairs."""
    rng = np.random.default_rng(42)
    total = n_left * n_right
    if total <= max_pairs:
        li = np.repeat(np.arange(n_left), n_right)
        ri = np.tile(np.arange(n_right), n_left)
    else:
        flat = rng.choice(total, size=max_pairs, replace=False)
        li = flat // n_right
        ri = flat % n_right
    if exclude_self:
        keep = li != ri
        li = li[keep]
        ri = ri[keep]
    return li, ri


# ---------------------------------------------------------------------------
# Per-query data loaders + viz
# ---------------------------------------------------------------------------

def viz_movie_q5(summary_rows):
    sp = get_spark()
    csv_path = _HOME + "/SemBench/files/movie/data/sf_1000/Reviews.csv"
    reviews = sp.read.option("header", "true").option("multiLine", "true").option("escape", '"').csv(csv_path)
    reviews = reviews.filter(
        (col("id") == "ant_man_and_the_wasp_quantumania") & col("reviewText").isNotNull()
    )
    pdf = reviews.select("reviewId", "reviewText").toPandas()
    rids = pdf["reviewId"].astype(str).tolist()
    texts = pdf["reviewText"].astype(str).tolist()
    n = len(texts)
    print(f"[movie_q5] embedding {n} reviews ...")
    row_emb = embed_texts(texts, EMBEDDING_MODEL)

    li, ri = subsample_pairs(n, n, MAX_PAIRS_PER_QUERY, exclude_self=True)
    left_emb = row_emb[li]
    right_emb = row_emb[ri]
    sims = cosine_pair(left_emb, right_emb)

    gt_path = _HOME + "/SemBench/files/movie/raw_results/ground_truth/Q5.csv"
    gt_df = pd.read_csv(gt_path)
    gt_pairs = set()
    for _, r in gt_df.iterrows():
        cols = list(gt_df.columns)
        v1, v2 = str(r[cols[1]]), str(r[cols[2]])
        gt_pairs.add(tuple(sorted([v1, v2])))

    labels = np.zeros(len(sims), dtype=np.int32)
    for k, (i, j) in enumerate(zip(li, ri)):
        pair = tuple(sorted([rids[i], rids[j]]))
        if pair in gt_pairs:
            labels[k] = 1
    plot_distribution("movie_q5", sims, labels, summary_rows)


def viz_movie_q7(summary_rows):
    sp = get_spark()
    csv_path = _HOME + "/SemBench/files/movie/data/sf_1000/Reviews.csv"
    reviews = sp.read.option("header", "true").option("multiLine", "true").option("escape", '"').csv(csv_path)
    reviews = reviews.filter(
        (col("id") == "ant_man_and_the_wasp_quantumania") & col("reviewText").isNotNull()
    )
    pdf = reviews.select("reviewId", "reviewText").toPandas()
    rids = pdf["reviewId"].astype(str).tolist()
    texts = pdf["reviewText"].astype(str).tolist()
    n = len(texts)
    print(f"[movie_q7] embedding {n} reviews ...")
    row_emb = embed_texts(texts, EMBEDDING_MODEL)

    li, ri = subsample_pairs(n, n, MAX_PAIRS_PER_QUERY, exclude_self=True)
    sims = cosine_pair(row_emb[li], row_emb[ri])

    gt_path = _HOME + "/SemBench/files/movie/raw_results/ground_truth/Q7.csv"
    gt_df = pd.read_csv(gt_path)
    gt_pairs = set()
    for _, r in gt_df.iterrows():
        cols = list(gt_df.columns)
        v1, v2 = str(r[cols[1]]), str(r[cols[2]])
        gt_pairs.add(tuple(sorted([v1, v2])))

    labels = np.zeros(len(sims), dtype=np.int32)
    for k, (i, j) in enumerate(zip(li, ri)):
        pair = tuple(sorted([rids[i], rids[j]]))
        if pair in gt_pairs:
            labels[k] = 1
    plot_distribution("movie_q7", sims, labels, summary_rows)


def viz_ecomm_q7(summary_rows):
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/ecomm/data/sf_4000"
    df = sp.read.parquet(os.path.join(data_dir, "styles_details.parquet"))
    df = df.filter(col("price") <= 500)
    df = df.withColumn("description_text", col("productDescriptors.description.value"))
    complex_fields = [f.name for f in df.schema.fields if isinstance(f.dataType, (MapType, StructType, ArrayType))]
    df = df.drop(*complex_fields)
    pdf = df.select("id", "productDisplayName", "description_text").toPandas()
    ids = pdf["id"].astype(str).tolist()
    texts = (pdf["productDisplayName"].fillna("") + " - " + pdf["description_text"].fillna("")).tolist()
    n = len(texts)
    print(f"[ecomm_q7] embedding {n} products ...")
    row_emb = embed_texts(texts, EMBEDDING_MODEL)

    li, ri = subsample_pairs(n, n, MAX_PAIRS_PER_QUERY, exclude_self=False)
    sims = cosine_pair(row_emb[li], row_emb[ri])

    gt_path = _HOME + "/SemBench/files/ecomm/raw_results/ground_truth/Q7.csv"
    gt_df = pd.read_csv(gt_path)
    gt_set = set()
    for _, r in gt_df.iterrows():
        s = str(r.iloc[0]).strip()
        if "-" in s:
            a, b = s.split("-", 1)
            gt_set.add((a.strip(), b.strip()))

    labels = np.zeros(len(sims), dtype=np.int32)
    for k, (i, j) in enumerate(zip(li, ri)):
        if (ids[i], ids[j]) in gt_set or (ids[j], ids[i]) in gt_set:
            labels[k] = 1
    plot_distribution("ecomm_q7", sims, labels, summary_rows)


def viz_cars_q10(summary_rows):
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/cars/data/sf_157376"
    cars = sp.read.option("header", "true").option("multiLine", "true").option("escape", '"').csv(
        os.path.join(data_dir, "car_data_157376.csv")
    )
    complaints = sp.read.option("header", "true").option("multiLine", "true").option("escape", '"').csv(
        os.path.join(data_dir, "text_complaints_data_157376.csv")
    )
    cars = cars.withColumn("registration_date", col("registration_date").cast("string"))
    joined = cars.join(complaints, on="car_id", how="inner").select("car_id", "summary").toPandas()
    cids = joined["car_id"].astype(str).tolist()
    texts = joined["summary"].astype(str).fillna("").tolist()
    n = len(texts)
    print(f"[cars_q10] embedding {n} complaints ...")
    complaint_emb = embed_texts(texts, EMBEDDING_MODEL)

    categories = [
        "ELECTRICAL SYSTEM", "POWER TRAIN", "ENGINE", "STEERING", "SERVICE BRAKES",
        "STRUCTURE", "AIR BAGS", "ENGINE AND ENGINE COOLING", "VEHICLE SPEED CONTROL",
        "VISIBILITY/WIPER", "FUEL/PROPULSION SYSTEM", "FORWARD COLLISION AVOIDANCE",
        "EXTERIOR LIGHTING", "SUSPENSION", "FUEL SYSTEM", "VISIBILITY", "WHEELS",
        "SEAT BELTS", "BACK OVER PREVENTION", "TIRES", "SEATS", "LATCHES/LOCKS/LINKAGES",
        "LANE DEPARTURE", "EQUIPMENT",
    ]
    cat_emb = embed_texts(categories, EMBEDDING_MODEL)
    K = len(categories)

    rng = np.random.default_rng(42)
    if n * K > MAX_PAIRS_PER_QUERY:
        sampled = rng.choice(n * K, size=MAX_PAIRS_PER_QUERY, replace=False)
        li = sampled // K
        ki = sampled % K
    else:
        li = np.repeat(np.arange(n), K)
        ki = np.tile(np.arange(K), n)
    sims = cosine_pair(complaint_emb[li], cat_emb[ki])

    gt_path = _HOME + "/SemBench/files/cars/raw_results/ground_truth/Q10.csv"
    gt_df = pd.read_csv(gt_path, dtype=str)
    gt_df.columns = [c.strip().lower() for c in gt_df.columns]
    gt_map = dict(zip(gt_df["car_id"].astype(str).str.strip(),
                      gt_df["problem_category"].astype(str).str.strip().str.lower()))
    cat_lower = [c.strip().lower() for c in categories]

    labels = np.zeros(len(sims), dtype=np.int32)
    for k_, (i, kk) in enumerate(zip(li, ki)):
        true_cat = gt_map.get(cids[i])
        if true_cat is not None and true_cat == cat_lower[kk]:
            labels[k_] = 1
    plot_distribution("cars_q10", sims, labels, summary_rows)


def viz_cars_q8(summary_rows):
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/cars/data/sf_157376"
    car_data = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(os.path.join(data_dir, "car_data_157376.csv"))
    image_data = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(os.path.join(data_dir, "image_car_data_157376.csv"))

    car_data = car_data.withColumn("registration_date", col("registration_date").cast("string"))
    image_data = image_data.withColumn(
        "image_path",
        concat(lit(_HOME + "/SemBench/"), col("image_path")),
    )
    merged = car_data.join(image_data, on="car_id", how="inner")
    pdf = merged.select("car_id", "image_path").toPandas()
    car_ids = pdf["car_id"].astype(str).tolist()
    img_paths = pdf["image_path"].astype(str).tolist()
    n = len(img_paths)
    print(f"[cars_q8] embedding {n} car images via SigLIP ...")
    img_emb = embed_images_siglip(img_paths, EMBEDDING_MODEL)

    prompt = (
        "You are given an image of a vehicle or its parts. "
        "Return true if car has both, puncture and paint scratches."
    )
    sims = cosine_to_prompt(img_emb, prompt)

    gt_path = _HOME + "/SemBench/files/cars/raw_results/ground_truth/Q8.csv"
    gt_df = pd.read_csv(gt_path, dtype=str)
    id_column = "car_id"

    def _normalize_id(val):
        s = str(val).strip()
        if s.lower() == "nan" or s == "":
            return None
        if re.fullmatch(r"\d+\.0", s):
            s = s.split(".")[0]
        return s if s.isdigit() else None

    gt_df[id_column] = gt_df[id_column].apply(_normalize_id)
    gt_df = gt_df.dropna(subset=[id_column])
    gt_ids = set(gt_df[id_column].tolist())

    normalized_car_ids = [_normalize_id(c) for c in car_ids]
    labels = np.array(
        [1 if cid is not None and cid in gt_ids else 0 for cid in normalized_car_ids],
        dtype=np.int32,
    )
    plot_distribution("cars_q8", sims, labels, summary_rows)


def viz_cars_q1(summary_rows):
    """Q1: "Find cars that were in a crash/accident/collision."

    Text sem_filter over complaint summaries. GT per SemBench evaluator:
    car_ids whose complaint has ``crash == True`` (same as Q1.csv).
    """
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/cars/data/sf_157376"
    complaints = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(os.path.join(data_dir, "text_complaints_data_157376.csv"))
    pdf = complaints.select("car_id", "summary").toPandas()
    car_ids = pdf["car_id"].astype(str).tolist()
    texts = pdf["summary"].astype(str).fillna("").tolist()
    n = len(texts)
    print(f"[cars_q1] embedding {n} complaints ...")
    text_emb = embed_texts(texts, EMBEDDING_MODEL)

    prompt = (
        "You are given a textual complaint entailing that the car was in a "
        "crash/accident/collision."
    )
    sims = cosine_to_prompt(text_emb, prompt)

    gt_path = _HOME + "/SemBench/files/cars/raw_results/ground_truth/Q1.csv"
    gt_df = pd.read_csv(gt_path, dtype=str)
    id_column = "car_id"

    def _normalize_id(val):
        s = str(val).strip()
        if s.lower() == "nan" or s == "":
            return None
        if re.fullmatch(r"\d+\.0", s):
            s = s.split(".")[0]
        return s if s.isdigit() else None

    gt_df[id_column] = gt_df[id_column].apply(_normalize_id)
    gt_df = gt_df.dropna(subset=[id_column])
    gt_ids = set(gt_df[id_column].tolist())

    normalized_car_ids = [_normalize_id(c) for c in car_ids]
    labels = np.array(
        [1 if cid is not None and cid in gt_ids else 0 for cid in normalized_car_ids],
        dtype=np.int32,
    )
    plot_distribution("cars_q1", sims, labels, summary_rows)


def viz_cars_q3(summary_rows):
    """Q3: "Find ten cars with manual transmission that are not damaged
    according to images. Return the VIN."

    Multimodal filter over Manual cars' images. GT per SemBench evaluator:
    VINs of manual-transmission cars that have at least one image with
    ``damage_status == 'no_damage'`` (Q3.csv, pre-LIMIT full set).
    The similarity distribution plotted here is over the full pre-LIMIT
    pool so that separability is informative; the LIMIT 10 is only
    enforced at evaluation time.
    """
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/cars/data/sf_157376"
    car_data = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(os.path.join(data_dir, "car_data_157376.csv"))
    image_data = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(os.path.join(data_dir, "image_car_data_157376.csv"))

    car_data = car_data.withColumn("registration_date", col("registration_date").cast("string"))
    image_data = image_data.withColumn(
        "image_path",
        concat(lit(_HOME + "/SemBench/"), col("image_path")),
    )
    manual_cars = car_data.filter(col("transmission") == lit("Manual"))
    merged = manual_cars.join(image_data, on="car_id", how="inner")

    pdf = merged.select("vin", "image_path").toPandas()
    vins = pdf["vin"].astype(str).str.strip().tolist()
    img_paths = pdf["image_path"].astype(str).tolist()
    n = len(img_paths)
    print(f"[cars_q3] embedding {n} manual-car images via SigLIP ...")
    img_emb = embed_images_siglip(img_paths, EMBEDDING_MODEL)

    prompt = (
        "You are given an image of a vehicle or its parts. "
        "Return true if the car is not damaged."
    )
    sims = cosine_to_prompt(img_emb, prompt)

    gt_path = _HOME + "/SemBench/files/cars/raw_results/ground_truth/Q3.csv"
    gt_df = pd.read_csv(gt_path, dtype=str)
    gt_col = "vin" if "vin" in gt_df.columns else gt_df.columns[0]
    gt_vins = {str(v).strip() for v in gt_df[gt_col].dropna() if str(v).strip()}

    labels = np.array(
        [1 if v in gt_vins else 0 for v in vins],
        dtype=np.int32,
    )
    plot_distribution("cars_q3", sims, labels, summary_rows)


def viz_cars_q4(summary_rows):
    """Q4: "What is the average age of cars with engine problems?"

    Text sem_filter over complaint summaries. Q4's SemBench GT is a
    scalar (average_age), so we derive the per-row GT the same way the
    SemBench evaluator does:
    ``component_class == 'ENGINE'`` → positive car_ids.

    The scale-factor-sampled complaints csv only contains
    ``[summary, complaint_id, car_id]``; the label column
    ``component_class`` lives in ``full_data/text_complaints_data_full.csv``
    so we join by complaint_id (exactly what CarsEvaluator._load_domain_data
    does).
    """
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/cars/data/sf_157376"
    full_data_dir = _HOME + "/SemBench/files/cars/data/full_data"
    complaints = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(os.path.join(data_dir, "text_complaints_data_157376.csv"))
    full_complaints = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(os.path.join(full_data_dir, "text_complaints_data_full.csv"))
    labelled = complaints.select("car_id", "complaint_id", "summary").join(
        full_complaints.select("complaint_id", "component_class"),
        on="complaint_id", how="inner",
    )
    pdf = labelled.select("car_id", "summary", "component_class").toPandas()
    car_ids = pdf["car_id"].astype(str).tolist()
    texts = pdf["summary"].astype(str).fillna("").tolist()
    component_classes = pdf["component_class"].astype(str).str.strip().tolist()
    n = len(texts)
    print(f"[cars_q4] embedding {n} complaints ...")
    text_emb = embed_texts(texts, EMBEDDING_MODEL)

    prompt = (
        "In the complaint, the car has some problems with engine / "
        "connected to engine."
    )
    sims = cosine_to_prompt(text_emb, prompt)

    labels = np.array(
        [1 if cc == "ENGINE" else 0 for cc in component_classes],
        dtype=np.int32,
    )
    plot_distribution("cars_q4", sims, labels, summary_rows)


def viz_ecomm_q6(summary_rows):
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/ecomm/data/sf_4000"
    styles = sp.read.parquet(os.path.join(data_dir, "styles_details.parquet"))
    image_map = sp.read.parquet(os.path.join(data_dir, "image_mapping.parquet"))
    styles = styles.filter(
        (col("masterCategory.typeName") == "Apparel")
        & (~col("subCategory.typeName").isin("Saree", "Apparel Set", "Loungewear and Nightwear"))
    )
    image_map = image_map.join(
        styles.select(col("id").cast("string").alias("style_id")),
        image_map["id"].cast("string") == col("style_id"),
        "inner",
    ).drop("style_id")
    image_map = image_map.withColumn(
        "image_filepath", concat(lit(data_dir + "/images/"), col("filename"))
    )
    pdf = image_map.select("id", "image_filepath").toPandas()
    img_ids = pdf["id"].astype(str).tolist()
    img_paths = pdf["image_filepath"].astype(str).tolist()
    n = len(img_paths)
    print(f"[ecomm_q6] embedding {n} images via SigLIP ...")
    img_emb = embed_images_siglip(img_paths, EMBEDDING_MODEL)

    categories = ["Dress", "Bottomwear", "Socks", "Topwear", "Innerwear"]
    cat_emb = embed_texts(categories, EMBEDDING_MODEL)
    K = len(categories)

    li = np.repeat(np.arange(n), K)
    ki = np.tile(np.arange(K), n)
    sims = cosine_pair(img_emb[li], cat_emb[ki])

    gt_path = _HOME + "/SemBench/files/ecomm/raw_results/ground_truth/Q6.csv"
    gt_df = pd.read_csv(gt_path)
    gt_df.columns = [c.strip() for c in gt_df.columns]
    id_col, cat_col = gt_df.columns[0], gt_df.columns[1]
    gt_map = dict(zip(gt_df[id_col].astype(str).str.strip(),
                      gt_df[cat_col].astype(str).str.strip().str.lower()))
    cat_lower = [c.strip().lower() for c in categories]

    labels = np.zeros(len(sims), dtype=np.int32)
    for k_, (i, kk) in enumerate(zip(li, ki)):
        true_cat = gt_map.get(img_ids[i])
        if true_cat is not None and true_cat == cat_lower[kk]:
            labels[k_] = 1
    plot_distribution("ecomm_q6", sims, labels, summary_rows)


def viz_ecomm_q13(summary_rows):
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/ecomm/data/sf_4000"
    styles = sp.read.parquet(os.path.join(data_dir, "styles_details.parquet"))
    image_map = sp.read.parquet(os.path.join(data_dir, "image_mapping.parquet"))
    styles = styles.withColumn("description_text", col("productDescriptors.description.value"))
    complex_fields = [f.name for f in styles.schema.fields if isinstance(f.dataType, (MapType, StructType, ArrayType))]
    styles = styles.drop(*complex_fields)
    styles = styles.withColumn("id_str", col("id").cast("string"))
    image_map = image_map.withColumn("image_id_str", col("id").cast("string"))
    merged = styles.join(
        image_map.select("image_id_str", "filename"),
        styles["id_str"] == image_map["image_id_str"], "inner",
    ).drop("image_id_str")
    merged = merged.withColumn("image_filepath", concat(lit(data_dir + "/images/"), col("filename")))
    pdf = merged.select("id", "image_filepath").toPandas()
    ids = pdf["id"].astype(str).tolist()
    img_paths = pdf["image_filepath"].astype(str).tolist()
    n = len(img_paths)
    print(f"[ecomm_q13] embedding {n} images via SigLIP ...")
    img_emb = embed_images_siglip(img_paths, EMBEDDING_MODEL)

    prompt = (
        "I am looking for a running shirt for men with a round neck and short sleeves, "
        "preferably in blue or black, but not bright colors like white. "
        "Also definitely not green. It should be suitable for outdoor running in warm weather."
    )
    sims = cosine_to_prompt(img_emb, prompt)

    gt_path = _HOME + "/SemBench/files/ecomm/raw_results/ground_truth/Q13.csv"
    gt_df = pd.read_csv(gt_path)
    gt_ids = set(str(x).strip() for x in gt_df.iloc[:, 0].tolist())

    labels = np.array([1 if i in gt_ids else 0 for i in ids], dtype=np.int32)
    plot_distribution("ecomm_q13", sims, labels, summary_rows)


def viz_mmqa_q2a(summary_rows):
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/mmqa/data/sf_800"
    ap_df = sp.read.option("header", "true").csv(os.path.join(data_dir, "ap_warrior.csv"))
    img_df = sp.read.option("header", "true").csv(os.path.join(data_dir, "thalamusdb_images.csv"))
    img_df_pdf = img_df.select("image_filename").toPandas()
    img_paths = [os.path.join(data_dir, "images", str(p)) for p in img_df_pdf["image_filename"]]
    img_filenames = img_df_pdf["image_filename"].astype(str).tolist()

    ap_pdf = ap_df.toPandas()
    track_col = "Track" if "Track" in ap_pdf.columns else ap_pdf.columns[1]
    id_col = "ID" if "ID" in ap_pdf.columns else ap_pdf.columns[0]
    track_texts = ap_pdf[track_col].astype(str).tolist()
    track_ids = ap_pdf[id_col].astype(str).tolist()
    n_left = len(track_texts)
    n_right = len(img_paths)
    print(f"[mmqa_q2a] embedding {n_left} tracks (text) + {n_right} images ...")
    track_emb = embed_texts(track_texts, EMBEDDING_MODEL)
    img_emb = embed_images_siglip(img_paths, EMBEDDING_MODEL)

    li = np.repeat(np.arange(n_left), n_right)
    ri = np.tile(np.arange(n_right), n_left)
    sims = cosine_pair(track_emb[li], img_emb[ri])

    gt_set = {
        (0, "117d500aaa630023c4038b8268b309c0.png"),
        (5, "117d500aaa630023c4038b8268b309c0.png"),
        (6, "117d500aaa630023c4038b8268b309c0.png"),
        (9, "117d500aaa630023c4038b8268b309c0.png"),
        (10, "117d500aaa630023c4038b8268b309c0.png"),
    }
    labels = np.zeros(len(sims), dtype=np.int32)
    for k_, (i, j) in enumerate(zip(li, ri)):
        try:
            tid = int(track_ids[i])
        except (ValueError, TypeError):
            continue
        if (tid, img_filenames[j]) in gt_set:
            labels[k_] = 1
    plot_distribution("mmqa_q2a", sims, labels, summary_rows)


def viz_mmqa_q7(summary_rows):
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/mmqa/data/sf_800"
    air_df = sp.read.option("header", "true").csv(os.path.join(data_dir, "tampa_international_airport.csv"))
    img_df = sp.read.option("header", "true").csv(os.path.join(data_dir, "thalamusdb_images.csv"))
    img_df_pdf = img_df.select("image_filename").toPandas()
    img_paths = [os.path.join(data_dir, "images", str(p)) for p in img_df_pdf["image_filename"]]
    img_filenames = img_df_pdf["image_filename"].astype(str).tolist()

    air_pdf = air_df.toPandas()
    air_pdf.columns = [c.strip() for c in air_pdf.columns]
    # Dedup by Airlines: the source CSV is one row per (airline, destination)
    # so each airline appears multiple times. gt_set keys on airline name
    # alone, so without dedup n_pos inflates by the #destinations-per-airline
    # factor (e.g. 5 GT airlines * 3 rows each = 15 "positives").
    air_pdf = air_pdf.drop_duplicates(subset=["Airlines"]).reset_index(drop=True)
    airline_col = "Airlines"
    dest_col = "Destinations"
    air_texts = (air_pdf[airline_col].astype(str) + " - " + air_pdf[dest_col].astype(str)).tolist()
    airline_names = air_pdf[airline_col].astype(str).tolist()
    n_left = len(air_texts)
    n_right = len(img_paths)
    print(f"[mmqa_q7] embedding {n_left} airline rows (text) + {n_right} images ...")
    text_emb = embed_texts(air_texts, EMBEDDING_MODEL)
    img_emb = embed_images_siglip(img_paths, EMBEDDING_MODEL)

    li = np.repeat(np.arange(n_left), n_right)
    ri = np.tile(np.arange(n_right), n_left)
    sims = cosine_pair(text_emb[li], img_emb[ri])

    gt_set = {
        ("British Airways", "cwncgabxti09zmf36t4phvaz1xzv10wu.png"),
        ("Delta Air Lines", "1sz0kf8wcmj0q8n3pu6mg61gl158vvz1.png"),
        ("Discover Airlines", "nqr1pjql5qs3dp8rz2a4zzq7rn0xyp6k.png"),
        ("Edelweiss Air", "6zpijbg5jv4jftpftpmla6xbizne28d9.png"),
        ("Virgin Atlantic", "antbx4oxst0z5o6pe2g1thrjr0dz073j.png"),
    }
    labels = np.zeros(len(sims), dtype=np.int32)
    for k_, (i, j) in enumerate(zip(li, ri)):
        if (airline_names[i], img_filenames[j]) in gt_set:
            labels[k_] = 1
    plot_distribution("mmqa_q7", sims, labels, summary_rows)


def viz_ecomm_q8(summary_rows):
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/ecomm/data/sf_4000"
    styles = sp.read.parquet(os.path.join(data_dir, "styles_details.parquet"))
    image_map = sp.read.parquet(os.path.join(data_dir, "image_mapping.parquet"))
    styles = styles.withColumn("description_text", col("productDescriptors.description.value"))
    complex_fields = [f.name for f in styles.schema.fields if isinstance(f.dataType, (MapType, StructType, ArrayType))]
    styles = styles.drop(*complex_fields)
    styles = styles.filter(length(col("description_text")) >= 3000)

    left_pdf = styles.select("id", "productDisplayName", "description_text").toPandas()
    style_ids = left_pdf["id"].astype(str).tolist()
    left_texts = (
        left_pdf["productDisplayName"].fillna("").astype(str)
        + " - "
        + left_pdf["description_text"].fillna("").astype(str)
    ).tolist()
    n_left = len(left_texts)

    right_pdf = image_map.select("id", "filename").toPandas()
    img_ids = right_pdf["id"].astype(str).tolist()
    img_paths = [os.path.join(data_dir, "images", str(p)) for p in right_pdf["filename"]]
    n_right = len(img_paths)

    print(f"[ecomm_q8] embedding {n_left} product texts + {n_right} images via SigLIP ...")
    text_emb = embed_texts(left_texts, EMBEDDING_MODEL)
    img_emb = embed_images_siglip(img_paths, EMBEDDING_MODEL)

    li, ri = subsample_pairs(n_left, n_right, MAX_PAIRS_PER_QUERY, exclude_self=False)
    sims = cosine_pair(text_emb[li], img_emb[ri])

    gt_path = _HOME + "/SemBench/files/ecomm/raw_results/ground_truth/Q8.csv"
    gt_df = pd.read_csv(gt_path)
    gt_set = set()
    for _, r in gt_df.iterrows():
        s = str(r.iloc[0]).strip()
        if "-" in s:
            a, b = s.split("-", 1)
            gt_set.add((a.strip(), b.strip()))

    labels = np.zeros(len(sims), dtype=np.int32)
    for k_, (i, j) in enumerate(zip(li, ri)):
        if (style_ids[i], img_ids[j]) in gt_set:
            labels[k_] = 1
    plot_distribution("ecomm_q8", sims, labels, summary_rows)


def viz_ecomm_q9(summary_rows):
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/ecomm/data/sf_4000"
    styles = sp.read.parquet(os.path.join(data_dir, "styles_details.parquet"))
    image_map = sp.read.parquet(os.path.join(data_dir, "image_mapping.parquet"))

    valid_colors = ["Black", "Blue", "Red", "White", "Orange", "Green"]
    # Aligned with run_ecomm_q9: keep only single-colour items
    # (colour1 == "" AND colour2 == "") so "same category + same colour"
    # is well-defined. Without these filters viz had ~12x more rows than
    # the runner, inflating the cross product accordingly.
    styles = styles.filter(
        col("baseColour").isin(valid_colors)
        & (col("colour1") == "")
        & (col("colour2") == "")
        & (col("price") < 800)
    )
    image_map = image_map.join(
        styles.select(col("id").cast("string").alias("style_id")),
        image_map["id"].cast("string") == col("style_id"),
        "inner",
    ).drop("style_id")

    pdf = image_map.select("id", "filename").toPandas()
    img_ids = pdf["id"].astype(str).tolist()
    img_paths = [os.path.join(data_dir, "images", str(p)) for p in pdf["filename"]]
    n = len(img_paths)
    print(f"[ecomm_q9] embedding {n} images via SigLIP ...")
    img_emb = embed_images_siglip(img_paths, EMBEDDING_MODEL)

    li, ri = subsample_pairs(n, n, MAX_PAIRS_PER_QUERY, exclude_self=True)
    sims = cosine_pair(img_emb[li], img_emb[ri])

    gt_path = _HOME + "/SemBench/files/ecomm/raw_results/ground_truth/Q9.csv"
    gt_df = pd.read_csv(gt_path)
    gt_set = set()
    for _, r in gt_df.iterrows():
        s = str(r.iloc[0]).strip()
        if "-" in s:
            a, b = s.split("-", 1)
            gt_set.add((a.strip(), b.strip()))

    labels = np.zeros(len(sims), dtype=np.int32)
    for k_, (i, j) in enumerate(zip(li, ri)):
        a, b = img_ids[i], img_ids[j]
        if (a, b) in gt_set or (b, a) in gt_set:
            labels[k_] = 1
    plot_distribution("ecomm_q9", sims, labels, summary_rows)


def viz_ecomm_q9_prompt_fused(summary_rows):
    """Alternate DIRECT axis: cos(prompt_emb, (left_img + right_img) / 2).

    Matches the behaviour of join_sim_mode="prompt_vs_fused" in
    proxy_model.py's proxy_join. If this axis separates GT positives
    from negatives better than raw cos(left, right), setting
    JOIN_SIM_MODE["ecomm_q9"]="prompt_vs_fused" in the grid should
    improve DIRECT's F1 on this query.
    """
    sp = get_spark()
    data_dir = _HOME + "/SemBench/files/ecomm/data/sf_4000"
    styles = sp.read.parquet(os.path.join(data_dir, "styles_details.parquet"))
    image_map = sp.read.parquet(os.path.join(data_dir, "image_mapping.parquet"))

    valid_colors = ["Black", "Blue", "Red", "White", "Orange", "Green"]
    styles = styles.filter(
        (col("baseColour").isin(valid_colors)) & (col("price") < 800)
    )
    image_map = image_map.join(
        styles.select(col("id").cast("string").alias("style_id")),
        image_map["id"].cast("string") == col("style_id"),
        "inner",
    ).drop("style_id")

    pdf = image_map.select("id", "filename").toPandas()
    img_ids = pdf["id"].astype(str).tolist()
    img_paths = [os.path.join(data_dir, "images", str(p)) for p in pdf["filename"]]
    n = len(img_paths)
    print(f"[ecomm_q9_prompt_fused] embedding {n} images via SigLIP ...")
    img_emb = embed_images_siglip(img_paths, EMBEDDING_MODEL)

    prompt = (
        "Determine whether both images display objects of the same category "
        "and share the same dominant surface color."
    )
    prompt_emb = embed_texts([prompt], EMBEDDING_MODEL).reshape(-1)
    d = min(len(prompt_emb), img_emb.shape[1])
    p_slice = prompt_emb[:d]
    img_emb_slice = img_emb[:, :d]
    norm_p = float(np.linalg.norm(p_slice)) + 1e-8

    li, ri = subsample_pairs(n, n, MAX_PAIRS_PER_QUERY, exclude_self=True)
    fused = (img_emb_slice[li] + img_emb_slice[ri]) * 0.5
    dots = fused @ p_slice
    norms_fused = np.linalg.norm(fused, axis=1) + 1e-8
    sims = (dots / (norms_fused * norm_p)).astype(np.float32)

    gt_path = _HOME + "/SemBench/files/ecomm/raw_results/ground_truth/Q9.csv"
    gt_df = pd.read_csv(gt_path)
    gt_set = set()
    for _, r in gt_df.iterrows():
        s = str(r.iloc[0]).strip()
        if "-" in s:
            a, b = s.split("-", 1)
            gt_set.add((a.strip(), b.strip()))

    labels = np.zeros(len(sims), dtype=np.int32)
    for k_, (i, j) in enumerate(zip(li, ri)):
        a, b = img_ids[i], img_ids[j]
        if (a, b) in gt_set or (b, a) in gt_set:
            labels[k_] = 1
    plot_distribution("ecomm_q9_prompt_fused", sims, labels, summary_rows)


# ---------------------------------------------------------------------------
# Animals benchmark (SemBench Q1 / Q3 / Q10).
#
# All three queries apply the same sem_filter predicate
#   "Does the image contain a zebra?"  over   animals/.../image_data.csv
# and differ only in the post-filter aggregation (count / top-City /
# top-(City,StationID)). The 1-D similarity axis DIRECT sorts on is
# therefore the SAME for all three, so we share one loader + embed path
# and emit three labelled entries (one per query tag) in the summary CSV
# for consistent slicing downstream.
#
# Per-row ground-truth "is zebra?" labels come from the best available
# column in image_data.csv (Species / Animal / Label / ...) with a
# filename-substring fallback for robustness.
# ---------------------------------------------------------------------------

_ANIMALS_PROMPT_FOR_VIZ = "Does the image contain a zebra?"


def _animals_load_and_label():
    """Return (img_paths, labels, column_used) for the animals image table."""
    sp = get_spark()
    scale = 1600
    data_dir = f_HOME + "/SemBench/files/animals/data/sf_{scale}"
    df = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("escape", '"') \
        .csv(os.path.join(data_dir, "image_data.csv"))
    pdf = df.toPandas()

    if "ImagePath" not in pdf.columns:
        raise RuntimeError(
            f"[animals] expected 'ImagePath' column in image_data.csv, got: {list(pdf.columns)}"
        )
    img_paths = pdf["ImagePath"].astype(str).tolist()

    # Find the best per-row zebra-label column. Try a few common names,
    # case-insensitively, and substring-match any value containing 'zebra'.
    label_candidates = [
        "Species", "species", "Animal", "animal", "Label", "label",
        "class", "Class", "Category", "category",
    ]
    label_col = next((c for c in label_candidates if c in pdf.columns), None)
    if label_col is not None:
        print(f"[animals] per-row labels from column '{label_col}'")
        series = pdf[label_col].astype(str).str.lower()
        labels = series.str.contains("zebra", regex=False, na=False).astype(int).to_numpy()
    else:
        print(
            f"[animals] no Species/Animal/Label/Category column found in "
            f"{list(pdf.columns)} — falling back to 'zebra' substring match on "
            "basename(ImagePath)"
        )
        labels = np.array(
            [1 if "zebra" in os.path.basename(p).lower() else 0 for p in img_paths],
            dtype=np.int32,
        )
    n_pos = int(labels.sum())
    print(f"[animals] total rows={len(img_paths)}, positives={n_pos}, "
          f"selectivity={n_pos / max(len(img_paths), 1):.1%}")
    return img_paths, labels, label_col


# Cache the embeddings+labels across the 3 viz calls since they're identical.
_ANIMALS_CACHE: dict = {"img_emb": None, "sims": None, "labels": None}


def _animals_compute_shared():
    """Compute (sims, labels) once; reuse for q1/q3/q10."""
    if _ANIMALS_CACHE["sims"] is not None:
        return _ANIMALS_CACHE["sims"], _ANIMALS_CACHE["labels"]
    img_paths, labels, _ = _animals_load_and_label()
    n = len(img_paths)
    print(f"[animals] embedding {n} images via SigLIP ...")
    img_emb = embed_images_siglip(img_paths, EMBEDDING_MODEL)
    sims = cosine_to_prompt(img_emb, _ANIMALS_PROMPT_FOR_VIZ)
    _ANIMALS_CACHE["img_emb"] = img_emb
    _ANIMALS_CACHE["sims"] = sims
    _ANIMALS_CACHE["labels"] = labels
    return sims, labels


def viz_animals_q1(summary_rows):
    sims, labels = _animals_compute_shared()
    plot_distribution("animals_q1", sims, labels, summary_rows)


def viz_animals_q3(summary_rows):
    sims, labels = _animals_compute_shared()
    plot_distribution("animals_q3", sims, labels, summary_rows)


def viz_animals_q10(summary_rows):
    sims, labels = _animals_compute_shared()
    plot_distribution("animals_q10", sims, labels, summary_rows)


# ---------------------------------------------------------------------------
# JoinML paper datasets (arXiv:2603.16153): VeRi, Flickr30K, Roxford, Company.
# Each one assumes the corresponding dataset/<name>_groundtruth.py has
# already produced <data_dir>/gt/{left.csv, right.csv, gt_pairwise.csv}.
# ---------------------------------------------------------------------------

def _viz_joinml_pair(query_tag, data_dir, left_name, right_name, gt_name,
                     left_kind, right_kind, left_col, right_col,
                     summary_rows):
    """Shared viz for the four JoinML joins.

    Sweeps the FULL |L|*|R| cross product via blocked matmul so the pos/neg
    similarity distributions reflect the whole dataset, not a subsample.
    Keeps every positive pair sim; Bernoulli-samples negatives at a fixed
    probability to hit VIZ_JOINML_NEG_SAMPLE on average.

    left_kind / right_kind: 'image' or 'text'.
    left_col / right_col: column holding image path (kind=='image') or
                          text string.
    """
    from scipy.sparse import csr_matrix

    gt_dir = Path(data_dir) / "gt"
    left_path = gt_dir / left_name
    right_path = gt_dir / right_name
    gt_path = gt_dir / gt_name
    missing = [p for p in (left_path, right_path, gt_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"[{query_tag}] missing GT artefact(s): {missing}. "
            f"Run dataset/{query_tag.split('_')[0]}_groundtruth.py first."
        )

    left_pdf = pd.read_csv(left_path)
    right_pdf = pd.read_csv(right_path)
    gt_pdf = pd.read_csv(gt_path)
    n_left = len(left_pdf)
    n_right = len(right_pdf)
    print(f"[{query_tag}] left={n_left:,}  right={n_right:,}  "
          f"positives={len(gt_pdf):,}  cross_product={n_left * n_right:,}")

    # Embed every row on both sides (cost |L|+|R|, not |L|*|R|).
    def _embed_all(pdf, col_name, kind):
        vals = pdf[col_name].astype(str).tolist()
        if kind == "image":
            print(f"  embedding {len(vals):,} images ({col_name}) ...")
            return embed_images_siglip(vals, EMBEDDING_MODEL)
        print(f"  embedding {len(vals):,} texts ({col_name}) ...")
        return embed_texts(vals, EMBEDDING_MODEL)

    left_emb = _embed_all(left_pdf, left_col, left_kind).astype(np.float32)
    right_emb = _embed_all(right_pdf, right_col, right_kind).astype(np.float32)

    # L2-normalise so raw matmul gives cosine similarity.
    left_emb /= (np.linalg.norm(left_emb, axis=1, keepdims=True) + 1e-8)
    right_emb /= (np.linalg.norm(right_emb, axis=1, keepdims=True) + 1e-8)

    # Positive pair set as a sparse |L|x|R| boolean matrix for fast block
    # masking. Drop any GT rows whose ids aren't in the side tables.
    left_ids = left_pdf["left_id"].astype(str).tolist()
    right_ids = right_pdf["right_id"].astype(str).tolist()
    left_idx_of = {lid: i for i, lid in enumerate(left_ids)}
    right_idx_of = {rid: i for i, rid in enumerate(right_ids)}
    gt_li = []
    gt_ri = []
    for l, r in zip(gt_pdf["left_id"].astype(str),
                    gt_pdf["right_id"].astype(str)):
        if l in left_idx_of and r in right_idx_of:
            gt_li.append(left_idx_of[l])
            gt_ri.append(right_idx_of[r])
    gt_li = np.asarray(gt_li, dtype=np.int64)
    gt_ri = np.asarray(gt_ri, dtype=np.int64)
    true_n_pos = int(len(gt_li))
    pos_matrix = csr_matrix(
        (np.ones(true_n_pos, dtype=np.int8), (gt_li, gt_ri)),
        shape=(n_left, n_right),
    )

    # Negative Bernoulli rate so the expected sample ≈ VIZ_JOINML_NEG_SAMPLE.
    true_total = n_left * n_right
    true_n_neg = true_total - true_n_pos
    if VIZ_JOINML_NEG_SAMPLE <= 0 or VIZ_JOINML_NEG_SAMPLE >= true_n_neg:
        neg_keep_prob = 1.0
    else:
        neg_keep_prob = VIZ_JOINML_NEG_SAMPLE / max(true_n_neg, 1)

    # --- Blocked matmul over the right side ---
    block = max(1, VIZ_JOINML_BLOCK)
    pos_sims_buf = []   # every positive sim we see
    neg_sims_buf = []   # Bernoulli-sampled negatives
    rng = np.random.default_rng(42)
    n_blocks = (n_right + block - 1) // block
    print(f"  blocked matmul: {n_blocks} blocks of <= {block} right-rows, "
          f"neg keep prob = {neg_keep_prob:.3e}")

    for bi, r_start in enumerate(range(0, n_right, block)):
        r_end = min(r_start + block, n_right)
        # (|L|, block) float32 — e.g. 32k*2048*4 = 256 MB for Flickr
        sims_block = left_emb @ right_emb[r_start:r_end].T
        pos_mask = pos_matrix[:, r_start:r_end].toarray().astype(bool)
        pos_vals = sims_block[pos_mask]
        pos_sims_buf.append(pos_vals)

        neg_vals = sims_block[~pos_mask]
        if neg_keep_prob >= 1.0:
            neg_sims_buf.append(neg_vals)
        else:
            mask = rng.random(neg_vals.shape, dtype=np.float32) < neg_keep_prob
            neg_sims_buf.append(neg_vals[mask])

        if (bi + 1) % max(1, n_blocks // 20) == 0 or bi + 1 == n_blocks:
            total_neg_so_far = sum(x.size for x in neg_sims_buf)
            total_pos_so_far = sum(x.size for x in pos_sims_buf)
            print(
                f"  block {bi+1}/{n_blocks}  pos_sims={total_pos_so_far:,}  "
                f"neg_sims={total_neg_so_far:,}"
            )

    pos_sims = np.concatenate(pos_sims_buf) if pos_sims_buf else np.empty(0, dtype=np.float32)
    neg_sims = np.concatenate(neg_sims_buf) if neg_sims_buf else np.empty(0, dtype=np.float32)

    sims = np.concatenate([pos_sims, neg_sims]).astype(np.float32)
    labels = np.zeros(len(sims), dtype=np.int32)
    labels[: len(pos_sims)] = 1

    # Pass TRUE counts so the summary CSV / plot labels reflect the full
    # cross product (|L|*|R|), not the Bernoulli-sampled negative count.
    plot_distribution(query_tag, sims, labels, summary_rows,
                      true_total_pairs=true_total,
                      true_n_pos=true_n_pos)


def viz_veri_q1(summary_rows):
    _viz_joinml_pair(
        query_tag="veri_q1",
        data_dir=_HOME + "/SemBench/files/veri",
        left_name="veri_left.csv", right_name="veri_right.csv",
        gt_name="gt_pairwise_veri.csv",
        left_kind="image", right_kind="image",
        left_col="left_image", right_col="right_image",
        summary_rows=summary_rows,
    )


def viz_flickr_q1(summary_rows):
    _viz_joinml_pair(
        query_tag="flickr_q1",
        data_dir=_HOME + "/SemBench/files/flickr30k",
        left_name="flickr_left.csv", right_name="flickr_right.csv",
        gt_name="gt_pairwise_flickr.csv",
        left_kind="image", right_kind="text",
        left_col="left_image", right_col="right_caption",
        summary_rows=summary_rows,
    )


def viz_roxford_q1(summary_rows):
    _viz_joinml_pair(
        query_tag="roxford_q1",
        data_dir=_HOME + "/SemBench/files/roxford5k",
        left_name="roxford_left.csv", right_name="roxford_right.csv",
        gt_name="gt_pairwise_roxford.csv",
        left_kind="image", right_kind="image",
        left_col="left_image", right_col="right_image",
        summary_rows=summary_rows,
    )


def viz_company_q1(summary_rows):
    _viz_joinml_pair(
        query_tag="company_q1",
        data_dir=_HOME + "/SemBench/files/company",
        left_name="company_left.csv", right_name="company_right.csv",
        gt_name="gt_pairwise_company.csv",
        left_kind="text", right_kind="text",
        left_col="left_description", right_col="right_description",
        summary_rows=summary_rows,
    )


# ---------------------------------------------------------------------------
# LRobench match queries (text-text semantic joins). Ground truth is a short
# fixed (left_id, right_id) list from the original LRobench pipelines. We
# embed the filtered left/right join keys separately and use pair cosine
# similarity as the 1D DIRECT axis.
# ---------------------------------------------------------------------------
LROBENCH_DATABASES_DIR = os.environ.get(
    "LROBENCH_DATABASES_DIR", "./databases"
)


def _viz_match_pair(query_tag, left_ids, left_texts, right_ids, right_texts,
                    gt_pairs, summary_rows):
    n_left = len(left_texts)
    n_right = len(right_texts)
    print(f"[{query_tag}] embedding {n_left} left + {n_right} right rows ...")
    left_emb = embed_texts(left_texts, EMBEDDING_MODEL)
    right_emb = embed_texts(right_texts, EMBEDDING_MODEL)

    li = np.repeat(np.arange(n_left), n_right)
    ri = np.tile(np.arange(n_right), n_left)
    sims = cosine_pair(left_emb[li], right_emb[ri])

    gt_set = {(str(a).strip(), str(b).strip()) for a, b in gt_pairs}
    lk = [str(x).strip() for x in left_ids]
    rk = [str(x).strip() for x in right_ids]
    labels = np.zeros(len(sims), dtype=np.int32)
    for k_, (i, j) in enumerate(zip(li, ri)):
        if (lk[i], rk[j]) in gt_set:
            labels[k_] = 1
    plot_distribution(query_tag, sims, labels, summary_rows)


def viz_match102(summary_rows):
    sp = get_spark()
    customers = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/car_retails/customers.csv")
    )
    customers = customers.filter(col("creditLimit") > 100000)
    cust_pdf = customers.select("customerNumber", "city").toPandas()
    left_ids = cust_pdf["customerNumber"].astype(int).tolist()
    left_texts = cust_pdf["city"].astype(str).tolist()

    offices = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/car_retails/offices.csv")
    )
    terr_pdf = offices.select("territory").distinct().toPandas()
    right_ids = terr_pdf["territory"].astype(str).tolist()
    right_texts = right_ids

    truth = [
        (114, 'APAC'), (119, 'EMEA'), (141, 'EMEA'), (146, 'EMEA'),
        (148, 'APAC'), (157, 'NA'),   (187, 'EMEA'), (227, 'EMEA'),
        (249, 'EMEA'), (259, 'EMEA'), (276, 'APAC'), (278, 'EMEA'),
        (286, 'NA'),   (298, 'EMEA'), (319, 'NA'),   (363, 'NA'),
        (386, 'EMEA'), (448, 'EMEA'), (458, 'EMEA'), (496, 'APAC'),
    ]
    _viz_match_pair("match102", left_ids, left_texts, right_ids, right_texts,
                    truth, summary_rows)


def viz_match104(summary_rows):
    sp = get_spark()
    comments = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/codebase_community/comments.csv")
    )
    # Pandas `comments[:10]` takes file order; orderBy(Id).limit(10) is the
    # stable Spark equivalent for the monotonically increasing Id column.
    comments = comments.orderBy(col("Id")).limit(10)
    cmt_pdf = comments.select("Id", "Text").toPandas()
    left_ids = cmt_pdf["Id"].astype(int).tolist()
    left_texts = cmt_pdf["Text"].astype(str).tolist()

    tags = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/codebase_community/tags.csv")
    )
    tags = tags.orderBy(col("Count").desc()).limit(10)
    tag_pdf = tags.select("Id", "TagName").toPandas()
    right_ids = tag_pdf["Id"].astype(int).tolist()
    right_texts = tag_pdf["TagName"].astype(str).tolist()

    truth = [(2, 41), (4, 9), (9, 41), (11, 41)]
    _viz_match_pair("match104", left_ids, left_texts, right_ids, right_texts,
                    truth, summary_rows)


def _viz_shipping(query_tag, revenue_filter, truth, summary_rows):
    sp = get_spark()
    customer = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/shipping/customer.csv")
    )
    customer = customer.filter(revenue_filter)
    cust_pdf = customer.select("cust_id", "state").toPandas()
    left_ids = cust_pdf["cust_id"].astype(int).tolist()
    left_texts = cust_pdf["state"].astype(str).tolist()

    city = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/shipping/city.csv")
    )
    city = city.filter(col("population") > 600000)
    state_pdf = city.select("state").distinct().toPandas()
    right_ids = state_pdf["state"].astype(str).tolist()
    right_texts = right_ids

    _viz_match_pair(query_tag, left_ids, left_texts, right_ids, right_texts,
                    truth, summary_rows)


def viz_match105(summary_rows):
    truth = [
        (381, 'Florida'),     (615, 'Florida'),     (618, 'Florida'),
        (1575, 'California'), (1685, 'California'), (1769, 'Texas'),
        (2001, 'California'), (2799, 'Texas'),      (3288, 'Texas'),
        (3660, 'Maryland'),   (4353, 'Texas'),      (4869, 'Arizona'),
    ]
    _viz_shipping("match105", col("cust_type") == "wholesaler", truth,
                  summary_rows)


def viz_match106(summary_rows):
    truth = [
        (721, 'Florida'),     (1724, 'Ohio'),       (2001, 'California'),
        (2434, 'Ohio'),       (2799, 'Texas'),      (3320, 'California'),
        (3447, 'California'),
    ]
    _viz_shipping("match106", col("annual_revenue") > 45000000, truth,
                  summary_rows)


def viz_match107(summary_rows):
    sp = get_spark()
    team = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/european_football_2/Team.csv")
    )
    team = team.filter(col("team_long_name").startswith("P"))
    team_pdf = team.select("id", "team_long_name").toPandas()
    left_ids = team_pdf["id"].astype(int).tolist()
    left_texts = team_pdf["team_long_name"].astype(str).tolist()

    country = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/european_football_2/Country.csv")
    )
    country_pdf = country.select("id", "name").toPandas()
    right_ids = country_pdf["id"].astype(int).tolist()
    right_texts = country_pdf["name"].astype(str).tolist()

    truth = [
        (3476, 1729),   (9548, 4769),   (20532, 10257), (21292, 10257),
        (23523, 10257), (26556, 13274), (29000, 13274), (31444, 15722),
        (31445, 15722), (31448, 15722), (31457, 15722), (32891, 15722),
        (33377, 15722), (36248, 17642), (41673, 19694),
    ]
    _viz_match_pair("match107", left_ids, left_texts, right_ids, right_texts,
                    truth, summary_rows)


def viz_match113(summary_rows):
    sp = get_spark()
    home = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/"
             "home_office_senior_officials_travel_data_return.csv")
    )
    left_pdf = (
        home.select(col("Name of Official").alias("name"))
            .dropna().dropDuplicates(["name"]).toPandas()
    )
    travel = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/"
             "travel-exp-April-June-2018.csv")
    )
    right_pdf = (
        travel.select(col("Senior Officials Name").alias("name"))
              .dropna().dropDuplicates(["name"]).toPandas()
    )
    left_names = left_pdf["name"].astype(str).tolist()
    right_names = right_pdf["name"].astype(str).tolist()
    truth = [
        ('Mark Sedwill ',  'Mark Bryson-Richardson'),
        ('Richard Clarke', 'Richard Montgomery'),
    ]
    _viz_match_pair("match113", left_names, left_names, right_names, right_names,
                    truth, summary_rows)


def viz_match121(summary_rows):
    sp = get_spark()
    april = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/01.Apr_2018.csv")
    )
    left_pdf = (
        april.select(col("Supplier").alias("supplier"))
             .dropna().dropDuplicates(["supplier"]).toPandas()
    )
    may = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/2015_05_expenditure.csv")
    )
    right_pdf = (
        may.select(col("Supplier").alias("supplier"))
           .dropna().dropDuplicates(["supplier"]).toPandas()
    )
    left_suppliers = left_pdf["supplier"].astype(str).tolist()
    right_suppliers = right_pdf["supplier"].astype(str).tolist()
    truth = [
        ('Nhs Supply Chain',                 'NHS SUPPLY CHAIN'),
        ('Novartis Pharmaceuticals Uk Ltd',  'NOVARTIS PHARMACEUTICALS UK LTD'),
        ('Roche Diagnostics Limited',        'ROCHE PRODUCTS LTD'),
        ('Csc',                              'CSC COMPUTER SCIENCES LTD'),
        ('Philips Healthcare',               'PHILIPS HEALTHCARE'),
        ('Nhs Blood And Transplant',         'NHS BLOOD & TRANSPLANT'),
        ('NHS Litigation Authority',         'NHS LITIGATION AUTHORITY'),
    ]
    _viz_match_pair("match121", left_suppliers, left_suppliers,
                    right_suppliers, right_suppliers, truth, summary_rows)


def viz_match202(summary_rows):
    """Yelp <-> Zomato entity resolution restricted to zip 60642.

    The text used for the embedding axis concatenates name/address/phone/
    cuisine on each side so the cosine similarity reflects all match
    fields, mirroring the LLM prompt in run_match202.
    """
    sp = get_spark()
    yelp = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/restaurants2/yelp.csv")
    ).filter(col("zip") == 60642)
    left_pdf = yelp.select(
        col("ID").cast("int").alias("id"),
        col("name"), col("address"), col("phone"), col("cuisine"),
    ).toPandas()
    zomato = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/restaurants2/zomato.csv")
    ).filter(col("zip") == 60642)
    right_pdf = zomato.select(
        col("ID").cast("int").alias("id"),
        col("name"), col("address"), col("phone"), col("cuisine"),
    ).toPandas()

    def _row_text(r):
        return (
            f"name={r['name']} | address={r['address']} | "
            f"phone={r['phone']} | cuisine={r['cuisine']}"
        )
    left_ids = left_pdf["id"].astype(int).tolist()
    left_texts = [_row_text(r) for _, r in left_pdf.iterrows()]
    right_ids = right_pdf["id"].astype(int).tolist()
    right_texts = [_row_text(r) for _, r in right_pdf.iterrows()]

    truth = [
        (2, 844), (72, 1020), (90, 744), (111, 564),
        (198, 1022), (275, 272), (333, 134), (418, 590),
    ]
    _viz_match_pair("match202", left_ids, left_texts, right_ids, right_texts,
                    truth, summary_rows)


def viz_match305(summary_rows):
    """Schema matching: each row describes one column (name + sample values).

    Mirrors run_match305 / _column_descriptor_df: every column from
    frpm.csv (left) and schools.csv (right) becomes a row whose embedding
    text is "name=<colname> | samples=<5 sample values>".
    """
    base = f"{LROBENCH_DATABASES_DIR}/california_schools"

    def _column_rows(csv_path, sample_n=5):
        pdf = pd.read_csv(csv_path, nrows=500)
        ids = []
        texts = []
        for c in pdf.columns:
            vals = pdf[c].dropna().astype(str).head(sample_n).tolist()
            samples = " | ".join(vals) if vals else ""
            ids.append(c)
            texts.append(f"name={c} | samples={samples}")
        return ids, texts

    left_ids, left_texts = _column_rows(f"{base}/frpm.csv")
    right_ids, right_texts = _column_rows(f"{base}/schools.csv")

    truth = [
        ('CDSCode',                  'CDSCode'),
        ('County Code',              'County'),
        ('District Code',            'NCESDist'),
        ('District Name',            'District'),
        ('School Name',              'School'),
        ('District Type ',           'DOCType'),
        ('School Type',              'SOCType'),
        ('Educational Option Type',  'EdOpsName'),
        ('Charter School (Y/N)',     'Charter'),
        ('Charter School Number',    'CharterNum'),
        ('Charter Funding Type',     'FundingType'),
    ]
    _viz_match_pair("match305", left_ids, left_texts, right_ids, right_texts,
                    truth, summary_rows)


# ---------------------------------------------------------------------------
# Combined grid PDF (2 rows x 7 cols) over a curated set of 14 queries.
#
# `plot_combined_grid_pdf` runs every viz_* runner in `queries`, captures
# the (sims, labels) it would have plotted by monkey-patching
# `plot_distribution`, then renders all 14 panels onto a single PDF
# figure. Each panel shows positive (blue) and negative (red) similarity
# histograms with N + positive-rate annotation. The cosine-similarity
# x-axis label is intentionally suppressed per request.
# ---------------------------------------------------------------------------


def _capture_sims_labels(query_runner):
    """Run a viz_* runner with `plot_distribution` patched to capture data.

    Returns a dict {query_tag, sims, labels}. Works for every viz_* in
    this module because they all funnel through `plot_distribution` for
    rendering; we swap that out for a no-op capture, run the loader, then
    restore.
    """
    captured: dict = {}
    g = globals()
    original = g["plot_distribution"]

    def _capture(query_tag, similarities, labels, summary_rows,
                 true_total_pairs=None, true_n_pos=None,
                 boundaries=None, sampled_indices=None):
        captured["query_tag"] = query_tag
        captured["sims"] = np.asarray(similarities, dtype=np.float32)
        captured["labels"] = np.asarray(labels, dtype=np.int32)
        captured["true_total_pairs"] = true_total_pairs
        captured["true_n_pos"] = true_n_pos

    g["plot_distribution"] = _capture
    try:
        _take_stashed_embeddings()  # clear any stale buffer
        query_runner([])
    finally:
        g["plot_distribution"] = original
    captured["embeddings"] = _take_stashed_embeddings()
    return captured


# 14-query layout: 2 rows x 7 columns. Note: the user's spec mentioned
# `mmqa_q6` but only `mmqa_q2a` and `mmqa_q7` are implemented in this
# repo (see SemBench groundtruths under files/mmqa/), so we use
# `mmqa_q7` as the second MMQA entry.
GRID14_QUERIES = [
    ("mmqa_q2a",   "MMQA Q2a"),
    ("mmqa_q7",    "MMQA Q7"),
    ("ecomm_q7",   "Ecomm Q7"),
    ("ecomm_q8",   "Ecomm Q8"),
    ("ecomm_q9",   "Ecomm Q9"),
    ("ecomm_q13",  "Ecomm Q13"),
    ("cars_q3",    "Cars Q3"),
    ("cars_q4",    "Cars Q4"),
    ("cars_q8",    "Cars Q8"),
    ("animals_q1", "Animals Q1"),
    ("match113",   "LRO Match 113"),
    ("match121",   "LRO Match 121"),
    ("match202",   "LRO Match 202"),
    ("match305",   "LRO Match 305"),
]


def plot_combined_grid_pdf(queries=None,
                            output_path=None,
                            rows: int = 2,
                            cols: int = 7) -> str:
    """Render `queries` (list of (tag, title)) as a rows x cols grid PDF.

    Defaults to GRID14_QUERIES (14 panels, 2 rows of 7). Each panel:
      * blue density histogram of GT positives,
      * red density histogram of negatives,
      * top-right inset with N and positive-rate %,
      * panel title (e.g. "MMQA Q2a"),
      * NO x-axis label (per request — the cosine-similarity equation
        is omitted to keep the grid uncluttered).

    Returns the absolute path of the saved PDF.
    """
    queries = list(queries) if queries is not None else list(GRID14_QUERIES)
    if output_path is None:
        output_path = os.path.join(results_dir, "similarity_dist_grid.pdf")

    if len(queries) > rows * cols:
        raise ValueError(
            f"layout {rows}x{cols} = {rows*cols} cannot fit {len(queries)} queries"
        )

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 3.2, rows * 2.8),
        squeeze=False,
    )

    pos_color = "#34495e"   # deep navy blue (matches reference figure)
    neg_color = "#c98176"   # warm coral red

    for idx, (tag, title) in enumerate(queries):
        ax = axes[idx // cols][idx % cols]
        runner = QUERY_VIZ_RUNNERS.get(tag)
        if runner is None:
            ax.text(0.5, 0.5, f"{tag}\n(no runner)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        try:
            print(f"[grid] capturing {tag} ...")
            captured = _capture_sims_labels(runner)
        except Exception as e:
            print(f"[grid] ERROR for {tag}: {e}")
            import traceback; traceback.print_exc()
            ax.text(0.5, 0.5, f"{tag}\n(error)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="red")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        sims = captured["sims"]
        labels = captured["labels"]
        true_total = captured.get("true_total_pairs")
        true_n_pos = captured.get("true_n_pos")

        pos_sims = sims[labels == 1]
        neg_sims = sims[labels == 0]
        if true_total and true_n_pos is not None:
            n_total = int(true_total)
            n_pos = int(true_n_pos)
        else:
            n_total = int(len(sims))
            n_pos = int(len(pos_sims))
        pos_rate = n_pos / max(n_total, 1)

        if len(sims) == 0:
            ax.text(0.5, 0.5, f"{tag}\n(empty)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        lo = float(sims.min()); hi = float(sims.max())
        if hi <= lo:
            hi = lo + 1e-3
        pad = (hi - lo) * 0.02
        edges = np.linspace(lo - pad, hi + pad, 41)

        if len(neg_sims) > 0:
            ax.hist(neg_sims, bins=edges, color=neg_color, alpha=0.75,
                    density=True, label="negatives", edgecolor="none")
        if len(pos_sims) > 0:
            ax.hist(pos_sims, bins=edges, color=pos_color, alpha=0.9,
                    density=True, label="positives", edgecolor="none")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        if idx % cols == 0:
            ax.set_ylabel("density", fontsize=9)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

        ax.text(
            0.97, 0.97,
            f"N={n_total:,}\npos rate {pos_rate:.2%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgray", alpha=0.9),
        )

    # Hide any leftover axes when len(queries) < rows*cols.
    for empty in range(len(queries), rows * cols):
        axes[empty // cols][empty % cols].axis("off")

    # Single shared legend at the bottom — clearer than per-panel legends.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=pos_color, edgecolor="none", label="positives"),
        Patch(facecolor=neg_color, edgecolor="none", label="negatives"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=2,
        frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    abs_path = os.path.abspath(output_path)
    print(f"[grid] saved combined PDF -> {abs_path}")
    return abs_path


# ---------------------------------------------------------------------------
# Axis-refinement grid: side-by-side similarity vs LR-blended axis.
#
# For every query we fit a LogisticRegression on a small uniform sample of
# the per-row embeddings (captured via _stash_per_row_embeddings in the
# cosine helpers), recover its decision_function over all rows, and form
# the refined axis
#     refined = (1 - lam) * z(sim) + lam * z(lr_score)
# exactly as cluster_proxy._direct_compute_phase_axis does.  Each query
# then renders two stacked histograms (top = SigLIP similarity, bottom =
# refined axis) so the visual claim "positives concentrate tighter near a
# single threshold after the LR blend" is on the page.
# ---------------------------------------------------------------------------


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd


def _normalize_axis(x: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """Display normalization for a phase axis.

    cluster_proxy sorts on the RAW axis (`_direct_compute_phase_axis` z-scores
    the blend *components* z_sim/z_lr but not the final blend, and returns raw
    cosine similarity when lambda=0), so the raw per-phase axis is not on a
    common scale. We therefore normalize each phase's axis purely for display;
    being monotonic, this does not change the ordering, the DIRECT threshold,
    or the ROC-AUC.

    - 'zscore'  : standardize to mean 0 / std 1 (matches `_direct_zscore`).
    - 'minmax'  : rescale to [0, 1] (归一化).
    """
    x = np.asarray(x, dtype=np.float64)
    if mode == "pm1":                       # min-max to [-1, 1]
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo < 1e-12:
            return np.zeros_like(x, dtype=np.float64)
        return 2.0 * (x - lo) / (hi - lo) - 1.0
    if mode == "minmax":
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo < 1e-12:
            return np.zeros_like(x, dtype=np.float64)
        return (x - lo) / (hi - lo)
    return _zscore(x)


def _fit_lr_score(embeddings: np.ndarray,
                   labels: np.ndarray,
                   sample_rate: float = 0.10,
                   min_sample: int = 30,
                   max_sample: int = 2000,
                   seed: int = 0,
                   lr_C: float = 1.0) -> np.ndarray | None:
    """Fit LogisticRegression on a stratified uniform subsample and return
    decision_function(embeddings) for all rows. Returns None if the labels
    are degenerate (one class only) or embeddings are missing.
    """
    from sklearn.linear_model import LogisticRegression
    n = len(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None
    rng = np.random.default_rng(int(seed))
    target = int(round(n * sample_rate))
    target = max(min_sample, min(target, max_sample, n))
    # Stratified by GT to guarantee both classes appear in the LR fit. We
    # use rate-matched proportions but floor each side at one and cap at
    # the available count.
    pos_rate = len(pos_idx) / max(n, 1)
    n_pos = max(1, min(int(round(target * pos_rate)), len(pos_idx)))
    n_neg = max(1, min(target - n_pos, len(neg_idx)))
    sampled = np.concatenate([
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=n_neg, replace=False),
    ])
    try:
        lr = LogisticRegression(C=float(lr_C), max_iter=300)
        lr.fit(embeddings[sampled], labels[sampled])
    except Exception as exc:
        print(f"  [axis-grid] LR fit failed: {exc}")
        return None
    return lr.decision_function(embeddings).astype(np.float32)


def _run_direct_trajectory(embeddings: np.ndarray,
                            similarities: np.ndarray,
                            labels: np.ndarray,
                            budget: int,
                            seed: int = 0,
                            n_phases: int = 4,
                            vs_ratio: float = 0.5,
                            cold_start_phases: int = 1,
                            lr_C: float = 1.0,
                            ) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Run the real `sample_direct` on real embeddings with a GT-backed
    oracle and return `(sampled_indices, sampled_labels, final_lambda)`.

    The function reproduces what LASO does at inference time but with the
    LLM oracle replaced by a numpy lookup against ground truth so we can
    capture the exact sample DIRECT would have drawn. Returns None if the
    sampler fails for this query.
    """
    from cluster_proxy import sample_direct as _sample_direct  # heavy import; defer

    similarities = np.asarray(similarities, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    n = int(len(similarities))
    texts = [f"row_{i}" for i in range(n)]

    def _label_fn(xs):
        out = np.empty(len(xs), dtype=np.int32)
        for k_, t in enumerate(xs):
            idx = int(str(t).split("_", 1)[1])
            out[k_] = int(labels[idx])
        return out

    np.random.seed(int(seed))
    try:
        sampled_idx, sampled_vals, _W, prof = _sample_direct(
            embeddings, similarities, texts, _label_fn,
            sample_size=int(budget),
            n_phases=int(n_phases),
            lr_blend_schedule="linear",
            cold_start_phases=int(cold_start_phases),
            vs_ratio=float(vs_ratio),
            lr_C=float(lr_C),
        )
    except Exception as exc:
        print(f"  [axis-grid] sample_direct failed: {exc}")
        import traceback; traceback.print_exc()
        return None
    lams = prof.get("phase_lambda", []) or [0.0]
    final_lam = float(lams[-1])
    return (np.asarray(sampled_idx, dtype=np.int64),
            np.asarray(sampled_vals, dtype=np.int32),
            final_lam)


def _compute_refined_axis(sims: np.ndarray,
                            embeddings: np.ndarray | None,
                            labels: np.ndarray,
                            sample_rate: float = 0.10,
                            lam: float = 1.0,
                            seed: int = 0,
                            use_direct: bool = True,
                            direct_budget_cap: int = 2000,
                            direct_n_phases: int = 4,
                            lr_C: float = 1.0,
                            ) -> dict | None:
    """Return `{refined, lr_score, sampled_idx, sampled_vals, final_lam,
    source}` where `source` is one of {'direct', 'stratified'}. Falls back
    to stratified-uniform if `use_direct` is True but the DIRECT trajectory
    fails or returns a single-class sample.
    """
    from sklearn.linear_model import LogisticRegression

    if embeddings is None or len(embeddings) != len(sims):
        return None
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None

    n = len(sims)
    sampled_idx: np.ndarray | None = None
    sampled_vals: np.ndarray | None = None
    source = "stratified"
    final_lam_recorded = float("nan")

    if use_direct:
        budget = int(round(n * sample_rate))
        budget = max(30, min(budget, int(direct_budget_cap), n))
        traj = _run_direct_trajectory(
            embeddings, sims, labels, budget,
            seed=seed, n_phases=direct_n_phases, lr_C=lr_C,
        )
        if traj is not None:
            s_idx, s_vals, final_lam_recorded = traj
            if len(np.unique(s_vals)) >= 2:
                sampled_idx, sampled_vals = s_idx, s_vals
                source = "direct"
            else:
                print(f"  [axis-grid] DIRECT sample is single-class; "
                      f"falling back to stratified.")

    if sampled_idx is None:
        # Stratified-uniform fallback so the panel still renders.
        rng = np.random.default_rng(int(seed))
        target = int(round(n * sample_rate))
        target = max(30, min(target, 2000, n))
        pos_rate = len(pos_idx) / max(n, 1)
        n_pos = max(1, min(int(round(target * pos_rate)), len(pos_idx)))
        n_neg = max(1, min(target - n_pos, len(neg_idx)))
        sampled_idx = np.concatenate([
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ])
        sampled_vals = labels[sampled_idx].astype(np.int32)
        source = "stratified"

    try:
        lr = LogisticRegression(C=float(lr_C), max_iter=300)
        lr.fit(embeddings[sampled_idx], sampled_vals)
    except Exception as exc:
        print(f"  [axis-grid] LR fit failed: {exc}")
        return None
    lr_score = lr.decision_function(embeddings).astype(np.float32)
    refined = ((1.0 - lam) * _zscore(sims) + lam * _zscore(lr_score)).astype(np.float32)
    return {
        "refined": refined,
        "lr_score": lr_score,
        "sampled_idx": np.asarray(sampled_idx, dtype=np.int64),
        "sampled_vals": np.asarray(sampled_vals, dtype=np.int32),
        "final_lam_recorded": float(final_lam_recorded),
        "source": source,
    }


def _save_paper_fig(fig, out_path) -> None:
    """Save `fig` as PNG (dpi 150, quick viewing) AND a sibling vector PDF (same
    basename, .pdf) for inserting into the paper via LaTeX \\includegraphics.
    Dense point-cloud artists are rasterized in-figure, so the PDF stays small
    while axes, text, threshold lines, and the (few) positives remain vector."""
    if not out_path:
        return
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")   # vector PDF (format from ext)


# Shared point palette + legend so the scatter and baseline figures use
# IDENTICAL colors and legend text.
_VIZ_COL_ALL = "#f0c9bf"       # population / all rows (faint pink)
_VIZ_COL_SAMPLED = "#8a3a1e"   # sampled / labeled rows (dark brown)
_VIZ_COL_POS = "#2a9d8f"       # ground-truth positive (teal)
_VIZ_COL_RING = "#e0a423"      # captured-positive ring (gold)


def _viz_point_legend_handles():
    """The shared 4-entry point legend (no threshold entry)."""
    from matplotlib.lines import Line2D
    return [
        # Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=_VIZ_COL_ALL,
        #        markeredgecolor="none", markersize=_fs(8), label="unsampled"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=_VIZ_COL_SAMPLED,
               markeredgecolor="none", markersize=_fs(8), label="sampled negative"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=_VIZ_COL_POS,
               markeredgecolor="white", markersize=_fs(10), label="positive"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=_VIZ_COL_POS,
               markeredgecolor=_VIZ_COL_RING, markeredgewidth=2, markersize=_fs(11),
               label="sampled positive"),
    ]


def _draw_dist_panel(ax,
                      values: np.ndarray,
                      labels: np.ndarray,
                      pos_color: str,
                      neg_color: str,
                      title: str | None = None,
                      ylabel: str | None = None,
                      n_total: int | None = None,
                      pos_rate: float | None = None,
                      xlabel: str | None = None,
                      annotate_box: bool = True) -> None:
    pos = values[labels == 1]
    neg = values[labels == 0]
    if len(values) == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        return
    lo = float(np.min(values)); hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1e-3
    pad = (hi - lo) * 0.02
    edges = np.linspace(lo - pad, hi + pad, 41)
    if len(neg) > 0:
        ax.hist(neg, bins=edges, color=neg_color, alpha=0.75, density=True,
                label="negatives", edgecolor="none")
    if len(pos) > 0:
        ax.hist(pos, bins=edges, color=pos_color, alpha=0.9, density=True,
                label="positives", edgecolor="none")
    if title is not None:
        ax.set_title(title, fontsize=10, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=8)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    if annotate_box and n_total is not None and pos_rate is not None:
        ax.text(
            0.97, 0.97,
            f"N={n_total:,}\npos {pos_rate:.2%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="lightgray", alpha=0.9),
        )


def plot_axis_refinement_grid(queries=None,
                                output_path=None,
                                sample_rate: float = 0.10,
                                lam: float = 1.0,
                                seed: int = 0,
                                use_direct: bool = True,
                                direct_budget_cap: int = 2000,
                                direct_n_phases: int = 4) -> str:
    """Render `queries` as a 2x7 (outer) x 2 (inner: sim on top, refined on
    bottom) figure. Top row of each cell is the SigLIP similarity axis;
    bottom row is the LR-blended refined axis with `lam` weight.

    Output: `results/similarity_dist/similarity_dist_axis_refinement.pdf`.
    """
    import matplotlib.gridspec as gridspec
    queries = list(queries) if queries is not None else list(GRID14_QUERIES)
    if output_path is None:
        output_path = os.path.join(
            results_dir, "similarity_dist_axis_refinement.pdf",
        )

    n_q = len(queries)
    outer_rows, outer_cols = 2, 7
    if n_q > outer_rows * outer_cols:
        raise ValueError(
            f"axis-refinement grid expects <= {outer_rows*outer_cols} "
            f"queries; got {n_q}"
        )

    pos_color = "#34495e"
    neg_color = "#c98176"

    fig = plt.figure(figsize=(outer_cols * 2.8, outer_rows * 4.4))
    outer = gridspec.GridSpec(outer_rows, outer_cols, figure=fig,
                              hspace=0.55, wspace=0.30)

    for idx, (tag, title) in enumerate(queries):
        r, c = idx // outer_cols, idx % outer_cols
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[r, c], hspace=0.40,
        )
        ax_sim = fig.add_subplot(inner[0, 0])
        ax_ref = fig.add_subplot(inner[1, 0])

        runner = QUERY_VIZ_RUNNERS.get(tag)
        if runner is None:
            for ax in (ax_sim, ax_ref):
                ax.text(0.5, 0.5, f"{tag}\n(no runner)", ha="center",
                        va="center", transform=ax.transAxes, fontsize=9,
                        color="gray")
                ax.set_xticks([]); ax.set_yticks([])
            ax_sim.set_title(title, fontsize=10, fontweight="bold")
            continue

        try:
            print(f"[axis-grid] capturing {tag} ...")
            captured = _capture_sims_labels(runner)
        except Exception as e:
            print(f"[axis-grid] ERROR for {tag}: {e}")
            import traceback; traceback.print_exc()
            for ax in (ax_sim, ax_ref):
                ax.text(0.5, 0.5, f"{tag}\n(error)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="red")
                ax.set_xticks([]); ax.set_yticks([])
            ax_sim.set_title(title, fontsize=10, fontweight="bold")
            continue

        sims = np.asarray(captured["sims"], dtype=np.float32)
        labels = np.asarray(captured["labels"], dtype=np.int32)
        embeddings = captured.get("embeddings")
        true_total = captured.get("true_total_pairs")
        true_n_pos = captured.get("true_n_pos")
        n_total = int(true_total) if true_total else int(len(sims))
        n_pos = int(true_n_pos) if true_n_pos is not None else int(np.sum(labels == 1))
        pos_rate = n_pos / max(n_total, 1)

        _draw_dist_panel(
            ax_sim, sims, labels, pos_color, neg_color,
            title=title, ylabel="density" if c == 0 else None,
            n_total=n_total, pos_rate=pos_rate,
        )
        ax_sim.text(0.02, 0.97, r"$\sigma_{\mathrm{sim}}$",
                    transform=ax_sim.transAxes, ha="left", va="top",
                    fontsize=8, color="#555")

        result = _compute_refined_axis(
            sims, embeddings, labels,
            sample_rate=sample_rate, lam=lam, seed=seed,
            use_direct=use_direct, direct_budget_cap=direct_budget_cap,
            direct_n_phases=direct_n_phases,
        )
        if result is None:
            ax_ref.text(0.5, 0.5, "LR refit unavailable\n(no embeddings or "
                                   "single-class labels)",
                        ha="center", va="center", transform=ax_ref.transAxes,
                        fontsize=8, color="gray")
            ax_ref.set_xticks([]); ax_ref.set_yticks([])
        else:
            refined = result["refined"]
            _draw_dist_panel(
                ax_ref, refined, labels, pos_color, neg_color,
                title=None, ylabel="density" if c == 0 else None,
                n_total=None, pos_rate=None, annotate_box=False,
            )
            sampled_n = int(len(result["sampled_idx"]))
            src_tag = "DIRECT" if result["source"] == "direct" else "stratified"
            ax_ref.text(0.02, 0.97,
                        rf"$(1{{-}}\lambda)\,z(\sigma_{{\mathrm{{sim}}}}) + "
                        rf"\lambda\,z(\sigma_{{\mathrm{{lr}}}})$, $\lambda={lam:.1f}$",
                        transform=ax_ref.transAxes, ha="left", va="top",
                        fontsize=7, color="#555")
            ax_ref.text(0.97, 0.97,
                        f"{src_tag} sample\nB={sampled_n:,}",
                        transform=ax_ref.transAxes, ha="right", va="top",
                        fontsize=7, color="#555",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="lightgray", alpha=0.9))

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=pos_color, edgecolor="none", label="positives"),
        Patch(facecolor=neg_color, edgecolor="none", label="negatives"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=2,
        frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.005),
    )

    fig.subplots_adjust(top=0.96, bottom=0.05, left=0.04, right=0.99)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    abs_path = os.path.abspath(output_path)
    print(f"[axis-grid] saved -> {abs_path}")
    return abs_path


# ---------------------------------------------------------------------------
# DIRECT multi-phase axis-refinement *evolution* (per-query, rows = phases)
#
# Shows how DIRECT + iterative active learning makes the pos/neg distribution
# gradually separable. Phase 0 uses the raw SigLIP cosine-similarity axis (the
# "single threshold separates pos/neg" baseline); later phases re-sort on an
# LR-refined axis fit on the labels DIRECT has gathered so far. Reading a
# figure top->bottom, the two class histograms pull apart and the per-phase
# ROC-AUC climbs.
#
# Faithful + viz-file-only: we run the REAL `cluster_proxy.sample_direct`, with
# (a) a ground-truth-by-index oracle replacing the LLM, (b) a monkey-patch of
# `cluster_proxy._direct_compute_phase_axis` to capture the EXACT axis each
# blended phase sorts on, and (c) per-phase sample segmentation via
# `profiling["phase_n_labeled_after"]`. No edits to cluster_proxy.py.
# ---------------------------------------------------------------------------
DIRECT_EVOLUTION_QUERIES = [
    ("mmqa_q2a",   "MMQA Q2a"),
    ("mmqa_q7",    "MMQA Q7"),
    ("ecomm_q7",   "Ecomm Q7"),
    ("ecomm_q8",   "Ecomm Q8"),
    ("ecomm_q9",   "Ecomm Q9"),
    ("ecomm_q13",  "Ecomm Q13"),
    ("cars_q1",    "Cars Q1"),
    ("cars_q3",    "Cars Q3"),
    ("cars_q8",    "Cars Q8"),
    ("animals_q1", "Animals Q1"),
    ("match113",   "LRO Match 113"),
    ("match121",   "LRO Match 121"),
    ("match202",   "LRO Match 202"),
    ("match305",   "LRO Match 305"),
]


def _run_direct_phase_trajectory(embeddings: np.ndarray,
                                  similarities: np.ndarray,
                                  labels: np.ndarray,
                                  budget: int,
                                  seed: int = 0,
                                  n_phases: int = 5,
                                  schedule: str = "linear",
                                  cold_start_phases: int = 1,
                                  vs_ratio: float = 0.5,
                                  B1: int = 5,
                                  lr_C: float = 1.0) -> dict | None:
    """Run multi-phase `sample_direct` with a GT oracle and capture, per phase,
    the EXACT axis it sorts on plus the rows it sampled.

    Returns a dict with keys (each a list of length P = #phases):
        'phase_axes'         list[np.ndarray]  axis values per row (len N)
        'phase_sampled_idx'  list[np.ndarray]  global row indices sampled
        'phase_sampled_vals' list[np.ndarray]  their {0,1} oracle labels
        'phase_lambda'       list[float]       blend weight lambda
        'phase_source'       list[str]         'siglip' | 'blend(...)' | 'skipped'
        'phase_n_labeled'    list[int]         cumulative #labels after phase
        'similarities'       np.ndarray
    Returns None on failure or degenerate (single-class) input.
    """
    import cluster_proxy as _cp  # heavy import; defer to call time

    similarities = np.asarray(similarities, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    n = int(len(similarities))
    if n == 0 or len(np.unique(labels)) < 2:
        return None
    # Opaque per-row tokens; the oracle recovers the global index from the
    # token, so we never need real `texts` and labelling is an O(1) lookup.
    texts = [f"row_{i}" for i in range(n)]

    call_order_idx: list = []  # global indices in oracle-call (time) order

    def _label_fn(xs):
        out = np.empty(len(xs), dtype=np.int32)
        for k_, t in enumerate(xs):
            gi = int(str(t).split("_", 1)[1])
            call_order_idx.append(gi)
            out[k_] = int(labels[gi])
        return out

    recorded_axes: list = []
    recorded_peaks: list = []
    _orig_axis_fn = _cp._direct_compute_phase_axis
    _orig_phase_fn = getattr(_cp, "_run_direct_single_phase", None)

    def _patched_axis_fn(sims_arg, lr_score, lam):
        ax = _orig_axis_fn(sims_arg, lr_score, lam)
        recorded_axes.append(np.asarray(ax, dtype=np.float64).copy())
        return ax

    def _patched_phase_fn(*a, **k):
        # `_run_direct_single_phase` runs once per non-skipped phase; capture the
        # peaks (best-hypothesis positions in sorted-axis space) it returns.
        res = _orig_phase_fn(*a, **k)
        try:
            recorded_peaks.append([int(p) for (p, _s) in res.get("peaks", [])])
        except Exception:
            recorded_peaks.append([])
        return res

    np.random.seed(int(seed))
    _cp._direct_compute_phase_axis = _patched_axis_fn
    if _orig_phase_fn is not None:
        _cp._run_direct_single_phase = _patched_phase_fn
    try:
        _idx, _vals, _W, prof = _cp.sample_direct(
            embeddings, similarities, texts, _label_fn,
            sample_size=int(budget),
            n_phases=int(n_phases),
            lr_blend_schedule=str(schedule),
            cold_start_phases=int(cold_start_phases),
            vs_ratio=float(vs_ratio),
            B1=int(B1),
            lr_C=float(lr_C),
        )
    except Exception as exc:
        print(f"  [phase-evo] sample_direct failed: {exc}")
        import traceback; traceback.print_exc()
        return None
    finally:
        _cp._direct_compute_phase_axis = _orig_axis_fn
        if _orig_phase_fn is not None:
            _cp._run_direct_single_phase = _orig_phase_fn

    sources = [str(s) for s in prof.get("phase_axis_source", [])]
    lams = [float(x) for x in prof.get("phase_lambda", [])]
    n_after = [int(x) for x in prof.get("phase_n_labeled_after", [])]
    P = len(sources)
    if P == 0:
        return None

    # Per-phase axis: 'blend(...)' phases consumed the patched fn in phase
    # order; 'siglip'/'skipped' phases sort on the raw similarity axis.
    blended = iter(recorded_axes)
    sim64 = similarities.astype(np.float64)
    phase_axes: list = []
    for p in range(P):
        if sources[p].startswith("blend"):
            try:
                phase_axes.append(next(blended))
            except StopIteration:
                phase_axes.append(sim64)
        else:
            phase_axes.append(sim64)

    # Per-phase threshold(s): `_run_direct_single_phase` is invoked once per
    # NON-skipped phase (in phase order), so consume recorded peaks for every
    # phase whose source isn't 'skipped'. Each entry is a list of best-hypothesis
    # positions in that phase's descending-axis sorted space (>1 if multi-modal).
    peaks_it = iter(recorded_peaks)
    phase_threshold_pos: list = []
    for p in range(P):
        if sources[p] == "skipped":
            phase_threshold_pos.append([])
            continue
        try:
            phase_threshold_pos.append([int(x) for x in next(peaks_it)])
        except StopIteration:
            phase_threshold_pos.append([])

    # Per-phase sampled rows: segment the oracle-call-ordered indices by the
    # cumulative label counts after each phase. Dedup defensively (first
    # occurrence wins) so the count invariant matches `phase_n_labeled_after`.
    seen: set = set()
    uniq: list = []
    for gi in call_order_idx:
        if gi not in seen:
            seen.add(gi)
            uniq.append(gi)
    call_idx = np.asarray(uniq, dtype=np.int64)

    phase_sampled_idx: list = []
    phase_sampled_vals: list = []
    prev = 0
    for p in range(P):
        end = n_after[p] if p < len(n_after) else len(call_idx)
        end = max(prev, min(int(end), len(call_idx)))
        seg = call_idx[prev:end]
        prev = end
        phase_sampled_idx.append(seg)
        phase_sampled_vals.append(
            labels[seg].astype(np.int32) if len(seg) else np.zeros(0, dtype=np.int32)
        )

    return {
        "similarities": similarities,
        "phase_axes": phase_axes,
        "phase_threshold_pos": phase_threshold_pos,
        "phase_sampled_idx": phase_sampled_idx,
        "phase_sampled_vals": phase_sampled_vals,
        "phase_lambda": lams + [float("nan")] * (P - len(lams)),
        "phase_source": sources,
        "phase_n_labeled": n_after + [0] * (P - len(n_after)),
    }


def _render_phase_evolution_figure(tag: str,
                                    title: str,
                                    labels: np.ndarray,
                                    phases: list,
                                    out_path: str | None,
                                    pdf=None,
                                    show_samples: bool = True,
                                    norm: str = "zscore",
                                    pos_color: str = "#34495e",
                                    neg_color: str = "#c98176") -> str | None:
    """Render the per-phase axis-refinement evolution for one query.

    `phases`: list of dicts with keys 'axis' (len-N array), 'threshold_pos',
    'sampled_idx', 'sampled_vals', 'lam', 'source', 'n_labeled', 'auc'.

    The pos/neg histograms are the FULL GROUND-TRUTH distribution over all N
    rows (`labels` is the ground-truth vector) along each phase's z-scored axis
    -- NOT the LLM/oracle-sampled subset. The bottom rug (only when
    `show_samples=True`) marks the rows DIRECT queried in that phase. Right
    column = ROC-AUC vs phase.

    Pure numpy + matplotlib (no sklearn / no data IO) so it is unit-testable
    without the heavy stack. Returns `out_path`.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    labels = np.asarray(labels, dtype=np.int32)
    P = len(phases)
    if P == 0:
        return None
    n_total = int(len(labels))
    n_pos = int(np.sum(labels == 1))
    pos_rate = n_pos / max(n_total, 1)

    fig = plt.figure(figsize=(8.4, max(2.6, 1.7 * P + 1.0)))
    gs = gridspec.GridSpec(P, 2, figure=fig, width_ratios=[3.4, 1.15],
                           hspace=0.58, wspace=0.30)
    ax_trend = fig.add_subplot(gs[:, 1])

    aucs: list = []
    for p, ph in enumerate(phases):
        ax = fig.add_subplot(gs[p, 0])
        axis_vals = np.asarray(ph["axis"], dtype=np.float64)
        is_cos_phase = str(ph.get("source", "")).startswith("siglip")
        if norm == "cosine01":
            disp = (axis_vals if is_cos_phase
                    else _normalize_axis(axis_vals, "minmax"))
            _xlab = "cosine similarity (phase 0)  →  refined axis [0,1]"
        elif norm == "minmax":
            disp = _normalize_axis(axis_vals, "minmax")
            _xlab = "normalized axis [0,1]  (phase 0 = cosine sim)"
        elif norm == "raw":
            disp = axis_vals
            _xlab = "raw axis value (no normalization)"
        elif norm == "pm1":
            disp = _normalize_axis(axis_vals, "pm1")
            _xlab = "normalized axis [-1,1]  (phase 0 = cosine sim)"
        else:
            disp = _normalize_axis(axis_vals, "zscore")
            _xlab = "standardized axis (z-score)  (phase 0 = cosine sim)"
        _draw_dist_panel(
            ax, disp, labels, pos_color, neg_color,
            title=None, ylabel="density",
            n_total=None, pos_rate=None, annotate_box=False,
            xlabel=_xlab if p == P - 1 else None,
        )
        if norm == "pm1":
            ax.set_xlim(-1.06, 1.06)
        elif norm in ("cosine01", "minmax"):
            ax.set_xlim(-0.02, 1.02)
        # Rug: the rows DIRECT sampled in *this* phase (pos navy / neg coral),
        # placed just above the x-axis (x in data coords, y in axes coords).
        s_idx = np.asarray(ph.get("sampled_idx", []), dtype=np.int64)
        if show_samples and len(s_idx):
            s_vals = np.asarray(ph.get("sampled_vals", []), dtype=np.int32)
            xr = disp[s_idx]
            tr = ax.get_xaxis_transform()
            for cls, col in ((1, pos_color), (0, neg_color)):
                xs_c = xr[s_vals == cls]
                if len(xs_c):
                    ax.plot(xs_c, np.full(len(xs_c), 0.045), marker="|",
                            linestyle="none", color=col, markersize=7,
                            markeredgewidth=1.0, transform=tr, clip_on=False,
                            alpha=0.9)
        # DIRECT decision threshold(s) on this phase's axis: argmax of the
        # accuracy vector (best separating hypothesis). Positions are in
        # descending-axis sorted space; map to display-x via the same z-score
        # used for the histogram, so the line lands where DIRECT cuts.
        thr_pos = list(ph.get("threshold_pos", []) or [])
        thr_frac = None
        if thr_pos:
            so = np.argsort(-axis_vals)
            N = len(so)
            for i_thr, j in enumerate(thr_pos):
                # j = #elements on the positive (high-axis) side; boundary sits
                # between sorted indices j-1 and j (acc has length N+1).
                j = max(0, min(int(j), N))
                if 0 < j < N:
                    xthr = 0.5 * (disp[so[j - 1]] + disp[so[j]])
                elif j <= 0:
                    xthr = float(disp[so[0]])        # nothing above threshold
                else:                                 # j >= N -> all positive
                    xthr = float(disp[so[N - 1]])
                if i_thr == 0:
                    ax.axvline(xthr, color="#111111", linestyle="--",
                               linewidth=1.4, alpha=0.9, zorder=5)
                    thr_frac = j / max(N, 1)          # share predicted positive
                else:
                    ax.axvline(xthr, color="#777777", linestyle=":",
                               linewidth=1.1, alpha=0.8, zorder=5)
        lam = ph.get("lam", float("nan"))
        src = str(ph.get("source", ""))
        nlab = int(ph.get("n_labeled", 0))
        auc = ph.get("auc", float("nan"))
        aucs.append(auc)
        if src.startswith("blend"):
            src_short = "LR-blend"
        elif src.startswith("siglip"):
            src_short = "SigLIP sim"
        else:
            src_short = src or "—"
        lam_str = f"{lam:.2f}" if lam == lam else "—"      # nan-safe
        auc_str = f"{auc:.3f}" if auc == auc else "—"
        ax.text(0.015, 0.95, f"phase {p} · {src_short} · λ={lam_str}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                color="#222")
        cut_str = f"\ncut={thr_frac:.0%}" if thr_frac is not None else ""
        ax.text(0.985, 0.95, f"|L|={nlab}\nAUC={auc_str}{cut_str}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                color="#222",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="lightgray", alpha=0.9))

    # Right column: ROC-AUC vs phase (quantifies "gradually separable").
    xs = np.arange(P)
    ys = np.array([a if (a == a) else np.nan for a in aucs], dtype=np.float64)
    ax_trend.plot(xs, ys, marker="o", color="#264653", linewidth=1.8,
                  markersize=4)
    ax_trend.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax_trend.set_ylim(0.45, 1.02)
    ax_trend.set_xticks(xs)
    ax_trend.set_xlabel("phase", fontsize=8)
    ax_trend.set_title("separability\n(ROC-AUC)", fontsize=8)
    ax_trend.tick_params(axis="both", labelsize=7)
    ax_trend.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(
        f"{title}  —  DIRECT multi-phase axis refinement\n"
        f"N={n_total:,}, pos rate {pos_rate:.2%}",
        fontsize=11, fontweight="bold",
    )
    legend_handles = [
        Patch(facecolor=pos_color, label="positives (ground truth)"),
        Patch(facecolor=neg_color, label="negatives (ground truth)"),
        Line2D([0], [0], color="#111111", linestyle="--",
               label="DIRECT threshold"),
    ]
    if show_samples:
        legend_handles.append(
            Line2D([0], [0], color="#555555", marker="|", linestyle="none",
                   markersize=8, label="DIRECT-sampled rows")
        )
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=len(legend_handles),
        frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.07, right=0.97)

    _save_paper_fig(fig, out_path)            # PNG + sibling vector PDF
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _render_phase_scatter_figure(tag: str,
                                  title: str,
                                  labels: np.ndarray,
                                  phases: list,
                                  out_path: str | None,
                                  pdf=None,
                                  cumulative: bool = False,
                                  max_cloud: int = 6000,
                                  seed: int = 0,
                                  norm: str = "zscore",
                                  budget_pct: float | None = None,
                                  jitter: np.ndarray | None = None,
                                  pos_color: str = _VIZ_COL_POS,
                                  cloud_color: str = _VIZ_COL_ALL,
                                  sampled_color: str = _VIZ_COL_SAMPLED,
                                  ring_color: str = _VIZ_COL_RING) -> str | None:
    """Per-phase 1D strip/jitter scatter of the GROUND-TRUTH points along each
    phase's axis, with the rows DIRECT labeled in that phase highlighted.

    Rows = phases (top->bottom). In each panel: all rows are a faint jittered
    cloud (subsampled to `max_cloud`); every ground-truth positive is a large
    green dot; captured positives (labeled positives) get a gold ring and
    ALWAYS accumulate across phases (a positive labeled in an earlier round
    stays ringed in later panels); the dark "labeled row" dots show this phase's
    NEW labels by default, or the cumulative labeled set when `cumulative=True`;
    the DIRECT threshold is a dashed cutoff. As the axis refines, the green
    positives migrate past the cutoff and get captured.

    Pure numpy + matplotlib (no sklearn / IO) so it is unit-testable.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    labels = np.asarray(labels, dtype=np.int32)
    P = len(phases)
    if P == 0:
        return None
    N = int(len(labels))
    pos_all = np.where(labels == 1)[0]
    neg_all = np.where(labels == 0)[0]
    n_pos = int(len(pos_all))

    rng = np.random.default_rng(int(seed))
    yj = (np.asarray(jitter, dtype=np.float64) if jitter is not None
          else rng.uniform(-1.0, 1.0, size=N))   # jitter (shared if passed in)
    cloud_neg = (rng.choice(neg_all, size=max_cloud, replace=False)
                 if len(neg_all) > max_cloud else neg_all)

    fig = plt.figure(figsize=(7.0, max(2.2, 1.3 * P + 0.9)))
    gs = gridspec.GridSpec(P, 1, figure=fig, hspace=0.66)

    cum_idx = np.zeros(0, dtype=np.int64)
    for p, ph in enumerate(phases):
        ax = fig.add_subplot(gs[p, 0])
        axis_vals = np.asarray(ph["axis"], dtype=np.float64)
        is_cos_phase = str(ph.get("source", "")).startswith("siglip")
        if norm == "cosine01":
            # phase 0 / cold-start: keep the RAW cosine similarity (it is already
            # a [0,1] score); refined phases: min-max to [0,1] so the few sparse
            # positives migrate toward 1.0 as the axis sharpens.
            disp = (axis_vals if is_cos_phase
                    else _normalize_axis(axis_vals, "minmax"))
        elif norm == "raw":
            disp = axis_vals          # NO normalization; auto x-range per panel
        else:
            disp = _normalize_axis(axis_vals, norm)

        new_idx = np.asarray(ph.get("sampled_idx", []), dtype=np.int64)
        cum_idx = (np.unique(np.concatenate([cum_idx, new_idx]))
                   if len(new_idx) else cum_idx)
        # Captured positives ALWAYS accumulate: a positive labeled in an earlier
        # round stays marked (gold ring) in every later panel.
        cum_mask = np.zeros(N, dtype=bool)
        if len(cum_idx):
            cum_mask[cum_idx] = True
        lab_pos = np.where(cum_mask & (labels == 1))[0]    # cumulative captured pos
        # Dark "labeled row" dots: this phase's NEW labels by default (round
        # activity); the whole cumulative set when `cumulative=True`.
        neg_src = cum_idx if cumulative else new_idx
        neg_mask = np.zeros(N, dtype=bool)
        if len(neg_src):
            neg_mask[neg_src] = True
        lab_neg = np.where(neg_mask & (labels == 0))[0]    # labeled negatives

        # 1) faint cloud of all (subsampled) rows
        ax.scatter(disp[cloud_neg], yj[cloud_neg], s=3, c=cloud_color,
                   alpha=0.18, linewidths=0, zorder=1, rasterized=True)
        # 2) rows DIRECT labeled this phase (negatives), darker
        if len(lab_neg):
            ax.scatter(disp[lab_neg], yj[lab_neg], s=14, c=sampled_color,
                       alpha=0.8, linewidths=0, zorder=2)
        # 3) all ground-truth positives, green
        if len(pos_all):
            ax.scatter(disp[pos_all], yj[pos_all], s=70, c=pos_color,
                       edgecolors="white", linewidths=0.6, zorder=3)
        # 4) captured positives (labeled & positive): gold ring
        if len(lab_pos):
            ax.scatter(disp[lab_pos], yj[lab_pos], s=120, facecolors=pos_color,
                       edgecolors=ring_color, linewidths=2.2, zorder=4)
        # 5) DIRECT threshold cutoff
        tpos = list(ph.get("threshold_pos", []) or [])
        if tpos:
            so = np.argsort(-axis_vals)
            Nn = len(so)
            j = max(0, min(int(tpos[0]), Nn))
            if 0 < j < Nn:
                xthr = 0.5 * (disp[so[j - 1]] + disp[so[j]])
            elif j <= 0:
                xthr = float(disp[so[0]])
            else:
                xthr = float(disp[so[Nn - 1]])
            ax.axvline(xthr, color="#555555", linestyle="--", linewidth=1.1,
                       zorder=5)
            ax.text(xthr, 1.06, "threshold", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=_fs(9.5), color="#555555")

        n_cap = int(len(lab_pos))                                   # cumulative
        new_cap = int(np.sum(labels[new_idx] == 1)) if len(new_idx) else 0
        ax.set_title(
            f"Phase {p} · captured {n_cap}/{n_pos} positives "
            f"(+{new_cap} this phase)",
            fontsize=_fs(12), fontweight="bold",
        )
        ax.set_yticks([])
        ax.set_ylim(-1.3, 1.3)
        if norm == "pm1":
            ax.set_xlim(-1.06, 1.06)       # every panel spans [-1, 1]
        elif norm in ("cosine01", "minmax"):
            ax.set_xlim(-0.02, 1.02)       # every panel spans [0, 1]
        ax.tick_params(axis="x", labelsize=_fs(9))
        if p == P - 1:
            if norm == "pm1":
                _xlab = ""
            elif norm == "cosine01":
                _xlab = "cosine similarity (phase 0)  →  refined axis, min-max [0,1]"
            elif norm == "minmax":
                _xlab = "normalized axis [0,1]   (phase 0 = cosine similarity)"
            elif norm == "raw":
                _xlab = "raw axis value (no normalization; cosine @ phase 0, LR-blend later)"
            else:
                _xlab = "standardized axis (z-score)   (phase 0 = cosine similarity)"
            ax.set_xlabel(_xlab, fontsize=_fs(11))
        ax.grid(True, axis="x", linestyle=":", alpha=0.3)

    _bp = f", Budget=10%" if budget_pct is not None else ""
    fig.suptitle(f"Active-learning sampling{_bp}",
                 fontsize=_fs(13.5), fontweight="bold")
    handles = _viz_point_legend_handles()        # shared 4-entry legend
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=_fs(11), bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.98)

    _save_paper_fig(fig, out_path)            # PNG + sibling vector PDF
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _render_phase_separability_summary(df, out_path: str) -> str:
    """One line per query: ROC-AUC of the phase axis vs phase index."""
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    cmap = None
    if hasattr(matplotlib, "colormaps"):
        try:
            cmap = matplotlib.colormaps.get_cmap("tab20")
        except Exception:
            cmap = None
    for i, (q, sub) in enumerate(df.groupby("query")):
        sub = sub.sort_values("phase")
        kw = {"color": cmap(i % 20)} if cmap is not None else {}
        ax.plot(sub["phase"].to_numpy(), sub["auc"].to_numpy(),
                marker="o", markersize=3, linewidth=1.4, label=str(q), **kw)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("DIRECT phase")
    ax.set_ylabel("ROC-AUC of phase axis")
    ax.set_title("Per-phase axis separability across queries")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    _save_paper_fig(fig, out_path)            # PNG + sibling vector PDF
    plt.close(fig)
    return out_path


def plot_direct_phase_evolution(queries=None,
                                 sample_rate: float | None = None,
                                 budget_cap: int | None = None,
                                 n_phases: int | None = None,
                                 schedule: str | None = None,
                                 cold_start_phases: int = 1,
                                 vs_ratio: float = 0.5,
                                 B1: int = 5,
                                 lr_C: float = 1.0,
                                 seed: int = 0,
                                 show_samples: bool | None = None,
                                 style: str | None = None,
                                 scatter_cumulative: bool | None = None,
                                 norm: str | None = None,
                                 combined_pdf: bool = True) -> list:
    """Render the DIRECT multi-phase axis-refinement evolution for `queries`
    (list of (tag, title); defaults to DIRECT_EVOLUTION_QUERIES).

    `style` ('hist' | 'scatter' | 'both', env VIZ_EVO_STYLE) selects the
    histogram panels (`{tag}_direct_phase_evolution.png`) and/or the strip/jitter
    scatter of labeled points (`{tag}_direct_phase_scatter.png`). Both share a
    combined multipage PDF and the `direct_phase_separability.csv` + cross-query
    AUC-vs-phase summary PNG. Other env overrides: VIZ_EVO_SAMPLE_RATE (0.05),
    VIZ_EVO_BUDGET_CAP (3000), VIZ_EVO_PHASES (5), VIZ_EVO_SCHEDULE (linear -
    lambda ramps 0->1 across phases; 'lr_only' snaps to 1 after cold start),
    VIZ_EVO_SCATTER_CUMULATIVE (0), VIZ_EVO_NORM (cosine01 [default] - phase 0 =
    raw cosine on a [0,1] axis, refined phases min-max'd to [0,1]; minmax =
    min-max every phase to [0,1]; zscore = standardize; raw = NO normalization,
    auto-zoom each panel to the data range - best when cosine sits in a narrow
    band. The algorithm always sorts on the raw axis; this is display-only).
    """
    from sklearn.metrics import roc_auc_score
    from matplotlib.backends.backend_pdf import PdfPages

    queries = list(queries) if queries is not None else list(DIRECT_EVOLUTION_QUERIES)
    if sample_rate is None:
        sample_rate = float(os.environ.get("VIZ_EVO_SAMPLE_RATE", "0.05"))
    if budget_cap is None:
        budget_cap = int(os.environ.get("VIZ_EVO_BUDGET_CAP", "3000"))
    if n_phases is None:
        n_phases = int(os.environ.get("VIZ_EVO_PHASES", "5"))
    if schedule is None:
        # 'linear' ramps the LR-blend weight lambda 0 -> 1 across phases (with
        # phase 0 = pure similarity during cold start); 'lr_only' would instead
        # snap to lambda=1 right after cold start.
        schedule = os.environ.get("VIZ_EVO_SCHEDULE", "linear")
    if show_samples is None:
        show_samples = os.environ.get("VIZ_EVO_SHOW_SAMPLES", "1") not in (
            "0", "false", "False",
        )
    if style is None:
        style = os.environ.get("VIZ_EVO_STYLE", "hist")  # hist | scatter | both
    if scatter_cumulative is None:
        scatter_cumulative = os.environ.get("VIZ_EVO_SCATTER_CUMULATIVE", "0") in (
            "1", "true", "True",
        )
    if norm is None:
        # pm1: min-max each phase's axis to [-1, 1]. (cosine01 = raw cosine on
        # phase 0 then [0,1]; minmax = [0,1] every phase; zscore = standardize;
        # raw = no normalization)
        norm = os.environ.get("VIZ_EVO_NORM", "pm1")
    want_hist = style in ("hist", "both")
    want_scatter = style in ("scatter", "both")

    pdf_hist = pdf_scatter = None
    pdf_hist_path = os.path.join(results_dir, "direct_phase_evolution.pdf")
    pdf_scatter_path = os.path.join(results_dir, "direct_phase_scatter.pdf")
    if combined_pdf and want_hist:
        pdf_hist = PdfPages(pdf_hist_path)
    if combined_pdf and want_scatter:
        pdf_scatter = PdfPages(pdf_scatter_path)

    summary_rows: list = []
    try:
        for tag, title in queries:
            runner = QUERY_VIZ_RUNNERS.get(tag)
            if runner is None:
                print(f"[phase-evo] no runner for {tag}; skipping")
                continue
            try:
                print(f"[phase-evo] capturing {tag} ...")
                cap = _capture_sims_labels(runner)
            except Exception as e:
                print(f"[phase-evo] ERROR capturing {tag}: {e}")
                import traceback; traceback.print_exc()
                continue
            sims = np.asarray(cap["sims"], dtype=np.float32)
            labels = np.asarray(cap["labels"], dtype=np.int32)
            emb = cap.get("embeddings")
            if emb is None or len(emb) != len(sims):
                print(f"[phase-evo] {tag}: embeddings unavailable "
                      f"(emb={None if emb is None else len(emb)}, "
                      f"sims={len(sims)}); skipping")
                continue
            if len(np.unique(labels)) < 2:
                print(f"[phase-evo] {tag}: single-class labels; skipping")
                continue
            n = int(len(sims))
            budget = max(30, min(int(round(n * sample_rate)), int(budget_cap)))
            budget = min(budget, n)  # never request more labels than rows
            print(f"[phase-evo] {tag}: N={n}, budget={budget}, "
                  f"n_phases={n_phases}, schedule={schedule}")
            traj = _run_direct_phase_trajectory(
                emb, sims, labels, budget, seed=seed, n_phases=n_phases,
                schedule=schedule, cold_start_phases=cold_start_phases,
                vs_ratio=vs_ratio, B1=B1, lr_C=lr_C,
            )
            if traj is None:
                print(f"[phase-evo] {tag}: trajectory failed; skipping")
                continue

            phases: list = []
            for p, axis in enumerate(traj["phase_axes"]):
                try:
                    auc = float(roc_auc_score(labels, np.asarray(axis, dtype=np.float64)))
                except Exception:
                    auc = float("nan")
                tpos = traj["phase_threshold_pos"][p]
                thr0 = int(tpos[0]) if tpos else -1
                # thr0 = #rows on positive side -> predicted-positive fraction.
                thr_frac = (thr0 / n) if thr0 >= 0 else float("nan")
                phases.append({
                    "axis": axis,
                    "threshold_pos": tpos,
                    "sampled_idx": traj["phase_sampled_idx"][p],
                    "sampled_vals": traj["phase_sampled_vals"][p],
                    "lam": traj["phase_lambda"][p],
                    "source": traj["phase_source"][p],
                    "n_labeled": traj["phase_n_labeled"][p],
                    "auc": auc,
                })
                summary_rows.append({
                    "query": tag,
                    "phase": p,
                    "lambda": traj["phase_lambda"][p],
                    "axis_source": traj["phase_source"][p],
                    "n_labeled": traj["phase_n_labeled"][p],
                    "auc": auc,
                    "threshold_pos": thr0,
                    "threshold_frac": thr_frac,
                    "n_total": n,
                    "pos_rate": float(np.mean(labels == 1)),
                    "budget": int(budget),
                })

            if want_hist:
                out_png = os.path.join(results_dir, f"{tag}_direct_phase_evolution.png")
                _render_phase_evolution_figure(tag, title, labels, phases, out_png,
                                               pdf=pdf_hist, show_samples=show_samples,
                                               norm=norm)
                print(f"[phase-evo] saved -> {os.path.abspath(out_png)}")
            if want_scatter:
                out_scatter = os.path.join(results_dir, f"{tag}_direct_phase_scatter.png")
                _render_phase_scatter_figure(tag, title, labels, phases, out_scatter,
                                             pdf=pdf_scatter,
                                             cumulative=scatter_cumulative, seed=seed,
                                             norm=norm,
                                             budget_pct=100.0 * budget / max(n, 1))
                print(f"[phase-evo] saved -> {os.path.abspath(out_scatter)}")
    finally:
        if pdf_hist is not None:
            pdf_hist.close()
            print(f"[phase-evo] combined hist PDF -> {os.path.abspath(pdf_hist_path)}")
        if pdf_scatter is not None:
            pdf_scatter.close()
            print(f"[phase-evo] combined scatter PDF -> {os.path.abspath(pdf_scatter_path)}")

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(results_dir, "direct_phase_separability.csv")
        df.to_csv(csv_path, index=False)
        print(f"[phase-evo] separability summary -> {os.path.abspath(csv_path)}")
        try:
            summ_png = os.path.join(results_dir, "direct_phase_separability.png")
            _render_phase_separability_summary(df, summ_png)
            print(f"[phase-evo] summary figure -> {os.path.abspath(summ_png)}")
        except Exception as e:
            print(f"[phase-evo] summary figure failed: {e}")
    return summary_rows


def _selftest_direct_phase_evolution() -> str:
    """Synthetic end-to-end check WITHOUT real data / embedding server / Spark.

    Fabricates a 2-cluster problem where positives are only weakly separable on
    the raw similarity axis but well separated in embedding space, so the
    LR-refined axis should sharpen across phases. Runs the REAL `sample_direct`.
    Needs numpy + sklearn + matplotlib (NOT runnable where sklearn is absent).
    """
    rng = np.random.default_rng(0)
    n, d = 1500, 16
    n_pos = int(n * 0.12)
    y = np.zeros(n, dtype=np.int32)
    y[:n_pos] = 1
    rng.shuffle(y)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    mu = np.zeros(d, dtype=np.float32)
    mu[:4] = 1.4
    emb[y == 1] += mu
    # Weakly informative similarity (one noisy projection) so phase 0 AUC is
    # modest and the later LR phases visibly improve it.
    sims = (emb[:, 0] * 0.25
            + rng.standard_normal(n).astype(np.float32)).astype(np.float32)

    traj = _run_direct_phase_trajectory(emb, sims, y, budget=200, seed=0,
                                        n_phases=5, schedule="linear")
    assert traj is not None, "trajectory returned None"
    from sklearn.metrics import roc_auc_score
    phases: list = []
    for p, axis in enumerate(traj["phase_axes"]):
        auc = float(roc_auc_score(y, np.asarray(axis, dtype=np.float64)))
        tpos = traj["phase_threshold_pos"][p]
        phases.append({
            "axis": axis,
            "threshold_pos": tpos,
            "sampled_idx": traj["phase_sampled_idx"][p],
            "sampled_vals": traj["phase_sampled_vals"][p],
            "lam": traj["phase_lambda"][p],
            "source": traj["phase_source"][p],
            "n_labeled": traj["phase_n_labeled"][p],
            "auc": auc,
        })
        print(f"  phase {p}: src={traj['phase_source'][p]} "
              f"lam={traj['phase_lambda'][p]:.2f} "
              f"|L|={traj['phase_n_labeled'][p]} AUC={auc:.3f} "
              f"sampled={len(traj['phase_sampled_idx'][p])} "
              f"threshold_pos={tpos}")
    out = os.path.join(results_dir, "_selftest_direct_phase_evolution.png")
    _render_phase_evolution_figure("selftest", "Synthetic self-test", y, phases, out)
    out_sc = os.path.join(results_dir, "_selftest_direct_phase_scatter.png")
    _render_phase_scatter_figure("selftest", "Synthetic self-test", y, phases, out_sc)
    print(f"[phase-evo selftest] scatter -> {os.path.abspath(out_sc)}")
    tot = int(sum(len(s) for s in traj["phase_sampled_idx"]))
    print(f"[phase-evo selftest] total sampled={tot}; "
          f"AUC[0]={phases[0]['auc']:.3f} -> AUC[-1]={phases[-1]['auc']:.3f}")
    print(f"[phase-evo selftest] figure -> {os.path.abspath(out)}")
    return out


# ---------------------------------------------------------------------------
# Sampling-baseline motivation figure (uniform vs top-K) on the cosine axis.
# Mirrors the "why naive sampling misses the rare positives" reference figure:
# two stacked strip plots over raw cosine similarity, marking which sparse
# ground-truth positives each strategy captures vs misses.
# ---------------------------------------------------------------------------
def _render_sampling_baseline_figure(tag: str,
                                     title: str,
                                     sims: np.ndarray,
                                     labels: np.ndarray,
                                     budget: int,
                                     out_path: str | None,
                                     seed: int = 0,
                                     pdf=None,
                                     jitter: np.ndarray | None = None,
                                     max_cloud: int = 12000) -> str | None:
    """(a) uniform sampling vs (b) top-K similarity sampling on the cosine axis.

    Pure numpy + matplotlib. Each panel jitters all rows along y; negatives are
    faint (population) with the sampled ones darkened; positives are large
    (missed = light teal, sampled = dark teal with a gold ring). Panel (b) adds
    the top-K cutoff sigma and annotates the below-cutoff misses.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    sims = np.asarray(sims, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n = int(len(sims))
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = int(len(pos_idx))
    B = int(max(1, min(int(budget), n)))

    # Dedicated stream for the uniform draw so scanning `seed` (VIZ_BASELINE_SEED)
    # cleanly changes which rows uniform picks (and whether it captures 0).
    uni_sel = np.random.default_rng(int(seed)).choice(n, size=B, replace=False)
    rng_aux = np.random.default_rng(int(seed) + 1)
    yj = (np.asarray(jitter, dtype=np.float64) if jitter is not None
          else rng_aux.uniform(-1.0, 1.0, size=n))   # jitter (shared if passed in)
    order = np.argsort(-sims)
    topk_sel = order[:B]                                # top-K by cosine
    cutoff = float(sims[order[B - 1]])                  # raw cosine threshold (sigma)
    disp = _normalize_axis(sims, "pm1")                 # x-axis -> [-1, 1]
    cutoff_x = float(disp[order[B - 1]])                # cutoff position on [-1, 1]
    pct = 100.0 * B / max(n, 1)

    NEG_POP, NEG_S = _VIZ_COL_ALL, _VIZ_COL_SAMPLED
    POS_MISS, POS_S, RING = _VIZ_COL_POS, _VIZ_COL_POS, _VIZ_COL_RING
    neg_cloud = (rng_aux.choice(neg_idx, size=max_cloud, replace=False)
                 if len(neg_idx) > max_cloud else neg_idx)

    fig = plt.figure(figsize=(7.0, 4.8))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.6)

    for row, (name, sel, is_topk) in enumerate([
        ("Uniform sampling", uni_sel, False),
        ("Top-$K$ similarity sampling", topk_sel, True),
    ]):
        ax = fig.add_subplot(gs[row, 0])
        sel_mask = np.zeros(n, dtype=bool)
        sel_mask[sel] = True
        ax.scatter(disp[neg_cloud], yj[neg_cloud], s=4, c=NEG_POP, alpha=0.25,
                   linewidths=0, zorder=1, rasterized=True)
        sn = neg_idx[sel_mask[neg_idx]]
        ax.scatter(disp[sn], yj[sn], s=12, c=NEG_S, alpha=0.8, linewidths=0,
                   zorder=2, rasterized=True)
        miss = pos_idx[~sel_mask[pos_idx]]
        got = pos_idx[sel_mask[pos_idx]]
        ax.scatter(disp[miss], yj[miss], s=120, c=POS_MISS, edgecolors="white",
                   linewidths=0.8, zorder=3)
        ax.scatter(disp[got], yj[got], s=150, facecolors=POS_S, edgecolors=RING,
                   linewidths=2.6, zorder=4)
        ncap = int(len(got))

        if is_topk:
            ax.axvline(cutoff_x, color="#999999", linestyle="--", linewidth=1.2, zorder=5)

        ax.set_title(f"{name}, Budget={pct:.0f}%\n"
                     f"captured {ncap}/{n_pos} positives", fontsize=_fs(12.5))
        ax.set_yticks([])
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(-1.06, 1.06)
        ax.tick_params(axis="x", labelsize=_fs(9))
        if row == 1:
            ax.set_xlabel("normalized similarity  [-1, 1]", fontsize=_fs(11))

        if not is_topk:
            nmiss = n_pos - ncap
            txt = (f"all {n_pos} positives miss the sample" if ncap == 0
                   else f"{nmiss} of {n_pos} positives miss the sample")
            if len(miss):
                k = int(miss[np.argmax(sims[miss])])
                ax.annotate(txt, xy=(disp[k], yj[k]), xycoords="data",
                            xytext=(0.60, 0.14), textcoords="axes fraction",
                            fontsize=_fs(10.5), color=POS_S, ha="left", va="center",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=POS_S),
                            arrowprops=dict(arrowstyle="->", color=POS_S, lw=1.2))
        else:
            below = miss[sims[miss] < cutoff]
            txt = f"positives below the cutoff\nare missed ({len(below)} of {n_pos})"
            if len(below):
                k = int(below[np.argmax(sims[below])])
                ax.annotate(txt, xy=(disp[k], yj[k]), xycoords="data",
                            xytext=(0.05, 0.16), textcoords="axes fraction",
                            fontsize=_fs(10.5), color=POS_S, ha="left", va="center",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=POS_S),
                            arrowprops=dict(arrowstyle="->", color=POS_S, lw=1.2))

    handles = _viz_point_legend_handles()        # shared 4-entry legend
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=_fs(11), bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(top=0.92, bottom=0.13, left=0.05, right=0.98)
    _save_paper_fig(fig, out_path)            # PNG + sibling vector PDF
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sampling_baseline(queries=None, budget=None, sample_rate=None,
                           seed: int = 0) -> list:
    """Render the uniform-vs-top-K sampling-baseline figure for `queries`.

    Per query -> `{tag}_sampling_baseline.png`. Budget B (= K) defaults to
    round(N * sample_rate) with sample_rate from VIZ_EVO_SAMPLE_RATE (0.05);
    set VIZ_BASELINE_BUDGET for an absolute B, VIZ_BASELINE_SEED for the uniform
    draw (try a few if you want the 'all positives missed' case).
    """
    queries = list(queries) if queries is not None else list(DIRECT_EVOLUTION_QUERIES)
    if sample_rate is None:
        sample_rate = float(os.environ.get("VIZ_EVO_SAMPLE_RATE", "0.05"))
    abs_budget = os.environ.get("VIZ_BASELINE_BUDGET")
    seed = int(os.environ.get("VIZ_BASELINE_SEED", str(seed)))
    saved: list = []
    for tag, title in queries:
        runner = QUERY_VIZ_RUNNERS.get(tag)
        if runner is None:
            print(f"[baseline] no runner for {tag}; skipping")
            continue
        try:
            print(f"[baseline] capturing {tag} ...")
            cap = _capture_sims_labels(runner)
        except Exception as e:
            print(f"[baseline] ERROR capturing {tag}: {e}")
            import traceback; traceback.print_exc()
            continue
        sims = np.asarray(cap["sims"], dtype=np.float64)
        labels = np.asarray(cap["labels"], dtype=np.int32)
        if len(sims) == 0 or len(np.unique(labels)) < 2:
            print(f"[baseline] {tag}: empty or single-class; skipping")
            continue
        n = int(len(sims))
        B = int(abs_budget) if abs_budget else max(1, int(round(n * sample_rate)))
        B = min(B, n)
        out = os.path.join(results_dir, f"{tag}_sampling_baseline.png")
        _render_sampling_baseline_figure(tag, title, sims, labels, B, out, seed=seed)
        print(f"[baseline] {tag}: N={n}, B=K={B} -> {os.path.abspath(out)}")
        saved.append(out)
    return saved


def plot_combined_scatter_baseline(queries=None, sample_rate=None, n_phases=None,
                                   schedule=None, seed: int = 0) -> list:
    """Render the Active-learning phase scatter AND the uniform/top-K sampling
    baseline for each query from the SAME captured data and the SAME y-jitter,
    so the ground-truth positives sit at identical positions in both figures.
    Writes TWO separate files per query (each PNG + vector PDF):
        {tag}_direct_phase_scatter.{png,pdf}   and  {tag}_sampling_baseline.{png,pdf}

    Sharing: both use the same (sims, labels), the same jitter (seeded by
    `seed`), and the same [-1,1] cosine normalization for the cosine axis, and
    the same budget B = round(N * sample_rate). The uniform draw seed is
    VIZ_BASELINE_SEED (independent of the jitter).
    """
    from sklearn.metrics import roc_auc_score

    queries = list(queries) if queries is not None else list(DIRECT_EVOLUTION_QUERIES)
    if sample_rate is None:
        sample_rate = float(os.environ.get("VIZ_EVO_SAMPLE_RATE", "0.05"))
    if n_phases is None:
        n_phases = int(os.environ.get("VIZ_EVO_PHASES", "5"))
    if schedule is None:
        schedule = os.environ.get("VIZ_EVO_SCHEDULE", "linear")
    budget_cap = int(os.environ.get("VIZ_EVO_BUDGET_CAP", "3000"))
    base_seed = int(os.environ.get("VIZ_BASELINE_SEED", "0"))
    saved: list = []
    for tag, title in queries:
        runner = QUERY_VIZ_RUNNERS.get(tag)
        if runner is None:
            print(f"[combined] no runner for {tag}; skipping")
            continue
        try:
            print(f"[combined] capturing {tag} ...")
            cap = _capture_sims_labels(runner)
        except Exception as e:
            print(f"[combined] ERROR capturing {tag}: {e}")
            import traceback; traceback.print_exc()
            continue
        sims = np.asarray(cap["sims"], dtype=np.float32)
        labels = np.asarray(cap["labels"], dtype=np.int32)
        emb = cap.get("embeddings")
        n = int(len(sims))
        if n == 0 or len(np.unique(labels)) < 2:
            print(f"[combined] {tag}: empty or single-class; skipping")
            continue
        budget = max(30, min(int(round(n * sample_rate)), budget_cap))
        budget = min(budget, n)
        pct = 100.0 * budget / max(n, 1)
        # ONE shared y-jitter so positives line up across both figures.
        jitter = np.random.default_rng(int(seed)).uniform(-1.0, 1.0, size=n)

        # (1) sampling baseline (cosine axis only — needs no embeddings)
        out_b = os.path.join(results_dir, f"{tag}_sampling_baseline.png")
        _render_sampling_baseline_figure(tag, title, sims, labels, budget, out_b,
                                         seed=base_seed, jitter=jitter)
        print(f"[combined] {tag}: baseline -> {os.path.abspath(out_b)}")
        saved.append(out_b)

        # (2) active-learning phase scatter (needs embeddings + DIRECT trajectory)
        if emb is None or len(emb) != n:
            print(f"[combined] {tag}: embeddings unavailable; phase scatter skipped")
            continue
        traj = _run_direct_phase_trajectory(emb, sims, labels, budget, seed=seed,
                                            n_phases=n_phases, schedule=schedule)
        if traj is None:
            print(f"[combined] {tag}: DIRECT trajectory failed; phase scatter skipped")
            continue
        phases = []
        for p, axis in enumerate(traj["phase_axes"]):
            try:
                auc = float(roc_auc_score(labels, np.asarray(axis, dtype=np.float64)))
            except Exception:
                auc = float("nan")
            phases.append({
                "axis": axis, "threshold_pos": traj["phase_threshold_pos"][p],
                "sampled_idx": traj["phase_sampled_idx"][p],
                "sampled_vals": traj["phase_sampled_vals"][p],
                "lam": traj["phase_lambda"][p], "source": traj["phase_source"][p],
                "n_labeled": traj["phase_n_labeled"][p], "auc": auc,
            })
        out_s = os.path.join(results_dir, f"{tag}_direct_phase_scatter.png")
        _render_phase_scatter_figure(tag, title, labels, phases, out_s,
                                     cumulative=False, seed=seed, norm="pm1",
                                     budget_pct=pct, jitter=jitter)
        print(f"[combined] {tag}: phase scatter -> {os.path.abspath(out_s)}")
        saved.append(out_s)
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
QUERY_VIZ_RUNNERS = {
    # "movie_q5": viz_movie_q5,
    # "movie_q7": viz_movie_q7,
    "ecomm_q7": viz_ecomm_q7,
   # "ecomm_q6": viz_ecomm_q6,
    "ecomm_q8": viz_ecomm_q8,
    "ecomm_q9": viz_ecomm_q9,
    #"ecomm_q9_prompt_fused": viz_ecomm_q9_prompt_fused,
    "ecomm_q13": viz_ecomm_q13,
    "cars_q1": viz_cars_q1,
    "cars_q3": viz_cars_q3,
    "cars_q4": viz_cars_q4,
    "cars_q8": viz_cars_q8,
   # "cars_q10": viz_cars_q10,
    "mmqa_q2a": viz_mmqa_q2a,
    "mmqa_q7": viz_mmqa_q7,
    # Animals benchmark
    "animals_q1": viz_animals_q1,
    "animals_q3": viz_animals_q3,
    "animals_q10": viz_animals_q10,
    # JoinML paper datasets
    "veri_q1": viz_veri_q1,
    "flickr_q1": viz_flickr_q1,
    "roxford_q1": viz_roxford_q1,
    "company_q1": viz_company_q1,
    # LRobench match queries (text-text semantic joins)
    "match102": viz_match102,
    "match104": viz_match104,
    "match105": viz_match105,
    "match106": viz_match106,
    "match107": viz_match107,
    "match113": viz_match113,
    "match121": viz_match121,
    "match202": viz_match202,
    "match305": viz_match305,
}


def main():
    filter_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    # `grid` -> render the curated 14-panel 2x7 PDF over GRID14_QUERIES.
    if filter_arg == "grid":
        plot_combined_grid_pdf()
        if spark:
            spark.stop()
        return
    # `axis_grid` -> render the 14-panel axis-refinement comparison
    # (similarity above, LR-blended axis below).
    if filter_arg == "axis_grid":
        sample_rate = float(os.environ.get("VIZ_AXIS_SAMPLE_RATE", "0.10"))
        lam = float(os.environ.get("VIZ_AXIS_LAMBDA", "1.0"))
        seed = int(os.environ.get("VIZ_AXIS_SEED", "0"))
        use_direct = os.environ.get("VIZ_AXIS_USE_DIRECT", "1") not in ("0", "false", "False")
        budget_cap = int(os.environ.get("VIZ_AXIS_DIRECT_BUDGET_CAP", "200000"))
        n_phases = int(os.environ.get("VIZ_AXIS_DIRECT_PHASES", "4"))
        plot_axis_refinement_grid(
            sample_rate=sample_rate, lam=lam, seed=seed,
            use_direct=use_direct, direct_budget_cap=budget_cap,
            direct_n_phases=n_phases,
        )
        if spark:
            spark.stop()
        return
    # `direct_evolution` -> per-query multi-phase axis-refinement evolution for
    # the 14 requested queries. `direct_evolution:<tag>` for a single query.
    # `direct_evolution_selftest` -> synthetic check (no data / Spark needed).
    if filter_arg == "direct_evolution_selftest":
        _selftest_direct_phase_evolution()
        if spark:
            spark.stop()
        return
    if filter_arg == "direct_evolution":
        plot_direct_phase_evolution()
        if spark:
            spark.stop()
        return
    if filter_arg.startswith("direct_evolution:"):
        tag = filter_arg.split(":", 1)[1]
        title = dict(DIRECT_EVOLUTION_QUERIES).get(tag, tag)
        plot_direct_phase_evolution(queries=[(tag, title)])
        if spark:
            spark.stop()
        return
    # Strip/jitter scatter of each phase's labeled points (GT positives green,
    # captured positives ringed). `direct_evolution_scatter[:<tag>]`.
    if filter_arg == "direct_evolution_scatter":
        plot_direct_phase_evolution(style="scatter")
        if spark:
            spark.stop()
        return
    if filter_arg.startswith("direct_evolution_scatter:"):
        tag = filter_arg.split(":", 1)[1]
        title = dict(DIRECT_EVOLUTION_QUERIES).get(tag, tag)
        plot_direct_phase_evolution(queries=[(tag, title)], style="scatter")
        if spark:
            spark.stop()
        return
    # Uniform-vs-top-K sampling baseline (the "why naive sampling misses the
    # rare positives" motivation figure). `sampling_baseline[:<tag>]`.
    if filter_arg == "sampling_baseline":
        plot_sampling_baseline()
        if spark:
            spark.stop()
        return
    if filter_arg.startswith("sampling_baseline:"):
        tag = filter_arg.split(":", 1)[1]
        title = dict(DIRECT_EVOLUTION_QUERIES).get(tag, tag)
        plot_sampling_baseline(queries=[(tag, title)])
        if spark:
            spark.stop()
        return
    # `combined[:<tag>]` -> phase scatter + sampling baseline from the SAME data
    # and SAME jitter (positives line up), written as two separate PDFs.
    if filter_arg == "combined":
        plot_combined_scatter_baseline()
        if spark:
            spark.stop()
        return
    if filter_arg.startswith("combined:"):
        tag = filter_arg.split(":", 1)[1]
        title = dict(DIRECT_EVOLUTION_QUERIES).get(tag, tag)
        plot_combined_scatter_baseline(queries=[(tag, title)])
        if spark:
            spark.stop()
        return
    if filter_arg in QUERY_VIZ_RUNNERS:
        queries = [filter_arg]
    elif filter_arg in ("ecomm", "movie", "mmqa", "cars", "animals",
                        "veri", "flickr", "roxford", "company", "match"):
        queries = [q for q in QUERY_VIZ_RUNNERS if q.startswith(filter_arg)]
    else:
        queries = list(QUERY_VIZ_RUNNERS.keys())

    print(f"Running similarity-distribution viz for: {queries}")
    summary_rows: list = []
    for q in queries:
        try:
            QUERY_VIZ_RUNNERS[q](summary_rows)
        except Exception as e:
            print(f"ERROR running viz for {q}: {e}")
            import traceback
            traceback.print_exc()

    if summary_rows:
        out_df = pd.DataFrame(summary_rows)
        out_df.to_csv(SEPARABILITY_CSV, index=False)
        print(f"\nSeparability summary saved -> {SEPARABILITY_CSV}")
        print(out_df.to_string(index=False))

    if spark:
        spark.stop()


if __name__ == "__main__":
    main()
