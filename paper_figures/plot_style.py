"""Shared color palette + matplotlib style for every paper figure script."""
from __future__ import annotations

import matplotlib.pyplot as plt


# ---- canonical 5-color palette (used on Figs 5, 6, 11 and re-used on others) ----
PALETTE = {
    "Lotus":              "#A7B4C7",
    "ThalamusDB":         "#7FA8C9",
    "Palimpzest":         "#C7B299",
    "UQE":                "#9A82A6",
    "BARGAIN":            "#4F9D9D",
    "Baseline(SparkSQL)": "#3A3A3A",
    "LASO (no router)":   "#E0A3A3",  # lighter LASO red: LASO without router
    "LASO (with router)": "#B85C5C",  # canonical LASO red: LASO with router
    "Ours":               "#B85C5C",  # back-compat alias for older scripts
}

SYS_LABELS = {
    "Lotus":              "Lotus",
    "ThalamusDB":         "ThalamusDB",
    "Palimpzest":         "Palimpzest",
    "UQE":                "UQE",
    "BARGAIN":            "BARGAIN",
    "Baseline(SparkSQL)": "No-proxy baseline",
    "LASO (no router)":   "LASO without router",
    "LASO (with router)": "LASO with router",
    "Ours":               "LASO",
}

# Order used on the per-query bar charts (Figures 6 and 7), left to right.
# UQE replaces Palimpzest as the proxy/AQP baseline; Palimpzest data is kept in
# the workbook but not plotted. BARGAIN is an additional AQP/proxy baseline.
# The two LASO variants sit at the right, with "without router" immediately to
# the LEFT of "with router" so the router's marginal effect reads left-to-right.
SYSTEMS = ["Lotus", "ThalamusDB", "UQE", "BARGAIN", "Baseline(SparkSQL)",
           "LASO (no router)", "LASO (with router)"]

# Phase palette for the time-breakdown figure: 6-stage pipeline.
# Embedding+Sampling are the two blues, Labeling is the dominant red, and the
# three small proxy/graph stages use a warm sage-green family that contrasts
# the blue/red duotone while staying low-saturation enough for stacked bars.
PHASE_PALETTE = {
    "Embedding":         "#4E78A0",  # muted steel blue
    "Sampling":          "#A8C0D9",  # pale slate blue
    "Labeling":          "#C46A6A",  # muted dusty red (dominant slice)
    "Label Propagation": "#7A9A6B",  # muted sage green (distinct hue, harmonious)
    "Proxy Train":       "#C8A45C",  # warm muted gold
    "Proxy Predict":     "#E0CB99",  # pale sand
    "LLM rollback":      "#8E6FA8",  # muted plum (router fallback MLLM overhead)
}


CANONICAL_QUERIES = (
    "mmqa_q2a", "mmqa_q7",
    "ecomm_q7", "ecomm_q8", "ecomm_q9", "ecomm_q13",
    "cars_q3", "cars_q4", "cars_q8", "animals_q1",
    "match113", "match121", "match202", "match305",
)


def short_q(q: str) -> str:
    s = str(q)
    # LRO-Bench joins are displayed with an LRO_ prefix, e.g. match305 -> LRO_305.
    if s.startswith("match"):
        return "LRO_" + s[len("match"):]
    if s.startswith("lro_"):
        return "LRO_" + s[len("lro_"):]
    return (s.replace("_q", " Q").replace("ecomm", "Ecomm").replace("mmqa", "MMQA")
              .replace("cars", "Cars").replace("animals", "Animals")
              .replace("movie", "Movie"))


def apply_paper_style(bold: bool = True) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelweight": "bold" if bold else "normal",
        "axes.titleweight": "bold" if bold else "normal",
        "legend.frameon": False,
    })


def order_queries(qs):
    # Only the canonical workload is plotted; non-canonical entries in a sheet
    # (e.g. incomplete movie_q5 rows with a single system) are dropped so they
    # never leak into a figure as a near-empty panel.
    qset = set(qs)
    return [q for q in CANONICAL_QUERIES if q in qset]
