from fixed_ce_config import runtime_options as _fixed_ce_runtime_options
import os as _os  # noqa: E402  (anonymized path config)
DATA_ROOT = _os.environ.get(
    "LASO_DATA_ROOT",
    _os.environ.get("LLMSQL_DATA_ROOT", _os.environ.get("PROBE_DATA_ROOT", "data")),
)
_os.environ.setdefault("LASO_DATA_ROOT", DATA_ROOT)
_os.environ.setdefault("LLMSQL_DATA_ROOT", DATA_ROOT)

import contextlib
import json
import os


import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if (_parent / "omnimcp" / "server").exists():
        sys.path.insert(0, str(_parent))
        break

import time
import re
import urllib.request
import urllib.error
from typing import Any, Dict, Optional
import numpy as np
if not hasattr(np, "object0"):
    np.object0 = np.object_
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, concat, lit, length
from pyspark.sql.types import MapType, StructType, ArrayType
from proxy_model import get_last_profiling, get_last_token_usage, reset_token_usage
import sem_ops_register  # noqa: F401  -- import registers df.sem_filter / df.sem_join (side effect)
from router_cap import split_total_budget
from learned_calibration import (
    LEARNED_CALIBRATION_MODE,
    LEARNED_CALIBRATION_RESULT_KEYS,
    LearnedCalibrationConfig,
    learned_calibration_total_budget,
)

MODEL = os.environ.get("SEMBENCH_MODEL", "google/gemma-4-26B-A4B-it")   # default oracle since 2026-09-24 (was Qwen/Qwen3-VL-30B-A3B-Instruct)
#MODEL = os.environ.get("SEMBENCH_MODEL", "Qwen/Qwen3.6-35B-A3B")

EMBEDDING_MODEL = f"{DATA_ROOT}/siglip-base-patch16-224"

# vLLM exposes Prometheus counters at /metrics on the same port as the OpenAI
# API. We snapshot them before/after each experiment so that the baseline
# (Java-side) path, which is invisible to the Python token counter, still
# reports a true request/token count.
VLLM_METRICS_URL = os.environ.get(
    "VLLM_METRICS_URL", "http://localhost:8000/metrics"
)

# ---------------------------------------------------------------------------
# Embedding model ablation grid
# ---------------------------------------------------------------------------
# For prompts that involve at least one image column:
EMBEDDING_MODELS_MULTIMODAL = [
    f"{DATA_ROOT}/siglip-base-patch16-224",   # SigLIP baseline
    #f"{DATA_ROOT}/clip-vit-base-patch32",                # CLIP base
    #f"{DATA_ROOT}/clip-vit-large-patch14-336",               # CLIP large
    #f"{DATA_ROOT}/gme-Qwen2-VL-2B-Instruct",     # GME-Qwen2-VL joint
]

# For text-only prompts (no image columns):
EMBEDDING_MODELS_TEXT = [
    f"{DATA_ROOT}/siglip-base-patch16-224",         # SigLIP text encoder
    #f"{DATA_ROOT}/all-MiniLM-L6-v2",           # MiniLM-L12-v2
]

# Modality per query: "multimodal" = has image columns, "text" = text-only
QUERY_MODALITY = {
    name: 'multimodal' if 'image' in workload['modality'] else 'text'
    for name, workload in json.loads((Path(__file__).parent/'experiments/paper_manifest.json').read_text())['workloads'].items()
}


def get_embedding_models_for_query(query_tag: str) -> list:
    """Return the embedding model list appropriate for this query's modality."""
    modality = QUERY_MODALITY.get(query_tag, "multimodal")
    return EMBEDDING_MODELS_MULTIMODAL if modality == "multimodal" else EMBEDDING_MODELS_TEXT


from proxy_model_backends import emb_short, release_backend  # noqa: E402


def _iter_emb_models(query_tag: str):
    """Iterate embedding models for a query with per-model GPU release.

    Each backend (SigLIP, CLIP-B, CLIP-L, GME-Qwen2-VL, ...) lazily loads
    its weights onto the GPU and previously held them forever in
    proxy_model_backends._BACKEND_CACHE. Iterating the four-model
    multimodal list therefore stacks all four models in VRAM, OOMing the
    last one (typically GME-2B) — which then escaped through
    run's outer try/except and aborted the whole
    query, leaving only SigLIP results on disk.

    This generator releases each backend (and its sibling SigLIP cache
    entry, when applicable) the moment the caller moves on, so each
    embedding loads into a clean GPU. Pair with _emb_scope at the call
    site to also contain a single embedding's failure.
    """
    for emb_model in get_embedding_models_for_query(query_tag):
        try:
            yield emb_model
        finally:
            release_backend(emb_model)


@contextlib.contextmanager
def _emb_scope(query_tag: str, emb_model: str):
    """Per-embedding error containment.

    Catches any exception raised while running the (ablation × budget ×
    k) cross-product for one embedding model so the remaining embeddings
    in the same query still execute. Use as:

        for emb_model in _iter_emb_models("ecomm_q7"):
            with _emb_scope("ecomm_q7", emb_model):
                for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                    ...
    """
    try:
        yield
    except Exception as e:
        print(f"[{query_tag}] embedding {emb_model} failed: {e}")
        import traceback
        traceback.print_exc()

BUDGET_PCTS = [10]

QUERY_BUDGET_PCTS: dict = {}

def get_budget_pcts_for_query(query_tag: str) -> list:
    """Return the budget-percentage list for `query_tag`.

    Falls back to the global BUDGET_PCTS when the query has no override.
    """
    return QUERY_BUDGET_PCTS.get(query_tag, BUDGET_PCTS)


KNN_K_VALUES = ["auto"]
RANDOM_SEED = int(os.environ.get("LASO_RANDOM_SEED", "0"))
# Patch R1: fraction of the total LLM budget the router may spend (env: LASO_ROUTER_CAP_FRAC).
ROUTER_CAP_FRAC = float(os.environ.get("LASO_ROUTER_CAP_FRAC", "0.3"))
# Preparation covers training and calibration. Routing and S3 can query the remaining population.
LC_SAMPLE_FRAC = float(os.environ.get("LASO_LC_SAMPLE_FRAC", "0.6"))
LC_CALIB_FRAC = float(os.environ.get("LASO_LC_CALIB_FRAC", "0.4"))
LC_ROUTE_FRAC = float(os.environ.get("LASO_LC_ROUTE_FRAC", "0.0"))
# Scales the total learned-calibration budget relative to the nominal budget_pct. 1.0 carves the
# calibration sample and the router out of the budget; 1.5 with fracs 2/3, 4/15, 1/15 instead keeps
# sampling at the full nominal budget and ADDS 0.4x for calibration and 0.1x for routing, matching how
# PROBE's deferral router is charged. The calibration sample is LLM-labelled, so those calls are part
# of the answer, not an overhead.
LC_BUDGET_SCALE = float(os.environ.get("LASO_LC_BUDGET_SCALE", "1.0"))
# Conservative fallback when the calibration sample A contains no positive: mix a Clopper-Pearson
# upper bound into the anchoring target (Kim et al., arXiv:2606.08090 §5.2). 0 disables it.
# 0.15 was picked by the sweep in results/lc_cp_sweep (5 lambda x 8 queries at 10%): it is the only
# value that never lost against refusing, and it sits at the Beta posterior median of the bound.
LC_CP_LAM = float(os.environ.get("LASO_LC_CP_LAM", "0.15"))
LC_CP_ALPHA = float(os.environ.get("LASO_LC_CP_ALPHA", "0.05"))
# Cost-quality dial within the remaining routing allowance. The 0.5 default was
# inherited from historical sweeps; it has not been retuned for the fixed-CE method.
LC_ALPHA = float(os.environ.get("LASO_LC_ALPHA", "0.5"))
# Neyman allocation of the calibration sample A (with an inclusion floor); 0 restores the even split.
LC_NEYMAN = os.environ.get("LASO_LC_NEYMAN", "1").lower() in {"1", "true", "yes", "on"}
# 1 (default): top-up rounds of A may draw on whatever part of B that L and A left unspent. With no
# routing share there is nothing else to draw on, so these rounds are rare.
LC_FALLBACK_UNLIMITED = os.environ.get("LASO_LC_FALLBACK_UNLIMITED", "0").lower() in {"1", "true", "yes", "on"}
# 1 (default): routing has no budget cap; alpha is the only thing deciding how many rows get verified.
LC_ROUTE_UNLIMITED = os.environ.get("LASO_LC_ROUTE_UNLIMITED", "1").lower() in {"1", "true", "yes", "on"}
# "all": sampling, calibration and routing share the total budget; "preparation": the budget covers
# sampling + calibration only and routing is outside it (the paper protocol, with LASO_LC_ROUTE_UNLIMITED=1).
# With the fixed-CE calibration, "all" caps uncapped routing at what is left of the total budget.
LC_BUDGET_SCOPE = os.environ.get("LASO_LC_BUDGET_SCOPE", "preparation").strip().lower()
# >0: routing gets its own budget of this multiple of B, on top of sampling+calibration (D <= mult*B).
LC_ROUTE_BUDGET_MULT = float(os.environ.get("LASO_LC_ROUTE_BUDGET_MULT", "0"))
# Calibrator arm: "platt" (a line in log-odds) or GAMCAL's spline (arXiv:2604.00660 App. A.2).
LC_CALIBRATOR = os.environ.get("LASO_LC_CALIBRATOR", "platt")
# Calibrator input: "proxy" = LR proxy logit (paper Sec. 5), "refined" = logit of the graph-refined score z_smooth.
LC_CALIB_INPUT = os.environ.get("LASO_LC_CALIB_INPUT", "refined").strip().lower()
LC_GAM_RIDGE = float(os.environ.get("LASO_LC_GAM_RIDGE", "1.0"))
LC_GAM_KNOTS = int(os.environ.get("LASO_LC_GAM_KNOTS", "8"))
LC_GAM_STOCHASTIC = os.environ.get("LASO_LC_GAM_STOCHASTIC", "0") == "1"
# Routing objective. cost_exp > 1 is what turns alpha from a switch into a dial under a budget cap:
# see plan_dual_threshold's docstring for the measured alpha -> routed-fraction tables.
# R in the appendix: cross-fitting folds of the calibration stage. Each fold re-runs graph
# refinement, so R is a compute knob as well as a statistical one; the effective value is clipped
# by the rarer class of L and is reported as lc_n_folds_used.
LC_FOLDS = int(os.environ.get("LASO_LC_FOLDS", "5"))
LC_ROUTE_OBJECTIVE = os.environ.get("LASO_LC_ROUTE_OBJECTIVE", "pool")
LC_ROUTE_COST_EXP = float(os.environ.get("LASO_LC_ROUTE_COST_EXP", "1.0"))

# Active-learning sampler config (paper §3).
def _cascade_decision_enabled(decision_mode: Optional[str] = None) -> bool:
    """Whether the active-learning + Bias Diffusion + Homogeneity Propagation path runs a post-Bias Diffusion + Homogeneity Propagation decision layer."""
    mode = str(decision_mode if decision_mode is not None else LEARNED_CALIBRATION_MODE)
    if mode == "none":
        return False
    return mode == LEARNED_CALIBRATION_MODE


LLM_BATCH_SIZE = 128


# Ablation grid: (label, sampling_strategy, refine_enabled)
# sampling_strategy: "activeSampling" = fixed predicate-aware CE + pilot-gated acquisition; legacy is explicit
#                    "random"     = uniform random
#                    "uqe_active" = UQE Algorithm 2 (arXiv 2407.09522)
#                    "top_k"      = deterministic top-K along fixed cosine
#                                   similarity axis (Proxy-100x §5.4 baseline)
# refine_enabled: True = Bias Diffusion + Homogeneity Propagation postprocess, False = LR-only prediction
ABLATION_CONFIGS = [('activeSampling+refine+learned_calibration','activeSampling',True)]


def _decision_mode_for_ablation(ablation_label: str) -> str:
    """Map experiment labels to the post-Bias Diffusion + Homogeneity Propagation policy under test."""
    label = str(ablation_label)
    if label.endswith("+trust_proxy"):
        return "none"
    if label.endswith("+learned_calibration"):
        return LEARNED_CALIBRATION_MODE
    # +dasp_topk is unified into the learned-deferral router (the DASP/top-k
    # decision path was removed); both map to the same policy.
    if label.endswith("+learned_deferral") or label.endswith("+dasp_topk"):
        raise ValueError("Retired router; use learned_calibration")
    if label.endswith("+lr_only"):
        # LR-only ablation bypasses Bias Diffusion + Homogeneity Propagation entirely; no post-Bias Diffusion + Homogeneity Propagation decision layer.
        return "none"
    return "learned_calibration"


def _deferral_policy_for_ablation(ablation_label: str) -> str:
    mode = _decision_mode_for_ablation(ablation_label)
    if mode == "none":
        return "trust_proxy"
    if mode == "learned_deferral":
        raise ValueError("Retired router; use learned_calibration")
    return mode


def _learned_calibration_config(n_items: int, budget_pct: float) -> LearnedCalibrationConfig:
    """Per-run config for the +learned_calibration ablation (total-budget semantics)."""
    return LearnedCalibrationConfig(
        total_budget=learned_calibration_total_budget(n_items, budget_pct * LC_BUDGET_SCALE),
        sample_frac=LC_SAMPLE_FRAC,
        calib_frac=LC_CALIB_FRAC,
        route_frac=LC_ROUTE_FRAC,
        alpha=LC_ALPHA,
        neyman=LC_NEYMAN,
        fallback_unlimited=LC_FALLBACK_UNLIMITED,
        route_unlimited=LC_ROUTE_UNLIMITED,
        route_budget_mult=LC_ROUTE_BUDGET_MULT,
        budget_scope=LC_BUDGET_SCOPE,
        cp_lam=LC_CP_LAM,
        cp_alpha=LC_CP_ALPHA,
        calibrator=LC_CALIBRATOR,
        calib_input=LC_CALIB_INPUT,
        gam_ridge=LC_GAM_RIDGE,
        gam_knots=LC_GAM_KNOTS,
        gam_stochastic=LC_GAM_STOCHASTIC,
        n_folds=LC_FOLDS,
        route_objective=LC_ROUTE_OBJECTIVE,
        route_cost_exp=LC_ROUTE_COST_EXP,
        seed=RANDOM_SEED,
        protocol=os.environ.get("PROBE_CALIBRATION_PROTOCOL", "paper"), fixed_calibration=True,
        rho_a=0. if LC_CALIB_INPUT == "refined" else .25, rho_b=0., rho_c=0.,
    )

spark = None
all_results = []
results_dir = "./results/activeSampling_grid/"
INCLUDE_NO_OPT = os.environ.get("LASO_INCLUDE_NO_OPT", "0").lower() in {"1", "true", "yes", "on"}
# Join pair representation for the LR proxy: "mean" reproduces the shipped behaviour,
# "interaction" uses [cos, Pu*Pv, |Pu-Pv|] (docs/design/2026-09-17-join-repr-experiments.md).
PAIR_FEATURE_MODE = os.environ.get("LASO_PAIR_FEATURE_MODE", "mean").strip().lower()
if PAIR_FEATURE_MODE not in {"mean", "interaction"}:
    raise SystemExit(f"LASO_PAIR_FEATURE_MODE must be mean or interaction, got {PAIR_FEATURE_MODE!r}")


# ---------------------------------------------------------------------------
# Environment-variable configuration overrides (the experiments/ command CLI).
# Every override defaults to the value defined above, so an UNSET environment
# reproduces the original hardcoded behaviour exactly. Each experiments/*.sh
# script exports only the variables it varies. See experiments/README.md.
# ---------------------------------------------------------------------------
_KNOWN_SAMPLERS = {"activeSampling", "random", "top_k", "uqe_active", "stratified"}
_KNOWN_ABLATION_PARTS = {"refine", "lr_only", "trust_proxy",
                         "learned_deferral", "dasp_topk", "learned_calibration"}


def _parse_ablation_label(label):
    """Map a label like 'activeSampling+refine+trust_proxy' to (label, sampler, refine_enabled).

    The router/decision policy is carried by the label suffix and resolved by
    the existing ``_decision_mode_for_ablation`` (+trust_proxy / +learned_deferral
    / +dasp_topk / +lr_only).
    """
    parts = label.split("+")
    strat = parts[0]
    if strat not in _KNOWN_SAMPLERS:
        raise ValueError(
            f"LASO_ABLATIONS: unknown sampler '{strat}' in '{label}' "
            f"(known samplers: {sorted(_KNOWN_SAMPLERS)})")
    for p in parts[1:]:
        if p not in _KNOWN_ABLATION_PARTS:
            raise ValueError(
                f"LASO_ABLATIONS: unknown component '{p}' in '{label}' "
                f"(known components: {sorted(_KNOWN_ABLATION_PARTS)})")
    refine_enabled = "refine" in parts[1:]
    return (label, strat, refine_enabled)


def _laso_env_overrides():
    global BUDGET_PCTS, QUERY_BUDGET_PCTS, KNN_K_VALUES
    global ABLATION_CONFIGS, EMBEDDING_MODEL
    global EMBEDDING_MODELS_MULTIMODAL, EMBEDDING_MODELS_TEXT, results_dir
    global INCLUDE_NO_OPT, RANDOM_SEED

    def _num(tok):
        tok = str(tok).strip()
        return int(tok) if tok.lstrip("-").isdigit() else float(tok)

    v = os.environ.get("LASO_BUDGET_PCTS")
    if v:
        BUDGET_PCTS = [_num(t) for t in v.replace(",", " ").split()]

    v = os.environ.get("LASO_QUERY_BUDGET_JSON")
    if v:
        import json as _json
        QUERY_BUDGET_PCTS = {
            **QUERY_BUDGET_PCTS,
            **{k: [_num(x) for x in vals] for k, vals in _json.loads(v).items()},
        }

    v = os.environ.get("LASO_KNN_K")
    if v:
        KNN_K_VALUES = [t.strip() for t in v.replace(",", " ").split()]

    v = os.environ.get("LASO_ABLATIONS")
    if v:
        ABLATION_CONFIGS = [_parse_ablation_label(lbl)
                            for lbl in v.replace(",", " ").split()]
    if os.environ.get("PROBE_ONLY_NO_OPT") == "1":
        ABLATION_CONFIGS = []

    v = os.environ.get("LASO_EMBEDDING_MODEL")
    if v:
        EMBEDDING_MODEL = v
        EMBEDDING_MODELS_MULTIMODAL = [v]
        EMBEDDING_MODELS_TEXT = [v]
    v = os.environ.get("LASO_EMB_MULTIMODAL")
    if v:
        EMBEDDING_MODELS_MULTIMODAL = [t.strip() for t in v.split(",") if t.strip()]
    v = os.environ.get("LASO_EMB_TEXT")
    if v:
        EMBEDDING_MODELS_TEXT = [t.strip() for t in v.split(",") if t.strip()]

    v = os.environ.get("LASO_RESULTS_DIR")
    if v:
        results_dir = v if v.endswith("/") else v + "/"

    v = os.environ.get("LASO_RANDOM_SEED")
    if v:
        RANDOM_SEED = int(v)

    v = os.environ.get("LASO_INCLUDE_NO_OPT")
    if v is not None:
        INCLUDE_NO_OPT = v.lower() in {"1", "true", "yes", "on"}


_laso_env_overrides()


def _laso_resolve_queries(argv):
    """Resolve the query set from LASO_QUERIES (if set) else argv[1] (else all).

    Tokens may be exact query ids or group prefixes (ecomm/movie/mmqa/cars/
    animals/veri/flickr/roxford/company/match).
    """
    _GROUPS = ("ecomm", "movie", "mmqa", "cars", "animals",
               "veri", "flickr", "roxford", "company", "match")
    env = os.environ.get("LASO_QUERIES")
    tokens = (env.replace(",", " ").split() if env
              else ([argv[1]] if len(argv) > 1 else ["all"]))
    resolved = []
    for tok in tokens:
        if tok in QUERY_RUNNERS:
            resolved.append(tok)
        elif tok in _GROUPS:
            resolved += [q for q in QUERY_RUNNERS if q.startswith(tok)]
        elif tok == "all":
            resolved = list(QUERY_RUNNERS.keys())
        else:
            raise ValueError(f"LASO_QUERIES: unknown query/group '{tok}'")
    # de-duplicate preserving order
    seen = set()
    return [q for q in resolved if not (q in seen or seen.add(q))]


def get_spark():
    global spark
    if spark is not None:
        return spark
    spark = SparkSession.builder.appName("PROBE") \
        .master("local[*]") \
        .config("spark.driver.memory", "32g") \
        .config("spark.executor.memory", "32g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.default.parallelism", "1") \
        .getOrCreate()
    return spark


def resolve_prediction_csv(output_path):
    p = Path(output_path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        parts = sorted(p.glob("part-*.csv"))
        if parts:
            return str(parts[0])
    return None


def calculate_f1_score(predicted_file_path, gt_path):
    gt_df = pd.read_csv(gt_path)
    gt_ids = set(str(x).strip() for x in gt_df.iloc[:, 0].tolist())
    pred_df = pd.read_csv(predicted_file_path)
    pred_ids = set(str(x).strip() for x in pred_df.iloc[:, 0].tolist())
    tp = len(gt_ids & pred_ids)
    fp = len(pred_ids - gt_ids)
    fn = len(gt_ids - pred_ids)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"GT: {len(gt_ids)}, Pred: {len(pred_ids)}, TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1


def calculate_f1_score_topk(predicted_file_path, gt_path, k: int = 100):
    """Top-K F1 scorer: sort both sides by id column ascending, take the top
    K, then compute set-based precision/recall/F1 on those two truncated
    sets. Useful for evaluating only the highest-ranked (by id) portion of
    a large join output — e.g. ecomm_q7 where the full GT has thousands of
    pairs and we want a stable, small-sample F1 for fast iteration.

    Sort is lexicographic on the string form of the id column so it's
    deterministic regardless of whether ids look numeric. Pass k=0 to fall
    back to the full-set calculation.
    """
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(predicted_file_path)
    gt_ids_sorted = sorted(str(x).strip() for x in gt_df.iloc[:, 0].tolist())
    pred_ids_sorted = sorted(str(x).strip() for x in pred_df.iloc[:, 0].tolist())
    if k and k > 0:
        gt_top = set(gt_ids_sorted[:k])
        pred_top = set(pred_ids_sorted[:k])
    else:
        gt_top = set(gt_ids_sorted)
        pred_top = set(pred_ids_sorted)
    tp = len(gt_top & pred_top)
    fp = len(pred_top - gt_top)
    fn = len(gt_top - pred_top)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(
        f"[top{k}] GT(top{k}): {len(gt_top)}/{len(gt_ids_sorted)}, "
        f"Pred(top{k}): {len(pred_top)}/{len(pred_ids_sorted)}, "
        f"TP: {tp}, FP: {fp}, FN: {fn}"
    )
    print(f"[top{k}] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1


# --- MMQA hardcoded ground truths ---
MMQA_Q2A_GT = {
    (0, "117d500aaa630023c4038b8268b309c0.png"),
    (5, "117d500aaa630023c4038b8268b309c0.png"),
    (6, "117d500aaa630023c4038b8268b309c0.png"),
    (9, "117d500aaa630023c4038b8268b309c0.png"),
    (10, "117d500aaa630023c4038b8268b309c0.png"),
}

MMQA_Q7_GT = {
    ("British Airways", "cwncgabxti09zmf36t4phvaz1xzv10wu.png"),
    ("Delta Air Lines", "1sz0kf8wcmj0q8n3pu6mg61gl158vvz1.png"),
    ("Discover Airlines", "nqr1pjql5qs3dp8rz2a4zzq7rn0xyp6k.png"),
    ("Edelweiss Air", "6zpijbg5jv4jftpftpmla6xbizne28d9.png"),
    ("Virgin Atlantic", "antbx4oxst0z5o6pe2g1thrjr0dz073j.png"),
}


def _score_mmqa_q2a(pred_path, _gt_path_unused):
    df = pd.read_csv(pred_path)
    id_col = "ID" if "ID" in df.columns else ("left_ID" if "left_ID" in df.columns else None)
    img_col = ("image_filename" if "image_filename" in df.columns
               else "right_image_filename" if "right_image_filename" in df.columns
               else None)
    if id_col is None or img_col is None:
        predicted_set = set(tuple(x) for x in df.values)
    else:
        predicted_set = set()
        for _, row in df.iterrows():
            try:
                key = (int(row[id_col]), str(row[img_col]).strip())
            except (ValueError, TypeError):
                continue
            predicted_set.add(key)

    tp = len(MMQA_Q2A_GT & predicted_set)
    fp = len(predicted_set - MMQA_Q2A_GT)
    fn = len(MMQA_Q2A_GT - predicted_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"  [mmqa_q2a] GT ({len(MMQA_Q2A_GT)} rows):")
    for t in sorted(MMQA_Q2A_GT):
        print(f"    {t}")
    print(f"  [mmqa_q2a] Predicted ({len(predicted_set)} rows):")
    for t in sorted(predicted_set, key=lambda x: (x[0], str(x[1])))[:20]:
        print(f"    {t}")
    if len(predicted_set) > 20:
        print(f"    ... ({len(predicted_set) - 20} more)")
    print(f"  [mmqa_q2a] TP={tp} FP={fp} FN={fn}  P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    return precision, recall, f1


def _score_mmqa_q7(pred_path, _gt_path_unused):
    df = pd.read_csv(pred_path)
    airline_col = ("Airlines" if "Airlines" in df.columns
                   else "left_Airlines" if "left_Airlines" in df.columns
                   else None)
    img_col = ("image_filename" if "image_filename" in df.columns
               else "right_image_filename" if "right_image_filename" in df.columns
               else None)
    if airline_col is None or img_col is None:
        predicted_set = set(tuple(x) for x in df.values)
    else:
        predicted_set = set()
        for _, row in df.iterrows():
            airline = str(row[airline_col]).strip()
            filename = str(row[img_col]).strip()
            predicted_set.add((airline, filename))

    tp = len(MMQA_Q7_GT & predicted_set)
    fp = len(predicted_set - MMQA_Q7_GT)
    fn = len(MMQA_Q7_GT - predicted_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"  [mmqa_q7] GT ({len(MMQA_Q7_GT)} rows):")
    for t in sorted(MMQA_Q7_GT):
        print(f"    {t}")
    print(f"  [mmqa_q7] Predicted ({len(predicted_set)} rows):")
    for t in sorted(predicted_set)[:20]:
        print(f"    {t}")
    if len(predicted_set) > 20:
        print(f"    ... ({len(predicted_set) - 20} more)")
    print(f"  [mmqa_q7] TP={tp} FP={fp} FN={fn}  P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Cars scale-factor switch: LASO_CARS_SF (fallback CARS_SF, default 157376).
# Read at call time; data dir, file names and GT paths all follow it so data
# and GT can never come from different scale factors.
# ---------------------------------------------------------------------------
def _cars_sf():
    return (os.environ.get("LASO_CARS_SF") or os.environ.get("CARS_SF") or "157376").strip()


def _cars_data_file(kind):
    """kind in {car_data, image_car_data, text_complaints_data}."""
    sf = _cars_sf()
    return f"{DATA_ROOT}/SemBench/files/cars/data/sf_{sf}/{kind}_{sf}.csv"


def _cars_gt_path(name):
    """name in {Q3, Q4, Q8, Q4_rows}; 157376 -> gt_1536 (unchanged), else gt_<sf>."""
    sf = _cars_sf()
    gt_dir = "gt_1536" if sf == "157376" else f"gt_{sf}"
    fname = f"Q4_ground_truth_rows_sf{sf}.csv" if name == "Q4_rows" else f"{name}_{sf}.csv"
    if name == "Q4" and sf != "157376":
        # No scalar Q4_<sf>.csv outside gt_1536; the Q4 scorer only uses the rows file.
        fname = f"Q4_ground_truth_rows_sf{sf}.csv"
    return f"{DATA_ROOT}/SemBench/files/cars/raw_results/ground_truth/{gt_dir}/{fname}"


def _score_cars_q8(pred_path, gt_path):
    """Full-set F1 for the paper candidate population."""
    gt_path = _cars_gt_path("Q8")
    return calculate_f1_score(pred_path, gt_path)


def _score_cars_q3(pred_path, gt_path):
    """Full-set F1 for the paper candidate population."""
    return calculate_f1_score(pred_path, gt_path)


def _score_cars_q4(pred_path, _gt_path_unused):
    """Evaluate Q4 as set retrieval over car_id (precision / recall / F1).

    Replaces the previous scalar relative-error scoring. That metric compared
    ``2026 - mean(year)`` over the predicted cars against the single value in
    ``Q4.csv``, but ``average_age`` of any sizeable subset of the fleet
    converges to the fleet mean, so the score only reflected how many cars were
    returned, not which ones. Measured: a random set of 21,879 cars scored
    0.9826 while the true engine-complaint set scored 0.9791 -- random beat
    ground truth in five of six sample sizes.

    Ground truth is the per-row file (21,879 engine-complaint car_ids).
    Returns ``(precision, recall, f1, extra)`` to keep the 4-tuple contract.
    """
    gt_rows_path = _cars_gt_path("Q4_rows")

    def _normalize_id(val):
        s = str(val).strip()
        if s.lower() == "nan" or s == "":
            return None
        if re.fullmatch(r"\d+\.0", s):
            s = s.split(".")[0]
        return s if s.isdigit() else None

    def _error_return(msg="", predicted_count=0, gt_count=0):
        if msg:
            print(f"  [cars_q4] {msg}")
        return 0.0, 0.0, 0.0, {
            "predicted_count": int(predicted_count),
            "gt_count": int(gt_count),
            "true_positives": 0,
        }

    try:
        gt_df = pd.read_csv(gt_rows_path, dtype=str)
    except Exception as e:
        return _error_return(msg=f"gt read error: {e}")
    gt_col = "car_id" if "car_id" in gt_df.columns else gt_df.columns[0]
    gt_ids = {
        cid for cid in (_normalize_id(v) for v in gt_df[gt_col].dropna())
        if cid is not None
    }

    try:
        pred_df = pd.read_csv(
            pred_path, on_bad_lines="skip", engine="python", quoting=3, dtype=str,
        )
    except Exception as e:
        return _error_return(msg=f"pred read error: {e}", gt_count=len(gt_ids))

    pred_col = "car_id" if "car_id" in pred_df.columns else pred_df.columns[0]
    pred_ids = {
        pid for pid in (_normalize_id(v) for v in pred_df[pred_col].dropna())
        if pid is not None
    }
    if not pred_ids:
        return _error_return(msg="empty prediction", gt_count=len(gt_ids))

    tp = len(pred_ids & gt_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(gt_ids) if gt_ids else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    print(
        f"  [cars_q4] GT={len(gt_ids)} Pred={len(pred_ids)} TP={tp} "
        f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}"
    )
    return precision, recall, f1, {
        "predicted_count": len(pred_ids),
        "gt_count": len(gt_ids),
        "true_positives": tp,
    }


def _create_pair_tuple(row, columns):
    if len(columns) >= 3:
        movie_id = row[columns[0]]
        val1 = row[columns[1]]
        val2 = row[columns[2]]
        if pd.notna(val1) and pd.notna(val2) and pd.notna(movie_id):
            return (movie_id, tuple(sorted([val1, val2])))
    return None


def calculate_pair_f1_factory(limit=None):
    _debug_printed = [False]

    def _score(pred_path, gt_path):
        pred_df = pd.read_csv(pred_path)
        gt_df = pd.read_csv(gt_path)
        sys_cols = list(pred_df.columns)
        gt_cols = list(gt_df.columns)

        def _build_pair_set(df, columns, tag):
            raw_tuples = [_create_pair_tuple(r, columns) for _, r in df.iterrows()]
            valid = [t for t in raw_tuples if t is not None]
            invalid = len(raw_tuples) - len(valid)
            pairs = set(valid)
            dedup_removed = len(valid) - len(pairs)
            reflexive = sum(1 for _, p in pairs if p[0] == p[1])
            per_movie_counts: Dict[object, int] = {}
            for mid, _ in pairs:
                per_movie_counts[mid] = per_movie_counts.get(mid, 0) + 1
            print(
                f"  [{tag}] rows={len(df)}  invalid_rows={invalid}  "
                f"before_dedup={len(valid)}  unique_pairs={len(pairs)}  "
                f"duplicates_removed={dedup_removed}  reflexive_pairs={reflexive}  "
                f"distinct_movies={len(per_movie_counts)}"
            )
            return pairs

        if not _debug_printed[0]:
            print(f"  [GT DEBUG] path={gt_path}")
            print(f"  [GT DEBUG] shape={gt_df.shape}  columns={gt_cols}")
            print(f"  [GT DEBUG] dtypes={dict(gt_df.dtypes.astype(str))}")
            print(f"  [GT DEBUG] first 5 rows:")
            for i, row in gt_df.head(5).iterrows():
                print(f"    {i}: {dict(row)}")
            n_unique_raw = len(gt_df.drop_duplicates())
            print(f"  [GT DEBUG] unique rows (drop_duplicates): {n_unique_raw}")
            print(f"  [PRED DEBUG] shape={pred_df.shape}  columns={sys_cols}")
            print(f"  [PRED DEBUG] first 3 rows:")
            for i, row in pred_df.head(3).iterrows():
                print(f"    {i}: {dict(row)}")
            _debug_printed[0] = True

        sys_pairs = _build_pair_set(pred_df, sys_cols, "SYS")
        gt_pairs = _build_pair_set(gt_df, gt_cols, "GT ")
        correct = sys_pairs & gt_pairs
        missed = gt_pairs - sys_pairs
        false_positives = sys_pairs - gt_pairs

        precision = len(correct) / len(sys_pairs) if sys_pairs else 0.0
        if limit is not None:
            recall = (min(len(correct), limit) / min(limit, len(gt_pairs))) if gt_pairs else 0.0
            recall_mode = f"top-{limit}"
        else:
            recall = len(correct) / len(gt_pairs) if gt_pairs else 0.0
            recall_mode = "full"
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(
            f"  Pairs - Correct: {len(correct)}, Sys: {len(sys_pairs)}, GT: {len(gt_pairs)}, "
            f"Missed: {len(missed)}, FalsePos: {len(false_positives)} "
            f"(recall mode: {recall_mode})"
        )
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return precision, recall, f1
    return _score


# Stage weights of the refinement (cluster_proxy.predict_with_score_refinement):
#   Stage B "Correct" — residual diffusion on the kNN graph, weight diffusion_alpha
#   Stage C "Smooth"  — homogeneity propagation, weight propagation_beta
# Unset reproduces the previous hardcoded 0.8/0.8, so an ablation can disable a stage without
# touching any other behaviour.
REFINE_DIFFUSION_ALPHA = float(os.environ.get("LASO_REFINE_ALPHA", "0.8"))
REFINE_PROPAGATION_BETA = float(os.environ.get("LASO_REFINE_BETA", "0.8"))
# T1, T2: the iteration caps of the two phases. Both phases stop early once the largest score change
# stays under 1e-4 for 5 iterations, so these are upper bounds; the row records the caps asked for and
# the profile records the iterations that ran.
REFINE_T1 = int(os.environ.get("LASO_REFINE_T1", "50"))
REFINE_T2 = int(os.environ.get("LASO_REFINE_T2", "50"))
# theta_ref: threshold on refined scores that yields the refined labels h (paper Alg. 3).
REFINE_THETA = float(os.environ.get("LASO_REFINE_THETA", "0.5"))


def _overrides_for(query_tag: str, default_threshold: float) -> Dict[str, float]:
    return {
        "confidence_threshold": default_threshold,
        "diffusion_alpha": REFINE_DIFFUSION_ALPHA,
        "propagation_beta": REFINE_PROPAGATION_BETA,
    }


def _effective_cfg(
    query_tag: str, ablation_label: str, default_threshold: float
) -> Dict[str, float]:
    """Uniform defaults: confidence_threshold, diffusion_alpha=propagation_beta=0.8, lr_C=1.0."""
    cfg = _overrides_for(query_tag, default_threshold)
    cfg["lr_C"] = 1.0
    return cfg


# ---------------------------------------------------------------------------
# vLLM call / token accounting
#
# Two independent sources are aggregated per experiment:
#   1. Python-side `_TOKEN_COUNTER` in proxy_model.py — populated only on the
#      proxy path (labeling the sampled rows). Reports exact per-request
#      prompt/completion/total tokens from the OpenAI response.
#   2. vLLM Prometheus counters at /metrics — cumulative across the whole
#      server process. We diff pre/post snapshots; this is the ground truth
#      for BOTH the proxy path and the no-opt baseline (whose HTTP traffic
#      originates in the Java/Scala backend and bypasses the Python counter).
# ---------------------------------------------------------------------------
_VLLM_METRIC_KEYS = ("prompt_tokens", "generation_tokens", "requests")


def snapshot_vllm_metrics(url: str = VLLM_METRICS_URL) -> Dict[str, int]:
    """Scrape cumulative counters from vLLM's Prometheus /metrics endpoint.

    Returns zeros (and silently continues) if the endpoint is unreachable, so
    the experiment itself never fails because of missing instrumentation.
    """
    counters = {k: 0 for k in _VLLM_METRIC_KEYS}
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError):
        return counters

    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        brace = line.find("{")
        space = line.find(" ")
        if brace == -1 or (space != -1 and space < brace):
            name_end = space
        else:
            name_end = brace
        if name_end == -1:
            continue
        name = line[:name_end]
        try:
            value = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        if name == "vllm:prompt_tokens_total":
            counters["prompt_tokens"] += int(value)
        elif name == "vllm:generation_tokens_total":
            counters["generation_tokens"] += int(value)
        elif name == "vllm:request_success_total":
            # labeled by finished_reason; sum all label combinations
            counters["requests"] += int(value)
    return counters


_PRE_RUN_STATE: Dict[str, Dict[str, int]] = {}


def _begin_run_measurements(tag: str) -> None:
    """Zero the Python counter and snapshot vLLM metrics for this run tag."""
    reset_token_usage()
    _PRE_RUN_STATE[tag] = snapshot_vllm_metrics()


def _collect_run_measurements(tag: str) -> Dict[str, int]:
    """Return Python + server-side LLM call/token counts for *tag*."""
    pre = _PRE_RUN_STATE.pop(tag, None) or {k: 0 for k in _VLLM_METRIC_KEYS}
    post = snapshot_vllm_metrics()
    py = get_last_token_usage()
    vllm_requests = max(0, post["requests"] - pre["requests"])
    vllm_prompt = max(0, post["prompt_tokens"] - pre["prompt_tokens"])
    vllm_gen = max(0, post["generation_tokens"] - pre["generation_tokens"])
    return {
        "py_requests": int(py.get("llm_requests", 0)),
        "py_prompt_tokens": int(py.get("prompt_tokens", 0)),
        "py_completion_tokens": int(py.get("completion_tokens", 0)),
        "py_total_tokens": int(py.get("total_tokens", 0)),
        "py_label_true": int(py.get("label_true", 0)),
        "py_label_false": int(py.get("label_false", 0)),
        "py_label_unparsed": int(py.get("label_unparsed", 0)),
        "vllm_requests": vllm_requests,
        "vllm_prompt_tokens": vllm_prompt,
        "vllm_generation_tokens": vllm_gen,
        "vllm_total_tokens": vllm_prompt + vllm_gen,
    }


def run_experiment_join(query_tag, left_df, right_df, join_prompt, gt_path,
                       total_pairs, mode, budget_pct, feature_mode, knn_k_val,
                       ablation_label, sampling_strategy, refine_enabled, exp_idx,
                       embedding_model=EMBEDDING_MODEL):
    _lc_cfg = None
    if _decision_mode_for_ablation(ablation_label) == LEARNED_CALIBRATION_MODE:
        _lc_cfg = _learned_calibration_config(total_pairs, budget_pct)
        _router_on, _router_cap, _budget = False, 0, _lc_cfg.sample_budget
        print(f"  budget split (learned_calibration): total={_lc_cfg.total_budget} "
              f"sample={_lc_cfg.sample_budget} calib={_lc_cfg.calib_budget} "
              f"route={_lc_cfg.route_budget}")
    else:
        _router_on = _cascade_decision_enabled(_decision_mode_for_ablation(ablation_label))
        _budget_total, _budget, _router_cap = split_total_budget(
            total_pairs, budget_pct, ROUTER_CAP_FRAC if _router_on else 0.0)
        print(f"  budget split: total={_budget_total} sample={_budget} router_cap={_router_cap if _router_on else 'n/a'}")
    _knn = 0 if knn_k_val == "auto" else int(knn_k_val)
    cfg = _effective_cfg(query_tag, ablation_label, default_threshold=REFINE_THETA)
    decision_mode = _decision_mode_for_ablation(ablation_label)
    deferral_policy = _deferral_policy_for_ablation(ablation_label)
    _emb_slug = emb_short(embedding_model)
    tag = f"{ablation_label}_{query_tag}_budget{budget_pct}pct_{feature_mode}_k{knn_k_val}_emb-{_emb_slug}"
    print("\n" + "=" * 60)
    print(f"[Exp {exp_idx}] {ablation_label} JOIN {query_tag}: budget={budget_pct}%, k={knn_k_val}, emb={_emb_slug}")
    print(
        f"  cfg: threshold={cfg['confidence_threshold']}, "
        f"diffusion_alpha={cfg['diffusion_alpha']}, propagation_beta={cfg['propagation_beta']}, "
        f"lr_C={cfg['lr_C']}, deferral_policy={deferral_policy}"
    )
    print("=" * 60)

    _begin_run_measurements(tag)
    start = time.time()
    joined_df = left_df.sem_join(
        right=right_df, prompt=join_prompt, model=MODEL,
        url=os.environ.get("PROBE_ORACLE_URL", "http://localhost:8000/v1/chat/completions"),
        batch_size=LLM_BATCH_SIZE, verbose=True,
        embedding_model=embedding_model,
        max_concurrent_requests=128,
        proxy_sample=True, sample_size=_budget,
        sampling_method=sampling_strategy,
        cluster_proxy_mode=mode,
        confidence_threshold=cfg["confidence_threshold"],
        diffusion_alpha=cfg["diffusion_alpha"],
        propagation_beta=cfg["propagation_beta"],
        refine_t1=REFINE_T1,
        refine_t2=REFINE_T2,
        refine_enabled=refine_enabled,
        lr_feature_mode=feature_mode,
        pair_feature_mode=PAIR_FEATURE_MODE,
        lr_class_weight_override="balanced",
        random_seed=RANDOM_SEED,
        **_fixed_ce_runtime_options(RANDOM_SEED, sampling_method=sampling_strategy),
        defer_enabled=_cascade_decision_enabled(decision_mode),
        router_max_calls=_router_cap if _router_on else None,
        decision_mode=decision_mode,
        learned_calibration_config=_lc_cfg,
        lr_C=cfg["lr_C"],
        exclude_self_pairs=(query_tag in {"ecomm_q9", "movie_q5", "movie_q7", "match122"}),
        self_pair_id_columns=("left_reviewId", "right_reviewId") if query_tag in {"movie_q5", "movie_q7"} else None,
        join_sim_mode="pair",
        knn_k=_knn,
        viz_tag=query_tag,
    )
    return joined_df, tag, start


def run_experiment_filter(query_tag, df, filter_prompt, gt_path,
                          total_rows, mode, budget_pct, feature_mode, knn_k_val,
                          ablation_label, sampling_strategy, refine_enabled, exp_idx,
                          embedding_model=EMBEDDING_MODEL):
    _lc_cfg = None
    if _decision_mode_for_ablation(ablation_label) == LEARNED_CALIBRATION_MODE:
        _lc_cfg = _learned_calibration_config(total_rows, budget_pct)
        _router_on, _router_cap, _budget = False, 0, _lc_cfg.sample_budget
        print(f"  budget split (learned_calibration): total={_lc_cfg.total_budget} "
              f"sample={_lc_cfg.sample_budget} calib={_lc_cfg.calib_budget} "
              f"route={_lc_cfg.route_budget}")
    else:
        _router_on = _cascade_decision_enabled(_decision_mode_for_ablation(ablation_label))
        _budget_total, _budget, _router_cap = split_total_budget(
            total_rows, budget_pct, ROUTER_CAP_FRAC if _router_on else 0.0)
        print(f"  budget split: total={_budget_total} sample={_budget} router_cap={_router_cap if _router_on else 'n/a'}")
    _knn = 0 if knn_k_val == "auto" else int(knn_k_val)
    cfg = _effective_cfg(query_tag, ablation_label, default_threshold=REFINE_THETA)
    decision_mode = _decision_mode_for_ablation(ablation_label)
    deferral_policy = _deferral_policy_for_ablation(ablation_label)
    _emb_slug = emb_short(embedding_model)
    tag = f"{ablation_label}_{query_tag}_budget{budget_pct}pct_{feature_mode}_k{knn_k_val}_emb-{_emb_slug}"
    print("\n" + "=" * 60)
    print(f"[Exp {exp_idx}] {ablation_label} FILTER {query_tag}: budget={budget_pct}%, k={knn_k_val}, emb={_emb_slug}")
    print(
        f"  cfg: threshold={cfg['confidence_threshold']}, "
        f"diffusion_alpha={cfg['diffusion_alpha']}, propagation_beta={cfg['propagation_beta']}, "
        f"lr_C={cfg['lr_C']}, deferral_policy={deferral_policy}"
    )
    print("=" * 60)


    _begin_run_measurements(tag)
    start = time.time()
    result_df = df.sem_filter(
        prompt=filter_prompt, model=MODEL,
        url=os.environ.get("PROBE_ORACLE_URL", "http://localhost:8000/v1/chat/completions"),
        batch_size=LLM_BATCH_SIZE, verbose=True,
        embedding_model=embedding_model,
        max_concurrent_requests=128,
        proxy_sample=True, sample_size=_budget,
        sampling_method=sampling_strategy,
        cluster_proxy_mode=mode,
        confidence_threshold=cfg["confidence_threshold"],
        diffusion_alpha=cfg["diffusion_alpha"],
        propagation_beta=cfg["propagation_beta"],
        refine_t1=REFINE_T1,
        refine_t2=REFINE_T2,
        refine_enabled=refine_enabled,
        lr_feature_mode=feature_mode,
        lr_class_weight_override="balanced",
        random_seed=RANDOM_SEED,
        **_fixed_ce_runtime_options(RANDOM_SEED, sampling_method=sampling_strategy),
        defer_enabled=_cascade_decision_enabled(decision_mode),
        router_max_calls=_router_cap if _router_on else None,
        decision_mode=decision_mode,
        learned_calibration_config=_lc_cfg,
        lr_C=cfg["lr_C"],
        knn_k=_knn,
        viz_tag=query_tag,
    )
    return result_df, tag, start


def run_no_opt_join_experiment(query_tag, left_df, right_df, join_prompt, exp_idx):
    """No-optimization baseline: full LLM call on every pair (proxy_sample=False)."""
    tag = f"no_opt_{query_tag}"
    print("\n" + "=" * 60)
    print(f"[Exp {exp_idx}] NO-OPT JOIN {query_tag} (proxy_sample=False, full LLM)")
    print("=" * 60)
    _begin_run_measurements(tag)
    start = time.time()
    from proxy_model import proxy_join
    joined_df = proxy_join(
        left_df, right_df, join_prompt, model=MODEL,
        url=os.environ.get("PROBE_ORACLE_URL", "http://localhost:8000/v1/chat/completions"),
        embedding_model=EMBEDDING_MODEL, max_tokens=16, max_concurrent_requests=128,
        exclude_self_pairs=(query_tag in {"ecomm_q9", "movie_q5", "movie_q7", "match122"}),
        self_pair_id_columns=("left_reviewId", "right_reviewId") if query_tag in {"movie_q5", "movie_q7"} else None,
        no_proxy=True,
    )
    return joined_df, tag, start


def run_no_opt_filter_experiment(query_tag, df, filter_prompt, exp_idx):
    """No-optimization baseline: full LLM call on every row (proxy_sample=False)."""
    tag = f"no_opt_{query_tag}"
    print("\n" + "=" * 60)
    print(f"[Exp {exp_idx}] NO-OPT FILTER {query_tag} (proxy_sample=False, full LLM)")
    print("=" * 60)
    _begin_run_measurements(tag)
    start = time.time()
    from proxy_model import proxy_filter
    # max_tokens=100 is what PROBE's filter calls get through sem_filter's default.
    result_df = proxy_filter(
        df, filter_prompt, model=MODEL, url=os.environ.get("PROBE_ORACLE_URL", "http://localhost:8000/v1/chat/completions"),
        embedding_model=EMBEDDING_MODEL, max_tokens=100, max_concurrent_requests=128, no_proxy=True,
    )
    return result_df, tag, start


def collect_result(tag, result_df, id_expr, gt_path, start, mode, budget_pct, feature_mode, knn_k_val, query_tag, ablation_label="activeSampling+refine", scorer=None, embedding_model=EMBEDDING_MODEL):
    result_df.cache()
    out_df = id_expr(result_df)
    num_results = out_df.count()
    duration = time.time() - start
    usage = _collect_run_measurements(tag)
    profiling = get_last_profiling()
    cluster_prof = profiling.get("cluster_profiling", {}) or {}
    refine_prof = cluster_prof.get("refinement_profiling", {}) or {}

    paper_stages = cluster_prof.get("paper_stage_seconds")
    if paper_stages is not None:
        paper_stages = dict(paper_stages)
        paper_stages["embedding"] = float(profiling.get("embedding_time", 0.)) + float(profiling.get("ce_time", 0.)) + float(profiling.get("pair_feature_time", 0.))
        stage_total = sum(paper_stages.values())
        overhead = duration - stage_total
        if overhead < -max(.1, .01*duration):
            raise ValueError("Paper stage timers overlap measured end-to-end duration")
        Path(f"{results_dir}{tag}_paper_timing.json").write_text(json.dumps(dict(
            stages=paper_stages, measured_stage_sum_s=stage_total, measured_end_to_end_s=duration,
            client_and_spark_overhead_s=max(0.,overhead),
            scope="Seven exclusive pipeline categories; Spark/data/result overhead shown separately; no inferred stage latency"),indent=2)+"\n")

    # ---- 4-phase breakdown that sums to End-to-End duration ----
    #
    #   End-to-End = Embedding + Sampling + Training + Prediction
    #
    # 1. Embedding:  embed_texts / embed_multimodal_rows / pair feature building
    # 2. Sampling:   everything related to selecting + labeling samples —
    #                data prep (toPandas, message building), similarity
    #                computation, KNN graph build, non-LLM sampling logic
    #                (activeSampling argsort/peak finding), vLLM oracle calls
    #                (labeling the sampled rows/pairs), result assembly
    #                (pandas -> Spark), and grid-runner Spark overhead
    #                (cache/count). Defined as the residual so the four
    #                phases sum exactly to the E2E duration.
    # 3. Training:   lr.fit() on the labeled sample only.
    # 4. Prediction: lr.predict() on the full set + Bias Diffusion + Homogeneity Propagation
    #                (graph normalization + residual diffusion + prediction
    #                smoothing). For LR-only ablations, this is just
    #                lr.predict() (Bias Diffusion + Homogeneity Propagation times are 0).
    phase_embedding = float(profiling.get("embedding_time", 0.0))
    phase_training = float(refine_prof.get("lr_fit_time", 0.0))
    phase_prediction = (
        float(refine_prof.get("lr_predict_time", 0.0))
        + float(refine_prof.get("graph_norm_time", 0.0))
        + float(refine_prof.get("bias_diffusion_time", 0.0))
        + float(refine_prof.get("homogeneity_propagation_time", 0.0))
    )
    phase_sampling = max(0.0, duration - phase_embedding - phase_training - phase_prediction)

    # Detailed sub-timings (diagnostic, not part of the 4-bucket sum)
    sub_labeling = float(profiling.get("vllm_invocation_time", 0.0))  # vLLM calls
    sub_data_prep = float(profiling.get("data_prep_time", 0.0))
    sub_similarity = float(profiling.get("similarity_time", 0.0))
    sub_pair_msg = float(profiling.get("pair_message_time", 0.0))
    sub_knn_build = float(cluster_prof.get("knn_graph_build_time", 0.0))
    knn_detail = cluster_prof.get("knn_graph_build_detail", {}) or {}
    knn_cache_hit = bool(cluster_prof.get("knn_cache_hit", False))
    sub_result_assembly = float(profiling.get("result_assembly_time", 0.0))
    sub_sampling_core = max(
        0.0,
        float(cluster_prof.get("sampling_time", 0.0)) - sub_labeling,
    )

    print(f"  -> {num_results} rows in {duration:.2f}s (End-to-End)")
    print(
        f"  4-phase: Embedding={phase_embedding:.2f}s  "
        f"Sampling={phase_sampling:.2f}s  "
        f"Training={phase_training:.2f}s  "
        f"Prediction={phase_prediction:.2f}s  "
        f"(sum={phase_embedding + phase_sampling + phase_training + phase_prediction:.2f}s)"
    )
    print(
        f"  Sampling detail: data_prep={sub_data_prep:.2f}s  "
        f"similarity={sub_similarity:.2f}s  pair_msg={sub_pair_msg:.2f}s  "
        f"knn_build={sub_knn_build:.2f}s  sampling_core={sub_sampling_core:.2f}s  "
        f"labeling(vLLM)={sub_labeling:.2f}s  result_assembly={sub_result_assembly:.2f}s"
    )
    if knn_detail:
        _hit_tag = "HIT" if knn_cache_hit else "MISS"
        _backend = str(knn_detail.get("backend", "sklearn"))
        _fallback_from = knn_detail.get("backend_fallback_from")
        _backend_str = f"{_fallback_from}->{_backend}" if _fallback_from else _backend
        print(
            f"  KNN build detail [backend={_backend_str} cache={_hit_tag} "
            f"N={int(knn_detail.get('n_samples', 0))} "
            f"d={int(knn_detail.get('dim', 0))} k={int(knn_detail.get('knn_k', 0))} "
            f"nnz={int(knn_detail.get('nnz', 0))}]: "
            f"cache_lookup={float(knn_detail.get('cache_lookup_time', 0.0)):.2f}s  "
            f"nn_fit={float(knn_detail.get('nn_fit_time', 0.0)):.2f}s  "
            f"nn_query={float(knn_detail.get('nn_query_time', 0.0)):.2f}s  "
            f"csr_build={float(knn_detail.get('csr_build_time', 0.0)):.2f}s  "
            f"symmetrize={float(knn_detail.get('symmetrize_time', 0.0)):.2f}s  "
            f"cache_save={float(knn_detail.get('cache_save_time', 0.0)):.2f}s"
        )
    print(
        f"  Prediction detail: lr_predict={refine_prof.get('lr_predict_time', 0.0):.2f}s  "
        f"graph_norm={refine_prof.get('graph_norm_time', 0.0):.2f}s  "
        f"correct={refine_prof.get('bias_diffusion_time', 0.0):.2f}s "
        f"(iters={refine_prof.get('bias_diffusion_iters', '-')})  "
        f"smooth={refine_prof.get('homogeneity_propagation_time', 0.0):.2f}s "
        f"(iters={refine_prof.get('homogeneity_propagation_iters', '-')})"
    )
    print(
        f"  vLLM usage: requests(py/server)={usage['py_requests']}/{usage['vllm_requests']}  "
        f"prompt_tok={usage['vllm_prompt_tokens']}  "
        f"gen_tok={usage['vllm_generation_tokens']}  "
        f"total_tok={usage['vllm_total_tokens']}"
    )
    deferral_policy = _deferral_policy_for_ablation(ablation_label)
    policy_total = int(cluster_prof.get("n_total", 0) or 0)
    policy_defer_to_llm = int(cluster_prof.get("defer_verify_count", 0) or 0)
    policy_trust_proxy = max(policy_total - policy_defer_to_llm, 0)
    if policy_total > 0:
        print(
            f"  Deferral policy: {deferral_policy} | "
            f"trust_proxy={policy_trust_proxy}/{policy_total}  "
            f"rollback_to_llm={policy_defer_to_llm}/{policy_total}"
        )

    output_path = f"{results_dir}{tag}_result.csv"
    def json_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported profiling value: {type(value).__name__}")
    Path(f"{results_dir}{tag}_profiling.json").write_text(
        json.dumps(profiling, default=json_value, sort_keys=True) + "\n")
    out_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
    pred_csv = resolve_prediction_csv(output_path)

    precision = recall = f1 = 0.0
    scorer_extra: Dict[str, Any] = {}
    if pred_csv:
        score_fn = scorer if scorer is not None else calculate_f1_score
        raw = None
        if scorer is not None:
            raw = score_fn(pred_csv, gt_path)
        elif gt_path:
            raw = score_fn(pred_csv, gt_path)
        # Scorers are allowed to return either the legacy 3-tuple
        # `(precision, recall, f1)` or an extended 4-tuple
        # `(precision, recall, f1, extra_dict)`. The extra dict is used
        # by the SemBench-native aggregation scorers (animals_q1/q3/q10)
        # to surface absolute_error / relative_error / mape / predicted_count
        # etc. alongside the derived F1 so a failure mode like "F1=0
        # because the count overshot by 10x" is traceable from the CSV.
        if raw is not None:
            if len(raw) == 4:
                precision, recall, f1, maybe_extra = raw
                if isinstance(maybe_extra, dict):
                    scorer_extra = maybe_extra
            else:
                precision, recall, f1 = raw
    result_df.unpersist()
    print(f"  >> [{query_tag}] {tag}  F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  latency={duration:.1f}s")
    out: Dict[str, Any] = {
        "query": query_tag, "experiment": tag,
        "ablation": ablation_label, "mode": mode,
        "deferral_policy": deferral_policy,
        "budget_pct": budget_pct, "feature_mode": feature_mode,
        "knn_k": knn_k_val,
        "embedding_model": embedding_model,
        "embedding_short": emb_short(embedding_model),
        "num_results": num_results, "precision": precision,
        "recall": recall, "f1": f1,
        "duration_s": round(duration, 2),
        # Effective refinement stage weights, so a stage ablation is checkable from the results.
        "refine_diffusion_alpha": REFINE_DIFFUSION_ALPHA,
        # Sampler rounds actually used (clipped by the budget in learned-calibration mode) and the
        # explore rate, so a hyperparameter sweep can be read off the result row.
        "sampler": (cluster_prof.get("activeSampling_profiling") or {}).get("sampler", "not_applicable"),
        "sampling_config": json.dumps((cluster_prof.get("activeSampling_profiling") or {}).get("config", {}), sort_keys=True),
        "pilot_regime": ((cluster_prof.get("activeSampling_profiling") or {}).get("pilot_state") or {}).get("regime"),
        # SAS gate decision (paper protocol): branch taken, and the gate trace (n0, k0, L0) as JSON.
        "pilot_mode": ((cluster_prof.get("activeSampling_profiling") or {}).get("pilot_state") or {}).get("mode"),
        "pilot_gate": json.dumps(next((t for t in (cluster_prof.get("activeSampling_profiling") or {}).get("trace") or []
                                       if t.get("stage") == "gate"), None), default=float),
        "sampling_batches": json.dumps(__import__("collections").Counter(t.get("stage") for t in (cluster_prof.get("activeSampling_profiling") or {}).get("trace") or [])),
        "refine_propagation_beta": REFINE_PROPAGATION_BETA,
        # Representation study: which model supplies the sampling axis and the proxy/graph features.
        "repr_mode": os.environ.get("PROBE_REPR", "ce"),
        # The oracle LLM that answered (SEMBENCH_MODEL) and how completions prompts were templated.
        "oracle_model": MODEL,
        "chat_template_mode": "hf" if os.environ.get("SEMBENCH_CHAT_TEMPLATE") == "hf" else "qwen_chatml",
        "no_opt_backend": "python",
        # Failed oracle requests (e.g. client timeouts) on the budgeted path, and whether the pipeline abstained
        # (then only already-labelled rows are returned and no calibration / routing diagnostics exist).
        "oracle_failures": (profiling.get("oracle_budget") or {}).get("failures"),
        "result_status": cluster_prof.get("result_status", cluster_prof.get("lc_status")),
        "lc_budget_scope_cfg": LC_BUDGET_SCOPE,
        "ce_model": json.loads(os.environ.get("PROBE_CE_CONFIG", "{}") or "{}").get("model", "Qwen/Qwen3-VL-Reranker-2B"),
        # Iteration caps asked for; bias_diffusion_iters / homogeneity_propagation_iters below
        # record what convergence actually needed.
        "refine_t1": REFINE_T1,
        "refine_t2": REFINE_T2,
        # Rows labeled by the proxy oracle (0 for the no-opt baseline, which
        # evaluates every row/pair via Java). Kept for back-compat.
        "llm_calls": cluster_prof.get("n_labeled", "-"),
        # Actual HTTP-level vLLM traffic observed on the server (Prometheus
        # counter delta). Works for both proxy and no-opt baseline paths.
        "vllm_requests_actual": usage["vllm_requests"],
        "vllm_prompt_tokens_actual": usage["vllm_prompt_tokens"],
        "vllm_completion_tokens_actual": usage["vllm_generation_tokens"],
        "vllm_total_tokens_actual": usage["vllm_total_tokens"],
        # Python-side counter (OpenAI SDK response.usage). Only populated on
        # the proxy path; zero for no-opt baseline because Java makes the
        # HTTP calls directly.
        "llm_requests_python": usage["py_requests"],
        "llm_prompt_tokens_python": usage["py_prompt_tokens"],
        "llm_completion_tokens_python": usage["py_completion_tokens"],
        "llm_total_tokens_python": usage["py_total_tokens"],
        # Oracle answers containing "true" / "false" / neither (the last still count as label 0).
        "oracle_answers_true": usage["py_label_true"],
        "oracle_answers_false": usage["py_label_false"],
        "oracle_answers_unparsed": usage["py_label_unparsed"],
        "policy_total_rows": policy_total,
        "policy_trust_proxy_count": policy_trust_proxy,
        "policy_rollback_to_llm_count": policy_defer_to_llm,
        "defer_candidates_before_cap": cluster_prof.get("defer_candidates_before_cap", "-"),
        "defer_router_cap": cluster_prof.get("defer_router_cap", "-"),
        "budget_sample_size": cluster_prof.get("budget_sample_size", "-"),
        # Final delegation rate: oracle answers (sampling + calibration + routing, or every candidate for
        # the no-proxy baseline) divided by the number of candidates.
        "delegation_rate_legacy": (
            (usage["py_label_true"] + usage["py_label_false"] + usage["py_label_unparsed"]) / policy_total
            if policy_total > 0 else None
        ),
        "policy_rollback_to_llm_rate": (
            policy_defer_to_llm / policy_total if policy_total > 0 else 0.0
        ),
        # 4-phase breakdown (seconds; sums to duration_s)
        "phase_embedding_s": round(phase_embedding, 3),
        # fixed CE scoring time (cold unless PROBE_CE_CACHE already holds the scores) and the baseline axis
        "sub_ce_s": round(float(profiling.get("ce_time", 0.0) or 0.0), 3),
        "baseline_axis": profiling.get("baseline_axis", "default"),
        "phase_sampling_s": round(phase_sampling, 3),
        "phase_training_s": round(phase_training, 3),
        "phase_prediction_s": round(phase_prediction, 3),
        # Sampling sub-phases (diagnostic; labeling = vLLM is part of Sampling)
        "sub_data_prep_s": round(sub_data_prep, 3),
        "sub_similarity_s": round(sub_similarity, 3),
        "sub_pair_msg_s": round(sub_pair_msg, 3),
        "sub_knn_build_s": round(sub_knn_build, 3),
        "sub_sampling_core_s": round(sub_sampling_core, 3),
        "sub_labeling_s": round(sub_labeling, 3),
        "sub_result_assembly_s": round(sub_result_assembly, 3),
        # Prediction sub-phases (diagnostic)
        "sub_lr_predict_s": round(refine_prof.get("lr_predict_time", 0.0), 3),
        "sub_graph_norm_s": round(refine_prof.get("graph_norm_time", 0.0), 3),
        "sub_correct_s": round(refine_prof.get("bias_diffusion_time", 0.0), 3),
        "sub_smooth_s": round(refine_prof.get("homogeneity_propagation_time", 0.0), 3),
        "bias_diffusion_iters": refine_prof.get("bias_diffusion_iters", "-"),
        "homogeneity_propagation_iters": refine_prof.get("homogeneity_propagation_iters", "-"),
    }

    # ------------------------------------------------------------------
    # defer dual-threshold profiling. Populated when
    # cluster_proxy_pipeline ran the defer block (sampling_method ==
    # 'activeSampling' + refine_enabled + defer_enabled). Used to plot the
    # F1-vs-delegation Pareto curve.
    # ------------------------------------------------------------------
    out["defer_enabled"] = bool(cluster_prof.get("defer_enabled", False))
    out["defer_cold_start"] = cluster_prof.get("defer_cold_start")
    out["defer_calibration_source"] = cluster_prof.get(
        "defer_calibration_source"
    )
    out["defer_n_cal"] = cluster_prof.get("defer_n_cal")
    out["defer_n_pos_cal"] = cluster_prof.get("defer_n_pos_cal")
    out["defer_n_neg_cal"] = cluster_prof.get("defer_n_neg_cal")
    out["defer_tau_low"] = cluster_prof.get("defer_tau_low")
    out["defer_tau_high"] = cluster_prof.get("defer_tau_high")
    out["defer_alpha"] = cluster_prof.get("defer_alpha")
    out["defer_beta"] = cluster_prof.get("defer_beta")
    out["defer_lam"] = cluster_prof.get("defer_lam")
    out["defer_n_knots"] = cluster_prof.get("defer_n_knots")
    out["defer_decision_mode"] = cluster_prof.get("defer_decision_mode")
    out["defer_accept_count"] = cluster_prof.get("defer_accept_count")
    out["defer_uncertain_count"] = cluster_prof.get("defer_uncertain_count")
    out["defer_reject_count"] = cluster_prof.get("defer_reject_count")
    out["defer_delegation_rate"] = cluster_prof.get("defer_delegation_rate")
    out["defer_expected_f_beta"] = cluster_prof.get("defer_expected_f_beta")
    out["defer_baseline_f_beta"] = cluster_prof.get("defer_baseline_f_beta")
    out["defer_objective_at_opt"] = cluster_prof.get("defer_objective_at_opt")
    out["defer_g_mean_avg"] = cluster_prof.get("defer_g_mean_avg")
    out["defer_g_se_avg"] = cluster_prof.get("defer_g_se_avg")
    out["defer_routing"] = cluster_prof.get("defer_routing")
    out["defer_verify_uncertain"] = cluster_prof.get("defer_verify_uncertain")
    out["defer_verify_count"] = cluster_prof.get("defer_verify_count")
    out["defer_verify_positive"] = cluster_prof.get("defer_verify_positive")
    out["defer_verify_time"] = cluster_prof.get("defer_verify_time")
    out["defer_fit_time"] = cluster_prof.get("defer_fit_time")
    out["defer_optimize_time"] = cluster_prof.get("defer_optimize_time")
    out["defer_total_time"] = cluster_prof.get("defer_total_time")
    if cluster_prof.get("defer_decision_mode") == LEARNED_CALIBRATION_MODE:
        for _key in LEARNED_CALIBRATION_RESULT_KEYS:
            out[_key] = cluster_prof.get(_key)
    for _key in ("bargain_mode", "bargain_target", "bargain_delta", "bargain_time_s", "bargain_oracle_rows",
                 "bargain_new_calls", "bargain_unresolved", "bargain_proxy_rows", "bargain_budget"):
        if _key in cluster_prof:
            out[_key] = cluster_prof.get(_key)
    # DASP decision-source breakdown: how many of the final positive /
    # negative predictions were decided by the proxy vs by the LLM
    # (split into B_proxy / B_defer / verify subsystems).
    for _key in (
        "dasp_pred_pos_total", "dasp_pred_pos_by_proxy",
        "dasp_pred_pos_by_b_proxy", "dasp_pred_pos_by_b_defer",
        "dasp_pred_pos_by_verify",
        "dasp_pred_neg_total", "dasp_pred_neg_by_proxy",
        "dasp_pred_neg_by_b_proxy", "dasp_pred_neg_by_b_defer",
        "dasp_pred_neg_by_verify",
    ):
        out[_key] = cluster_prof.get(_key)
    # Merge SemBench-native aggregation metrics (absolute_error /
    # relative_error / mape / predicted_count / gt_count / etc.) when
    # the scorer returned a 4-tuple. Prefixed with `sembench_` so they
    # don't collide with row-level precision/recall/f1 semantics used
    # by the classification scorers. Values are JSON-stringified when
    # they're containers so pandas.to_csv writes a single cell.
    import json as _json
    for _k, _v in scorer_extra.items():
        col = f"sembench_{_k}"
        if isinstance(_v, (list, tuple, set, dict)):
            try:
                out[col] = _json.dumps(
                    sorted(_v) if isinstance(_v, set) else _v,
                    default=str,
                )
            except (TypeError, ValueError):
                out[col] = str(_v)
        else:
            out[col] = _v
    oracle = profiling.get("oracle_budget") or {}
    if os.environ.get('PROBE_PAPER_JOB') and pred_csv:
        from experiments.evaluation_export import export_evaluation
        reference = MMQA_Q2A_GT if query_tag == 'mmqa_q2a' else MMQA_Q7_GT if query_tag == 'mmqa_q7' else None
        effective_gt = _cars_gt_path('Q4_rows') if query_tag == 'cars_q4' else gt_path
        annotations = f"{DATA_ROOT}/SemBench/files/animals/data/sf_{os.environ.get('LASO_ANIMALS_SF','1600')}/image_data.csv" if query_tag == 'animals_q1' else None
        export_evaluation(query_tag, id_expr, pred_csv, effective_gt, f1,
                          mmqa_truth=reference, annotation_source=annotations)
        if reference is not None:
            # The scorer uses an embedded reference. Archive exactly those
            # bytes rather than reporting an unrelated or absent GT file.
            gt_path = str(Path(results_dir)/f'{query_tag}_reference.json')
            Path(gt_path).write_text(json.dumps(sorted(reference))+'\n')
        elif query_tag == 'cars_q4':
            gt_path = effective_gt
    out.update(oracle_unique_labels=oracle.get("unique_labels"),
               oracle_physical_attempts=oracle.get("candidate_attempts"),
               oracle_http_requests=oracle.get("http_requests"),
               oracle_failures=oracle.get("failures"),
               delegation_rate=(oracle.get("unique_labels", 0) / policy_total
                                if oracle and policy_total else out["delegation_rate_legacy"]),
               ce_provenance=json.dumps(profiling.get("ce_provenance", {}), sort_keys=True),
               gt_source_sha256=(__import__("hashlib").sha256(Path(gt_path).read_bytes()).hexdigest()
                                 if gt_path and Path(gt_path).is_file() else None))
    return out


def _phase_breakdown_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean 4-phase timing breakdown grouped by ablation."""
    cols = ["phase_embedding_s", "phase_sampling_s", "phase_training_s",
            "phase_prediction_s", "duration_s"]
    existing = [c for c in cols if c in df.columns]
    if not existing or "ablation" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby("ablation")[existing].mean().reset_index()
    return agg.round(3)


def maybe_append_no_opt_join(query_tag, left_df, right_df, prompt, exp_idx, id_expr,
                             gt_path=None, scorer=None, embedding_model=None):
    """Append the no-proxy baseline row when LASO_INCLUDE_NO_OPT is enabled."""
    if not INCLUDE_NO_OPT:
        return exp_idx
    emb_model = embedding_model or EMBEDDING_MODEL
    no_opt_df, no_opt_tag, no_opt_start = run_no_opt_join_experiment(
        query_tag, left_df, right_df, prompt, exp_idx,
    )
    no_opt_r = collect_result(
        no_opt_tag, no_opt_df, id_expr, gt_path, no_opt_start,
        "no_opt", 100, "n/a", "n/a", query_tag,
        ablation_label="no_opt", scorer=scorer, embedding_model=emb_model,
    )
    all_results.append(no_opt_r)
    return exp_idx + 1


def maybe_append_no_opt_filter(query_tag, df, prompt, exp_idx, id_expr,
                               gt_path=None, scorer=None, embedding_model=None):
    """Append the no-proxy baseline row when LASO_INCLUDE_NO_OPT is enabled."""
    if not INCLUDE_NO_OPT:
        return exp_idx
    emb_model = embedding_model or EMBEDDING_MODEL
    no_opt_df, no_opt_tag, no_opt_start = run_no_opt_filter_experiment(
        query_tag, df, prompt, exp_idx,
    )
    no_opt_r = collect_result(
        no_opt_tag, no_opt_df, id_expr, gt_path, no_opt_start,
        "no_opt", 100, "n/a", "n/a", query_tag,
        ablation_label="no_opt", scorer=scorer, embedding_model=emb_model,
    )
    all_results.append(no_opt_r)
    return exp_idx + 1


def save_query_results(query_tag, rows):
    if not rows:
        print(f"[{query_tag}] no rows to save")
        return
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = f"{results_dir}{query_tag}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[{query_tag}] {len(rows)} experiments saved -> {csv_path}")

    agg = pd.DataFrame()
    if "ablation" in df.columns and "f1" in df.columns:
        summary_path = f"{results_dir}{query_tag}_summary.csv"
        agg_specs = {
            "f1_mean": ("f1", "mean"),
            "f1_max": ("f1", "max"),
            "precision_mean": ("precision", "mean"),
            "recall_mean": ("recall", "mean"),
            "latency_mean": ("duration_s", "mean"),
            "n_experiments": ("f1", "size"),
        }
        # Aggregate LLM-call / token usage when the instrumentation columns
        # are present (populated by _collect_run_measurements).
        usage_cols = [
            ("vllm_requests_actual", "vllm_requests_mean"),
            ("vllm_prompt_tokens_actual", "vllm_prompt_tokens_mean"),
            ("vllm_completion_tokens_actual", "vllm_completion_tokens_mean"),
            ("vllm_total_tokens_actual", "vllm_total_tokens_mean"),
            ("llm_requests_python", "llm_requests_python_mean"),
            ("llm_total_tokens_python", "llm_total_tokens_python_mean"),
        ]
        for src, dst in usage_cols:
            if src in df.columns:
                agg_specs[dst] = (src, "mean")
        agg = df.groupby("ablation").agg(**agg_specs).reset_index().sort_values(
            "f1_mean", ascending=False
        )
        agg.to_csv(summary_path, index=False)
        print(f"[{query_tag}] summary saved -> {summary_path}")
        print(f"[{query_tag}] per-ablation F1 summary:")
        print(agg.round(4).to_string(index=False))

    phase_agg = _phase_breakdown_summary(df)
    if not phase_agg.empty:
        print(f"[{query_tag}] per-ablation 4-phase timing (mean seconds):")
        print(phase_agg.to_string(index=False))

# ---------------------------------------------------------------------------
# Ecomm Q7: sem_join (product descriptions same category + brand)
# ---------------------------------------------------------------------------
def run_ecomm_q7():
    sp = get_spark()
    # scale factor 由 LASO_ECOMM_SF 选择(250 或 4000),GT 目录随之切换,
    # 避免在两次运行之间反复编辑本文件。
    _sf = os.environ.get("LASO_ECOMM_SF", "250")
    data_dir = f"{DATA_ROOT}/SemBench/files/ecomm/data/sf_{_sf}"
    df = sp.read.parquet(os.path.join(data_dir, "styles_details.parquet"))
    df = df.filter(col("price") <= 500)
    df = df.withColumn("description_text", col("productDescriptors.description.value"))
    complex_fields = [f.name for f in df.schema.fields if isinstance(f.dataType, (MapType, StructType, ArrayType))]
    df = df.drop(*complex_fields)

    left_df = df.select(col("id").alias("left_id"), col("productDisplayName").alias("left_productDisplayName"),
                        col("description_text").alias("left_description_text"))
    right_df = df.select(col("id").alias("right_id"), col("productDisplayName").alias("right_productDisplayName"),
                         col("description_text").alias("right_description_text"))

    prompt = """You will be given two product descriptions. Do both product descriptions describe
products of the same category from the same brand, e.g., both are t-shirts from Adidas?

The first product description is:
{left_productDisplayName} - {left_description_text}

The second product description is:
{right_productDisplayName} - {right_description_text}"""

    _gtdir = "ground_truth" if _sf == "250" else "ground_truth_sf4000"
    gt_path = f"{DATA_ROOT}/SemBench/files/ecomm/raw_results/{_gtdir}/Q7.csv"
    total_pairs = left_df.count() * right_df.count()
    print(f"[ecomm_q7] total_pairs={total_pairs}")

    def id_expr(jdf):
        return jdf.withColumn("id", concat_ws("-", col("left_id").cast("string"), col("right_id").cast("string"))).select("id")

    # Top-K F1: sort prediction ids and GT ids (lexicographic on the id
    # string), take the top K of each, compute P/R/F1 on those. Override K
    # via ECOMM_Q7_TOPK env var (default 100; set to 0 for full-set F1).
    ECOMM_Q7_TOPK = int(os.environ.get("ECOMM_Q7_TOPK", "0"))
    ecomm_q7_scorer = lambda p, g: calculate_f1_score_topk(p, g, k=ECOMM_Q7_TOPK)
    print(f"[ecomm_q7] scoring mode: top-{ECOMM_Q7_TOPK} F1" if ECOMM_Q7_TOPK > 0 else "[ecomm_q7] scoring mode: full-set F1")

    exp_idx = 1
    for emb_model in _iter_emb_models("ecomm_q7"):
        with _emb_scope("ecomm_q7", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("ecomm_q7"):
                    for k_val in KNN_K_VALUES:
                        joined_df, tag, start = run_experiment_join("ecomm_q7", left_df, right_df, prompt, gt_path, total_pairs, "propagation", bp, "embedding", k_val, abl_label, abl_strategy, abl_cs, exp_idx, embedding_model=emb_model)
                        r = collect_result(tag, joined_df, id_expr, gt_path, start, "propagation", bp, "embedding", k_val, "ecomm_q7", ablation_label=abl_label, embedding_model=emb_model, scorer=ecomm_q7_scorer)
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_join("ecomm_q7", left_df, right_df, prompt, exp_idx, id_expr, gt_path, scorer=ecomm_q7_scorer)


# ---------------------------------------------------------------------------
# Ecomm Q8: sem_join (image-text matching)
# ---------------------------------------------------------------------------
def run_ecomm_q8():
    sp = get_spark()
    # scale factor 由 LASO_ECOMM_SF 选择(250 或 4000),GT 目录随之切换,
    # 避免在两次运行之间反复编辑本文件。
    _sf = os.environ.get("LASO_ECOMM_SF", "250")
    data_dir = f"{DATA_ROOT}/SemBench/files/ecomm/data/sf_{_sf}"
    styles = sp.read.parquet(os.path.join(data_dir, "styles_details.parquet"))
    image_map = sp.read.parquet(os.path.join(data_dir, "image_mapping.parquet"))
    styles = styles.withColumn("description_text", col("productDescriptors.description.value"))
    complex_fields = [f.name for f in styles.schema.fields if isinstance(f.dataType, (MapType, StructType, ArrayType))]
    styles = styles.drop(*complex_fields)
    styles = styles.filter(length(col("description_text")) >= 3000)

    left_df = styles.select(
        col("id").alias("style_id"), col("productDisplayName").alias("left_productDisplayName"),
        col("description_text").alias("left_description_text"),
    )
    right_df = image_map.select(
        col("id").alias("image_id"),
        concat(lit(data_dir + "/images/"), col("filename")).alias("right_image_filepath"),
    )

    prompt = """Does the image fit the product description?
Image: {image:right_image_filepath}
Product name: {left_productDisplayName}
Product description: {left_description_text}"""

    _gtdir = "ground_truth" if _sf == "250" else "ground_truth_sf4000"
    gt_path = f"{DATA_ROOT}/SemBench/files/ecomm/raw_results/{_gtdir}/Q8.csv"
    total_pairs = left_df.count() * right_df.count()
    print(f"[ecomm_q8] total_pairs={total_pairs}")

    def id_expr(jdf):
        return jdf.withColumn("id", concat_ws("-", col("style_id").cast("string"), col("image_id").cast("string"))).select("id")

    exp_idx = 1
    for emb_model in _iter_emb_models("ecomm_q8"):
        with _emb_scope("ecomm_q8", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("ecomm_q8"):
                    for k_val in KNN_K_VALUES:
                        joined_df, tag, start = run_experiment_join("ecomm_q8", left_df, right_df, prompt, gt_path, total_pairs, "propagation", bp, "embedding", k_val, abl_label, abl_strategy, abl_cs, exp_idx, embedding_model=emb_model)
                        r = collect_result(tag, joined_df, id_expr, gt_path, start, "propagation", bp, "embedding", k_val, "ecomm_q8", ablation_label=abl_label, embedding_model=emb_model)
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_join("ecomm_q8", left_df, right_df, prompt, exp_idx, id_expr, gt_path)


# ---------------------------------------------------------------------------
# Ecomm Q9: sem_join (image-image same category + color self-join)
# ---------------------------------------------------------------------------
def run_ecomm_q9():
    sp = get_spark()
    # scale factor 由 LASO_ECOMM_SF 选择,GT 目录随之切换(ecomm 已统一为 sf_250)
    _sf = os.environ.get("LASO_ECOMM_SF", "250")
    data_dir = f"{DATA_ROOT}/SemBench/files/ecomm/data/sf_{_sf}"
    styles = sp.read.parquet(os.path.join(data_dir, "styles_details.parquet"))
    image_map = sp.read.parquet(os.path.join(data_dir, "image_mapping.parquet"))

    valid_colors = ["Black", "Blue", "Red", "White", "Orange", "Green"]
    # Match sembench_q9_cluster_proxy.py exactly: keep only single-colour
    # items (colour1 == "" AND colour2 == "") so "same category + same
    # colour" is well-defined. Without this filter, ~12x more styles slip
    # through and no_opt predicts 280 pairs (mostly multi-colour FPs) in
    # ~740s instead of 24 pairs in ~55s.
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
    image_map = image_map.withColumn("image_filepath", concat(lit(data_dir + "/images/"), col("filename")))

    left_df = image_map.select(col("id").alias("left_id"), col("image_filepath").alias("left_image_filepath"))
    right_df = image_map.select(col("id").alias("right_id"), col("image_filepath").alias("right_image_filepath"))

    prompt = """Determine whether both images display objects of the same category
(e.g., both are shoes, both are bags, etc.) and whether these objects
share the same dominant surface color. Disregard any logos, text, or
printed graphics on the objects. There might be other objects in the
images. Only focus on the main object. Base your comparison solely on
object type and overall surface color.
Left image: {image:left_image_filepath}
Right image: {image:right_image_filepath}"""

    _gtdir = "ground_truth" if _sf == "250" else "ground_truth_sf4000"
    gt_path = f"{DATA_ROOT}/SemBench/files/ecomm/raw_results/{_gtdir}/Q9.csv"
    # Budget the eligible physical candidates, matching exclude_self_pairs.
    # Repeated IDs exclude all equal-ID pairs, not only the positional diagonal.
    multiplicities = image_map.groupBy("id").count().collect()
    total_pairs = sum(int(r["count"]) for r in multiplicities) ** 2 - sum(int(r["count"]) ** 2 for r in multiplicities)
    print(f"[ecomm_q9] eligible total_pairs={total_pairs}")

    def id_expr(jdf):
        return jdf.filter(col("left_id") != col("right_id")).withColumn(
            "id", concat_ws("-", col("left_id").cast("string"), col("right_id").cast("string"))).select("id")

    exp_idx = 1
    for emb_model in _iter_emb_models("ecomm_q9"):
        with _emb_scope("ecomm_q9", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("ecomm_q9"):
                    for k_val in KNN_K_VALUES:
                        joined_df, tag, start = run_experiment_join("ecomm_q9", left_df, right_df, prompt, gt_path, total_pairs, "propagation", bp, "embedding", k_val, abl_label, abl_strategy, abl_cs, exp_idx, embedding_model=emb_model)
                        r = collect_result(tag, joined_df, id_expr, gt_path, start, "propagation", bp, "embedding", k_val, "ecomm_q9", ablation_label=abl_label, embedding_model=emb_model)
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_join("ecomm_q9", left_df, right_df, prompt, exp_idx, id_expr, gt_path)


# ---------------------------------------------------------------------------
# Ecomm Q13: sem_filter (multimodal filter — running shirt)
# ---------------------------------------------------------------------------
def run_ecomm_q13():
    sp = get_spark()
    # scale factor 由 LASO_ECOMM_SF 选择,GT 目录随之切换(ecomm 已统一为 sf_250)
    _sf = os.environ.get("LASO_ECOMM_SF", "250")
    data_dir = f"{DATA_ROOT}/SemBench/files/ecomm/data/sf_{_sf}"
    styles = sp.read.parquet(os.path.join(data_dir, "styles_details.parquet"))
    image_map = sp.read.parquet(os.path.join(data_dir, "image_mapping.parquet"))
    styles = styles.withColumn("description_text", col("productDescriptors.description.value"))
    complex_fields = [f.name for f in styles.schema.fields if isinstance(f.dataType, (MapType, StructType, ArrayType))]
    styles = styles.drop(*complex_fields)

    styles = styles.withColumn("id_str", col("id").cast("string"))
    image_map = image_map.withColumn("image_id_str", col("id").cast("string"))
    merged = styles.join(
        image_map.select(col("image_id_str"), col("filename")),
        styles["id_str"] == image_map["image_id_str"], "inner",
    ).drop("image_id_str")
    merged = merged.withColumn("image_filepath", concat(lit(data_dir + "/images/"), col("filename")))

    prompt = """You will receive a description of what a customer is looking for together with an image and a textual description of the product.
Determine if they both match.

I am looking for a running shirt for men with a round neck and short sleeves,
preferably in blue or black, but not bright colors like white.
Also definitely not green.
It should be suitable for outdoor running in warm weather.
If the t-shirt is not green, it should at least feature a striped design.

The product has the following image {image:image_filepath} and textual description {productDisplayName} {description_text}"""

    _gtdir = "ground_truth" if _sf == "250" else "ground_truth_sf4000"
    gt_path = f"{DATA_ROOT}/SemBench/files/ecomm/raw_results/{_gtdir}/Q13.csv"
    total_rows = merged.count()
    print(f"[ecomm_q13] total_rows={total_rows}")

    def id_expr(rdf):
        return rdf.select("id")

    exp_idx = 1
    for emb_model in _iter_emb_models("ecomm_q13"):
        with _emb_scope("ecomm_q13", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("ecomm_q13"):
                    for k_val in KNN_K_VALUES:
                        result_df, tag, start = run_experiment_filter("ecomm_q13", merged, prompt, gt_path, total_rows, "propagation", bp, "embedding", k_val, abl_label, abl_strategy, abl_cs, exp_idx, embedding_model=emb_model)
                        r = collect_result(tag, result_df, id_expr, gt_path, start, "propagation", bp, "embedding", k_val, "ecomm_q13", ablation_label=abl_label, embedding_model=emb_model)
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_filter("ecomm_q13", merged, prompt, exp_idx, id_expr, gt_path)


# ---------------------------------------------------------------------------
# Movie Q5: sem_join (same sentiment reviews)
# ---------------------------------------------------------------------------
def run_movie_q5():
    sp = get_spark()
    csv_path = f"{DATA_ROOT}/SemBench/files/movie/data/sf_1000/Reviews.csv"
    reviews = sp.read.option("header", "true").csv(csv_path)
    reviews = reviews.filter(
        (col("id") == "ant_man_and_the_wasp_quantumania") & col("reviewText").isNotNull()
    )
    left_df = reviews.select(
        col("id").alias("left_id"), col("reviewId").alias("left_reviewId"),
        col("reviewText").alias("left_reviewText"),
    )
    right_df = reviews.select(
        col("id").alias("right_id"), col("reviewId").alias("right_reviewId"),
        col("reviewText").alias("right_reviewText"),
    )

    prompt = """These two movie reviews express the same sentiment - either both are positive or both are negative.
Review 1: "{left_reviewText}"
Review 2: "{right_reviewText}" """

    gt_path = f"{DATA_ROOT}/SemBench/files/movie/raw_results/ground_truth/Q5.csv"
    # Source rows can repeat review IDs. Charge only eligible physical pairs,
    # excluding every pair with equal review IDs, not just the positional diagonal.
    multiplicities = reviews.groupBy("reviewId").count().collect()
    total_pairs = sum(int(r["count"]) for r in multiplicities) ** 2 - sum(int(r["count"]) ** 2 for r in multiplicities)
    print(f"[movie_q5] eligible total_pairs={total_pairs}")

    def id_expr(jdf):
        # Drop trivial self-pairs (A,A); keep ALL other predicted pairs and
        # evaluate full precision/recall against the entire GT set
        # (no top-10 cap on either predictions or recall denominator).
        return jdf.filter(col("left_reviewId") != col("right_reviewId")).select(
            col("left_id").alias("id"),
            col("left_reviewId").alias("reviewId"),
            col("right_reviewId").alias("reviewId2"),
        )

    scorer = calculate_pair_f1_factory(limit=None)
    exp_idx = 1
    for emb_model in _iter_emb_models("movie_q5"):
        with _emb_scope("movie_q5", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("movie_q5"):
                    for k_val in KNN_K_VALUES:
                        joined_df, tag, start = run_experiment_join("movie_q5", left_df, right_df, prompt, gt_path, total_pairs, "propagation", bp, "embedding", k_val, abl_label, abl_strategy, abl_cs, exp_idx, embedding_model=emb_model)
                        r = collect_result(tag, joined_df, id_expr, gt_path, start, "propagation", bp, "embedding", k_val, "movie_q5", ablation_label=abl_label, scorer=scorer, embedding_model=emb_model)
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_join("movie_q5", left_df, right_df, prompt, exp_idx, id_expr, gt_path, scorer=scorer)


# ---------------------------------------------------------------------------
# Movie Q7: sem_join (opposite sentiment reviews)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MMQA Q2a: sem_join (racetrack + image logo)
# ---------------------------------------------------------------------------
def run_mmqa_q2a():
    sp = get_spark()
    # scale factor 由 LASO_MMQA_SF 选择(默认 800 = PROBE 已跑的配置)
    _msf = os.environ.get("LASO_MMQA_SF", "800")
    data_dir = f"{DATA_ROOT}/SemBench/files/mmqa/data/sf_{_msf}"
    ap_df = sp.read.option("header", "true").csv(os.path.join(data_dir, "ap_warrior.csv"))
    img_df = sp.read.option("header", "true").csv(os.path.join(data_dir, "thalamusdb_images.csv"))
    img_df = img_df.withColumn("image_filepath", concat(lit(data_dir + "/images/"), col("image_filename")))

    left_df = ap_df
    right_df = img_df

    prompt = """You will be provided with a horse racetrack name and an image.
Determine if the image shows the logo of the racetrack and A.P. Warrior was a contender.
Racetrack: {text:Track}
Image: {image:image_filepath}"""

    total_pairs = left_df.count() * right_df.count()
    print(f"[mmqa_q2a] total_pairs={total_pairs}")

    def id_expr(jdf):
        id_name = "ID" if "ID" in jdf.columns else "left_ID"
        img_name = "image_filename" if "image_filename" in jdf.columns else "right_image_filename"
        return jdf.select(col(id_name).alias("ID"), col(img_name).alias("image_filename"))

    exp_idx = 1
    for emb_model in _iter_emb_models("mmqa_q2a"):
        with _emb_scope("mmqa_q2a", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("mmqa_q2a"):
                    for k_val in KNN_K_VALUES:
                        joined_df, tag, start = run_experiment_join("mmqa_q2a", left_df, right_df, prompt, None, total_pairs, "propagation", bp, "embedding", k_val, abl_label, abl_strategy, abl_cs, exp_idx, embedding_model=emb_model)
                        r = collect_result(tag, joined_df, id_expr, None, start, "propagation", bp, "embedding", k_val, "mmqa_q2a", ablation_label=abl_label, scorer=_score_mmqa_q2a, embedding_model=emb_model)
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_join("mmqa_q2a", left_df, right_df, prompt, exp_idx, id_expr, None, scorer=_score_mmqa_q2a)


# ---------------------------------------------------------------------------
# MMQA Q7: sem_join (airline + destination + image logo)
# ---------------------------------------------------------------------------
def run_mmqa_q7():
    sp = get_spark()
    # scale factor 由 LASO_MMQA_SF 选择(默认 200 = PROBE 已跑的配置)
    _msf = os.environ.get("LASO_MMQA_SF", "200")
    data_dir = f"{DATA_ROOT}/SemBench/files/mmqa/data/sf_{_msf}"
    airport_df = sp.read.option("header", "true").csv(os.path.join(data_dir, "tampa_international_airport.csv"))
    img_df = sp.read.option("header", "true").csv(os.path.join(data_dir, "thalamusdb_images.csv"))
    img_df = img_df.withColumn("image_filepath", concat(lit(data_dir + "/images/"), col("image_filename")))

    # Dedup by Airlines: the source CSV is one row per (airline, destination)
    # so each airline appears multiple times. The Q7 GT keys on airline name
    # alone, so without dedup n_pos inflates by the #destinations-per-airline
    # factor (e.g. 5 GT airlines * 3 rows each = 15 "positives"). Matches the
    # dedup that viz_similarity_distribution.viz_mmqa_q7 already applied.
    airport_df = airport_df.dropDuplicates(["Airlines"])

    left_df = airport_df
    right_df = img_df

    prompt = """You will be provided with an airline name {text:Airlines} and its destination {text:Destinations} and an image {image:image_filepath}.
Determine for each airline if the airline's destination is in Europe and if the image shows the logo of the airline."""

    total_pairs = left_df.count() * right_df.count()
    print(f"[mmqa_q7] total_pairs={total_pairs}")

    def id_expr(jdf):
        airline_name = "Airlines" if "Airlines" in jdf.columns else "left_Airlines"
        img_name = "image_filename" if "image_filename" in jdf.columns else "right_image_filename"
        return jdf.select(col(airline_name).alias("Airlines"), col(img_name).alias("image_filename"))

    exp_idx = 1
    for emb_model in _iter_emb_models("mmqa_q7"):
        with _emb_scope("mmqa_q7", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("mmqa_q7"):
                    for k_val in KNN_K_VALUES:
                        joined_df, tag, start = run_experiment_join("mmqa_q7", left_df, right_df, prompt, None, total_pairs, "propagation", bp, "embedding", k_val, abl_label, abl_strategy, abl_cs, exp_idx, embedding_model=emb_model)
                        r = collect_result(tag, joined_df, id_expr, None, start, "propagation", bp, "embedding", k_val, "mmqa_q7", ablation_label=abl_label, scorer=_score_mmqa_q7, embedding_model=emb_model)
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_join("mmqa_q7", left_df, right_df, prompt, exp_idx, id_expr, None, scorer=_score_mmqa_q7)


# ---------------------------------------------------------------------------
# Classify scorers (for CLASSIFY-style queries decomposed as sem_join over
# (rows × categories) pairs).  A row with multiple True category matches
# keeps the first one; a row with zero True matches gets `fallback_category`.
# ---------------------------------------------------------------------------
def make_classify_macro_f1_scorer(id_col, cat_col, all_ids_df, fallback_category):
    def _score(pred_path, gt_path):
        from sklearn.metrics import precision_recall_fscore_support

        def _norm(v):
            if pd.isna(v) or str(v).lower() == "nan":
                return None
            return str(v).strip().lower().replace("\n", "")

        pred_df = pd.read_csv(pred_path, on_bad_lines="skip", engine="python", dtype=str)
        pred_df.columns = [c.strip().lower() for c in pred_df.columns]
        if id_col not in pred_df.columns or cat_col not in pred_df.columns:
            print(f"  [classify-f1] missing columns — have {list(pred_df.columns)}, need {id_col}, {cat_col}")
            return 0.0, 0.0, 0.0
        pred_df[id_col] = pred_df[id_col].astype(str).str.strip()
        pred_df[cat_col] = pred_df[cat_col].apply(_norm)
        pred_df = pred_df.dropna(subset=[cat_col]).drop_duplicates(subset=[id_col])

        gt_df = pd.read_csv(gt_path, dtype=str)
        gt_df.columns = [c.strip().lower() for c in gt_df.columns]
        gt_df[id_col] = gt_df[id_col].astype(str).str.strip()
        gt_df[cat_col] = gt_df[cat_col].apply(_norm)

        merged = gt_df[[id_col, cat_col]].merge(
            pred_df[[id_col, cat_col]].rename(columns={cat_col: "pred"}),
            on=id_col, how="left",
        )
        n_missing = int(merged["pred"].isna().sum())
        merged["pred"] = merged["pred"].fillna(_norm(fallback_category))

        merged = merged.sort_values(id_col)
        precision, recall, f1, _ = precision_recall_fscore_support(
            merged[cat_col], merged["pred"],
            average="macro", zero_division=0,
        )
        n_matched = len(merged) - n_missing
        print(
            f"  Classify macro - GT: {len(gt_df)}, Pred matched: {n_matched}, "
            f"Fallback-filled: {n_missing} (-> '{fallback_category}')"
        )
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return float(precision), float(recall), float(f1)
    return _score


# ---------------------------------------------------------------------------
# Cars Q1: sem_filter over complaint text
#   "Find cars that were in a crash/accident/collision."
# Per SemBench evaluator, GT = car_ids whose complaint has ``crash == True``.
# We run sem_filter on the full complaints table and dedupe to car_id.
# ---------------------------------------------------------------------------

    # no_opt_df, no_opt_tag, no_opt_start = run_no_opt_filter_experiment(
    #     "cars_q1", complaints_df, prompt, exp_idx,
    # )
    # no_opt_r = collect_result(
    #     no_opt_tag, no_opt_df, id_expr, gt_path, no_opt_start,
    #     "no_opt", 100, "n/a", "n/a",
    #     "cars_q1", ablation_label="no_opt", scorer=_score_cars_q1,
    #     embedding_model=EMBEDDING_MODEL,
    # )
    # all_results.append(no_opt_r)


# ---------------------------------------------------------------------------
# Cars Q3: sem_filter over vehicle images (multimodal)
#   "Find ten cars with manual transmission that are not damaged
#    according to images. Return the VIN."
# The `transmission == 'Manual'` predicate is a deterministic pre-filter;
# the LLM only sees Manual cars + their images and must decide
# "not damaged". Proxy output is evaluated against the full pre-LIMIT GT
# (Q3.csv, `vin` column).
# ---------------------------------------------------------------------------
def run_cars_q3():
    sp = get_spark()
    # scale factor 由 LASO_CARS_SF(回退 CARS_SF)选择(默认 157376 = PROBE 已跑的配置)
    _csf = _cars_sf()
    car_data = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(_cars_data_file("car_data"))
    image_data = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(_cars_data_file("image_car_data"))

    car_data = car_data.withColumn("registration_date", col("registration_date").cast("string"))
    image_data = image_data.withColumn(
        "image_path",
        concat(lit(f"{DATA_ROOT}/SemBench/"), col("image_path")),
    )

    # Deterministic pre-filter: only Manual-transmission cars.
    manual_cars = car_data.filter(col("transmission") == lit("Manual"))
    merged = manual_cars.join(image_data, on="car_id", how="inner")

    prompt = ("You are given an image of a vehicle or its parts. "
              "Return true if the car is NOT damaged (pristine / no visible damage). "
              "Image: {image:image_path}")

    gt_path = _cars_gt_path("Q3")
    total_rows = merged.count()
    print(f"[cars_q3] sf={_csf} total_rows={total_rows} (after transmission==Manual pre-filter)")

    def id_expr(rdf):
        # Ground truth uses `vin`; keep one row per VIN and enforce the
        # LIMIT 10 from the SemBench Q3 spec (mirrors lotus `.head(10)`).
        return rdf.select(col("vin").alias("vin"))

    exp_idx = 1
    for emb_model in _iter_emb_models("cars_q3"):
        with _emb_scope("cars_q3", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("cars_q3"):
                    for k_val in KNN_K_VALUES:
                        result_df, tag, start = run_experiment_filter(
                            "cars_q3", merged, prompt, gt_path,
                            total_rows, "propagation", bp, "embedding",
                            k_val, abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model,
                        )
                        r = collect_result(
                            tag, result_df, id_expr, gt_path, start,
                            "propagation", bp, "embedding", k_val,
                            "cars_q3", ablation_label=abl_label, scorer=_score_cars_q3,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_filter("cars_q3", merged, prompt, exp_idx, id_expr, gt_path, scorer=_score_cars_q3)


# ---------------------------------------------------------------------------
# Cars Q4: sem_filter over complaint text
#   "What is the average age of cars with engine problems?"
# The benchmark's ground truth is a scalar (average_age), so we evaluate
# the filter per-row by re-deriving the "engine-problem car_ids" set the
# same way the SemBench evaluator does (complaint.component_class == 'ENGINE')
# and scoring F1 on car_id sets. The scorer additionally logs the resulting
# aggregate avg_age for comparison to the Q4.csv scalar.
# ---------------------------------------------------------------------------
def run_cars_q4():
    sp = get_spark()
    # scale factor 由 LASO_CARS_SF(回退 CARS_SF)选择(默认 157376 = PROBE 已跑的配置)
    _csf = _cars_sf()
    complaints = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(_cars_data_file("text_complaints_data"))

    complaints_df = complaints.select(
        col("car_id").cast("string").alias("car_id"),
        col("summary").alias("summary"),
    )

    prompt = ("You are given a textual complaint about a vehicle. Return true if "
              "the complaint describes a problem with the car's engine (or a "
              "component directly connected to the engine), false otherwise.\n"
              "Complaint: {summary}")

    # gt_path points at the scalar Q4.csv for bookkeeping; the scorer
    # re-derives per-row GT internally from the Q4 rows file (sf follows _cars_sf()).
    gt_path = _cars_gt_path("Q4")
    total_rows = complaints_df.count()
    print(f"[cars_q4] sf={_csf} total_rows={total_rows}")

    def id_expr(rdf):
        # Q4 is scored against the scalar average_age, so we need the full
        # set of predicted positives (no LIMIT) to compute the mean year.
        return rdf.select(col("car_id").cast("string").alias("car_id"))

    exp_idx = 1
    for emb_model in _iter_emb_models("cars_q4"):
        with _emb_scope("cars_q4", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("cars_q4"):
                    for k_val in KNN_K_VALUES:
                        result_df, tag, start = run_experiment_filter(
                            "cars_q4", complaints_df, prompt, gt_path,
                            total_rows, "propagation", bp, "embedding",
                            k_val, abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model,
                        )
                        r = collect_result(
                            tag, result_df, id_expr, gt_path, start,
                            "propagation", bp, "embedding", k_val,
                            "cars_q4", ablation_label=abl_label, scorer=_score_cars_q4,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_filter("cars_q4", complaints_df, prompt, exp_idx, id_expr, gt_path, scorer=_score_cars_q4)


# ---------------------------------------------------------------------------
# Cars Q10: CLASSIFY complaints into 24 problem categories (text)
# Reframed as sem_join (complaints × categories).  For each complaint, the
# predicted category is the first sem_join match; unmatched complaints fall
# back to the most common category ('engine').
# ---------------------------------------------------------------------------

    # no_opt_df, no_opt_tag, no_opt_start = run_no_opt_join_experiment(
    #     "cars_q10", complaints_df, categories_df, prompt, exp_idx,
    # )
    # no_opt_r = collect_result(
    #     no_opt_tag, no_opt_df, id_expr, gt_path, no_opt_start,
    #     "no_opt", 100, "n/a", "n/a",
    #     "cars_q10", ablation_label="no_opt", scorer=scorer,
    #     embedding_model=EMBEDDING_MODEL,
    # )
    # all_results.append(no_opt_r)


# ---------------------------------------------------------------------------
# Cars Q8: sem_filter (multimodal — puncture AND paint scratches)
# ---------------------------------------------------------------------------
def run_cars_q7():
    """Paper case-study Q7: dented images, N=30,130 at sf_157376."""
    from experiments.workload_ground_truth import cars_q7_image_ground_truth
    sp = get_spark()
    images = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"').csv(_cars_data_file("image_car_data"))
    images = images.withColumn("image_path", concat(lit(f"{DATA_ROOT}/SemBench/"), col("image_path")))
    prompt = ("You are given an image of a vehicle or its parts. "
              "Return true if the car is dented. Image: {image:image_path}")
    gt_path = cars_q7_image_ground_truth(DATA_ROOT, _cars_sf(), results_dir)
    total_rows = images.count()
    def id_expr(rdf):
        return rdf.select(col("car_id").cast("string").alias("car_id"))
    exp_idx = 1
    for emb_model in _iter_emb_models("cars_q7"):
        with _emb_scope("cars_q7", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("cars_q7"):
                    for k_val in KNN_K_VALUES:
                        result_df, tag, start = run_experiment_filter(
                            "cars_q7", images, prompt, gt_path, total_rows, "propagation", bp,
                            "embedding", k_val, abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model)
                        all_results.append(collect_result(
                            tag, result_df, id_expr, gt_path, start, "propagation", bp,
                            "embedding", k_val, "cars_q7", ablation_label=abl_label,
                            embedding_model=emb_model))
                        exp_idx += 1
    maybe_append_no_opt_filter("cars_q7", images, prompt, exp_idx, id_expr, gt_path)


def run_cars_q8():
    sp = get_spark()
    # scale factor 由 LASO_CARS_SF(回退 CARS_SF)选择(默认 157376 = PROBE 已跑的配置)
    _csf = _cars_sf()
    car_data = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(_cars_data_file("car_data"))
    image_data = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(_cars_data_file("image_car_data"))

    car_data = car_data.withColumn("registration_date", col("registration_date").cast("string"))
    image_data = image_data.withColumn(
        "image_path",
        concat(lit(f"{DATA_ROOT}/SemBench/"), col("image_path")),
    )

    merged = car_data.join(image_data, on="car_id", how="inner")

    prompt = ("You are given an image of a vehicle or its parts. "
              "Return true if car has both, puncture and paint scratches. "
              "Image: {image:image_path}")

    gt_path = _cars_gt_path("Q8")
    total_rows = merged.count()
    print(f"[cars_q8] sf={_csf} total_rows={total_rows}")

    def id_expr(rdf):
        return rdf.select("car_id")

    exp_idx = 1
    for emb_model in _iter_emb_models("cars_q8"):
        with _emb_scope("cars_q8", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("cars_q8"):
                    for k_val in KNN_K_VALUES:
                        result_df, tag, start = run_experiment_filter(
                            "cars_q8", merged, prompt, gt_path,
                            total_rows, "propagation", bp, "embedding",
                            k_val, abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model,
                        )
                        r = collect_result(
                            tag, result_df, id_expr, gt_path, start,
                            "propagation", bp, "embedding", k_val,
                            "cars_q8", ablation_label=abl_label, scorer=_score_cars_q8,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_filter("cars_q8", merged, prompt, exp_idx, id_expr, gt_path, scorer=_score_cars_q8)


# ---------------------------------------------------------------------------
# Ecomm Q6: CLASSIFY apparel images into 5 categories
# Reframed as sem_join (images × categories) with natural-language
# category descriptions for richer semantic matching.
# ---------------------------------------------------------------------------
def run_ecomm_q6():
    sp = get_spark()
    data_dir = f"{DATA_ROOT}/SemBench/files/ecomm/data/sf_4000"
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

    images_df = image_map.select(
        col("id").cast("string").alias("left_id"),
        col("image_filepath").alias("left_image_filepath"),
    )

    category_descriptions = [
        ("Dress",
         "A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consists of a bodice attached to a skirt."),
        ("Bottomwear",
         "Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings."),
        ("Socks",
         "Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth."),
        ("Topwear",
         "Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets."),
        ("Innerwear",
         "Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts."),
    ]
    cat_values_sql = ", ".join(
        "('{}', '{}')".format(
            name.replace("'", "''"),
            desc.replace("'", "''"),
        )
        for name, desc in category_descriptions
    )
    categories_df = sp.sql(
        f"SELECT col1 AS right_category, col2 AS right_description FROM VALUES {cat_values_sql}"
    )

    prompt = """Does this product image show a "{right_category}"?
Category definition: {right_description}
Answer True if the image depicts this category, False otherwise.

Image: {image:left_image_filepath}"""

    gt_path = f"{DATA_ROOT}/SemBench/files/ecomm/raw_results/ground_truth_sf4000/Q6.csv"
    n_images = images_df.count()
    total_pairs = n_images * len(category_descriptions)
    print(f"[ecomm_q6] n_images={n_images}, n_categories={len(category_descriptions)}, total_pairs={total_pairs}")

    def id_expr(jdf):
        id_name = "left_id" if "left_id" in jdf.columns else "id"
        cat_name = "right_category" if "right_category" in jdf.columns else "category"
        return jdf.select(
            col(id_name).cast("string").alias("id"),
            col(cat_name).alias("category"),
        )

    scorer = make_classify_macro_f1_scorer(
        id_col="id",
        cat_col="category",
        all_ids_df=images_df,
        fallback_category="Topwear",
    )

    exp_idx = 1
    for emb_model in _iter_emb_models("ecomm_q6"):
        with _emb_scope("ecomm_q6", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("ecomm_q6"):
                    for k_val in KNN_K_VALUES:
                        joined_df, tag, start = run_experiment_join(
                            "ecomm_q6", images_df, categories_df, prompt, gt_path,
                            total_pairs, "propagation", bp, "embedding",
                            k_val, abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model,
                        )
                        r = collect_result(
                            tag, joined_df, id_expr, gt_path, start,
                            "propagation", bp, "embedding", k_val,
                            "ecomm_q6", ablation_label=abl_label, scorer=scorer,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_join("ecomm_q6", images_df, categories_df, prompt, exp_idx, id_expr, gt_path, scorer=scorer)


# ---------------------------------------------------------------------------
# Animals benchmark (SemBench Q1 / Q3 / Q10): sem_filter over zebra images.
# All three share the same predicate but differ in the post-filter aggregation
# (count / top-city / top-(city,stationID)). Dataset columns:
#   ImagePath, City, StationID, id, ...
# Ground-truth files under .../animals/raw_results/ground_truth/Q{1,3,10}.csv
# ---------------------------------------------------------------------------

def _animals_load_df():
    # File-order IDs distinguish repeated image rows and match evaluation annotations.
    from proxy_model import _pandas_to_spark
    scale = int(os.environ.get('LASO_ANIMALS_SF','1600'))
    frame = pd.read_csv(f'{DATA_ROOT}/SemBench/files/animals/data/sf_{scale}/image_data.csv')
    frame['row_id'] = np.arange(len(frame))
    return _pandas_to_spark(get_spark(),frame.drop(columns=['Species']))


ANIMALS_PROMPT = "Does the image contain a zebra? Image: {image:ImagePath}"


def run_animals_q1():
    df = _animals_load_df()
    prompt = ANIMALS_PROMPT
    from experiments.workload_ground_truth import animals_q1_ground_truth
    gt_path = animals_q1_ground_truth(DATA_ROOT,os.environ.get('LASO_ANIMALS_SF','1600'),results_dir)
    total_rows = df.count()
    print(f"[animals_q1] total_rows={total_rows}")

    # Report row-level F1 for the paper's semantic filter.
    def id_expr(rdf):
        return rdf.select("row_id")

    exp_idx = 1
    for emb_model in _iter_emb_models("animals_q1"):
        with _emb_scope("animals_q1", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("animals_q1"):
                    for k_val in KNN_K_VALUES:
                        result_df, tag, start = run_experiment_filter(
                            "animals_q1", df, prompt, gt_path,
                            total_rows, "propagation", bp, "embedding",
                            k_val, abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model,
                        )
                        r = collect_result(
                            tag, result_df, id_expr, gt_path, start,
                            "propagation", bp, "embedding", k_val,
                            "animals_q1", ablation_label=abl_label,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_filter("animals_q1", df, prompt, exp_idx, id_expr, gt_path)


    # no_opt_df, no_opt_tag, no_opt_start = run_no_opt_filter_experiment(
    #     "animals_q3", df, prompt, exp_idx,
    # )
    # no_opt_r = collect_result(
    #     no_opt_tag, no_opt_df, id_expr, gt_path, no_opt_start,
    #     "no_opt", 100, "n/a", "n/a",
    #     "animals_q3", ablation_label="no_opt", scorer=_score_animals_q3,
    #     embedding_model=EMBEDDING_MODEL,
    # )
    # all_results.append(no_opt_r)


    # no_opt_df, no_opt_tag, no_opt_start = run_no_opt_filter_experiment(
    #     "animals_q10", df, prompt, exp_idx,
    # )
    # no_opt_r = collect_result(
    #     no_opt_tag, no_opt_df, id_expr, gt_path, no_opt_start,
    #     "no_opt", 100, "n/a", "n/a",
    #     "animals_q10", ablation_label="no_opt"0,
    #     embedding_model=EMBEDDING_MODEL,
    # )
    # all_results.append(no_opt_r)


# ---------------------------------------------------------------------------
# JoinML paper datasets (arXiv:2603.16153): VeRi, Flickr30K, Roxford, Company.
#
# Each runner assumes the corresponding dataset/<name>_groundtruth.py script
# has been executed to materialise three CSVs under <data_dir>/gt/:
#   <name>_left.csv, <name>_right.csv, gt_pairwise_<name>.csv (label=1 only).
# ---------------------------------------------------------------------------

    # no_opt_df, no_opt_tag, no_opt_start = run_no_opt_join_experiment(
    #     query_tag, left_df, right_df, prompt, exp_idx,
    # )
    # no_opt_r = collect_result(
    #     no_opt_tag, no_opt_df, id_expr, gt_path, no_opt_start,
    #     "no_opt", 100, "n/a", "n/a",
    #     query_tag, ablation_label="no_opt", scorer=scorer,
    #     embedding_model=EMBEDDING_MODEL,
    # )
    # all_results.append(no_opt_r)


# ---------------------------------------------------------------------------
# VeRi Q1: sem_join (same vehicle across two camera groups)
# Prompt verbatim from JoinML paper §C.2.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Flickr30K Q1: sem_join (caption describes image)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Roxford Q1: sem_join (same landmark across gallery and queries)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Company Q1: sem_join (entity resolution on company descriptions)
# ---------------------------------------------------------------------------

LROBENCH_DATABASES_DIR = os.environ.setdefault(
    "LROBENCH_DATABASES_DIR", f"{DATA_ROOT}/databases"
)


def _write_match_gt_csv(query_tag, pairs):
    os.makedirs(results_dir, exist_ok=True)
    gt_path = f"{results_dir}{query_tag}_gt.csv"
    pd.DataFrame(
        {"id": [f"{a}-{b}" for a, b in pairs]}
    ).to_csv(gt_path, index=False)
    return gt_path


def _run_match_join(query_tag, left_df, right_df, prompt, gt_path,
                    left_id_col, right_id_col):
    # Collapse any shuffle-induced partition split (e.g. `.distinct()` and
    # `.orderBy().limit()` produce 200 partitions by default). The proxy
    # pipeline zips the two sides' row-indexed RDDs internally and fails
    # with "Can't zip RDDs with unequal numbers of partitions" whenever
    # left/right have different partition counts. All LRobench match
    # inputs are tiny (≤ a few thousand rows), so coalescing to a single
    # partition on each side is safe.
    left_df = left_df.coalesce(1)
    right_df = right_df.coalesce(1)

    total_pairs = left_df.count() * right_df.count()
    print(f"[{query_tag}] total_pairs={total_pairs}  "
          f"left={left_df.count()}  right={right_df.count()}")

    def id_expr(jdf):
        return jdf.withColumn(
            "id",
            concat_ws(
                "-",
                col(left_id_col).cast("string"),
                col(right_id_col).cast("string"),
            ),
        ).select("id")

    exp_idx = 1
    for emb_model in _iter_emb_models(query_tag):
        with _emb_scope(query_tag, emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query(query_tag):
                    for k_val in KNN_K_VALUES:
                        joined_df, tag, start = run_experiment_join(
                            query_tag, left_df, right_df, prompt, gt_path,
                            total_pairs, "propagation", bp, "embedding",
                            k_val, abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model,
                        )
                        r = collect_result(
                            tag, joined_df, id_expr, gt_path, start,
                            "propagation", bp, "embedding", k_val,
                            query_tag, ablation_label=abl_label,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_join(query_tag, left_df, right_df, prompt, exp_idx, id_expr, gt_path)


# ---------------------------------------------------------------------------
# Additional LRobench match queries
# (match113, match121, match122, match125: CELL-level text joins
#  match202, match203: ROW-level entity resolution
#  match305: COLUMN-level schema matching)
# ---------------------------------------------------------------------------
def run_match113():
    sp = get_spark()
    home = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/"
             "home_office_senior_officials_travel_data_return.csv")
    )
    left_df = (
        home.select(col("Name of Official").alias("left_name"))
            .dropna().dropDuplicates(["left_name"])
    )

    travel = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/"
             "travel-exp-April-June-2018.csv")
    )
    right_df = (
        travel.select(col("Senior Officials Name").alias("right_name"))
              .dropna().dropDuplicates(["right_name"])
    )

    prompt = """Determine whether the two senior government official names share the same first name (first given name, ignoring middle names / hyphenated parts).

Official A: {left_name}
Official B: {right_name}"""

    truth = [
        ('Mark Sedwill ',  'Mark Bryson-Richardson'),
        ('Richard Clarke', 'Richard Montgomery'),
    ]
    gt_path = _write_match_gt_csv("match113", truth)
    _run_match_join("match113", left_df, right_df, prompt, gt_path,
                    "left_name", "right_name")


def run_match121():
    sp = get_spark()
    april = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/01.Apr_2018.csv")
    )
    left_df = (
        april.select(col("Supplier").alias("left_Supplier"))
             .dropna().dropDuplicates(["left_Supplier"])
    )

    may = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/2015_05_expenditure.csv")
    )
    right_df = (
        may.select(col("Supplier").alias("right_Supplier"))
           .dropna().dropDuplicates(["right_Supplier"])
    )

    prompt = """Determine whether these two supplier names refer to the same organisation (ignoring case, abbreviations, punctuation, trailing legal suffixes such as Ltd/Limited, and minor formatting differences).

Supplier A: {left_Supplier}
Supplier B: {right_Supplier}"""

    truth = [
        ('Nhs Supply Chain',                 'NHS SUPPLY CHAIN'),
        ('Novartis Pharmaceuticals Uk Ltd',  'NOVARTIS PHARMACEUTICALS UK LTD'),
        ('Roche Diagnostics Limited',        'ROCHE PRODUCTS LTD'),
        ('Csc',                              'CSC COMPUTER SCIENCES LTD'),
        ('Philips Healthcare',               'PHILIPS HEALTHCARE'),
        ('Nhs Blood And Transplant',         'NHS BLOOD & TRANSPLANT'),
        ('NHS Litigation Authority',         'NHS LITIGATION AUTHORITY'),
    ]
    gt_path = _write_match_gt_csv("match121", truth)
    _run_match_join("match121", left_df, right_df, prompt, gt_path,
                    "left_Supplier", "right_Supplier")


def run_match202():
    """LRO-Bench 202: high-rated movies, 24 x 16 = 384 candidates."""
    sp = get_spark()
    imdb = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/movies/imdb.csv")
    )
    imdb = imdb.filter(col("Rating").cast("double") > 8.3)
    left_df = imdb.select(
        col("ID").alias("left_ID"),
        col("Title").alias("left_Title"),
        col("Year").alias("left_Year"),
        col("Director").alias("left_Director"),
        col("Genre").alias("left_Genre"),
    )
    tomato = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/movies/rotten_tomatoes.csv")
    )
    tomato = tomato.filter(col("Rating").cast("double") > 8.5)
    right_df = tomato.select(
        col("ID").alias("right_ID"),
        col("Title").alias("right_Title"),
        col("Year").alias("right_Year"),
        col("Director").alias("right_Director"),
        col("Genre").alias("right_Genre"),
    )
    prompt = """Determine whether the two records refer to the same movie (same film, identified by matching title + year + director even if formatting differs).

IMDb: title={left_Title} | year={left_Year} | director={left_Director} | genre={left_Genre}
Rotten Tomatoes: title={right_Title} | year={right_Year} | director={right_Director} | genre={right_Genre}"""
    truth = [('a-456', 'b-434'), ('a-569', 'b-534'), ('a-701', 'b-653')]
    gt_path = _write_match_gt_csv("match202", truth)
    _run_match_join("match202", left_df, right_df, prompt, gt_path,
                    "left_ID", "right_ID")


def _column_descriptor_df(sp, csv_path, side, sample_n=5):
    """Build a Spark DataFrame where each row describes one column of `csv_path`:
       ({side}_colname, {side}_samples).  Used for schema-matching queries."""
    pdf = pd.read_csv(csv_path, nrows=500)
    rows = []
    for c in pdf.columns:
        vals = pdf[c].dropna().astype(str).head(sample_n).tolist()
        samples = " | ".join(vals) if vals else ""
        rows.append({f"{side}_colname": c, f"{side}_samples": samples})
    desc_pdf = pd.DataFrame(rows)
    # Use proxy_model's converter: PySpark 3.3's createDataFrame calls
    # pandas.DataFrame.iteritems(), removed in pandas 2.x. Calling
    # sp.createDataFrame directly only worked when some earlier query had already
    # installed that compatibility alias, so match305 failed when run first.
    from proxy_model import _pandas_to_spark
    return _pandas_to_spark(sp, desc_pdf)


def run_match305():
    sp = get_spark()
    left_df = _column_descriptor_df(
        sp, f"{LROBENCH_DATABASES_DIR}/california_schools/frpm.csv", "left"
    )
    right_df = _column_descriptor_df(
        sp, f"{LROBENCH_DATABASES_DIR}/california_schools/schools.csv", "right"
    )

    prompt = """Determine whether the two columns from different tables describe the same real-world feature (same semantic attribute), using the column names and sample values.

Table A column: name={left_colname} | samples={left_samples}
Table B column: name={right_colname} | samples={right_samples}"""

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
    gt_path = _write_match_gt_csv("match305", truth)
    _run_match_join("match305", left_df, right_df, prompt, gt_path,
                    "left_colname", "right_colname")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
QUERY_RUNNERS = {
    "ecomm_q6": run_ecomm_q6,
    "ecomm_q7": run_ecomm_q7,
    "ecomm_q8": run_ecomm_q8,
    "ecomm_q9": run_ecomm_q9,
    "ecomm_q13": run_ecomm_q13,
    "movie_q5": run_movie_q5,
    "mmqa_q2a": run_mmqa_q2a,
    "mmqa_q7": run_mmqa_q7,

    "cars_q3": run_cars_q3,
    "cars_q4": run_cars_q4,
    "cars_q7": run_cars_q7,
    "cars_q8": run_cars_q8,
    # Animals benchmark (SemBench Q1/Q3/Q10)
    "animals_q1": run_animals_q1,
    # # JoinML paper datasets
    # LRobench match queries (text-text semantic joins)
    "match113": run_match113,
    "match121": run_match121,
    "match202": run_match202,
    "match305": run_match305,
}

if __name__ == "__main__":
    queries = _laso_resolve_queries(sys.argv)

    if os.environ.get("LASO_DRYRUN"):
        # Echo the fully resolved configuration and exit WITHOUT starting
        # Spark / the LLM endpoint. Used to verify experiments/*.sh wiring.
        print("=== LASO_DRYRUN: resolved configuration ===")
        print(f"queries        = {queries}")
        print(f"ABLATION_CONFIGS = {ABLATION_CONFIGS}")
        print(f"BUDGET_PCTS    = {BUDGET_PCTS}")
        print(f"QUERY_BUDGET_PCTS = {QUERY_BUDGET_PCTS}")
        print(f"KNN_K_VALUES   = {KNN_K_VALUES}")
        from dataclasses import asdict as _config_asdict
        resolved_sampling = {
            method: _fixed_ce_runtime_options(RANDOM_SEED, sampling_method=method)
            for method in sorted({entry[1] for entry in ABLATION_CONFIGS})
        }
        print("SAMPLING_RUNTIME = " + json.dumps(resolved_sampling, default=_config_asdict, sort_keys=True))
        print(f"EMBEDDING_MODEL = {EMBEDDING_MODEL}")
        print(f"EMBEDDING_MODELS_MULTIMODAL = {EMBEDDING_MODELS_MULTIMODAL}")
        print(f"EMBEDDING_MODELS_TEXT = {EMBEDDING_MODELS_TEXT}")
        print(f"MODEL (SEMBENCH_MODEL) = {MODEL}")
        print(f"DATA_ROOT      = {DATA_ROOT}")
        print(f"LROBENCH_DATABASES_DIR = {LROBENCH_DATABASES_DIR}")
        print(f"RANDOM_SEED    = {RANDOM_SEED}")
        print(f"INCLUDE_NO_OPT = {INCLUDE_NO_OPT}")
        print(f"results_dir    = {results_dir}")
        sys.exit(0)

    os.makedirs(results_dir, exist_ok=True)
    for q in queries:
        q_start = len(all_results)
        try:
            QUERY_RUNNERS[q]()
        except Exception as e:
            print(f"ERROR running {q}: {e}")
            import traceback
            traceback.print_exc()
        q_rows = all_results[q_start:]
        save_query_results(q, q_rows)

    print("\n" + "=" * 80)
    print("activeSampling GRID SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(all_results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary_df.to_string(index=False))

    _run_tag = "_".join(sorted(queries)) if len(queries) <= 3 else "selected_queries"
    _run_tag = _run_tag.replace("/", "_").replace(" ", "_")
    summary_path = f"{results_dir}{_run_tag}_grid_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    if spark:
        spark.stop()
    print("\nDone.")
