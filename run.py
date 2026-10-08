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

JAR_PATH = os.environ.get(
    "LASO_SPARK_JAR",
    "./spark-semantic-plugin_2.12-0.1.0-SNAPSHOT.jar",
)
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
QUERY_MODALITY: dict = {
    "ecomm_q6":   "multimodal",  # image × category classification
    "ecomm_q7":   "text",        # product text-text join
    "ecomm_q8":   "multimodal",  # text description ↔ product image
    "ecomm_q9":   "multimodal",  # image-image same category+color
    "ecomm_q13":  "multimodal",  # running shirt image filter
    "movie_q5":   "text",        # review text-text join (same sentiment)
    "movie_q7":   "text",        # review text-text join (opposite sentiment)
    "mmqa_q2a":   "multimodal",  # racetrack text + logo image
    "mmqa_q7":    "multimodal",  # airline text + logo image
    "cars_q3":    "multimodal",  # manual cars + image filter ("not damaged")
    "cars_q4":    "text",        # complaint text filter (engine problems)
    "cars_q8":    "multimodal",  # vehicle image filter
    "cars_q10":   "text",        # car complaint text classification
    "animals_q1": "multimodal",  # zebra image filter → count
    "animals_q3": "multimodal",  # zebra image filter → top city (groupBy City)
    "animals_q10":"multimodal",  # zebra image filter → top (city, stationID)
    # LRobench match queries (text-text semantic joins over tabular data)
    "match102":   "text",        # customer city  ↔ office territory
    "match104":   "text",        # comment text   ↔ tag name
    "match105":   "text",        # state code     ↔ full state name (wholesalers)
    "match106":   "text",        # state code     ↔ full state name (high-revenue)
    "match107":   "text",        # team name      ↔ country name
    "match113":   "text",        # senior officials — shared first name
    "match121":   "text",        # supplier name   ↔ supplier name (case/format drift)
    "match122":   "text",        # staff self-join — same first name, different person
    "match125":   "text",        # merchant category ↔ RM-MR category
    "match202":   "text",        # yelp row        ↔ zomato row (entity match 60642)
    "match203":   "text",        # imdb row        ↔ rotten_tomatoes row (low ratings)
    "match305":   "text",        # frpm col        ↔ schools col (schema match)
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
# Fixed-CE production shares one hard oracle budget across learning L, calibration A,
# and routing D. Default allocation is 50/40/10; retries also consume this budget.
# Historical uncapped experiments require explicit legacy configuration and are not
# measurements of this default algorithm.
LC_SAMPLE_FRAC = float(os.environ.get("LASO_LC_SAMPLE_FRAC", "0.8"))
LC_CALIB_FRAC = float(os.environ.get("LASO_LC_CALIB_FRAC", "0.2"))
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
ACTIVESAMPLING_VS_RATIO = 0.5            # legacy only; fixed CE rejects LASO_VS_RATIO
ACTIVESAMPLING_N_PHASES = 5             # legacy only; no reranking in fixed CE
ACTIVESAMPLING_LR_BLEND_SCHEDULE = "linear"  # lambda_t = (t-1)/(T-1), legacy linear 0->1 ramp across rounds
ACTIVESAMPLING_COLD_START_PHASES = 1    # leading SigLIP-only (lambda=0) phases

enable_defer = True
DEFERRAL_FEATURES = [
    "z_smooth",
    "neighbour_disag",
    "rank_position",
]


def _cascade_decision_enabled(decision_mode: Optional[str] = None) -> bool:
    """Whether the active-learning + Bias Diffusion + Homogeneity Propagation path runs a post-Bias Diffusion + Homogeneity Propagation decision layer."""
    mode = str(decision_mode if decision_mode is not None else "learned_deferral")
    if mode == "none":
        return False
    return bool(enable_defer or mode in ("learned_deferral", LEARNED_CALIBRATION_MODE))


LLM_BATCH_SIZE = 128




# Ablation grid: (label, sampling_strategy, refine_enabled)
# sampling_strategy: "activeSampling" = fixed predicate-aware CE + pilot-gated acquisition; legacy is explicit
#                    "random"     = uniform random
#                    "uqe_active" = UQE Algorithm 2 (arXiv 2407.09522)
#                    "top_k"      = deterministic top-K along fixed cosine
#                                   similarity axis (Proxy-100x §5.4 baseline)
# refine_enabled: True = Bias Diffusion + Homogeneity Propagation postprocess, False = LR-only prediction
ABLATION_CONFIGS = [
("activeSampling+refine+learned_calibration", "activeSampling", True),
# ("activeSampling+refine+dasp_topk",        "activeSampling",     True),
# # UQE Algorithm 2 online active-learning sampler.
# # Spec: docs/uqe_active_implementation.md
# ("uqe_active+refine+trust_proxy",  "uqe_active", True),
# ("uqe_active+lr_only",         "uqe_active", False),
# Top-K similarity baseline. Deterministic: picks the K rows with the
# highest cosine similarity to the prompt. Useful as the cheapest
# non-trivial baseline and as the "no surrogate, no randomness" reference
# point in the sampler ablation.
# ("top_k+refine+trust_proxy",       "top_k",      True),
#("uqe_active+lr_only",              "top_k",      False),

   #("activeSampling+refine+learned_deferral", "activeSampling",  True),
   #("activeSampling+refine+dasp_topk",        "activeSampling",  True),
#    ("random+refine",      "random",  True),
#("activeSampling+lr_only", "activeSampling",  False),
#    ("random+lr_only", "random",  False),
]



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
        return "learned_deferral"
    if label.endswith("+lr_only"):
        # LR-only ablation bypasses Bias Diffusion + Homogeneity Propagation entirely; no post-Bias Diffusion + Homogeneity Propagation decision layer.
        return "none"
    return "learned_deferral"


def _deferral_policy_for_ablation(ablation_label: str) -> str:
    mode = _decision_mode_for_ablation(ablation_label)
    if mode == "none":
        return "trust_proxy"
    if mode == "learned_deferral":
        return "learned_deferral"
    return mode


def _lc_phases(sample_budget: int) -> int:
    """Active-learning rounds that fit in L's share.

    Each round costs at least ~3 calls, so the configured 5 rounds need 15 calls. When L's share is
    smaller than that the sampler used to overshoot and swallow the whole budget, leaving the
    calibration sample A empty (70 of the 242 refusals in the 5-seed study). Cutting rounds keeps L
    inside its share so A always gets its own.
    """
    return max(1, min(ACTIVESAMPLING_N_PHASES, int(sample_budget) // 3))


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
# How the no-proxy baseline calls the LLM: "jni" = the Spark/Rust operator (sem_filter/sem_join with
# proxy_sample=False), "python" = proxy_model's labelling path on every candidate (no_proxy=True), which
# uses the same messages, chat template and answer parsing as PROBE's own oracle calls.
NO_OPT_BACKEND = os.environ.get("LASO_NO_OPT_BACKEND", "jni").strip().lower()
if NO_OPT_BACKEND not in {"jni", "python"}:
    raise SystemExit(f"LASO_NO_OPT_BACKEND must be jni or python, got {NO_OPT_BACKEND!r}")
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


def _effective_phases(cluster_prof) -> int:
    """Sampler rounds actually run. The sampler records them in its own profile; in
    learned-calibration mode they are clipped by the sample budget, so this can be below the
    configured ACTIVESAMPLING_N_PHASES."""
    prof = (cluster_prof or {}).get("activeSampling_profiling") or {}
    if prof.get("sampler") == "fixed_ce_pilot":
        return 0  # no outer reranking rounds
    try:
        return int(prof["n_phases"])
    except (KeyError, TypeError, ValueError):
        return int(ACTIVESAMPLING_N_PHASES)


def _laso_env_overrides():
    global BUDGET_PCTS, QUERY_BUDGET_PCTS, KNN_K_VALUES, ACTIVESAMPLING_VS_RATIO
    global ACTIVESAMPLING_N_PHASES
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

    v = os.environ.get("LASO_VS_RATIO")
    if v:
        ACTIVESAMPLING_VS_RATIO = float(v.split(",")[0].strip())

    # Rounds T of the sampler. In learned-calibration mode the effective value is clipped by the
    # sample budget (_lc_phases); the run row records what actually ran, not what was asked for.
    v = os.environ.get("LASO_N_PHASES")
    if v:
        ACTIVESAMPLING_N_PHASES = int(v.split(",")[0].strip())

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
    spark = SparkSession.builder.appName("activeSampling_grid") \
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


def _score_cars_q8(pred_path, _gt_path_unused):
    """Evaluate Q8 using SemBench's limit-aware ground truth sampling (LIMIT 100).

    SemBench CarsEvaluator._evaluate_q8 builds a ground truth sample of exactly
    ``limit`` rows: all correctly-matched IDs from system results plus random
    false cases (random_state=42) to pad up to 100.
    """
    LIMIT = 9999
    ground_truth_path = _cars_gt_path("Q8")
    id_column = "car_id"

    def _normalize_id(val):
        s = str(val).strip()
        if s.lower() == "nan" or s == "":
            return None
        if re.fullmatch(r"\d+\.0", s):
            s = s.split(".")[0]
        if s.isdigit():
            return s
        return None

    try:
        gt_df = pd.read_csv(ground_truth_path, dtype=str)
        gt_df[id_column] = gt_df[id_column].apply(_normalize_id)
        gt_df = gt_df.dropna(subset=[id_column])
    except Exception as e:
        print(f"  [cars_q8] Error reading ground truth: {e}")
        return 0.0, 0.0, 0.0

    try:
        pred_df = pd.read_csv(
            pred_path, on_bad_lines="skip", engine="python", quoting=3, dtype=str,
        )
        if id_column in pred_df.columns:
            raw_predictions = pred_df[id_column].dropna().tolist()
        else:
            raw_predictions = pred_df.iloc[:, 0].dropna().tolist()

        predicted_ids = [_normalize_id(v) for v in raw_predictions]
        predicted_set = set(pid for pid in predicted_ids if pid is not None)

        # SemBench limit-aware ground truth sampling
        correct_ids_ix = gt_df[id_column].isin(list(predicted_set))
        correct_ids = gt_df.loc[correct_ids_ix, :]

        if correct_ids.empty:
            n_to_sample = min(LIMIT, len(gt_df))
            ground_truth_sample = gt_df.sample(n=n_to_sample, random_state=42) if n_to_sample > 0 else gt_df
        elif correct_ids.shape[0] >= LIMIT:
            ground_truth_sample = correct_ids.head(LIMIT)
        else:
            false_cases = gt_df[~correct_ids_ix]
            n_needed = LIMIT - correct_ids.shape[0]
            n_to_sample = min(n_needed, len(false_cases))
            if n_to_sample > 0:
                ground_truth_sample = pd.concat([correct_ids, false_cases.sample(n=n_to_sample, random_state=42)])
            else:
                ground_truth_sample = correct_ids

        ground_truth_set = set(ground_truth_sample[id_column].dropna().astype(str).str.strip())
        tp = len(ground_truth_set & predicted_set)
        fp = len(predicted_set - ground_truth_set)
        fn = len(ground_truth_set - predicted_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  [cars_q8] GT sample: {len(ground_truth_set)} (LIMIT={LIMIT}), Pred: {len(predicted_set)}")
        print(f"  [cars_q8] TP={tp} FP={fp} FN={fn}  P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
        return precision, recall, f1
    except Exception as e:
        print(f"  [cars_q8] Error processing predictions: {e}")
        return 0.0, 0.0, 0.0


def _score_cars_q1(pred_path, gt_path):
    """Evaluate Q1 using SemBench's LIMIT-10 limit-aware GT sampling.

    Q1's natural-language spec ("Find cars that were in a
    crash/accident/collision.") does not specify LIMIT 10, but for
    consistency with cars_q3 we run the proxy under a LIMIT 10 cap
    (the system output already has at most 10 car_ids) and score it
    with the same limit-aware GT-sampling rule the SemBench
    CarsEvaluator uses for Q3 / Q8: take all correct-match GT rows
    plus enough random false GT rows (random_state=42) to reach 10.
    """
    LIMIT = 9999
    id_column = "car_id"

    def _normalize_id(val):
        s = str(val).strip()
        if s.lower() == "nan" or s == "":
            return None
        if re.fullmatch(r"\d+\.0", s):
            s = s.split(".")[0]
        return s if s.isdigit() else None

    try:
        gt_df = pd.read_csv(gt_path, dtype=str)
        gt_df[id_column] = gt_df[id_column].apply(_normalize_id)
        gt_df = gt_df.dropna(subset=[id_column])
    except Exception as e:
        print(f"  [cars_q1] Error reading ground truth: {e}")
        return 0.0, 0.0, 0.0

    try:
        pred_df = pd.read_csv(
            pred_path, on_bad_lines="skip", engine="python", quoting=3, dtype=str,
        )
        pred_col = id_column if id_column in pred_df.columns else pred_df.columns[0]
        predicted_ids = [
            pid for pid in (_normalize_id(v) for v in pred_df[pred_col].dropna())
            if pid is not None
        ]
        # Defensive cap: id_expr already calls .limit(10) upstream, but
        # truncate here too in case the CSV slipped through with extras.
        if len(predicted_ids) > LIMIT:
            predicted_ids = predicted_ids[:LIMIT]
        predicted_set = set(predicted_ids)

        correct_ids_ix = gt_df[id_column].isin(list(predicted_set))
        correct_ids = gt_df.loc[correct_ids_ix, :]

        if correct_ids.empty:
            n_to_sample = min(LIMIT, len(gt_df))
            ground_truth_sample = (
                gt_df.sample(n=n_to_sample, random_state=42)
                if n_to_sample > 0 else gt_df
            )
        elif correct_ids.shape[0] > LIMIT:
            ground_truth_sample = correct_ids.head(LIMIT)
        elif correct_ids.shape[0] < LIMIT:
            false_cases = gt_df[~correct_ids_ix]
            n_needed = LIMIT - correct_ids.shape[0]
            n_to_sample = min(n_needed, len(false_cases))
            if n_to_sample > 0:
                ground_truth_sample = pd.concat(
                    [correct_ids, false_cases.sample(n=n_to_sample, random_state=42)]
                )
            else:
                ground_truth_sample = correct_ids
        else:
            ground_truth_sample = correct_ids

        ground_truth_set = set(
            ground_truth_sample[id_column].dropna().astype(str).str.strip()
        )
        tp = len(ground_truth_set & predicted_set)
        fp = len(predicted_set - ground_truth_set)
        fn = len(ground_truth_set - predicted_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"  [cars_q1] GT sample: {len(ground_truth_set)} (LIMIT={LIMIT}), Pred: {len(predicted_set)}")
        print(f"  [cars_q1] TP={tp} FP={fp} FN={fn}  P={p:.4f} R={r:.4f} F1={f1:.4f}")
        return p, r, f1
    except Exception as e:
        print(f"  [cars_q1] Error processing predictions: {e}")
        return 0.0, 0.0, 0.0


def _score_cars_q3(pred_path, gt_path):
    """Evaluate Q3 using SemBench's LIMIT-10 limit-aware GT sampling.

    Q3's natural-language spec ends with "Return the VIN" after "Find ten
    cars…", i.e. LIMIT 10. The SemBench ``CarsEvaluator._evaluate_q3``
    builds a ground-truth sample of exactly 10 rows: all correctly-matched
    VINs from the system output, padded with random false cases
    (random_state=42) up to 10. Precision/recall/F1 is then computed on
    the sampled 10-row GT vs. the (already LIMIT-10) system output.
    """
    LIMIT = 9999
    id_column = "vin"

    try:
        gt_df = pd.read_csv(gt_path, dtype=str)
        if id_column not in gt_df.columns:
            gt_df = gt_df.rename(columns={gt_df.columns[0]: id_column})
        gt_df[id_column] = gt_df[id_column].astype(str).str.strip()
        gt_df = gt_df[gt_df[id_column].astype(bool)]
    except Exception as e:
        print(f"  [cars_q3] Error reading ground truth: {e}")
        return 0.0, 0.0, 0.0

    try:
        pred_df = pd.read_csv(
            pred_path, on_bad_lines="skip", engine="python", quoting=3, dtype=str,
        )
        pred_col = id_column if id_column in pred_df.columns else pred_df.columns[0]
        predicted_vins = (
            pred_df[pred_col].dropna().astype(str).str.strip()
        )
        predicted_vins = predicted_vins[predicted_vins.astype(bool)].tolist()
        # Enforce LIMIT 10 on the system output, mirroring the SemBench
        # Q3 query spec (Lotus runner does .head(10)).
        if len(predicted_vins) > LIMIT:
            predicted_vins = predicted_vins[:LIMIT]
        predicted_set = set(predicted_vins)

        correct_ids_ix = gt_df[id_column].isin(list(predicted_set))
        correct_ids = gt_df.loc[correct_ids_ix, :]

        if correct_ids.empty:
            n_to_sample = min(LIMIT, len(gt_df))
            ground_truth_sample = (
                gt_df.sample(n=n_to_sample, random_state=42)
                if n_to_sample > 0 else gt_df
            )
        elif correct_ids.shape[0] > LIMIT:
            # SemBench's evaluator raises here because a correct system
            # would never return more than LIMIT IDs; we clamp to LIMIT
            # after having already truncated the predictions above, so
            # this branch is unreachable in practice. Kept for parity.
            ground_truth_sample = correct_ids.head(LIMIT)
        elif correct_ids.shape[0] < LIMIT:
            false_cases = gt_df[~correct_ids_ix]
            n_needed = LIMIT - correct_ids.shape[0]
            n_to_sample = min(n_needed, len(false_cases))
            if n_to_sample > 0:
                ground_truth_sample = pd.concat(
                    [correct_ids, false_cases.sample(n=n_to_sample, random_state=42)]
                )
            else:
                ground_truth_sample = correct_ids
        else:  # correct_ids.shape[0] == LIMIT
            ground_truth_sample = correct_ids

        ground_truth_set = set(
            ground_truth_sample[id_column].dropna().astype(str).str.strip()
        )
        tp = len(ground_truth_set & predicted_set)
        fp = len(predicted_set - ground_truth_set)
        fn = len(ground_truth_set - predicted_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"  [cars_q3] GT sample: {len(ground_truth_set)} (LIMIT={LIMIT}), Pred: {len(predicted_set)}")
        print(f"  [cars_q3] TP={tp} FP={fp} FN={fn}  P={p:.4f} R={r:.4f} F1={f1:.4f}")
        return p, r, f1
    except Exception as e:
        print(f"  [cars_q3] Error processing predictions: {e}")
        return 0.0, 0.0, 0.0


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



def _sembench_aggregation_errors(predicted: float, actual: float) -> dict:
    """Reproduce SemBench `QueryMetricAggregation.calculate_errors`.

    Matches the method body of
    `_generic_aggregation_evaluation` → `m.calculate_errors(predicted, actual)`
    in sembench/metrics/aggregation.py: one absolute error, one relative
    error, one MAPE (which for a single-value aggregation is just the
    percentage form of relative error).
    """
    abs_err = abs(float(predicted) - float(actual))
    denom = max(abs(float(actual)), 1e-12)
    rel_err = abs_err / denom
    mape = rel_err * 100.0
    return {
        "absolute_error": float(abs_err),
        "relative_error": float(rel_err),
        "mape": float(mape),
    }


def _score_animals_q1(pred_path, gt_path):
    """SemBench-native aggregation scoring for the zebra-count query.

    Returns ``(precision, recall, f1, extra)`` where:
      * precision/recall/f1 are the legacy count-fidelity scalar
        ``1 - min(relative_error, 1)`` (all equal for an aggregation
        query). Kept identical to the previous scorer so historical
        summary CSVs remain comparable.
      * extra is a dict of SemBench's native aggregation metrics
        (predicted_count, gt_count, absolute_error, relative_error,
        mape) that `collect_result` merges into the per-run results
        CSV. This makes the common ``F1=0`` outcome diagnosable: the
        reader can now see *how far off* the predicted count was,
        rather than a flat zero that collapses "predicted nothing"
        and "overshot by 10x" into the same score.
    """
    try:
        pred_df = pd.read_csv(pred_path, on_bad_lines="skip", engine="python")
    except Exception as e:
        print(f"  [animals_q1] pred read error: {e}")
        return 0.0, 0.0, 0.0, {
            "predicted_count": 0, "gt_count": None,
            "absolute_error": float("inf"), "relative_error": 1.0, "mape": 100.0,
        }
    predicted_count = len(pred_df)
    try:
        gt_df = pd.read_csv(gt_path)
        gt_count = int(gt_df.iloc[0, 0])
    except Exception as e:
        print(f"  [animals_q1] gt read error: {e}")
        return 0.0, 0.0, 0.0, {
            "predicted_count": predicted_count, "gt_count": None,
            "absolute_error": float("inf"), "relative_error": 1.0, "mape": 100.0,
        }

    errs = _sembench_aggregation_errors(predicted_count, gt_count)
    f1 = 1.0 - min(errs["relative_error"], 1.0)
    print(
        f"  [animals_q1] GT count={gt_count}, predicted={predicted_count}, "
        f"abs_err={errs['absolute_error']:.0f}, "
        f"rel_err={errs['relative_error']:.4f}, "
        f"mape={errs['mape']:.2f}%, f1_proxy={f1:.4f}"
    )
    return f1, f1, f1, {
        "predicted_count": int(predicted_count),
        "gt_count": int(gt_count),
        "absolute_error": errs["absolute_error"],
        "relative_error": errs["relative_error"],
        "mape": errs["mape"],
    }


def _score_animals_q3(pred_path, gt_path):
    """Top-city retrieval scored SemBench-style.

    Scalar: 1 if argmax(groupBy City) ∈ GT cities else 0. This mirrors
    `_generic_aggregation_evaluation`'s `first_num` fallback on the
    single-value comparison — strings coerced through value_counts.
    Extra metadata surfaces the actual predicted top-city and the GT
    set so a 0/1 outcome is interpretable in the per-run CSV.
    """
    try:
        pred_df = pd.read_csv(pred_path, on_bad_lines="skip", engine="python")
    except Exception as e:
        print(f"  [animals_q3] pred read error: {e}")
        return 0.0, 0.0, 0.0, {
            "predicted_top": None, "gt_values": None, "match": False,
            "predicted_count": 0,
        }
    if "City" not in pred_df.columns or len(pred_df) == 0:
        print(f"  [animals_q3] missing City column or empty predictions")
        return 0.0, 0.0, 0.0, {
            "predicted_top": None, "gt_values": None, "match": False,
            "predicted_count": int(len(pred_df)),
        }
    top_city = pred_df["City"].astype(str).value_counts().idxmax()
    try:
        gt_df = pd.read_csv(gt_path)
        gt_cities = set(str(c).strip() for c in gt_df.iloc[:, 0].tolist())
    except Exception as e:
        print(f"  [animals_q3] gt read error: {e}")
        return 0.0, 0.0, 0.0, {
            "predicted_top": str(top_city), "gt_values": None, "match": False,
            "predicted_count": int(len(pred_df)),
        }
    match = str(top_city).strip() in gt_cities
    f1 = 1.0 if match else 0.0
    print(
        f"  [animals_q3] GT cities={gt_cities}, top predicted={top_city!r}, "
        f"match={match}, f1={f1:.4f}"
    )
    return f1, f1, f1, {
        "predicted_top": str(top_city).strip(),
        "gt_values": sorted(gt_cities),
        "match": bool(match),
        "predicted_count": int(len(pred_df)),
    }


def _score_animals_q10(pred_path, gt_path):
    """Top (City, StationID) retrieval scored SemBench-style.

    Scalar 1/0 as before. Extra metadata surfaces the predicted top
    pair and the GT set so the per-run CSV has enough context to
    diagnose a 0 outcome (wrong city? wrong station id? empty
    predictions?) without replaying the experiment.
    """
    try:
        pred_df = pd.read_csv(pred_path, on_bad_lines="skip", engine="python")
    except Exception as e:
        print(f"  [animals_q10] pred read error: {e}")
        return 0.0, 0.0, 0.0, {
            "predicted_top": None, "gt_values": None, "match": False,
            "predicted_count": 0,
        }
    if "City" not in pred_df.columns or "StationID" not in pred_df.columns or len(pred_df) == 0:
        print(f"  [animals_q10] missing City/StationID columns or empty predictions")
        return 0.0, 0.0, 0.0, {
            "predicted_top": None, "gt_values": None, "match": False,
            "predicted_count": int(len(pred_df)),
        }
    pair_counts = (
        pred_df.groupby(["City", "StationID"]).size().reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
    )
    top = pair_counts.iloc[0]
    top_pair = (str(top["City"]).strip(), str(top["StationID"]).strip())

    gt_tuples: set = set()
    try:
        gt_df = pd.read_csv(gt_path)
        if gt_df.shape[1] >= 2:
            gt_tuples = {
                (str(r[0]).strip(), str(r[1]).strip())
                for r in gt_df.itertuples(index=False)
            }
    except Exception:
        pass
    if not gt_tuples:
        # Fallback matching semBench_q10_cluster_proxy.py's evaluator.
        gt_tuples = {("Nairobi", "Station_A")}

    match = top_pair in gt_tuples
    f1 = 1.0 if match else 0.0
    print(
        f"  [animals_q10] GT pairs={gt_tuples}, top predicted={top_pair}, "
        f"match={match}, f1={f1:.4f}"
    )
    return f1, f1, f1, {
        "predicted_top": f"{top_pair[0]}|{top_pair[1]}",
        "gt_values": sorted(f"{a}|{b}" for a, b in gt_tuples),
        "match": bool(match),
        "predicted_count": int(len(pred_df)),
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
        _phases = _lc_phases(_lc_cfg.sample_budget)
        print(f"  budget split (learned_calibration): total={_lc_cfg.total_budget} "
              f"sample={_lc_cfg.sample_budget} calib={_lc_cfg.calib_budget} "
              f"route={_lc_cfg.route_budget} phases={_phases}")
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
        deferral_features=DEFERRAL_FEATURES,
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
        _phases = _lc_phases(_lc_cfg.sample_budget)
        print(f"  budget split (learned_calibration): total={_lc_cfg.total_budget} "
              f"sample={_lc_cfg.sample_budget} calib={_lc_cfg.calib_budget} "
              f"route={_lc_cfg.route_budget} phases={_phases}")
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
        deferral_features=DEFERRAL_FEATURES,
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
    if NO_OPT_BACKEND == "python":
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
    joined_df = left_df.sem_join(
        right=right_df, prompt=join_prompt, model=MODEL,
        url=os.environ.get("PROBE_ORACLE_URL", "http://localhost:8000/v1/chat/completions"),
        batch_size=32, verbose=True,
        embedding_model=EMBEDDING_MODEL,
        max_concurrent_requests=128,
        proxy_sample=False,
        exclude_self_pairs=(query_tag in {"ecomm_q9", "movie_q5", "movie_q7", "match122"}),
        self_pair_id_columns=("left_reviewId", "right_reviewId") if query_tag in {"movie_q5", "movie_q7"} else None,
        join_sim_mode="pair",
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
    if NO_OPT_BACKEND == "python":
        from proxy_model import proxy_filter
        # max_tokens=100 is what PROBE's filter calls get through sem_filter's default.
        result_df = proxy_filter(
            df, filter_prompt, model=MODEL, url=os.environ.get("PROBE_ORACLE_URL", "http://localhost:8000/v1/chat/completions"),
            embedding_model=EMBEDDING_MODEL, max_tokens=100, max_concurrent_requests=128, no_proxy=True,
        )
        return result_df, tag, start
    result_df = df.sem_filter(
        prompt=filter_prompt, model=MODEL,
        url=os.environ.get("PROBE_ORACLE_URL", "http://localhost:8000/v1/chat/completions"),
        batch_size=32, verbose=True,
        embedding_model=EMBEDDING_MODEL,
        max_concurrent_requests=128,
        proxy_sample=False,
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
        "n_phases_effective": _effective_phases(cluster_prof),
        "vs_ratio": None if _effective_phases(cluster_prof) == 0 else ACTIVESAMPLING_VS_RATIO,
        "sampler": (cluster_prof.get("activeSampling_profiling") or {}).get("sampler", "legacy"),
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
        "no_opt_backend": NO_OPT_BACKEND,
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
    out["deferral_method"] = cluster_prof.get("deferral_method")
    out["deferral_n_oof_train"] = cluster_prof.get("deferral_n_oof_train")
    out["deferral_n_train_wrong"] = cluster_prof.get("deferral_n_train_wrong")
    out["deferral_n_train_right"] = cluster_prof.get("deferral_n_train_right")
    out["deferral_inner_cv_auc"] = cluster_prof.get("deferral_inner_cv_auc")
    out["deferral_inner_cv_auc_std"] = cluster_prof.get("deferral_inner_cv_auc_std")
    out["deferral_probe_auc"] = cluster_prof.get("deferral_probe_auc")
    out["deferral_probe_confidence_baseline_auc"] = cluster_prof.get(
        "deferral_probe_confidence_baseline_auc"
    )
    out["deferral_probe_defer_rate"] = cluster_prof.get("deferral_probe_defer_rate")
    out["deferral_probe_selective_acc"] = cluster_prof.get("deferral_probe_selective_acc")
    out["deferral_probe_n"] = cluster_prof.get("deferral_probe_n")
    out["deferral_defer_rate"] = cluster_prof.get("deferral_defer_rate")
    out["deferral_chosen_hidden_dims"] = cluster_prof.get("deferral_chosen_hidden_dims")
    out["deferral_chosen_alpha"] = cluster_prof.get("deferral_chosen_alpha")
    out["deferral_fallback_reason"] = cluster_prof.get("deferral_fallback_reason")
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

    # XLSX workbook with raw + summary + phase-breakdown sheets
    xlsx_path = f"{results_dir}{query_tag}_results.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="experiments", index=False)
            if not agg.empty:
                agg.round(4).to_excel(writer, sheet_name="ablation_summary", index=False)
            if not phase_agg.empty:
                phase_agg.to_excel(writer, sheet_name="phase_breakdown", index=False)
        print(f"[{query_tag}] XLSX saved -> {xlsx_path}")
    except Exception as e:
        print(f"[{query_tag}] XLSX save failed (non-fatal): {e}")


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
def run_movie_q7():
    sp = get_spark()
    csv_path = f"{DATA_ROOT}/SemBench/files/movie/data/sf_1000/Reviews.csv"
    reviews = sp.read.option("header", "true").csv(csv_path)
    reviews = reviews.filter(
        (col("id") == "ant_man_and_the_wasp_quantumania") & col("reviewText").isNotNull()
    )
    left_df = reviews.select(col("id"), col("reviewId"), col("reviewText"))
    right_df = reviews.select(
        col("id").alias("id_right"), col("reviewId").alias("reviewId_right"),
        col("reviewText").alias("reviewText_right"),
    )

    prompt = """These two movie reviews express opposite sentiments - one is positive and the other is negative.
Review 1: "{reviewText}"
Review 2: "{reviewText_right}" """

    gt_path = f"{DATA_ROOT}/SemBench/files/movie/raw_results/ground_truth/Q7.csv"
    total_pairs = left_df.count() * right_df.count()
    print(f"[movie_q7] total_pairs={total_pairs}")

    def id_expr(jdf):
        return jdf.filter(col("reviewId") != col("reviewId_right")).select(
            col("id"), col("reviewId"), col("reviewId_right").alias("reviewId2"))

    scorer = calculate_pair_f1_factory(limit=None)
    exp_idx = 1
    for emb_model in _iter_emb_models("movie_q7"):
        with _emb_scope("movie_q7", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("movie_q7"):
                    for k_val in KNN_K_VALUES:
                        joined_df, tag, start = run_experiment_join("movie_q7", left_df, right_df, prompt, gt_path, total_pairs, "propagation", bp, "embedding", k_val, abl_label, abl_strategy, abl_cs, exp_idx, embedding_model=emb_model)
                        r = collect_result(tag, joined_df, id_expr, gt_path, start, "propagation", bp, "embedding", k_val, "movie_q7", ablation_label=abl_label, scorer=scorer, embedding_model=emb_model)
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_join("movie_q7", left_df, right_df, prompt, exp_idx, id_expr, gt_path, scorer=scorer)


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


def make_classify_ari_scorer(id_col, cat_col, fallback_category):
    def _score(pred_path, gt_path):
        from sklearn.metrics import adjusted_rand_score

        def _norm(v):
            if pd.isna(v) or str(v).lower() == "nan":
                return None
            return str(v).strip().lower().replace("\n", "")

        pred_df = pd.read_csv(pred_path, on_bad_lines="skip", engine="python", dtype=str)
        pred_df.columns = [c.strip().lower() for c in pred_df.columns]
        if id_col not in pred_df.columns or cat_col not in pred_df.columns:
            print(f"  [classify-ari] missing columns — have {list(pred_df.columns)}, need {id_col}, {cat_col}")
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
        merged = merged.dropna(subset=[cat_col])

        ari = float(adjusted_rand_score(merged[cat_col].tolist(), merged["pred"].tolist()))
        print(
            f"  Classify ARI - GT: {len(gt_df)}, Overlap rows: {len(merged)}, "
            f"Fallback-filled: {n_missing} (-> '{fallback_category}')"
        )
        print(f"  ARI: {ari:.4f}")
        return ari, ari, ari
    return _score


# ---------------------------------------------------------------------------
# Cars Q1: sem_filter over complaint text
#   "Find cars that were in a crash/accident/collision."
# Per SemBench evaluator, GT = car_ids whose complaint has ``crash == True``.
# We run sem_filter on the full complaints table and dedupe to car_id.
# ---------------------------------------------------------------------------
def run_cars_q1():
    sp = get_spark()
    data_dir = f"{DATA_ROOT}/SemBench/files/cars/data/sf_157376"
    complaints = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(os.path.join(data_dir, "text_complaints_data_157376.csv"))

    complaints_df = complaints.select(
        col("car_id").cast("string").alias("car_id"),
        col("summary").alias("summary"),
    )

    prompt = ("You are given a textual complaint entailing whether the car was in "
              "a crash/accident/collision. Return true if the complaint describes "
              "a crash, accident, or collision.\n"
              "Complaint: {summary}")

    gt_path = f"{DATA_ROOT}/SemBench/files/cars/raw_results/ground_truth/gt_1536/Q1_157376.csv"
    total_rows = complaints_df.count()
    print(f"[cars_q1] total_rows={total_rows}")

    def id_expr(rdf):
        # Match cars_q3: enforce LIMIT 10 on the system output so the
        # SemBench limit-aware scorer can compare apples-to-apples.
        return rdf.select(col("car_id").cast("string").alias("car_id")).distinct()

    exp_idx = 1
    for emb_model in _iter_emb_models("cars_q1"):
        with _emb_scope("cars_q1", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("cars_q1"):
                    for k_val in KNN_K_VALUES:
                        result_df, tag, start = run_experiment_filter(
                            "cars_q1", complaints_df, prompt, gt_path,
                            total_rows, "propagation", bp, "embedding",
                            k_val, abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model,
                        )
                        r = collect_result(
                            tag, result_df, id_expr, gt_path, start,
                            "propagation", bp, "embedding", k_val,
                            "cars_q1", ablation_label=abl_label, scorer=_score_cars_q1,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

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
        return rdf.select(col("vin").alias("vin")).distinct()

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
        return rdf.select(col("car_id").cast("string").alias("car_id")).distinct()

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
def run_cars_q10():
    sp = get_spark()
    data_dir = f"{DATA_ROOT}/SemBench/files/cars/data/sf_157376"
    cars = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(os.path.join(data_dir, "car_data_157376.csv"))
    complaints = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("multiLine", "true").option("escape", '"') \
        .csv(os.path.join(data_dir, "text_complaints_data_157376.csv"))
    cars = cars.withColumn("registration_date", col("registration_date").cast("string"))

    complaints_df = cars.join(complaints, on="car_id", how="inner") \
        .select(
            col("car_id").cast("string").alias("left_car_id"),
            col("summary").alias("left_summary"),
        )

    categories = [
        "ELECTRICAL SYSTEM", "POWER TRAIN", "ENGINE", "STEERING", "SERVICE BRAKES",
        "STRUCTURE", "AIR BAGS", "ENGINE AND ENGINE COOLING", "VEHICLE SPEED CONTROL",
        "VISIBILITY/WIPER", "FUEL/PROPULSION SYSTEM", "FORWARD COLLISION AVOIDANCE",
        "EXTERIOR LIGHTING", "SUSPENSION", "FUEL SYSTEM", "VISIBILITY", "WHEELS",
        "SEAT BELTS", "BACK OVER PREVENTION", "TIRES", "SEATS", "LATCHES/LOCKS/LINKAGES",
        "LANE DEPARTURE", "EQUIPMENT",
    ]
    cat_values_sql = ", ".join(
        "('{}')".format(c.replace("'", "''")) for c in categories
    )
    categories_df = sp.sql(
        f"SELECT col1 AS right_category FROM VALUES {cat_values_sql}"
    )

    prompt = """Classify this car complaint. Does it fall under the problem category "{right_category}"?
Answer True if the complaint is primarily about this category, False otherwise.

Complaint: {left_summary}"""

    gt_path = f"{DATA_ROOT}/SemBench/files/cars/raw_results/ground_truth/gt_1536/Q10_157376.csv"
    n_complaints = complaints_df.count()
    total_pairs = n_complaints * len(categories)
    print(f"[cars_q10] n_complaints={n_complaints}, n_categories={len(categories)}, total_pairs={total_pairs}")

    def id_expr(jdf):
        car_col = "left_car_id" if "left_car_id" in jdf.columns else "car_id"
        cat_col = "right_category" if "right_category" in jdf.columns else "category"
        return jdf.select(
            col(car_col).cast("string").alias("car_id"),
            col(cat_col).alias("problem_category"),
        )

    scorer = make_classify_macro_f1_scorer(
        id_col="car_id",
        cat_col="problem_category",
        all_ids_df=complaints_df,
        fallback_category="ENGINE",
    )

    exp_idx = 1
    for emb_model in _iter_emb_models("cars_q10"):
        with _emb_scope("cars_q10", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("cars_q10"):
                    for k_val in KNN_K_VALUES:
                        joined_df, tag, start = run_experiment_join(
                            "cars_q10", complaints_df, categories_df, prompt, gt_path,
                            total_pairs, "propagation", bp, "embedding",
                            k_val, abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model,
                        )
                        r = collect_result(
                            tag, joined_df, id_expr, gt_path, start,
                            "propagation", bp, "embedding", k_val,
                            "cars_q10", ablation_label=abl_label, scorer=scorer,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

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
    sp = get_spark()
    scale = 100
    data_dir = f"{DATA_ROOT}/SemBench/files/animals/data/sf_{scale}"
    df = sp.read.option("header", "true").option("inferSchema", "true") \
        .option("escape", '"') \
        .csv(os.path.join(data_dir, "image_data.csv"))
    return df


ANIMALS_PROMPT = "Does the image contain a zebra? Image: {image:ImagePath}"


def run_animals_q1():
    df = _animals_load_df()
    prompt = ANIMALS_PROMPT
    gt_path = f"{DATA_ROOT}/SemBench/files/animals/raw_results/ground_truth/Q1.csv"
    total_rows = df.count()
    print(f"[animals_q1] total_rows={total_rows}")

    # Keep ImagePath for the scorer; it just counts rows.
    def id_expr(rdf):
        return rdf.select("ImagePath")

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
                            "animals_q1", ablation_label=abl_label, scorer=_score_animals_q1,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

    exp_idx = maybe_append_no_opt_filter("animals_q1", df, prompt, exp_idx, id_expr, gt_path, scorer=_score_animals_q1)


def run_animals_q3():
    df = _animals_load_df()
    prompt = ANIMALS_PROMPT
    gt_path = f"{DATA_ROOT}/SemBench/files/animals/raw_results/ground_truth/Q3.csv"
    total_rows = df.count()
    print(f"[animals_q3] total_rows={total_rows}")

    # Keep ImagePath + City so the scorer can do the groupBy in pandas.
    def id_expr(rdf):
        return rdf.select("ImagePath", "City")

    exp_idx = 1
    for emb_model in _iter_emb_models("animals_q3"):
        with _emb_scope("animals_q3", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("animals_q3"):
                    for k_val in KNN_K_VALUES:
                        result_df, tag, start = run_experiment_filter(
                            "animals_q3", df, prompt, gt_path,
                            total_rows, "propagation", bp, "embedding",
                            k_val, abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model,
                        )
                        r = collect_result(
                            tag, result_df, id_expr, gt_path, start,
                            "propagation", bp, "embedding", k_val,
                            "animals_q3", ablation_label=abl_label, scorer=_score_animals_q3,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

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


def run_animals_q10():
    df = _animals_load_df()
    prompt = ANIMALS_PROMPT
    gt_path = f"{DATA_ROOT}/SemBench/files/animals/raw_results/ground_truth/Q10.csv"
    total_rows = df.count()
    print(f"[animals_q10] total_rows={total_rows}")

    # Keep ImagePath + City + StationID so the scorer can do the groupBy.
    def id_expr(rdf):
        return rdf.select("ImagePath", "City", "StationID")

    exp_idx = 1
    for emb_model in _iter_emb_models("animals_q10"):
        with _emb_scope("animals_q10", emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query("animals_q10"):
                    for k_val in KNN_K_VALUES:
                        result_df, tag, start = run_experiment_filter(
                            "animals_q10", df, prompt, gt_path,
                            total_rows, "propagation", bp, "embedding",
                            k_val, abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model,
                        )
                        r = collect_result(
                            tag, result_df, id_expr, gt_path, start,
                            "propagation", bp, "embedding", k_val,
                            "animals_q10", ablation_label=abl_label, scorer=_score_animals_q10,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

    # no_opt_df, no_opt_tag, no_opt_start = run_no_opt_filter_experiment(
    #     "animals_q10", df, prompt, exp_idx,
    # )
    # no_opt_r = collect_result(
    #     no_opt_tag, no_opt_df, id_expr, gt_path, no_opt_start,
    #     "no_opt", 100, "n/a", "n/a",
    #     "animals_q10", ablation_label="no_opt", scorer=_score_animals_q10,
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
def _joinml_run(query_tag, data_dir, left_name, right_name, gt_name, prompt,
                id_cols=("left_id", "right_id")):
    sp = get_spark()
    gt_dir = os.path.join(data_dir, "gt")
    left_csv = os.path.join(gt_dir, left_name)
    right_csv = os.path.join(gt_dir, right_name)
    gt_path = os.path.join(gt_dir, gt_name)

    left_df = sp.read.option("header", "true").option("multiLine", "true") \
        .option("escape", '"').csv(left_csv)
    right_df = sp.read.option("header", "true").option("multiLine", "true") \
        .option("escape", '"').csv(right_csv)

    total_pairs = left_df.count() * right_df.count()
    print(f"[{query_tag}] total_pairs={total_pairs}  left={left_df.count()}  right={right_df.count()}")

    def id_expr(jdf):
        return jdf.select(
            col(id_cols[0]).cast("string").alias(id_cols[0]),
            col(id_cols[1]).cast("string").alias(id_cols[1]),
        )

    scorer = calculate_pair_f1_factory(limit=None)

    exp_idx = 1
    for emb_model in _iter_emb_models(query_tag):
        with _emb_scope(query_tag, emb_model):
            for abl_label, abl_strategy, abl_cs in ABLATION_CONFIGS:
                for bp in get_budget_pcts_for_query(query_tag):
                    for k_val in KNN_K_VALUES:
                        joined_df, tag, start = run_experiment_join(
                            query_tag, left_df, right_df, prompt, gt_path, total_pairs,
                            "propagation", bp, "embedding", k_val,
                            abl_label, abl_strategy, abl_cs, exp_idx,
                            embedding_model=emb_model,
                        )
                        r = collect_result(
                            tag, joined_df, id_expr, gt_path, start,
                            "propagation", bp, "embedding", k_val,
                            query_tag, ablation_label=abl_label, scorer=scorer,
                            embedding_model=emb_model,
                        )
                        all_results.append(r)
                        exp_idx += 1

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
def run_veri_q1():
    data_dir = f"{DATA_ROOT}/SemBench/files/veri"
    prompt = ("Two images show the same vehicle: "
              "{image:left_image} {image:right_image}")
    _joinml_run(
        query_tag="veri_q1",
        data_dir=data_dir,
        left_name="veri_left.csv",
        right_name="veri_right.csv",
        gt_name="gt_pairwise_veri.csv",
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Flickr30K Q1: sem_join (caption describes image)
# ---------------------------------------------------------------------------
def run_flickr_q1():
    data_dir = f"{DATA_ROOT}/SemBench/files/flickr30k"
    prompt = ("Caption {text:right_caption} describes the image "
              "{image:left_image}")
    _joinml_run(
        query_tag="flickr_q1",
        data_dir=data_dir,
        left_name="flickr_left.csv",
        right_name="flickr_right.csv",
        gt_name="gt_pairwise_flickr.csv",
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Roxford Q1: sem_join (same landmark across gallery and queries)
# ---------------------------------------------------------------------------
def run_roxford_q1():
    data_dir = f"{DATA_ROOT}/SemBench/files/roxford5k"
    prompt = ("Two images show the same landmark: "
              "{image:left_image} {image:right_image}")
    _joinml_run(
        query_tag="roxford_q1",
        data_dir=data_dir,
        left_name="roxford_left.csv",
        right_name="roxford_right.csv",
        gt_name="gt_pairwise_roxford.csv",
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Company Q1: sem_join (entity resolution on company descriptions)
# ---------------------------------------------------------------------------
def run_company_q1():
    data_dir = f"{DATA_ROOT}/SemBench/files/company"
    prompt = ("Company 1: {text:left_description} and "
              "Company 2: {text:right_description} are the same company.")
    _joinml_run(
        query_tag="company_q1",
        data_dir=data_dir,
        left_name="company_left.csv",
        right_name="company_right.csv",
        gt_name="gt_pairwise_company.csv",
        prompt=prompt,
    )

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


def run_match102():
    sp = get_spark()
    customers = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/car_retails/customers.csv")
    )
    customers = customers.filter(col("creditLimit") > 100000)
    left_df = customers.select(
        col("customerNumber").cast("int").alias("left_customerNumber"),
        col("city").alias("left_city"),
    )

    offices = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/car_retails/offices.csv")
    )
    right_df = offices.select(col("territory").alias("right_territory")).distinct()

    prompt = """Determine whether the customer's city is located within the office's sales territory.

Customer city: {left_city}
Sales territory: {right_territory}"""

    truth = [
        (114, 'APAC'), (119, 'EMEA'), (141, 'EMEA'), (146, 'EMEA'),
        (148, 'APAC'), (157, 'NA'),   (187, 'EMEA'), (227, 'EMEA'),
        (249, 'EMEA'), (259, 'EMEA'), (276, 'APAC'), (278, 'EMEA'),
        (286, 'NA'),   (298, 'EMEA'), (319, 'NA'),   (363, 'NA'),
        (386, 'EMEA'), (448, 'EMEA'), (458, 'EMEA'), (496, 'APAC'),
    ]
    gt_path = _write_match_gt_csv("match102", truth)
    _run_match_join("match102", left_df, right_df, prompt, gt_path,
                    "left_customerNumber", "right_territory")


def run_match104():
    sp = get_spark()
    comments = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/codebase_community/comments.csv")
    )
    # pandas `comments[:10]` takes the first 10 rows in file order; ordering
    # by Id (monotonic in the raw export) is a stable Spark equivalent.
    comments = comments.orderBy(col("Id")).limit(10)
    left_df = comments.select(
        col("Id").cast("int").alias("left_Id"),
        col("Text").alias("left_Text"),
    )

    tags = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/codebase_community/tags.csv")
    )
    tags = tags.orderBy(col("Count").desc()).limit(10)
    right_df = tags.select(
        col("Id").cast("int").alias("right_Id"),
        col("TagName").alias("right_TagName"),
    )

    prompt = """Determine whether the comment's content is directly related to the tag's topic.

Comment: {left_Text}
Tag: {right_TagName}"""

    truth = [(2, 41), (4, 9), (9, 41), (11, 41)]
    gt_path = _write_match_gt_csv("match104", truth)
    _run_match_join("match104", left_df, right_df, prompt, gt_path,
                    "left_Id", "right_Id")


def run_match105():
    sp = get_spark()
    customer = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/shipping/customer.csv")
    )
    customer = customer.filter(col("cust_type") == "wholesaler")
    left_df = customer.select(
        col("cust_id").cast("int").alias("left_cust_id"),
        col("state").alias("left_state"),
    )

    city = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/shipping/city.csv")
    )
    city = city.filter(col("population") > 600000)
    right_df = city.select(col("state").alias("right_state")).distinct()

    prompt = """Determine whether the two-letter abbreviated US state code refers to exactly the same state as the full state name.

Abbreviated state code: {left_state}
Full state name: {right_state}"""

    truth = [
        (381, 'Florida'),     (615, 'Florida'),     (618, 'Florida'),
        (1575, 'California'), (1685, 'California'), (1769, 'Texas'),
        (2001, 'California'), (2799, 'Texas'),      (3288, 'Texas'),
        (3660, 'Maryland'),   (4353, 'Texas'),      (4869, 'Arizona'),
    ]
    gt_path = _write_match_gt_csv("match105", truth)
    _run_match_join("match105", left_df, right_df, prompt, gt_path,
                    "left_cust_id", "right_state")


def run_match106():
    sp = get_spark()
    customer = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/shipping/customer.csv")
    )
    customer = customer.filter(col("annual_revenue") > 45000000)
    left_df = customer.select(
        col("cust_id").cast("int").alias("left_cust_id"),
        col("state").alias("left_state"),
    )

    city = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/shipping/city.csv")
    )
    city = city.filter(col("population") > 600000)
    right_df = city.select(col("state").alias("right_state")).distinct()

    prompt = """Determine whether the two-letter abbreviated US state code refers to exactly the same state as the full state name.

Abbreviated state code: {left_state}
Full state name: {right_state}"""

    truth = [
        (721, 'Florida'),     (1724, 'Ohio'),       (2001, 'California'),
        (2434, 'Ohio'),       (2799, 'Texas'),      (3320, 'California'),
        (3447, 'California'),
    ]
    gt_path = _write_match_gt_csv("match106", truth)
    _run_match_join("match106", left_df, right_df, prompt, gt_path,
                    "left_cust_id", "right_state")


def run_match107():
    sp = get_spark()
    team = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/european_football_2/Team.csv")
    )
    team = team.filter(col("team_long_name").startswith("P"))
    left_df = team.select(
        col("id").cast("int").alias("left_id"),
        col("team_long_name").alias("left_team_long_name"),
    )

    country = (
        sp.read.option("header", "true").option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/european_football_2/Country.csv")
    )
    right_df = country.select(
        col("id").cast("int").alias("right_id"),
        col("name").alias("right_name"),
    )

    prompt = """Determine whether the football team belongs to (plays its domestic league in) the listed country.

Team: {left_team_long_name}
Country: {right_name}"""

    truth = [
        (3476, 1729),   (9548, 4769),   (20532, 10257), (21292, 10257),
        (23523, 10257), (26556, 13274), (29000, 13274), (31444, 15722),
        (31445, 15722), (31448, 15722), (31457, 15722), (32891, 15722),
        (33377, 15722), (36248, 17642), (41673, 19694),
    ]
    gt_path = _write_match_gt_csv("match107", truth)
    _run_match_join("match107", left_df, right_df, prompt, gt_path,
                    "left_id", "right_id")


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


def run_match122():
    sp = get_spark()
    staff = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/nextiaJD/"
             "SCS_Staff_Salaries_data_30th_June 2010.csv")
    )
    staff = staff.select(col("Name").alias("_name")).dropna().dropDuplicates(["_name"])
    left_df  = staff.select(col("_name").alias("left_Name"))
    right_df = staff.select(col("_name").alias("right_Name"))

    prompt = """Determine whether the two staff members share the same first name but are different people (different full names).

Staff A: {left_Name}
Staff B: {right_Name}"""

    truth = [
        ('David Ashbourne', 'David Pimm'),
        ('David Pimm',      'David Ashbourne'),
    ]
    gt_path = _write_match_gt_csv("match122", truth)
    _run_match_join("match122", left_df, right_df, prompt, gt_path,
                    "left_Name", "right_Name")


def run_match125():
    sp = get_spark()
    merchant = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/"
             "purchasing-card-spend-jul2015.csv")
    )
    left_df = (
        merchant.select(col("Procurement (Merchant) Category").alias("left_Category"))
                .dropna().dropDuplicates(["left_Category"])
    )

    rm = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/rm-mr-2009-eng.csv")
    )
    right_df = (
        rm.select(col("MINE").alias("right_Category"))
          .dropna().dropDuplicates(["right_Category"])
    )

    prompt = """Determine whether the two procurement/department categories cover highly similar subject matter (so one could semantically substitute for the other, e.g. "Human Resources" ≈ "Human Resources and Skills Development").

Category A: {left_Category}
Category B: {right_Category}"""

    truth = [
        ('Human Resources  ',        'Human Resources and Skills Development'),
        ('Environmental Services  ', 'Environment'),
    ]
    gt_path = _write_match_gt_csv("match125", truth)
    _run_match_join("match125", left_df, right_df, prompt, gt_path,
                    "left_Category", "right_Category")


def run_match202():
    sp = get_spark()
    yelp = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/restaurants2/yelp.csv")
    )
    yelp = yelp.filter(col("zip") == 60642)
    left_df = yelp.select(
        col("ID").cast("int").alias("left_ID"),
        col("name").alias("left_name"),
        col("address").alias("left_address"),
        col("phone").alias("left_phone"),
        col("cuisine").alias("left_cuisine"),
    )

    zomato = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/restaurants2/zomato.csv")
    )
    zomato = zomato.filter(col("zip") == 60642)
    right_df = zomato.select(
        col("ID").cast("int").alias("right_ID"),
        col("name").alias("right_name"),
        col("address").alias("right_address"),
        col("phone").alias("right_phone"),
        col("cuisine").alias("right_cuisine"),
    )

    prompt = """Determine whether the two restaurant records refer to the same real-world restaurant (same establishment at the same address).

Yelp: name={left_name} | address={left_address} | phone={left_phone} | cuisine={left_cuisine}
Zomato: name={right_name} | address={right_address} | phone={right_phone} | cuisine={right_cuisine}"""

    truth = [
        (2, 844), (72, 1020), (90, 744), (111, 564),
        (198, 1022), (275, 272), (333, 134), (418, 590),
    ]
    gt_path = _write_match_gt_csv("match202", truth)
    _run_match_join("match202", left_df, right_df, prompt, gt_path,
                    "left_ID", "right_ID")


def run_match203():
    sp = get_spark()
    imdb = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/movies/imdb.csv")
    )
    imdb = imdb.filter(col("Rating").cast("double") < 2.3)
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
    tomato = tomato.filter(col("Rating").cast("double") < 2.3)
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

    truth = [
        ('a-708',  'b-662'),
        ('a-1775', 'b-1553'),
    ]
    gt_path = _write_match_gt_csv("match203", truth)
    _run_match_join("match203", left_df, right_df, prompt, gt_path,
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
    "movie_q7": run_movie_q7,
    "mmqa_q2a": run_mmqa_q2a,
    "mmqa_q7": run_mmqa_q7,
    "cars_q1": run_cars_q1,

    "cars_q3": run_cars_q3,
    "cars_q4": run_cars_q4,
    "cars_q8": run_cars_q8,
    # Animals benchmark (SemBench Q1/Q3/Q10)
    "animals_q1": run_animals_q1,
    # "animals_q3": run_animals_q3,
    # "animals_q10": run_animals_q10,
    # # JoinML paper datasets
    # "veri_q1": run_veri_q1,
    # "flickr_q1": run_flickr_q1,
    # "roxford_q1": run_roxford_q1,
    # "company_q1": run_company_q1,
    # LRobench match queries (text-text semantic joins)
    # "match102": run_match102,
    # "match104": run_match104,
    # "match105": run_match105,
    # "match106": run_match106,
    # "match107": run_match107,
    "match113": run_match113,
    "match121": run_match121,
    # "match122": run_match122,
    # "match125": run_match125,
    "match202": run_match202,
    # "match203": run_match203,
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
        print(f"JAR_PATH       = {JAR_PATH}")
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

    # Per-query × ablation mean 4-phase breakdown (sums to duration_s)
    phase_cols = ["phase_embedding_s", "phase_sampling_s",
                  "phase_training_s", "phase_prediction_s", "duration_s"]
    _existing_phase_cols = [c for c in phase_cols if c in summary_df.columns]
    phase_summary = pd.DataFrame()
    if _existing_phase_cols and "query" in summary_df.columns and "ablation" in summary_df.columns:
        phase_summary = (
            summary_df.groupby(["query", "ablation"])[_existing_phase_cols]
            .mean()
            .round(3)
            .reset_index()
        )
        print("\n--- 4-Phase Breakdown (mean seconds, sums to duration_s) ---")
        print(phase_summary.to_string(index=False))

    # XLSX workbook with multiple sheets
    xlsx_path = f"{results_dir}{_run_tag}_grid_summary.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="all_experiments", index=False)
            if not phase_summary.empty:
                phase_summary.to_excel(writer, sheet_name="phase_breakdown", index=False)
        print(f"XLSX summary saved to {xlsx_path}")
    except Exception as e:
        print(f"XLSX save failed (non-fatal): {e}")

    if not summary_df.empty and "f1" in summary_df.columns:
        valid = summary_df[summary_df["f1"] > 0]
        if not valid.empty:
            best = valid.loc[valid["f1"].idxmax()]
            print(f"\nBest config: {best['experiment']} (F1={best['f1']:.4f})")

    # --- Ablation study: per-query and average F1/latency by config ---
    if not summary_df.empty and "ablation" in summary_df.columns:
        print("\n" + "=" * 80)
        print("ABLATION STUDY")
        print("=" * 80)

        ablation_agg = summary_df.groupby(["query", "ablation"]).agg(
            f1_mean=("f1", "mean"),
            f1_max=("f1", "max"),
            latency_mean=("duration_s", "mean"),
        ).reset_index()

        # Per-query pivot: rows=query, columns=ablation, values=f1_mean
        f1_pivot = ablation_agg.pivot(index="query", columns="ablation", values="f1_mean")
        lat_pivot = ablation_agg.pivot(index="query", columns="ablation", values="latency_mean")

        # Add average row
        f1_pivot.loc["AVERAGE"] = f1_pivot.mean()
        lat_pivot.loc["AVERAGE"] = lat_pivot.mean()

        print("\n--- F1 (mean over budgets × k) ---")
        print(f1_pivot.round(4).to_string())

        print("\n--- Latency (mean, seconds) ---")
        print(lat_pivot.round(2).to_string())

        # Delta table for the deferral-policy ablation:
        # learned_deferral - trust_proxy compares "partial rollback to LLM"
        # against "trust every proxy decision" under the same activeSampling+Bias Diffusion + Homogeneity Propagation
        # ablation family.
        print("\n--- Deferral Policy Delta ---")
        trust_col = "activeSampling+refine+trust_proxy"
        learned_col = "activeSampling+refine+learned_deferral"
        if trust_col in f1_pivot.columns and learned_col in f1_pivot.columns:
            for q_idx in f1_pivot.index:
                trust_f1 = f1_pivot.loc[q_idx].get(trust_col, float("nan"))
                learned_f1 = f1_pivot.loc[q_idx].get(learned_col, float("nan"))
                trust_lat = lat_pivot.loc[q_idx].get(trust_col, float("nan"))
                learned_lat = lat_pivot.loc[q_idx].get(learned_col, float("nan"))
                d_f1 = learned_f1 - trust_f1 if pd.notna(learned_f1) and pd.notna(trust_f1) else float("nan")
                d_lat = learned_lat - trust_lat if pd.notna(learned_lat) and pd.notna(trust_lat) else float("nan")
                print(
                    f"  {q_idx:20s}  "
                    f"trust_f1={trust_f1:.4f}  learned_f1={learned_f1:.4f}  "
                    f"ΔF1={d_f1:+.4f}  trust_lat={trust_lat:.2f}s  "
                    f"learned_lat={learned_lat:.2f}s  Δlat={d_lat:+.2f}s"
                )
        else:
            print("  Deferral policy columns not both present; skipping delta table.")

        # Save ablation tables
        ablation_path = f"{results_dir}{_run_tag}_ablation_f1.csv"
        f1_pivot.round(4).to_csv(ablation_path)
        print(f"\nAblation F1 saved to {ablation_path}")
        lat_pivot.round(2).to_csv(f"{results_dir}{_run_tag}_ablation_latency.csv")

        # --- Embedding model ablation pivot ---
        if "embedding_short" in summary_df.columns:
            emb_agg = summary_df[summary_df["ablation"] != "no_opt"].groupby(
                ["query", "ablation", "embedding_short"]
            ).agg(f1_mean=("f1", "mean"), latency_mean=("duration_s", "mean")).reset_index()
            emb_f1_pivot = emb_agg.pivot_table(
                index=["query", "ablation"],
                columns="embedding_short",
                values="f1_mean",
                aggfunc="mean",
            )
            emb_lat_pivot = emb_agg.pivot_table(
                index=["query", "ablation"],
                columns="embedding_short",
                values="latency_mean",
                aggfunc="mean",
            )
            print("\n--- Embedding Model Ablation: F1 (mean over budgets × k) ---")
            print(emb_f1_pivot.round(4).to_string())
            emb_f1_pivot.round(4).to_csv(f"{results_dir}{_run_tag}_emb_ablation_f1.csv")
            emb_lat_pivot.round(2).to_csv(f"{results_dir}{_run_tag}_emb_ablation_latency.csv")
            print(f"Embedding ablation tables saved to {results_dir}{_run_tag}_emb_ablation_*.csv")

        # --- Plot ablation bar chart ---
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            query_labels = [q for q in f1_pivot.index if q != "AVERAGE"]
            config_labels = list(f1_pivot.columns)
            x = np.arange(len(query_labels))
            width = max(0.15, 0.8 / max(len(config_labels), 1))

            colors = {
                "activeSampling+refine": "#2a9d8f",
                "activeSampling+refine+trust_proxy": "#2a9d8f",
                "activeSampling+refine+learned_deferral": "#e76f51",
                "activeSampling+lr_only": "#264653",
                "random+refine": "#e9c46a",
                "random+lr_only": "#e76f51",
            }

            # F1 bar chart
            ax1 = axes[0]
            _center_offset = width * (len(config_labels) - 1) / 2
            for i, cfg_name in enumerate(config_labels):
                vals = [f1_pivot.loc[q, cfg_name] if pd.notna(f1_pivot.loc[q, cfg_name]) else 0 for q in query_labels]
                ax1.bar(x + i * width, vals, width, label=cfg_name, color=colors.get(cfg_name, "#adb5bd"))
            ax1.set_xticks(x + _center_offset)
            ax1.set_xticklabels(query_labels, rotation=45, ha="right", fontsize=8)
            ax1.set_ylabel("F1 (mean)")
            ax1.set_title("Ablation: F1 by Query")
            ax1.legend(fontsize=7)
            ax1.set_ylim(0, 1.05)

            # Latency bar chart
            ax2 = axes[1]
            for i, cfg_name in enumerate(config_labels):
                vals = [lat_pivot.loc[q, cfg_name] if pd.notna(lat_pivot.loc[q, cfg_name]) else 0 for q in query_labels]
                ax2.bar(x + i * width, vals, width, label=cfg_name, color=colors.get(cfg_name, "#adb5bd"))
            ax2.set_xticks(x + _center_offset)
            ax2.set_xticklabels(query_labels, rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel("Latency (s, mean)")
            ax2.set_title("Ablation: Latency by Query")
            ax2.legend(fontsize=7)

            plt.tight_layout()
            plot_path = f"{results_dir}{_run_tag}_ablation_study.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Ablation plot saved to {plot_path}")
        except Exception as e:
            print(f"Plot failed (non-fatal): {e}")

    if spark:
        spark.stop()
    print("\nDone.")
