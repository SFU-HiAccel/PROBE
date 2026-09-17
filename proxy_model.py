import asyncio
import atexit
import collections
import hashlib
import logging
import os
import pickle
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
if not hasattr(np, 'object0'):
    np.object0 = np.object_
import pandas as pd
from openai import AsyncOpenAI, OpenAI
from pyspark.sql import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

logger = logging.getLogger("proxy_model")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PROXY MODEL] %(message)s"))
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Profiling support — stores per-phase wall-clock seconds for the last run.
# Access via ``get_last_profiling()`` after calling proxy_filter / proxy_join.
# ---------------------------------------------------------------------------

_PROFILING_KEYS = ("embedding_time", "lr_training_time", "vllm_invocation_time", "total_time")


# ---------------------------------------------------------------------------
# Oracle budget tracking — limits total vLLM calls during proxy training.
# When calls exceed budget_size the training loop stops immediately.
# ---------------------------------------------------------------------------

def _new_budget_tracker(budget_size: Optional[int] = None) -> Optional[dict]:
    """Create a budget tracker dict, or None if no limit."""
    if budget_size is None or budget_size <= 0:
        return None
    return {"used": 0, "limit": budget_size}


def _budget_exhausted(tracker: Optional[dict]) -> bool:
    """Return True if the oracle budget has been reached."""
    if tracker is None:
        return False
    return tracker["used"] >= tracker["limit"]


def _budget_remaining(tracker: Optional[dict]) -> int:
    """Return remaining budget (capped at a large int when no tracker)."""
    if tracker is None:
        return 2**31
    return max(0, tracker["limit"] - tracker["used"])


def _new_profiling() -> dict:
    """Return a fresh profiling accumulator."""
    return {k: 0.0 for k in _PROFILING_KEYS}


# Mutable container — updated in-place so all importers see the same data.
_last_profiling: dict = _new_profiling()


def _set_last_profiling(prof: dict) -> None:
    """Copy *prof* values into the shared ``_last_profiling`` dict in-place."""
    _last_profiling.clear()
    _last_profiling.update(prof)


def get_last_profiling() -> dict:
    """Return profiling dict from the most recent proxy_filter/proxy_join call."""
    return dict(_last_profiling)


# ---------------------------------------------------------------------------
# LLM token accounting — accumulates prompt/completion/total tokens across
# every oracle call made during a proxy run. Thread-safe (the async path
# launches many concurrent requests via a semaphore).
# ---------------------------------------------------------------------------

_TOKEN_COUNTER_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "llm_requests")
_TOKEN_COUNTER_LOCK = threading.Lock()
_TOKEN_COUNTER: Dict[str, int] = {k: 0 for k in _TOKEN_COUNTER_KEYS}


def reset_token_usage() -> None:
    """Zero out the token accumulator. Call before a pipeline run."""
    with _TOKEN_COUNTER_LOCK:
        for k in _TOKEN_COUNTER_KEYS:
            _TOKEN_COUNTER[k] = 0


def get_last_token_usage() -> Dict[str, int]:
    """Return a snapshot of the cumulative LLM token counters."""
    with _TOKEN_COUNTER_LOCK:
        return dict(_TOKEN_COUNTER)


def _record_llm_usage(usage) -> None:
    """Accumulate prompt/completion/total tokens from an LLM response.

    *usage* may be a pydantic object (OpenAI SDK) or a plain dict (raw httpx).
    Missing fields are treated as 0; ``total`` falls back to ``prompt + completion``.
    """
    if usage is None:
        with _TOKEN_COUNTER_LOCK:
            _TOKEN_COUNTER["llm_requests"] += 1
        return
    if hasattr(usage, "prompt_tokens"):
        p = int(getattr(usage, "prompt_tokens", 0) or 0)
        c = int(getattr(usage, "completion_tokens", 0) or 0)
        t = int(getattr(usage, "total_tokens", 0) or (p + c))
    else:
        try:
            p = int(usage.get("prompt_tokens", 0) or 0)
            c = int(usage.get("completion_tokens", 0) or 0)
            t = int(usage.get("total_tokens", 0) or (p + c))
        except AttributeError:
            p = c = t = 0
    with _TOKEN_COUNTER_LOCK:
        _TOKEN_COUNTER["prompt_tokens"] += p
        _TOKEN_COUNTER["completion_tokens"] += c
        _TOKEN_COUNTER["total_tokens"] += t
        _TOKEN_COUNTER["llm_requests"] += 1


def _log_profiling(prof: dict) -> None:
    """Log a profiling summary.

    Four-stage breakdown (sums to total_time minus "Other overhead"):
      1. Embedding  — SigLIP/CLIP/... encoding of rows or pairs
      2. Labeling   — vLLM / LLM oracle calls on the sampled subset
      3. Training   — LR fit on the labeled sample
      4. Predicting — LR predict on the full universe + Bias Diffusion + Homogeneity Propagation
                      graph diffusion (graph norm + residual + smoothing)

    KNN-graph construction, data prep, and Spark overhead are folded into
    "Other overhead" (the residual between Total and the four stages).
    """
    total = prof["total_time"]
    emb = prof["embedding_time"]
    lr = prof["lr_training_time"]
    vllm = prof["vllm_invocation_time"]
    cp = prof.get("cluster_profiling") or {}
    graph_prof = (
        cp.get("activeSampling_profiling")
        or cp.get("random_profiling")
        or {}
    )
    refine_prof = cp.get("refinement_profiling") or {}

    is_cs = bool(refine_prof)

    logger.info("=== Profiling Summary ===")

    def _pct(x: float) -> str:
        return f"({100 * x / max(total, 1e-9):5.1f}%)"

    if is_cs:
        lr_fit_time = float(refine_prof.get("lr_fit_time", 0.0))
        lr_predict_time = float(refine_prof.get("lr_predict_time", 0.0))
        bias_diffusion_time = float(refine_prof.get("bias_diffusion_time", 0.0))
        homogeneity_propagation_time = float(refine_prof.get("homogeneity_propagation_time", 0.0))
        graph_norm_time = float(refine_prof.get("graph_norm_time", 0.0))

        # If lr_fit/lr_predict aren't separately tracked, fall back to the
        # combined lr_training_time (put all of it in training).
        if lr_fit_time == 0.0 and lr_predict_time == 0.0:
            lr_fit_time = float(refine_prof.get("lr_training_time", 0.0))

        stage_embedding = emb
        stage_labeling = vllm
        stage_training = lr_fit_time
        stage_predicting = (
            lr_predict_time + graph_norm_time + bias_diffusion_time + homogeneity_propagation_time
        )

        accounted = stage_embedding + stage_labeling + stage_training + stage_predicting
        other = max(0.0, total - accounted)

        logger.info(f"  [1] Embedding:            {stage_embedding:8.2f}s  {_pct(stage_embedding)}")
        logger.info(f"  [2] Labeling:             {stage_labeling:8.2f}s  {_pct(stage_labeling)}")
        logger.info(f"  [3] Training:             {stage_training:8.2f}s  {_pct(stage_training)}")
        logger.info(f"  [4] Predicting:           {stage_predicting:8.2f}s  {_pct(stage_predicting)}")
        logger.info(f"      LR predict:           {lr_predict_time:8.2f}s")
        logger.info(f"      Graph normalisation:  {graph_norm_time:8.2f}s")
        logger.info(
            f"      Bias Diffusion (residual):   {bias_diffusion_time:8.2f}s  "
            f"iters={refine_prof.get('bias_diffusion_iters', '-')}"
        )
        logger.info(
            f"      Homogeneity Propagation (prediction):  {homogeneity_propagation_time:8.2f}s  "
            f"iters={refine_prof.get('homogeneity_propagation_iters', '-')}"
        )
        logger.info(f"  Other overhead:           {other:8.2f}s  {_pct(other)}")
        logger.info(
            f"  Labeled samples: {cp.get('n_labeled', '?')} "
            f"(pos: {cp.get('n_pos_labeled', '?')})"
        )
    else:
        # LR-only (no Bias Diffusion + Homogeneity Propagation). Training and Predicting are derived from
        # lr_fit_time / lr_predict_time if available, else the combined
        # lr_training_time is attributed to Training.
        lr_fit_time = float(prof.get("lr_fit_time", 0.0))
        lr_predict_time = float(prof.get("lr_predict_time", 0.0))
        if lr_fit_time == 0.0 and lr_predict_time == 0.0:
            lr_fit_time = lr

        stage_embedding = emb
        stage_labeling = vllm
        stage_training = lr_fit_time
        stage_predicting = lr_predict_time

        accounted = stage_embedding + stage_labeling + stage_training + stage_predicting
        other = max(0.0, total - accounted)

        logger.info(f"  [1] Embedding:            {stage_embedding:8.2f}s  {_pct(stage_embedding)}")
        logger.info(f"  [2] Labeling:             {stage_labeling:8.2f}s  {_pct(stage_labeling)}")
        logger.info(f"  [3] Training:             {stage_training:8.2f}s  {_pct(stage_training)}")
        logger.info(f"  [4] Predicting:           {stage_predicting:8.2f}s  {_pct(stage_predicting)}")
        logger.info(f"  Other overhead:           {other:8.2f}s  {_pct(other)}")
        if cp:
            stratum_time = cp.get("stratum_time", 0)
            logger.info("  --- Proxy Breakdown ---")
            logger.info(f"    Stratum Division time: {stratum_time:8.2f}s")
            logger.info(f"    Sampling:              {cp.get('sampling_time', 0):8.2f}s")
            logger.info(f"    Label propagation:     {cp.get('propagation_time', 0):8.2f}s")
            logger.info(f"    Labeled samples:    {cp.get('n_labeled', '?')} (pos: {cp.get('n_pos_labeled', '?')})")
    logger.info(f"  Total:                    {total:8.2f}s")

FILTER_SYSTEM_PROMPT = (
    "You are a helpful data analyst. You will receive data and a query. "
    "Answer ONLY 'True' or 'False'."
)

JOIN_SYSTEM_PROMPT = (
    "You are a helpful data analyst. You will receive data from two tables and a query. "
    "Answer ONLY 'True' or 'False'."
)

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_TYPED_PLACEHOLDER_RE = re.compile(r"\{(\w+):(\w+)\}")


def _pandas_to_spark(spark, pdf: pd.DataFrame, schema=None) -> "DataFrame":
    """Convert pandas DataFrame to Spark DataFrame without triggering iteritems().

    PySpark 3.3.1's createDataFrame uses pandas iteritems() which was removed
    in pandas 2.0+. Workaround: convert to list of Row objects.
    """
    if len(pdf) == 0:
        from pyspark.sql.types import StructType
        s = schema or StructType()
        jschema = spark._jsparkSession.parseDataType(s.json())
        empty = spark._jvm.java.util.ArrayList()
        jdf = spark._jsparkSession.createDataFrame(empty, jschema)
        from pyspark.sql import DataFrame as SparkDF
        return SparkDF(jdf, spark)

    # Prefer Spark's pandas conversion path first. This avoids Python-RDD
    # pickling in environments where PySpark 3.3.x + Python 3.12 can fail with:
    # "_pickle.PicklingError: Could not serialize object: IndexError..."
    #
    # PySpark <= 3.3 expects pandas.DataFrame.iteritems(), which was removed
    # in pandas 2.x. Add a local compatibility alias when needed.
    if not hasattr(pd.DataFrame, "iteritems"):
        pd.DataFrame.iteritems = pd.DataFrame.items

    try:
        converted = pdf.map(_to_python_native)
        return spark.createDataFrame(converted, schema=schema)
    except Exception as e:
        logger.warning(
            f"createDataFrame(pandas) failed ({e}); falling back to Row conversion."
        )

    from pyspark.sql import Row
    columns = pdf.columns.tolist()
    rows = [
        Row(**{c: _to_python_native(row[c]) for c in columns})
        for _, row in pdf.iterrows()
    ]
    return spark.createDataFrame(rows, schema=schema)


def _to_python_native(val):
    """Convert numpy/pandas types to Python native types for Spark Row."""
    if pd.isna(val):
        return None
    if hasattr(val, 'item'):  # numpy scalar
        return val.item()
    return val


def _extract_placeholders(prompt: str) -> List[str]:
    return _PLACEHOLDER_RE.findall(prompt)


def _extract_typed_placeholders(prompt: str) -> dict:
    """Extract typed placeholders like {image:col} or {text:col}.

    Returns dict mapping column_name -> column_type.
    E.g. "{image:image_filepath}" -> {"image_filepath": "image"}
    """
    matches = _TYPED_PLACEHOLDER_RE.findall(prompt)
    return {col_name: col_type for col_type, col_name in matches}


def _get_image_columns(prompt: str) -> List[str]:
    """Return column names that are typed as 'image' in the prompt."""
    typed = _extract_typed_placeholders(prompt)
    return [col for col, ctype in typed.items() if ctype == "image"]


def _get_text_columns(prompt: str) -> List[str]:
    """Return column names that are typed as 'text' in the prompt."""
    typed = _extract_typed_placeholders(prompt)
    return [col for col, ctype in typed.items() if ctype == "text"]


def _get_all_columns(prompt: str) -> List[str]:
    """Return all column names from both simple {col} and typed {type:col} placeholders."""
    simple = _PLACEHOLDER_RE.findall(prompt)
    typed = _extract_typed_placeholders(prompt)
    return simple + list(typed.keys())


def _replace_placeholder(text: str, col: str, val: str) -> str:
    """Replace both {col} and {type:col} forms in text."""
    # Replace typed form {type:col} first (e.g. {text:Track}, {image:path})
    text = re.sub(r"\{\w+:" + re.escape(col) + r"\}", lambda _: val, text)
    # Replace simple form {col}
    text = text.replace(f"{{{col}}}", val)
    return text


def build_row_text(row: dict, prompt: str, columns: List[str]) -> str:
    text = prompt
    for col in columns:
        val = str(row.get(col, ""))
        text = _replace_placeholder(text, col, val)
    return text


def build_pair_text(
    left_row: dict,
    right_row: dict,
    prompt: str,
    left_columns: List[str],
    right_columns: List[str],
) -> str:
    text = prompt
    for col in left_columns:
        val = str(left_row.get(col, ""))
        text = _replace_placeholder(text, col, val)
    for col in right_columns:
        val = str(right_row.get(col, ""))
        text = _replace_placeholder(text, col, val)
    return text


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}


def _is_image_like(value: str) -> bool:
    """Check if a value looks like an image path or URL (mirrors Rust is_image_like)."""
    lower = value.lower().strip()
    if any(lower.endswith(ext) for ext in _IMAGE_EXTENSIONS):
        return True
    return lower.startswith(("file://", "http://", "https://", "data:image/"))


def _normalize_image_url(raw: str) -> str:
    """Ensure image paths have a file:// prefix for vLLM."""
    raw = raw.strip()
    if raw.startswith(("file://", "http://", "https://", "data:")):
        return raw
    return f"file://{raw}"


def build_row_message(
    row: dict, prompt: str, columns: List[str], image_columns: List[str],
) -> Union[str, list]:
    """Build an OpenAI-format user message content for a row.

    For text-only rows, returns a plain string.
    For multimodal rows (with image columns), returns a list of content blocks:
      [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "file://..."}}]

    This matches the message format expected by vLLM vision-language models.
    """
    if not image_columns:
        return build_row_text(row, prompt, columns)

    # Build text prompt with image placeholders stripped
    text_prompt = prompt
    media_contents = []
    for col in columns:
        val = str(row.get(col, ""))
        if col in image_columns:
            if val and val != "NULL":
                media_contents.append({
                    "type": "image_url",
                    "image_url": {"url": _normalize_image_url(val)},
                })
            # Strip the placeholder from text
            text_prompt = re.sub(r"\{\w+:" + re.escape(col) + r"\}", "", text_prompt)
            text_prompt = text_prompt.replace(f"{{{col}}}", "")
        else:
            text_prompt = _replace_placeholder(text_prompt, col, val)

    content = [{"type": "text", "text": text_prompt.strip()}]
    content.extend(media_contents)
    return content


def build_pair_message(
    left_row: dict,
    right_row: dict,
    prompt: str,
    left_columns: List[str],
    right_columns: List[str],
    image_columns: List[str],
) -> Union[str, list]:
    """Build an OpenAI-format user message content for a pair (join).

    Same as build_row_message but for left+right row pairs.
    """
    if not image_columns:
        return build_pair_text(left_row, right_row, prompt, left_columns, right_columns)

    text_prompt = prompt
    media_contents = []

    for col in left_columns:
        val = str(left_row.get(col, ""))
        if col in image_columns:
            if val and val != "NULL":
                media_contents.append({
                    "type": "image_url",
                    "image_url": {"url": _normalize_image_url(val)},
                })
            text_prompt = re.sub(r"\{\w+:" + re.escape(col) + r"\}", "", text_prompt)
            text_prompt = text_prompt.replace(f"{{{col}}}", "")
        else:
            text_prompt = _replace_placeholder(text_prompt, col, val)

    for col in right_columns:
        val = str(right_row.get(col, ""))
        if col in image_columns:
            if val and val != "NULL":
                media_contents.append({
                    "type": "image_url",
                    "image_url": {"url": _normalize_image_url(val)},
                })
            text_prompt = re.sub(r"\{\w+:" + re.escape(col) + r"\}", "", text_prompt)
            text_prompt = text_prompt.replace(f"{{{col}}}", "")
        else:
            text_prompt = _replace_placeholder(text_prompt, col, val)

    content = [{"type": "text", "text": text_prompt.strip()}]
    content.extend(media_contents)
    return content


def _get_siglip_version(model_path: str) -> Optional[int]:
    """Return SigLIP version (1 or 2) from config, or None if not a SigLIP model."""
    if not os.path.isdir(model_path):
        return None
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        import json
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type", "").lower()
        if model_type == "siglip2":
            return 2
        if "siglip" in model_type:
            return 1
        return None
    except Exception:
        return None


def _is_siglip_model(model_path: str) -> bool:
    """Detect whether a local model directory contains a SigLIP (v1 or v2) model."""
    return _get_siglip_version(model_path) is not None


# Cache for loaded SigLIP models to avoid reloading
_siglip_cache: dict = {}


def release_siglip_cache_entry(model_path: str) -> None:
    """Evict one SigLIP entry and free its GPU memory.

    Called by proxy_model_backends.release_backend so the legacy SigLIP
    helpers don't pin weights on the GPU after the corresponding backend
    has been dropped.
    """
    entry = _siglip_cache.pop(model_path, None)
    if entry is None:
        return
    for key in ("model", "text_forward", "vision_forward", "processor", "tokenizer"):
        entry[key] = None
    import gc
    gc.collect()
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass


def _extract_features_tensor(out):
    import torch as _t
    if isinstance(out, _t.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if isinstance(v, _t.Tensor):
            return v
    last_hidden = getattr(out, "last_hidden_state", None)
    if isinstance(last_hidden, _t.Tensor):
        return last_hidden.mean(dim=1)
    if isinstance(out, dict):
        for key in ("image_embeds", "text_embeds", "pooler_output"):
            v = out.get(key)
            if isinstance(v, _t.Tensor):
                return v
        v = out.get("last_hidden_state")
        if isinstance(v, _t.Tensor):
            return v.mean(dim=1)
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], _t.Tensor):
        return out[0]
    raise TypeError(
        f"SigLIP feature head returned unsupported type {type(out).__name__}; "
        f"expected Tensor or BaseModelOutputWithPooling-like object"
    )


def _siglip_pick_attn_impl(version: int, device: str) -> str:
    """Pick attention backend: prefer flash_attention_2 on Ampere+ when
    installed, fall back to sdpa, else eager. SM 8.0+ for FA2.
    """
    if device != "cuda":
        return "eager"
    import torch
    try:
        cap_major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        cap_major = 0
    if cap_major >= 8:
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            pass
    return "sdpa" if version == 2 else "eager"


def _load_siglip(model_path: str):
    """Load and cache SigLIP/SigLIP2 model, processor, and tokenizer.

    Multi-GPU strategy: **manual sharding** — one full model copy per
    visible GPU, no ``nn.DataParallel``. RTX 6000 Ada (and other Ada
    workstation cards) ship without NVLink, so DP's per-step
    scatter/gather over PCIe is the bottleneck. Sharding splits the input
    list across replicas and runs each chunk independently with zero
    cross-GPU traffic, giving near-linear scaling.

    Override knobs:
      LLMSQL_MULTI_GPU=N  cap device count (0 disables multi-GPU)
      LLMSQL_SIGLIP_ATTN  force attn impl: flash_attention_2 / sdpa / eager
    """
    if model_path in _siglip_cache:
        return _siglip_cache[model_path]

    import os as _os
    import torch
    from torch import nn as _nn
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    version = _get_siglip_version(model_path) or 1
    logger.info(f"Loading SigLIP v{version} model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    attn_impl = _os.environ.get("LLMSQL_SIGLIP_ATTN") or _siglip_pick_attn_impl(version, device)
    load_kwargs: Dict[str, Any] = {}
    if attn_impl != "eager":
        load_kwargs["attn_implementation"] = attn_impl

    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if device == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None

    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    env_cap = _os.environ.get("LLMSQL_MULTI_GPU")
    if env_cap is not None:
        try:
            n_gpus = min(n_gpus, max(0, int(env_cap)))
        except ValueError:
            pass

    class _TextForward(_nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, input_ids, attention_mask=None, **kwargs):
            out = self.base.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            return _extract_features_tensor(out)

    class _VisionForward(_nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, pixel_values, **kwargs):
            out = self.base.get_image_features(pixel_values=pixel_values, **kwargs)
            return _extract_features_tensor(out)

    def _build_replica(dev: str) -> Dict[str, Any]:
        def _attempt(kw: Dict[str, Any]):
            mdl = AutoModel.from_pretrained(model_path, **kw).to(dev).eval()
            # Smoke-test the chosen attention backend with a tiny text
            # forward. transformers can pair attn_implementation=
            # flash_attention_2 with a flash_attn version where
            # process_flash_kwargs_fn ends up None, which throws
            # TypeError on the FIRST forward rather than at load. Failing
            # here lets us cleanly fall back to sdpa.
            if "attn_implementation" in kw:
                with torch.no_grad():
                    probe = tokenizer(["x"], return_tensors="pt").to(dev)
                    mdl.get_text_features(**probe)
            return mdl

        try:
            mdl = _attempt(load_kwargs)
        except (ValueError, ImportError, RuntimeError, TypeError) as exc:
            if "attn_implementation" in load_kwargs:
                logger.info(
                    f"attn={load_kwargs['attn_implementation']} unavailable on {dev} "
                    f"({exc.__class__.__name__}: {exc}); falling back to sdpa/eager"
                )
                fb_kwargs = {k: v for k, v in load_kwargs.items() if k != "attn_implementation"}
                if version == 2:
                    fb_kwargs["attn_implementation"] = "sdpa"
                else:
                    fb_kwargs.pop("attn_implementation", None)
                mdl = _attempt(fb_kwargs)
            else:
                raise
        try:
            mdl = torch.compile(mdl, mode="reduce-overhead")
        except Exception as exc:
            logger.info(f"torch.compile not available on {dev}, eager: {exc}")
        return {
            "model": mdl,
            "text_forward": _TextForward(mdl).to(dev).eval(),
            "vision_forward": _VisionForward(mdl).to(dev).eval(),
            "device": dev,
        }

    replicas: List[Dict[str, Any]] = []
    if n_gpus >= 2:
        for gpu_idx in range(n_gpus):
            replicas.append(_build_replica(f"cuda:{gpu_idx}"))
        logger.info(
            f"SigLIP v{version} sharded across {n_gpus} GPUs (manual replicas, "
            f"no DataParallel — avoids PCIe scatter/gather on no-NVLink cards)"
        )
    else:
        replicas.append(_build_replica(device))

    canonical = replicas[0]["model"]
    # ``torch.compile`` wraps the module; metadata reads still go through
    # the underlying ``_orig_mod`` if compiled, else the model itself.
    config_owner = getattr(canonical, "_orig_mod", canonical)

    entry = {
        "model": config_owner,           # for .config metadata reads
        "replicas": replicas,
        "processor": processor,
        "tokenizer": tokenizer,
        "device": device,
        "amp_dtype": amp_dtype,
        "version": version,
        "n_gpus": len(replicas) if device == "cuda" else 0,
        "attn_impl": attn_impl,
    }
    _siglip_cache[model_path] = entry
    logger.info(
        f"SigLIP v{version} loaded: device={device} amp_dtype={amp_dtype} "
        f"n_replicas={entry['n_gpus']} attn={attn_impl}"
    )
    return entry


def _siglip_shard_dispatch(
    replicas: List[Dict[str, Any]],
    items: List[Any],
    work_fn,
) -> np.ndarray:
    """Split ``items`` across replicas and run ``work_fn(replica, shard)``
    in parallel threads. Each thread is GPU-bound on its own device, so
    the GIL is released inside CUDA kernels — no cross-GPU traffic.
    """
    if len(replicas) == 1 or len(items) == 0:
        return work_fn(replicas[0], items)

    n_shards = min(len(replicas), len(items))
    shard_size = (len(items) + n_shards - 1) // n_shards
    shards = [items[i * shard_size : (i + 1) * shard_size] for i in range(n_shards)]

    with ThreadPoolExecutor(max_workers=n_shards) as pool:
        futures = [pool.submit(work_fn, replicas[i], shards[i]) for i in range(n_shards)]
        results = [f.result() for f in futures]
    return np.concatenate(results, axis=0)


def embed_texts_siglip(texts: List[str], model_path: str, batch_size: int = 512) -> np.ndarray:
    """Embed texts using SigLIP/SigLIP2 text encoder.

    bf16 autocast on GPU, dynamic padding (only pad to longest in-batch),
    inference_mode, and manual cross-GPU sharding when multiple visible.
    """
    import torch

    entry = _load_siglip(model_path)
    replicas = entry["replicas"]
    tokenizer = entry["tokenizer"]
    amp_dtype = entry["amp_dtype"]
    text_dim = entry["model"].config.text_config.hidden_size

    if not texts:
        return np.zeros((0, text_dim), dtype=np.float32)

    def _process_shard(replica: Dict[str, Any], shard_texts: List[str]) -> np.ndarray:
        if not shard_texts:
            return np.zeros((0, text_dim), dtype=np.float32)
        dev = replica["device"]
        device_type = dev.split(":")[0]  # "cuda:0" -> "cuda"
        text_forward = replica["text_forward"]
        out_chunks: List[np.ndarray] = []
        for i in range(0, len(shard_texts), batch_size):
            batch = shard_texts[i : i + batch_size]
            # Dynamic padding: only pad to longest in batch (was max_length=64
            # for every sample regardless of actual length).
            inputs = tokenizer(
                batch, padding=True, truncation=True, max_length=64,
                return_tensors="pt",
            ).to(dev, non_blocking=True)
            with torch.inference_mode():
                if amp_dtype is not None:
                    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                        feats = text_forward(**inputs)
                else:
                    feats = text_forward(**inputs)
                if not torch.is_tensor(feats):
                    if hasattr(feats, "pooler_output") and torch.is_tensor(feats.pooler_output):
                        feats = feats.pooler_output
                    elif isinstance(feats, (tuple, list)) and feats and torch.is_tensor(feats[0]):
                        feats = feats[0]
                    elif hasattr(feats, "last_hidden_state") and torch.is_tensor(feats.last_hidden_state):
                        feats = feats.last_hidden_state[:, 0, :]
                    else:
                        raise TypeError(
                            f"Unexpected SigLIP text output type: {type(feats)!r}"
                        )
                feats = feats.float()
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            out_chunks.append(feats.cpu().numpy())
        return np.concatenate(out_chunks, axis=0).astype(np.float32)

    return _siglip_shard_dispatch(replicas, list(texts), _process_shard)


def _siglip_text_dim(model_path: str) -> int:
    """Return the text embedding dimension for a SigLIP model."""
    entry = _load_siglip(model_path)
    return entry["model"].config.text_config.hidden_size


def _load_image(path: str):
    """Load a single image, returning (image, True) or (None, False)."""
    from PIL import Image
    try:
        return Image.open(path).convert("RGB"), True
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}, using zero embedding")
        return None, False


def embed_images_siglip(image_paths: List[str], model_path: str, batch_size: int = 256) -> np.ndarray:
    """Embed images using SigLIP's vision encoder.

    bf16 autocast, parallel I/O (32-thread pool), inference_mode, and
    manual cross-GPU sharding when multiple visible.
    """
    import torch

    entry = _load_siglip(model_path)
    replicas = entry["replicas"]
    processor = entry["processor"]
    amp_dtype = entry["amp_dtype"]
    vision_dim = entry["model"].config.vision_config.hidden_size

    if not image_paths:
        return np.zeros((0, vision_dim), dtype=np.float32)

    def _process_shard(replica: Dict[str, Any], shard_paths: List[str]) -> np.ndarray:
        if not shard_paths:
            return np.zeros((0, vision_dim), dtype=np.float32)
        dev = replica["device"]
        device_type = dev.split(":")[0]
        vision_forward = replica["vision_forward"]
        out_chunks: List[np.ndarray] = []
        for i in range(0, len(shard_paths), batch_size):
            batch_paths = shard_paths[i : i + batch_size]

            # Parallel PIL decode. PIL releases the GIL on I/O so threads
            # help; for very large batches a process pool would help more,
            # but threads avoid the IPC overhead for typical batches.
            with ThreadPoolExecutor(max_workers=min(32, len(batch_paths))) as pool:
                results = list(pool.map(_load_image, batch_paths))
            images = [img for img, ok in results if ok]
            valid_mask = [ok for _, ok in results]

            if not images:
                out_chunks.append(np.zeros((len(batch_paths), vision_dim), dtype=np.float32))
                continue

            inputs = processor(images=images, return_tensors="pt").to(dev, non_blocking=True)
            with torch.inference_mode():
                if amp_dtype is not None:
                    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                        feats = vision_forward(**inputs)
                else:
                    feats = vision_forward(**inputs)
                feats = feats.float()
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)

            img_embs = feats.cpu().numpy()
            # Reconstruct full batch with zeros for failed images
            result = np.zeros((len(batch_paths), img_embs.shape[1]), dtype=np.float32)
            valid_idx = 0
            for j, valid in enumerate(valid_mask):
                if valid:
                    result[j] = img_embs[valid_idx]
                    valid_idx += 1
            out_chunks.append(result)
        return np.concatenate(out_chunks, axis=0).astype(np.float32)

    return _siglip_shard_dispatch(replicas, list(image_paths), _process_shard)


# ---------------------------------------------------------------------------
# Embedding cache (two layers: in-process LRU + disk)
# ---------------------------------------------------------------------------
# The grid harness calls embed_rows_independent / embed_multimodal_rows with
# the same (records, prompt, columns, embedding_model) tuple once per
# ablation config (activeSampling+refine, activeSampling+lr_only, random+refine, random+lr_only),
# and also once per budget/k_val within a query. For expensive backends
# (gme-Qwen2-VL at 7B, gte-Qwen2-VL at 7B, jina-v4) the embedding step can
# take minutes and dominates the grid cell wall time.
#
# Layer 1 — in-process OrderedDict (LRU, bounded by LLMSQL_EMBED_CACHE_ENTRIES)
# Layer 2 — disk cache at LLMSQL_EMBED_CACHE_DIR (default ~/.cache/llmsql/
#   embeddings). Pickled, atomically written. Survives process restarts so
#   re-running the grid after a crash or across multiple invocations reuses
#   already-computed embeddings.
#
# Env vars:
#   LLMSQL_EMBED_CACHE         master switch (default 1)
#   LLMSQL_EMBED_CACHE_ENTRIES in-memory LRU size (default 32)
#   LLMSQL_EMBED_CACHE_DISK    disk layer on/off (default 1)
#   LLMSQL_EMBED_CACHE_DIR     disk cache directory
#
# Bump LLMSQL_EMBED_CACHE_VERSION to invalidate existing on-disk entries
# without deleting the directory (e.g. after changing the fingerprint
# function or backend behaviour).
# ---------------------------------------------------------------------------

_EMBED_CACHE_VERSION = os.environ.get("LLMSQL_EMBED_CACHE_VERSION", "v1")
_EMBED_CACHE: collections.OrderedDict = collections.OrderedDict()
_EMBED_CACHE_MAX_ENTRIES = int(os.environ.get("LLMSQL_EMBED_CACHE_ENTRIES", "32"))
_EMBED_CACHE_ENABLED = os.environ.get("LLMSQL_EMBED_CACHE", "1").strip() not in ("0", "false", "False")
_EMBED_CACHE_DISK_ENABLED = os.environ.get("LLMSQL_EMBED_CACHE_DISK", "1").strip() not in ("0", "false", "False")
_EMBED_CACHE_DIR = os.environ.get(
    "LLMSQL_EMBED_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "llmsql", "embeddings"),
)


def _records_fingerprint(records: List[dict], columns: List[str]) -> str:
    """Sampled hash of a list of row dicts restricted to `columns`.

    Hashes len(records) plus the first, middle, and last record's values
    for the named columns. Stable for the grid's dataset re-use pattern
    (same records across ablation configs) and cheap even for 10k+ rows.
    """
    h = hashlib.sha256()
    h.update(str(len(records)).encode())
    h.update(b"||")
    if records:
        idxs = sorted({0, len(records) // 2, len(records) - 1})
        for i in idxs:
            row = records[i]
            h.update(f"[{i}]".encode())
            for col in columns:
                h.update(col.encode())
                h.update(b"=")
                h.update(str(row.get(col, "")).encode(errors="replace"))
                h.update(b"|")
    return h.hexdigest()


def _cache_key_to_filename(key: tuple, viz_tag: Optional[str] = None) -> str:
    """Stable filesystem-safe filename for a cache key.

    When viz_tag is provided (e.g. "ecomm_q6", "mmqa_q2a") the filename is
    "{sanitized_viz_tag}_{sha256}" so entries are grep-able by query name
    and a single query's cache can be wiped with `rm <viz_tag>_*.pkl`.
    Without viz_tag we fall back to just the sha256 (old layout).
    """
    h = hashlib.sha256()
    h.update(_EMBED_CACHE_VERSION.encode())
    h.update(b"\x00")
    for part in key:
        h.update(str(part).encode(errors="replace"))
        h.update(b"\x00")
    digest = h.hexdigest()
    if not viz_tag:
        return digest
    safe_tag = re.sub(r"[^\w\-.]", "_", str(viz_tag))[:64]
    return f"{safe_tag}_{digest}"


def _disk_cache_path(key, viz_tag: Optional[str] = None) -> str:
    return os.path.join(
        _EMBED_CACHE_DIR, _cache_key_to_filename(key, viz_tag) + ".pkl"
    )


def _disk_cache_get(key, viz_tag: Optional[str] = None):
    if not _EMBED_CACHE_DISK_ENABLED:
        return None
    path = _disk_cache_path(key, viz_tag)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"[embed_cache] disk load failed for {path}: {e}")
        return None


def _disk_cache_put(key, value, viz_tag: Optional[str] = None) -> None:
    if not _EMBED_CACHE_DISK_ENABLED:
        return
    try:
        os.makedirs(_EMBED_CACHE_DIR, exist_ok=True)
        final_path = _disk_cache_path(key, viz_tag)
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Atomic rename — safe if two workers race on the same key.
        os.replace(tmp_path, final_path)
    except Exception as e:
        logger.warning(f"[embed_cache] disk save failed: {e}")


def _embed_cache_get(key, viz_tag: Optional[str] = None):
    """Two-layer lookup: in-memory first, fall through to disk. On disk hit,
    populate the in-memory layer for cheap subsequent lookups.
    """
    if not _EMBED_CACHE_ENABLED:
        return None
    v = _EMBED_CACHE.get(key)
    if v is not None:
        _EMBED_CACHE.move_to_end(key)
        return v
    v = _disk_cache_get(key, viz_tag)
    if v is not None:
        _EMBED_CACHE[key] = v
        _EMBED_CACHE.move_to_end(key)
        while len(_EMBED_CACHE) > _EMBED_CACHE_MAX_ENTRIES:
            _EMBED_CACHE.popitem(last=False)
        fname = _cache_key_to_filename(key, viz_tag)
        logger.info(f"[embed_cache] DISK HIT {fname[:80]} → rehydrated memory")
    return v


def _embed_cache_put(key, value, viz_tag: Optional[str] = None) -> None:
    if not _EMBED_CACHE_ENABLED:
        return
    _EMBED_CACHE[key] = value
    _EMBED_CACHE.move_to_end(key)
    while len(_EMBED_CACHE) > _EMBED_CACHE_MAX_ENTRIES:
        _EMBED_CACHE.popitem(last=False)
    _disk_cache_put(key, value, viz_tag)


def clear_embedding_cache(disk: bool = False, viz_tag: Optional[str] = None) -> None:
    """Clear cached embeddings.

    * disk=False (default): only wipes the in-process memory cache.
    * disk=True: also deletes on-disk pickles. If viz_tag is provided,
      deletes only files whose name starts with "{viz_tag}_"; otherwise
      removes the whole cache directory.
    """
    if viz_tag is None:
        _EMBED_CACHE.clear()
    else:
        safe_tag = re.sub(r"[^\w\-.]", "_", str(viz_tag))[:64]
        for k in list(_EMBED_CACHE.keys()):
            if safe_tag in str(k):
                _EMBED_CACHE.pop(k, None)

    if disk and os.path.isdir(_EMBED_CACHE_DIR):
        if viz_tag is None:
            import shutil
            shutil.rmtree(_EMBED_CACHE_DIR, ignore_errors=True)
        else:
            safe_tag = re.sub(r"[^\w\-.]", "_", str(viz_tag))[:64]
            prefix = safe_tag + "_"
            for name in os.listdir(_EMBED_CACHE_DIR):
                if name.startswith(prefix) and name.endswith(".pkl"):
                    try:
                        os.remove(os.path.join(_EMBED_CACHE_DIR, name))
                    except OSError:
                        pass


def embedding_cache_info() -> dict:
    """Summary of cache state — useful for debugging from the REPL."""
    disk_entries = 0
    disk_bytes = 0
    by_tag: dict = {}
    if os.path.isdir(_EMBED_CACHE_DIR):
        for name in os.listdir(_EMBED_CACHE_DIR):
            if not name.endswith(".pkl"):
                continue
            path = os.path.join(_EMBED_CACHE_DIR, name)
            try:
                size = os.path.getsize(path)
                disk_bytes += size
                disk_entries += 1
                # Extract viz_tag prefix (everything up to the last "_<sha256>.pkl")
                stem = name[:-4]  # strip ".pkl"
                if "_" in stem:
                    # The sha256 is 64 hex chars; anything before the last
                    # 64-char hex block is the viz_tag.
                    rest = stem
                    # Find the last underscore followed by 64 hex chars
                    m = re.search(r"_[0-9a-f]{64}$", rest)
                    tag = rest[: m.start()] if m else "<untagged>"
                else:
                    tag = "<untagged>"
                by_tag.setdefault(tag, {"entries": 0, "bytes": 0})
                by_tag[tag]["entries"] += 1
                by_tag[tag]["bytes"] += size
            except OSError:
                pass
    return {
        "memory_entries": len(_EMBED_CACHE),
        "memory_max": _EMBED_CACHE_MAX_ENTRIES,
        "disk_entries": disk_entries,
        "disk_bytes": disk_bytes,
        "disk_dir": _EMBED_CACHE_DIR,
        "by_tag": by_tag,
        "enabled": _EMBED_CACHE_ENABLED,
        "disk_enabled": _EMBED_CACHE_DISK_ENABLED,
        "version": _EMBED_CACHE_VERSION,
    }


def embed_multimodal_pairs(
    left_records: List[dict],
    right_records: List[dict],
    pair_indices: List[Tuple[int, int]],
    prompt: str,
    embedding_model: str,
    left_columns: List[str],
    right_columns: List[str],
) -> np.ndarray:
    """Embed pairs using the configured embedding backend.

    For each pair, text columns are encoded with the text encoder and image
    columns are encoded with the vision encoder. The resulting embeddings are
    mean-pooled into a single vector (valid when backend.shared_space is True).
    """
    from proxy_model_backends import get_backend
    backend = get_backend(embedding_model)

    image_cols = set(_get_image_columns(prompt))
    text_cols_typed = set(_get_text_columns(prompt))

    # Classify columns into image and text
    all_cols = left_columns + right_columns
    img_col_names = [c for c in all_cols if c in image_cols]
    txt_col_names = [c for c in all_cols if c not in image_cols]

    parts = []

    # Embed text columns
    if txt_col_names:
        left_txt_cols = [c for c in left_columns if c not in image_cols]
        right_txt_cols = [c for c in right_columns if c not in image_cols]
        texts = []
        for li, ri in pair_indices:
            text_parts = []
            for col in left_txt_cols:
                text_parts.append(str(left_records[li].get(col, "")))
            for col in right_txt_cols:
                text_parts.append(str(right_records[ri].get(col, "")))
            texts.append(" ".join(text_parts))
        parts.append(backend.embed_texts(texts))

    # Embed image columns
    for img_col in img_col_names:
        is_left = img_col in left_columns
        image_paths = []
        for li, ri in pair_indices:
            record = left_records[li] if is_left else right_records[ri]
            image_paths.append(str(record.get(img_col, "")))
        parts.append(backend.embed_images(image_paths))

    if not parts:
        # Fallback: embed pair texts as regular text
        pair_texts = [
            build_pair_text(left_records[i], right_records[j], prompt, left_columns, right_columns)
            for i, j in pair_indices
        ]
        return backend.embed_texts(pair_texts)

    # Mean-pool all modality embeddings into a single vector.
    if len(parts) == 1:
        return parts[0].astype(np.float32)
    stacked = np.stack(parts, axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


def embed_multimodal_rows(
    records: List[dict],
    prompt: str,
    columns: List[str],
    embedding_model: str,
    viz_tag: Optional[str] = None,
) -> np.ndarray:
    """Embed rows using the configured embedding backend (for filter).

    When the backend exposes a native embed_fused(texts, paths) method (e.g.
    gme-Qwen2-VL) AND the row has both text and image columns, we use it to
    get a single joint embedding per row. Otherwise we embed text and image
    sides separately and mean-pool.

    Memoised across grid cells keyed on
    (embedding_model, prompt, columns, records_fingerprint). Controlled by
    LLMSQL_EMBED_CACHE / LLMSQL_EMBED_CACHE_ENTRIES env vars. viz_tag (e.g.
    "ecomm_q13") is not part of the key but is used as a filename prefix on
    disk so entries are grep-able by query name.
    """
    cache_key = (
        "mm_rows", embedding_model, prompt, tuple(columns),
        _records_fingerprint(records, columns),
    )
    cached = _embed_cache_get(cache_key, viz_tag=viz_tag)
    if cached is not None:
        logger.info(
            f"[embed_cache] HIT embed_multimodal_rows "
            f"(tag={viz_tag}, model={os.path.basename(embedding_model)}, "
            f"n={len(records)}, dim={cached.shape[1]})"
        )
        return cached

    from proxy_model_backends import get_backend
    backend = get_backend(embedding_model)

    image_cols = set(_get_image_columns(prompt))
    txt_cols = [c for c in columns if c not in image_cols]
    img_cols_present = [c for c in columns if c in image_cols]

    # Native joint text+image path (GMEQwen2VLBackend): use exactly one text
    # concatenation and one primary image column to pair per row.
    fuse_fn = getattr(backend, "embed_fused", None)
    if callable(fuse_fn) and txt_cols and img_cols_present:
        primary_img = img_cols_present[0]
        texts = [
            " ".join(str(row.get(c, "")) for c in txt_cols) for row in records
        ]
        paths = [str(row.get(primary_img, "")) for row in records]
        embs = fuse_fn(texts, paths)
        # If there are additional image columns, mean-pool them in (rare).
        for extra in img_cols_present[1:]:
            extra_paths = [str(row.get(extra, "")) for row in records]
            embs = (embs + backend.embed_images(extra_paths)) * 0.5
        result = embs.astype(np.float32)
        _embed_cache_put(cache_key, result, viz_tag=viz_tag)
        return result

    # Generic separate-encoder path (SigLIP/CLIP/Jina/GTE-VL).
    parts = []
    if txt_cols:
        texts = [
            " ".join(str(row.get(c, "")) for c in txt_cols) for row in records
        ]
        parts.append(backend.embed_texts(texts))
    for img_col in img_cols_present:
        image_paths = [str(row.get(img_col, "")) for row in records]
        parts.append(backend.embed_images(image_paths))

    if not parts:
        texts = [build_row_text(row, prompt, columns) for row in records]
        result = backend.embed_texts(texts)
    elif len(parts) == 1:
        result = parts[0].astype(np.float32)
    else:
        stacked = np.stack(parts, axis=0)
        result = np.mean(stacked, axis=0).astype(np.float32)

    _embed_cache_put(cache_key, result, viz_tag=viz_tag)
    return result


def embed_rows_independent(
    left_records: List[dict],
    right_records: List[dict],
    prompt: str,
    embedding_model: str,
    left_columns: List[str],
    right_columns: List[str],
    feature_mode: str = "concat",
    viz_tag: Optional[str] = None,
) -> dict:
    """Embed left and right table rows independently (not as pairs).

    Returns a dict with keys:
      - left_embs: np.ndarray (N × d) — combined left embeddings
      - right_embs: np.ndarray (M × d) — combined right embeddings
      - left_text_embs: Optional[np.ndarray] — text-only embeddings for left
      - right_text_embs: Optional[np.ndarray] — text-only embeddings for right
      - left_img_embs: Optional[np.ndarray] — image-only embeddings for left
      - right_img_embs: Optional[np.ndarray] — image-only embeddings for right

    Memoised across grid cells keyed on
    (embedding_model, prompt, left_columns, right_columns, left_fp, right_fp,
     feature_mode). Controlled by LLMSQL_EMBED_CACHE / LLMSQL_EMBED_CACHE_ENTRIES.
    viz_tag (e.g. "movie_q5") prefixes the on-disk filename for grep-able
    per-query cache inspection; it is not part of the key.
    """
    cache_key = (
        "indep_rows", embedding_model, prompt,
        tuple(left_columns), tuple(right_columns), feature_mode,
        _records_fingerprint(left_records, left_columns),
        _records_fingerprint(right_records, right_columns),
    )
    cached = _embed_cache_get(cache_key, viz_tag=viz_tag)
    if cached is not None:
        logger.info(
            f"[embed_cache] HIT embed_rows_independent "
            f"(tag={viz_tag}, model={os.path.basename(embedding_model)}, "
            f"n_left={len(left_records)}, n_right={len(right_records)})"
        )
        return cached

    from proxy_model_backends import get_backend
    backend = get_backend(embedding_model)

    image_cols = set(_get_image_columns(prompt))

    left_txt_cols = [c for c in left_columns if c not in image_cols]
    right_txt_cols = [c for c in right_columns if c not in image_cols]
    left_img_cols = [c for c in left_columns if c in image_cols]
    right_img_cols = [c for c in right_columns if c in image_cols]

    result = {
        "left_text_embs": None, "right_text_embs": None,
        "left_img_embs": None, "right_img_embs": None,
    }

    left_parts, right_parts = [], []

    # Text embeddings
    if left_txt_cols:
        left_texts = [" ".join(str(r.get(c, "")) for c in left_txt_cols) for r in left_records]
        left_text_embs = backend.embed_texts(left_texts)
        result["left_text_embs"] = left_text_embs
        left_parts.append(left_text_embs)
    if right_txt_cols:
        right_texts = [" ".join(str(r.get(c, "")) for c in right_txt_cols) for r in right_records]
        right_text_embs = backend.embed_texts(right_texts)
        result["right_text_embs"] = right_text_embs
        right_parts.append(right_text_embs)

    # Image embeddings (one per image column)
    for img_col in left_img_cols:
        paths = [str(r.get(img_col, "")) for r in left_records]
        embs = backend.embed_images(paths)
        if result["left_img_embs"] is None:
            result["left_img_embs"] = embs
        else:
            # Mean-pool multiple image columns (rare but possible)
            result["left_img_embs"] = (result["left_img_embs"] + embs) / 2.0
        left_parts.append(embs)
    for img_col in right_img_cols:
        paths = [str(r.get(img_col, "")) for r in right_records]
        embs = backend.embed_images(paths)
        if result["right_img_embs"] is None:
            result["right_img_embs"] = embs
        else:
            result["right_img_embs"] = (result["right_img_embs"] + embs) / 2.0
        right_parts.append(embs)

    # Fallback: embed as plain text if no typed columns found
    if not left_parts:
        left_texts = [" ".join(str(r.get(c, "")) for c in left_columns) for r in left_records]
        left_text_embs = backend.embed_texts(left_texts)
        result["left_text_embs"] = left_text_embs
        left_parts.append(left_text_embs)
    if not right_parts:
        right_texts = [" ".join(str(r.get(c, "")) for c in right_columns) for r in right_records]
        right_text_embs = backend.embed_texts(right_texts)
        result["right_text_embs"] = right_text_embs
        right_parts.append(right_text_embs)

    def _combine(parts: List[np.ndarray]) -> np.ndarray:
        if len(parts) == 1:
            return parts[0].astype(np.float32)
        # Mean-pool all modality embeddings into a single vector.
        stacked = np.stack(parts, axis=0)
        return np.mean(stacked, axis=0).astype(np.float32)

    result["left_embs"] = _combine(left_parts)
    result["right_embs"] = _combine(right_parts)

    _embed_cache_put(cache_key, result, viz_tag=viz_tag)
    return result


def topk_prefilter_pairs(
    left_embs: np.ndarray,
    right_embs: np.ndarray,
    k: int,
    left_text_embs: Optional[np.ndarray] = None,
    right_text_embs: Optional[np.ndarray] = None,
    left_img_embs: Optional[np.ndarray] = None,
    right_img_embs: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """Select top-K candidate pairs per left row using cosine similarity.

    Uses the best available modality overlap for cross-table similarity:
    1. If both sides have text embeddings: use text similarity
    2. If both sides have image embeddings: use image similarity
    3. Cross-modal (text vs image): use SigLIP shared-space similarity
    4. Fallback: use the full combined embeddings

    Returns list of (left_idx, right_idx) candidate pairs.
    """
    sim_components = []

    # Text-text similarity
    if left_text_embs is not None and right_text_embs is not None:
        ln = left_text_embs / (np.linalg.norm(left_text_embs, axis=1, keepdims=True) + 1e-8)
        rn = right_text_embs / (np.linalg.norm(right_text_embs, axis=1, keepdims=True) + 1e-8)
        sim_components.append(ln @ rn.T)

    # Image-image similarity
    if left_img_embs is not None and right_img_embs is not None:
        ln = left_img_embs / (np.linalg.norm(left_img_embs, axis=1, keepdims=True) + 1e-8)
        rn = right_img_embs / (np.linalg.norm(right_img_embs, axis=1, keepdims=True) + 1e-8)
        sim_components.append(ln @ rn.T)

    # Cross-modal: text vs image (SigLIP shared space — 768-dim each)
    if not sim_components:
        lt = left_text_embs if left_text_embs is not None else left_img_embs
        rt = right_text_embs if right_text_embs is not None else right_img_embs
        if lt is not None and rt is not None:
            # Ensure same dimensionality for cross-modal (SigLIP: both 768)
            d = min(lt.shape[1], rt.shape[1])
            ln = lt[:, :d] / (np.linalg.norm(lt[:, :d], axis=1, keepdims=True) + 1e-8)
            rn = rt[:, :d] / (np.linalg.norm(rt[:, :d], axis=1, keepdims=True) + 1e-8)
            sim_components.append(ln @ rn.T)

    # Fallback: full embedding similarity
    if not sim_components:
        d = min(left_embs.shape[1], right_embs.shape[1])
        ln = left_embs[:, :d] / (np.linalg.norm(left_embs[:, :d], axis=1, keepdims=True) + 1e-8)
        rn = right_embs[:, :d] / (np.linalg.norm(right_embs[:, :d], axis=1, keepdims=True) + 1e-8)
        sim_components.append(ln @ rn.T)

    # Average similarity across modalities
    sim_matrix = np.mean(sim_components, axis=0)  # N × M

    n_left = len(left_embs)
    n_right = len(right_embs)
    k = min(k, n_right)

    pairs = []
    for i in range(n_left):
        topk_indices = np.argsort(sim_matrix[i])[-k:]
        for j in topk_indices:
            pairs.append((i, int(j)))

    logger.info(
        f"Top-K prefilter: {n_left}×{n_right}={n_left*n_right} → {len(pairs)} candidates (K={k})"
    )
    return pairs


def compute_similarity_features(
    pair_indices: List[Tuple[int, int]],
    left_embs: np.ndarray,
    right_embs: np.ndarray,
    left_text_embs: Optional[np.ndarray] = None,
    right_text_embs: Optional[np.ndarray] = None,
    left_img_embs: Optional[np.ndarray] = None,
    right_img_embs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute cosine similarity features for each pair.

    Returns an array of shape (len(pair_indices), n_features) where n_features
    is 1-3 depending on available modalities:
      - overall cosine similarity (always present)
      - text-text cosine similarity (if both sides have text)
      - image-image cosine similarity (if both sides have images)
    """
    n_pairs = len(pair_indices)
    li_arr = np.array([p[0] for p in pair_indices])
    ri_arr = np.array([p[1] for p in pair_indices])

    def _cos_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity for aligned row pairs."""
        dot = np.sum(a * b, axis=1)
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        return dot / (na * nb + 1e-8)

    features = []

    overall_cos = _cos_batch(left_embs[li_arr], right_embs[ri_arr])
    features.append(overall_cos.reshape(-1, 1))

    # Per-modality similarities
    if left_text_embs is not None and right_text_embs is not None:
        text_cos = _cos_batch(left_text_embs[li_arr], right_text_embs[ri_arr])
        features.append(text_cos.reshape(-1, 1))

    if left_img_embs is not None and right_img_embs is not None:
        img_cos = _cos_batch(left_img_embs[li_arr], right_img_embs[ri_arr])
        features.append(img_cos.reshape(-1, 1))

    return np.concatenate(features, axis=1).astype(np.float32)


def _build_pair_embeddings_from_independent(
    pair_indices: List[Tuple[int, int]],
    left_embs: np.ndarray,
    right_embs: np.ndarray,
) -> np.ndarray:
    """Build D-dim pair features by mean-pooling (left[li], right[ri]).

    Two performance choices vs. the previous version:
      1. Index extraction via ``np.asarray(pair_indices, dtype=np.int64)``
         instead of two ``np.fromiter`` over generators — single C-level
         conversion, ~50× faster for n_pairs ≥ 1M.
      2. GPU offload when CUDA is visible: the gather + fused add+scale
         is memory-bandwidth bound, and on Ada/Hopper the device
         bandwidth (~960 GB/s on RTX 6000 Ada) is ~20× host RAM. Left
         and right are sent to GPU once; we chunk the index array so
         peak VRAM stays small. Falls back transparently to a chunked
         CPU pass when no GPU is available.
    """
    n = len(pair_indices)
    D = int(left_embs.shape[1])
    if n == 0:
        return np.empty((0, D), dtype=np.float32)

    # Vectorized index extraction — a list of 2-tuples maps directly to
    # an (n, 2) ndarray in one C copy.
    pair_arr = np.asarray(pair_indices, dtype=np.int64)
    if pair_arr.ndim != 2 or pair_arr.shape[1] != 2:
        # Defensive fallback for unusual inputs (e.g. ragged sequences).
        li_arr = np.fromiter((p[0] for p in pair_indices), dtype=np.int64, count=n)
        ri_arr = np.fromiter((p[1] for p in pair_indices), dtype=np.int64, count=n)
    else:
        li_arr = np.ascontiguousarray(pair_arr[:, 0])
        ri_arr = np.ascontiguousarray(pair_arr[:, 1])

    # GPU path: keep left/right on device once, stream index chunks
    # through. Threshold guards small inputs where the H2D/D2H round
    # trip would dominate. Set ``LLMSQL_PAIR_EMBED_CPU=1`` to force the
    # CPU path even when CUDA is visible (e.g. when GPUs are reserved
    # for the LLM oracle and you want the pair-embed work off-device).
    _GPU_THRESHOLD = 100_000
    _force_cpu = os.environ.get("LLMSQL_PAIR_EMBED_CPU", "0") == "1"
    if n >= _GPU_THRESHOLD and not _force_cpu:
        try:
            import torch
            if torch.cuda.is_available():
                dev = "cuda:0"
                # Resident on device for the whole pass.
                l_dev = torch.from_numpy(np.ascontiguousarray(left_embs)).to(dev, non_blocking=True)
                r_dev = torch.from_numpy(np.ascontiguousarray(right_embs)).to(dev, non_blocking=True)
                out = np.empty((n, D), dtype=np.float32)
                # Chunk size set so peak VRAM ≈ 2 × chunk × D × 4 bytes.
                # 2M × 768 × 4 = ~6 GB — well within RTX 6000 Ada's 48 GB.
                chunk = max(1, min(n, 2_000_000))
                for s in range(0, n, chunk):
                    e = min(s + chunk, n)
                    li_t = torch.from_numpy(li_arr[s:e]).to(dev, non_blocking=True)
                    ri_t = torch.from_numpy(ri_arr[s:e]).to(dev, non_blocking=True)
                    # In-place to halve VRAM traffic vs (a + b) * 0.5.
                    chunk_out = l_dev.index_select(0, li_t)
                    chunk_out.add_(r_dev.index_select(0, ri_t))
                    chunk_out.mul_(0.5)
                    out[s:e] = chunk_out.cpu().numpy()
                # Free promptly so the next caller doesn't OOM.
                del l_dev, r_dev
                torch.cuda.empty_cache()
                return out
        except Exception as exc:
            logger.warning(
                f"_build_pair_embeddings GPU path failed ({exc}); falling back to CPU"
            )

    # ── CPU path: threaded chunked add into a pre-divided source ──
    #
    # Two CPU optimizations on top of the original chunked loop:
    #
    # (1) Pre-divide left/right by 2 ONCE up front (cheap: their row
    #     count is ≪ n_pairs). Then ``out[s:e] = lh[li] + rh[ri]`` is a
    #     single ufunc with one write to ``out`` per chunk — no
    #     follow-up ``out *= 0.5`` pass. Cuts ``out`` traffic in half.
    #
    # (2) Parallel chunks via ThreadPoolExecutor. ``np.add`` and fancy
    #     indexing release the GIL during the C-level work, so threads
    #     genuinely run concurrently and saturate DRAM bandwidth across
    #     cores. On a 16-core box this is typically 4–8× faster than
    #     single-threaded for memory-bound gather+add.
    out = np.empty((n, D), dtype=np.float32)

    # One-time pre-divide. For typical n_pairs ≫ max(n_left, n_right)
    # this is virtually free (e.g. 10k × 768 × 4 = 30 MB).
    left_half = np.multiply(left_embs, np.float32(0.5))
    right_half = np.multiply(right_embs, np.float32(0.5))

    n_workers = max(1, min(os.cpu_count() or 4, 16))
    # Smaller chunks (200k) → better L3 cache locality and finer-grained
    # work units for the thread pool.
    chunk = max(1, min(n, 200_000))

    def _add_chunk(s: int) -> None:
        e = min(s + chunk, n)
        np.add(
            left_half[li_arr[s:e]],
            right_half[ri_arr[s:e]],
            out=out[s:e],
        )

    if n_workers > 1 and n >= 50_000:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for fut in [pool.submit(_add_chunk, s) for s in range(0, n, chunk)]:
                fut.result()
    else:
        for s in range(0, n, chunk):
            _add_chunk(s)

    return out


def embed_texts(texts: List[str], embedding_model: str) -> np.ndarray:
    from proxy_model_backends import get_backend
    return get_backend(embedding_model).embed_texts(texts)


def embed_query(query: str, embedding_model: str, task: Optional[str] = None) -> np.ndarray:
    """Embed a user's predicate/prompt as a QUERY (not a document).

    For asymmetric retrieval models (e.g. gte-Qwen2-7B-instruct) this applies
    the model's query-side instruction prefix. For symmetric models (SigLIP,
    CLIP, Jina-CLIP, GTE-Qwen2-VL, SentenceTransformer, OpenAI) the default
    implementation just calls embed_texts, so the behaviour is unchanged.

    Returns a (1, D) float32 array matching embed_texts([query]).
    """
    from proxy_model_backends import get_backend
    backend = get_backend(embedding_model)
    fn = getattr(backend, "embed_query", None)
    if callable(fn):
        try:
            return fn(query, task=task)
        except TypeError:
            return fn(query)
    return backend.embed_texts([query])


def _call_llm_single(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: Union[str, list],
    max_tokens: int,
    temperature: float,
) -> int:
    """Call LLM for a single row (sync fallback). user_content can be a
    plain string or a list of content blocks (for multimodal messages)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        **_labeller_sampling_overrides(),
        extra_body=_labeller_extra_body(),
    )
    _record_llm_usage(getattr(resp, "usage", None))
    content = (resp.choices[0].message.content or "").strip().lower()
    if "true" in content:
        return 1
    return 0


# --- Expert-call retry policy --------------------------------------------------
# A failed expert request used to become label 0 immediately. Over ~1M concurrent
# requests a transient client-side error (httpx timeout / HTTP-2 stream reset,
# often with an empty message) is expected, and one such error aborted a
# multi-hour budget-F1 run. Transient failures are retried with exponential
# backoff; only persistent failures fall back to label 0, and those still log
# "defaulting to 0" so run watchers can stop the experiment.
LLM_MAX_ATTEMPTS = max(1, int(os.environ.get("PROBE_LLM_MAX_ATTEMPTS", "4")))
LLM_RETRY_BASE_DELAY_S = float(os.environ.get("PROBE_LLM_RETRY_BASE_DELAY_S", "1.0"))
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUS or status_code >= 500


async def _retry_backoff(attempt: int) -> None:
    # Read the module attribute at call time so tests can set it to 0.
    delay = LLM_RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
    if delay > 0:
        await asyncio.sleep(delay)


def _describe_exception(e: BaseException) -> str:
    msg = str(e)
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


async def _call_llm_single_async(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_content: Union[str, list],
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> int:
    """Call LLM for a single row via async I/O with concurrency control."""
    async with semaphore:
        for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **_labeller_sampling_overrides(),
                    extra_body=_labeller_extra_body(),
                )
                _record_llm_usage(getattr(resp, "usage", None))
                content = (resp.choices[0].message.content or "").strip().lower()
                return 1 if "true" in content else 0
            except Exception as e:
                if attempt < LLM_MAX_ATTEMPTS:
                    logger.warning(f"Async expert request attempt {attempt}/{LLM_MAX_ATTEMPTS} errored "
                                   f"({_describe_exception(e)}); retrying")
                    await _retry_backoff(attempt)
                    continue
                logger.warning(f"Async LLM call failed after {attempt} attempts: "
                               f"{_describe_exception(e)}, defaulting to 0")
                return 0
        return 0


async def _call_llm_single_raw_async(
    http_client,
    url: str,
    model: str,
    system_prompt: str,
    user_content: Union[str, list],
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> int:
    """Raw httpx POST, bypasses AsyncOpenAI + Pydantic for lower per-call overhead.

    Mirrors the Rust baseline in
    sparkai/HetuDataU/.../query_utils.rs::execute_query_multimodal_async,
    which sends a plain JSON POST to /v1/chat/completions under a semaphore.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
        **_labeller_sampling_overrides(),
        **_labeller_extra_body(),
    }
    async with semaphore:
        for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
            try:
                resp = await http_client.post(url, json=body)
            except Exception as e:
                if attempt < LLM_MAX_ATTEMPTS:
                    logger.warning(f"Expert request attempt {attempt}/{LLM_MAX_ATTEMPTS} errored "
                                   f"({_describe_exception(e)}); retrying")
                    await _retry_backoff(attempt)
                    continue
                logger.warning(f"Raw async LLM call failed after {attempt} attempts: "
                               f"{_describe_exception(e)}, defaulting to 0")
                return 0
            if resp.status_code != 200:
                if _is_retryable_status(resp.status_code) and attempt < LLM_MAX_ATTEMPTS:
                    logger.warning(f"Expert request attempt {attempt}/{LLM_MAX_ATTEMPTS} got HTTP "
                                   f"{resp.status_code}; retrying")
                    await _retry_backoff(attempt)
                    continue
                logger.warning(
                    f"Raw LLM call HTTP {resp.status_code}: {resp.text[:200]}, defaulting to 0"
                )
                return 0
            payload = resp.json()
            _record_llm_usage(payload.get("usage"))
            choices = payload.get("choices") or []
            if not choices:
                return 0
            msg = choices[0].get("message") or {}
            content = (msg.get("content") or "").strip().lower()
            return 1 if "true" in content else 0
        return 0


# --- Cached raw httpx client + event loop ----------------------------------
# activeSampling sampling calls label_rows_with_llm ~10 times per pipeline run (one
# per version-space round) and many iterative samplers do the same. Creating
# a fresh client and event loop on every call forces TCP handshakes and a
# new connection pool every batch, which crushes the end-to-end throughput
# advantage of async I/O.
#
# The Rust baseline (sparkai/HetuDataU/.../query_utils.rs) uses a
# reqwest::Client with pool_max_idle_per_host=64 + 30s tcp_keepalive and
# max_concurrent_requests=256. Python's default httpx pool is 10 per host
# which bottlenecks concurrency regardless of the asyncio.Semaphore cap.
# We replicate the Rust tuning here and bypass AsyncOpenAI's pydantic
# validation on the hot path by posting raw JSON via httpx directly.
from typing import Any as _Any  # noqa: E402
_async_httpx_cache: Dict[str, _Any] = {}
_async_client_cache: Dict[str, AsyncOpenAI] = {}
_async_client_lock = threading.Lock()
_shared_loop: Optional[asyncio.AbstractEventLoop] = None
_shared_loop_lock = threading.Lock()


def _make_tuned_httpx_client():
    import httpx
    limits = httpx.Limits(
        max_connections=512,
        max_keepalive_connections=256,
        keepalive_expiry=30.0,
    )
    timeout = httpx.Timeout(
        connect=10.0, read=600.0, write=60.0, pool=30.0,
    )
    try:
        return httpx.AsyncClient(limits=limits, timeout=timeout, http2=True)
    except Exception:
        return httpx.AsyncClient(limits=limits, timeout=timeout)


def _get_shared_httpx_client(base_url: str):
    with _async_client_lock:
        client = _async_httpx_cache.get(base_url)
        if client is None:
            client = _make_tuned_httpx_client()
            _async_httpx_cache[base_url] = client
        return client


def _get_shared_async_client(base_url: str) -> AsyncOpenAI:
    with _async_client_lock:
        client = _async_client_cache.get(base_url)
        if client is None:
            try:
                import httpx
                limits = httpx.Limits(
                    max_connections=512,
                    max_keepalive_connections=256,
                    keepalive_expiry=30.0,
                )
                timeout = httpx.Timeout(
                    connect=10.0, read=600.0, write=60.0, pool=30.0,
                )
                try:
                    http_client = httpx.AsyncClient(
                        limits=limits, timeout=timeout, http2=True,
                    )
                except Exception:
                    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
                client = AsyncOpenAI(
                    base_url=base_url, api_key="dummy",
                    http_client=http_client, timeout=600.0,
                )
            except Exception:
                client = AsyncOpenAI(base_url=base_url, api_key="dummy", timeout=600.0)
            _async_client_cache[base_url] = client
        return client


def _get_shared_event_loop() -> asyncio.AbstractEventLoop:
    global _shared_loop
    with _shared_loop_lock:
        if _shared_loop is None or _shared_loop.is_closed():
            new_loop: Optional[asyncio.AbstractEventLoop] = None
            try:
                import uvloop
                new_loop = uvloop.new_event_loop()
                logger.info("proxy_model: using uvloop for faster async I/O")
            except Exception:
                new_loop = None
            if new_loop is None:
                new_loop = asyncio.new_event_loop()
            _shared_loop = new_loop
        return _shared_loop


def _close_shared_async_resources() -> None:
    global _shared_loop
    loop = _shared_loop
    oai_clients = list(_async_client_cache.values())
    hx_clients = list(_async_httpx_cache.values())
    _async_client_cache.clear()
    _async_httpx_cache.clear()
    if loop is None or loop.is_closed():
        return

    async def _close_all():
        for c in oai_clients:
            try:
                await c.close()
            except Exception as e:
                logger.debug(f"AsyncOpenAI close failed: {e}")
        for c in hx_clients:
            try:
                await c.aclose()
            except Exception as e:
                logger.debug(f"httpx close failed: {e}")

    try:
        loop.run_until_complete(_close_all())
    except Exception as e:
        logger.debug(f"Shared loop shutdown failed: {e}")
    try:
        loop.close()
    except Exception:
        pass
    _shared_loop = None


atexit.register(_close_shared_async_resources)


# --- Batched /v1/completions path (text-only) -----------------------------
# vLLM's /v1/completions endpoint accepts `prompt: ["p1", "p2", ...]` and
# returns N completions in a single HTTP response. Batching this way
# eliminates per-row HTTP/JSON overhead and asyncio task scheduling cost,
# which dominates the QA phase where ~half the budget gets labelled in a
# single label_fn() call.
#
# vLLM's chat template is applied server-side only on /v1/chat/completions.
# For /v1/completions we must format the prompt ourselves. We default to
# the Qwen-style ChatML format used by Qwen2/Qwen2.5/Qwen3-VL/Qwen3.6 models.
# Override via env var SEMBENCH_CHAT_TEMPLATE if a different model needs
# a different format.
_QWEN_CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_CHAT_TEMPLATE = os.environ.get("SEMBENCH_CHAT_TEMPLATE", _QWEN_CHAT_TEMPLATE)

# --- Qwen3 / Qwen3.6 hybrid-thinking mode ---------------------------------
# Qwen3 / Qwen3.6 are hybrid reasoning models: by default they emit a
# <think>...</think> chain-of-thought before the final answer. For binary
# True/False labelling that CoT is wasted tokens and can also cause our
# `"true" in content` parser to misfire on stray tokens inside <think>.
#
# The official way to suppress thinking is to pass
#     chat_template_kwargs={"enable_thinking": False}
# to vLLM's /v1/chat/completions. Server-side, the chat template injects
# `<think>\n\n</think>\n\n` after `<|im_start|>assistant\n` to short-circuit
# the thinking turn. For /v1/completions we apply the chat template
# ourselves, so we inject the same suffix into the prompt directly.
#
# Toggle with SEMBENCH_ENABLE_THINKING=1 to re-enable CoT.
_ENABLE_THINKING = os.environ.get("SEMBENCH_ENABLE_THINKING", "0").lower() not in (
    "0", "false", "no", "",
)
_QWEN_NO_THINK_SUFFIX = "<think>\n\n</think>\n\n"


def _chat_template_kwargs() -> Dict[str, bool]:
    """Return the chat_template_kwargs payload to send with chat completions.

    Used by /v1/chat/completions paths (server-side template rendering)."""
    return {"enable_thinking": bool(_ENABLE_THINKING)}


# --- Deterministic sampling overrides for binary labelling ----------------
# vLLM's --override-generation-config sets server-side defaults for every
# request, and any sampling param the client does NOT explicitly send falls
# through to those defaults. For Qwen3-recommended chat sampling
# (temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5) this badly
# corrupts True/False oracle labelling:
#   * presence_penalty>0 penalises tokens that have already appeared in the
#     prompt — including "True"/"False" from the system instructions —
#     producing a systematic bias toward whichever label is rarer in the
#     prompt context. We've observed this manifest as a high false-positive
#     rate (negatives flipped to positives).
#   * top_p<1 + top_k>1 + temperature>0 turn a deterministic classification
#     into a stochastic one, hurting reproducibility.
# We therefore pin neutral values on every oracle request so they override
# the server defaults. Callers can still override `temperature` per-call
# (default 0.0 from label_rows_with_llm).
def _labeller_sampling_overrides() -> Dict[str, float]:
    """OpenAI-standard sampling params, pinned to neutral values."""
    return {
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }


def _labeller_extra_body() -> Dict[str, Any]:
    """vLLM-only sampling extensions (top_k) plus chat_template_kwargs.

    Used in `extra_body` for the OpenAI SDK paths. Raw-httpx paths can
    splat this dict directly into the JSON body."""
    return {
        "top_k": -1,  # vLLM convention: -1 disables top-k filtering
        "repetition_penalty": 1.0,
        "chat_template_kwargs": _chat_template_kwargs(),
    }


def _format_chat_prompt(system_prompt: str, user_content: str) -> str:
    prompt = _CHAT_TEMPLATE.format(system=system_prompt, user=user_content)
    if not _ENABLE_THINKING and _CHAT_TEMPLATE is _QWEN_CHAT_TEMPLATE:
        # Mirror the server-side `enable_thinking=False` behaviour for the
        # /v1/completions path, where vLLM does not apply a chat template.
        prompt = prompt + _QWEN_NO_THINK_SUFFIX
    return prompt


def _all_text(row_texts: List[Union[str, list]]) -> bool:
    for t in row_texts:
        if not isinstance(t, str):
            return False
    return True


def _parse_label_from_text(text: str) -> int:
    return 1 if "true" in (text or "").strip().lower() else 0


async def _label_chunk_via_completions_async(
    http_client,
    completions_url: str,
    model: str,
    system_prompt: str,
    chunk_texts: List[str],
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> List[int]:
    prompts = [_format_chat_prompt(system_prompt, t) for t in chunk_texts]
    # /v1/completions does not apply the chat template server-side, so
    # chat_template_kwargs is meaningless here — but we still need to pin
    # the sampling overrides to neutralise the vLLM server's
    # --override-generation-config defaults.
    completion_body_overrides = dict(_labeller_extra_body())
    completion_body_overrides.pop("chat_template_kwargs", None)
    body = {
        "model": model,
        "prompt": prompts,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
        **_labeller_sampling_overrides(),
        **completion_body_overrides,
    }
    async with semaphore:
        for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
            try:
                resp = await http_client.post(completions_url, json=body)
            except Exception as e:
                if attempt < LLM_MAX_ATTEMPTS:
                    logger.warning(f"Batched expert request attempt {attempt}/{LLM_MAX_ATTEMPTS} errored "
                                   f"({_describe_exception(e)}); retrying")
                    await _retry_backoff(attempt)
                    continue
                logger.warning(f"Batched /v1/completions failed after {attempt} attempts: "
                               f"{_describe_exception(e)}, defaulting to 0")
                return [0] * len(chunk_texts)
            if resp.status_code != 200:
                if _is_retryable_status(resp.status_code) and attempt < LLM_MAX_ATTEMPTS:
                    logger.warning(f"Batched expert request attempt {attempt}/{LLM_MAX_ATTEMPTS} got HTTP "
                                   f"{resp.status_code}; retrying")
                    await _retry_backoff(attempt)
                    continue
                # Keep "defaulting to 0" in the message so run watchers catch persistent failures.
                logger.warning(
                    f"Batched /v1/completions HTTP {resp.status_code}: "
                    f"{resp.text[:200]}, defaulting to 0 (all-zero chunk)"
                )
                return [0] * len(chunk_texts)
            payload = resp.json()
            _record_llm_usage(payload.get("usage"))
            choices = payload.get("choices") or []
            results = [0] * len(chunk_texts)
            for c in choices:
                idx = c.get("index")
                text = c.get("text") or ""
                if isinstance(idx, int) and 0 <= idx < len(chunk_texts):
                    results[idx] = _parse_label_from_text(text)
            return results
        return [0] * len(chunk_texts)


async def _label_rows_completions_batched_async(
    row_texts: List[Union[str, list]],
    model: str,
    url: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    max_concurrent_requests: int,
    chat_batch_size: int,
) -> np.ndarray:
    text_only: List[str] = [t for t in row_texts if isinstance(t, str)]
    if len(text_only) != len(row_texts):
        raise ValueError(
            "_label_rows_completions_batched_async requires all rows to be plain text"
        )

    base_url = url.rsplit("/chat/completions", 1)[0].rsplit("/completions", 1)[0]
    completions_url = base_url.rstrip("/") + "/completions"
    http_client = _get_shared_httpx_client(base_url)
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    chunks: List[List[str]] = [
        text_only[i : i + chat_batch_size]
        for i in range(0, len(text_only), chat_batch_size)
    ]
    tasks = [
        _label_chunk_via_completions_async(
            http_client, completions_url, model, system_prompt, chunk,
            max_tokens, temperature, semaphore,
        )
        for chunk in chunks
    ]
    chunk_results = await asyncio.gather(*tasks)

    flat: List[int] = []
    for cr in chunk_results:
        flat.extend(cr)
    return np.array(flat, dtype=np.int32)


async def _label_rows_async(
    row_texts: List[Union[str, list]],
    model: str,
    url: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    max_concurrent_requests: int,
    chat_batch_size: int = 32,
) -> np.ndarray:
    """Label rows using async I/O.

    For text-only row_texts, batches multiple prompts per HTTP request via
    vLLM's /v1/completions endpoint (chat_batch_size prompts per request).
    This dramatically lowers per-row HTTP/JSON overhead and is the main
    Query-Around-Hypothesis speedup.

    For multimodal row_texts (image inputs), keeps the per-row
    /v1/chat/completions path because /v1/completions does not accept
    image content.
    """
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    base_url = url.rsplit("/chat/completions", 1)[0]

    if row_texts and _all_text(row_texts) and chat_batch_size > 1:
        try:
            logger.info(
                f"Labelling {len(row_texts)} text rows via /v1/completions "
                f"in chunks of {chat_batch_size} (concurrency={max_concurrent_requests})"
            )
            return await _label_rows_completions_batched_async(
                row_texts, model, url, system_prompt,
                max_tokens, temperature,
                max_concurrent_requests, chat_batch_size,
            )
        except Exception as e:
            logger.info(f"Batched completions path failed ({e}); falling back to per-row")

    try:
        http_client = _get_shared_httpx_client(base_url)
        tasks = [
            _call_llm_single_raw_async(
                http_client, url, model, system_prompt, text,
                max_tokens, temperature, semaphore,
            )
            for text in row_texts
        ]
        results = await asyncio.gather(*tasks)
        return np.array(results, dtype=np.int32)
    except Exception as e:
        logger.info(f"Raw httpx path unavailable ({e}); falling back to AsyncOpenAI")
        client = _get_shared_async_client(base_url)
        tasks = [
            _call_llm_single_async(
                client, model, system_prompt, text,
                max_tokens, temperature, semaphore,
            )
            for text in row_texts
        ]
        results = await asyncio.gather(*tasks)
        return np.array(results, dtype=np.int32)


def _get_or_create_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Return a long-lived event loop for running async labelling batches.

    If we're already inside a running loop (Jupyter / Spark UDF), patch it
    with nest_asyncio when available; otherwise fall back to the
    ThreadPoolExecutor path by returning None.  In the common driver-thread
    case we return a single shared loop across all batches so the cached
    AsyncOpenAI client stays bound to it.
    """
    try:
        loop = asyncio.get_running_loop()
        try:
            import nest_asyncio
            nest_asyncio.apply(loop)
            return loop
        except ImportError:
            return None
    except RuntimeError:
        return _get_shared_event_loop()


def label_rows_with_llm(
    row_texts: List[Union[str, list]],
    model: str,
    url: str,
    system_prompt: str = FILTER_SYSTEM_PROMPT,
    max_tokens: int = 16,
    temperature: float = 0.0,
    max_concurrent_requests: int = 256,
    budget: Optional[dict] = None,
    chat_batch_size: int = 32,
) -> np.ndarray:
    """Label rows with LLM. Uses async I/O for higher vLLM throughput.

    If *budget* is provided, at most ``_budget_remaining(budget)`` rows will
    be sent to the LLM.  The remaining rows are assigned label 0 (negative).
    The budget counter is incremented by the number of rows actually labelled.

    For text-only ``row_texts`` we batch ``chat_batch_size`` prompts per HTTP
    request via vLLM's /v1/completions endpoint. Multimodal rows always use
    the per-row /v1/chat/completions path.
    """
    if len(row_texts) == 0:
        return np.array([], dtype=np.int32)

    remaining = _budget_remaining(budget)
    if remaining <= 0:
        logger.info("Oracle budget exhausted — skipping LLM labelling, returning all-zero labels.")
        return np.zeros(len(row_texts), dtype=np.int32)

    # Truncate to remaining budget
    truncated = False
    if remaining < len(row_texts):
        logger.info(
            f"Oracle budget: only labelling {remaining}/{len(row_texts)} rows "
            f"(budget {budget['used']}/{budget['limit']})"
        )
        truncated = True
        full_count = len(row_texts)
        row_texts = row_texts[:remaining]

    labels = None
    loop = _get_or_create_event_loop()
    if loop is not None:
        try:
            labels = loop.run_until_complete(
                _label_rows_async(
                    row_texts, model, url, system_prompt,
                    max_tokens, temperature, max_concurrent_requests,
                    chat_batch_size,
                )
            )
        except RuntimeError:
            pass  # Fall through to sync fallback

    if labels is None:
        # Sync fallback using ThreadPoolExecutor
        client = OpenAI(
            base_url=url.rsplit("/chat/completions", 1)[0],
            api_key="dummy", timeout=300,
        )
        labels = np.zeros(len(row_texts), dtype=np.int32)
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            futures = {}
            for i, text in enumerate(row_texts):
                future = executor.submit(
                    _call_llm_single, client, model, system_prompt,
                    text, max_tokens, temperature,
                )
                futures[future] = i
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    labels[idx] = future.result()
                except Exception as e:
                    logger.warning(f"LLM call failed for row {idx}: {e}, defaulting to 0")
                    labels[idx] = 0

    # Update budget counter
    if budget is not None:
        budget["used"] += len(labels)
        logger.info(f"Oracle budget: {budget['used']}/{budget['limit']} calls used")

    # Pad with zeros if we truncated the input
    if truncated:
        padded = np.zeros(full_count, dtype=np.int32)
        padded[:len(labels)] = labels
        return padded

    return labels


def sample_random(total: int, sample_size: int) -> np.ndarray:
    n = min(sample_size, total)
    return np.random.choice(total, n, replace=False)


def sample_topk(
    embeddings: np.ndarray, prompt_embedding: np.ndarray, sample_size: int
) -> np.ndarray:
    norms_e = np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
    norms_p = np.linalg.norm(prompt_embedding).reshape(1)
    similarities = (embeddings @ prompt_embedding.T).flatten() / (
        norms_e.flatten() * norms_p.flatten() + 1e-8
    )
    n = min(sample_size, len(similarities))
    return np.argsort(similarities)[-n:]


def _compute_similarities(
    embeddings: np.ndarray,
    prompt_embedding: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarities between row embeddings and prompt embedding."""
    norms_e = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms_p = np.linalg.norm(prompt_embedding, keepdims=True)
    return (embeddings @ prompt_embedding.T).flatten() / (
        norms_e.flatten() * norms_p.flatten() + 1e-8
    )


def sample_stratified(
    embeddings: np.ndarray,
    prompt_embedding: np.ndarray,
    texts: List[Union[str, list]],
    sample_size: int,
    model: str,
    url: str,
    system_prompt: str = FILTER_SYSTEM_PROMPT,
    max_tokens: int = 16,
    temperature: float = 0.0,
    max_concurrent_requests: int = 256,
    n_strata: int = 5,
    n_rounds: int = 5,
    budget: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive stratified active sampling with importance weighting and
    variance-based budget allocation.

    Three key mechanisms:
      1. Strata partition by cosine-similarity quantiles.
      2. Seed round uniform → then Neyman variance-based allocation:
         budget_k ∝ |S_k| × σ_k.  Strata with high label variance
         (boundary regions) get more LLM budget; pure strata get none.
      3. Importance weighting: low-similarity rows that turn out positive
         get higher training weight (inverse-propensity style), because
         they are rare false-negative signals critical for proxy quality.

    Returns:
        (indices, labels, sample_weights)
        sample_weights[i] = training weight for row i.  Low-similarity
        positives get weight > 1 (up-weighted); high-similarity negatives
        get weight ≈ 1 (neutral).
    """
    total = len(texts)
    sample_size = min(sample_size, total)

    # --- Partition into strata by similarity ---
    similarities = _compute_similarities(embeddings, prompt_embedding)
    quantiles = np.linspace(0, 1, n_strata + 1)
    thresholds = np.quantile(similarities, quantiles)
    row_strata = np.digitize(similarities, thresholds[1:-1])  # 0..n_strata-1

    # Pre-compute rows per stratum
    stratum_rows = [np.where(row_strata == k)[0] for k in range(n_strata)]

    labeled_set: set = set()
    all_indices: List[int] = []
    all_labels: List[int] = []

    # Per-stratum variance (initialized uniform; updated after seed round)
    stratum_variance = np.ones(n_strata, dtype=float) * 0.25  # max variance for binary

    # Budget per round
    round_budget = max(1, sample_size // n_rounds)

    # Current proxy model (None until first fit after seed round)
    current_lr = None

    def _fit_proxy():
        """Fit proxy on current labeled data, return (model, f1)."""
        lab = np.array(all_labels, dtype=np.int32)
        if len(lab) < 2 or len(np.unique(lab)) < 2:
            return None, 0.0
        idx = np.array(all_indices)
        lr = LogisticRegression(
            class_weight="balanced", max_iter=500, solver="lbfgs",
        )
        lr.fit(embeddings[idx], lab)
        preds = lr.predict(embeddings[idx])
        f1 = float(f1_score(lab, preds, average="macro", zero_division=0))
        return lr, f1

    def _select_within_stratum(
        stratum_unlabeled: np.ndarray, budget_k: int, lr_model,
    ) -> np.ndarray:
        """Select the most informative rows within a stratum."""
        n_select = min(budget_k, len(stratum_unlabeled))
        if n_select <= 0:
            return np.array([], dtype=int)

        if lr_model is None or len(np.unique(np.array(all_labels, dtype=np.int32))) < 2:
            return np.random.choice(stratum_unlabeled, n_select, replace=False)

        probas = lr_model.predict_proba(embeddings[stratum_unlabeled])
        pos_cls = list(lr_model.classes_).index(1) if 1 in lr_model.classes_ else 0
        pos_p = probas[:, pos_cls]

        num_definite = int(np.sum(np.array(all_labels) == 1))
        num_unknown = total - len(all_indices)
        if num_definite > 0 and num_unknown > 0:
            eer = _compute_eer_scores(pos_p, num_definite, num_unknown)
            top_idx = np.argsort(eer)[-n_select:]
        else:
            uncertainty = 1.0 - np.abs(pos_p - 0.5) * 2
            top_idx = np.argsort(uncertainty)[-n_select:]

        return stratum_unlabeled[top_idx]

    def _sample_from_stratum_active(
        k: int, budget_k: int, lr_model,
    ) -> Tuple[int, int]:
        """Select and label rows from stratum k using active learning."""
        unlabeled = np.array([i for i in stratum_rows[k] if i not in labeled_set])
        if len(unlabeled) == 0 or budget_k <= 0:
            return 0, 0

        selected = _select_within_stratum(unlabeled, budget_k, lr_model)
        if len(selected) == 0:
            return 0, 0

        selected_texts = [texts[i] for i in selected]
        selected_labels = label_rows_with_llm(
            selected_texts, model, url, system_prompt,
            max_tokens, temperature, max_concurrent_requests,
            budget=budget,
        )
        all_indices.extend(selected.tolist())
        all_labels.extend(selected_labels.tolist())
        labeled_set.update(selected.tolist())
        return len(selected), int((selected_labels == 1).sum())

    def _compute_stratum_variance():
        """Compute per-stratum label variance σ²_k = p_k(1-p_k)."""
        for k in range(n_strata):
            stratum_labeled = [
                all_labels[j] for j, idx in enumerate(all_indices)
                if row_strata[idx] == k
            ]
            if len(stratum_labeled) >= 2:
                p_k = np.mean(stratum_labeled)
                stratum_variance[k] = p_k * (1 - p_k)
            else:
                stratum_variance[k] = 0.25  # unknown → assume max variance

    for rnd in range(n_rounds):
        remaining = sample_size - len(all_indices)
        if remaining <= 0:
            break
        if _budget_exhausted(budget):
            logger.info("Oracle budget exhausted — stopping stratified sampling.")
            break
        this_round = min(round_budget, remaining)

        if rnd == 0:
            # --- Seed round: uniform allocation, random within stratum ---
            per_stratum = max(1, this_round // n_strata)
            for k in range(n_strata):
                if sample_size - len(all_indices) <= 0 or _budget_exhausted(budget):
                    break
                n_sampled, n_pos = _sample_from_stratum_active(k, per_stratum, None)
                sim_lo, sim_hi = thresholds[k], thresholds[k + 1]
                logger.info(
                    f"[Stratified] Seed stratum {k}/{n_strata} "
                    f"sim=[{sim_lo:.3f},{sim_hi:.3f}]: "
                    f"sampled {n_sampled}, pos={n_pos}"
                )

            current_lr, baseline_f1 = _fit_proxy()
            _compute_stratum_variance()
            logger.info(
                f"[Stratified] Seed round done: {len(all_indices)} labeled, "
                f"proxy F1={baseline_f1:.4f}, "
                f"variances={[f'{v:.3f}' for v in stratum_variance]}"
            )
        else:
            # --- Adaptive round: Neyman variance-based allocation ---
            # budget_k ∝ |S_k| × σ_k  (Neyman optimal allocation)
            stratum_sizes = np.array([len(stratum_rows[k]) for k in range(n_strata)], dtype=float)
            sigma_k = np.sqrt(stratum_variance + 1e-8)
            neyman_weights = stratum_sizes * sigma_k
            neyman_sum = neyman_weights.sum()
            if neyman_sum > 0:
                weights = neyman_weights / neyman_sum
            else:
                weights = np.ones(n_strata) / n_strata
            stratum_budgets = np.round(this_round * weights).astype(int)
            diff = this_round - stratum_budgets.sum()
            stratum_budgets[np.argmax(neyman_weights)] += diff

            f1_before = _fit_proxy()[1]

            for k in range(n_strata):
                if sample_size - len(all_indices) <= 0 or _budget_exhausted(budget):
                    break
                n_sampled, n_pos = _sample_from_stratum_active(
                    k, int(stratum_budgets[k]), current_lr,
                )
                if n_sampled == 0:
                    continue

                current_lr, f1_after_k = _fit_proxy()
                _compute_stratum_variance()

                sim_lo, sim_hi = thresholds[k], thresholds[k + 1]
                logger.info(
                    f"[Stratified] Round {rnd} stratum {k}/{n_strata} "
                    f"sim=[{sim_lo:.3f},{sim_hi:.3f}]: "
                    f"sampled {n_sampled}, pos={n_pos}, "
                    f"variance={stratum_variance[k]:.4f}, "
                    f"F1_delta={f1_after_k - f1_before:+.4f}"
                )
                f1_before = f1_after_k

            logger.info(
                f"[Stratified] Round {rnd} done: {len(all_indices)} labeled, "
                f"proxy F1={f1_before:.4f}, "
                f"variances={[f'{v:.3f}' for v in stratum_variance]}"
            )

    # --- Compute importance weights ---
    # Low-similarity positives are rare and informative → higher weight.
    # Weight = inverse of stratum's positive rate (capped).
    # High-sim stratum with 50% positives → weight 2.
    # Low-sim stratum with 1% positives  → weight 100 (capped to 10).
    final_indices = np.array(all_indices)
    final_labels = np.array(all_labels, dtype=np.int32)
    sample_weights = np.ones(len(final_labels), dtype=np.float32)

    # Compute per-stratum positive rate
    for k in range(n_strata):
        stratum_mask = np.array([row_strata[idx] == k for idx in final_indices])
        stratum_labels = final_labels[stratum_mask]
        if len(stratum_labels) == 0:
            continue
        p_k = max(stratum_labels.mean(), 0.01)  # floor to avoid div-by-zero

        # For positives in this stratum: weight = 1 / p_k (capped at 10)
        # Intuition: positive in a 1%-positive stratum is 100× rarer than
        # positive in a 50%-positive stratum → deserves more influence.
        pos_weight = min(1.0 / p_k, 10.0)
        for j in range(len(final_indices)):
            if stratum_mask[j] and final_labels[j] == 1:
                sample_weights[j] = pos_weight

    final_pos = int((final_labels == 1).sum())
    _, final_f1 = _fit_proxy()
    weighted_pos = float(sample_weights[final_labels == 1].sum()) if final_pos > 0 else 0
    logger.info(
        f"[Stratified] Complete: {len(final_indices)} labeled, "
        f"{final_pos} positives (effective weighted={weighted_pos:.1f}), "
        f"proxy F1={final_f1:.4f}"
    )

    return final_indices, final_labels, sample_weights


def sample_active(
    embeddings: np.ndarray,
    texts: List[Union[str, list]],
    model: str,
    url: str,
    system_prompt: str,
    sample_size: int,
    max_tokens: int = 16,
    temperature: float = 0.0,
    max_concurrent_requests: int = 256,
    initial_size: int = 50,
    batch_size: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    # NOTE: LEGACY, NON-PAPER PATH. This is a uniform-pilot + uncertainty
    # sampler and does NOT implement the paper's budget-aware sampling
    # (Alg. 1/2). The paper's sampler lives in
    # cluster_proxy.py::sample_activeSampling (reached via
    # proxy_filter/proxy_join -> cluster_proxy_pipeline with
    # sampling_method="activeSampling"). Kept only for legacy callers.
    total = len(texts)
    initial_size = min(initial_size, sample_size, total)

    all_indices = np.random.choice(total, initial_size, replace=False)
    pilot_texts = [texts[i] for i in all_indices]
    all_labels = label_rows_with_llm(
        pilot_texts, model, url, system_prompt, max_tokens, temperature, max_concurrent_requests
    )

    remaining_budget = sample_size - initial_size
    labeled_set = set(all_indices.tolist())

    while remaining_budget > 0:
        if len(np.unique(all_labels)) < 2:
            next_batch_size = min(batch_size, remaining_budget)
            unlabeled = np.array([i for i in range(total) if i not in labeled_set])
            if len(unlabeled) == 0:
                break
            new_idx = np.random.choice(
                unlabeled, min(next_batch_size, len(unlabeled)), replace=False
            )
        else:
            lr = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
            lr.fit(embeddings[all_indices], all_labels)

            unlabeled = np.array([i for i in range(total) if i not in labeled_set])
            if len(unlabeled) == 0:
                break

            probas = lr.predict_proba(embeddings[unlabeled])[:, 1]
            uncertainty = np.abs(probas - 0.5)
            next_batch_size = min(batch_size, remaining_budget, len(unlabeled))
            most_uncertain = np.argsort(uncertainty)[:next_batch_size]
            new_idx = unlabeled[most_uncertain]

        new_texts = [texts[i] for i in new_idx]
        new_labels = label_rows_with_llm(
            new_texts, model, url, system_prompt, max_tokens, temperature, max_concurrent_requests
        )

        all_indices = np.concatenate([all_indices, new_idx])
        all_labels = np.concatenate([all_labels, new_labels])
        labeled_set.update(new_idx.tolist())
        remaining_budget -= len(new_idx)

    return all_indices, all_labels


def _handle_imbalance(
    embeddings: np.ndarray, labels: np.ndarray, strategy: str
) -> Tuple[np.ndarray, np.ndarray]:
    if strategy == "weighted":
        return embeddings, labels
    elif strategy == "downsample":
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) < 2:
            return embeddings, labels
        min_count = counts.min()
        indices: List[int] = []
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            chosen = np.random.choice(cls_idx, min_count, replace=False)
            indices.extend(chosen.tolist())
        indices_arr = np.array(indices)
        np.random.shuffle(indices_arr)
        return embeddings[indices_arr], labels[indices_arr]
    elif strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE

            sm = SMOTE(random_state=42)
            return sm.fit_resample(embeddings, labels)
        except ImportError:
            logger.warning("imbalanced-learn not installed, falling back to weighted strategy")
            return embeddings, labels
    else:
        return embeddings, labels


class _ConstantClassifier:
    """Fallback classifier when training data contains only one class.

    Mimics the LogisticRegression interface (classes_, predict, predict_proba)
    so downstream code works without special-casing.
    """

    def __init__(self, single_class: int, n_features: int):
        self.classes_ = np.array([0, 1])
        self._single_class = single_class
        self._n_features = n_features

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._single_class, dtype=np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if self._single_class == 1:
            return np.column_stack([np.zeros(n), np.ones(n)])
        else:
            return np.column_stack([np.ones(n), np.zeros(n)])


def fit_proxy_model(
    embeddings: np.ndarray,
    labels: np.ndarray,
    imbalance_strategy: str = "weighted",
    verbose: bool = False,
    prof: Optional[dict] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> LogisticRegression:
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        single_class = int(unique_classes[0])
        logger.warning(
            f"Only one class ({single_class}) found in {len(labels)} samples. "
            f"Returning constant classifier (all predictions = {single_class})."
        )
        return _ConstantClassifier(single_class, embeddings.shape[1])

    train_emb, train_labels = _handle_imbalance(embeddings, labels, imbalance_strategy)

    # Align sample_weight with potentially resampled data
    train_weight = None
    if sample_weight is not None and len(sample_weight) == len(labels):
        if len(train_labels) == len(labels):
            train_weight = sample_weight
        # If imbalance handling changed the size, drop custom weights
        # (class_weight="balanced" will handle it)

    class_weight = "balanced" if imbalance_strategy == "weighted" else None
    lr = LogisticRegression(
        class_weight=class_weight, max_iter=1000, solver="lbfgs", warm_start=False
    )
    t0 = time.time()
    lr.fit(train_emb, train_labels, sample_weight=train_weight)
    lr_dur = time.time() - t0
    if prof is not None:
        prof["lr_training_time"] += lr_dur

    log_training_metrics(lr, embeddings, labels, verbose)
    return lr


def log_training_metrics(
    model: LogisticRegression,
    embeddings: np.ndarray,
    labels: np.ndarray,
    verbose: bool = False,
    prefix: str = "Training",
) -> dict:
    """Log and return proxy model metrics. Always prints key stats."""
    preds = model.predict(embeddings)
    probas = model.predict_proba(embeddings)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division="warn")
    try:
        ll = log_loss(labels, probas)
    except ValueError:
        ll = float("nan")

    # ROC-AUC and Brier score (need both classes present)
    pos_idx_cls = list(np.unique(labels)).index(1) if 1 in np.unique(labels) else -1
    if pos_idx_cls >= 0 and len(np.unique(labels)) >= 2:
        pos_probas = probas[:, pos_idx_cls]
        try:
            auc = roc_auc_score(labels, pos_probas)
        except ValueError:
            auc = float("nan")
        try:
            brier = brier_score_loss(labels, pos_probas)
        except ValueError:
            brier = float("nan")
    else:
        auc = float("nan")
        brier = float("nan")

    classes, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(classes.tolist(), counts.tolist()))
    imbalance_ratio = counts.max() / max(counts.min(), 1)

    # Compute per-class metrics for skew diagnosis
    pos_count = int((labels == 1).sum())
    neg_count = int((labels == 0).sum())
    pos_rate = pos_count / max(len(labels), 1)

    logger.info(f"=== {prefix} Proxy Metrics ===")
    logger.info(f"Sample size: {len(labels)}  |  Positive: {pos_count} ({pos_rate:.1%})  |  Negative: {neg_count}")
    logger.info(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
    logger.info(f"Accuracy: {acc:.4f}  |  Macro F1: {f1:.4f}  |  Log-Loss: {ll:.4f}")
    logger.info(f"ROC-AUC: {auc:.4f}  |  Brier Score: {brier:.4f}")

    if pos_count > 0 and neg_count > 0:
        # Per-class precision/recall for skew diagnosis
        from sklearn.metrics import precision_recall_fscore_support
        p_per, r_per, f1_per, _ = precision_recall_fscore_support(
            labels, preds, labels=[0, 1], zero_division=0
        )
        logger.info(
            f"Class 0 (neg): P={p_per[0]:.3f} R={r_per[0]:.3f} F1={f1_per[0]:.3f}  |  "
            f"Class 1 (pos): P={p_per[1]:.3f} R={r_per[1]:.3f} F1={f1_per[1]:.3f}"
        )
        if imbalance_ratio > 10:
            logger.warning(
                f"HIGH IMBALANCE ({imbalance_ratio:.0f}:1). "
                f"Consider using sampling_method='online' for active learning refinement."
            )

    if verbose:
        report = classification_report(labels, preds, zero_division="warn")
        cm = confusion_matrix(labels, preds)
        logger.info(f"Classification Report:\n{report}")
        logger.info(f"Confusion Matrix:\n{cm}")

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "log_loss": float(ll),
        "roc_auc": float(auc),
        "brier_score": float(brier),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "imbalance_ratio": float(imbalance_ratio),
    }


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def save_proxy_artifact(
    path: str,
    model: LogisticRegression,
    training_embeddings: np.ndarray,
    training_labels: np.ndarray,
    prompt: str,
    embedding_model: str,
    num_updates: int = 0,
    metrics: Optional[dict] = None,
) -> None:
    artifact = {
        "model": model,
        "training_embeddings": training_embeddings,
        "training_labels": training_labels,
        "prompt_hash": _prompt_hash(prompt),
        "embedding_model": embedding_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "num_updates": num_updates,
        "metrics": metrics or {},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info(f"Saved proxy model to: {path}")


def load_proxy_artifact(path: str, prompt: str, embedding_model: str) -> dict:
    with open(path, "rb") as f:
        artifact = pickle.load(f)

    if artifact["prompt_hash"] != _prompt_hash(prompt):
        logger.warning(
            "Prompt hash mismatch: saved model was trained for a different query. "
            "Results may be inaccurate."
        )

    if artifact["embedding_model"] != embedding_model:
        raise ValueError(
            f"Embedding model mismatch: saved={artifact['embedding_model']}, "
            f"current={embedding_model}. Cannot reuse model with different embeddings."
        )

    logger.info(f"Loaded proxy model from: {path} ({artifact['num_updates']} previous updates)")
    return artifact


def _log_prediction_summary(
    probas: np.ndarray,
    threshold: float,
    total_rows: int,
    sample_size: int,
    total_should_call: Optional[int] = None,
    proxy_model=None,
    train_embeddings: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
) -> None:
    positive_count = int((probas >= threshold).sum())
    negative_count = total_rows - positive_count
    logger.info("=== Prediction Summary ===")
    logger.info(f"Total rows predicted: {total_rows}")
    logger.info(f"Predicted positive (pass): {positive_count} ({100*positive_count/max(total_rows,1):.1f}%)")
    logger.info(f"Predicted negative (fail): {negative_count} ({100*negative_count/max(total_rows,1):.1f}%)")
    logger.info(f"Decision threshold: {threshold:.3f}")
    if len(probas) > 0:
        logger.info(f"Max positive probability: {float(np.max(probas)):.4f}")
    if positive_count > 0:
        logger.info(f"Avg confidence (positive): {probas[probas >= threshold].mean():.4f}")
    if negative_count > 0:
        logger.info(f"Avg confidence (negative): {(1 - probas[probas < threshold]).mean():.4f}")

    # Log proxy model quality metrics if available
    if proxy_model is not None and train_embeddings is not None and train_labels is not None:
        try:
            train_preds = proxy_model.predict(train_embeddings)
            train_probas = proxy_model.predict_proba(train_embeddings)
            train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)
            train_acc = accuracy_score(train_labels, train_preds)
            pos_cls_idx = list(proxy_model.classes_).index(1) if 1 in proxy_model.classes_ else 0
            pos_probs = train_probas[:, pos_cls_idx]
            try:
                train_auc = roc_auc_score(train_labels, pos_probs)
            except ValueError:
                train_auc = float("nan")
            try:
                train_brier = brier_score_loss(train_labels, pos_probs)
            except ValueError:
                train_brier = float("nan")
            logger.info(
                f"Proxy quality (on training data): "
                f"F1={train_f1:.4f}  Acc={train_acc:.4f}  "
                f"AUC={train_auc:.4f}  Brier={train_brier:.4f}"
            )
        except Exception as e:
            logger.debug(f"Could not compute proxy metrics: {e}")
    should_call = total_should_call if total_should_call is not None else total_rows
    logger.info(
        f"LLM calls issued: {sample_size} / {should_call} "
        f"({100*sample_size/max(should_call,1):.1f}%)"
    )
    logger.info(
        f"LLM calls saved: {should_call - sample_size} / {should_call} "
        f"({100*(should_call - sample_size)/max(should_call,1):.1f}%)"
    )


def _compute_eer_scores(
    pos_probas: np.ndarray,
    num_definite: int,
    num_unknown: int,
) -> np.ndarray:
    """Compute Expected Error Reduction (EER) scores for unlabeled rows.

    EER(i) = e_current - E[e | eval i]
    where E[e | eval i] = p_i * e_pass + (1-p_i) * e_fail

    Based on Roy & McCallum (ICML 2001) adapted for semantic query AQP.
    """
    I = num_definite
    M = num_definite + num_unknown
    if I == 0 and M == 0:
        return np.zeros(len(pos_probas))

    e_current = (M / max(I, 1)) - 1 if I > 0 else float("inf")

    eer_scores = np.zeros(len(pos_probas))
    for idx in range(len(pos_probas)):
        p = pos_probas[idx]

        # Error if this row passes (I increases by 1)
        e_pass = (M - I - 1) / (I + 1) if (I + 1) > 0 else float("inf")
        # Error if this row fails (M decreases by 1)
        if I > 0:
            e_fail = (M - 1 - I) / I
        elif M - 1 > 0:
            e_fail = float("inf")
        else:
            e_fail = 0.0

        expected_error = p * e_pass + (1 - p) * e_fail

        if np.isfinite(e_current) and np.isfinite(expected_error):
            eer_scores[idx] = e_current - expected_error
        elif np.isinf(e_current) and np.isfinite(expected_error):
            eer_scores[idx] = 1e6  # Maximum priority
        else:
            eer_scores[idx] = 0.0

    return eer_scores


def _select_verify_batch(
    pos_probas: np.ndarray,
    unlabeled_indices: np.ndarray,
    refine_batch_size: int,
    confidence_threshold: float,
    verify_positive: bool,
    num_definite: int = 0,
    num_unknown: int = 0,
    use_eer: bool = True,
) -> np.ndarray:
    """Select which unlabeled rows to verify with LLM.

    Strategies (in priority order):
    1. EER (use_eer=True): Expected Error Reduction — selects rows that
       maximally reduce the retrieval error bound M/I - 1.
    2. verify_positive=True: 50% high-confidence positives + 50% boundary-uncertain.
    3. verify_positive=False: Pure uncertainty sampling (|p - 0.5|).
    """
    if use_eer and num_unknown > 0:
        eer_scores = _compute_eer_scores(pos_probas, num_definite, num_unknown)
        verify_local = np.argsort(eer_scores)[-refine_batch_size:]
        return unlabeled_indices[verify_local]

    if verify_positive:
        # Send all confident positives to LLM for verification/training
        candidate_order = np.argsort(-pos_probas)
        uncertain_order = np.argsort(np.abs(pos_probas - confidence_threshold))
        n_confident = max(1, int(refine_batch_size))
        n_uncertain = refine_batch_size - n_confident
        verify_local = np.unique(np.concatenate([
            candidate_order[:n_confident],
            uncertain_order[:n_uncertain],
        ]))[:refine_batch_size]
    else:
        uncertainty = np.abs(pos_probas - 0.5)
        verify_local = np.argsort(uncertainty)[:refine_batch_size]
    return unlabeled_indices[verify_local]


def _online_refine(
    lr_model,
    embeddings: np.ndarray,
    texts: List[Union[str, list]],
    sample_indices: np.ndarray,
    sample_labels: np.ndarray,
    model: str,
    url: str,
    system_prompt: str,
    confidence_threshold: float,
    imbalance_strategy: str,
    max_tokens: int = 16,
    temperature: float = 0.0,
    max_concurrent_requests: int = 256,
    refine_batch_size: int = 50,
    max_refine_rounds: int = 100,
    verify_positive: bool = True,
    verbose: bool = False,
    prof: Optional[dict] = None,
    budget: Optional[dict] = None,
    deferral_enabled: bool = False,
    deferral_hidden_dims: Sequence[int] = (32, 16),
    deferral_dagger_beta: float = 0.00,
):
    """Online active learning: refine proxy during inference.

    NOTE: LEGACY, NON-PAPER PATH. This EER-driven online refinement with a
    DAgger MLP deferral head is NOT the paper's method. The paper's graph
    label propagation (Alg. 3) and proxy-expert router (Alg. 4) are
    implemented in cluster_proxy.py (predict_with_score_refinement and the
    fit_deferral_classifier call site), reached via proxy_filter/proxy_join
    -> cluster_proxy_pipeline. This function is kept only for legacy callers.

    After initial proxy is built from sample_size rows, process remaining
    rows in batches. For proxy-positive predictions, verify with LLM.
    Use verified (label, embedding) pairs to update the proxy.

    This addresses extreme class imbalance (e.g. 5 positives in 2600 rows)
    by actively seeking and verifying positive predictions, enriching the
    training set with real positives found during inference.

    Batch selection uses Expected Error Reduction (EER) scoring
    (Roy & McCallum, ICML 2001) to select rows that maximally reduce
    the retrieval error bound M/I - 1.

    Convergence: stops when (1) mean positive confidence >= threshold,
    (2) F1 plateaus for 3 consecutive rounds, or (3) no new positives
    found with sufficient positives already discovered.

    Returns: (refined_model, all_labeled_indices, all_labels)
    """
    total = len(texts)
    labeled_set = set(sample_indices.tolist())
    all_indices = list(sample_indices)
    all_labels = list(sample_labels)
    current_model = lr_model
    deferral_clf = None
    if deferral_enabled:
        from learned_deferral import DeferralClassifier

        deferral_clf = DeferralClassifier(
            hidden_dims=tuple(int(x) for x in deferral_hidden_dims),
            random_state=42,
        )
        if prof is not None:
            prof["deferral_online_updates"] = 0
            prof["deferral_online_random_probe"] = 0

    # Track how many new positives we find
    initial_pos = int((sample_labels == 1).sum())
    new_verified = 0

    # F1-plateau convergence tracking
    f1_history: List[float] = []
    _F1_PLATEAU_WINDOW = 3
    _F1_PLATEAU_THRESHOLD = 0.01

    for round_num in range(max_refine_rounds):
        if _budget_exhausted(budget):
            logger.info("Oracle budget exhausted — stopping online refinement.")
            break

        # Predict on all unlabeled rows
        unlabeled_mask = np.array([i not in labeled_set for i in range(total)])
        unlabeled_indices = np.where(unlabeled_mask)[0]

        if len(unlabeled_indices) == 0:
            break

        probas = current_model.predict_proba(embeddings[unlabeled_indices])
        pos_idx = list(current_model.classes_).index(1) if 1 in current_model.classes_ else 0
        pos_probas = probas[:, pos_idx]

        # --- Convergence check 1: stop when proxy confidence > threshold ---
        all_probas = current_model.predict_proba(embeddings)[:, pos_idx]
        predicted_positive_mask = all_probas >= confidence_threshold
        if predicted_positive_mask.sum() > 0:
            mean_pos_confidence = float(all_probas[predicted_positive_mask].mean())
        else:
            mean_pos_confidence = 0.0

        # Compute train F1 for logging and plateau detection
        all_lab_arr_check = np.array(all_labels, dtype=np.int32)
        all_idx_arr_check = np.array(all_indices)
        train_preds = current_model.predict(embeddings[all_idx_arr_check])
        train_f1 = float(f1_score(all_lab_arr_check, train_preds, average="macro", zero_division=0))
        f1_history.append(train_f1)

        # EER error bound for logging
        num_definite = int(np.sum(np.array(all_labels) == 1))
        num_unknown = len(unlabeled_indices)
        eer_error = (num_definite + num_unknown) / max(num_definite, 1) - 1 if num_definite > 0 else float("inf")

        logger.info(
            f"Online round {round_num + 1}: "
            f"mean_pos_confidence={mean_pos_confidence:.4f}, "
            f"train_f1={train_f1:.4f}, "
            f"eer_error={eer_error:.4f}, "
            f"threshold={confidence_threshold:.4f}"
        )

        if mean_pos_confidence >= confidence_threshold:
            logger.info(
                f"Convergence reached: mean positive confidence ({mean_pos_confidence:.4f}) "
                f">= threshold ({confidence_threshold:.4f}). Stopping refinement."
            )
            break

        # --- Convergence check 2: F1 plateau ---
        if len(f1_history) >= _F1_PLATEAU_WINDOW:
            recent_f1 = f1_history[-_F1_PLATEAU_WINDOW:]
            f1_range = max(recent_f1) - min(recent_f1)
            if f1_range < _F1_PLATEAU_THRESHOLD and train_f1 > 0.5:
                logger.info(
                    f"F1 plateau detected: last {_F1_PLATEAU_WINDOW} rounds "
                    f"F1 range={f1_range:.4f} < {_F1_PLATEAU_THRESHOLD}. "
                    f"Stopping refinement."
                )
                break

        # Select rows to verify. When learned deferral is enabled, reserve a
        # decaying beta fraction for uniform DAgger probes and use f(x) to
        # rank the remaining budget once the deferral classifier is fitted.
        random_probe_global = np.zeros(0, dtype=np.int64)
        non_probe_budget = int(refine_batch_size)
        if deferral_enabled and float(deferral_dagger_beta) > 0.0:
            beta_t = max(0.05, float(deferral_dagger_beta) * (0.95 ** round_num))
            probe_n = min(
                len(unlabeled_indices),
                max(0, int(round(non_probe_budget * beta_t))),
            )
            if probe_n > 0:
                rng = np.random.default_rng(42 + round_num)
                random_probe_global = rng.choice(
                    unlabeled_indices, size=probe_n, replace=False
                ).astype(np.int64)
                non_probe_budget = max(0, non_probe_budget - probe_n)
                if prof is not None:
                    prof["deferral_online_random_probe"] += int(probe_n)

        pool_indices = np.setdiff1d(
            unlabeled_indices, random_probe_global, assume_unique=False
        )
        pool_pos = pos_probas[np.isin(unlabeled_indices, pool_indices)]
        if (
            deferral_enabled
            and deferral_clf is not None
            and getattr(deferral_clf, "_fitted", False)
            and non_probe_budget > 0
            and len(pool_indices) > 0
        ):
            from learned_deferral import build_online_deferral_features

            X_pool = build_online_deferral_features(
                pool_pos, indices=pool_indices, total=total
            )
            defer_proba = deferral_clf.predict_defer_proba(X_pool)
            order = np.argsort(-defer_proba)
            verify_main = pool_indices[order[: min(non_probe_budget, len(order))]]
        elif non_probe_budget > 0 and len(pool_indices) > 0:
            verify_main = _select_verify_batch(
                pool_pos,
                pool_indices,
                non_probe_budget,
                confidence_threshold,
                verify_positive,
                num_definite=num_definite,
                num_unknown=num_unknown,
                use_eer=True,
            )
        else:
            verify_main = np.zeros(0, dtype=np.int64)
        verify_global = np.unique(
            np.concatenate([verify_main, random_probe_global]).astype(np.int64)
        )
        if verify_global.size == 0:
            logger.info("No rows selected for online verification; stopping refinement.")
            break
        proba_by_global = dict(zip(unlabeled_indices.tolist(), pos_probas.tolist()))
        pre_verify_probas = np.asarray(
            [proba_by_global[int(i)] for i in verify_global.tolist()], dtype=np.float64
        )
        pre_verify_pred = pre_verify_probas >= float(confidence_threshold)

        # Send to LLM for verification
        verify_texts = [texts[i] for i in verify_global]
        t0_vllm = time.time()
        verify_labels = label_rows_with_llm(
            verify_texts, model, url, system_prompt,
            max_tokens, temperature, max_concurrent_requests,
            budget=budget,
        )
        if prof is not None:
            prof["vllm_invocation_time"] += time.time() - t0_vllm
        if deferral_enabled and deferral_clf is not None:
            from learned_deferral import build_online_deferral_features

            X_new = build_online_deferral_features(
                pre_verify_probas, indices=verify_global, total=total
            )
            z_new = (
                pre_verify_pred.astype(np.int32)
                != np.asarray(verify_labels, dtype=np.int32)
            ).astype(np.int32)
            deferral_clf.partial_fit(X_new, z_new)
            if prof is not None:
                prof["deferral_online_updates"] += int(len(z_new))

        # Update training set
        all_indices.extend(verify_global.tolist())
        all_labels.extend(verify_labels.tolist())
        labeled_set.update(verify_global.tolist())
        new_verified += len(verify_labels)

        new_pos_in_batch = int((verify_labels == 1).sum())
        total_pos = int(np.sum(np.array(all_labels) == 1))
        logger.info(
            f"Online round {round_num + 1}: "
            f"verified {len(verify_labels)} rows, "
            f"found {new_pos_in_batch} positives "
            f"(total positives: {total_pos}, total labeled: {len(all_labels)})"
        )

        # Refit proxy with enriched training set
        all_idx_arr = np.array(all_indices)
        all_lab_arr = np.array(all_labels, dtype=np.int32)
        current_model = fit_proxy_model(
            embeddings[all_idx_arr], all_lab_arr, imbalance_strategy, verbose,
            prof=prof,
        )

        # Early stop if no new positives found and we have enough
        if new_pos_in_batch == 0 and total_pos >= 3:
            logger.info("No new positives found in this round and have sufficient positives. Stopping refinement.")
            break

    final_pos = int(np.sum(np.array(all_labels) == 1))
    logger.info(
        f"Online refinement complete: {initial_pos} → {final_pos} positives "
        f"(+{final_pos - initial_pos} discovered), "
        f"{new_verified} extra LLM calls for verification"
    )
    log_training_metrics(current_model, embeddings[np.array(all_indices)],
                         np.array(all_labels, dtype=np.int32), verbose, prefix="Refined")

    return current_model, np.array(all_indices), np.array(all_labels, dtype=np.int32)


def proxy_filter(
    df: DataFrame,
    prompt: str,
    model: str,
    url: str,
    embedding_model: str,
    sample_size: int = 200,
    sampling_method: str = "random",
    imbalance_strategy: str = "weighted",
    proxy_model_path: str = "",
    proxy_warm_start: bool = False,
    max_tokens: int = 100,
    batch_size: int = 32,
    verbose: bool = False,
    max_concurrent_requests: int = 256,
    temperature: float = 0.0,
    confidence_threshold: float = 0.5,
    max_refine_rounds: int = 100,
    refine_batch_size: int = 32,
    budget_size: Optional[int] = None,
    cluster_proxy_mode: str = "propagation",
    n_clusters: int = 0,
    knn_k: int = 0,
    lr_feature_mode: str = "embedding",
    n_strata: int = 10,
    bas_pilot_ratio: float = 0.15,
    allocation_method: str = "neyman",
    reg_method: str = "exponential",
    reg_strength: float = 0.5,
    auto_k: bool = False,
    refine: bool = False,
    diffusion_alpha: float = 0.8,
    propagation_beta: float = 0.8,
    refine_num_iter: int = 50,
    refine_tol: float = 1e-4,
    refine_patience: int = 5,
    refine_enabled: bool = True,
    activeSampling_B1: int = 5,
    activeSampling_vs_ratio: float = 0.5,
    activeSampling_n_modes: int = 1,
    activeSampling_k_intervals: Optional[int] = None,
    activeSampling_k_auto: bool = False,
    activeSampling_k_max: int = 4,
    activeSampling_k_gain_tol: float = 0.02,
    activeSampling_class_balance: bool = False,
    activeSampling_n_phases: int = 1,
    activeSampling_lr_blend_schedule: str = "linear",
    activeSampling_cold_start_phases: int = 1,
    activeSampling_lr_explore_frac: float = 0.0,
    activeSampling_phase_budget_split: str = "equal",
    defer_enabled: bool = False,
    verify_uncertain: bool = True,
    router_max_calls: Optional[int] = None,
    decision_mode: str = "force_label",
    learned_calibration_config: Optional[Any] = None,
    deferral_features: Optional[Sequence[str]] = None,
    deferral_rank_axis: str = "blended",
    deferral_hidden_dims_grid: Optional[Sequence[Sequence[int]]] = None,
    deferral_alpha_grid: Optional[Sequence[float]] = None,
    deferral_k_folds: int = 5,
    deferral_inner_cv_k: int = 3,
    deferral_dagger_beta: float = 0.00,
    deferral_min_inner_cv_auc: float = 0.60,
    deferral_min_probe_auc: float = 0.60,
    lr_C: float = 1.0,
    base_threshold: Optional[float] = None,
    lr_class_weight_override: Optional[str] = None,
    random_seed: Optional[int] = 42,
    viz_tag: str = "",
) -> DataFrame:
    spark = df.sparkSession
    start_time = time.time()
    prof = _new_profiling()
    budget = _new_budget_tracker(budget_size)

    logger.info(f"Starting proxy filter (sampling_method={sampling_method}, sample_size={sample_size}, budget_size={budget_size})")

    t0_prep = time.time()
    pdf = df.toPandas()
    total_rows = len(pdf)

    if total_rows == 0:
        return df

    columns = _get_all_columns(prompt)
    if not columns:
        columns = pdf.columns.tolist()

    records = pdf.to_dict("records")

    image_cols = _get_image_columns(prompt)

    # Build LLM messages: multimodal content arrays for image prompts,
    # plain strings for text-only prompts.
    texts = [build_row_message(row, prompt, columns, image_cols) for row in records]
    prof["data_prep_time"] = time.time() - t0_prep

    logger.info(f"Embedding {total_rows} rows...")
    t0_emb = time.time()
    if image_cols:
        # Multimodal query — route through the backend's text+image encoders
        # via embed_rows. Works for any backend that supports embed_images
        # (SigLIP, CLIP, Jina-CLIP, GTE-Qwen2-VL). Text-only backends routed
        # here would raise NotImplementedError, but the grid's QUERY_MODALITY
        # dict prevents that mismatch.
        logger.info(f"Using multimodal embedding (image columns: {image_cols})")
        embeddings = embed_multimodal_rows(records, prompt, columns, embedding_model, viz_tag=viz_tag)
    else:
        # Text-only query — plain string inputs for the text embedder.
        text_strings = [t if isinstance(t, str) else " ".join(
            b.get("text", "") for b in t if isinstance(b, dict)
        ) for t in texts]
        embeddings = embed_texts(text_strings, embedding_model)
    prof["embedding_time"] += time.time() - t0_emb

    # ---- Proxy pipeline: stratified / kmeans_stratified / activeSampling / random ----
    _CLUSTER_METHODS = ("stratified", "kmeans_stratified", "activeSampling", "random", "uqe_active", "top_k")
    if sampling_method in _CLUSTER_METHODS:
        from cluster_proxy import cluster_proxy_pipeline

        def _make_label_fn():
            def _label(row_texts: List[Union[str, list]]) -> np.ndarray:
                return label_rows_with_llm(
                    row_texts, model, url,
                    system_prompt=FILTER_SYSTEM_PROMPT,
                    max_tokens=max_tokens, temperature=temperature,
                    max_concurrent_requests=max_concurrent_requests,
                    budget=budget,
                )
            return _label

        sim_features = None
        filter_sims = None
        _sim_methods = ("stratified", "kmeans_stratified", "activeSampling", "top_k")
        if lr_feature_mode == "similarity" or sampling_method in _sim_methods:
            t0_sim = time.time()
            prompt_emb = embed_query(prompt, embedding_model)
            norms_e = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            norms_p = np.linalg.norm(prompt_emb) + 1e-8
            cos_sim = (embeddings @ prompt_emb.T) / (norms_e * norms_p)
            filter_sims = cos_sim.flatten().astype(np.float32)
            if lr_feature_mode == "similarity":
                sim_features = cos_sim.astype(np.float32)
                logger.info(f"Proxy: using similarity features (dim={sim_features.shape[1]})")
            prof["similarity_time"] = time.time() - t0_sim

        positive_mask, cluster_prof = cluster_proxy_pipeline(
            embeddings, texts, _make_label_fn,
            sample_size=sample_size,
            cluster_proxy_mode=cluster_proxy_mode,
            knn_k=knn_k,
            confidence_threshold=confidence_threshold,
            lr_features=sim_features,
            sampling_method=sampling_method,
            similarities=filter_sims,
            n_strata=n_strata,
            bas_pilot_ratio=bas_pilot_ratio,
            allocation_method=allocation_method,
            reg_method=reg_method,
            reg_strength=reg_strength,
            n_clusters=n_clusters,
            auto_k=auto_k,
            refine=refine,
            diffusion_alpha=diffusion_alpha,
            propagation_beta=propagation_beta,
            refine_num_iter=refine_num_iter,
            refine_tol=refine_tol,
            refine_patience=refine_patience,
            refine_enabled=refine_enabled,
            activeSampling_B1=activeSampling_B1,
            activeSampling_vs_ratio=activeSampling_vs_ratio,
            activeSampling_n_modes=activeSampling_n_modes,
            activeSampling_k_intervals=activeSampling_k_intervals,
            activeSampling_k_auto=activeSampling_k_auto,
            activeSampling_k_max=activeSampling_k_max,
            activeSampling_k_gain_tol=activeSampling_k_gain_tol,
            activeSampling_class_balance=activeSampling_class_balance,
            activeSampling_n_phases=activeSampling_n_phases,
            activeSampling_lr_blend_schedule=activeSampling_lr_blend_schedule,
            activeSampling_cold_start_phases=activeSampling_cold_start_phases,
            activeSampling_lr_explore_frac=activeSampling_lr_explore_frac,
            activeSampling_phase_budget_split=activeSampling_phase_budget_split,
            defer_enabled=defer_enabled,
            verify_uncertain=verify_uncertain,
            router_max_calls=router_max_calls,
            decision_mode=decision_mode,
            learned_calibration_config=learned_calibration_config,
            deferral_features=deferral_features,
            deferral_rank_axis=deferral_rank_axis,
            deferral_hidden_dims_grid=deferral_hidden_dims_grid,
            deferral_alpha_grid=deferral_alpha_grid,
            deferral_k_folds=deferral_k_folds,
            deferral_inner_cv_k=deferral_inner_cv_k,
            deferral_dagger_beta=deferral_dagger_beta,
            deferral_min_inner_cv_auc=deferral_min_inner_cv_auc,
            deferral_min_probe_auc=deferral_min_probe_auc,
            lr_C=lr_C,
            base_threshold=base_threshold,
            lr_class_weight_override=lr_class_weight_override,
            random_seed=random_seed,
            viz_tag=viz_tag,
        )

        t0_result = time.time()
        result_pdf = pdf[positive_mask].reset_index(drop=True)
        result_spark_df = _pandas_to_spark(spark, result_pdf, schema=df.schema)
        prof["result_assembly_time"] = time.time() - t0_result

        duration = time.time() - start_time
        prof["total_time"] = duration
        prof["vllm_invocation_time"] = cluster_prof.get("vllm_invocation_time", 0.0)
        prof["lr_training_time"] = cluster_prof.get("lr_training_time", 0.0)
        prof["cluster_profiling"] = cluster_prof
        _set_last_profiling(prof)
        logger.info(f"Cluster proxy filter completed in {duration:.2f}s ({len(result_pdf)}/{total_rows} rows kept)")
        if budget is not None:
            logger.info(f"Oracle budget usage: {budget['used']}/{budget['limit']} vLLM calls")
        _log_profiling(prof)
        return result_spark_df

    if proxy_model_path and os.path.exists(proxy_model_path) and not proxy_warm_start:
        artifact = load_proxy_artifact(proxy_model_path, prompt, embedding_model)
        lr_model = artifact["model"]
        actual_sample_size = 0
    elif proxy_model_path and os.path.exists(proxy_model_path) and proxy_warm_start:
        artifact = load_proxy_artifact(proxy_model_path, prompt, embedding_model)
        prev_embeddings = artifact["training_embeddings"]
        prev_labels = artifact["training_labels"]
        prev_metrics = artifact.get("metrics", {})
        num_updates = artifact["num_updates"]

        logger.info("=== Warm-Start Update ===")
        logger.info(f"Previous training size: {len(prev_labels)} samples")

        sample_idx, sample_labels, _sw = _do_sampling(
            embeddings, texts, prompt, model, url, sample_size,
            sampling_method, embedding_model, max_tokens, temperature,
            max_concurrent_requests, prof=prof, budget=budget,
        )

        combined_emb = np.concatenate([prev_embeddings, embeddings[sample_idx]])
        combined_labels = np.concatenate([prev_labels, sample_labels])
        # Combine weights: previous samples get weight 1.0
        combined_sw = None
        if _sw is not None:
            combined_sw = np.concatenate([np.ones(len(prev_labels)), _sw])

        logger.info(f"New sample size: {len(sample_labels)} samples")
        logger.info(f"Combined training size: {len(combined_labels)} samples")
        if "f1_macro" in prev_metrics:
            logger.info(f"Previous Macro F1: {prev_metrics['f1_macro']:.4f}")

        lr_model = fit_proxy_model(combined_emb, combined_labels, imbalance_strategy, verbose, prof=prof, sample_weight=combined_sw)

        preds = lr_model.predict(combined_emb)
        new_metrics = {
            "accuracy": float(accuracy_score(combined_labels, preds)),
            "f1_macro": float(f1_score(combined_labels, preds, average="macro", zero_division="warn")),
            "log_loss": float(log_loss(combined_labels, lr_model.predict_proba(combined_emb)))
            if len(np.unique(combined_labels)) > 1
            else float("nan"),
        }

        if "f1_macro" in prev_metrics:
            diff = new_metrics["f1_macro"] - prev_metrics["f1_macro"]
            logger.info(f"Updated Macro F1:  {new_metrics['f1_macro']:.4f} ({diff:+.4f})")

        save_proxy_artifact(
            proxy_model_path, lr_model, combined_emb, combined_labels,
            prompt, embedding_model, num_updates + 1, new_metrics,
        )
        actual_sample_size = len(sample_labels)
    else:
        sample_idx, sample_labels, sample_sw = _do_sampling(
            embeddings, texts, prompt, model, url, sample_size,
            sampling_method, embedding_model, max_tokens, temperature,
            max_concurrent_requests, prof=prof, budget=budget,
        )

        actual_sample_size = len(sample_labels)

        lr_model = fit_proxy_model(
            embeddings[sample_idx], sample_labels, imbalance_strategy, verbose,
            prof=prof, sample_weight=sample_sw,
        )

        # Online active learning: only when explicitly requested
        if not _budget_exhausted(budget) and sampling_method == "online":
            logger.info("Online mode requested — starting online refinement...")
            lr_model, sample_idx, sample_labels = _online_refine(
                lr_model, embeddings, texts, sample_idx, sample_labels,
                model, url, FILTER_SYSTEM_PROMPT, confidence_threshold,
                imbalance_strategy, max_tokens, temperature,
                max_concurrent_requests, refine_batch_size,
                max_refine_rounds, verbose=verbose, prof=prof,
                budget=budget,
                deferral_enabled=(decision_mode == "learned_deferral"),
                deferral_hidden_dims=(
                    tuple(deferral_hidden_dims_grid[0])
                    if deferral_hidden_dims_grid
                    else (32, 16)
                ),
                deferral_dagger_beta=deferral_dagger_beta,
            )
            actual_sample_size = len(sample_labels)

        if proxy_model_path:
            preds = lr_model.predict(embeddings[sample_idx])
            metrics = {
                "accuracy": float(accuracy_score(sample_labels, preds)),
                "f1_macro": float(f1_score(sample_labels, preds, average="macro", zero_division="warn")),
                "log_loss": float(log_loss(sample_labels, lr_model.predict_proba(embeddings[sample_idx])))
                if len(np.unique(sample_labels)) > 1
                else float("nan"),
            }
            save_proxy_artifact(
                proxy_model_path, lr_model, embeddings[sample_idx], sample_labels,
                prompt, embedding_model, 0, metrics,
            )

    probas = lr_model.predict_proba(embeddings)
    positive_class_idx = list(lr_model.classes_).index(1) if 1 in lr_model.classes_ else 0
    positive_probas = probas[:, positive_class_idx]

    # Pass proxy model + training data to log quality metrics in summary
    _train_emb = embeddings[sample_idx] if sample_idx is not None else None
    _train_lab = sample_labels if sample_labels is not None else None
    _log_prediction_summary(
        positive_probas, confidence_threshold, total_rows, actual_sample_size,
        proxy_model=lr_model, train_embeddings=_train_emb, train_labels=_train_lab,
    )

    mask = positive_probas >= confidence_threshold
    result_pdf = pdf[mask].reset_index(drop=True)

    duration = time.time() - start_time
    prof["total_time"] = duration
    _set_last_profiling(prof)
    logger.info(f"Proxy filter completed in {duration:.2f}s ({len(result_pdf)}/{total_rows} rows kept)")
    if budget is not None:
        logger.info(f"Oracle budget usage: {budget['used']}/{budget['limit']} vLLM calls")
    _log_profiling(prof)

    return _pandas_to_spark(spark, result_pdf, schema=df.schema)


def proxy_join(
    left_df: DataFrame,
    right_df: DataFrame,
    prompt: str,
    model: str,
    url: str,
    embedding_model: str,
    sample_size: int = 500,
    sampling_method: str = "random",
    imbalance_strategy: str = "weighted",
    proxy_model_path: str = "",
    proxy_warm_start: bool = False,
    max_tokens: int = 16,
    batch_size: int = 32,
    verbose: bool = False,
    max_concurrent_requests: int = 256,
    temperature: float = 0.0,
    true_criteria: str = "True",
    refine_batch_size: int = 32,
    max_refine_rounds: int = 100,
    confidence_threshold: float = 0.5,
    topk_prefilter: bool = False,
    topk_k: int = 0,
    exclude_self_pairs: bool = False,
    join_sim_mode: str = "pair",
    lr_feature_mode: str = "embedding",
    budget_size: Optional[int] = None,
    cluster_proxy_mode: str = "propagation",
    n_clusters: int = 0,
    knn_k: int = 0,
    n_strata: int = 10,
    bas_pilot_ratio: float = 0.15,
    allocation_method: str = "neyman",
    reg_method: str = "exponential",
    reg_strength: float = 0.5,
    auto_k: bool = False,
    refine: bool = False,
    diffusion_alpha: float = 0.8,
    propagation_beta: float = 0.8,
    refine_num_iter: int = 50,
    refine_tol: float = 1e-4,
    refine_patience: int = 5,
    refine_enabled: bool = True,
    activeSampling_B1: int = 5,
    activeSampling_vs_ratio: float = 0.5,
    activeSampling_n_modes: int = 1,
    activeSampling_k_intervals: Optional[int] = None,
    activeSampling_k_auto: bool = False,
    activeSampling_k_max: int = 4,
    activeSampling_k_gain_tol: float = 0.02,
    activeSampling_class_balance: bool = False,
    activeSampling_n_phases: int = 1,
    activeSampling_lr_blend_schedule: str = "linear",
    activeSampling_cold_start_phases: int = 1,
    activeSampling_lr_explore_frac: float = 0.0,
    activeSampling_phase_budget_split: str = "equal",
    defer_enabled: bool = False,
    verify_uncertain: bool = True,
    router_max_calls: Optional[int] = None,
    decision_mode: str = "force_label",
    learned_calibration_config: Optional[Any] = None,
    deferral_features: Optional[Sequence[str]] = None,
    deferral_rank_axis: str = "blended",
    deferral_hidden_dims_grid: Optional[Sequence[Sequence[int]]] = None,
    deferral_alpha_grid: Optional[Sequence[float]] = None,
    deferral_k_folds: int = 5,
    deferral_inner_cv_k: int = 3,
    deferral_dagger_beta: float = 0.00,
    deferral_min_inner_cv_auc: float = 0.60,
    deferral_min_probe_auc: float = 0.60,
    lr_C: float = 1.0,
    base_threshold: Optional[float] = None,
    lr_class_weight_override: Optional[str] = None,
    random_seed: Optional[int] = 42,
    viz_tag: str = "",
) -> DataFrame:
    """Proxy-based semantic join with optional top-K pre-filtering."""
    spark = left_df.sparkSession
    start_time = time.time()
    prof = _new_profiling()
    budget = _new_budget_tracker(budget_size)

    t0_prep = time.time()
    left_pdf = left_df.toPandas()
    right_pdf = right_df.toPandas()
    n_left = len(left_pdf)
    n_right = len(right_pdf)
    total_pairs = n_left * n_right

    logger.info(
        f"Starting proxy join: {n_left} x {n_right} = {total_pairs} pairs "
        f"(sampling_method={sampling_method}, sample_size={sample_size}, budget_size={budget_size}, "
        f"topk_prefilter={topk_prefilter}, lr_feature_mode={lr_feature_mode})"
    )
    if lr_feature_mode not in {"embedding", "similarity", "average"}:
        raise ValueError(
            f"Unsupported lr_feature_mode='{lr_feature_mode}'. "
            "Expected one of: embedding, similarity, average."
        )

    if total_pairs == 0:
        combined_schema = left_df.schema
        for field in right_df.schema:
            combined_schema = combined_schema.add(field)
        return _pandas_to_spark(spark, pd.DataFrame(), schema=combined_schema)

    # Extract columns from both simple {col} and typed {type:col} placeholders
    all_cols = _get_all_columns(prompt)
    left_columns = [c for c in all_cols if c in left_pdf.columns]
    right_columns = [c for c in all_cols if c in right_pdf.columns]
    if not left_columns and not right_columns:
        left_columns = left_pdf.columns.tolist()
        right_columns = right_pdf.columns.tolist()

    left_records = left_pdf.to_dict("records")
    right_records = right_pdf.to_dict("records")

    image_cols = _get_image_columns(prompt)
    prof["data_prep_time"] = time.time() - t0_prep

    # ---- Embedding Phase ----
    # Always embed left and right sides independently. All supported
    # backends (SigLIP, CLIP, Jina-CLIP, GTE-Qwen2-VL, SentenceTransformer,
    # OpenAI) expose the same embed_texts / embed_images interface via
    # proxy_model_backends.get_backend, so embed_rows_independent works
    # for any backend. Per-side embeddings are also required by activeSampling /
    # stratified / kmeans_stratified for pair-similarity sampling and by
    # topk_prefilter.
    t0_emb = time.time()
    logger.info(f"Embedding left ({n_left}) and right ({n_right}) rows independently...")
    indep = embed_rows_independent(
        left_records, right_records, prompt, embedding_model,
        left_columns, right_columns,
        feature_mode="average" if lr_feature_mode == "average" else "concat",
        viz_tag=viz_tag,
    )
    left_embs = indep["left_embs"]
    right_embs = indep["right_embs"]
    left_text_embs = indep["left_text_embs"]
    right_text_embs = indep["right_text_embs"]
    left_img_embs = indep["left_img_embs"]
    right_img_embs = indep["right_img_embs"]

    # ---- Top-K Pre-filtering ----
    if topk_prefilter and left_embs is not None:
        k = topk_k if topk_k > 0 else max(10, int(max(n_left, n_right) * 0.2))
        pair_indices = topk_prefilter_pairs(
            left_embs, right_embs, k,
            left_text_embs, right_text_embs,
            left_img_embs, right_img_embs,
        )
        effective_pairs = len(pair_indices)
    else:
        pair_indices = [(i, j) for i in range(n_left) for j in range(n_right)]
        effective_pairs = total_pairs

    if exclude_self_pairs and n_left == n_right:
        n_before = len(pair_indices)
        pair_indices = [(i, j) for (i, j) in pair_indices if i != j]
        n_after = len(pair_indices)
        logger.info(
            f"exclude_self_pairs=True: dropped {n_before - n_after} reflexive pairs "
            f"from cartesian product (now {n_after} pairs)"
        )
        effective_pairs = len(pair_indices)
        total_pairs = effective_pairs
        effective_pairs = total_pairs

    # ---- Build LR features ----
    # pair_embeddings: features used for LR proxy training
    # sampling_embeddings: per-pair embeddings used for similarity / KNN
    #   (must match prompt embedding dimension for cosine similarity)
    if lr_feature_mode == "similarity":
        logger.info(f"Building similarity features for {effective_pairs} pairs...")
        pair_embeddings = compute_similarity_features(
            pair_indices, left_embs, right_embs,
            left_text_embs, right_text_embs,
            left_img_embs, right_img_embs,
        )
        logger.info(f"Similarity feature dim: {pair_embeddings.shape[1]}")
        sampling_embeddings = _build_pair_embeddings_from_independent(
            pair_indices, left_embs, right_embs,
        )
    else:
        logger.info(f"Building fused embedding features for {effective_pairs} pairs...")
        pair_embeddings = _build_pair_embeddings_from_independent(
            pair_indices, left_embs, right_embs,
        )
        logger.info(f"Fused embedding dim: {pair_embeddings.shape[1]}")
        sampling_embeddings = pair_embeddings

    prof["embedding_time"] += time.time() - t0_emb

    # Build LLM messages for the candidate pairs (for sampling/labeling)
    t0_pair_msg = time.time()
    pair_texts = [
        build_pair_message(
            left_records[i], right_records[j], prompt,
            left_columns, right_columns, image_cols,
        )
        for i, j in pair_indices
    ]
    prof["pair_message_time"] = time.time() - t0_pair_msg

    # ---- Cluster / stratified / kmeans_stratified / galaxy / activeSampling proxy pipeline ----
    _CLUSTER_JOIN_METHODS = ("stratified", "kmeans_stratified", "activeSampling", "random", "uqe_active", "top_k")
    if sampling_method in _CLUSTER_JOIN_METHODS:
        from cluster_proxy import cluster_proxy_pipeline

        def _make_join_label_fn():
            def _label(row_texts: List[Union[str, list]]) -> np.ndarray:
                return label_rows_with_llm(
                    row_texts, model, url,
                    system_prompt=JOIN_SYSTEM_PROMPT,
                    max_tokens=max_tokens, temperature=temperature,
                    max_concurrent_requests=max_concurrent_requests,
                    budget=budget,
                )
            return _label

        sim_features = None
        join_sims = None
        _sim_join_methods = ("stratified", "kmeans_stratified", "activeSampling", "top_k")
        if lr_feature_mode == "similarity" and pair_embeddings is not sampling_embeddings:
            sim_features = pair_embeddings
            logger.info(f"Cluster proxy join: using similarity features (dim={sim_features.shape[1]})")
        if sampling_method in _sim_join_methods and left_embs is not None and right_embs is not None:
            t0_sim = time.time()

            # Vectorized index extraction (~50× faster than the
            # comprehension when n_pairs is large).
            pair_arr = np.asarray(pair_indices, dtype=np.int64)
            li = np.ascontiguousarray(pair_arr[:, 0])
            ri = np.ascontiguousarray(pair_arr[:, 1])
            n_pairs = li.shape[0]
            n_left, D_left = left_embs.shape
            n_right, D_right = right_embs.shape

            # Precompute per-row L2 norms (and squared norms) ONCE over
            # n_left + n_right rows — the cosine branches below only
            # need to look these up by index, never recompute.
            left_sq_norms = np.einsum("ij,ij->i", left_embs, left_embs).astype(np.float32)
            right_sq_norms = np.einsum("ij,ij->i", right_embs, right_embs).astype(np.float32)
            left_norms_full = np.sqrt(left_sq_norms) + np.float32(1e-8)
            right_norms_full = np.sqrt(right_sq_norms) + np.float32(1e-8)

            # Pairwise dot matrix M[i, j] = left[i] · right[j]. Built
            # via a single BLAS gemm: O(n_left · n_right · D) flops,
            # O(n_left · n_right · 4) bytes. For typical sem-join
            # sizes (n_left·n_right ≤ 100M) this is faster *and*
            # cheaper in memory than materializing (N, D) gather
            # arrays — the gather costs N·D·4 bytes which usually
            # dominates because n_left·n_right ≤ N anyway.
            #
            # Threshold guards the pathological case where pairs come
            # from a sparse / arbitrary list and n_left·n_right ≫ N.
            # Above the threshold we fall back to a chunked per-pair
            # dot product via einsum (still avoiding the full (N, D)
            # gather alloc).
            _PAIRWISE_FULL_MAX = int(os.environ.get(
                "LLMSQL_JOIN_PAIRWISE_MAX", str(200_000_000)
            ))
            use_full_matrix = (n_left * n_right) <= _PAIRWISE_FULL_MAX

            if use_full_matrix:
                t0_mm = time.time()
                # float32 GEMM via BLAS (numpy uses MKL/OpenBLAS).
                M = np.ascontiguousarray(left_embs).astype(np.float32) @ \
                    np.ascontiguousarray(right_embs).astype(np.float32).T
                _mm_dt = time.time() - t0_mm
            else:
                M = None
                _mm_dt = 0.0

            def _per_pair_lr_dot() -> np.ndarray:
                """per-pair l[li] · r[ri] — uses M if available, else einsum chunks."""
                if M is not None:
                    return M[li, ri]
                out = np.empty(n_pairs, dtype=np.float32)
                chunk = 1_000_000
                for s in range(0, n_pairs, chunk):
                    e = min(s + chunk, n_pairs)
                    out[s:e] = np.einsum(
                        "ij,ij->i",
                        left_embs[li[s:e]],
                        right_embs[ri[s:e]],
                    ).astype(np.float32, copy=False)
                return out

            if join_sim_mode == "prompt_vs_fused":
                # cos(prompt, (l[li] + r[ri]) / 2) without ever
                # materializing the (N, D) fused matrix.
                #
                #   prompt · fused[k] = (lp[li[k]] + rp[ri[k]]) / 2
                #   ||fused[k]||² = (||l[li]||² + ||r[ri]||² + 2·l[li]·r[ri]) / 4
                prompt_emb = embed_query(prompt, embedding_model)
                p_vec = prompt_emb.reshape(-1)
                d = min(len(p_vec), D_left)
                p_slice = p_vec[:d].astype(np.float32, copy=False)
                # If d == D_left we can use the precomputed M / norms;
                # otherwise we slice the embeddings to dim d and redo.
                if d == D_left and d == D_right:
                    lp = left_embs.astype(np.float32, copy=False) @ p_slice
                    rp = right_embs.astype(np.float32, copy=False) @ p_slice
                    ll_sq = left_sq_norms
                    rr_sq = right_sq_norms
                else:
                    L = left_embs[:, :d].astype(np.float32, copy=False)
                    R = right_embs[:, :d].astype(np.float32, copy=False)
                    lp = L @ p_slice
                    rp = R @ p_slice
                    ll_sq = np.einsum("ij,ij->i", L, L).astype(np.float32)
                    rr_sq = np.einsum("ij,ij->i", R, R).astype(np.float32)
                lr_dot = _per_pair_lr_dot()
                dots = (lp[li] + rp[ri]) * np.float32(0.5)
                fused_norm_sq = (ll_sq[li] + rr_sq[ri] + 2.0 * lr_dot) * np.float32(0.25)
                fused_norm = np.sqrt(np.maximum(fused_norm_sq, 0.0)) + np.float32(1e-8)
                norm_p = float(np.linalg.norm(p_slice)) + 1e-8
                join_sims = (dots / (fused_norm * norm_p)).astype(np.float32)
                logger.info(
                    f"join_sim_mode=prompt_vs_fused: cos(prompt, (left+right)/2) "
                    f"for N={n_pairs} pairs (dim={d}) | "
                    f"pairwise_matmul={_mm_dt*1000:.0f}ms "
                    f"path={'full_matrix' if M is not None else 'chunked_einsum'}"
                )
            else:
                # cos(l[li], r[ri]) — same trick.
                dots = _per_pair_lr_dot()
                norms_l = left_norms_full[li]
                norms_r = right_norms_full[ri]
                join_sims = (dots / (norms_l * norms_r)).astype(np.float32)
                logger.info(
                    f"Computed pair cosine similarities for stratified sampling "
                    f"(N={n_pairs}) | pairwise_matmul={_mm_dt*1000:.0f}ms "
                    f"path={'full_matrix' if M is not None else 'chunked_einsum'}"
                )
            prof["similarity_time"] = time.time() - t0_sim

        positive_mask, cluster_prof = cluster_proxy_pipeline(
            sampling_embeddings, pair_texts, _make_join_label_fn,
            sample_size=sample_size,
            cluster_proxy_mode=cluster_proxy_mode,
            knn_k=knn_k,
            confidence_threshold=confidence_threshold,
            lr_features=sim_features,
            sampling_method=sampling_method,
            similarities=join_sims,
            n_strata=n_strata,
            bas_pilot_ratio=bas_pilot_ratio,
            allocation_method=allocation_method,
            reg_method=reg_method,
            reg_strength=reg_strength,
            n_clusters=n_clusters,
            auto_k=auto_k,
            refine=refine,
            diffusion_alpha=diffusion_alpha,
            propagation_beta=propagation_beta,
            refine_num_iter=refine_num_iter,
            refine_tol=refine_tol,
            refine_patience=refine_patience,
            refine_enabled=refine_enabled,
            activeSampling_B1=activeSampling_B1,
            activeSampling_vs_ratio=activeSampling_vs_ratio,
            activeSampling_n_modes=activeSampling_n_modes,
            activeSampling_k_intervals=activeSampling_k_intervals,
            activeSampling_k_auto=activeSampling_k_auto,
            activeSampling_k_max=activeSampling_k_max,
            activeSampling_k_gain_tol=activeSampling_k_gain_tol,
            activeSampling_class_balance=activeSampling_class_balance,
            activeSampling_n_phases=activeSampling_n_phases,
            activeSampling_lr_blend_schedule=activeSampling_lr_blend_schedule,
            activeSampling_cold_start_phases=activeSampling_cold_start_phases,
            activeSampling_lr_explore_frac=activeSampling_lr_explore_frac,
            activeSampling_phase_budget_split=activeSampling_phase_budget_split,
            defer_enabled=defer_enabled,
            verify_uncertain=verify_uncertain,
            router_max_calls=router_max_calls,
            decision_mode=decision_mode,
            learned_calibration_config=learned_calibration_config,
            deferral_features=deferral_features,
            deferral_rank_axis=deferral_rank_axis,
            deferral_hidden_dims_grid=deferral_hidden_dims_grid,
            deferral_alpha_grid=deferral_alpha_grid,
            deferral_k_folds=deferral_k_folds,
            deferral_inner_cv_k=deferral_inner_cv_k,
            deferral_dagger_beta=deferral_dagger_beta,
            deferral_min_inner_cv_auc=deferral_min_inner_cv_auc,
            deferral_min_probe_auc=deferral_min_probe_auc,
            lr_C=lr_C,
            base_threshold=base_threshold,
            lr_class_weight_override=lr_class_weight_override,
            random_seed=random_seed,
            viz_tag=viz_tag,
        )

        t0_result = time.time()
        matched_indices = [pair_indices[i] for i in range(effective_pairs) if positive_mask[i]]

        if not matched_indices:
            from pyspark.sql.types import StructType, StructField, StringType
            left_cols = list(left_pdf.columns)
            right_cols = list(right_pdf.columns)
            overlapping = set(left_cols) & set(right_cols)
            fields = []
            for c in left_cols:
                name = f"left_{c}" if c in overlapping else c
                fields.append(StructField(name, StringType(), True))
            for c in right_cols:
                name = f"right_{c}" if c in overlapping else c
                fields.append(StructField(name, StringType(), True))
            schema = StructType(fields)
            jvm = spark._jvm
            jschema = spark._jsparkSession.parseDataType(schema.json())
            empty_rdd = jvm.java.util.ArrayList()
            jdf = spark._jsparkSession.createDataFrame(empty_rdd, jschema)
            from pyspark.sql import DataFrame as SparkDataFrame
            result_spark_df = SparkDataFrame(jdf, spark)
        else:
            left_cols = list(left_pdf.columns)
            right_cols = list(right_pdf.columns)
            overlapping = set(left_cols) & set(right_cols)
            rows = []
            for li, ri in matched_indices:
                left_row = {(f"left_{c}" if c in overlapping else c): v for c, v in left_records[li].items()}
                right_row = {(f"right_{c}" if c in overlapping else c): v for c, v in right_records[ri].items()}
                rows.append({**left_row, **right_row})

            result_pdf = pd.DataFrame(rows)
            logger.info(f"{len(result_pdf)}/{effective_pairs} candidate pairs matched")
            result_spark_df = _pandas_to_spark(spark, result_pdf)
        prof["result_assembly_time"] = time.time() - t0_result

        duration = time.time() - start_time
        prof["total_time"] = duration
        prof["vllm_invocation_time"] = cluster_prof.get("vllm_invocation_time", 0.0)
        prof["lr_training_time"] = cluster_prof.get("lr_training_time", 0.0)
        prof["cluster_profiling"] = cluster_prof
        _set_last_profiling(prof)
        logger.info(
            f"Cluster proxy join completed in {duration:.2f}s "
            f"({len(matched_indices)}/{effective_pairs} pairs matched)"
        )
        if budget is not None:
            logger.info(f"Oracle budget usage: {budget['used']}/{budget['limit']} vLLM calls")
        _log_profiling(prof)
        return result_spark_df

    # ---- Proxy Model Training ----
    if proxy_model_path and os.path.exists(proxy_model_path) and not proxy_warm_start:
        artifact = load_proxy_artifact(proxy_model_path, prompt, embedding_model)
        lr_model = artifact["model"]
        actual_sample_size = 0
    elif proxy_model_path and os.path.exists(proxy_model_path) and proxy_warm_start:
        artifact = load_proxy_artifact(proxy_model_path, prompt, embedding_model)
        prev_embeddings = artifact["training_embeddings"]
        prev_labels = artifact["training_labels"]
        prev_metrics = artifact.get("metrics", {})
        num_updates = artifact["num_updates"]

        logger.info("=== Warm-Start Update ===")
        logger.info(f"Previous training size: {len(prev_labels)} samples")

        s_idx, s_labels, _sw = _do_sampling(
            sampling_embeddings, pair_texts, prompt, model, url, sample_size,
            sampling_method, embedding_model, max_tokens, temperature,
            max_concurrent_requests, system_prompt=JOIN_SYSTEM_PROMPT, prof=prof,
            budget=budget,
        )

        combined_emb = np.concatenate([prev_embeddings, pair_embeddings[s_idx]])
        combined_labels = np.concatenate([prev_labels, s_labels])
        combined_sw = None
        if _sw is not None:
            combined_sw = np.concatenate([np.ones(len(prev_labels)), _sw])

        lr_model = fit_proxy_model(combined_emb, combined_labels, imbalance_strategy, verbose, prof=prof, sample_weight=combined_sw)

        preds = lr_model.predict(combined_emb)
        new_metrics = {
            "accuracy": float(accuracy_score(combined_labels, preds)),
            "f1_macro": float(f1_score(combined_labels, preds, average="macro", zero_division="warn")),
            "log_loss": float(log_loss(combined_labels, lr_model.predict_proba(combined_emb)))
            if len(np.unique(combined_labels)) > 1
            else float("nan"),
        }

        save_proxy_artifact(
            proxy_model_path, lr_model, combined_emb, combined_labels,
            prompt, embedding_model, num_updates + 1, new_metrics,
        )
        actual_sample_size = len(s_labels)
    else:
        s_idx, s_labels, s_sw = _do_sampling(
            sampling_embeddings, pair_texts, prompt, model, url, sample_size,
            sampling_method, embedding_model, max_tokens, temperature,
            max_concurrent_requests, system_prompt=JOIN_SYSTEM_PROMPT, prof=prof,
            budget=budget,
        )

        actual_sample_size = len(s_labels)

        lr_model = fit_proxy_model(
            pair_embeddings[s_idx], s_labels, imbalance_strategy, verbose,
            prof=prof, sample_weight=s_sw,
        )

        # Online active learning: only when explicitly requested
        if not _budget_exhausted(budget) and sampling_method == "online":
            logger.info("Online mode requested — starting online refinement...")
            lr_model, s_idx, s_labels = _online_refine(
                lr_model, pair_embeddings, pair_texts, s_idx, s_labels,
                model, url, JOIN_SYSTEM_PROMPT, confidence_threshold,
                imbalance_strategy, max_tokens, temperature,
                max_concurrent_requests, refine_batch_size,
                max_refine_rounds, verbose=verbose, prof=prof,
                budget=budget,
            )
            actual_sample_size = len(s_labels)

        if proxy_model_path:
            preds = lr_model.predict(pair_embeddings[s_idx])
            metrics = {
                "accuracy": float(accuracy_score(s_labels, preds)),
                "f1_macro": float(f1_score(s_labels, preds, average="macro", zero_division="warn")),
                "log_loss": float(log_loss(s_labels, lr_model.predict_proba(pair_embeddings[s_idx])))
                if len(np.unique(s_labels)) > 1
                else float("nan"),
            }
            save_proxy_artifact(
                proxy_model_path, lr_model, pair_embeddings[s_idx], s_labels,
                prompt, embedding_model, 0, metrics,
            )

    # ---- Prediction ----
    probas = lr_model.predict_proba(pair_embeddings)
    positive_class_idx = list(lr_model.classes_).index(1) if 1 in lr_model.classes_ else 0
    positive_probas = probas[:, positive_class_idx]

    _train_emb_j = pair_embeddings[s_idx] if s_idx is not None else None
    _train_lab_j = s_labels if s_labels is not None else None
    _log_prediction_summary(
        positive_probas,
        confidence_threshold,
        effective_pairs,
        actual_sample_size,
        total_should_call=total_pairs,
        proxy_model=lr_model, train_embeddings=_train_emb_j, train_labels=_train_lab_j,
    )

    duration = time.time() - start_time
    prof["total_time"] = duration
    _set_last_profiling(prof)
    logger.info(
        f"Proxy join completed in {duration:.2f}s "
        f"(topk={topk_prefilter}, lr_mode={lr_feature_mode}, "
        f"candidates={effective_pairs}/{total_pairs})"
    )
    if budget is not None:
        logger.info(f"Oracle budget usage: {budget['used']}/{budget['limit']} vLLM calls")
    _log_profiling(prof)

    matched_mask = positive_probas >= confidence_threshold
    matched_indices = [pair_indices[i] for i in range(effective_pairs) if matched_mask[i]]

    if not matched_indices:
        max_proba = float(np.max(positive_probas)) if len(positive_probas) > 0 else float("nan")
        if confidence_threshold > 0.5:
            logger.warning(
                "No proxy-positive matches at current threshold "
                f"({confidence_threshold:.2f}); max positive probability={max_proba:.4f}. "
                "Consider lowering confidence_threshold (e.g. 0.5~0.8) for recall."
            )
        logger.info("No pairs matched the confidence threshold.")
        from pyspark.sql.types import StructType, StructField, StringType
        left_cols = list(left_pdf.columns)
        right_cols = list(right_pdf.columns)
        overlapping = set(left_cols) & set(right_cols)
        fields = []
        for c in left_cols:
            name = f"left_{c}" if c in overlapping else c
            fields.append(StructField(name, StringType(), True))
        for c in right_cols:
            name = f"right_{c}" if c in overlapping else c
            fields.append(StructField(name, StringType(), True))
        schema = StructType(fields)
        jvm = spark._jvm
        jschema = spark._jsparkSession.parseDataType(schema.json())
        empty_rdd = jvm.java.util.ArrayList()
        jdf = spark._jsparkSession.createDataFrame(empty_rdd, jschema)
        from pyspark.sql import DataFrame
        return DataFrame(jdf, spark)

    left_cols = list(left_pdf.columns)
    right_cols = list(right_pdf.columns)
    overlapping = set(left_cols) & set(right_cols)
    rows = []
    for li, ri in matched_indices:
        left_row = {(f"left_{c}" if c in overlapping else c): v for c, v in left_records[li].items()}
        right_row = {(f"right_{c}" if c in overlapping else c): v for c, v in right_records[ri].items()}
        rows.append({**left_row, **right_row})

    result_pdf = pd.DataFrame(rows)
    logger.info(f"{len(result_pdf)}/{effective_pairs} candidate pairs matched")

    return _pandas_to_spark(spark, result_pdf)


def _do_sampling(
    embeddings: np.ndarray,
    texts: List[Union[str, list]],
    prompt: str,
    model: str,
    url: str,
    sample_size: int,
    sampling_method: str,
    embedding_model: str,
    max_tokens: int = 16,
    temperature: float = 0.0,
    max_concurrent_requests: int = 256,
    system_prompt: str = FILTER_SYSTEM_PROMPT,
    prof: Optional[dict] = None,
    budget: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Returns (indices, labels, sample_weights).
    sample_weights is non-None only for stratified sampling (active/online)."""
    total = len(texts)
    sample_size = min(sample_size, total)

    if sampling_method in ("active", "online"):
        # Adaptive stratified active sampling with importance weighting
        # and Neyman variance-based budget allocation.
        image_cols = _get_image_columns(prompt)
        if image_cols and _is_siglip_model(embedding_model):
            text_dim = _siglip_text_dim(embedding_model)
            prompt_emb = embed_texts_siglip([prompt], embedding_model)[0]
            row_emb = embeddings[:, :text_dim]
        else:
            prompt_emb = embed_query(prompt, embedding_model)[0]
            row_emb = embeddings
        t0 = time.time()
        indices, labels, sample_weights = sample_stratified(
            row_emb, prompt_emb, texts, sample_size,
            model, url, system_prompt,
            max_tokens, temperature, max_concurrent_requests,
            budget=budget,
        )
        if prof is not None:
            prof["vllm_invocation_time"] += time.time() - t0
        return indices, labels, sample_weights

    if sampling_method == "topk":
        image_cols = _get_image_columns(prompt)
        if image_cols and _is_siglip_model(embedding_model):
            text_dim = _siglip_text_dim(embedding_model)
            text_subspace = embeddings[:, :text_dim]
            prompt_emb = embed_texts_siglip([prompt], embedding_model)
            indices = sample_topk(text_subspace, prompt_emb[0], sample_size)
            logger.info(f"topk sampling on text subspace (dim={text_dim}) of multimodal SigLIP embeddings")
        else:
            prompt_emb = embed_query(prompt, embedding_model)
            indices = sample_topk(embeddings, prompt_emb[0], sample_size)
    else:
        indices = sample_random(total, sample_size)

    sample_texts = [texts[i] for i in indices]
    t0 = time.time()
    labels = label_rows_with_llm(
        sample_texts, model, url, system_prompt,
        max_tokens, temperature, max_concurrent_requests,
        budget=budget,
    )
    if prof is not None:
        prof["vllm_invocation_time"] += time.time() - t0
    return indices, labels, None  # no importance weights for random/topk
