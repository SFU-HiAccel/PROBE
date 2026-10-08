"""Embedding model backend registry for LLMSQL proxy pipeline.

Each backend wraps one embedding model family and exposes a uniform interface:
  .embed_texts(texts, batch_size) -> np.ndarray  (N, D) float32, L2-normalised
  .embed_images(paths, batch_size) -> np.ndarray (N, D) float32, L2-normalised
  .shared_space: bool  -- True means text+image encoders project to the same
                          contrastive latent space so mean-pool fusion is valid.

Backends are lazy-loaded and cached in _BACKEND_CACHE keyed by model_id.

Supported model families
------------------------
  SigLIP / SigLIP2  : google/siglip-base-patch16-224 (local dir)
  CLIP              : openai/clip-vit-base-patch32, openai/clip-vit-large-patch14
  Jina-CLIP v2      : jinaai/jina-clip-v2
  GTE-Qwen2-VL      : Alibaba-NLP/gte-Qwen2-VL-7B-Instruct  (7B, opt-in)
  SentenceTransformer: any local ST model, e.g. all-MiniLM-L12-v2 (text-only)
  OpenAI / HTTP API : any model_id recognised by the OpenAI embeddings endpoint
"""
from __future__ import annotations

import os
# Set BEFORE any transformers/tokenizers import in this module. Harmless if
# the entry point already set it; otherwise this suppresses the fork warning
# that appears when PySpark (or any other fork-based parallelism) spawns
# workers after a Rust-backed tokenizer has already been initialised.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKEND_CACHE: dict = {}


def get_backend(model_id: str):
    """Return (and cache) the appropriate backend for *model_id*."""
    if model_id not in _BACKEND_CACHE:
        _BACKEND_CACHE[model_id] = _create_backend(model_id)
    return _BACKEND_CACHE[model_id]


def release_backend(model_id: str) -> None:
    """Drop one backend from the cache and free its GPU memory.

    Use between embedding-model swaps in ablation grids to avoid
    accumulating multiple multi-GB models on the same GPU.
    """
    entry = _BACKEND_CACHE.pop(model_id, None)
    if entry is not None:
        for attr in (
            "_model", "_text_model", "_vision_model",
            "_processor", "_tokenizer",
        ):
            if hasattr(entry, attr):
                setattr(entry, attr, None)
    # Also drop the SigLIP-specific cache in proxy_model (kept separate
    # because SigLIPBackend delegates to legacy module-level helpers).
    try:
        from proxy_model import release_siglip_cache_entry
        release_siglip_cache_entry(model_id)
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_all_backends() -> None:
    """Drop every cached backend. Useful after a query is done."""
    for mid in list(_BACKEND_CACHE.keys()):
        release_backend(mid)


def emb_short(model_id: str) -> str:
    """Short slug for result tags and filenames (strips org prefix, caps at 30 chars)."""
    name = os.path.basename(model_id.rstrip("/"))
    name = re.sub(
        r"^(google|openai|jinaai|Alibaba-NLP|sentence-transformers)/", "", name
    )
    return name[:30]


# ---------------------------------------------------------------------------
# Backend detection helpers
# ---------------------------------------------------------------------------

def _read_config_model_type(model_path: str) -> str:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return ""
    try:
        import json
        with open(config_path) as f:
            return json.load(f).get("model_type", "").lower()
    except Exception:
        return ""


def _safe_open_image(path: str, min_size: int = 32, fallback_size: int = 448):
    """Open an image path robustly for any Qwen2-VL-style processor.

    * Broken or missing files return a blank fallback_size × fallback_size RGB.
    * Images smaller than (min_size × min_size) are upscaled with BICUBIC so
      they don't trip Qwen2-VL's smart_resize("height/width must be larger
      than factor:28") check. min_size=32 leaves a small safety margin over
      the factor=28 requirement; fallback_size=448 matches Qwen2-VL's
      default spatial_factor.
    """
    from PIL import Image
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w < min_size or h < min_size:
            scale = max(min_size / max(w, 1), min_size / max(h, 1))
            new_w = max(min_size, int(round(w * scale)))
            new_h = max(min_size, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return img
    except Exception:
        return Image.new("RGB", (fallback_size, fallback_size))


def _default_use_fast_true(*processor_classes):
    """Context manager: make the given Auto* processor classes default to
    use_fast=True inside the `with` block. Used when loading models whose
    remote (trust_remote_code) code instantiates processors without passing
    use_fast, producing a "Using a slow image processor" warning.
    """
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        originals = []
        for cls in processor_classes:
            orig = cls.__dict__.get("from_pretrained")
            if orig is None:
                continue
            originals.append((cls, orig))
            unwrapped = orig.__func__ if isinstance(orig, classmethod) else orig

            def _make_patched(fn):
                def _patched(c, *args, **kwargs):
                    kwargs.setdefault("use_fast", True)
                    return fn(c, *args, **kwargs)
                return _patched

            cls.from_pretrained = classmethod(_make_patched(unwrapped))
        try:
            yield
        finally:
            for cls, orig in originals:
                cls.from_pretrained = orig

    return _cm()


def _is_jina_v4(lower: str) -> bool:
    return "jina-embeddings-v4" in lower or "jina_embeddings_v4" in lower


def _is_jina_clip(lower: str) -> bool:
    # jina-clip-v1 / jina-clip-v2 (but NOT jina-embeddings-v4, which is checked first)
    return "jina-clip" in lower or "jina_clip" in lower


def _is_gte_qwen_vl(lower: str) -> bool:
    return "gte-qwen2-vl" in lower or "gte_qwen2_vl" in lower


def _is_gme_qwen_vl(lower: str) -> bool:
    return "gme-qwen2-vl" in lower or "gme_qwen2_vl" in lower


def _is_gte_qwen2_text(lower: str) -> bool:
    # Match gte-Qwen2-7B-instruct (text-only) but NOT gte-Qwen2-VL-*
    return "gte-qwen2" in lower and "vl" not in lower


def _is_blip(model_id: str, lower: str) -> bool:
    if "salesforce/blip" in lower or "blip-itm" in lower:
        return True
    if os.path.isdir(model_id):
        mt = _read_config_model_type(model_id)
        return mt in ("blip", "blip-2", "blip_image_text_matching")
    return False


def _is_clip(model_id: str, lower: str) -> bool:
    if "openai/clip" in lower:
        return True
    if os.path.isdir(model_id):
        return _read_config_model_type(model_id) == "clip"
    return False


def _is_siglip(model_id: str, lower: str) -> bool:
    if "siglip" in lower:
        return True
    if os.path.isdir(model_id):
        mt = _read_config_model_type(model_id)
        return "siglip" in mt
    return False


def _create_backend(model_id: str):
    lower = model_id.lower()
    if _is_jina_v4(lower):
        return JinaEmbeddingsV4Backend(model_id)
    if _is_jina_clip(lower):
        return JinaClipBackend(model_id)
    if _is_gme_qwen_vl(lower):
        return GMEQwen2VLBackend(model_id)
    if _is_gte_qwen_vl(lower):
        return GTEQwenVLBackend(model_id)
    if _is_gte_qwen2_text(lower):
        return GTEQwen2Backend(model_id)
    if _is_blip(model_id, lower):
        return BLIPBackend(model_id)
    if _is_clip(model_id, lower):
        return CLIPBackend(model_id)
    if _is_siglip(model_id, lower):
        return SigLIPBackend(model_id)
    if os.path.isdir(model_id):
        return SentenceTransformerBackend(model_id)
    # Non-local path: OpenAI-compatible API or HF model name via ST
    if lower.startswith("sentence-transformers/") or lower.startswith("all-"):
        return SentenceTransformerBackend(model_id)
    return OpenAIBackend(model_id)


# ---------------------------------------------------------------------------
# SigLIPBackend — delegates to proxy_model's existing siglip functions.
# The lazy import (inside each method) avoids a circular import since
# proxy_model.py imports get_backend from this module.
# ---------------------------------------------------------------------------

class SigLIPBackend:
    shared_space = True  # text+image share the same contrastive 768-dim space

    def __init__(self, model_id: str):
        self.model_id = model_id

    def embed_texts(self, texts: List[str], batch_size: int = 512) -> np.ndarray:
        from proxy_model import embed_texts_siglip  # lazy — avoids circular import
        return embed_texts_siglip(texts, self.model_id, batch_size)

    def embed_images(self, paths: List[str], batch_size: int = 256) -> np.ndarray:
        from proxy_model import embed_images_siglip  # lazy
        return embed_images_siglip(paths, self.model_id, batch_size)


# ---------------------------------------------------------------------------
# CLIPBackend — openai/clip-vit-* via transformers CLIPModel
# ---------------------------------------------------------------------------

class CLIPBackend:
    shared_space = True  # contrastive-trained shared text+image space

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor
        device = os.environ.get("PROBE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.startswith("cuda") else None
        self._model = CLIPModel.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_id, use_fast=True)
        self._tokenizer = self._processor.tokenizer
        self._device = device

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self._load()
        import torch
        from proxy_model import _extract_features_tensor  # lazy — avoids circular import
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=77, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad(), torch.amp.autocast(self._device.split(':')[0]):
                feats = self._model.get_text_features(**inputs)
            feats = _extract_features_tensor(feats)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            results.append(feats.float().cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)

    def embed_images(self, paths: List[str], batch_size: int = 64) -> np.ndarray:
        self._load()
        import torch
        from PIL import Image
        from proxy_model import _extract_features_tensor  # lazy
        results = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    if os.environ.get("PROBE_PAPER_JOB"):
                        raise
                    imgs.append(Image.new("RGB", (224, 224)))
            inputs = self._processor(images=imgs, return_tensors="pt").to(self._device)
            with torch.no_grad(), torch.amp.autocast(self._device.split(':')[0]):
                feats = self._model.get_image_features(**inputs)
            feats = _extract_features_tensor(feats)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            results.append(feats.float().cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# JinaClipBackend — jinaai/jina-clip-v2 (1024-dim, trust_remote_code=True)
# ---------------------------------------------------------------------------

class JinaClipBackend:
    shared_space = True  # contrastive-trained shared text+image space

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else None,
        ).to(device).eval()
        self._processor = AutoImageProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._device = device

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self._load()
        import torch
        from proxy_model import _extract_features_tensor  # lazy
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                out = self._model.get_text_features(**inputs)
            out = _extract_features_tensor(out)
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
            results.append(out.float().cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)

    def embed_images(self, paths: List[str], batch_size: int = 64) -> np.ndarray:
        self._load()
        import torch
        from PIL import Image
        from proxy_model import _extract_features_tensor  # lazy
        results = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (224, 224)))
            inputs = self._processor(images=imgs, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model.get_image_features(**inputs)
            out = _extract_features_tensor(out)
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
            results.append(out.float().cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# JinaEmbeddingsV4Backend — jinaai/jina-embeddings-v4 (2048-dim, task-switched)
#
# Distinct from jina-clip-v2 (which uses get_text_features / get_image_features).
# V4 uses encode_text / encode_image with task-specific adapters:
#   task ∈ {"retrieval", "text-matching", "code"}
#   prompt_name ∈ {"query", "passage"}   — only for the retrieval/code tasks
#
# Requirements:
#   transformers >= 4.52.0  (conflicts with gme-Qwen2-VL which needs <4.52.0)
#   peft >= 0.15.2
#
# Env vars:
#   LLMSQL_JINA_V4_TASK         default task (default "retrieval")
#   LLMSQL_JINA_V4_TRUNCATE_DIM Matryoshka dim ∈ {128,256,512,1024,2048}
# ---------------------------------------------------------------------------

class JinaEmbeddingsV4Backend:
    shared_space = True  # retrieval task aligns text and image in one space

    DEFAULT_TASK = "retrieval"

    # VRAM-aware defaults. Jina v4 is ~3.8B params (fp16 ≈ 7.6 GB). The rest
    # of VRAM is consumed by per-image visual tokens (~1000–1600 tokens
    # depending on resolution) times attention memory. The auto-tuner below
    # picks batches by total VRAM; env vars override if set.
    #
    # Guide (roughly measured at default image resolution ~1.6 M pixels):
    #   <24 GB  → text 64,  image 32,  workers 8
    #   24–40 GB → text 128, image 64,  workers 16
    #   40–60 GB → text 128, image 96,  workers 24
    #   60–80 GB → text 256, image 160, workers 32
    #   >=80 GB  → text 512, image 256, workers 48    (e.g. RTX PRO 6000 96GB)
    _VRAM_TIERS: list = [
        # (min_gb, text, image, workers)
        (80, 512, 256, 48),
        (60, 256, 160, 32),
        (40, 128, 96,  24),
        (24, 128, 64,  16),
        (0,  64,  32,  8),
    ]

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._device = None
        self._task = os.environ.get("LLMSQL_JINA_V4_TASK", self.DEFAULT_TASK)
        td = os.environ.get("LLMSQL_JINA_V4_TRUNCATE_DIM", "").strip()
        self._truncate_dim = int(td) if td.isdigit() else None

        def _env_int(key: str) -> Optional[int]:
            raw = os.environ.get(key, "").strip()
            return int(raw) if raw.isdigit() else None

        # Explicit env overrides win; otherwise we pick by VRAM at _load() time.
        self._text_batch_override = _env_int("LLMSQL_JINA_V4_TEXT_BATCH")
        self._image_batch_override = _env_int("LLMSQL_JINA_V4_IMAGE_BATCH")
        self._image_workers_override = _env_int("LLMSQL_JINA_V4_IMAGE_WORKERS")
        self._text_batch = 0
        self._image_batch = 0
        self._image_workers = 0
        # Optional cap on image token count — lower = faster + less VRAM at
        # the cost of fine-grained visual detail. Jina v4 / Qwen2-VL scales
        # quadratically with resolution, so clamping max_pixels is the
        # single biggest lever for image-batch throughput.
        mp = os.environ.get("LLMSQL_JINA_V4_MAX_PIXELS", "").strip()
        self._max_pixels = int(mp) if mp.isdigit() else None

    def _pick_batches_from_vram(self, total_gb: float) -> tuple:
        for min_gb, t, i, w in self._VRAM_TIERS:
            if total_gb >= min_gb:
                return t, i, w
        return 32, 16, 4

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Pick batch sizes based on available VRAM unless env vars override.
        if device == "cuda":
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            auto_text, auto_image, auto_workers = self._pick_batches_from_vram(total_gb)
        else:
            total_gb = 0.0
            auto_text, auto_image, auto_workers = 16, 4, 2

        self._text_batch = self._text_batch_override or auto_text
        self._image_batch = self._image_batch_override or auto_image
        self._image_workers = self._image_workers_override or auto_workers

        import logging
        logging.getLogger(__name__).info(
            f"JinaEmbeddingsV4Backend: device={device}, vram={total_gb:.0f} GB, "
            f"text_batch={self._text_batch}, image_batch={self._image_batch}, "
            f"image_workers={self._image_workers}"
            + (f", max_pixels={self._max_pixels}" if self._max_pixels else "")
        )

        self._model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        self._device = device

    def _extra_kwargs(self) -> dict:
        kw = {}
        if self._truncate_dim:
            kw["truncate_dim"] = self._truncate_dim
        return kw

    def _load_images_parallel(self, paths: List[str]):
        """Decode images concurrently so PIL I/O overlaps GPU forward."""
        from concurrent.futures import ThreadPoolExecutor
        if self._image_workers <= 1 or len(paths) <= 1:
            return [_safe_open_image(p) for p in paths]
        with ThreadPoolExecutor(max_workers=self._image_workers) as pool:
            return list(pool.map(_safe_open_image, paths))

    def _to_numpy(self, out) -> np.ndarray:
        """Convert Jina v4's encode_text / encode_image output to an ndarray.

        The model may return any of:
          * a single torch.Tensor of shape (N, D)  (batched)
          * a list/tuple of torch.Tensor, each of shape (D,)  (per-item)
          * a list/tuple of torch.Tensor of shape (1, D)
          * a numpy array / list of lists (already CPU)
        All CUDA tensors must be moved to CPU before numpy conversion.
        """
        import torch
        if isinstance(out, torch.Tensor):
            return out.detach().float().cpu().numpy().astype(np.float32)
        if isinstance(out, (list, tuple)):
            if len(out) == 0:
                return np.zeros((0, 2048), dtype=np.float32)
            first = out[0]
            if isinstance(first, torch.Tensor):
                cpu = [t.detach().float().cpu() for t in out]
                if cpu[0].dim() == 1:
                    stacked = torch.stack(cpu, dim=0)
                else:
                    stacked = torch.cat(cpu, dim=0)
                return stacked.numpy().astype(np.float32)
        # numpy array or list-of-lists on CPU
        return np.asarray(out, dtype=np.float32)

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None,
                    prompt_name: str = "passage") -> np.ndarray:
        """Default document-side encoding (prompt_name='passage').

        Passes the full list to encode_text and lets its internal DataLoader
        batch on the GPU side at the requested batch_size.
        """
        self._load()
        import torch
        batch_size = batch_size or self._text_batch
        if not texts:
            return np.zeros((0, 2048), dtype=np.float32)
        with torch.inference_mode():
            out = self._model.encode_text(
                texts=texts, task=self._task, prompt_name=prompt_name,
                batch_size=batch_size,
                **self._extra_kwargs(),
            )
        return self._to_numpy(out)

    def embed_images(self, paths: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Pre-decodes PIL in a thread pool, then passes the whole list to
        encode_image with a large internal batch_size so the GPU stays saturated.
        """
        self._load()
        import torch
        batch_size = batch_size or self._image_batch
        if not paths:
            return np.zeros((0, 2048), dtype=np.float32)
        batch_imgs = self._load_images_parallel(paths)
        kwargs = dict(self._extra_kwargs())
        if self._max_pixels is not None:
            kwargs["max_pixels"] = self._max_pixels
        with torch.inference_mode():
            out = self._model.encode_image(
                images=batch_imgs, task=self._task,
                batch_size=batch_size,
                **kwargs,
            )
        return self._to_numpy(out)

    def embed_query(self, query: str, task: Optional[str] = None,
                    batch_size: int = 1) -> np.ndarray:
        # `task` at this method boundary is the Jina task family selector.
        # Default honours the backend-level task; callers rarely override.
        self._load()
        import torch
        task_name = task or self._task
        with torch.inference_mode():
            out = self._model.encode_text(
                texts=[query], task=task_name, prompt_name="query",
                batch_size=1,
                **self._extra_kwargs(),
            )
        return self._to_numpy(out)


# ---------------------------------------------------------------------------
# BLIPBackend — Salesforce/blip-itm-{base,large}-coco (Image-Text Matching)
#
# BLIP is primarily a cross-encoder (ITM head gives pair scores), but
# underneath it has a dual-encoder alignment head (vision_proj + text_proj)
# that projects image and text into a shared 256-dim space. We use that
# alignment head to produce CLIP-style independent embeddings so the
# pipeline's cosine/similarity machinery works unchanged.
# ---------------------------------------------------------------------------

class BLIPBackend:
    shared_space = True  # vision_proj and text_proj share a 256-dim space

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import BlipProcessor, BlipForImageTextRetrieval
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else None
        self._processor = BlipProcessor.from_pretrained(self.model_id, use_fast=True)
        self._model = BlipForImageTextRetrieval.from_pretrained(
            self.model_id, torch_dtype=dtype,
        ).to(device).eval()
        self._device = device

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self._load()
        import torch
        import torch.nn.functional as F
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._processor(
                text=batch, padding=True, truncation=True,
                max_length=35, return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                text_out = self._model.text_encoder(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    return_dict=True,
                )
                cls = text_out.last_hidden_state[:, 0, :]
                feats = self._model.text_proj(cls)
                feats = F.normalize(feats, p=2, dim=-1)
            results.append(feats.float().cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)

    def embed_images(self, paths: List[str], batch_size: int = 16) -> np.ndarray:
        self._load()
        import torch
        import torch.nn.functional as F
        from PIL import Image
        results = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (384, 384)))
            inputs = self._processor(images=imgs, return_tensors="pt").to(self._device)
            with torch.no_grad():
                vision_out = self._model.vision_model(
                    pixel_values=inputs.pixel_values, return_dict=True,
                )
                cls = vision_out.last_hidden_state[:, 0, :]
                feats = self._model.vision_proj(cls)
                feats = F.normalize(feats, p=2, dim=-1)
            results.append(feats.float().cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)

    # Optional: expose BLIP's cross-encoder ITM head for pair-scoring.
    # Not used by the current sampling/C&S pipeline (which needs independent
    # embeddings), but available for direct (text, image) rescoring.
    def score_itm(self, texts: List[str], paths: List[str],
                  batch_size: int = 8) -> np.ndarray:
        """Return P(match=1) from the ITM head for each paired (text, image).

        Runs the cross-encoder forward; slower than embed_* but often more
        accurate than cosine(text_emb, image_emb).
        """
        if len(texts) != len(paths):
            raise ValueError(
                f"score_itm: len(texts)={len(texts)} != len(paths)={len(paths)}"
            )
        self._load()
        import torch
        import torch.nn.functional as F
        from PIL import Image
        scores = []
        for i in range(0, len(texts), batch_size):
            t_batch = texts[i : i + batch_size]
            p_batch = paths[i : i + batch_size]
            imgs = []
            for p in p_batch:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (384, 384)))
            inputs = self._processor(
                images=imgs, text=t_batch, padding=True, truncation=True,
                max_length=35, return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs)[0]      # (B, 2): [no-match, match]
                probs = F.softmax(logits, dim=-1)[:, 1]
            scores.append(probs.float().cpu().numpy())
        return np.concatenate(scores, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# GTEQwenVLBackend — Alibaba-NLP/gte-Qwen2-VL-7B-Instruct (7B, heavy)
# Mean-pools last_hidden_state over the attention mask.
# ---------------------------------------------------------------------------

class GTEQwenVLBackend:
    shared_space = True  # retrieval-tuned; text and image occupy the same space

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device).eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=True
        )
        self._device = device

    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        import torch
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def embed_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        self._load()
        import torch
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
            with torch.no_grad():
                out = self._model(**inputs)
            pooled = self._mean_pool(out.last_hidden_state, inputs["attention_mask"])
            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
            results.append(pooled.float().cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)

    def embed_images(self, paths: List[str], batch_size: int = 4) -> np.ndarray:
        self._load()
        import torch
        results = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            # _safe_open_image handles missing files + upscales tiny images
            # so the Qwen2-VL processor's smart_resize(factor=28) check passes.
            imgs = [_safe_open_image(p) for p in batch_paths]
            messages = [
                [{"role": "user", "content": [{"type": "image", "image": img}]}]
                for img in imgs
            ]
            texts_fmt = [
                self._processor.apply_chat_template(m, tokenize=False)
                for m in messages
            ]
            inputs = self._processor(
                text=texts_fmt, images=imgs, return_tensors="pt", padding=True
            ).to(self._device)
            with torch.no_grad():
                out = self._model(**inputs)
            pooled = self._mean_pool(out.last_hidden_state, inputs["attention_mask"])
            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
            results.append(pooled.float().cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# GMEQwen2VLBackend — Alibaba-NLP/gme-Qwen2-VL-7B-Instruct (General Multimodal
# Embedder). Unlike the separate-encoders pattern used by SigLIP/CLIP/Jina/
# GTE-VL, GME exposes a third mode: get_fused_embeddings(texts, images) that
# encodes a (text, image) pair jointly into a SINGLE vector — different from
# encoding them separately and mean-pooling.
#
# Public methods (follow the HF model card exactly):
#   embed_texts(texts, is_query=False, instruction=None)
#   embed_images(paths, is_query=False)
#   embed_fused(texts, paths, is_query=False)   ← joint text+image
#   embed_query(q, task=None)                   ← query-side text with instruction
#
# Queries use a task-specific `instruction=`; candidates/corpus items omit it.
# ---------------------------------------------------------------------------

class GMEQwen2VLBackend:
    shared_space = True

    DEFAULT_TASK = (
        "Given a semantic predicate, retrieve rows or pairs that satisfy the predicate."
    )

    # Default batch sizes tuned for ~40GB A100 at fp16. Override per-deployment
    # via LLMSQL_GME_{TEXT,IMAGE,FUSED}_BATCH env vars (bigger GPU → push
    # higher; 24GB cards may need smaller).
    DEFAULT_TEXT_BATCH = 8
    DEFAULT_IMAGE_BATCH = 4
    DEFAULT_FUSED_BATCH = 4
    DEFAULT_IMAGE_WORKERS = 8

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._device = None
        self._default_task = os.environ.get(
            "LLMSQL_GME_TASK", self.DEFAULT_TASK
        )

        def _env_int(key: str, default: int) -> int:
            raw = os.environ.get(key, "").strip()
            return int(raw) if raw.isdigit() else default

        self._text_batch = _env_int("LLMSQL_GME_TEXT_BATCH", self.DEFAULT_TEXT_BATCH)
        self._image_batch = _env_int("LLMSQL_GME_IMAGE_BATCH", self.DEFAULT_IMAGE_BATCH)
        self._fused_batch = _env_int("LLMSQL_GME_FUSED_BATCH", self.DEFAULT_FUSED_BATCH)
        self._image_workers = _env_int(
            "LLMSQL_GME_IMAGE_WORKERS", self.DEFAULT_IMAGE_WORKERS
        )

    def _load(self):
        if self._model is not None:
            return
        import torch
        # Remote code shipped with the checkpoint constrains transformers
        # version; surface a helpful error rather than a cryptic one.
        try:
            from transformers.utils.versions import require_version
            require_version(
                "transformers<4.52.0",
                "gme-Qwen2-VL requires transformers<4.52.0 "
                "(pip install 'transformers==4.51.3')",
            )
        except Exception:
            pass
        from transformers import AutoModel, AutoImageProcessor, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Prefer SDPA attention on CUDA when the model+transformers version
        # support it; fall back to the default backend otherwise.
        load_kwargs = dict(
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
        )
        # The GME remote code loads its own image processor without use_fast,
        # producing a "Using a slow image processor" warning. Default
        # use_fast=True on the Auto* processor loaders for the duration of
        # this load so the remote code picks up the fast (torchvision) path.
        with _default_use_fast_true(AutoImageProcessor, AutoProcessor):
            try:
                self._model = AutoModel.from_pretrained(
                    self.model_id, attn_implementation="sdpa", **load_kwargs,
                ).eval()
            except (TypeError, ValueError):
                # Older remote-code classes don't accept attn_implementation.
                self._model = AutoModel.from_pretrained(
                    self.model_id, **load_kwargs,
                ).eval()
        self._device = device

    # ---- helpers ---------------------------------------------------------

    def _load_images_parallel(self, paths: List[str]):
        """Decode images concurrently to hide PIL I/O behind GPU compute.

        Uses _safe_open_image so missing files become blank placeholders and
        tiny images (e.g. 16×16) are upscaled to at least 32×32 before the
        Qwen2-VL image processor's smart_resize(factor=28) check.
        """
        from concurrent.futures import ThreadPoolExecutor

        if self._image_workers <= 1 or len(paths) <= 1:
            return [_safe_open_image(p) for p in paths]
        with ThreadPoolExecutor(max_workers=self._image_workers) as pool:
            return list(pool.map(_safe_open_image, paths))

    # ---- single-modality text --------------------------------------------

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        is_query: bool = False,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        self._load()
        import torch
        batch_size = batch_size or self._text_batch
        kwargs = {"is_query": is_query}
        if instruction is not None:
            kwargs["instruction"] = instruction
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.inference_mode():
                out = self._model.get_text_embeddings(texts=batch, **kwargs)
            results.append(out.float().cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32) if results \
            else np.zeros((0, 3584), dtype=np.float32)

    # ---- single-modality image -------------------------------------------

    def embed_images(
        self,
        paths: List[str],
        batch_size: Optional[int] = None,
        is_query: bool = False,
    ) -> np.ndarray:
        self._load()
        import torch
        batch_size = batch_size or self._image_batch
        results = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            imgs = self._load_images_parallel(batch_paths)
            with torch.inference_mode():
                out = self._model.get_image_embeddings(images=imgs, is_query=is_query)
            results.append(out.float().cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32) if results \
            else np.zeros((0, 3584), dtype=np.float32)

    # ---- joint text+image (the GME-specific mode) ------------------------

    def embed_fused(
        self,
        texts: List[str],
        paths: List[str],
        batch_size: Optional[int] = None,
        is_query: bool = False,
    ) -> np.ndarray:
        """Joint (text, image) → single embedding per item via GME's native fuser.

        texts[i] and paths[i] are paired: item i is the (text, image) tuple.
        Falls back to text-only / image-only when the other side is missing
        (empty string / empty path) to keep batches uniform.
        """
        if len(texts) != len(paths):
            raise ValueError(
                f"embed_fused: texts (n={len(texts)}) and paths (n={len(paths)}) "
                "must have the same length."
            )
        self._load()
        import torch
        batch_size = batch_size or self._fused_batch
        results = []
        for i in range(0, len(texts), batch_size):
            t_batch = texts[i : i + batch_size]
            p_batch = paths[i : i + batch_size]
            imgs = self._load_images_parallel(p_batch)
            with torch.inference_mode():
                out = self._model.get_fused_embeddings(
                    texts=t_batch, images=imgs, is_query=is_query,
                )
            results.append(out.float().cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32) if results \
            else np.zeros((0, 3584), dtype=np.float32)

    # ---- query-side text with instruction prefix -------------------------

    def embed_query(
        self,
        query: str,
        task: Optional[str] = None,
        batch_size: int = 1,
    ) -> np.ndarray:
        effective_task = task or self._default_task
        return self.embed_texts(
            [query], batch_size=batch_size, is_query=True, instruction=effective_task,
        )


# ---------------------------------------------------------------------------
# GTEQwen2Backend — Alibaba-NLP/gte-Qwen2-7B-instruct (text-only, 3584-dim)
#
# The model itself has no vision tower; for multimodal queries each image is
# first converted to a text caption, and the caption is embedded as text.
#
# Captioning mode is controlled by env vars:
#   LLMSQL_CAPTION_URL     OpenAI-compatible /v1 base URL of a vision LLM
#                          (e.g. the vLLM server already running Qwen3-VL).
#   LLMSQL_CAPTION_MODEL   Model name to pass to that endpoint.
#   LLMSQL_CAPTION_PROMPT  Optional caption prompt (default: single-sentence).
#
# If LLMSQL_CAPTION_URL is not set, falls back to embedding the image's
# file-stem (underscore-split) as plain text. This keeps the pipeline running
# but is a degraded baseline — suitable only as an ablation sanity-check.
# Captions are cached in-memory per backend instance to avoid recomputation.
# ---------------------------------------------------------------------------

class GTEQwen2Backend:
    """Alibaba-NLP/gte-Qwen2-7B-instruct — text-only 3584-dim embedder.

    Implementation follows the official HuggingFace model card code example
    (https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct):
      * AutoTokenizer + AutoModel with trust_remote_code=True
      * max_length = 8192
      * last-token (EOS) pooling
      * L2 normalization via F.normalize(p=2, dim=1)
      * Asymmetric retrieval: QUERIES get an "Instruct: {task}\\nQuery: {q}"
        prefix, DOCUMENTS are embedded raw. Callers hit `embed_query(...)`
        for the former and `embed_texts(...)` for the latter.

    model_id can be either the HuggingFace ID or a local checkpoint dir;
    AutoModel.from_pretrained handles both.

    Image inputs are captioned via a vision LLM (see env vars below) and
    the caption is embedded as a document.

    Env vars
    --------
    LLMSQL_GTE_QWEN2_TASK   Task description baked into the query prefix.
                            Default: semantic-predicate retrieval task.
    LLMSQL_CAPTION_URL      Vision-LLM /v1 base URL for image captioning.
    LLMSQL_CAPTION_MODEL    Caption-LLM model name for that endpoint.
    LLMSQL_CAPTION_PROMPT   Caption prompt (default: one-sentence).
    """

    shared_space = True  # text-only; "images" go through captioning → same text space

    DEFAULT_CAPTION_PROMPT = "Describe this image in one concise sentence."
    DEFAULT_TASK = (
        "Given a semantic predicate, retrieve rows or pairs that satisfy the predicate."
    )

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._device = None
        self._default_task = os.environ.get(
            "LLMSQL_GTE_QWEN2_TASK", self.DEFAULT_TASK
        )
        self._caption_url = os.environ.get("LLMSQL_CAPTION_URL", "").strip()
        self._caption_model = os.environ.get("LLMSQL_CAPTION_MODEL", "default")
        self._caption_prompt = os.environ.get(
            "LLMSQL_CAPTION_PROMPT", self.DEFAULT_CAPTION_PROMPT
        )
        self._caption_cache: dict = {}
        self._caption_fallback_warned = False

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # model_id may be a HuggingFace ID or a local directory — both work.
        self._model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._device = device

    @staticmethod
    def _last_token_pool(last_hidden, attention_mask):
        """Last-token (EOS) pooling — copied verbatim from the HF model card."""
        import torch
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return last_hidden[:, -1]
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        return last_hidden[torch.arange(batch_size, device=last_hidden.device), seq_lens]

    @staticmethod
    def _format_query(task: str, query: str) -> str:
        """Matches get_detailed_instruct from the HF model card."""
        return f"Instruct: {task}\nQuery: {query}"

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Shared encode path used by both embed_texts and embed_query."""
        self._load()
        import torch
        import torch.nn.functional as F
        inputs = self._tokenizer(
            texts, max_length=8192, padding=True, truncation=True,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        pooled = self._last_token_pool(out.last_hidden_state, inputs["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.float().cpu().numpy().astype(np.float32)

    def embed_texts(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """Embed as DOCUMENTS — no instruction prefix, per HF model card."""
        results = []
        for i in range(0, len(texts), batch_size):
            results.append(self._encode_batch(texts[i : i + batch_size]))
        return np.concatenate(results, axis=0) if results else np.zeros((0, 3584), dtype=np.float32)

    def embed_query(self, query: str, task: Optional[str] = None, batch_size: int = 4) -> np.ndarray:
        """Embed as a QUERY — wrap with the Instruct/Query prefix first."""
        effective_task = task or self._default_task
        formatted = self._format_query(effective_task, query)
        return self._encode_batch([formatted])

    def embed_images(self, paths: List[str], batch_size: int = 4) -> np.ndarray:
        """Caption each image via a vision LLM and embed the captions as documents."""
        captions = [self._caption(p) for p in paths]
        return self.embed_texts(captions, batch_size=batch_size)

    # ---- Image → caption ------------------------------------------------

    def _caption(self, path: str) -> str:
        if path in self._caption_cache:
            return self._caption_cache[path]

        if not self._caption_url:
            caption = self._filename_caption(path)
        else:
            try:
                caption = self._caption_via_vlm(path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"GTEQwen2Backend: caption call failed for {path!r} ({e}); "
                    "falling back to filename."
                )
                caption = self._filename_caption(path)

        self._caption_cache[path] = caption
        return caption

    def _filename_caption(self, path: str) -> str:
        if not self._caption_fallback_warned and not self._caption_url:
            import logging
            logging.getLogger(__name__).warning(
                "GTEQwen2Backend: LLMSQL_CAPTION_URL is not set; embedding image "
                "file-stems as text. For meaningful multimodal results, point "
                "LLMSQL_CAPTION_URL at a vision-LLM /v1 endpoint and set "
                "LLMSQL_CAPTION_MODEL."
            )
            self._caption_fallback_warned = True
        stem = os.path.basename(path).rsplit(".", 1)[0]
        return stem.replace("_", " ").replace("-", " ")

    def _caption_via_vlm(self, path: str) -> str:
        import base64
        from openai import OpenAI
        client = OpenAI(base_url=self._caption_url, api_key="dummy")
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "jpeg"
        if ext == "jpg":
            ext = "jpeg"
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")
        resp = client.chat.completions.create(
            model=self._caption_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self._caption_prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/{ext};base64,{img_b64}"}},
                ],
            }],
            max_tokens=80,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip() or self._filename_caption(path)


# ---------------------------------------------------------------------------
# SentenceTransformerBackend — text-only (e.g. all-MiniLM-L12-v2)
# ---------------------------------------------------------------------------

class SentenceTransformerBackend:
    shared_space = False  # text-only; embed_images raises NotImplementedError

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._st = None

    def _load(self):
        if self._st is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer(self.model_id)

    def embed_texts(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        self._load()
        embs = self._st.encode(
            texts, batch_size=batch_size,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(embs, dtype=np.float32)

    def embed_images(self, paths: List[str], batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError(
            f"SentenceTransformer backend '{self.model_id}' is text-only "
            "and cannot embed images. Use a multimodal backend for image queries."
        )


# ---------------------------------------------------------------------------
# OpenAIBackend — OpenAI embeddings API or any HTTP-compatible endpoint
# ---------------------------------------------------------------------------

class OpenAIBackend:
    shared_space = False  # text-only

    def __init__(self, model_id: str):
        self.model_id = model_id

    def embed_texts(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        from openai import OpenAI
        base_url = self.model_id if self.model_id.startswith("http") else None
        if base_url:
            parts = base_url.rsplit("/v1", 1)
            api_base = parts[0] + "/v1" if len(parts) > 1 else base_url
            client = OpenAI(base_url=api_base, api_key="dummy")
            model_name = "default"
        else:
            client = OpenAI()
            model_name = self.model_id

        all_embeddings: list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(input=batch, model=model_name)
            for item in resp.data:
                all_embeddings.append(item.embedding)
        return np.array(all_embeddings, dtype=np.float32)

    def embed_images(self, paths: List[str], batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError(
            "OpenAI embedding API does not support image embedding directly."
        )
