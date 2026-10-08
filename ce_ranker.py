"""Predicate-aware fixed CE scores for filters and joins, with content caching.

No feedback/GT parameter exists. Only explicitly selected row columns are read.
Disk cache identity covers the actual backend source, model snapshot, processor
files, predicate, original IDs, candidate text, image bytes and preprocessing.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

MODEL = 'Qwen/Qwen3-VL-Reranker-2B'
# Predicate-aware multimodal cross-encoders with a backend in ce_ranker_backend.BACKENDS.
SUPPORTED_MODELS = (MODEL, 'jinaai/jina-reranker-m0')
_MODELS = {}


@dataclass(frozen=True)
class CEConfig:
    model: str = MODEL
    revision: str | None = None
    device: str = 'cuda'
    batch_size: int = 16
    max_pixels: int = 262144
    max_text_chars: int = 2000

    def __post_init__(self):
        if self.revision is None:
            object.__setattr__(self, 'revision',
                '4bd860ac4f15ad1897a214615cccc700f8f71818' if self.model == MODEL else 'main')
        if self.batch_size < 1 or self.max_pixels < 4096 or self.max_text_chars < 1:
            raise ValueError('invalid CE preprocessing/batch configuration')
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f'fixed CE supports {SUPPORTED_MODELS}, got {self.model!r}')


def row_contents(records, columns, image_columns=()):
    """Preserve selected column names. Reject multiple images rather than dropping one."""
    images = set(image_columns)
    out = []
    for record in records:
        parts, paths = [], []
        for column in columns:
            value = record.get(column)
            if value is None:
                continue
            if column in images:
                if str(value).strip() and str(value) != 'NULL':
                    paths.append(str(value).removeprefix('file://').strip())
            else:
                parts.append(f'{column}: {value}')
        if len(paths) > 1:
            raise ValueError('CE supports one image per side; refusing to silently discard images')
        out.append({'text': '\n'.join(parts) or None, 'image': paths[0] if paths else None})
    return out


def _digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _snapshot(config):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(config.model, revision=config.revision))


def _model_identity(path):
    # HF snapshots have immutable commit names. Fingerprint tokenizer/processor
    # and config content as well, so local changes cannot reuse a stale score.
    path = Path(path).resolve()
    assets = {p.name: _digest_file(p) for p in sorted(path.iterdir())
              if p.is_file() and p.suffix in ('.json', '.txt', '.jinja', '.model')}
    if not re.fullmatch('[0-9a-f]{40}', path.name):
        raise ValueError('CE model must resolve to an immutable HF snapshot commit')
    return {'revision': path.name, 'assets': assets}


def score_candidates(candidate_ids, left, right, predicate, *, kind, config=None,
                     cache_dir=None, model_path=None, encoder=None, return_hidden=False):
    """Aligned content lists -> (scores, provenance); optional encoder enables tests.

    With `return_hidden`, returns (scores, provenance, hidden): the last-token hidden state of the
    same forward pass, one row per candidate. It is cached (float16) next to the scores; a score-only
    cache entry is re-encoded once to add it.

    `model_path` must be an immutable local HF snapshot. For filters, left is
    the predicate query and right is each row. Joins supply left/right rows.
    Truncated images raise an error; no candidate is silently removed.
    """
    cfg = config or CEConfig()
    ids = list(candidate_ids)
    if kind not in ('filter', 'join') or not predicate.strip():
        raise ValueError('kind and nonempty predicate are required')
    if len(ids) != len(left) or len(ids) != len(right):
        raise ValueError('unaligned CE candidates')
    canonical_ids = [_json(i) for i in ids]
    if len(set(canonical_ids)) != len(ids):
        raise ValueError('duplicate CE candidate IDs')
    if any(row.get('image') for rows in (left, right) for row in rows):
        from PIL import ImageFile
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            raise ValueError('CE requires strict image decoding; run with '
                             'PIL.ImageFile.LOAD_TRUNCATED_IMAGES=False in an isolated process')
    path = Path(model_path) if model_path is not None else _snapshot(cfg)
    model_identity = _model_identity(path)
    file_hashes = {}
    def content_identity(rows):
        result = []
        for row in rows:
            image = row.get('image')
            if image and image not in file_hashes:
                file_hashes[image] = _digest_file(image)
            result.append({'text': row.get('text'), 'image': file_hashes.get(image)})
        return result
    manifest = {
        'schema': 1, 'kind': kind, 'predicate': predicate,
        'ids': ids, 'left': content_identity(left), 'right': content_identity(right),
        'model': model_identity, 'config': asdict(cfg),
        'score': 'yes_minus_no_logit' if cfg.model == MODEL else 'score_head_logit',
        'truncated_image_policy': 'error',
        'backend': _digest_file(Path(__file__).with_name('ce_ranker_backend.py')),
        'provider': _digest_file(__file__),
    }
    # Also distinguish dependency implementations that build model inputs.
    from importlib.metadata import version
    manifest['libraries'] = {name: version(name) for name in ('transformers', 'torch', 'pillow')}
    fingerprint = hashlib.sha256(_json(manifest).encode()).hexdigest()
    target = Path(cache_dir) / (fingerprint + '.npz') if cache_dir else None
    if target is not None and target.exists():
        with np.load(target, allow_pickle=False) as data:
            scores = data['scores']
            hidden = data['hidden'] if 'hidden' in data.files else None
            valid = str(data['fingerprint']) == fingerprint and scores.shape == (len(ids),) and np.isfinite(scores).all()
        if not valid:
            raise ValueError(f'invalid CE cache: {target}')
        if not return_hidden:
            return scores, {'cache_hit': True, 'fingerprint': fingerprint, 'manifest': manifest}
        if hidden is not None and hidden.shape[0] == len(ids):
            return scores, {'cache_hit': True, 'fingerprint': fingerprint, 'manifest': manifest}, hidden
    if not ids:
        scores = np.empty(0, dtype=np.float32)
        hidden = np.empty((0, 0), dtype=np.float16)
    else:
        if encoder is None:
            from ce_ranker_backend import BACKENDS
            key = (str(path), _json(model_identity), cfg.device, cfg.max_pixels, cfg.max_text_chars)
            if key not in _MODELS:
                _MODELS[key] = BACKENDS[cfg.model](cfg.device, str(path), cfg.max_pixels, cfg.max_text_chars)
            encoder = _MODELS[key]
            encoder._images.clear()
        scores, hidden = encoder.encode(left, right, predicate, cfg.batch_size)
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (len(ids),) or not np.isfinite(scores).all():
            raise ValueError('invalid CE scores')
        if hidden is not None:
            hidden = np.asarray(hidden, dtype=np.float16)
            if hidden.ndim != 2 or hidden.shape[0] != len(ids) or not np.isfinite(hidden).all():
                raise ValueError('invalid CE hidden states')
        elif return_hidden:
            raise ValueError('the CE encoder returned no hidden states')
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=target.parent, suffix='.npz')
        try:
            with os.fdopen(fd, 'wb') as handle:
                extra = {} if hidden is None else {'hidden': hidden}
                np.savez_compressed(handle, scores=scores, fingerprint=fingerprint,
                                    manifest=_json(manifest), **extra)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    info = {'cache_hit': False, 'fingerprint': fingerprint, 'manifest': manifest}
    return (scores, info, hidden) if return_hidden else (scores, info)


def score_filter(records, columns, image_columns, predicate, candidate_ids, **kwargs):
    right = row_contents(records, columns, image_columns)
    left = [{'text': predicate, 'image': None} for _ in right]
    return score_candidates(candidate_ids, left, right, predicate, kind='filter', **kwargs)


def score_join(left_records, right_records, pairs, left_columns, right_columns,
               image_columns, predicate, candidate_ids, **kwargs):
    left_rows = row_contents(left_records, left_columns, image_columns)
    right_rows = row_contents(right_records, right_columns, image_columns)
    pairs = np.asarray(pairs)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError('pairs must be an integer Nx2 array')
    if pairs.size and ((pairs < 0).any() or (pairs[:, 0] >= len(left_rows)).any() or (pairs[:, 1] >= len(right_rows)).any()):
        raise ValueError('pair index out of range')
    left = [left_rows[i] for i, _ in pairs]
    right = [right_rows[j] for _, j in pairs]
    return score_candidates(candidate_ids, left, right, predicate, kind='join', **kwargs)
