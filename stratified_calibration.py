"""Score-stratified calibration sample A for learned calibration (design3 §4).

Rows outside the active sample L are split into equal-width bins of the frozen
proxy score s. Every non-empty stratum receives at least one draw, the rest of
the quota is spread evenly (capped by stratum size), and each drawn row carries
the inverse inclusion weight N_j / n_j.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

__all__ = [
    "StratifiedSample",
    "build_score_strata",
    "allocate_equal",
    "draw_stratified_sample",
]


@dataclass(frozen=True)
class StratifiedSample:
    """Calibration sample A with inverse-inclusion weights."""

    indices: np.ndarray        # int64 row ids, grouped by stratum (low score first)
    weights: np.ndarray        # float64 omega_i = N_j / n_j, aligned with indices
    stratum_sizes: np.ndarray  # int64 N_j per stratum
    sample_counts: np.ndarray  # int64 n_j per stratum

    @property
    def size(self) -> int:
        return int(self.indices.size)


def _empty_sample() -> StratifiedSample:
    return StratifiedSample(
        indices=np.zeros(0, dtype=np.int64),
        weights=np.zeros(0, dtype=np.float64),
        stratum_sizes=np.zeros(0, dtype=np.int64),
        sample_counts=np.zeros(0, dtype=np.int64),
    )


def build_score_strata(
    pool_idx: np.ndarray, scores: np.ndarray, n_bins: int, max_strata: int
) -> List[np.ndarray]:
    """Equal-width score bins over ``pool_idx``, merged down to ``max_strata``.

    Empty bins are dropped. While there are too many strata, the adjacent pair
    with the fewest rows is merged (ties: the lower-score pair).
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if max_strata < 1:
        raise ValueError(f"max_strata must be >= 1, got {max_strata}")
    pool_idx = np.asarray(pool_idx, dtype=np.int64).reshape(-1)
    pool_scores = np.asarray(scores, dtype=np.float64)[pool_idx]
    if not np.all(np.isfinite(pool_scores)):
        raise ValueError("scores must be finite")
    clipped = np.clip(pool_scores, 0.0, 1.0)
    bin_id = np.minimum((clipped * n_bins).astype(np.int64), n_bins - 1)
    strata = [pool_idx[bin_id == b] for b in range(n_bins) if np.any(bin_id == b)]
    while len(strata) > max_strata:
        pair_sizes = [strata[j].size + strata[j + 1].size for j in range(len(strata) - 1)]
        j = int(np.argmin(pair_sizes))
        strata = strata[:j] + [np.concatenate([strata[j], strata[j + 1]])] + strata[j + 2:]
    return strata


def allocate_equal(sizes: np.ndarray, m: int) -> np.ndarray:
    """Split ``m`` draws evenly across strata, capped by size, leftovers low-score first."""
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    if m < sizes.size or m > int(sizes.sum()):
        raise ValueError(f"m={m} must be in [{sizes.size}, {int(sizes.sum())}]")
    counts = np.zeros_like(sizes)
    remaining = int(m)
    while remaining > 0:
        open_strata = np.flatnonzero(counts < sizes)
        share = remaining // open_strata.size
        if share == 0:
            counts[open_strata[:remaining]] += 1
            break
        add = np.minimum(sizes[open_strata] - counts[open_strata], share)
        counts[open_strata] += add
        remaining -= int(add.sum())
    return counts


def draw_stratified_sample(
    pool_idx: np.ndarray,
    scores: np.ndarray,
    m: int,
    *,
    n_bins: int,
    rng: np.random.Generator,
) -> StratifiedSample:
    """Draw exactly ``min(m, |pool|)`` rows without replacement, stratified on ``scores``."""
    pool_idx = np.asarray(pool_idx, dtype=np.int64).reshape(-1)
    m = min(int(m), int(pool_idx.size))
    if m <= 0:
        return _empty_sample()
    strata = build_score_strata(pool_idx, scores, n_bins, max_strata=m)
    sizes = np.array([st.size for st in strata], dtype=np.int64)
    counts = allocate_equal(sizes, m)
    picked, weights = [], []
    for stratum, n_j in zip(strata, counts.tolist()):
        chosen = rng.choice(stratum, size=n_j, replace=False)
        picked.append(np.sort(chosen).astype(np.int64))
        weights.append(np.full(n_j, stratum.size / n_j, dtype=np.float64))
    return StratifiedSample(
        indices=np.concatenate(picked),
        weights=np.concatenate(weights),
        stratum_sizes=sizes,
        sample_counts=counts,
    )
