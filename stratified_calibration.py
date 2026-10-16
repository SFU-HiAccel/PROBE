"""Score-stratified calibration sample A for learned calibration (design3 §4).

Rows outside the active sample L are split into equal-width bins of the frozen
proxy score s. Draws are spread evenly across strata (capped by stratum size,
leftovers low score first) and each drawn row carries the inverse inclusion
weight N_j / n_j.

The strata are fixed once, so A can grow in rounds (the insufficient-evidence
fallback): each round draws without replacement from the rows of each stratum
not sampled yet, keeping the cumulative per-stratum counts as equal as
possible. After any number of rounds, A within stratum j is a simple random
sample of size n_j (cumulative) from N_j rows, so weights are N_j / n_j.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

__all__ = [
    "StratifiedSample",
    "StratifiedDesign",
    "build_score_strata",
    "allocate_equal",
    "allocate_cumulative",
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


def allocate_cumulative(sizes: np.ndarray, counts: np.ndarray, m: int) -> np.ndarray:
    """Add ``m`` draws to ``counts``, one at a time, to the open stratum with the fewest draws.

    Ties go to the lower-score stratum. Returns the new cumulative counts.
    """
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    new_counts = np.asarray(counts, dtype=np.int64).reshape(-1).copy()
    if new_counts.shape != sizes.shape or np.any(new_counts > sizes) or np.any(new_counts < 0):
        raise ValueError("counts must be within [0, sizes]")
    if m < 0 or m > int((sizes - new_counts).sum()):
        raise ValueError(f"m={m} exceeds remaining capacity {int((sizes - new_counts).sum())}")
    for _ in range(int(m)):
        open_strata = np.flatnonzero(new_counts < sizes)
        j = open_strata[int(np.argmin(new_counts[open_strata]))]
        new_counts[j] += 1
    return new_counts


def allocate_neyman(sizes: np.ndarray, counts: np.ndarray, m: int, p_hat: np.ndarray,
                    *, floor_frac: float = 0.2) -> np.ndarray:
    """Add ``m`` draws with Neyman allocation on a floor, returning the new cumulative counts.

    Neyman (1934) minimises the variance of the estimated total by giving stratum j a share
    proportional to ``N_j * sigma_j`` with ``sigma_j = sqrt(p_j (1 - p_j))``. Under a rare positive
    rate that tilts the sample towards the strata where positives can plausibly live, which is what an
    equal split fails to do: on mmqa_q7 it spread 1080 draws evenly and caught none of the 5 positives.

    ``floor_frac`` of the draws is handed out by the equal rule first, so every stratum keeps a
    non-zero inclusion probability. Without that floor a stratum could get zero draws, its rows would
    have pi = 0, and the Horvitz-Thompson estimate would no longer cover the pool (the coverage bias
    Owen & Zhou's defensive mixing guards against).
    """
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    p = np.clip(np.asarray(p_hat, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if p.shape != sizes.shape:
        raise ValueError("p_hat must have one entry per stratum")
    m = int(m)
    if m <= 0:
        return counts.copy()

    n_floor = min(m, max(int(np.count_nonzero((counts == 0) & (sizes > 0))),
                         int(np.ceil(floor_frac * m))))
    target = allocate_cumulative(sizes, counts, n_floor)          # spread the floor evenly
    left = m - n_floor
    if left <= 0:
        return target

    room = sizes - target
    sigma = np.sqrt(p * (1.0 - p))
    share = sizes.astype(np.float64) * sigma
    share = np.where(room > 0, share, 0.0)
    if share.sum() <= 0:                                          # no information to tilt on
        return allocate_cumulative(sizes, target, min(left, int(room.sum())))
    want = np.floor(left * share / share.sum()).astype(np.int64)
    want = np.minimum(want, room)
    target = target + want
    spare = left - int(want.sum())
    if spare > 0:                                                 # leftovers by the equal rule
        target = allocate_cumulative(sizes, target, min(spare, int((sizes - target).sum())))
    return target


def allocate_equal(sizes: np.ndarray, m: int) -> np.ndarray:
    """Split ``m`` draws evenly across strata, capped by size, leftovers low-score first."""
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    if m < sizes.size or m > int(sizes.sum()):
        raise ValueError(f"m={m} must be in [{sizes.size}, {int(sizes.sum())}]")
    return allocate_cumulative(sizes, np.zeros_like(sizes), m)


@dataclass(frozen=True)
class StratifiedDesign:
    """Fixed score strata plus the rows sampled from each so far (immutable)."""

    strata: Tuple[np.ndarray, ...]   # row ids per stratum, fixed
    sampled: Tuple[np.ndarray, ...]  # sorted row ids drawn so far per stratum
    p_hat: Optional[np.ndarray] = None   # per-stratum positive rate used for Neyman allocation

    @classmethod
    def build(cls, pool_idx: np.ndarray, scores: np.ndarray, *, n_bins: int, max_strata: int,
              neyman: bool = False) -> "StratifiedDesign":
        strata = tuple(build_score_strata(pool_idx, scores, n_bins, max_strata))
        p_hat = None
        if neyman:
            # The proxy score is the only estimate of the positive rate available before labelling.
            sc = np.asarray(scores, dtype=np.float64)
            p_hat = np.array([float(np.clip(sc[st].mean(), 0.0, 1.0)) for st in strata])
        return cls(strata=strata, sampled=tuple(np.zeros(0, dtype=np.int64) for _ in strata), p_hat=p_hat)

    @property
    def sizes(self) -> np.ndarray:
        return np.array([s.size for s in self.strata], dtype=np.int64)

    @property
    def counts(self) -> np.ndarray:
        return np.array([s.size for s in self.sampled], dtype=np.int64)

    @property
    def remaining(self) -> int:
        return int((self.sizes - self.counts).sum())

    def draw(self, m: int, rng: np.random.Generator) -> Tuple["StratifiedDesign", np.ndarray]:
        """Draw up to ``m`` new rows; returns the updated design and the new row ids."""
        m = max(0, min(int(m), self.remaining))
        if m == 0:
            return self, np.zeros(0, dtype=np.int64)
        if self.p_hat is None:
            target = allocate_cumulative(self.sizes, self.counts, m)
        else:
            target = allocate_neyman(self.sizes, self.counts, m, self.p_hat)
        new_sampled, new_rows = [], []
        for stratum, taken, want in zip(self.strata, self.sampled, target.tolist()):
            extra = want - taken.size
            if extra > 0:
                available = np.setdiff1d(stratum, taken, assume_unique=True) if taken.size else stratum
                chosen = rng.choice(available, size=extra, replace=False).astype(np.int64)
                new_rows.append(np.sort(chosen))
                taken = np.sort(np.concatenate([taken, chosen]))
            new_sampled.append(taken)
        rows = np.concatenate(new_rows) if new_rows else np.zeros(0, dtype=np.int64)
        return StratifiedDesign(strata=self.strata, sampled=tuple(new_sampled), p_hat=self.p_hat), rows

    def sample(self) -> StratifiedSample:
        """All rows drawn so far with weights N_j / n_j (strata with n_j = 0 contribute nothing)."""
        if not self.strata:
            return _empty_sample()
        weights = [np.full(taken.size, stratum.size / taken.size, dtype=np.float64)
                   for stratum, taken in zip(self.strata, self.sampled) if taken.size]
        return StratifiedSample(
            indices=np.concatenate(self.sampled).astype(np.int64),
            weights=np.concatenate(weights) if weights else np.zeros(0, dtype=np.float64),
            stratum_sizes=self.sizes,
            sample_counts=self.counts,
        )


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
    design = StratifiedDesign.build(pool_idx, scores, n_bins=n_bins, max_strata=m)
    design, _ = design.draw(m, rng)
    return design.sample()
