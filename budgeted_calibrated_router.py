"""Budgeted dual-threshold routing for learned calibration (design3 §6-§7).

With frozen calibrated probabilities p on the unverified rows, the expected
counts are
    T  = C + sum_{h=1} p,   FP = sum_{h=1} (1 - p),   FN = sum_{h=0} p
where C is the number of verified positives. Verifying a predicted positive
removes its (1 - p) from FP; verifying a predicted negative moves its p from FN
into T. Within each group the rows with the largest expected error come first
(predicted positives by ascending s, predicted negatives by descending s), so
enumerating the split k_plus + k_minus = r over prefix sums finds the exact
maximiser of F_tilde = 2T / (2T + FP + FN) for this single allocation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ExpectedCounts", "RoutingPlan", "expected_counts", "plan_dual_threshold", "apply_verified_labels"]


@dataclass(frozen=True)
class ExpectedCounts:
    tp: float
    fp: float
    fn: float

    def f1(self) -> float:
        den = 2.0 * self.tp + self.fp + self.fn
        return float("nan") if den <= 0.0 else 2.0 * self.tp / den


@dataclass(frozen=True)
class RoutingPlan:
    verify_idx: np.ndarray  # int64 row ids to send to the LLM, sorted
    k_plus: int             # predicted positives verified (lowest s first)
    k_minus: int            # predicted negatives verified (highest s first)
    tau_plus: float         # s of the last verified predicted positive (nan if none)
    tau_minus: float        # s of the last verified predicted negative (nan if none)
    f_tilde: float          # objective value of the chosen split


def expected_counts(p: np.ndarray, h: np.ndarray, n_verified_pos: int) -> ExpectedCounts:
    p = np.asarray(p, dtype=np.float64)
    pos = np.asarray(h) > 0.5
    return ExpectedCounts(
        tp=float(n_verified_pos) + float(p[pos].sum()),
        fp=float((1.0 - p[pos]).sum()),
        fn=float(p[~pos].sum()),
    )


def plan_dual_threshold(
    row_idx: np.ndarray,
    p: np.ndarray,
    s: np.ndarray,
    h: np.ndarray,
    counts: ExpectedCounts,
    budget: int,
) -> RoutingPlan:
    """Choose which unverified rows to verify with at most ``budget`` LLM calls.

    ``p``, ``s`` and ``h`` are aligned with ``row_idx``. ``s`` is any ordering key
    monotone in ``p`` within each ``h`` group (e.g. the raw proxy logit); ``tau_plus``
    and ``tau_minus`` are reported on the same scale. Ties in ``s`` are broken by row
    id, so exactly ``min(budget, len(row_idx))`` rows are returned.
    """
    row_idx = np.asarray(row_idx, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    pos = np.asarray(h) > 0.5

    pos_order = np.lexsort((row_idx[pos], s[pos]))
    neg_order = np.lexsort((row_idx[~pos], -s[~pos]))
    pos_rows, pos_s = row_idx[pos][pos_order], s[pos][pos_order]
    neg_rows, neg_s = row_idx[~pos][neg_order], s[~pos][neg_order]
    s_plus = np.concatenate([[0.0], np.cumsum(1.0 - p[pos][pos_order])])
    s_minus = np.concatenate([[0.0], np.cumsum(p[~pos][neg_order])])

    m_plus, m_minus = pos_rows.size, neg_rows.size
    r = max(0, min(int(budget), m_plus + m_minus))
    k = np.arange(max(0, r - m_minus), min(r, m_plus) + 1)
    k_neg = r - k
    tp = counts.tp + s_minus[k_neg]
    fp = np.maximum(counts.fp - s_plus[k], 0.0)
    fn = np.maximum(counts.fn - s_minus[k_neg], 0.0)
    den = 2.0 * tp + fp + fn
    f = np.full(k.size, np.nan)
    np.divide(2.0 * tp, den, out=f, where=den > 0.0)
    best = int(np.argmax(np.where(np.isnan(f), -np.inf, f)))
    k_plus = int(k[best])
    k_minus = r - k_plus
    return RoutingPlan(
        verify_idx=np.sort(np.concatenate([pos_rows[:k_plus], neg_rows[:k_minus]])),
        k_plus=k_plus,
        k_minus=k_minus,
        tau_plus=float(pos_s[k_plus - 1]) if k_plus > 0 else float("nan"),
        tau_minus=float(neg_s[k_minus - 1]) if k_minus > 0 else float("nan"),
        f_tilde=float(f[best]),
    )


def apply_verified_labels(counts: ExpectedCounts, p: np.ndarray, h: np.ndarray, y: np.ndarray) -> ExpectedCounts:
    """Replace the expected contribution of newly verified rows with their LLM labels."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pos = np.asarray(h) > 0.5
    return ExpectedCounts(
        tp=counts.tp + float((y[pos] - p[pos]).sum()) + float(y[~pos].sum()),
        fp=counts.fp - float((1.0 - p[pos]).sum()),
        fn=counts.fn - float(p[~pos].sum()),
    )
