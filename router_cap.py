"""Router call cap (patch R1, budget-F1 experiment protocol §4.1).

PROBE's learned router previously escalated every row it flagged, so total LLM
spend was unbounded (reviewer R2/O1). These helpers bound it: the sampler keeps
the full budget ``B`` and the router may spend at most ``floor(cap_frac * B)``
additional calls. An earlier 70/30 split starved the sampler at small budgets
(match202 F1 1.00 -> 0.21) while the router used none of its reserved quota.
"""
from typing import Optional, Tuple

import numpy as np

MIN_TOTAL_BUDGET = 10


def split_total_budget(
    n_items: int, budget_pct: float, cap_frac: float
) -> Tuple[int, int, int]:
    """Return ``(total, sample_budget, router_cap)`` for one run.

    ``total`` keeps run.py's existing floor of ``MIN_TOTAL_BUDGET`` calls.
    Sampling receives all of it, identical to the unpatched run.py sample size.
    The router may add at most ``router_cap = floor(cap_frac * total)`` calls,
    so a run spends at most ``total + router_cap`` expert calls.
    """
    if not 0.0 <= cap_frac <= 1.0:
        raise ValueError(f"cap_frac must be in [0, 1], got {cap_frac}")
    total = max(MIN_TOTAL_BUDGET, int(n_items * budget_pct / 100))
    return total, total, int(cap_frac * total)


def cap_router_candidates(
    cand_idx: np.ndarray,
    defer_proba: np.ndarray,
    router_max_calls: Optional[int],
) -> np.ndarray:
    """Keep at most ``router_max_calls`` escalation candidates.

    Rows most likely to be wrong (highest ``defer_proba``) are kept. The result
    is sorted by row index. ``None`` disables the cap.
    """
    cand_idx = np.asarray(cand_idx, dtype=np.int64)
    if router_max_calls is None or cand_idx.size <= router_max_calls:
        return cand_idx
    if router_max_calls <= 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-np.asarray(defer_proba)[cand_idx], kind="stable")
    return np.sort(cand_idx[order[:router_max_calls]])
