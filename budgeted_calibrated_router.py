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

__all__ = ["ExpectedCounts", "RoutingPlan", "expected_counts", "best_splits", "plan_dual_threshold",
           "apply_verified_labels"]


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
    f_tilde: float          # expected F1 of the chosen plan
    n_calls: int = 0        # calls the plan spends (== k_plus + k_minus)
    alpha: float = 1.0      # cost-quality dial the plan was chosen under
    objective: str = "pool"           # which scalarisation chose it
    f_curve: tuple = ()               # expected F1 at r = 0%, 10%, ..., 100% of r_max
    f_gain: float = float("nan")      # f(r_max) - f(0): how much routing can buy at all


def expected_counts(p: np.ndarray, h: np.ndarray, n_verified_pos: int) -> ExpectedCounts:
    p = np.asarray(p, dtype=np.float64)
    pos = np.asarray(h) > 0.5
    return ExpectedCounts(
        tp=float(n_verified_pos) + float(p[pos].sum()),
        fp=float((1.0 - p[pos]).sum()),
        fn=float(p[~pos].sum()),
    )


def best_splits(a: np.ndarray, b: np.ndarray, counts: ExpectedCounts, r_max: int,
                max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """For every r = 0..r_max, the split k_plus + k_minus = r maximising the plug-in F1.

    ``a`` holds (1 - p) of the predicted positives and ``b`` holds p of the predicted negatives, each
    sorted in DESCENDING order (the routing order). Returns ``(k_plus[r], f[r])``; f is nan where the
    F1 denominator is 0. Among maximisers the smallest k_plus is preferred; when several splits tie
    exactly (e.g. identical p), which one wins can differ from the former O(R^2) scan by float rounding,
    but the F1 value is the same.

    Why this is exact in O(R log R) instead of scanning every split (O(R^2)): with prefix sums S+ and
    S-, F1(k) = N(k) / D(k) for k = k_plus at fixed r, and F1(k) >= lam iff
        phi_lam(k) = N(k) - lam * D(k) = const + lam * S+(k) + (2 - lam) * S-(r - k) >= 0.
    S+ and S- are sums of the k largest terms, hence concave, and lam in [0, 1] keeps both coefficients
    non-negative, so phi_lam is concave in k and its maximiser is the first k whose increment
        phi_lam(k + 1) - phi_lam(k) = lam * a[k] - (2 - lam) * b[r - k - 1]
    is <= 0 (a binary search). Dinkelbach's iteration lam <- F1(argmax phi_lam) then increases lam
    strictly until max phi_lam = 0, i.e. lam = max F1, and terminates on a finite set; it is run for
    all r at once, vectorised.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sp = np.concatenate([[0.0], np.cumsum(a)])
    sm = np.concatenate([[0.0], np.cumsum(b)])
    r = np.arange(int(r_max) + 1)
    lo = np.maximum(0, r - b.size)
    hi = np.minimum(r, a.size)

    def f1(k: np.ndarray) -> np.ndarray:
        tp = counts.tp + sm[r - k]
        den = 2.0 * tp + np.maximum(counts.fp - sp[k], 0.0) + np.maximum(counts.fn - sm[r - k], 0.0)
        out = np.full(k.shape, np.nan)
        np.divide(2.0 * tp, den, out=out, where=den > 0.0)
        return out

    def argmax_phi(lam: np.ndarray) -> np.ndarray:
        left, right = lo.copy(), hi.copy()
        while True:
            open_ = left < right
            if not open_.any():
                return left
            mid = (left + right) // 2
            # guard the indices of closed intervals; their result is discarded
            m = np.where(open_, mid, lo)
            nb = np.clip(r - m - 1, 0, max(b.size - 1, 0))
            ga = a[np.clip(m, 0, max(a.size - 1, 0))] if a.size else np.zeros_like(lam)
            gb = b[nb] if b.size else np.zeros_like(lam)
            stop = lam * ga - (2.0 - lam) * gb <= 0.0
            right = np.where(open_ & stop, mid, right)
            left = np.where(open_ & ~stop, mid + 1, left)

    k = lo.copy()
    f = f1(k)
    lam = np.where(np.isfinite(f), f, 0.0)
    for _ in range(max_iter):
        k_new = argmax_phi(lam)
        f_new = f1(k_new)
        better = np.isfinite(f_new) & (f_new > lam + 1e-15)
        if not better.any():
            break
        k = np.where(better, k_new, k)
        lam = np.where(better, f_new, lam)
        f = np.where(better, f_new, f)
    # lam is now max F1 for each r; the smallest maximiser of phi_lam is the smallest F1 maximiser.
    k_min = argmax_phi(lam)
    f_min = f1(k_min)
    use = np.isfinite(f_min) & (f_min >= f - 1e-15) & (k_min <= k)
    k = np.where(use, k_min, k)
    f = np.where(use, f_min, f)
    return k.astype(np.int64), f


def plan_dual_threshold(
    row_idx: np.ndarray,
    p: np.ndarray,
    s: np.ndarray,
    h: np.ndarray,
    counts: ExpectedCounts,
    budget: int,
    alpha: float = 1.0,
    n_pool: Optional[int] = None,
    objective: str = "pool",
    cost_exp: float = 1.0,
) -> RoutingPlan:
    """Choose which unverified rows to verify with at most ``budget`` LLM calls.

    ``alpha`` trades expected quality against the calls spent, in the spirit of GAMCAL's cost-quality
    dial (arXiv:2604.00660) but under a hard budget rather than as an unconstrained scalarisation.
    ``alpha = 1`` always spends the whole budget, reproducing the pre-alpha plan.

    ``objective`` picks how the two terms are made commensurate, which is what decides whether alpha
    behaves like a dial or like a switch:

    ``"pool"``       ``alpha * f(r) - (1 - alpha) * r / n_pool`` -- the original. The cost term is the
                     fraction of the WHOLE POOL routed, so with a budget of 10% of the pool it can
                     never exceed 0.1 while ``f`` spans [0, 1]. Routing is then cut only when
                     ``(1-alpha)/alpha * r/n_pool > f(r) - f(0)``, i.e. when the F1 a full budget buys
                     falls under 0.025 at alpha=0.8, 0.10 at alpha=0.5 and 0.40 at alpha=0.2. Most
                     queries sit far from those thresholds, so alpha reads as a step, not a dial.
    ``"budget"``     ``alpha * f(r) - (1 - alpha) * r / r_max`` -- cost is the fraction of the ROUTING
                     BUDGET spent, so both terms span [0, 1] and alpha trades marginal F1 per unit of
                     budget at a rate of (1-alpha)/alpha.
    ``"normalized"`` ``alpha * (f(r) - f(0)) / (f(r_max) - f(0)) - (1 - alpha) * r / r_max`` -- both
                     terms are normalised to [0, 1] per query, so alpha means the same thing whether a
                     query's routing buys 0.5 F1 or 0.02.
    ``"gamcal"``     GAMCAL Eq. 31 (arXiv:2604.00660 §5.4), minimising
                     ``alpha * (1 - f(r)) / (1 - f(0)) + (1 - alpha) * r / r_max``. The error is divided
                     by the error of the no-delegation proxy-only policy, so BOTH terms equal exactly
                     1.0 at r = 0 and alpha is a true convex interpolation between them. Note GAMCAL
                     runs uncapped (their rho = 1, so their cost really is the delegation rate over the
                     whole pool and spans [0, 1]); under a budget cap the same role is played by
                     ``r / r_max``, which is why this uses the budget rather than ``n_pool``.

    ``n_pool`` is only used by ``"pool"``.

    ``cost_exp`` (gamma) raises the cost term to a power: ``((1 - alpha) * (r / r_max) ** gamma``.
    It is the knob that decides whether alpha resolves at all, and it matters more than the
    normalisation. The objective is linear in ``r``, and measured ``f(r)`` is very nearly linear too --
    the routed rows are the ones closest to the threshold, so at a budget of 10% of the pool they all
    carry about the same information and the marginal gain barely decays (measured
    ``f'(0)/f'(r_max)`` is only 1.1--1.6, and a *softer* calibrator makes it flatter, not steeper).
    Maximising (near-linear minus linear) always lands on an endpoint, which is why alpha reads as a
    switch. With ``gamma > 1`` the marginal cost rises with ``r`` and meets the near-constant marginal
    gain at an interior point, so the routed fraction moves smoothly with alpha. Measured on the
    curve from a 2000-row pool at a 10% budget:

        alpha       0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8
        gamma = 1     0%    0%    0%    0%   40%  100%  100%  100%
        gamma = 2     7%   15%   24%   35%   50%   70%   94%  100%

    ``gamma = 1`` (the default) keeps the original behaviour.

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
    r_max = max(0, min(int(budget), m_plus + m_minus))
    alpha = float(min(max(alpha, 0.0), 1.0))
    scale = float(n_pool) if n_pool else float(max(1, m_plus + m_minus))

    # f(r) over the whole range: needed by "normalized", and worth reporting either way -- a query
    # whose f_gain is ~0 cannot be steered by alpha under any scalarisation, and that is invisible
    # from the chosen plan alone. best_splits finds every r's best split in O(R log R).
    k_best, f_best_r = best_splits(1.0 - p[pos][pos_order], p[~pos][neg_order], counts, r_max)
    splits = list(zip(k_best.tolist(), f_best_r.tolist()))
    f_all = np.array([f for _, f in splits], dtype=np.float64)
    finite = f_all[np.isfinite(f_all)]
    f0 = float(f_all[0]) if np.isfinite(f_all[0]) else (float(finite[0]) if finite.size else float("nan"))
    f_hi = float(np.nanmax(f_all)) if finite.size else float("nan")
    f_gain = f_hi - f0 if np.isfinite(f_hi) and np.isfinite(f0) else float("nan")
    curve = tuple(round(float(f_all[min(r_max, int(round(q * r_max)))]), 6) for q in np.linspace(0, 1, 11)) \
        if r_max >= 0 else ()

    cost_scale = scale if objective == "pool" else float(max(1, r_max))

    if alpha >= 1.0:
        r_best, (k_plus, f_best) = r_max, splits[r_max]
    else:
        r_best, k_plus, f_best, obj_best = 0, 0, float("nan"), -np.inf
        # "normalized" divides the quality gain by what routing can buy on THIS query, so alpha means
        # the same thing everywhere. A query with no gain to buy falls back to the raw F1 term.
        q_den = f_gain if (objective == "normalized" and np.isfinite(f_gain) and f_gain > 1e-9) else None
        # GAMCAL divides the error by the no-delegation policy's error, so both terms are 1.0 at r=0.
        # Guard the near-perfect-proxy case the paper does not mention: f(0) -> 1 makes it explode.
        err0 = max(1.0 - f0, 1e-6) if np.isfinite(f0) else 1.0
        gamma = float(max(cost_exp, 1e-6))
        for r in range(r_max + 1):
            kp, f_r = splits[r]
            if np.isnan(f_r):
                continue
            cost = (r / cost_scale) ** gamma if gamma != 1.0 else (r / cost_scale)
            if objective == "gamcal":
                obj = -(alpha * ((1.0 - f_r) / err0) + (1.0 - alpha) * cost)   # Eq. 31, negated to maximise
            else:
                q_term = (f_r - f0) / q_den if q_den is not None else f_r
                obj = alpha * q_term - (1.0 - alpha) * cost
            if obj > obj_best:
                r_best, k_plus, f_best, obj_best = r, kp, f_r, obj
    k_minus = r_best - k_plus
    return RoutingPlan(
        verify_idx=np.sort(np.concatenate([pos_rows[:k_plus], neg_rows[:k_minus]])),
        k_plus=k_plus,
        k_minus=k_minus,
        tau_plus=float(pos_s[k_plus - 1]) if k_plus > 0 else float("nan"),
        tau_minus=float(neg_s[k_minus - 1]) if k_minus > 0 else float("nan"),
        f_tilde=float(f_best),
        n_calls=int(r_best),
        alpha=alpha,
        objective=str(objective),
        f_curve=curve,
        f_gain=float(f_gain),
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
