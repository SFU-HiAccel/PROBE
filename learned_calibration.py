"""Learned calibration: stratified sample A + active sample L -> Platt -> dual thresholds.

Entry points used by ``cluster_proxy.cluster_proxy_pipeline`` when
``decision_mode == "learned_calibration"`` (design3.md, design3_algorithms.tex):

* ``compute_oof_features``  out-of-fold (logit, refined label) for L
* ``run_learned_calibration``  stages 2-5 of the algorithm

Both take their proxy/propagation and LLM access as callables, so this module
has no dependency on cluster_proxy / proxy_model and is testable offline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from budgeted_calibrated_router import apply_verified_labels, expected_counts, plan_dual_threshold
from router_cap import MIN_TOTAL_BUDGET
from stratified_calibration import draw_stratified_sample
from weighted_calibrator import calibrate, predict_proba

__all__ = [
    "LEARNED_CALIBRATION_MODE",
    "LEARNED_CALIBRATION_RESULT_KEYS",
    "LearnedCalibrationConfig",
    "OofFeatures",
    "LearnedCalibrationResult",
    "learned_calibration_total_budget",
    "compute_oof_features",
    "run_learned_calibration",
]

LEARNED_CALIBRATION_MODE = "learned_calibration"

# profiling keys copied into run.py result rows (only for this mode)
LEARNED_CALIBRATION_RESULT_KEYS = (
    "lc_status", "lc_reason", "lc_total_budget", "lc_sample_budget", "lc_calib_budget",
    "lc_calls_L", "lc_calls_A", "lc_calls_D", "lc_calls_total", "lc_budget_overrun",
    "lc_n_strata", "lc_A_pos", "lc_L_pos", "lc_ess_pos", "lc_max_pos_weight_share",
    "lc_lambda_used", "lc_oof_reason", "lc_a", "lc_b", "lc_c", "lc_delta", "lc_t", "lc_b_final",
    "lc_intercept_residual", "lc_k_plus", "lc_k_minus", "lc_tau_plus", "lc_tau_minus",
    "lc_planned_f1", "lc_estimated_f1", "lc_D_pos", "lc_time_s",
)

# refine_fn(train_idx, train_y) -> (refined_label_mask[N], lr_logit[N] or None)
RefineFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]]
# label_rows(row_ids) -> 0/1 LLM labels aligned with row_ids
LabelRowsFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LearnedCalibrationConfig:
    total_budget: int
    sample_frac: float = 0.6
    calib_frac: float = 0.1
    n_bins: int = 10
    n_folds: int = 5
    lam: float = 0.25
    rho_a: float = 0.25
    rho_b: float = 0.25
    rho_c: float = 0.25
    rho_delta: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.total_budget < 1:
            raise ValueError(f"total_budget must be >= 1, got {self.total_budget}")
        if not 0.0 < self.sample_frac <= 1.0:
            raise ValueError(f"sample_frac must be in (0, 1], got {self.sample_frac}")
        if not 0.0 <= self.calib_frac <= 1.0 or self.sample_frac + self.calib_frac > 1.0:
            raise ValueError("calib_frac must be in [0, 1] and sample_frac + calib_frac <= 1")
        if self.n_bins < 1 or self.n_folds < 2:
            raise ValueError("n_bins must be >= 1 and n_folds >= 2")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {self.lam}")
        if min(self.rho_a, self.rho_b, self.rho_c, self.rho_delta) < 0.0:
            raise ValueError("ridge weights must be >= 0")

    @property
    def sample_budget(self) -> int:
        return max(1, int(self.sample_frac * self.total_budget))

    @property
    def calib_budget(self) -> int:
        return int(self.calib_frac * self.total_budget)

    @property
    def rho(self) -> Tuple[float, float, float, float]:
        return (self.rho_a, self.rho_b, self.rho_c, self.rho_delta)


@dataclass(frozen=True)
class OofFeatures:
    x: Optional[np.ndarray]  # out-of-fold logits aligned with labeled_indices
    h: Optional[np.ndarray]  # out-of-fold refined labels (0.0 / 1.0)
    reason: str              # "" when available


@dataclass(frozen=True)
class LearnedCalibrationResult:
    final_pos: np.ndarray    # bool[N]
    calib_idx: np.ndarray    # rows in A (LLM-labelled)
    verify_idx: np.ndarray   # rows in D (LLM-labelled by routing or exploration)
    status: str              # "routed" | "insufficient_evidence"
    diagnostics: Dict[str, Any]


def learned_calibration_total_budget(n_items: int, budget_pct: float) -> int:
    """Total LLM-call budget B, with the same floor run.py uses today."""
    return max(MIN_TOTAL_BUDGET, int(n_items * budget_pct / 100))


def compute_oof_features(
    refine_fn: RefineFn,
    labeled_indices: np.ndarray,
    labeled_values: np.ndarray,
    *,
    n_folds: int,
    seed: int,
) -> OofFeatures:
    """Each labelled row's logit/refined label from a model that never saw its label."""
    idx = np.asarray(labeled_indices, dtype=np.int64).reshape(-1)
    y = np.asarray(labeled_values, dtype=np.int32).reshape(-1)
    counts = np.bincount(y, minlength=2)
    k = min(int(n_folds), int(counts.min()))
    if k < 2:
        return OofFeatures(None, None, "too_few_labels_per_class")
    x_oof = np.zeros(idx.size, dtype=np.float64)
    h_oof = np.zeros(idx.size, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    for train_pos, hold_pos in splitter.split(idx, y):
        mask, logit = refine_fn(idx[train_pos], y[train_pos])
        if logit is None:
            return OofFeatures(None, None, "fold_logit_unavailable")
        x_oof[hold_pos] = np.asarray(logit, dtype=np.float64)[idx[hold_pos]]
        h_oof[hold_pos] = np.asarray(mask, dtype=bool)[idx[hold_pos]].astype(np.float64)
    return OofFeatures(x_oof, h_oof, "")


def _label(label_rows: LabelRowsFn, rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.zeros(0, dtype=np.int32)
    labels = np.asarray(label_rows(rows), dtype=np.int32).reshape(-1)
    if labels.size != rows.size:
        raise ValueError(f"label_rows returned {labels.size} labels for {rows.size} rows")
    return labels


def run_learned_calibration(
    *,
    x: Optional[np.ndarray],
    h: np.ndarray,
    labeled_indices: np.ndarray,
    labeled_values: np.ndarray,
    oof: OofFeatures,
    label_rows: LabelRowsFn,
    config: LearnedCalibrationConfig,
) -> LearnedCalibrationResult:
    """Stages 2-5: sample A, calibrate on A+L, choose dual thresholds, verify D."""
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    n = h.size
    l_idx = np.asarray(labeled_indices, dtype=np.int64).reshape(-1)
    l_y = np.asarray(labeled_values, dtype=np.int32).reshape(-1)
    rng = np.random.default_rng(int(config.seed))

    verified = np.zeros(n, dtype=bool)
    y_known = np.zeros(n, dtype=np.int32)
    verified[l_idx] = True
    y_known[l_idx] = l_y
    remaining = config.total_budget - l_idx.size
    diag: Dict[str, Any] = {
        "lc_total_budget": int(config.total_budget),
        "lc_sample_budget": int(config.sample_budget),
        "lc_calib_budget": int(config.calib_budget),
        "lc_calls_L": int(l_idx.size),
        "lc_L_pos": int(l_y.sum()),
        "lc_oof_reason": oof.reason,
    }

    # Stage 2: stratified calibration sample A on the frozen proxy score s
    pool = np.flatnonzero(~verified)
    a_idx = np.zeros(0, dtype=np.int64)
    fit = None
    if x is None:
        reason = "proxy_logit_unavailable"
    else:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        s = expit(x)
        m_a = max(0, min(config.calib_budget, pool.size, remaining))
        sample = draw_stratified_sample(pool, s, m_a, n_bins=config.n_bins, rng=rng)
        a_idx = sample.indices
        a_y = _label(label_rows, a_idx)
        remaining -= a_idx.size
        verified[a_idx] = True
        y_known[a_idx] = a_y
        diag["lc_n_strata"] = int(sample.stratum_sizes.size)

        # Stage 3: A+L weighted Platt fit, then A-only intercept shift
        use_l = oof.x is not None
        fit = calibrate(
            x_a=x[a_idx], h_a=h[a_idx], y_a=a_y, w_a=sample.weights,
            x_l=oof.x if use_l else None, h_l=oof.h if use_l else None, y_l=l_y if use_l else None,
            h_pool=h[pool], lam=config.lam, rho=config.rho,
        )
        reason = fit.reason
        diag["lc_A_pos"] = int(a_y.sum())
        diag["lc_ess_pos"] = fit.diagnostics.get("ess_pos")
        diag["lc_max_pos_weight_share"] = fit.diagnostics.get("max_pos_weight_share")
        diag["lc_lambda_used"] = fit.diagnostics.get("lambda_used")
    diag["lc_calls_A"] = int(a_idx.size)

    unverified = np.flatnonzero(~verified)
    budget_left = max(0, remaining)
    if fit is not None and fit.status == "ok":
        # Stages 4-5: dual thresholds on calibrated probabilities, one allocation
        params = fit.params
        p = predict_proba(params, x[unverified], h[unverified])
        counts = expected_counts(p, h[unverified], n_verified_pos=int(y_known[verified].sum()))
        # Order by the raw logit: p is monotone in x within each h group, and x does
        # not saturate the way sigmoid(x) does for |x| > ~37.
        plan = plan_dual_threshold(unverified, p, x[unverified], h[unverified], counts, budget_left)
        d_idx = plan.verify_idx
        status = "routed"
        diag.update({
            "lc_a": params.a, "lc_b": params.b, "lc_c": params.c, "lc_delta": params.delta,
            "lc_t": params.t, "lc_b_final": params.b_final,
            "lc_intercept_residual": fit.diagnostics.get("intercept_residual"),
            "lc_k_plus": plan.k_plus, "lc_k_minus": plan.k_minus,
            "lc_tau_plus": float(expit(plan.tau_plus)), "lc_tau_minus": float(expit(plan.tau_minus)),
            "lc_planned_f1": plan.f_tilde,
        })
    else:
        # insufficient evidence: spend what is left uniformly, report no quality estimate
        take = min(budget_left, unverified.size)
        d_idx = np.sort(rng.choice(unverified, size=take, replace=False)) if take else np.zeros(0, dtype=np.int64)
        status = "insufficient_evidence"
        plan = None

    d_y = _label(label_rows, d_idx)
    final_pos = h > 0.5
    final_pos[verified] = y_known[verified] == 1
    final_pos[d_idx] = d_y == 1

    if plan is not None:
        in_d = np.isin(unverified, d_idx)
        after = apply_verified_labels(counts, p[in_d], h[unverified][in_d], d_y)
        diag["lc_estimated_f1"] = after.f1()
    else:
        diag["lc_estimated_f1"] = float("nan")

    calls_total = int(l_idx.size + a_idx.size + d_idx.size)
    diag.update({
        "lc_status": status,
        "lc_reason": reason,
        "lc_calls_D": int(d_idx.size),
        "lc_D_pos": int(d_y.sum()),
        "lc_calls_total": calls_total,
        "lc_budget_overrun": max(0, calls_total - int(config.total_budget)),
    })
    return LearnedCalibrationResult(final_pos, a_idx, d_idx, status, diag)
