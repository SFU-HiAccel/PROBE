"""Weighted Platt calibrator for learned calibration (design3 §5).

Model on training rows:   z = a*x + b + c*h + delta*I[row in L],  a >= 0
Deployed probability:     p = sigmoid(a*x + (b + t) + c*h)

x is the raw proxy LR logit, h the refined (graph-propagated) label. The fit
mixes the active sample L (mean BCE, weight lam) with the stratified sample A
(inverse-inclusion-weighted BCE, weight 1 - lam) plus ridge terms. The final
intercept shift t is then fit on A alone so that the weighted predicted
positive rate on A equals the weighted observed rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit
from scipy.stats import beta

__all__ = [
    "PlattParams",
    "CalibrationFit",
    "stable_bce",
    "predict_proba",
    "fit_intercept_shift",
    "positive_weight_diagnostics",
    "calibrate",
]


@dataclass(frozen=True)
class PlattParams:
    a: float
    b: float
    c: float
    delta: float
    t: float

    @property
    def b_final(self) -> float:
        return self.b + self.t


@dataclass(frozen=True)
class CalibrationFit:
    params: Optional[PlattParams]
    status: str  # "ok" | "insufficient_evidence" | "solver_failed"
    reason: str
    diagnostics: Dict[str, Any]


def stable_bce(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Elementwise BCE of logits ``z`` against 0/1 labels ``y`` (no overflow)."""
    return np.logaddexp(0.0, z) - y * z


def predict_proba(params: PlattParams, x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return expit(params.a * np.asarray(x, dtype=np.float64)
                 + params.b_final + params.c * np.asarray(h, dtype=np.float64))


def _objective(
    theta: np.ndarray,
    a_rows: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    l_rows: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    lam: float,
    rho: Tuple[float, float, float, float],
) -> Tuple[float, np.ndarray]:
    a, b, c, d = theta
    x_a, h_a, y_a, w_a = a_rows
    w_sum = float(w_a.sum())
    z_a = a * x_a + b + c * h_a
    g_a = w_a * (expit(z_a) - y_a) / w_sum
    loss = (1.0 - lam) * float(np.sum(w_a * stable_bce(z_a, y_a))) / w_sum
    grad = (1.0 - lam) * np.array([np.sum(g_a * x_a), np.sum(g_a), np.sum(g_a * h_a), 0.0])
    if l_rows is not None and lam > 0.0:
        x_l, h_l, y_l = l_rows
        z_l = a * x_l + b + c * h_l + d
        g_l = (expit(z_l) - y_l) / y_l.size
        loss += lam * float(np.mean(stable_bce(z_l, y_l)))
        grad += lam * np.array([np.sum(g_l * x_l), np.sum(g_l), np.sum(g_l * h_l), np.sum(g_l)])
    rho_a, rho_b, rho_c, rho_d = rho
    loss += 0.5 * (rho_a * (a - 1.0) ** 2 + rho_b * b ** 2 + rho_c * c ** 2 + rho_d * d ** 2)
    grad += np.array([rho_a * (a - 1.0), rho_b * b, rho_c * c, rho_d * d])
    return loss, grad


def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial rate; 1.0 when n == 0."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def stratified_positive_target(y: np.ndarray, w: np.ndarray, *, lam: float, alpha: float,
                               strata: Optional[np.ndarray] = None) -> Tuple[float, Dict[str, Any]]:
    """Expected positives in the pool, mixing the measured rate with a Clopper-Pearson upper bound.

    Follows Kim et al. (arXiv:2606.08090 §5.2): ``(1-lam) * measured + lam * CP_upper`` per score range,
    so a range the sample covers densely keeps its measured rate while a sparse one is inflated. Strata
    are recovered from the inverse-inclusion weights (all rows of stratum j share w = N_j / n_j), and the
    failure probability is split across strata (Bonferroni). With no positive anywhere in A the measured
    part is 0 and the bound alone carries the target, which is what makes the intercept solvable.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return 0.0, {"cp_strata": 0}
    groups = w if strata is None else np.asarray(strata).reshape(-1)
    if groups.shape != y.shape:
        raise ValueError("strata must align with calibration labels")
    uniq = np.unique(groups)
    per_stratum_alpha = float(alpha) / max(1, uniq.size)
    measured = upper = 0.0
    for group in uniq:
        rows = groups == group
        n_j = int(rows.sum())
        k_j = int(y[rows].sum())
        size_j = float(w[rows].sum())
        measured += size_j * (k_j / n_j)
        upper += size_j * clopper_pearson_upper(k_j, n_j, per_stratum_alpha)
    target = (1.0 - lam) * measured + lam * upper
    return target, {"cp_strata": int(uniq.size), "cp_measured_pos": measured, "cp_upper_pos": upper,
                    "cp_alpha_per_stratum": per_stratum_alpha}


def fit_intercept_shift(z: np.ndarray, y: np.ndarray, w: np.ndarray, *, max_doublings: int = 60,
                        target: Optional[float] = None) -> float:
    """Solve ``sum w*sigmoid(z+t) = target`` for t, where target defaults to ``sum w*y``."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    target = float(np.sum(w * y)) if target is None else float(target)
    if not 0.0 < target < float(np.sum(w)):
        raise ValueError("intercept shift needs both label classes in the weighted sample")

    def residual(t: float) -> float:
        return float(np.sum(w * expit(z + t))) - target

    lo, hi = -1.0, 1.0
    for _ in range(max_doublings):
        if residual(lo) <= 0.0:
            break
        lo *= 2.0
    for _ in range(max_doublings):
        if residual(hi) >= 0.0:
            break
        hi *= 2.0
    if residual(lo) > 0.0 or residual(hi) < 0.0:
        raise RuntimeError("could not bracket the intercept shift")
    return float(brentq(residual, lo, hi, xtol=1e-12))


def positive_weight_diagnostics(y: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    """K_A, positive-weight ESS and the largest single positive-weight share (§5.5)."""
    pos_w = np.asarray(w, dtype=np.float64) * np.asarray(y, dtype=np.float64)
    k = int(np.sum(np.asarray(y) > 0.5))
    if k == 0:
        return {"k_pos": 0, "ess_pos": float("nan"), "max_pos_weight_share": float("nan")}
    total = float(pos_w.sum())
    return {
        "k_pos": k,
        "ess_pos": total ** 2 / float(np.sum(pos_w ** 2)),
        "max_pos_weight_share": float(pos_w.max()) / total,
    }


def _l_enters_fit(x_l, h_l, y_l, lam: float) -> bool:
    """The out-of-fold rows of L enter the loss only when they are present and weighted (lam > 0)."""
    return (x_l is not None and h_l is not None and y_l is not None
            and np.asarray(y_l).size > 0 and float(lam) > 0.0)


def _evidence_problem(y_a: np.ndarray, h_a: np.ndarray, h_pool: np.ndarray,
                      h_l: Optional[np.ndarray] = None) -> str:
    """Why the fit is not identified, or "" when it is.

    Every refined-label value that occurs in the pool must be seen by the fit, or the c*h term is not identified
    for that group. The fit sees A and, when its term is on, the out-of-fold rows of L (pass their h as h_l).
    """
    if y_a.size == 0:
        return "empty_calibration_sample"
    if np.unique(y_a).size < 2:
        return "single_class_in_A"
    seen = set(np.unique(h_a).tolist())
    if h_l is not None:
        seen |= set(np.unique(np.asarray(h_l, dtype=np.float64)).tolist())
    if set(np.unique(h_pool).tolist()) - seen:
        return "h_value_missing_in_A_and_L"
    return ""


def _calibrate_paper(x_a, y_a, w_a, x_l, y_l, *, cp_lam, cp_alpha, strata):
    """Section 5: shared slope, independent A/L intercepts, no ridge penalty.

    A contributes hard labels with weights omega/mean(omega). Only L uses
    Platt targets, computed from L's own class counts. If A has no positives,
    fit the slope on L alone and anchor the A intercept to the CP heuristic.
    """
    diag = {"n_a": int(y_a.size), **positive_weight_diagnostics(y_a, w_a)}
    if not y_a.size:
        return CalibrationFit(None, "insufficient_evidence", "empty_calibration_sample", diag)
    if not (x_a.shape == y_a.shape == w_a.shape) or not np.all(w_a > 0):
        raise ValueError("calibration arrays must align and weights must be positive")
    use_l = x_l is not None and y_l is not None and np.asarray(y_l).size > 0
    xl = np.asarray(x_l, dtype=float).reshape(-1) if use_l else np.empty(0)
    yl = np.asarray(y_l, dtype=float).reshape(-1) if use_l else np.empty(0)
    if xl.shape != yl.shape or not all(np.all(np.isfinite(v)) for v in (x_a, y_a, w_a, xl, yl)):
        raise ValueError("calibration inputs must be finite and aligned")
    if not all(np.all((v == 0) | (v == 1)) for v in (y_a, yl)):
        raise ValueError("calibration labels must be binary")
    cp = bool(not y_a.any() and cp_lam > 0 and use_l and np.unique(yl).size == 2)
    if np.unique(y_a).size < 2 and not cp:
        return CalibrationFit(None, "insufficient_evidence", "single_class_in_A", diag)
    npos = int(yl.sum())
    target_l = np.where(yl > .5, (npos + 1.) / (npos + 2.), 1. / (len(yl) - npos + 2.))
    wa = w_a / w_a.mean()
    scale = len(yl) if cp else len(y_a) + len(yl)

    def objective(theta):
        a, c, cl = theta
        loss, grad = 0., np.zeros(3)
        if not cp:
            z = a * x_a + c
            residual = wa * (expit(z) - y_a)
            loss += float(np.dot(wa, stable_bce(z, y_a)))
            grad[:2] += [np.dot(residual, x_a), residual.sum()]
        if use_l:
            z = a * xl + cl
            residual = expit(z) - target_l
            loss += float(stable_bce(z, target_l).sum())
            grad += [np.dot(residual, xl), 0., residual.sum()]
        return loss / scale, grad / scale

    result = minimize(objective, [1., 0., 0.], jac=True, method="L-BFGS-B",
                      bounds=[(0., None), (0., 0.) if cp else (None, None),
                              (None, None) if use_l else (0., 0.)],
                      options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000})
    diag.update(n_l=len(yl), lambda_used=(len(yl) / scale),
                solver_success=bool(result.success), solver_message=str(result.message),
                loss="sum_A omega/mean(omega) BCE(hard) + sum_L BCE(Platt_L)")
    if not result.success or not np.all(np.isfinite(result.x)):
        return CalibrationFit(None, "solver_failed", "lbfgsb_failed", diag)
    a, c, cl = map(float, result.x)
    target = float(np.dot(w_a, y_a))
    if cp:
        target, cp_diag = stratified_positive_target(y_a, w_a, lam=cp_lam, alpha=cp_alpha, strata=strata)
        diag.update(cp_diag, cp_target_pos=target, cp_used=True)
    try:
        # Enforce Eq. anchor to numerical precision. This is the first-order
        # condition for the independent A intercept, not an extra fitting loss.
        c = fit_intercept_shift(a * x_a, y_a, w_a, target=target)
    except (RuntimeError, ValueError) as exc:
        return CalibrationFit(None, "solver_failed", f"intercept_fit_failed: {exc}", diag)
    diag["intercept_residual"] = float(np.dot(w_a, expit(a * x_a + c)) - target)
    params = PlattParams(a=a, b=c, c=0., delta=cl-c if use_l else 0., t=0.)
    return CalibrationFit(params, "ok", "cp_conservative" if cp else "", diag)


def calibrate(
    *,
    x_a: np.ndarray,
    h_a: np.ndarray,
    y_a: np.ndarray,
    w_a: np.ndarray,
    x_l: Optional[np.ndarray],
    h_l: Optional[np.ndarray],
    y_l: Optional[np.ndarray],
    h_pool: np.ndarray,
    lam: float,
    rho: Tuple[float, float, float, float],
    cp_lam: float = 0.0,
    cp_alpha: float = 0.05,
    paper: bool = False,
    strata: Optional[np.ndarray] = None,
) -> CalibrationFit:
    """Fit the A+L weighted Platt model, then the A-only intercept shift."""
    x_a, h_a, y_a, w_a = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (x_a, h_a, y_a, w_a))
    h_pool = np.asarray(h_pool, dtype=np.float64).reshape(-1)
    if paper:
        return _calibrate_paper(x_a, y_a, w_a, x_l, y_l,
                                cp_lam=cp_lam, cp_alpha=cp_alpha, strata=strata)
    diagnostics: Dict[str, Any] = {"n_a": int(y_a.size), **positive_weight_diagnostics(y_a, w_a)}
    h_l_fit = h_l if _l_enters_fit(x_l, h_l, y_l, lam) else None
    problem = _evidence_problem(y_a, h_a, h_pool, h_l_fit)
    # which refined-label values each set holds (and their counts), to diagnose an unidentified c*h term
    _vals = lambda v: "none" if v is None else ";".join(
        f"{int(k)}:{int(n)}" for k, n in zip(*np.unique(np.asarray(v, dtype=np.float64), return_counts=True)))
    diagnostics.update(h_pool_values=_vals(h_pool), h_a_values=_vals(h_a), h_l_values=_vals(h_l_fit))
    # A with no positive at all normally stops the calibration: the intercept equation
    # sum w*sigmoid(z+t) = 0 has no finite solution. With cp_lam > 0 the target becomes a
    # Clopper-Pearson mix instead of the measured 0, so the level is anchored conservatively and the
    # shape still comes from L. Without L there is nothing to give the shape, so keep refusing.
    conservative = (problem == "single_class_in_A" and cp_lam > 0.0 and float(y_a.sum()) == 0.0
                    and _l_enters_fit(x_l, h_l, y_l, lam) and np.asarray(y_l).sum() > 0)
    if problem and not conservative:
        return CalibrationFit(None, "insufficient_evidence", problem, diagnostics)

    l_rows = None
    if x_l is not None and h_l is not None and y_l is not None and np.asarray(y_l).size > 0:
        l_rows = tuple(np.asarray(v, dtype=np.float64).reshape(-1) for v in (x_l, h_l, y_l))
    lam_used = float(lam) if l_rows is not None else 0.0
    diagnostics["lambda_used"] = lam_used
    diagnostics["n_l"] = 0 if l_rows is None else int(l_rows[2].size)

    result = minimize(
        _objective,
        x0=np.array([1.0, 0.0, 0.0, 0.0]),
        args=((x_a, h_a, y_a, w_a), l_rows, lam_used, rho),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None), (None, None), (None, None), (None, None)],
    )
    diagnostics["solver_success"] = bool(result.success)
    diagnostics["solver_message"] = str(result.message)
    a, b, c, d = (float(v) for v in result.x)
    if not result.success or not np.all(np.isfinite(result.x)):
        return CalibrationFit(None, "solver_failed", "lbfgsb_failed", diagnostics)

    z_a = a * x_a + b + c * h_a
    cp_target = None
    if conservative:
        cp_target, cp_diag = stratified_positive_target(y_a, w_a, lam=cp_lam, alpha=cp_alpha)
        diagnostics.update(cp_diag)
        diagnostics["cp_target_pos"] = float(cp_target)
        diagnostics["cp_used"] = True
    try:
        t = fit_intercept_shift(z_a, y_a, w_a, target=cp_target)
    except (RuntimeError, ValueError) as exc:
        return CalibrationFit(None, "solver_failed", f"intercept_shift_failed: {exc}", diagnostics)
    params = PlattParams(a=a, b=b, c=c, delta=d, t=t)
    diagnostics["intercept_residual"] = float(np.sum(w_a * expit(z_a + t)) - np.sum(w_a * y_a))
    return CalibrationFit(params, "ok", "cp_conservative" if conservative else "", diagnostics)
