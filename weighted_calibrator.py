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


def fit_intercept_shift(z: np.ndarray, y: np.ndarray, w: np.ndarray, *, max_doublings: int = 60) -> float:
    """Solve ``sum w*sigmoid(z+t) = sum w*y`` for t (unique when 0 < sum w*y < sum w)."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    target = float(np.sum(w * y))
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


def _evidence_problem(y_a: np.ndarray, h_a: np.ndarray, h_pool: np.ndarray) -> str:
    if y_a.size == 0:
        return "empty_calibration_sample"
    if np.unique(y_a).size < 2:
        return "single_class_in_A"
    missing = set(np.unique(h_pool).tolist()) - set(np.unique(h_a).tolist())
    if missing:
        return "h_value_missing_in_A"
    return ""


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
) -> CalibrationFit:
    """Fit the A+L weighted Platt model, then the A-only intercept shift."""
    x_a, h_a, y_a, w_a = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (x_a, h_a, y_a, w_a))
    h_pool = np.asarray(h_pool, dtype=np.float64).reshape(-1)
    diagnostics: Dict[str, Any] = {"n_a": int(y_a.size), **positive_weight_diagnostics(y_a, w_a)}
    problem = _evidence_problem(y_a, h_a, h_pool)
    if problem:
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
    try:
        t = fit_intercept_shift(z_a, y_a, w_a)
    except (RuntimeError, ValueError) as exc:
        return CalibrationFit(None, "solver_failed", f"intercept_shift_failed: {exc}", diagnostics)
    params = PlattParams(a=a, b=b, c=c, delta=d, t=t)
    diagnostics["intercept_residual"] = float(np.sum(w_a * expit(z_a + t)) - np.sum(w_a * y_a))
    return CalibrationFit(params, "ok", "", diagnostics)
