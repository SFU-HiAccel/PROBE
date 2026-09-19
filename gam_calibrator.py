"""GAMCAL-style calibrator: a monotone-prior spline in log-odds space, with pointwise uncertainty.

A drop-in alternative to the Platt calibrator in ``weighted_calibrator.py``, following GAMCAL
(arXiv:2604.00660, "Streaming Model Cascades for Semantic SQL", Snowflake). Platt fits a LINE in
log-odds space; this fits a SPLINE, and additionally reports the standard error of that fit, which
Platt has no analogue for.

What the paper specifies, and what is ours
------------------------------------------
Paper, Eq. 24 (§5.2):      log( g(s) / (1 - g(s)) ) = f(s),  f a cubic B-spline, g' >= 0 enforced
Paper, Eq. 26 (§5.2):      g~(s, q) = sigmoid( f_hat(s) + Phi^-1(q) * se(s) )
Paper, Eq. 35-37 (§A.2):   f(s) = sum_j theta_j * phi_j(logit s), ridge toward a non-zero prior mean
                           mu_theta that encodes the identity map, Laplace covariance for se.

This module implements the Appendix A.2 backend, for three reasons the paper itself gives or that
apply to our setting: A.2 is "preferable when oracle samples are scarce" (our regime -- A is tens of
rows); it accepts sample weights, which the primary variant does not (GAMCAL samples uniformly and
argues weights are therefore unnecessary -- we sample stratified, so we need them); and its prior
mean means that at large lambda it provably COINCIDES with Platt scaling, which makes the A/B a clean
one-knob comparison rather than two unrelated models.

Ours, not the paper's:
  * Extra parametric columns ``h`` (graph-propagated label) and ``1[L]`` (row came from the active
    sample), matching what the Platt model already conditions on. GAMCAL is strictly univariate, so a
    faithful arm is available via ``n_extra=0``; without it the A/B would confound "spline vs line"
    with "loses two features".
  * The A-only intercept shift, kept from the Platt path so the deployed positive rate still matches
    the weighted observed rate on A.
  * Weights normalised to sum to |S|. The paper's lambda is a fixed user input with no GCV/REML, so
    the weight scale would otherwise silently change what a given lambda means.

Known weakness, flagged rather than hidden: under stratified (not uniform) sampling the Laplace
``se`` is a pseudo-likelihood standard error and is under-dispersed, so Eq. 26's exploration is
weaker than the paper's. A design-based sandwich covariance would be the principled fix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.preprocessing import SplineTransformer

from weighted_calibrator import (
    CalibrationFit,
    fit_intercept_shift,
    positive_weight_diagnostics,
    stable_bce,
    stratified_positive_target,
    _evidence_problem,
)

__all__ = ["GamParams", "calibrate_gam", "DEFAULT_N_KNOTS", "DEFAULT_LOGIT_CLIP"]

# The paper never states a knot count, for either variant (see its Alg. 2 input list: lambda and
# n_min are inputs, p is not). 8 knots on a sample of tens of rows keeps the basis well under the
# label count while still bending; it is our choice and should be reported as such.
DEFAULT_N_KNOTS = 8
DEFAULT_LOGIT_CLIP = 12.0   # also unspecified by the paper; +-12 in log-odds is already p in 1e-6..1-1e-6


@dataclass(frozen=True)
class GamParams:
    """Fitted spline coefficients plus everything needed to score new rows."""
    theta: np.ndarray               # spline + extra-column coefficients
    t: float                        # A-only intercept shift, as in the Platt path
    cov: np.ndarray                 # Laplace covariance (Eq. 37), for se(s)
    spline: Any = field(repr=False) # fitted SplineTransformer
    clip: float = DEFAULT_LOGIT_CLIP
    n_extra: int = 2                # parametric columns after the basis: h, 1[L]

    def design(self, x: np.ndarray, h: np.ndarray, in_l: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=np.float64).reshape(-1, 1), -self.clip, self.clip)
        cols = [self.spline.transform(x)]
        if self.n_extra >= 1:
            cols.append(np.asarray(h, dtype=np.float64).reshape(-1, 1))
        if self.n_extra >= 2:
            l = np.zeros(x.shape[0]) if in_l is None else np.asarray(in_l, dtype=np.float64)
            cols.append(l.reshape(-1, 1))
        return np.hstack(cols)

    def linear_predictor(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """f_hat(s) on deployment rows: the 1[L] column is 0, as in Platt's deployed form."""
        return self.design(x, h) @ self.theta + self.t

    def predict(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        return expit(self.linear_predictor(x, h))

    def se(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Pointwise standard error of f_hat (Eq. 37): sqrt(phi^T Sigma phi)."""
        d = self.design(x, h)
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.cov, d), 0.0))

    def predict_stochastic(self, x: np.ndarray, h: np.ndarray, q: np.ndarray) -> np.ndarray:
        """GAMCAL Eq. 26: g~(s, q) = sigmoid(f_hat(s) + Phi^-1(q) * se(s)).

        ``q`` is drawn ONCE per row and frozen for its lifetime (paper §5.3), so routing stays
        consistent across refits. Rows in poorly-calibrated regions get more dispersed scores and are
        pulled into the uncertain band -- this is GAMCAL's whole exploration mechanism, and it is what
        the Platt arm has no counterpart for.
        """
        z = self.linear_predictor(x, h) + norm.ppf(np.clip(np.asarray(q, dtype=np.float64), 1e-6, 1 - 1e-6)) \
            * self.se(x, h)
        return expit(z)


def _identity_prior(n_basis: int, clip: float) -> np.ndarray:
    """mu_theta encoding the identity map in log-odds (Eq. 36's prior mean).

    With a large ridge the fit is pulled to this, which reproduces Platt scaling -- the paper's
    stated reason for the prior, and what makes the GAM arm degrade to the Platt arm rather than to
    something arbitrary.
    """
    return np.linspace(-clip, clip, n_basis)


def _objective(theta: np.ndarray, X: np.ndarray, y: np.ndarray, w: np.ndarray,
               mu: np.ndarray, lam: float) -> Tuple[float, np.ndarray]:
    z = X @ theta
    loss = float(np.sum(w * stable_bce(z, y)) + 0.5 * lam * np.sum((theta - mu) ** 2))
    grad = X.T @ (w * (expit(z) - y)) + lam * (theta - mu)
    return loss, grad


def calibrate_gam(
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
    ridge: float = 1.0,
    n_knots: int = DEFAULT_N_KNOTS,
    clip: float = DEFAULT_LOGIT_CLIP,
    univariate: bool = False,
    cp_lam: float = 0.0,
    cp_alpha: float = 0.05,
) -> CalibrationFit:
    """Fit the A+L weighted spline, then the A-only intercept shift.

    Signature mirrors ``weighted_calibrator.calibrate`` so the two are interchangeable.
    ``lam`` keeps its meaning there (the A/L mixing weight); ``ridge`` is GAMCAL's smoothing
    lambda on ``||theta - mu||^2``. ``univariate=True`` drops h and 1[L] for the paper-faithful arm.
    """
    x_a, h_a, y_a, w_a = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (x_a, h_a, y_a, w_a))
    h_pool = np.asarray(h_pool, dtype=np.float64).reshape(-1)
    diagnostics: Dict[str, Any] = {"n_a": int(y_a.size), "gam_n_knots": int(n_knots),
                                   "gam_ridge": float(ridge), "gam_univariate": bool(univariate),
                                   **positive_weight_diagnostics(y_a, w_a)}

    problem = _evidence_problem(y_a, h_a, h_pool)
    conservative = (problem == "single_class_in_A" and cp_lam > 0.0 and float(y_a.sum()) == 0.0
                    and x_l is not None and y_l is not None and np.asarray(y_l).sum() > 0)
    if problem and not conservative:
        return CalibrationFit(None, "insufficient_evidence", problem, diagnostics)

    has_l = x_l is not None and h_l is not None and y_l is not None and np.asarray(y_l).size > 0
    lam_used = float(lam) if has_l else 0.0
    diagnostics["lambda_used"] = lam_used
    diagnostics["n_l"] = int(np.asarray(y_l).size) if has_l else 0

    # One design matrix over A and L. A carries inverse-inclusion weights scaled to (1 - lam) of the
    # total mass and L a flat (lam) share, matching how the Platt objective mixes the two samples.
    xs, hs, ys, ws, in_l = [x_a], [h_a], [y_a], [], []
    wa = w_a / max(w_a.sum(), 1e-12) * (1.0 - lam_used if has_l else 1.0)
    ws.append(wa)
    in_l.append(np.zeros(y_a.size))
    if has_l:
        x_l2, h_l2, y_l2 = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (x_l, h_l, y_l))
        xs.append(x_l2); hs.append(h_l2); ys.append(y_l2)
        ws.append(np.full(y_l2.size, lam_used / max(y_l2.size, 1)))
        in_l.append(np.ones(y_l2.size))
    x_all = np.concatenate(xs); h_all = np.concatenate(hs)
    y_all = np.concatenate(ys); w_all = np.concatenate(ws); l_all = np.concatenate(in_l)
    # The paper's lambda is a fixed input with no GCV, so the weight SCALE would otherwise change
    # what a given lambda means. Normalise to sum to the sample count.
    w_all = w_all / max(w_all.sum(), 1e-12) * w_all.size

    xc = np.clip(x_all, -clip, clip).reshape(-1, 1)
    # Knots uniform in LOG-ODDS, not in the raw score: for a low-base-rate filter that puts
    # resolution where the rows are instead of crowding it all near p = 0.
    n_knots = int(max(3, min(n_knots, max(3, np.unique(xc).size))))
    spline = SplineTransformer(n_knots=n_knots, degree=3, extrapolation="linear",
                               include_bias=True, knots="uniform").fit(xc)
    n_extra = 0 if univariate else 2
    proto = GamParams(theta=np.zeros(1), t=0.0, cov=np.zeros((1, 1)), spline=spline,
                      clip=clip, n_extra=n_extra)
    X = proto.design(x_all, h_all, l_all)
    n_basis = X.shape[1] - n_extra
    mu = np.zeros(X.shape[1])
    mu[:n_basis] = _identity_prior(n_basis, clip)

    res = minimize(_objective, x0=mu, args=(X, y_all, w_all, mu, float(ridge)),
                   jac=True, method="L-BFGS-B")
    diagnostics["solver_success"] = bool(res.success)
    diagnostics["solver_message"] = str(res.message)
    if not res.success or not np.all(np.isfinite(res.x)):
        return CalibrationFit(None, "solver_failed", "lbfgsb_failed", diagnostics)
    theta = np.asarray(res.x, dtype=np.float64)

    # Eq. 37: Laplace covariance at the optimum.
    p_hat = expit(X @ theta)
    H = X.T @ (X * (p_hat * (1.0 - p_hat) * w_all)[:, None]) + float(ridge) * np.eye(X.shape[1])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    params = GamParams(theta=theta, t=0.0, cov=cov, spline=spline, clip=clip, n_extra=n_extra)
    z_a = params.design(x_a, h_a) @ theta
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

    params = GamParams(theta=theta, t=float(t), cov=cov, spline=spline, clip=clip, n_extra=n_extra)
    diagnostics["intercept_residual"] = float(np.sum(w_a * expit(z_a + t)) - np.sum(w_a * y_a))
    diagnostics["gam_se_median"] = float(np.median(params.se(x_a, h_a)))
    # A monotone g is what the paper enforces by construction; the A.2 backend only encourages it
    # through mu_theta, so check rather than assume.
    grid = np.linspace(-clip, clip, 201)
    p_grid = params.predict(grid, np.zeros_like(grid))
    diagnostics["gam_monotone"] = bool(np.all(np.diff(p_grid) >= -1e-9))
    return CalibrationFit(params, "ok", "cp_conservative" if conservative else "", diagnostics)
