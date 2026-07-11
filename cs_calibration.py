"""GAMCAL-style dual-threshold calibration for the DIRECT + Correct & Smooth path.

Implements the calibration + optimisation stage of GAMCAL (Liskowski &
Schmaus, "Streaming Model Cascades for Semantic SQL", 2026 — see
``docs/model_cascade_summary.md``) adapted to LLMSQL's offline pipeline.

What this module does:

1. Fits a monotone-prior logistic GAM (cubic B-spline basis +
   Laplace approximation, the Appendix A.2 variant) on labelled
   ``z_smooth`` outputs from Correct & Smooth. The Appendix A.2
   variant is used in place of pyGAM because it relies only on
   scipy + sklearn, both already in the project's dependencies.

2. Provides per-row calibrated probabilities ``g(s)`` and pointwise
   uncertainty ``se(s)``. Uses the stochastic quantile mechanism
   (Eq. 26) to push poorly-calibrated rows into the uncertain region.

3. Optimises a pair of thresholds ``(tau_low, tau_high)`` against
   the closed-form cost-quality objective (Eq. 31) via differential
   evolution. ``tau_low <= tau_high`` is enforced by the standard
   ``y2``-fraction reparameterisation (Eq. 32).

The exposed entry point is :func:`fit_and_optimize_dual_thresholds`,
which the C&S pipeline calls to pick (tau_low, tau_high) for the
DIRECT + Correct & Smooth path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.preprocessing import SplineTransformer

logger = logging.getLogger(__name__)


_SCORE_EPS = 1e-4  # avoid logit(0) / logit(1) blow-ups
_LOGIT_CLAMP = 8.0  # log-odds clamp; |logit| <= 8 covers p in [3.4e-4, 1-3.4e-4]
_SE_MAX = 6.0  # cap se(s) to avoid overflow in sigmoid


def _safe_logit(p: np.ndarray) -> np.ndarray:
    """Clipped logit; output is clamped to [-_LOGIT_CLAMP, _LOGIT_CLAMP]."""
    p = np.clip(p.astype(np.float64), _SCORE_EPS, 1.0 - _SCORE_EPS)
    out = np.log(p) - np.log1p(-p)
    return np.clip(out, -_LOGIT_CLAMP, _LOGIT_CLAMP)


@dataclass
class CalibrationResult:
    """Output of :class:`SplineGAMCalibrator.fit`."""

    method: str
    n_labeled: int
    n_pos: int
    n_neg: int
    fit_time: float
    n_basis: int
    lam: float
    converged: bool


class SplineGAMCalibrator:
    """Logistic GAM with cubic B-spline basis + Laplace-approx uncertainty.

    Implements Appendix A.2 of the model-cascades paper. The roughness
    of the calibration curve is controlled by ``lam`` (the L2 penalty
    on coefficient deviations from a monotone identity-in-log-odds
    prior). Larger ``lam`` -> smoother fit, useful when oracle labels
    are scarce.

    Notes:
        Monotonicity is encouraged but not strictly enforced. The
        identity-in-log-odds prior plus the smoothness of the cubic
        spline basis produces approximately monotone calibration
        curves in practice. For DIRECT + Correct & Smooth, the
        labelled set is centred on the version-space boundary so a
        monotonic prior is a sensible inductive bias.
    """

    def __init__(
        self,
        n_knots: int = 8,
        degree: int = 3,
        lam: float = 1.0,
        max_iter: int = 200,
    ) -> None:
        if n_knots < 4:
            raise ValueError(f"n_knots must be >= 4, got {n_knots}")
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if lam <= 0.0:
            raise ValueError(f"lam must be > 0, got {lam}")
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.lam = float(lam)
        self.max_iter = int(max_iter)

        # Lazily initialised in fit()
        self._spline: Optional[SplineTransformer] = None
        self._theta: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._mu_prior: Optional[np.ndarray] = None
        self._fitted = False
        self._info: Optional[CalibrationResult] = None

    @property
    def info(self) -> CalibrationResult:
        if self._info is None:
            raise RuntimeError("SplineGAMCalibrator.fit() not called yet")
        return self._info

    # --- internals --------------------------------------------------

    def _design(self, s: np.ndarray) -> np.ndarray:
        """Return the spline design matrix for raw scores ``s``."""
        if self._spline is None:
            raise RuntimeError("spline transformer not fitted")
        ell = _safe_logit(np.asarray(s, dtype=np.float64).reshape(-1, 1))
        return self._spline.transform(ell)

    def _make_prior(self, d: int) -> np.ndarray:
        """Identity-in-log-odds prior: monotone increasing baseline.

        ``mu_j = ell_min + (ell_max - ell_min) * j / (d - 1)`` produces
        coefficients that, when multiplied by a non-negative
        partition-of-unity B-spline basis, recover an approximately
        identity calibration in log-odds space.
        """
        ell_min, ell_max = -_LOGIT_CLAMP, _LOGIT_CLAMP
        if d <= 1:
            return np.zeros(d, dtype=np.float64)
        return ell_min + (ell_max - ell_min) * np.arange(d) / (d - 1)

    def _neg_log_likelihood(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        w: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Penalised neg log-likelihood + analytical gradient.

        When ``w`` is provided, sample contributions are weighted:
        ``loss = -Σ w_i [y_i log h_i + (1 - y_i) log (1 - h_i)]``
        and the gradient becomes ``Xᵀ (w * (h - y)) + lam (θ - μ)``.
        ``w`` is expected to be 1D and the same length as ``y``; the
        roughness penalty is *not* scaled — it is anchored to the
        identity-in-log-odds prior regardless of effective sample size.
        """
        z = X @ theta
        z = np.clip(z, -_LOGIT_CLAMP * 2.0, _LOGIT_CLAMP * 2.0)
        h = expit(z)
        # log loss with float64 stability
        eps = 1e-12
        per_sample = y * np.log(h + eps) + (1.0 - y) * np.log(1.0 - h + eps)
        if w is None:
            loss = -float(np.sum(per_sample))
            residual = h - y
        else:
            loss = -float(np.dot(w, per_sample))
            residual = w * (h - y)
        diff = theta - self._mu_prior
        loss += 0.5 * self.lam * float(diff @ diff)
        grad = X.T @ residual + self.lam * diff
        return loss, grad

    # --- public API -------------------------------------------------

    def fit(
        self,
        s_labeled: np.ndarray,
        y_labeled: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """Fit the calibrator on labelled scores.

        Args:
            s_labeled: 1D array of raw scores in [0, 1] (e.g. ``z_smooth``
                values at the labelled rows).
            y_labeled: 1D binary array (0/1) of oracle labels.
            sample_weight: optional 1D array of non-negative per-sample
                weights, same length as ``y_labeled``. When provided the
                weighted log-likelihood is optimised and the Laplace
                covariance uses the weighted Fisher information. Use this
                to upweight calibration samples that came from a more
                reliable / more recent base learner (e.g. samples
                predicted by a later DIRECT phase whose LR saw more
                training data). Pass ``None`` (default) for uniform
                weights — equivalent to the previous behaviour.

        Returns:
            CalibrationResult metadata. The fitted model is stored on
            ``self`` and consumed by :meth:`predict`.
        """
        s = np.asarray(s_labeled, dtype=np.float64).reshape(-1)
        y = np.asarray(y_labeled, dtype=np.float64).reshape(-1)
        if s.shape[0] != y.shape[0]:
            raise ValueError(
                f"shape mismatch: s_labeled={s.shape}, y_labeled={y.shape}"
            )
        if s.shape[0] < 8:
            raise ValueError(
                f"need >= 8 labelled samples to fit GAM, got {s.shape[0]}"
            )
        n_pos = int(np.sum(y > 0.5))
        n_neg = int(np.sum(y <= 0.5))
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"need both classes in labelled set: n_pos={n_pos} n_neg={n_neg}"
            )

        # Validate + normalise sample_weight. We rescale to mean=1 so
        # the scale of `lam` (the roughness penalty) keeps the same
        # interpretation regardless of weight magnitudes.
        if sample_weight is None:
            w_arr: Optional[np.ndarray] = None
        else:
            w_arr = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if w_arr.shape[0] != y.shape[0]:
                raise ValueError(
                    f"sample_weight length ({w_arr.shape[0]}) != "
                    f"y_labeled length ({y.shape[0]})"
                )
            if np.any(w_arr < 0.0):
                raise ValueError("sample_weight must be non-negative")
            w_sum = float(w_arr.sum())
            if w_sum <= 0.0:
                raise ValueError("sample_weight sums to zero")
            # Normalise so mean weight = 1 (preserves NLL scale ↔ lam)
            w_arr = w_arr * (float(w_arr.shape[0]) / w_sum)

        t0 = time.time()

        # Fit B-spline basis on the *clamped log-odds* of the labelled
        # scores so the knot range covers the full operating range
        # rather than the (potentially narrow) DIRECT-sample boundary.
        ell = _safe_logit(s.reshape(-1, 1))
        # use a wider sentinel so the basis extrapolates linearly to
        # both tails of the score axis even when DIRECT samples cluster
        knot_anchor = np.array(
            [[-_LOGIT_CLAMP], [_LOGIT_CLAMP]], dtype=np.float64
        )
        ell_for_fit = np.vstack([ell, knot_anchor])

        spline = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            knots="uniform",
            include_bias=True,
            extrapolation="linear",
        )
        spline.fit(ell_for_fit)
        self._spline = spline

        X = spline.transform(ell)
        d = X.shape[1]
        self._mu_prior = self._make_prior(d)
        theta0 = self._mu_prior.copy()

        result = minimize(
            self._neg_log_likelihood,
            theta0,
            args=(X, y, w_arr),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-9, "gtol": 1e-7},
        )
        self._theta = result.x

        # Laplace approximation: weighted Fisher information.
        #   H = X^T diag(w_i * h_i * (1 - h_i)) X + lam * I
        # When ``w_arr is None`` this collapses to the unweighted case.
        z = X @ self._theta
        z = np.clip(z, -_LOGIT_CLAMP * 2.0, _LOGIT_CLAMP * 2.0)
        h = expit(z)
        w_diag = h * (1.0 - h)
        if w_arr is not None:
            w_diag = w_diag * w_arr
        H = (X.T * w_diag) @ X + self.lam * np.eye(d, dtype=np.float64)
        # symmetrise + add jitter for stable inversion
        H = 0.5 * (H + H.T) + 1e-8 * np.eye(d, dtype=np.float64)
        try:
            self._cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            # extremely degenerate case — fall back to pseudo-inverse
            self._cov = np.linalg.pinv(H)

        self._fitted = True
        self._info = CalibrationResult(
            method="spline_gam_laplace",
            n_labeled=int(s.shape[0]),
            n_pos=n_pos,
            n_neg=n_neg,
            fit_time=time.time() - t0,
            n_basis=d,
            lam=self.lam,
            converged=bool(result.success),
        )
        return self._info

    def predict(
        self, s: np.ndarray, return_se: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Calibrated probability mean (and optional pointwise se).

        Returns:
            mean: array of P(y=1 | A=s).
            se: array of standard errors of the *log-odds* prediction
                (i.e. of f̂(s)), suitable for plugging into Eq. 26;
                ``None`` if ``return_se=False``.
        """
        if not self._fitted:
            raise RuntimeError("SplineGAMCalibrator.fit() not called yet")
        X = self._design(s)
        f_hat = X @ self._theta
        f_hat = np.clip(f_hat, -_LOGIT_CLAMP * 2.0, _LOGIT_CLAMP * 2.0)
        mean = expit(f_hat)
        if not return_se:
            return mean.astype(np.float64), None
        # pointwise se(f̂(s)) = sqrt(phi(s)^T Sigma phi(s))
        var = np.einsum("ij,jk,ik->i", X, self._cov, X)
        var = np.maximum(var, 0.0)
        se = np.sqrt(var)
        se = np.minimum(se, _SE_MAX)
        return mean.astype(np.float64), se.astype(np.float64)


# ---------------------------------------------------------------------
# Stochastic calibrated score (Eq. 26)
# ---------------------------------------------------------------------


def stochastic_calibrated_score(
    f_hat: np.ndarray, se: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Eq. 26: g̃(s, q) = sigmoid( f̂(s) + Φ⁻¹(q) · se(s) ).

    Args:
        f_hat: log-odds prediction per row.
        se: pointwise standard error of f̂.
        q: per-row quantile draws in (0, 1), frozen for the lifetime
            of the row.
    """
    q = np.clip(q, 1e-6, 1.0 - 1e-6)
    z_q = norm.ppf(q)
    raw = f_hat + z_q * se
    raw = np.clip(raw, -_LOGIT_CLAMP * 2.0, _LOGIT_CLAMP * 2.0)
    return expit(raw)


# ---------------------------------------------------------------------
# Closed-form expected F-beta and dual-threshold optimisation (Eq. 27-32)
# ---------------------------------------------------------------------


def _expected_metrics(
    g: np.ndarray, tau_low: float, tau_high: float
) -> Tuple[float, float, float, float]:
    """Eq. 27-29 + delegation. Treats g as P(y=1).

    Assumes the uncertain region is fully oracle-resolved (so it
    contributes only to TP through the oracle, never to FP/FN).

    Returns:
        (E[TP], E[FP], E[FN], delegation).
    """
    accept = g >= tau_high
    uncertain = (g >= tau_low) & (g < tau_high)
    reject = g < tau_low

    e_tp_accept = float(g[accept].sum())
    e_tp_uncertain = float(g[uncertain].sum())
    e_tp = e_tp_accept + e_tp_uncertain

    e_fp = float((1.0 - g[accept]).sum())  # accept-region false positives
    e_fn = float(g[reject].sum())  # reject-region false negatives

    deleg = float(uncertain.mean())
    return e_tp, e_fp, e_fn, deleg


def expected_f_beta(
    g: np.ndarray, tau_low: float, tau_high: float, beta: float = 1.0
) -> float:
    """Eq. 30 with handling of the all-zero edge case."""
    e_tp, e_fp, e_fn, _ = _expected_metrics(g, tau_low, tau_high)
    if e_tp <= 0.0:
        return 0.0
    b2 = float(beta) * float(beta)
    denom = (1.0 + b2) * e_tp + e_fn + b2 * e_fp
    if denom <= 0.0:
        return 0.0
    return (1.0 + b2) * e_tp / denom


# Two parameterisations of (tau_low, tau_high) over [0, 1]^2 search space:
#
#   "force_label" (paper Eq. 32, default):
#       tau_low  = y1            in [0, 1]
#       tau_high = y1 + (1 - y1) * y2   in [tau_low, 1]
#       The accept region (g >= tau_high) is force-marked positive and
#       the reject region (g < tau_low) is force-marked negative,
#       regardless of where 0.5 sits.
#
#   "trust_proxy":
#       tau_low  = 0.5 * y1                 in [0, 0.5]
#       tau_high = 0.5 + 0.5 * y2           in [0.5, 1]
#       The optimiser is constrained to never put tau_high below 0.5
#       or tau_low above 0.5, so the accept region is always a subset
#       of "rows the proxy itself classifies as positive". Robust to
#       cold-start GAM fits and small alpha; recommended for low-
#       selectivity workloads.
DECISION_MODE_FORCE_LABEL = "force_label"
DECISION_MODE_TRUST_PROXY = "trust_proxy"
_VALID_DECISION_MODES = (DECISION_MODE_FORCE_LABEL, DECISION_MODE_TRUST_PROXY)


def _decode_thresholds(
    y1: float, y2: float, decision_mode: str
) -> Tuple[float, float]:
    """Map (y1, y2) in [0,1]^2 to (tau_low, tau_high) per decision_mode."""
    y1 = min(max(float(y1), 0.0), 1.0)
    y2 = min(max(float(y2), 0.0), 1.0)
    if decision_mode == DECISION_MODE_TRUST_PROXY:
        tau_low = 0.5 * y1
        tau_high = 0.5 + 0.5 * y2
    elif decision_mode == DECISION_MODE_FORCE_LABEL:
        tau_low = y1
        tau_high = y1 + (1.0 - y1) * y2
    else:
        raise ValueError(
            f"unknown decision_mode {decision_mode!r}, "
            f"expected one of {_VALID_DECISION_MODES}"
        )
    return tau_low, tau_high


def _cost_quality_objective(
    y: np.ndarray,
    g: np.ndarray,
    alpha: float,
    beta: float,
    baseline_err: float,
    decision_mode: str,
) -> float:
    """Eq. 31 instantiation, evaluated at (tau_low, tau_high) encoded by y."""
    tau_low, tau_high = _decode_thresholds(y[0], y[1], decision_mode)
    f1 = expected_f_beta(g, tau_low, tau_high, beta=beta)
    err = (1.0 - f1) / max(baseline_err, 1e-9)
    _, _, _, deleg = _expected_metrics(g, tau_low, tau_high)
    return alpha * err + (1.0 - alpha) * deleg


def optimize_dual_thresholds(
    g: np.ndarray,
    alpha: float = 0.5,
    beta: float = 1.0,
    seed: Optional[int] = 42,
    popsize: int = 15,
    maxiter: int = 60,
    tol: float = 1e-6,
    decision_mode: str = DECISION_MODE_FORCE_LABEL,
) -> Tuple[float, float, Dict[str, float]]:
    """Differential evolution over the (y1, y2) reparameterisation.

    Args:
        g: per-row calibrated probabilities (mean OR stochastic — caller
            decides which to feed in).
        alpha: cost-quality knob in [0, 1]. Higher α → wider uncertain
            band (more delegation, higher F1).
        beta: F_beta weight (β=1 gives F1).
        seed: RNG seed for differential_evolution reproducibility.
        popsize, maxiter, tol: DE hyperparameters.
        decision_mode: ``"force_label"`` (paper §3.2 default — accept
            region always marked positive, reject region always marked
            negative) or ``"trust_proxy"`` (constrained search so
            tau_low <= 0.5 <= tau_high; recommended for low-selectivity
            workloads).

    Returns:
        (tau_low, tau_high, info_dict).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0,1], got {alpha}")
    if decision_mode not in _VALID_DECISION_MODES:
        raise ValueError(
            f"unknown decision_mode {decision_mode!r}, "
            f"expected one of {_VALID_DECISION_MODES}"
        )
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    if g.size == 0:
        return 0.0, 1.0, {"reason": "empty g"}
    g = np.clip(g, 0.0, 1.0)

    # Baseline error at no-delegation, threshold 0.5 (Eq. 31 normaliser)
    baseline_f1 = expected_f_beta(g, 0.5, 0.5, beta=beta)
    baseline_err = max(1.0 - baseline_f1, 1e-3)

    # The (y1, y2) bounds are [0,1]^2 in BOTH modes; the parameterisation
    # inside _decode_thresholds is what differentiates them. Keeping the
    # search box uniform lets DE's Sobol init and population behaviour
    # remain comparable across modes.
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    t0 = time.time()
    result = differential_evolution(
        _cost_quality_objective,
        bounds=bounds,
        args=(g, alpha, beta, baseline_err, decision_mode),
        popsize=int(popsize),
        maxiter=int(maxiter),
        tol=float(tol),
        seed=seed,
        polish=True,
        init="sobol",
        updating="deferred",
        workers=1,  # single-thread; DE objective is cheap (vectorised numpy)
    )
    opt_time = time.time() - t0

    tau_low, tau_high = _decode_thresholds(
        float(result.x[0]), float(result.x[1]), decision_mode
    )

    e_tp, e_fp, e_fn, deleg = _expected_metrics(g, tau_low, tau_high)
    f1_at_opt = expected_f_beta(g, tau_low, tau_high, beta=beta)

    info: Dict[str, float] = {
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "alpha": float(alpha),
        "beta": float(beta),
        "decision_mode": str(decision_mode),
        "expected_f_beta": float(f1_at_opt),
        "baseline_f_beta": float(baseline_f1),
        "delegation_rate": float(deleg),
        "n_accept": int(np.sum(g >= tau_high)),
        "n_reject": int(np.sum(g < tau_low)),
        "n_uncertain": int(np.sum((g >= tau_low) & (g < tau_high))),
        "expected_tp": float(e_tp),
        "expected_fp": float(e_fp),
        "expected_fn": float(e_fn),
        "objective_at_opt": float(result.fun),
        "optimize_time": float(opt_time),
        "de_iterations": int(getattr(result, "nit", -1)),
        "de_converged": bool(result.success),
    }
    return tau_low, tau_high, info


# ---------------------------------------------------------------------
# End-to-end entry point used by cluster_proxy_pipeline
# ---------------------------------------------------------------------


@dataclass
class GamcalDecision:
    """Result of :func:`fit_and_optimize_dual_thresholds`."""

    tau_low: float
    tau_high: float
    g_mean: np.ndarray            # calibrated probability per row, shape (N,)
    g_stochastic: np.ndarray      # Eq. 26 per-row stochastic score, shape (N,)
    g_se: np.ndarray              # log-odds standard error per row, shape (N,)
    accept_mask: np.ndarray       # bool (N,): g_stochastic >= tau_high
    reject_mask: np.ndarray       # bool (N,): g_stochastic < tau_low
    uncertain_mask: np.ndarray    # bool (N,): in the open band
    info: Dict[str, float]


def fit_and_optimize_dual_thresholds(
    z_smooth: np.ndarray,
    s_labeled: np.ndarray,
    y_labeled: np.ndarray,
    *,
    alpha: float = 0.5,
    beta: float = 1.0,
    n_knots: int = 8,
    lam: float = 1.0,
    seed: Optional[int] = 42,
    popsize: int = 15,
    maxiter: int = 60,
    de_tol: float = 1e-6,
    use_stochastic_routing: bool = True,
    decision_mode: str = DECISION_MODE_FORCE_LABEL,
    score_eval: str = "all",  # "all" | "labeled"
    sample_weight: Optional[np.ndarray] = None,
) -> GamcalDecision:
    """Full GAMCAL stage on top of Correct & Smooth's z_smooth output.

    Args:
        z_smooth: (N,) C&S output for every row.
        s_labeled: (M,) z_smooth values at labelled rows. May come from
            an OOF buffer (preferred for DIRECT) or in-bag z_smooth.
        y_labeled: (M,) binary labels.
        alpha: cost-quality tradeoff parameter.
        beta: F_beta weight.
        n_knots, lam: GAM hyperparameters.
        seed, popsize, maxiter, de_tol: DE hyperparameters.
        use_stochastic_routing: when True, route each row by the
            stochastic g̃ from Eq. 26; when False, route by g_mean
            (deterministic).
        score_eval: which set of g values to feed into the threshold
            optimiser. "all" (default, paper's behaviour) optimises
            over the full population; "labeled" restricts the
            objective to labelled rows only (useful for diagnostics).
        sample_weight: optional per-sample weights for the GAM fit
            (not the threshold optimiser). Same length as
            ``y_labeled``, non-negative, and rescaled inside the
            calibrator so the mean weight is 1. Use this to upweight
            calibration rows whose OOF scores were produced by a
            more reliable / more recent base learner (e.g. later
            DIRECT phases). ``None`` (default) = uniform.

    Returns:
        :class:`GamcalDecision` with thresholds, per-row scores, masks,
        and a profiling dict.
    """
    z_smooth = np.asarray(z_smooth, dtype=np.float64).reshape(-1)
    s_labeled = np.asarray(s_labeled, dtype=np.float64).reshape(-1)
    y_labeled = np.asarray(y_labeled, dtype=np.float64).reshape(-1)

    n_total = z_smooth.shape[0]
    if n_total == 0:
        raise ValueError("z_smooth is empty")
    if s_labeled.shape[0] != y_labeled.shape[0]:
        raise ValueError(
            f"labeled length mismatch: s={s_labeled.shape} y={y_labeled.shape}"
        )

    # 1. Fit calibrator
    cal = SplineGAMCalibrator(n_knots=n_knots, lam=lam)
    cal_info = cal.fit(s_labeled, y_labeled, sample_weight=sample_weight)

    # 2. Per-row mean + se on the deployment set
    g_mean, g_se = cal.predict(z_smooth, return_se=True)
    if g_se is None:  # defensive — predict returned None despite return_se=True
        g_se = np.zeros_like(g_mean)

    # 3. Stochastic g̃ via per-row frozen quantile (Eq. 26)
    rng = np.random.default_rng(seed)
    q = rng.uniform(low=1e-6, high=1.0 - 1e-6, size=n_total)
    f_hat = _safe_logit(g_mean)
    g_stoch = stochastic_calibrated_score(f_hat, g_se, q)

    # 4. Optimise (tau_low, tau_high)
    routing_score = g_stoch if use_stochastic_routing else g_mean
    if score_eval == "labeled":
        opt_g = routing_score[: s_labeled.shape[0]]  # caller responsibility
    else:
        opt_g = routing_score
    tau_low, tau_high, opt_info = optimize_dual_thresholds(
        opt_g, alpha=alpha, beta=beta, seed=seed,
        popsize=popsize, maxiter=maxiter, tol=de_tol,
        decision_mode=decision_mode,
    )

    # 5. Build region masks on the deployment scores
    accept_mask = routing_score >= tau_high
    reject_mask = routing_score < tau_low
    uncertain_mask = (~accept_mask) & (~reject_mask)

    info: Dict[str, float] = {}
    info.update(opt_info)
    info["calibration_method"] = cal_info.method
    info["calibration_n_labeled"] = float(cal_info.n_labeled)
    info["calibration_n_pos"] = float(cal_info.n_pos)
    info["calibration_n_neg"] = float(cal_info.n_neg)
    info["calibration_fit_time"] = float(cal_info.fit_time)
    info["calibration_n_basis"] = float(cal_info.n_basis)
    info["calibration_lam"] = float(cal_info.lam)
    info["calibration_converged"] = float(cal_info.converged)
    info["routing"] = "stochastic" if use_stochastic_routing else "mean"
    info["g_mean_min"] = float(g_mean.min())
    info["g_mean_max"] = float(g_mean.max())
    info["g_mean_avg"] = float(g_mean.mean())
    info["g_se_avg"] = float(g_se.mean())

    return GamcalDecision(
        tau_low=float(tau_low),
        tau_high=float(tau_high),
        g_mean=g_mean,
        g_stochastic=g_stoch,
        g_se=g_se,
        accept_mask=accept_mask,
        reject_mask=reject_mask,
        uncertain_mask=uncertain_mask,
        info=info,
    )


__all__ = [
    "SplineGAMCalibrator",
    "CalibrationResult",
    "GamcalDecision",
    "DECISION_MODE_FORCE_LABEL",
    "DECISION_MODE_TRUST_PROXY",
    "stochastic_calibrated_score",
    "expected_f_beta",
    "optimize_dual_thresholds",
    "fit_and_optimize_dual_thresholds",
]
