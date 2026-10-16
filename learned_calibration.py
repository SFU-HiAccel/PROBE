"""Learned calibration: stratified sample A + active sample L -> Platt -> dual thresholds.

Entry points used by ``cluster_proxy.cluster_proxy_pipeline`` when
``decision_mode == "learned_calibration"`` (design3.md, design3_algorithms.tex):

* ``compute_oof_features``  out-of-fold (logit, refined label) for L
* ``run_learned_calibration``  stages 2-5 of the algorithm

Both take their proxy/propagation and LLM access as callables, so this module
has no dependency on cluster_proxy / proxy_model and is testable offline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from budgeted_calibrated_router import apply_verified_labels, expected_counts, plan_dual_threshold
from router_cap import MIN_TOTAL_BUDGET
from stratified_calibration import StratifiedDesign
from weighted_calibrator import CalibrationFit, PlattParams, calibrate, predict_proba

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
CALIBRATORS = ("platt", "gam", "gam_univariate", "none")
# What the calibrator reads as its score input: the LR proxy logit u (paper Sec. 5) or the logit of the
# graph-refined score z_smooth (the output of bias diffusion + homogeneity propagation).
CALIB_INPUTS = ("proxy", "refined")
REFINED_EPS = 1e-6   # z_smooth is pinned to exactly 0/1 on L; clip before the logit
# Identity map in log-odds: predict_proba(IDENTITY, x, h) == sigmoid(x).
IDENTITY_PARAMS = PlattParams(a=1.0, b=0.0, c=0.0, delta=0.0, t=0.0)

# profiling keys copied into run.py result rows (only for this mode)
LEARNED_CALIBRATION_RESULT_KEYS = (
    "lc_status", "lc_reason", "lc_total_budget", "lc_sample_budget", "lc_calib_budget",
    "lc_calls_L", "lc_calls_A", "lc_calls_D", "lc_calls_total", "lc_budget_overrun",
    "lc_n_strata", "lc_A_pos", "lc_L_pos", "lc_ess_pos", "lc_max_pos_weight_share",
    "lc_lambda_used", "lc_oof_reason", "lc_n_folds_used", "lc_a", "lc_b", "lc_c", "lc_delta", "lc_t", "lc_b_final",
    "lc_intercept_residual", "lc_k_plus", "lc_k_minus", "lc_tau_plus", "lc_tau_minus",
    "lc_planned_f1", "lc_estimated_f1", "lc_D_pos", "lc_time_s", "lc_fallback_rounds",
    "lc_route_budget", "lc_fallback_budget", "lc_fallback_used",
    "lc_cp_used", "lc_cp_target_pos", "lc_cp_measured_pos", "lc_cp_upper_pos", "lc_cp_strata",
    "lc_alpha", "lc_plan_calls", "lc_alloc",
    "lc_time_sample_s", "lc_time_fit_s", "lc_time_route_s",
    "lc_calibrator", "lc_route_objective", "lc_route_cost_exp", "lc_f_gain", "lc_f_curve",
    "lc_gam_se_median", "lc_gam_monotone", "lc_gam_n_knots", "lc_gam_ridge",
    "lc_upper_missed", "lc_upper_pool", "lc_bound_confidence", "lc_bound_scope",
    "lc_calibration_design", "lc_calib_input", "lc_h_pool_values", "lc_h_a_values", "lc_h_l_values", "lc_audit_path", "lc_audit_n", "lc_audit_time_s",
)

# refine_fn(train_idx, train_y) -> (refined_label_mask[N], lr_logit[N] or None)
RefineFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]]
# label_rows(row_ids) -> 0/1 LLM labels aligned with row_ids
LabelRowsFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LearnedCalibrationConfig:
    total_budget: int
    budget_scope: str = "all"  # preparation: L+A only; router may use all remaining IDs
    sample_frac: float = 0.5
    calib_frac: float = 0.4
    route_frac: float = 0.1
    n_bins: int = 10
    n_folds: int = 5
    lam: float = 0.25
    rho_a: float = 0.25
    rho_b: float = 0.25
    rho_c: float = 0.25
    rho_delta: float = 1.0
    alpha: float = 1.0           # 1: spend the whole routing budget; <1: trade quality against calls
    neyman: bool = True          # allocate A by Neyman on a floor instead of evenly across strata
    fallback_unlimited: bool = False  # let the stratified fallback draw on the whole remaining budget
    route_unlimited: bool = False     # routing is not capped by a share: alpha alone decides how far it goes
    route_budget_mult: float = 0.0    # >0: routing may spend up to mult x the nominal budget, charged on top
    cp_lam: float = 0.0          # >0: mix a Clopper-Pearson upper bound in when A has no positive
    cp_alpha: float = 0.05       # failure probability of that bound, split across strata
    # --- calibrator arm: "platt" (a line in log-odds) or GAMCAL's spline (arXiv:2604.00660) ---
    # "none" is the w/o-calibration ablation: A is still drawn (same calls), but routing uses the raw
    # proxy probability sigmoid(x) as p instead of a fitted calibrator.
    calibrator: str = "platt"    # "platt" | "gam" | "gam_univariate" (the last drops h and 1[L]) | "none"
    calib_input: str = "proxy"   # "proxy": LR logit u; "refined": logit of the graph-refined score (x_cal)
    gam_ridge: float = 1.0       # GAMCAL's smoothing lambda; the paper takes it as a fixed input
    gam_knots: int = 8           # knot count -- unspecified by the paper, so ours to declare
    gam_stochastic: bool = False # route on Eq. 26's g~(s,q) instead of the posterior mean
    # --- routing objective: see plan_dual_threshold for why cost_exp is what makes alpha resolve ---
    route_objective: str = "pool"   # "pool" | "budget" | "normalized" | "gamcal"
    route_cost_exp: float = 1.0     # gamma on the cost term; >1 turns alpha from a switch into a dial
    seed: int = 42
    fixed_calibration: bool = False  # legacy callers opt in; fixed-CE production enforces True
    protocol: str = "legacy"  # paper entry points select the current manuscript explicitly
    certificate_delta: float = .05

    def __post_init__(self) -> None:
        if self.protocol not in ("paper", "legacy"):
            raise ValueError("protocol must be paper or legacy")
        if self.calib_input not in CALIB_INPUTS:
            raise ValueError(f"calib_input must be one of {CALIB_INPUTS}, got {self.calib_input!r}")
        if self.calibrator not in CALIBRATORS:
            raise ValueError(f"calibrator must be one of {CALIBRATORS}, got {self.calibrator!r}")
        if self.budget_scope not in ("all", "preparation"):
            raise ValueError("budget_scope must be all or preparation")
        if self.budget_scope == "preparation" and (not self.route_unlimited or self.route_budget_mult > 0):
            raise ValueError("preparation scope requires unlimited routing without a multiplier")
        if self.total_budget < 0:
            raise ValueError(f"total_budget must be >= 0, got {self.total_budget}")
        if not 0 < self.certificate_delta < 1:
            raise ValueError("certificate_delta must be in (0,1)")
        if not 0.0 < self.sample_frac <= 1.0:
            raise ValueError(f"sample_frac must be in (0, 1], got {self.sample_frac}")
        if not 0.0 <= self.calib_frac <= 1.0 or self.sample_frac + self.calib_frac > 1.0:
            raise ValueError("calib_frac must be in [0, 1] and sample_frac + calib_frac <= 1")
        if not 0.0 <= self.route_frac <= 1.0 or self.sample_frac + self.calib_frac + self.route_frac > 1.0 + 1e-9:
            raise ValueError("route_frac must be in [0, 1] and the three shares must sum to <= 1")
        if self.n_bins < 1 or self.n_folds < 2:
            raise ValueError("n_bins must be >= 1 and n_folds >= 2")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {self.lam}")
        if min(self.rho_a, self.rho_b, self.rho_c, self.rho_delta) < 0.0:
            raise ValueError("ridge weights must be >= 0")

    @property
    def sample_budget(self) -> int:
        if self.protocol == "paper":
            return (3 * self.total_budget) // 5
        return min(self.total_budget, max(1, int(self.sample_frac * self.total_budget)))

    @property
    def calib_budget(self) -> int:
        if self.protocol == "paper":
            return self.total_budget - self.sample_budget
        return int(self.calib_frac * self.total_budget)

    @property
    def route_budget(self) -> int:
        """Share reserved for routing D. The stratified fallback also draws from it, so a run that
        keeps failing to calibrate spends at most this much beyond L + A instead of eating the
        whole remaining budget."""
        return int(self.route_frac * self.total_budget)

    @property
    def rho(self) -> Tuple[float, float, float, float]:
        return (self.rho_a, self.rho_b, self.rho_c, self.rho_delta)


@dataclass(frozen=True)
class OofFeatures:
    x: Optional[np.ndarray]  # out-of-fold logits aligned with labeled_indices
    h: Optional[np.ndarray]  # out-of-fold refined labels (0.0 / 1.0)
    reason: str              # "" when available
    n_folds_used: int = 0    # folds actually run: min(R, rarer class of L); 0 when unavailable


@dataclass(frozen=True)
class LearnedCalibrationResult:
    final_pos: np.ndarray    # bool[N]
    calib_idx: np.ndarray    # rows in A (LLM-labelled)
    verify_idx: np.ndarray   # rows in D (LLM-labelled by routing or exploration)
    status: str              # "routed" | "insufficient_evidence"
    diagnostics: Dict[str, Any]
    params: Any = None       # fitted calibrator (PlattParams / GamParams), None when not routed


def refined_logit(z: np.ndarray) -> np.ndarray:
    """Logit of a graph-refined score in [0, 1], clipped so rows pinned to 0/1 stay finite."""
    z = np.clip(np.asarray(z, dtype=np.float64), REFINED_EPS, 1.0 - REFINED_EPS)
    return np.log(z / (1.0 - z))


def learned_calibration_total_budget(n_items: int, budget_pct: float) -> int:
    """Preparation labels B = floor(N * budget_pct / 100), including tiny queries."""
    if n_items < 0 or not np.isfinite(budget_pct) or not 0 <= budget_pct <= 100:
        raise ValueError("invalid candidate count or percentage")
    return int(n_items * budget_pct / 100)


def paper_calibration_config(total_budget: int, **overrides) -> LearnedCalibrationConfig:
    """Current paper protocol; historical configurations remain explicitly constructible."""
    settings = dict(total_budget=total_budget, protocol="paper", budget_scope="preparation",
                    sample_frac=.6, calib_frac=.4, route_frac=0., route_unlimited=True,
                    fixed_calibration=True, calib_input="refined", alpha=.5, cp_lam=.15,
                    rho_a=0., rho_b=0., rho_c=0., rho_delta=0., seed=0)
    settings.update(overrides)
    return LearnedCalibrationConfig(**settings)


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
    # A class rarer than R caps the folds; report the value that ran, not the one requested.
    x_oof = np.zeros(idx.size, dtype=np.float64)
    h_oof = np.zeros(idx.size, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    for train_pos, hold_pos in splitter.split(idx, y):
        mask, logit = refine_fn(idx[train_pos], y[train_pos])
        if logit is None:
            return OofFeatures(None, None, "fold_logit_unavailable", k)
        x_oof[hold_pos] = np.asarray(logit, dtype=np.float64)[idx[hold_pos]]
        h_oof[hold_pos] = np.asarray(mask, dtype=bool)[idx[hold_pos]].astype(np.float64)
    return OofFeatures(x_oof, h_oof, "", k)


def _label(label_rows: LabelRowsFn, rows: np.ndarray, stage="A") -> np.ndarray:
    if rows.size == 0:
        return np.zeros(0, dtype=np.int32)
    from oracle_budget import binary_labels_or_raise
    callback = getattr(label_rows, "for_stage", None)
    values = callback(rows, stage) if callback is not None else label_rows(rows)
    labels = binary_labels_or_raise(values).reshape(-1)
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
    x_cal: Optional[np.ndarray] = None,
) -> LearnedCalibrationResult:
    """Stages 2-5: sample A, calibrate on A+L, choose dual thresholds, verify D.

    Paper A is stratified on refined scores. Legacy A uses the LR proxy score. The calibrator reads ``x_cal``
    when given (e.g. the logit of the graph-refined score, calib_input="refined") and ``x`` otherwise;
    ``oof.x`` must be on the same scale as the calibrator input.
    """
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
        "lc_n_folds_used": int(oof.n_folds_used),
    }

    # Stages 2-3: stratified calibration sample A on the frozen proxy score s. The first draw is
    # calib_budget; if the calibrator still lacks evidence, further rounds (the fallback) come out of
    # the routing share, capped at route_budget in total. Budget the fallback does not use stays with
    # routing, and a run that never calibrates stops after L + A + route_budget instead of consuming B.
    pool = np.flatnonzero(~verified)
    fit = None
    reason = "proxy_logit_unavailable" if x is None else "empty_calibration_sample"
    if x is not None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        s = expit(x_cal) if config.protocol == "paper" and x_cal is not None else expit(x)
    # calibrator input: fit, predicted probabilities and the routing order all use xc
    xc = x if x_cal is None or x is None else np.asarray(x_cal, dtype=np.float64).reshape(-1)
    use_l = oof.x is not None
    design = None
    draws = 0
    # Capped at D's share by default; unlimited lets it keep drawing while any budget is left, which is
    # the behaviour the 5-seed study had, and makes the fallback's real appetite visible.
    fallback_left = max(0, remaining - config.calib_budget) if config.fallback_unlimited \
        else max(0, min(config.route_budget, remaining - config.calib_budget))
    if config.fixed_calibration:
        fallback_left = 0  # one fixed A draw, irrespective of its observed labels
    diag["lc_fallback_budget"] = int(fallback_left)
    t_sample = t_fit = t_route = 0.0
    m = max(0, min(config.calib_budget, pool.size, remaining))
    if m > 0:
        # Without a proxy logit there is nothing to stratify on: one stratum (uniform rounds).
        scores = s if x is not None else np.zeros(n, dtype=np.float64)
        design = StratifiedDesign.build(pool, scores, n_bins=config.n_bins, max_strata=m,
                                        neyman=config.neyman and x is not None)
    while design is not None and m > 0:
        design, new_rows = design.draw(m, rng)
        _t0 = time.perf_counter()
        new_y = _label(label_rows, new_rows)
        t_sample += time.perf_counter() - _t0
        draws += 1
        remaining -= new_rows.size
        verified[new_rows] = True
        y_known[new_rows] = new_y
        if x is not None and not (config.protocol == "paper" and np.all(verified)):
            _t0 = time.perf_counter()
            sample = design.sample()
            common = dict(
                x_a=xc[sample.indices], h_a=h[sample.indices], y_a=y_known[sample.indices],
                w_a=sample.weights,
                x_l=oof.x if use_l else None, h_l=oof.h if use_l else None, y_l=l_y if use_l else None,
                h_pool=h[pool], lam=config.lam, cp_lam=config.cp_lam, cp_alpha=config.cp_alpha,
            )
            if config.calibrator == "none":
                fit = CalibrationFit(IDENTITY_PARAMS, "ok", "uncalibrated", {})
            elif str(config.calibrator).startswith("gam"):
                from gam_calibrator import calibrate_gam
                fit = calibrate_gam(**common, ridge=config.gam_ridge, n_knots=config.gam_knots,
                                    univariate=config.calibrator == "gam_univariate")
            else:
                fit = calibrate(**common, rho=config.rho, paper=config.protocol == "paper",
                                strata=np.repeat(np.arange(len(sample.sample_counts)), sample.sample_counts))
            reason = fit.reason
            t_fit += time.perf_counter() - _t0
            if fit.status != "insufficient_evidence":
                break  # "ok" -> route; "solver_failed" -> stop retrying
        # Fallback rounds are paid for by the routing share, not by whatever budget is left.
        m = max(0, min(config.calib_budget, fallback_left, design.remaining, remaining))
        fallback_left -= m

    a_idx = design.sample().indices if design is not None else np.zeros(0, dtype=np.int64)
    diag["lc_calls_A"] = int(a_idx.size)
    diag["lc_fallback_used"] = int(diag["lc_fallback_budget"] - max(0, fallback_left))
    diag["lc_fallback_rounds"] = max(0, draws - 1)
    if x is not None:
        diag["lc_n_strata"] = 0 if design is None else len(design.strata)
        diag["lc_alloc"] = "neyman" if (design is not None and design.p_hat is not None) else "equal"
        diag["lc_A_pos"] = int(y_known[a_idx].sum())
    if fit is not None:
        for key in ("cp_used", "cp_target_pos", "cp_measured_pos", "cp_upper_pos", "cp_strata",
                    "h_pool_values", "h_a_values", "h_l_values"):
            if key in fit.diagnostics:
                diag[f"lc_{key}"] = fit.diagnostics[key]
        diag["lc_ess_pos"] = fit.diagnostics.get("ess_pos")
        diag["lc_max_pos_weight_share"] = fit.diagnostics.get("max_pos_weight_share")
        diag["lc_lambda_used"] = fit.diagnostics.get("lambda_used")

    unverified = np.flatnonzero(~verified)
    # Routing spends what is left of its own share; the fallback may already have taken part of it.
    if config.route_budget_mult > 0.0:
        # Routing is charged on top of the sampling+calibration budget but bounded: D <= mult * B.
        budget_left = int(min(unverified.size, int(config.route_budget_mult * config.total_budget)))
    elif config.route_unlimited:
        # No cap at all: every unverified row is eligible and the alpha objective decides where to stop.
        # With alpha = 1 this verifies the whole pool, i.e. it degenerates to calling the LLM on everything.
        budget_left = int(unverified.size)
    elif config.fallback_unlimited:
        budget_left = max(0, remaining)
    else:
        budget_left = max(0, min(remaining, config.route_budget - diag["lc_fallback_used"]))
    if config.fixed_calibration and config.budget_scope == "all":
        budget_left = min(budget_left, max(0, remaining))
    diag["lc_route_budget"] = int(budget_left)
    if fit is not None and fit.status == "ok":
        # Stages 4-5: dual thresholds on calibrated probabilities, one allocation
        params = fit.params
        if hasattr(params, "predict"):          # GAM arm
            if config.gam_stochastic:
                # Eq. 26: q is drawn once per row and frozen, so routing stays consistent across
                # refits (paper §5.3). Rows in poorly calibrated regions get more dispersed scores.
                q = np.random.default_rng(int(config.seed) + 1).random(n)
                p = params.predict_stochastic(xc[unverified], h[unverified], q[unverified])
            else:
                p = params.predict(xc[unverified], h[unverified])
        else:
            p = predict_proba(params, xc[unverified], h[unverified])
        counts = expected_counts(p, h[unverified], n_verified_pos=int(y_known[verified].sum()))
        # Order by the raw logit: p is monotone in x within each h group, and x does
        # not saturate the way sigmoid(x) does for |x| > ~37.
        routing_axis = p if hasattr(params, 'predict') else xc[unverified]
        plan = plan_dual_threshold(unverified, p, routing_axis, h[unverified], counts, budget_left,
                                   alpha=config.alpha, n_pool=n,
                                   objective=config.route_objective, cost_exp=config.route_cost_exp)
        d_idx = plan.verify_idx
        if fit.diagnostics.get("cp_used"):
            # Level anchored by the Clopper-Pearson bound rather than an observed positive in A.
            status = "routed_conservative"
        else:
            status = "routed" if draws <= 1 else "routed_after_fallback"
        diag.update({
            "lc_t": params.t,
            "lc_intercept_residual": fit.diagnostics.get("intercept_residual"),
            "lc_k_plus": plan.k_plus, "lc_k_minus": plan.k_minus,
            "lc_tau_plus": float(expit(plan.tau_plus)), "lc_tau_minus": float(expit(plan.tau_minus)),
            "lc_planned_f1": plan.f_tilde, "lc_alpha": plan.alpha, "lc_plan_calls": plan.n_calls,
            "lc_calibrator": str(config.calibrator),
            "lc_route_objective": str(plan.objective), "lc_route_cost_exp": float(config.route_cost_exp),
            # f(r) is what decides whether alpha can resolve at all: a near-linear curve has its
            # optimum at an endpoint under any linear scalarisation. Recording the gain and the
            # curve makes a degenerate dial diagnosable from the output instead of by re-deriving it.
            "lc_f_gain": plan.f_gain,
            "lc_f_curve": ";".join(f"{v:.6f}" for v in plan.f_curve),
        })
        if hasattr(params, "theta"):            # GAM arm
            diag.update({"lc_gam_se_median": fit.diagnostics.get("gam_se_median"),
                         "lc_gam_monotone": fit.diagnostics.get("gam_monotone"),
                         "lc_gam_n_knots": fit.diagnostics.get("gam_n_knots"),
                         "lc_gam_ridge": fit.diagnostics.get("gam_ridge")})
        else:                                   # Platt arm
            diag.update({"lc_a": params.a, "lc_b": params.b, "lc_c": params.c,
                         "lc_delta": params.delta, "lc_b_final": params.b_final})
    else:
        # insufficient evidence after all rounds: keep refusing, report no quality estimate
        d_idx = np.zeros(0, dtype=np.int64)
        status = "insufficient_evidence"
        plan = None

    _t0 = time.perf_counter()
    d_y = _label(label_rows, d_idx, "D")
    t_route = time.perf_counter() - _t0
    final_pos = (np.zeros(n, dtype=bool) if config.protocol == "legacy" and
                 config.fixed_calibration and plan is None else h > 0.5)
    final_pos[verified] = y_known[verified] == 1
    final_pos[d_idx] = d_y == 1

    if plan is not None:
        in_d = np.isin(unverified, d_idx)
        after = apply_verified_labels(counts, p[in_d], h[unverified][in_d], d_y)
        diag["lc_estimated_f1"] = after.f1()
    else:
        diag["lc_estimated_f1"] = float("nan")

    if config.fixed_calibration:
        from calibration_bounds import missed_positive_bound
        upper_missed, upper_pool = missed_positive_bound(
            pool, design, y_known, d_idx, d_y, final_pos, config.certificate_delta)
        diag.update(lc_upper_missed=upper_missed, lc_upper_pool=upper_pool,
                    lc_bound_confidence=1-config.certificate_delta,
                    lc_bound_scope="stationary_oracle_labels_fixed_A_design",
                    lc_calibration_design="fixed_single_draw")
    else:
        diag["lc_bound_scope"] = "no_fixed_time_certificate_for_legacy_adaptive_stopping"

    calls_total = int(l_idx.size + a_idx.size + d_idx.size)
    diag.update({
        "lc_status": status,
        "lc_reason": reason,
        "lc_calls_D": int(d_idx.size),
        "lc_D_pos": int(d_y.sum()),
        "lc_calls_total": calls_total,
        # Stage timers for the latency breakdown: labelling A, fitting the calibrator, labelling D.
        "lc_time_sample_s": t_sample,
        "lc_time_fit_s": t_fit,
        "lc_time_route_s": t_route,
        "lc_budget_scope": config.budget_scope,
        "lc_calib_input": config.calib_input,
        "lc_budget_overrun": max(0, (int(l_idx.size+a_idx.size) if config.budget_scope == "preparation" else calls_total) - int(config.total_budget)),
    })
    return LearnedCalibrationResult(final_pos, a_idx, d_idx, status, diag,
                                    fit.params if plan is not None else None)
