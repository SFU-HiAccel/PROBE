"""Contracts at public boundaries of the current manuscript protocol."""
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit

from fixed_ce_config import runtime_options
from fixed_ce_sampling import SamplingConfig
from learned_calibration import (paper_calibration_config, OofFeatures,
                                run_learned_calibration)
from weighted_calibrator import calibrate, predict_proba, stable_bce


@pytest.mark.parametrize("b", range(21))
def test_exact_prepare_split_and_public_sampler(b):
    cfg = paper_calibration_config(b)
    assert cfg.sample_budget == (4 * b) // 5
    assert cfg.sample_budget + cfg.calib_budget == b
    assert cfg.route_unlimited and cfg.budget_scope == "preparation"
    sampler = runtime_options(environ={})["fixed_ce_config"]
    assert (sampler.protocol, sampler.pilot_mode, sampler.pilot_fraction, sampler.n_min) == (
        "paper", "uniform", .3, 2)
    assert SamplingConfig() == sampler


def test_paper_platt_matches_independent_three_parameter_loss():
    xa = np.array([-3., -1., 0., 2., 3.])
    ya = np.array([0., 0., 1., 0., 1.])
    w = np.array([4., 3., 1., 2., 1.])
    xl = np.array([-2., -.5, 1., 3.])
    yl = np.array([0., 1., 1., 1.])
    npos = int(ya.sum() + yl.sum()); nneg = len(ya) + len(yl) - npos
    target = lambda y: np.where(y > .5, (npos + 1) / (npos + 2), 1 / (nneg + 2))
    def loss(t):
        a, b, xi = t
        return (.75 * np.average(stable_bce(a * xa + b, target(ya)), weights=w)
                + .25 * np.mean(stable_bce(a * xl + b + xi, target(yl))) + .5 * xi * xi)
    reference = minimize(loss, [1., 0., 0.], bounds=[(0., None), (None, None), (None, None)])
    fit = calibrate(x_a=xa, h_a=(xa > 0), y_a=ya, w_a=w,
                    x_l=xl, h_l=(xl > 0), y_l=yl, h_pool=np.array([0., 1.]),
                    lam=.25, rho=(9., 9., 9., 9.), paper=True)
    assert fit.status == "ok" and fit.params.c == 0.
    np.testing.assert_allclose([fit.params.a, fit.params.b, fit.params.delta], reference.x, atol=1e-4)
    p = predict_proba(fit.params, xa, np.zeros_like(xa))
    np.testing.assert_allclose(p, predict_proba(fit.params, xa, np.ones_like(xa)))
    assert abs(np.dot(w, p) - np.dot(w, ya)) < 1e-8


def test_failed_paper_fit_preserves_refined_labels_and_zero_alpha_never_routes():
    n = 60; h = (np.arange(n) % 3 == 0).astype(float)
    l = np.arange(8); labels = (l % 2).astype(int)
    cfg = paper_calibration_config(10, alpha=0)
    result = run_learned_calibration(x=np.linspace(-2, 2, n), x_cal=np.linspace(2, -2, n),
        h=h, labeled_indices=l, labeled_values=labels, oof=OofFeatures(None, None, "fixture"),
        label_rows=lambda rows: np.zeros(len(rows), dtype=int), config=cfg)
    assert len(result.verify_idx) == 0 and result.params is None
    remaining = np.setdiff1d(np.arange(n), np.r_[l, result.calib_idx])
    np.testing.assert_array_equal(result.final_pos[remaining], h[remaining] > .5)
    np.testing.assert_array_equal(result.final_pos[l], labels.astype(bool))


def test_calibration_design_uses_refined_score_and_covers_each_stratum(monkeypatch):
    import learned_calibration as lc
    real = lc.StratifiedDesign.build
    seen = {}
    def build(pool, scores, **kwargs):
        seen["scores"] = scores.copy()
        return real(pool, scores, **kwargs)
    monkeypatch.setattr(lc.StratifiedDesign, "build", build)
    n = 100; x = np.linspace(-4, 4, n); xc = -x
    result = lc.run_learned_calibration(x=x, x_cal=xc, h=(xc > 0).astype(float),
        labeled_indices=np.arange(40), labeled_values=np.arange(40) % 2,
        oof=OofFeatures(None, None, "fixture"), label_rows=lambda ids: ids % 2,
        config=paper_calibration_config(50, alpha=0))
    np.testing.assert_allclose(seen["scores"], expit(xc))
    assert len(result.calib_idx) == 10 and len(result.verify_idx) == 0
