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
    assert cfg.sample_budget == (3 * b) // 5
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
    npos = int(yl.sum()); nneg = len(yl) - npos
    target = lambda y: np.where(y > .5, (npos + 1) / (npos + 2), 1 / (nneg + 2))
    def loss(t):
        a, c, cl = t
        return (np.dot(w / w.mean(), stable_bce(a * xa + c, ya))
                + np.sum(stable_bce(a * xl + cl, target(yl))))
    reference = minimize(loss, [1., 0., 0.], bounds=[(0., None), (None, None), (None, None)],
                         options={"ftol": 1e-13, "gtol": 1e-8})
    fit = calibrate(x_a=xa, h_a=(xa > 0), y_a=ya, w_a=w,
                    x_l=xl, h_l=(xl > 0), y_l=yl, h_pool=np.array([0., 1.]),
                    lam=.25, rho=(9., 9., 9., 9.), paper=True)
    assert fit.status == "ok" and fit.params.c == 0.
    np.testing.assert_allclose([fit.params.a, fit.params.b_final, fit.params.b + fit.params.delta], reference.x, atol=1e-4)
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
    assert len(result.calib_idx) == 10 and len(result.verify_idx) == 0  # only ten preparation labels remain
    assert result.diagnostics['lc_n_strata'] > 2


def test_zero_positive_A_fits_slope_on_L_alone():
    xl = np.array([-3., -1., .5, 2., 4.])
    yl = np.array([0., 0., 1., 1., 1.])
    fits = [calibrate(x_a=xa, h_a=np.zeros(6), y_a=np.zeros(6), w_a=np.ones(6)*10,
                      x_l=xl, h_l=np.zeros(5), y_l=yl, h_pool=np.zeros(1),
                      lam=.25, rho=(1.,)*4, paper=True, cp_lam=.15,
                      strata=np.repeat([0, 1], 3))
            for xa in (np.linspace(-6, -2, 6), np.linspace(1, 9, 6))]
    assert all(f.status == 'ok' for f in fits)
    assert fits[0].params.a == pytest.approx(fits[1].params.a, abs=1e-9)
    for fit in fits:
        assert fit.diagnostics['cp_strata'] == 2
        assert abs(fit.diagnostics['intercept_residual']) < 1e-8


def test_calibration_floor_keeps_score_bins_when_m_is_small():
    from stratified_calibration import StratifiedDesign
    design = StratifiedDesign.build(np.arange(100), np.linspace(0, 1, 100),
                                     n_bins=10, max_strata=10, neyman=True)
    design, ids = design.draw(10, np.random.default_rng(0))
    assert len(ids) == 10
    np.testing.assert_array_equal(design.counts, np.ones(10))
    assert design.sample().weights.sum() == 100


@pytest.mark.parametrize('n,expected', [(99, 1), (266, 2), (25440, 254)])
def test_sas_default_batch_scales_with_universe(n, expected):
    from fixed_ce_sampling import sample_fixed_ce
    batches = []
    def label(ids):
        batches.append(len(ids))
        return [0] * len(ids)
    sample_fixed_ce(range(n), -np.arange(n, dtype=float), lambda _: 0, min(1000, n),
                    SamplingConfig(pilot_fraction=0), batch_oracle=label)
    assert max(batches) == expected
