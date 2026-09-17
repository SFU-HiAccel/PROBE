"""Tests for the weighted Platt calibrator (design3 §5, §9)."""
import numpy as np
import pytest
from scipy.special import expit

from weighted_calibrator import (
    PlattParams,
    calibrate,
    fit_intercept_shift,
    positive_weight_diagnostics,
    predict_proba,
    stable_bce,
)

NO_RIDGE = (1e-8, 1e-8, 1e-8, 1e-8)


def _world(n, seed, a=1.5, b=-1.0, c=0.8):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 2.0, n)
    h = (x > 0).astype(float)
    y = (rng.random(n) < expit(a * x + b + c * h)).astype(float)
    return x, h, y


class TestStableBce:
    def test_matches_naive_formula_for_moderate_logits(self):
        z = np.array([-3.0, 0.0, 2.5])
        y = np.array([0.0, 1.0, 1.0])
        naive = -(y * np.log(expit(z)) + (1 - y) * np.log(1 - expit(z)))
        assert np.allclose(stable_bce(z, y), naive)

    def test_finite_for_extreme_logits(self):
        assert np.all(np.isfinite(stable_bce(np.array([-800.0, 800.0]), np.array([1.0, 0.0]))))


class TestInterceptShift:
    def test_matches_weighted_positive_rate(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=200)
        y = (rng.random(200) < 0.2).astype(float)
        w = rng.uniform(1.0, 5.0, 200)
        t = fit_intercept_shift(z, y, w)
        assert np.sum(w * expit(z + t)) == pytest.approx(np.sum(w * y), abs=1e-8)

    def test_rejects_single_class(self):
        with pytest.raises(ValueError):
            fit_intercept_shift(np.zeros(5), np.zeros(5), np.ones(5))


class TestCalibrate:
    def test_recovers_parameters_from_a_only(self):
        x, h, y = _world(20000, seed=1)
        fit = calibrate(x_a=x, h_a=h, y_a=y, w_a=np.ones_like(x), x_l=None, h_l=None, y_l=None,
                        h_pool=h, lam=0.25, rho=NO_RIDGE)
        assert fit.status == "ok"
        assert fit.diagnostics["lambda_used"] == 0.0
        assert fit.params.a == pytest.approx(1.5, abs=0.15)
        assert fit.params.b_final == pytest.approx(-1.0, abs=0.15)
        assert fit.params.c == pytest.approx(0.8, abs=0.25)

    def test_slope_is_never_negative(self):
        x, h, y = _world(3000, seed=2, a=-1.0, c=0.0)
        fit = calibrate(x_a=x, h_a=h, y_a=y, w_a=np.ones_like(x), x_l=None, h_l=None, y_l=None,
                        h_pool=h, lam=0.0, rho=NO_RIDGE)
        assert fit.status == "ok"
        assert fit.params.a >= 0.0

    def test_intercept_shift_removes_enriched_l_offset_on_a(self):
        x_a, h_a, y_a = _world(1500, seed=3)
        x_l, h_l, _ = _world(400, seed=4)
        y_l = (np.random.default_rng(5).random(400) < expit(1.5 * x_l + 1.5 + 0.8 * h_l)).astype(float)
        w_a = np.full(x_a.size, 7.0)
        fit = calibrate(x_a=x_a, h_a=h_a, y_a=y_a, w_a=w_a, x_l=x_l, h_l=h_l, y_l=y_l,
                        h_pool=h_a, lam=0.25, rho=(0.25, 0.25, 0.25, 1.0))
        assert fit.status == "ok"
        p = predict_proba(fit.params, x_a, h_a)
        assert np.sum(w_a * p) == pytest.approx(np.sum(w_a * y_a), abs=1e-6)
        assert abs(fit.diagnostics["intercept_residual"]) < 1e-6

    @pytest.mark.parametrize("labels,reason", [
        (np.zeros(50), "single_class_in_A"),
        (np.ones(50), "single_class_in_A"),
    ])
    def test_single_class_a_is_insufficient(self, labels, reason):
        x, h, _ = _world(50, seed=6)
        fit = calibrate(x_a=x, h_a=h, y_a=labels, w_a=np.ones(50), x_l=None, h_l=None, y_l=None,
                        h_pool=h, lam=0.25, rho=NO_RIDGE)
        assert (fit.status, fit.reason, fit.params) == ("insufficient_evidence", reason, None)

    def test_missing_refined_label_value_is_insufficient(self):
        x, _, y = _world(50, seed=7)
        fit = calibrate(x_a=x, h_a=np.zeros(50), y_a=y, w_a=np.ones(50), x_l=None, h_l=None, y_l=None,
                        h_pool=np.array([0.0, 1.0]), lam=0.25, rho=NO_RIDGE)
        assert fit.reason == "h_value_missing_in_A"


class TestPositiveWeightDiagnostics:
    def test_values(self):
        d = positive_weight_diagnostics(np.array([1, 0, 1]), np.array([2.0, 5.0, 2.0]))
        assert d == {"k_pos": 2, "ess_pos": pytest.approx(2.0), "max_pos_weight_share": pytest.approx(0.5)}

    def test_no_positives(self):
        d = positive_weight_diagnostics(np.zeros(3), np.ones(3))
        assert d["k_pos"] == 0 and np.isnan(d["ess_pos"])


def test_predict_proba_uses_b_final_not_delta():
    params = PlattParams(a=1.0, b=0.5, c=0.0, delta=9.0, t=-0.5)
    assert predict_proba(params, np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.5)
