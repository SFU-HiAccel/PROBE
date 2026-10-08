"""The w/o-calibration ablation arm (calibrator="none") and the calibration audit helpers."""
import asyncio

import numpy as np
import pytest
from scipy.special import expit

from calibration_audit import audit_labels, stratified_audit_sample, weighted_brier, weighted_ece
from learned_calibration import LearnedCalibrationConfig, OofFeatures, run_learned_calibration
from weighted_calibrator import predict_proba


def _world(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 2.0, n)
    # Miscalibrated on purpose: the true log-odds are 1.3 x - 1, not x.
    y_true = (rng.random(n) < expit(1.3 * x - 1.0)).astype(np.int32)
    h = (x > 0.5).astype(float)
    labeled = np.sort(rng.choice(n, size=150, replace=False))
    return x, h, y_true, labeled


def _run(calibrator, seed=0, alpha=0.5, y_override=None):
    x, h, y_true, labeled = _world(seed=seed)
    y = y_true if y_override is None else y_override
    calls = []

    def label_rows(rows):
        calls.extend(int(r) for r in rows)
        return y[rows]

    cfg = LearnedCalibrationConfig(total_budget=400, seed=1, calibrator=calibrator, alpha=alpha,
                                   route_unlimited=True)
    res = run_learned_calibration(x=x, h=h, labeled_indices=labeled, labeled_values=y[labeled],
                                  oof=OofFeatures(None, None, "skipped"), label_rows=label_rows, config=cfg)
    return res, x, h, calls


class TestUncalibratedArm:
    def test_routes_on_the_raw_proxy_probability(self):
        res, x, h, _ = _run("none")
        assert res.status == "routed"
        assert res.diagnostics["lc_reason"] == "uncalibrated"
        assert res.diagnostics["lc_calibrator"] == "none"
        p = predict_proba(res.params, x, h)
        assert np.allclose(p, expit(x))

    def test_spends_the_same_calibration_sample_as_platt(self):
        cal, *_ = _run("platt")
        raw, *_ = _run("none")
        assert raw.diagnostics["lc_calls_A"] == cal.diagnostics["lc_calls_A"]
        assert np.array_equal(np.sort(raw.calib_idx), np.sort(cal.calib_idx))

    def test_single_class_sample_does_not_trigger_fallback_rounds(self):
        x, *_ = _world()
        res, *_ = _run("none", y_override=np.zeros(x.size, dtype=np.int32))
        assert res.status == "routed"
        assert res.diagnostics["lc_fallback_rounds"] == 0

    def test_alpha_zero_matches_platt(self):
        cal, *_ = _run("platt", alpha=0.0)
        raw, *_ = _run("none", alpha=0.0)
        assert raw.verify_idx.size == cal.verify_idx.size == 0
        assert np.array_equal(raw.final_pos, cal.final_pos)

    def test_platt_result_exposes_its_params(self):
        res, *_ = _run("platt")
        assert res.params is not None and hasattr(res.params, "a")

    def test_unknown_calibrator_is_rejected(self):
        with pytest.raises(ValueError):
            LearnedCalibrationConfig(total_budget=100, calibrator="isotonic")


class TestAuditSample:
    def test_small_pool_is_labelled_in_full(self):
        pool = np.arange(50, 150)
        idx, w = stratified_audit_sample(pool, np.linspace(-3, 3, 200), 500, n_bins=10, seed=0)
        assert np.array_equal(np.sort(idx), pool)
        assert np.allclose(w, 1.0)

    def test_large_pool_is_stratified_and_weighted(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=20000)
        pool = np.arange(20000)
        idx, w = stratified_audit_sample(pool, x, 1000, n_bins=20, seed=0)
        assert idx.size == 1000 and np.unique(idx).size == 1000
        assert np.isin(idx, pool).all()
        assert w.sum() == pytest.approx(pool.size)
        # equal allocation: every score stratum is represented, including the top one
        top = x >= np.quantile(x, 0.95)
        assert np.isin(idx, np.flatnonzero(top)).sum() >= 40


class TestMetrics:
    def test_ece_is_zero_for_perfectly_calibrated_bins(self):
        p = np.repeat([0.1, 0.5, 0.9], 10)
        y = np.concatenate([[1] + [0] * 9, [1] * 5 + [0] * 5, [1] * 9 + [0]])
        assert weighted_ece(p, y, np.ones_like(p), n_bins=10) == pytest.approx(0.0)

    def test_ece_and_brier_of_a_constant_forecast(self):
        y = np.array([1, 0, 0, 0])
        p = np.full(4, 0.5)
        assert weighted_ece(p, y, np.ones(4), n_bins=10) == pytest.approx(0.25)
        assert weighted_brier(p, y, np.ones(4)) == pytest.approx(0.25)


class TestAuditLabels:
    def test_uses_a_fresh_ledger_and_the_session_request(self):
        class _Request:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def __call__(self, key):
                return int(key % 2), {}

        class _Session:
            request_factory = _Request
            concurrency = 4

        labels = audit_labels(_Session(), np.array([3, 4, 5]))
        assert labels.tolist() == [1, 0, 1]
        asyncio.get_event_loop_policy()  # no loop left running


class TestRunAudit:
    def test_writes_an_npz_without_touching_the_result(self, tmp_path):
        from calibration_audit import run_audit

        res, x, h, _ = _run("platt")
        _, _, y_true, labeled = _world()

        class _Request:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def __call__(self, key):
                return int(y_true[key]), {}

        class _Session:
            request_factory = _Request

        before = res.final_pos.copy()
        diag = run_audit(result=res, x=x, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                         session=_Session(), config=LearnedCalibrationConfig(total_budget=400, seed=1),
                         out_dir=str(tmp_path), n=500)
        assert np.array_equal(res.final_pos, before)
        z = np.load(diag["lc_audit_path"])
        assert diag["lc_audit_n"] == 500 == z["audit_idx"].size
        touched = np.concatenate([labeled, res.calib_idx, res.verify_idx])
        assert not np.isin(z["audit_idx"], touched).any()
        assert np.array_equal(z["audit_y"], y_true[z["audit_idx"]])
        assert np.allclose(z["p_cal"], predict_proba(res.params, x, h))
        assert str(z["meta_calibrator"]) == "platt"
