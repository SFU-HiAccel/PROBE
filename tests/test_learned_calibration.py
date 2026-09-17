"""Tests for the learned-calibration orchestrator (design3 §2-§7, §9)."""
import numpy as np
import pytest
from scipy.special import expit

from learned_calibration import (
    LearnedCalibrationConfig,
    OofFeatures,
    compute_oof_features,
    learned_calibration_total_budget,
    run_learned_calibration,
)


def _world(n=3000, seed=0, positive_shift=-1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 2.0, n)
    y_true = (rng.random(n) < expit(1.3 * x + positive_shift)).astype(np.int32)
    h = (x > 0.5).astype(float)
    labeled = np.sort(rng.choice(n, size=150, replace=False))
    return x, h, y_true, labeled


class _CountingLabeller:
    def __init__(self, y_true):
        self.y_true = y_true
        self.rows = []

    def __call__(self, rows):
        self.rows.extend(int(r) for r in rows)
        return self.y_true[rows]


class TestConfig:
    def test_budget_split(self):
        cfg = LearnedCalibrationConfig(total_budget=100)
        assert (cfg.sample_budget, cfg.calib_budget) == (60, 10)

    def test_rejects_fractions_over_one(self):
        with pytest.raises(ValueError):
            LearnedCalibrationConfig(total_budget=100, sample_frac=0.95, calib_frac=0.1)

    def test_total_budget_floor(self):
        assert learned_calibration_total_budget(50, 1.0) == 10
        assert learned_calibration_total_budget(10000, 5.0) == 500


class TestComputeOofFeatures:
    def test_held_out_labels_never_reach_their_own_fold(self):
        n = 200
        labeled = np.arange(0, 100, dtype=np.int64)
        y = np.array([0, 1] * 50, dtype=np.int32)
        seen_train = []

        def refine_fn(train_idx, train_y):
            seen_train.append(set(train_idx.tolist()))
            return np.zeros(n, dtype=bool), np.arange(n, dtype=float)

        oof = compute_oof_features(refine_fn, labeled, y, n_folds=5, seed=0)
        assert oof.reason == ""
        assert len(seen_train) == 5
        held = [set(labeled.tolist()) - tr for tr in seen_train]
        assert set().union(*held) == set(labeled.tolist())
        assert all(not (h & tr) for h, tr in zip(held, seen_train))
        assert np.allclose(oof.x, labeled.astype(float))

    def test_too_few_labels_per_class(self):
        oof = compute_oof_features(lambda a, b: (None, None), np.arange(5), np.array([1, 0, 0, 0, 0]),
                                   n_folds=5, seed=0)
        assert (oof.x, oof.reason) == (None, "too_few_labels_per_class")


class TestRunLearnedCalibration:
    def test_routes_within_budget_and_keeps_llm_labels(self):
        x, h, y_true, labeled = _world()
        labeller = _CountingLabeller(y_true)
        cfg = LearnedCalibrationConfig(total_budget=400, seed=1)
        oof = OofFeatures(x[labeled] + 0.1, h[labeled], "")
        res = run_learned_calibration(x=x, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                                      oof=oof, label_rows=labeller, config=cfg)
        assert res.status == "routed"
        assert len(labeller.rows) == len(set(labeller.rows))
        assert len(labeller.rows) + labeled.size <= cfg.total_budget
        assert res.diagnostics["lc_calls_total"] == labeled.size + len(labeller.rows)
        assert res.diagnostics["lc_budget_overrun"] == 0
        assert not set(res.calib_idx.tolist()) & set(labeled.tolist())
        assert not set(res.verify_idx.tolist()) & (set(labeled.tolist()) | set(res.calib_idx.tolist()))
        touched = np.concatenate([labeled, res.calib_idx, res.verify_idx])
        assert np.array_equal(res.final_pos[touched], y_true[touched] == 1)
        untouched = np.setdiff1d(np.arange(x.size), touched)
        assert np.array_equal(res.final_pos[untouched], h[untouched] > 0.5)

    def test_routing_improves_f1_over_refined_labels(self):
        x, h, y_true, labeled = _world(seed=2)
        cfg = LearnedCalibrationConfig(total_budget=600, seed=3)
        res = run_learned_calibration(x=x, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                                      oof=OofFeatures(None, None, "skipped"),
                                      label_rows=_CountingLabeller(y_true), config=cfg)

        def f1(pred):
            tp = np.sum(pred & (y_true == 1))
            return 2 * tp / (np.sum(pred) + np.sum(y_true == 1))

        assert res.diagnostics["lc_lambda_used"] == 0.0
        assert f1(res.final_pos) > f1(h > 0.5)

    def test_all_negative_world_is_insufficient_but_spends_budget_uniformly(self):
        x, h, _, labeled = _world(seed=4)
        y_true = np.zeros(x.size, dtype=np.int32)
        labeller = _CountingLabeller(y_true)
        cfg = LearnedCalibrationConfig(total_budget=300, seed=5)
        res = run_learned_calibration(x=x, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                                      oof=OofFeatures(None, None, "skipped"), label_rows=labeller, config=cfg)
        assert res.status == "insufficient_evidence"
        assert res.diagnostics["lc_reason"] == "single_class_in_A"
        assert labeled.size + len(labeller.rows) == cfg.total_budget
        assert np.isnan(res.diagnostics["lc_estimated_f1"])

    def test_missing_logit_skips_calibration(self):
        x, h, y_true, labeled = _world(seed=6)
        labeller = _CountingLabeller(y_true)
        cfg = LearnedCalibrationConfig(total_budget=200, seed=7)
        res = run_learned_calibration(x=None, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                                      oof=OofFeatures(None, None, "skipped"), label_rows=labeller, config=cfg)
        assert (res.status, res.diagnostics["lc_reason"]) == ("insufficient_evidence", "proxy_logit_unavailable")
        assert res.calib_idx.size == 0 and res.verify_idx.size == 50

    def test_active_sample_over_budget_makes_no_further_calls(self):
        x, h, y_true, labeled = _world(seed=8)
        labeller = _CountingLabeller(y_true)
        cfg = LearnedCalibrationConfig(total_budget=100, seed=9)
        res = run_learned_calibration(x=x, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                                      oof=OofFeatures(None, None, "skipped"), label_rows=labeller, config=cfg)
        assert labeller.rows == []
        assert res.diagnostics["lc_budget_overrun"] == labeled.size - 100


def test_routing_orders_rows_by_raw_logit_not_saturated_probability(monkeypatch):
    import learned_calibration as lc

    x, h, y_true, labeled = _world(seed=10)
    x = x * 30.0  # push many logits past float64 sigmoid saturation
    captured = {}
    real_plan = lc.plan_dual_threshold

    def spy(row_idx, p, s, h_arg, counts, budget):
        captured["rows"], captured["key"] = row_idx, s
        return real_plan(row_idx, p, s, h_arg, counts, budget)

    monkeypatch.setattr(lc, "plan_dual_threshold", spy)
    res = run_learned_calibration(x=x, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                                  oof=OofFeatures(None, None, "skipped"),
                                  label_rows=_CountingLabeller(y_true),
                                  config=LearnedCalibrationConfig(total_budget=400, seed=11))
    assert res.status == "routed"
    assert np.array_equal(captured["key"], x[captured["rows"]])
    tau = res.diagnostics["lc_tau_plus"]
    assert np.isnan(tau) or 0.0 <= tau <= 1.0
