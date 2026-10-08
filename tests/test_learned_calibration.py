"""Tests for the learned-calibration orchestrator (design3 §2-§7, §9)."""
import time

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
        # L 50% + A 40% + D 10% of B; the fallback is paid for out of D's share.
        assert (cfg.sample_budget, cfg.calib_budget, cfg.route_budget) == (50, 40, 10)

    def test_rejects_fractions_over_one(self):
        with pytest.raises(ValueError):
            LearnedCalibrationConfig(total_budget=100, sample_frac=0.95, calib_frac=0.1)

    def test_total_budget_floor(self):
        assert learned_calibration_total_budget(50, 1.0) == 0
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

    def test_all_negative_world_stops_after_the_routing_share_then_refuses(self):
        x, h, _, labeled = _world(seed=4)
        y_true = np.zeros(x.size, dtype=np.int32)
        labeller = _CountingLabeller(y_true)
        cfg = LearnedCalibrationConfig(total_budget=300, seed=5)
        res = run_learned_calibration(x=x, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                                      oof=OofFeatures(None, None, "skipped"), label_rows=labeller, config=cfg)
        assert res.status == "insufficient_evidence"
        assert res.diagnostics["lc_reason"] == "single_class_in_A"
        # A gets its own 40% plus one fallback round paid from D's 10%; the rest of B is NOT spent.
        assert res.calib_idx.size == cfg.calib_budget + cfg.route_budget
        # spend is L + A + D's share and never more than B; whatever the fallback leaves is not spent
        assert len(labeller.rows) == cfg.calib_budget + cfg.route_budget
        assert labeled.size + len(labeller.rows) <= cfg.total_budget
        assert res.verify_idx.size == 0
        assert res.diagnostics["lc_fallback_rounds"] == 1
        assert res.diagnostics["lc_fallback_used"] == cfg.route_budget
        assert np.isnan(res.diagnostics["lc_estimated_f1"])

    def test_missing_logit_spends_leftover_in_rounds_without_calibration(self):
        x, h, y_true, labeled = _world(seed=6)
        labeller = _CountingLabeller(y_true)
        cfg = LearnedCalibrationConfig(total_budget=200, seed=7)
        res = run_learned_calibration(x=None, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                                      oof=OofFeatures(None, None, "skipped"), label_rows=labeller, config=cfg)
        assert (res.status, res.diagnostics["lc_reason"]) == ("insufficient_evidence", "proxy_logit_unavailable")
        assert res.verify_idx.size == 0 and res.calib_idx.size == 50
        assert np.array_equal(res.final_pos[res.calib_idx], y_true[res.calib_idx] == 1)

    def test_fallback_round_that_finds_evidence_switches_the_router_on(self, monkeypatch):
        import learned_calibration as lc
        from weighted_calibrator import CalibrationFit

        x, h, y_true, labeled = _world(seed=12)
        real_calibrate = lc.calibrate
        calls = []

        def flaky_calibrate(**kw):
            calls.append(kw["y_a"].size)
            if len(calls) == 1:
                return CalibrationFit(None, "insufficient_evidence", "single_class_in_A", {})
            return real_calibrate(**kw)

        monkeypatch.setattr(lc, "calibrate", flaky_calibrate)
        labeller = _CountingLabeller(y_true)
        cfg = LearnedCalibrationConfig(total_budget=400, seed=13)
        res = run_learned_calibration(x=x, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                                      oof=OofFeatures(None, None, "skipped"), label_rows=labeller, config=cfg)
        # first draw = calib_budget (160), fallback round = what is left of D's share (40)
        assert calls == [cfg.calib_budget, cfg.calib_budget + cfg.route_budget]
        assert res.status == "routed_after_fallback"
        assert res.diagnostics["lc_fallback_rounds"] == 1
        assert res.calib_idx.size == cfg.calib_budget + cfg.route_budget
        assert labeled.size + len(labeller.rows) <= cfg.total_budget
        assert len(labeller.rows) == len(set(labeller.rows))
        # the fallback used up D's share, so nothing is left to route
        assert res.diagnostics["lc_fallback_used"] == cfg.route_budget
        assert res.verify_idx.size == 0

    def test_first_round_success_keeps_plain_routed_status(self):
        x, h, y_true, labeled = _world(seed=14)
        res = run_learned_calibration(x=x, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                                      oof=OofFeatures(None, None, "skipped"),
                                      label_rows=_CountingLabeller(y_true),
                                      config=LearnedCalibrationConfig(total_budget=600, seed=15))
        assert res.status == "routed"
        assert res.diagnostics["lc_fallback_rounds"] == 0

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

    def spy(row_idx, p, s, h_arg, counts, budget, **kw):
        captured["rows"], captured["key"] = row_idx, s
        return real_plan(row_idx, p, s, h_arg, counts, budget, **kw)

    monkeypatch.setattr(lc, "plan_dual_threshold", spy)
    res = run_learned_calibration(x=x, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                                  oof=OofFeatures(None, None, "skipped"),
                                  label_rows=_CountingLabeller(y_true),
                                  config=LearnedCalibrationConfig(total_budget=400, seed=11))
    assert res.status == "routed"
    assert np.array_equal(captured["key"], x[captured["rows"]])
    tau = res.diagnostics["lc_tau_plus"]
    assert np.isnan(tau) or 0.0 <= tau <= 1.0


class TestConservativeClopperPearson:
    """A with no positive: cp_lam > 0 anchors the level on a bound instead of refusing."""

    def test_zero_positive_A_routes_conservatively_instead_of_refusing(self):
        x, h, y_true, labeled = _world(seed=21)
        # Force A to see only negatives by labelling every row outside L as 0.
        y_a_world = y_true.copy()
        y_a_world[np.setdiff1d(np.arange(x.size), labeled)] = 0
        labeller = _CountingLabeller(y_a_world)
        common = dict(x=x, h=h, labeled_indices=labeled, labeled_values=y_true[labeled],
                      oof=OofFeatures(x[labeled], h[labeled], ""), label_rows=labeller)

        strict = run_learned_calibration(config=LearnedCalibrationConfig(total_budget=400, seed=3), **common)
        assert strict.status == "insufficient_evidence"
        assert strict.verify_idx.size == 0

        labeller2 = _CountingLabeller(y_a_world)
        common2 = dict(common, label_rows=labeller2)
        lenient = run_learned_calibration(
            config=LearnedCalibrationConfig(total_budget=400, seed=3, cp_lam=0.06), **common2)
        assert lenient.status == "routed_conservative"
        assert lenient.verify_idx.size > 0
        d = lenient.diagnostics
        assert d["lc_cp_used"] is True
        # measured positives are 0, so the bound alone carries the target
        assert d["lc_cp_measured_pos"] == 0.0
        assert 0.0 < d["lc_cp_target_pos"] <= d["lc_cp_upper_pos"]

    def test_bound_shrinks_towards_the_measured_rate_as_the_sample_grows(self):
        from weighted_calibrator import clopper_pearson_upper, stratified_positive_target

        assert clopper_pearson_upper(0, 10, 0.05) > clopper_pearson_upper(0, 1000, 0.05)
        assert clopper_pearson_upper(0, 1000, 0.05) == pytest.approx(3.0 / 1000, rel=0.1)
        # dense sample: mixing barely moves the target away from the measured count
        y = np.array([1.0] * 50 + [0.0] * 450)
        w = np.full(500, 2.0)
        target, diag = stratified_positive_target(y, w, lam=0.06, alpha=0.05)
        assert diag["cp_measured_pos"] == pytest.approx(100.0)
        assert target == pytest.approx(100.0, rel=0.05)


class TestStageTimers:
    def test_reports_sample_fit_and_route_times(self):
        """The A-labelling, fitting and routing times are reported and sum to lc_time_s."""
        import numpy as np
        from learned_calibration import LearnedCalibrationConfig, OofFeatures, run_learned_calibration

        rng = np.random.default_rng(0)
        n = 800
        y = (rng.random(n) < 0.25).astype(int)
        x = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1.2, n)
        h = (x > 0).astype(float)
        cfg = LearnedCalibrationConfig(total_budget=200, sample_frac=0.8, calib_frac=0.2,
                                       route_frac=0.0, route_unlimited=True, alpha=0.9, seed=0)
        l_idx = rng.choice(n, cfg.sample_budget, replace=False)

        def label_rows(rows):
            time.sleep(0.01)          # make the labelling cost visible in the timers
            return y[rows]

        t0 = time.perf_counter()
        res = run_learned_calibration(
            x=x, h=h, labeled_indices=l_idx, labeled_values=y[l_idx],
            oof=OofFeatures(None, None, "n/a"), label_rows=label_rows, config=cfg)
        elapsed = time.perf_counter() - t0
        d = res.diagnostics
        for key in ("lc_time_sample_s", "lc_time_fit_s", "lc_time_route_s"):
            assert key in d and d[key] >= 0.0
        assert res.calib_idx.size > 0 and d["lc_time_sample_s"] > 0.0
        if res.verify_idx.size:
            assert d["lc_time_route_s"] > 0.0
        parts = d["lc_time_sample_s"] + d["lc_time_fit_s"] + d["lc_time_route_s"]
        assert parts <= elapsed + 1e-6           # the parts are disjoint spans of this call
        assert parts >= 0.5 * elapsed            # and cover most of it
