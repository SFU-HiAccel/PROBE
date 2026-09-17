"""Tests for budgeted dual-threshold routing (design3 §7, §9)."""
from itertools import combinations

import numpy as np
import pytest

from budgeted_calibrated_router import (
    ExpectedCounts,
    apply_verified_labels,
    expected_counts,
    plan_dual_threshold,
)


def _f_tilde_for_subset(subset, p, h, counts):
    tp, fp, fn = counts.tp, counts.fp, counts.fn
    for i in subset:
        if h[i] > 0.5:
            fp -= 1.0 - p[i]
        else:
            tp += p[i]
            fn -= p[i]
    den = 2 * tp + fp + fn
    return float("nan") if den <= 0 else 2 * tp / den


class TestPlanDualThreshold:
    @pytest.mark.parametrize("seed", range(40))
    def test_matches_brute_force_over_all_subsets(self, seed):
        rng = np.random.default_rng(seed)
        n = 7
        s = rng.random(n)
        h = (rng.random(n) < 0.5).astype(float)
        p = np.clip(np.where(h > 0.5, s, s * 0.6) + rng.normal(0, 0.05, n), 0.01, 0.99)
        # rank-consistent within each group, as with a calibrator monotone in s
        p[h > 0.5] = np.sort(p[h > 0.5])[np.argsort(np.argsort(s[h > 0.5]))]
        p[h <= 0.5] = np.sort(p[h <= 0.5])[np.argsort(np.argsort(s[h <= 0.5]))]
        counts = expected_counts(p, h, n_verified_pos=int(rng.integers(0, 4)))
        budget = int(rng.integers(0, n + 2))
        plan = plan_dual_threshold(np.arange(n), p, s, h, counts, budget)
        r = min(budget, n)
        best = max(_f_tilde_for_subset(c, p, h, counts) for size in range(r + 1)
                   for c in combinations(range(n), size))
        assert plan.f_tilde == pytest.approx(best, abs=1e-9)
        assert plan.verify_idx.size == r

    def test_zero_budget_verifies_nothing(self):
        p = np.array([0.2, 0.9]); h = np.array([0.0, 1.0]); s = np.array([0.2, 0.9])
        counts = expected_counts(p, h, 3)
        plan = plan_dual_threshold(np.array([10, 11]), p, s, h, counts, 0)
        assert plan.verify_idx.size == 0 and np.isnan(plan.tau_plus) and np.isnan(plan.tau_minus)
        assert plan.f_tilde == pytest.approx(counts.f1())

    def test_budget_above_pool_verifies_everything(self):
        p = np.array([0.2, 0.9, 0.6]); h = np.array([0.0, 1.0, 1.0]); s = p.copy()
        plan = plan_dual_threshold(np.array([5, 7, 9]), p, s, h, expected_counts(p, h, 0), 99)
        assert plan.verify_idx.tolist() == [5, 7, 9]

    def test_tied_scores_never_exceed_budget(self):
        s = np.full(10, 0.5); h = np.ones(10); p = np.full(10, 0.5)
        plan = plan_dual_threshold(np.arange(10), p, s, h, expected_counts(p, h, 0), 4)
        assert plan.verify_idx.tolist() == [0, 1, 2, 3]

    def test_verifies_lowest_score_positives_and_highest_score_negatives(self):
        row = np.arange(6)
        s = np.array([0.95, 0.55, 0.75, 0.05, 0.45, 0.25])
        h = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        p = s.copy()
        plan = plan_dual_threshold(row, p, s, h, expected_counts(p, h, 0), 2)
        assert set(plan.verify_idx.tolist()) <= {1, 4}
        assert plan.k_plus + plan.k_minus == 2


class TestApplyVerifiedLabels:
    def test_matches_recomputation_from_scratch(self):
        rng = np.random.default_rng(0)
        p = rng.random(30); h = (rng.random(30) < 0.4).astype(float); y = (rng.random(30) < 0.5).astype(float)
        before = expected_counts(p, h, n_verified_pos=4)
        done = np.arange(10)
        after = apply_verified_labels(before, p[done], h[done], y[done])
        rest = np.arange(10, 30)
        expected = expected_counts(p[rest], h[rest], n_verified_pos=4 + int(y[done].sum()))
        assert (after.tp, after.fp, after.fn) == pytest.approx((expected.tp, expected.fp, expected.fn))


def test_f1_zero_denominator_is_nan():
    assert np.isnan(ExpectedCounts(0.0, 0.0, 0.0).f1())
