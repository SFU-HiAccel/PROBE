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


class TestCostQualityAlpha:
    """alpha decides how much of the routing budget is worth spending (budget stays a hard cap)."""

    def _instance(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.05, 0.95, size=n)
        h = (p > 0.5).astype(float)
        idx = np.arange(n)
        counts = expected_counts(p, h, n_verified_pos=3)
        return idx, p, h, counts

    def test_alpha_one_spends_the_whole_budget(self):
        idx, p, h, counts = self._instance()
        plan = plan_dual_threshold(idx, p, p, h, counts, budget=10, alpha=1.0)
        assert plan.n_calls == 10
        assert plan.k_plus + plan.k_minus == 10
        # identical to calling without the dial at all
        base = plan_dual_threshold(idx, p, p, h, counts, budget=10)
        assert (plan.k_plus, plan.k_minus) == (base.k_plus, base.k_minus)

    def test_lower_alpha_never_spends_more(self):
        idx, p, h, counts = self._instance(seed=5)
        calls = [plan_dual_threshold(idx, p, p, h, counts, budget=20, alpha=a, n_pool=len(idx)).n_calls
                 for a in (1.0, 0.9, 0.6, 0.3, 0.05)]
        assert calls == sorted(calls, reverse=True)
        assert calls[-1] < calls[0]

    def test_plan_maximises_the_alpha_objective(self):
        idx, p, h, counts = self._instance(n=25, seed=9)
        alpha, budget = 0.7, 12
        plan = plan_dual_threshold(idx, p, p, h, counts, budget=budget, alpha=alpha, n_pool=len(idx))
        obj = lambda pl: alpha * pl.f_tilde - (1 - alpha) * pl.n_calls / len(idx)
        for r in range(budget + 1):
            other = plan_dual_threshold(idx, p, p, h, counts, budget=r, alpha=1.0)
            assert obj(plan) >= obj(other) - 1e-12


def _brute_best_splits(p, h, counts, r_max):
    """Reference: for every r, scan every split k_plus (the pre-optimisation O(R^2) planner)."""
    pos = np.asarray(h) > 0.5
    a = np.sort(1.0 - p[pos])[::-1]
    b = np.sort(p[~pos])[::-1]
    sp = np.concatenate([[0.0], np.cumsum(a)])
    sm = np.concatenate([[0.0], np.cumsum(b)])
    ks, fs = [], []
    for r in range(r_max + 1):
        k = np.arange(max(0, r - b.size), min(r, a.size) + 1)
        tp = counts.tp + sm[r - k]
        den = 2 * tp + np.maximum(counts.fp - sp[k], 0) + np.maximum(counts.fn - sm[r - k], 0)
        f = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)
        j = int(np.argmax(np.where(np.isnan(f), -np.inf, f)))
        ks.append(int(k[j])); fs.append(float(f[j]))
    return np.array(ks), np.array(fs)


class TestFastSplitSearch:
    @pytest.mark.parametrize("seed", range(40))
    def test_matches_quadratic_reference(self, seed):
        from budgeted_calibrated_router import best_splits
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 400))
        p = rng.beta(0.3, 3.0, n) if seed % 3 else np.round(rng.random(n), 1)   # every third: many ties
        h = (rng.random(n) < 0.3).astype(float)
        counts = expected_counts(p, h, n_verified_pos=int(rng.integers(0, 5)))
        r_max = int(rng.integers(0, n + 1))
        k_ref, f_ref = _brute_best_splits(p, h, counts, r_max)
        pos = h > 0.5
        a = np.sort(1.0 - p[pos])[::-1]
        b = np.sort(p[~pos])[::-1]
        k_new, f_new = best_splits(a, b, counts, r_max)
        # f_new is F1 at k_new, so matching f_ref means k_new is a maximiser. Among plans whose F1
        # ties exactly, the reference's pick is decided by float rounding, so k itself is not compared.
        np.testing.assert_allclose(f_new, f_ref, rtol=0, atol=1e-12)
        r = np.arange(r_max + 1)
        assert np.all((k_new >= np.maximum(0, r - b.size)) & (k_new <= np.minimum(r, a.size)))
