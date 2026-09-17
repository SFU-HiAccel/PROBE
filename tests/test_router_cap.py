"""Tests for the router call cap (patch R1, budget-F1 experiment protocol §4.1).

Sampling receives the full LLM budget ``B``. The router may escalate at most
``floor(cap_frac * B)`` additional rows, so total calls stay within
``B + floor(cap_frac * B)``. Sampling is not reduced to make room for routing:
an earlier 70/30 split starved the sampler at small budgets (match202 F1 fell
from 1.00 to 0.21) while the router spent none of its reserved quota.
"""
import numpy as np
import pytest

from router_cap import cap_router_candidates, split_total_budget


class TestSplitTotalBudget:
    def test_sampling_gets_full_budget_and_router_cap_is_extra(self):
        total, sample, cap = split_total_budget(n_items=10_000, budget_pct=10, cap_frac=0.3)
        assert total == 1000
        assert sample == 1000
        assert cap == 300

    def test_sampling_budget_always_equals_total(self):
        for n in (266, 1426, 9828, 27000):
            for pct in (0.1, 0.5, 1, 5, 10):
                total, sample, cap = split_total_budget(n, pct, 0.3)
                assert sample == total
                assert cap == int(0.3 * total)

    def test_matches_unpatched_sampling_budget(self):
        # run.py previously used max(10, int(N * pct / 100)) as the sample size.
        for n, pct in ((270, 10), (306, 10), (266, 10), (100, 10), (27000, 1)):
            _, sample, _ = split_total_budget(n, pct, 0.3)
            assert sample == max(10, int(n * pct / 100))

    def test_floor_of_ten_applies_to_total(self):
        # match113: N=266, 0.1% -> 0.266 rows -> total floored to 10
        total, sample, cap = split_total_budget(n_items=266, budget_pct=0.1, cap_frac=0.3)
        assert total == 10
        assert sample == 10
        assert cap == 3

    def test_zero_cap_frac_disables_router_budget(self):
        total, sample, cap = split_total_budget(n_items=1000, budget_pct=10, cap_frac=0.0)
        assert cap == 0
        assert sample == total == 100

    def test_invalid_cap_frac_rejected(self):
        with pytest.raises(ValueError):
            split_total_budget(n_items=1000, budget_pct=10, cap_frac=1.5)
        with pytest.raises(ValueError):
            split_total_budget(n_items=1000, budget_pct=10, cap_frac=-0.1)


class TestCapRouterCandidates:
    def test_keeps_highest_probability_rows(self):
        cand_idx = np.arange(50)
        defer_proba = np.zeros(100)
        defer_proba[cand_idx] = np.linspace(0.51, 0.99, 50)  # row 49 is most likely wrong
        kept = cap_router_candidates(cand_idx, defer_proba, router_max_calls=10)
        assert kept.size == 10
        assert set(kept.tolist()) == set(range(40, 50))

    def test_result_is_sorted_by_row_index(self):
        cand_idx = np.array([7, 3, 11, 5])
        defer_proba = np.zeros(20)
        defer_proba[[7, 3, 11, 5]] = [0.9, 0.95, 0.6, 0.8]
        kept = cap_router_candidates(cand_idx, defer_proba, router_max_calls=3)
        assert kept.tolist() == [3, 5, 7]

    def test_no_cap_returns_all(self):
        cand_idx = np.arange(50)
        defer_proba = np.full(50, 0.7)
        kept = cap_router_candidates(cand_idx, defer_proba, router_max_calls=None)
        np.testing.assert_array_equal(kept, cand_idx)

    def test_under_cap_returns_all(self):
        cand_idx = np.array([2, 4, 6])
        defer_proba = np.full(10, 0.7)
        kept = cap_router_candidates(cand_idx, defer_proba, router_max_calls=10)
        np.testing.assert_array_equal(kept, cand_idx)

    def test_zero_cap_returns_empty(self):
        cand_idx = np.arange(5)
        defer_proba = np.full(5, 0.9)
        kept = cap_router_candidates(cand_idx, defer_proba, router_max_calls=0)
        assert kept.size == 0

    def test_ties_break_deterministically(self):
        cand_idx = np.arange(10)
        defer_proba = np.full(10, 0.8)
        a = cap_router_candidates(cand_idx, defer_proba, router_max_calls=4)
        b = cap_router_candidates(cand_idx, defer_proba, router_max_calls=4)
        np.testing.assert_array_equal(a, b)
