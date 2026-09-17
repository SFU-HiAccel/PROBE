"""Tests for the score-stratified calibration sample A (design3 §4, §9)."""
import numpy as np
import pytest

from stratified_calibration import allocate_equal, build_score_strata, draw_stratified_sample


class TestAllocateEqual:
    def test_caps_small_strata_and_refills_others(self):
        assert allocate_equal(np.array([1, 5, 5]), 7).tolist() == [1, 3, 3]

    def test_leftovers_go_to_low_score_strata_first(self):
        assert allocate_equal(np.array([5, 5, 5]), 4).tolist() == [2, 1, 1]
        assert allocate_equal(np.array([2, 2, 2]), 5).tolist() == [2, 2, 1]

    def test_rejects_quota_outside_feasible_range(self):
        with pytest.raises(ValueError):
            allocate_equal(np.array([3, 3]), 1)
        with pytest.raises(ValueError):
            allocate_equal(np.array([3, 3]), 7)


class TestBuildScoreStrata:
    def test_drops_empty_bins_and_keeps_score_one_in_last_bin(self):
        scores = np.array([0.05, 0.06, 0.95, 1.0])
        strata = build_score_strata(np.arange(4), scores, n_bins=10, max_strata=10)
        assert [s.tolist() for s in strata] == [[0, 1], [2, 3]]

    def test_merges_smallest_adjacent_pair_until_within_limit(self):
        scores = np.array([0.05] * 5 + [0.15] + [0.25] + [0.95] * 5)
        strata = build_score_strata(np.arange(12), scores, n_bins=10, max_strata=3)
        assert [s.size for s in strata] == [5, 2, 5]

    def test_equal_scores_share_a_stratum(self):
        scores = np.full(8, 0.42)
        strata = build_score_strata(np.arange(8), scores, n_bins=10, max_strata=8)
        assert len(strata) == 1

    def test_rejects_non_finite_scores(self):
        with pytest.raises(ValueError):
            build_score_strata(np.arange(2), np.array([0.1, np.nan]), n_bins=10, max_strata=2)


class TestDrawStratifiedSample:
    def test_fixed_size_disjoint_from_excluded_rows_and_weights_sum_to_pool(self):
        rng = np.random.default_rng(0)
        scores = rng.random(1000)
        pool = np.arange(100, 1000)
        sample = draw_stratified_sample(pool, scores, 50, n_bins=10, rng=rng)
        assert sample.size == 50
        assert np.unique(sample.indices).size == 50
        assert np.all(sample.indices >= 100)
        assert sample.weights.sum() == pytest.approx(pool.size)
        assert np.all(sample.sample_counts >= 1)
        assert np.all(sample.sample_counts <= sample.stratum_sizes)

    def test_zero_quota_returns_empty_sample(self):
        sample = draw_stratified_sample(np.arange(10), np.random.default_rng(1).random(10), 0,
                                        n_bins=10, rng=np.random.default_rng(1))
        assert sample.size == 0

    def test_quota_larger_than_pool_takes_everything(self):
        scores = np.linspace(0, 1, 20)
        sample = draw_stratified_sample(np.arange(20), scores, 99, n_bins=10,
                                        rng=np.random.default_rng(2))
        assert sorted(sample.indices.tolist()) == list(range(20))
        assert np.allclose(sample.weights, 1.0)

    def test_inclusion_frequency_matches_n_over_N(self):
        scores = np.array([0.05] * 10 + [0.95] * 2)
        hits = np.zeros(12)
        rng = np.random.default_rng(3)
        n_rep = 4000
        for _ in range(n_rep):
            hits[draw_stratified_sample(np.arange(12), scores, 4, n_bins=10, rng=rng).indices] += 1
        # strata sizes (10, 2), quota 4 -> counts (2, 2): pi = 0.2 and 1.0
        assert np.allclose(hits[:10] / n_rep, 0.2, atol=0.03)
        assert np.allclose(hits[10:] / n_rep, 1.0)
