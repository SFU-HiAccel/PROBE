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


from stratified_calibration import StratifiedDesign, allocate_cumulative  # noqa: E402


class TestAllocateCumulative:
    def test_equalizes_cumulative_counts_low_score_first(self):
        assert allocate_cumulative(np.array([5, 5, 5]), np.array([2, 1, 1]), 2).tolist() == [2, 2, 2]
        assert allocate_cumulative(np.array([5, 5, 5]), np.array([2, 2, 2]), 1).tolist() == [3, 2, 2]

    def test_respects_capacity(self):
        assert allocate_cumulative(np.array([1, 5, 5]), np.array([1, 1, 1]), 4).tolist() == [1, 3, 3]

    def test_fewer_draws_than_strata_is_allowed(self):
        assert allocate_cumulative(np.array([4, 4, 4]), np.zeros(3, dtype=np.int64), 1).tolist() == [1, 0, 0]

    def test_matches_allocate_equal_from_zero(self):
        sizes = np.array([1, 5, 5])
        assert allocate_cumulative(sizes, np.zeros(3, dtype=np.int64), 7).tolist() == allocate_equal(sizes, 7).tolist()

    def test_rejects_more_draws_than_remaining_capacity(self):
        with pytest.raises(ValueError):
            allocate_cumulative(np.array([2, 2]), np.array([2, 1]), 2)


class TestStratifiedDesignRounds:
    def test_rounds_never_repeat_rows_and_weights_track_cumulative_counts(self):
        rng = np.random.default_rng(0)
        scores = rng.random(500)
        pool = np.arange(500)
        design = StratifiedDesign.build(pool, scores, n_bins=10, max_strata=10)
        seen = []
        for m in (10, 10, 7):
            design, new_rows = design.draw(m, rng)
            assert new_rows.size == m
            seen.extend(new_rows.tolist())
        assert len(seen) == len(set(seen)) == 27
        sample = design.sample()
        assert sorted(sample.indices.tolist()) == sorted(seen)
        assert sample.weights.sum() == pytest.approx(500)
        assert np.all(sample.sample_counts == np.array([s.size for s in design.sampled]))

    def test_draw_is_capped_by_remaining_rows(self):
        design = StratifiedDesign.build(np.arange(6), np.linspace(0, 1, 6), n_bins=2, max_strata=2)
        design, first = design.draw(4, np.random.default_rng(1))
        design, second = design.draw(99, np.random.default_rng(2))
        assert first.size == 4 and second.size == 2
        assert design.remaining == 0
        assert np.allclose(design.sample().weights, 1.0)

    def test_first_round_matches_draw_stratified_sample(self):
        scores = np.random.default_rng(3).random(300)
        a = draw_stratified_sample(np.arange(300), scores, 20, n_bins=10, rng=np.random.default_rng(4))
        design = StratifiedDesign.build(np.arange(300), scores, n_bins=10, max_strata=20)
        design, _ = design.draw(20, np.random.default_rng(4))
        b = design.sample()
        assert np.array_equal(a.indices, b.indices) and np.allclose(a.weights, b.weights)


class TestNeymanAllocation:
    """Neyman tilts the draws towards strata that can hold positives, but never starves one."""

    def test_tilts_towards_the_informative_strata(self):
        from stratified_calibration import allocate_neyman
        sizes = np.array([1000, 1000, 1000])
        counts = np.zeros(3, dtype=np.int64)
        p_hat = np.array([0.001, 0.02, 0.4])           # rising positive rate with the score
        got = allocate_neyman(sizes, counts, 90, p_hat)
        assert got.sum() == 90
        assert got[2] > got[1] > got[0]

    def test_every_stratum_keeps_a_nonzero_inclusion_probability(self):
        from stratified_calibration import allocate_neyman
        sizes = np.array([500, 500, 500, 500])
        p_hat = np.array([0.0, 0.0, 0.0, 0.9])         # only one stratum looks informative
        got = allocate_neyman(sizes, np.zeros(4, dtype=np.int64), 40, p_hat)
        assert got.sum() == 40
        assert (got > 0).all(), f"a stratum received no draws: {got}"

    def test_design_weights_still_invert_the_inclusion_probability(self):
        from stratified_calibration import StratifiedDesign
        rng = np.random.default_rng(0)
        pool = np.arange(600)
        scores = np.linspace(0.0, 1.0, 600)
        design = StratifiedDesign.build(pool, scores, n_bins=6, max_strata=6, neyman=True)
        design, _ = design.draw(60, rng)
        smp = design.sample()
        for size, n, w in zip(smp.stratum_sizes, smp.sample_counts, np.unique(smp.weights)):
            pass                                        # weights are checked per row below
        for stratum, taken in zip(design.strata, design.sampled):
            if taken.size:
                w = stratum.size / taken.size
                assert np.isclose(smp.weights[np.isin(smp.indices, taken)], w).all()
