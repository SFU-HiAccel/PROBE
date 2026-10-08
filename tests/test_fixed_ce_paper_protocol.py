"""Paper Algorithms 1-2 (SamplingConfig(protocol='paper'))."""
import numpy as np
import pytest
from scipy.stats import hypergeom

from fixed_ce_sampling import (SamplingConfig, _stratum_quota, one_sided_lower, one_sided_upper,
                               paper_config, sample_fixed_ce)


def paper(**kw):
    return SamplingConfig(protocol='paper', pilot_mode='uniform', pilot_fraction=.2, **kw)


@pytest.mark.parametrize('N,n,k', [(1000, 40, 0), (1000, 40, 3), (50, 50, 5), (200, 10, 10)])
def test_one_sided_upper_is_largest_consistent_K(N, n, k):
    K = one_sided_upper(N, n, k, .05)
    assert hypergeom.cdf(k, N, K, n) > .05
    if K < N - n + k:
        assert hypergeom.cdf(k, N, K + 1, n) <= .05
    assert one_sided_upper(N, 0, 0) == N


def test_pilot_has_no_batch_minimum_and_gate_is_once():
    # B_L = 50 -> m0 = 10 even though batch_size = 32.
    res = sample_fixed_ce(range(2000), -np.arange(2000.), lambda _: 0, 50, paper(batch_size=32))
    assert res.pilot.budget == 10
    gates = [t for t in res.trace if t['stage'] == 'gate']
    assert len(gates) == 1 and gates[0]['n0'] == 10


@pytest.mark.parametrize('N,n,k', [(1000, 40, 0), (1000, 40, 3), (1000, 40, 12), (50, 50, 5), (200, 10, 10)])
def test_one_sided_lower_is_smallest_consistent_K(N, n, k):
    K = one_sided_lower(N, n, k, .05)
    assert hypergeom.sf(k - 1, N, K, n) > .05
    if K > k:
        assert hypergeom.sf(k - 1, N, K - 1, n) <= .05
    assert one_sided_lower(N, 0, 0) == 0
    assert one_sided_lower(N, n, 0) == 0


def test_no_pilot_label_stays_multi():
    # Lower-bound gate: without evidence that positives are common, the query stays highly selective.
    res = sample_fixed_ce(range(100), np.zeros(100), lambda _: 0, 4, paper(batch_size=2))
    assert res.pilot.budget == 0 and res.pilot.mode == 'multi'


def test_small_negative_pilot_stays_multi():
    res = sample_fixed_ce(range(1000), -np.arange(1000.), lambda _: 0, 50,
                          paper_config(batch_size=8))
    gate = [t for t in res.trace if t['stage'] == 'gate'][0]
    assert gate['n0'] == 15 and gate['k0'] == 0 and gate['L0'] == 0
    assert res.pilot.mode == 'multi' and res.pilot.regime == 'highly_selective'


def test_common_positives_go_stratified():
    # 40% positives, 90-row pilot: the lower bound clears p_star = 0.1.
    res = sample_fixed_ce(range(2000), -np.arange(2000.), lambda i: int(i % 5 < 2), 300,
                          paper_config(batch_size=32))
    gate = [t for t in res.trace if t['stage'] == 'gate'][0]
    assert gate['L0'] > .1 and res.pilot.mode == 'stratified'
    assert res.pilot.regime == 'not_highly_selective'


def test_paper_config_defaults():
    cfg = paper_config()
    assert (cfg.protocol, cfg.pilot_mode, cfg.pilot_fraction, cfg.p_star, cfg.delta_sel, cfg.n_min) == \
        ('paper', 'uniform', .3, .05, .05, 2)
    assert paper_config(p_star=.2).p_star == .2


def test_scan_phase_takes_top_ranks_until_first_positive():
    N = 5000
    scores = -np.arange(N, dtype=float)            # rank == id
    positives = {700}
    res = sample_fixed_ce(range(N), scores, lambda i: int(i in positives), 600,
                          paper(batch_size=40, explore_fraction=.25, p_star=.2))
    assert res.pilot.mode == 'multi'
    scan = [t for t in res.trace if t['stage'] == 'scan']
    assert scan and all(t['n'] == 30 for t in scan[:3])   # 3/4 of each batch
    multi_ids = [i for t in res.trace if t['stage'] == 'multi' for i in t['ids']]
    scanned = sorted(i for i in multi_ids if i < 400)
    assert scanned[:30] == list(range(30))                  # high-to-low from rank 0


def test_interval_phase_fixes_c_at_first_positive_and_keeps_exploring():
    N = 3000
    scores = -np.arange(N, dtype=float)
    res = sample_fixed_ce(range(N), scores, lambda i: int(i < 20), 400,
                          paper(batch_size=32, p_star=.2))
    cs = {t['c'] for t in res.trace if t['stage'] == 'windows'}
    assert len(cs) == 1
    assert len(res.attempted_ids) == 400 == len(set(res.attempted_ids))
    assert all(i in res.labels for i in range(20))


def test_stratified_quota_reaches_every_stratum_and_respects_budget():
    N = 1024
    res = sample_fixed_ce(range(N), -np.arange(N, dtype=float), lambda i: int(i % 3 == 0), 300,
                          paper(batch_size=32, train_strata=6))
    assert res.pilot.mode == 'stratified'
    assert len(res.attempted_ids) == 300 == len(set(res.attempted_ids))
    takes = [t['take'] for t in res.trace if t['stage'] == 'strata_alloc']
    assert all(min(t) >= 1 for t in takes[:3])


def test_stratum_quota_rotation_and_caps():
    take = _stratum_quota([5, 0, 5, 5], [0, 0, 0, 0], 2, 1)
    assert take.sum() == 2 and take[1] == 0
    take = _stratum_quota([1, 100], [1, 0], 10, 0)
    assert take.tolist() == [1, 9]


def test_cache_is_free_and_public_default_is_paper():
    cache = {i: 0 for i in range(50)}
    calls = []
    sample_fixed_ce(range(500), np.zeros(500), lambda i: calls.append(i) or 0, 40,
                    paper(batch_size=8), cache)
    assert not set(calls) & set(range(50)) and len(calls) == 40
    assert SamplingConfig().protocol == 'paper'
    assert SamplingConfig(protocol='legacy').protocol == 'legacy'


def test_explore_phase_needs_n_min_positives():
    # One positive at the top, the next far down: with n_min = 2 the scan continues after the first hit.
    N = 5000
    positives = {3, 4000}
    def run(n_min):
        res = sample_fixed_ce(range(N), -np.arange(N, dtype=float), lambda i: int(i in positives), 600,
                              paper(batch_size=40, explore_fraction=.25, p_star=.2, n_min=n_min))
        return sum(t['stage'] == 'scan' for t in res.trace), res
    scans_1, res_1 = run(1)
    scans_2, res_2 = run(2)
    assert 3 in res_1.labels and 3 in res_2.labels
    assert scans_1 == 1 and scans_2 > scans_1


def test_n_min_must_be_positive():
    with pytest.raises(ValueError):
        SamplingConfig(n_min=0)
