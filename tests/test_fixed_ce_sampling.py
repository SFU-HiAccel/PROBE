import numpy as np
import pytest
from scipy.stats import hypergeom

from fixed_ce_sampling import SamplingConfig as _SamplingConfig, sample_fixed_ce


@pytest.mark.parametrize('budget', [0, 1, 2, 7, 30])
def test_budget_cache_and_zero_positive(budget):
    calls = []
    cache = {3: 0}
    def oracle(i):
        calls.append(i)
        return 0
    result = sample_fixed_ce(range(20), np.zeros(20), oracle, budget,
                             SamplingConfig(batch_size=2, pilot_mode='uniform'), cache)
    assert len(calls) == min(budget, 19)
    assert len(calls) == len(set(calls))
    assert 3 not in calls
    assert result.order == list(range(20))
    assert result.verified_positives == []
    assert set(result.unresolved_ids) == set(range(20)) - set(cache)
    assert result.pilot.regime != 'not_highly_selective'


def test_pilot_not_selective_and_stratified_reuses_labels():
    result = sample_fixed_ce(range(100), np.arange(100), lambda _: 1, 70,
                             SamplingConfig(batch_size=32, pilot_mode='uniform'))
    assert result.pilot.regime == 'not_highly_selective'
    assert result.pilot.mode == 'stratified'
    assert len(result.labels) == 70
    assert len(result.attempted_ids) == len(set(result.attempted_ids))


def test_high_selectivity_requires_evidence():
    result = sample_fixed_ce(range(1000), np.arange(1000), lambda _: 0, 100,
                             SamplingConfig(batch_size=80, pilot_mode='uniform'))
    assert result.pilot.regime == 'highly_selective'
    tiny = sample_fixed_ce(range(1000), np.arange(1000), lambda _: 0, 2, SamplingConfig())
    assert tiny.pilot.regime == 'highly_selective'
    assert tiny.pilot.mode == 'multi'


def test_audit_prefix_does_not_skip_cached_ids():
    cfg = SamplingConfig(batch_size=12, pilot_mode='uniform', seed=18)
    first = sample_fixed_ce(range(100), np.arange(100), lambda i: i % 2, 20, cfg)
    cache = {i: i % 2 for i in first.pilot.audit_ids[:6]}
    second = sample_fixed_ce(range(100), np.arange(100), lambda i: i % 2, 20, cfg, cache)
    assert second.pilot.audit_ids == first.pilot.audit_ids
    assert second.pilot.interval == first.pilot.interval
    assert len(second.attempted_ids) == 20
    assert not set(first.pilot.audit_ids[:6]) & set(second.attempted_ids)


def test_failed_audit_abstains_and_attempts_are_capped():
    def failing(_):
        raise RuntimeError('offline')
    result = sample_fixed_ce(range(20), np.arange(20), failing, 7,
                             SamplingConfig(batch_size=2, pilot_mode='uniform'))
    assert len(result.attempted_ids) == len(result.failed_ids) == 7
    assert len(result.unresolved_ids) == 20
    assert result.pilot.regime == 'highly_selective'
    assert result.pilot.interval == (0., 1.)




def test_seed_followup_and_global_rng_isolation():
    # Isolated positives survive the interval penalty and receive explicit service.
    cfg = SamplingConfig(batch_size=4, pilot_mode='uniform', seed=4)
    cache = {10: 1, 70: 1}
    np.random.seed(12)
    state = np.random.get_state()
    result = sample_fixed_ce(range(100), -np.arange(100), lambda i: int(i in (10,70)), 24, cfg, cache)
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    served = set().union(*(set(t['served_seed_ids']) for t in result.trace if t['stage'] == 'windows'))
    assert {10,70} <= served


def test_validation_and_empty_pool():
    with pytest.raises(ValueError):
        sample_fixed_ce([1,1], [0,1], lambda _: 0, 2, SamplingConfig())
    with pytest.raises(ValueError):
        sample_fixed_ce([1], [float('nan')], lambda _: 0, 2, SamplingConfig())
    result = sample_fixed_ce([], [], lambda _: 0, 5)
    assert result.stop_reason == 'pool_labelled'
    assert not result.attempted_ids


def test_batch_and_scalar_oracles_have_identical_sampling():
    cfg = SamplingConfig(batch_size=7, seed=12)
    ids = [f'row-{i}' for i in range(80)]
    labels = {key: int(i % 5 == 0) for i, key in enumerate(ids)}
    calls = []
    def batch(keys):
        calls.append(keys)
        return [labels[k] for k in keys]
    scalar = sample_fixed_ce(ids, np.arange(80), labels.__getitem__, 45, cfg)
    batched = sample_fixed_ce(ids, np.arange(80), None, 45, cfg, batch_oracle=batch)
    assert scalar.labels == batched.labels
    assert scalar.attempted_ids == batched.attempted_ids
    assert scalar.pilot == batched.pilot
    assert sum(map(len, calls)) == 45
    assert max(map(len, calls)) <= 7
    assert len(calls) < 45


@pytest.mark.parametrize('response', ['exception', 'short', 'partial'])
def test_batch_failure_consumes_reservation_without_fabrication(response):
    def batch(keys):
        if response == 'exception':
            raise RuntimeError('transport failed')
        if response == 'short':
            return [1] * (len(keys)-1)
        return [None if i % 2 else 1 for i in range(len(keys))]
    result = sample_fixed_ce(range(30), np.arange(30), None, 11,
                             SamplingConfig(batch_size=4), batch_oracle=batch)
    assert len(result.attempted_ids) == 11
    assert len(result.labels) + len(result.failed_ids) == 11
    assert set(result.failed_ids) <= set(result.unresolved_ids)
    if response != 'partial':
        assert not result.labels


def test_text_adapter_preserves_original_ids_and_local_positions():
    from fixed_ce_sampling import sample_fixed_ce_texts
    texts = ['yes', 'no', 'yes', 'no']
    cache = {'c': 1}
    seen = []
    def label(texts):
        seen.extend(texts)
        return [int(t == 'yes') for t in texts]
    ix, y, graph, profile = sample_fixed_ce_texts(
        [2, 3, 1, 4], texts, label, 3,
        config=SamplingConfig(batch_size=2), original_ids=list('abcd'), label_cache=cache)
    assert dict(zip(ix, y)) == {0:1, 1:0, 2:1, 3:0}
    assert len(seen) == 3
    assert profile['sort_count'] == 1
    assert profile['_fixed_order'].tolist() == [3,1,0,2]
    assert graph is None
    assert not profile['unresolved_ids']


def SamplingConfig(**kwargs):
    # Explicit small batches exercise the paper hard-budget contract.
    kwargs.setdefault("protocol", "paper")
    kwargs.setdefault("pilot_fraction", .1)
    kwargs.setdefault("pilot_mode", "uniform")
    kwargs.setdefault("n_min", 1)
    return _SamplingConfig(**kwargs)
