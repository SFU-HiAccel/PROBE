"""Actual cluster pipeline entry uses fixed sampling, final LR and graph."""
import numpy as np
import pytest
import cluster_proxy as cp
from fixed_ce_sampling import SamplingConfig as _SamplingConfig


@pytest.mark.parametrize('budget', [0, 1, 2, 24])
@pytest.mark.parametrize('refine', [False, True])
def test_fixed_pipeline_bypasses_dynamic_axis(monkeypatch, budget, refine):
    monkeypatch.setattr(cp, '_KNN_CACHE_ENABLED', False)
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy dynamic sampling is reachable')
    assert not hasattr(cp, 'sample_activeSampling')
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(80,8))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    calls = []
    def label(batch):
        ids = list(map(int,batch)); calls.extend(ids)
        return np.array([int(i % 3 == 0) for i in ids])
    scores = np.arange(80, dtype=float)
    mask, profile = cp.cluster_proxy_pipeline(
        embeddings, list(map(str,range(80))), lambda: label, budget,
        sampling_method='activeSampling',
        activeSampling_backend='fixed_ce', fixed_ce_scores=scores,
        fixed_ce_config=SamplingConfig(batch_size=4, seed=7),
        knn_k=5, defer_enabled=False, refine_enabled=refine,
    )
    assert len(calls) == len(set(calls)) == (80 if budget < 2 else budget)
    info = profile['activeSampling_profiling']
    assert info['sampler'] == 'fixed_ce_pilot' and info['sort_count'] == 1
    assert info['_fixed_order'].tolist() == list(range(79,-1,-1))
    assert mask.shape == (80,)
    assert profile['n_labeled'] == budget


def test_fixed_pipeline_requires_explicit_ce_scores():
    with pytest.raises(ValueError, match='explicit fixed_ce_scores'):
        cp.cluster_proxy_pipeline(np.eye(4), list('abcd'), lambda: lambda _: [], 0,
            sampling_method='activeSampling', similarities=np.arange(4))


def test_shared_transport_caps_calibration_and_preserves_new_positives(monkeypatch):
    from oracle_budget import PipelineOracle
    from learned_calibration import LearnedCalibrationConfig
    monkeypatch.setattr(cp, '_KNN_CACHE_ENABLED', False)
    calls = []
    class Request:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def __call__(self, key):
            calls.append(key)
            return int(len(calls) > 5), {'prompt_tokens':2,'completion_tokens':1}
    # Duplicate text deliberately tests that production sampling/calibration
    # identifies records by original ID, not by prompt content or object identity.
    texts = ['same text'] * 30
    oracle = PipelineOracle(texts, limit=7, request_factory=Request)
    cfg = LearnedCalibrationConfig(total_budget=10, sample_frac=.5,calib_frac=.4,route_frac=.1)
    rng = np.random.default_rng(8)
    emb = rng.normal(size=(30,5))
    mask, profile = cp.cluster_proxy_pipeline(
        emb,texts,lambda:oracle,5,sampling_method='activeSampling',
        similarities=np.arange(30),fixed_ce_scores=np.arange(30),
        activeSampling_backend='fixed_ce',fixed_ce_config=SamplingConfig(batch_size=2),
        knn_k=3,refine_enabled=True,defer_enabled=True,
        decision_mode='learned_calibration',learned_calibration_config=cfg)
    assert len(calls) == len(set(calls)) == 7
    assert oracle.ledger.used == 7
    assert profile['result_status'] == 'llm_executed_with_failures'
    assert profile['discovery_status'] == 'positive_found'
    assert mask.sum() == 2  # discovered in S3 fallback, after all-negative L
    assert len(profile['unresolved_ids']) == 23
    assert profile['quality_certificate'] is None
    assert {e['stage'] for e in oracle.ledger.events} == {'L','fallback'}


def test_zero_hits_do_not_certify_empty_result(monkeypatch):
    from oracle_budget import PipelineOracle
    monkeypatch.setattr(cp,'_KNN_CACHE_ENABLED',False)
    class Request:
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def __call__(self,key): return 0, {}
    texts = list(map(str,range(20)))
    oracle = PipelineOracle(texts,limit=2,request_factory=Request)
    mask, profile = cp.cluster_proxy_pipeline(
        np.random.default_rng(3).normal(size=(20,4)),texts,lambda:oracle,2,
        sampling_method='activeSampling',similarities=np.arange(20),
        fixed_ce_config=SamplingConfig(),fixed_ce_scores=np.arange(20),activeSampling_backend='fixed_ce',knn_k=3)
    assert not mask.any()
    assert profile['result_status'] == 'llm_executed_with_failures'
    assert profile['discovery_status'] == 'no_positive_found'
    assert len(profile['unresolved_ids']) == 18


@pytest.mark.parametrize('positive,failed_id', [(False,None),(True,None),(False,3)])
def test_census_status_requires_successful_labels_for_entire_pool(monkeypatch,positive,failed_id):
    from oracle_budget import PipelineOracle
    monkeypatch.setattr(cp,'_KNN_CACHE_ENABLED',False)
    n=12
    class Request:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def __call__(self,key):
            if key==failed_id:raise TimeoutError('one unverified candidate')
            return int(positive),{}
    texts=['duplicate content']*n
    oracle=PipelineOracle(texts,limit=n,request_factory=Request)
    mask,profile=cp.cluster_proxy_pipeline(
        np.random.default_rng(41).normal(size=(n,4)),texts,lambda:oracle,n,
        sampling_method='activeSampling',fixed_ce_scores=np.arange(n),
        activeSampling_backend='fixed_ce',fixed_ce_config=SamplingConfig(batch_size=4),
        knn_k=3,refine_enabled=True,defer_enabled=False)
    assert oracle.ledger.used==n
    if failed_id is not None:
        assert profile['result_status']=='llm_executed_with_failures'
        assert profile['discovery_status']=='no_positive_found'
        assert profile['unresolved_ids']==[failed_id]
        assert not mask.any()
    else:
        assert profile['result_status']=='llm_executed'
        assert profile['unresolved_ids']==[]
        assert profile['discovery_status']==('positive_found' if positive else 'verified_empty')
        assert np.array_equal(mask,np.full(n,positive))
    assert profile['quality_certificate'] is None


@pytest.mark.parametrize('rule', ['none', 'all'])
def test_paper_protocol_single_class_hands_whole_query_to_llm(monkeypatch, rule):
    """Paper Alg. 1 lines 12-13: L without both classes -> every row labelled by the LLM."""
    monkeypatch.setattr(cp, '_KNN_CACHE_ENABLED', False)
    rng = np.random.default_rng(3)
    embeddings = rng.normal(size=(60, 8))
    calls = []
    def oracle(batch):
        ids = list(map(int, batch)); calls.extend(ids)
        return np.array([int(rule == 'all') for _ in ids])
    mask, profile = cp.cluster_proxy_pipeline(
        embeddings, list(map(str, range(60))), lambda: oracle, 10,
        sampling_method='activeSampling', activeSampling_backend='fixed_ce',
        fixed_ce_scores=np.arange(60, dtype=float),
        fixed_ce_config=SamplingConfig(batch_size=4, protocol='paper', pilot_mode='uniform',
                                       pilot_fraction=.2),
        knn_k=5, defer_enabled=False, refine_enabled=True,
    )
    assert profile['whole_query_fallback'] is True
    assert sorted(calls) == list(range(60))          # every row labelled exactly once
    assert profile['whole_query_fallback_calls'] == 50
    assert mask.tolist() == [rule == 'all'] * 60


def SamplingConfig(**kwargs):
    # Explicit small batches exercise the paper hard-budget contract.
    kwargs.setdefault("protocol", "paper")
    kwargs.setdefault("pilot_fraction", .1)
    kwargs.setdefault("pilot_mode", "uniform")
    kwargs.setdefault("n_min", 1)
    return _SamplingConfig(**kwargs)
