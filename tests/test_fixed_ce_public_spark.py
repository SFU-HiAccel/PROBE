"""Real Spark filter/join boundaries with deterministic model/transport doubles.

Exercises production wrappers, sampler, final fit/graph, calibration and output
assembly. It does not evaluate real model accuracy or network transport.
"""
import numpy as np
import pytest
from pyspark.sql import SparkSession
import proxy_model as pm
import cluster_proxy as cp
import ce_ranker
import oracle_budget
from fixed_ce_sampling import SamplingConfig as _SamplingConfig
from learned_calibration import LearnedCalibrationConfig


@pytest.fixture(scope='module')
def spark():
    session=(SparkSession.builder.master('local[1]').appName('fixed-ce-public-contract')
             .config('spark.ui.enabled','false').config('spark.driver.host','127.0.0.1')
             .config('spark.sql.shuffle.partitions','1')
             .config('spark.sql.execution.arrow.pyspark.enabled','true').getOrCreate())
    session.sparkContext.setLogLevel('ERROR')
    yield session
    session.stop()


@pytest.mark.parametrize('kind',['filter','join'])
@pytest.mark.parametrize('scenario',['mixed','zero','failure'])
def test_public_spark_fixed_axis_and_budget(spark,monkeypatch,kind,scenario):
    monkeypatch.setenv('PROBE_REPR', 'mixed')
    calls=[];scored=[]
    def forbidden(*args,**kwargs):
        raise AssertionError('dynamic sampling reached from fixed CE public wrapper')
    monkeypatch.setattr(cp,'sample_activeSampling',forbidden)
    monkeypatch.setattr(cp,'_KNN_CACHE_ENABLED',False)
    def embeddings(rows,*args,**kwargs):
        return np.random.default_rng(3).normal(size=(len(rows),5))
    monkeypatch.setattr(pm,'embed_texts',embeddings)
    monkeypatch.setattr(pm,'embed_query',lambda *a,**k:np.ones((1,5)))
    def sides(left,right,*args,**kwargs):
        l,r=embeddings(left),embeddings(right)
        return dict(left_embs=l,right_embs=r,left_text_embs=l,right_text_embs=r,
                    left_img_embs=None,right_img_embs=None)
    monkeypatch.setattr(pm,'embed_rows_independent',sides)
    def ce_filter(records,columns,images,prompt,ids,**kwargs):
        assert prompt=='Relevant? {text}' and columns==['text']
        assert 'gt' not in columns
        scored.append(list(ids))
        return np.arange(len(ids),dtype=float),{'fingerprint':'test-filter'}
    def ce_join(left,right,pairs,lc,rc,images,prompt,ids,**kwargs):
        assert prompt=='Related? {left_text} {right_text}'
        assert lc==['left_text'] and rc==['right_text']
        assert ids==[[i,j] for i,j in pairs]
        scored.append(list(ids))
        return np.arange(len(ids),dtype=float),{'fingerprint':'test-join'}
    monkeypatch.setattr(ce_ranker,'score_filter',ce_filter)
    monkeypatch.setattr(ce_ranker,'score_join',ce_join)
    class Request:
        def __init__(self,*args,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def __call__(self,key):
            calls.append(key)
            if scenario=='failure':raise RuntimeError('injected transport failure')
            return int(scenario=='mixed' and key%3==0),{}
    # Keep _make_fixed_ce_oracle itself: its production budget calculation is tested.
    monkeypatch.setattr(oracle_budget,'OpenAIOracleRequest',Request)
    cfg=LearnedCalibrationConfig(total_budget=12,sample_frac=.5,calib_frac=.4,
        route_frac=.1,fixed_calibration=True,seed=0)
    kwargs=dict(model='deterministic-test',url='unused',embedding_model='test',
        sample_size=cfg.sample_budget,sampling_method='activeSampling',budget_size=10,
        fixed_ce_config=SamplingConfig(batch_size=2,seed=0),
        decision_mode='learned_calibration',learned_calibration_config=cfg,
        defer_enabled=True,refine_enabled=True,knn_k=3)
    if kind=='filter':
        # Duplicate content and an extra label-looking column test stable identity
        # and predicate-selected fields, not use of evaluation data by the scorer.
        df=spark.range(24).selectExpr('id', "'same' AS text", 'CAST(id % 2 AS INT) AS gt')
        output=pm.proxy_filter(df,'Relevant? {text}',**kwargs)
    else:
        left=spark.range(4).selectExpr('id AS left_id', "'same' AS left_text")
        right=spark.range(6).selectExpr('id AS right_id', "'same' AS right_text")
        output=pm.proxy_join(left,right,'Related? {left_text} {right_text}',**kwargs)
    count=output.count()  # actual Spark assembly, including empty join path
    prof=pm.get_last_profiling();cluster=prof['cluster_profiling']
    assert len(scored)==1 and len(scored[0])==24
    assert len(calls)<=10
    if scenario!='failure':assert len(calls)==len(set(calls))
    # Failed candidates may be retried by a later stage, but each try is charged.
    assert prof['oracle_budget']['candidate_attempts']==len(calls)
    assert prof['oracle_budget']['limit']==10
    sampling=cluster.get('activeSampling_profiling')
    if sampling is not None:
        assert sampling['sort_count']==1
        assert sampling['_fixed_order'].tolist()==list(range(23,-1,-1))
    if scenario!='mixed':
        assert count==0 and cluster['result_status']=='abstain'
        assert cluster['quality_certificate'] is None
        if scenario=='failure':assert len(cluster['unresolved_ids'])==24


def SamplingConfig(**kwargs):
    # These tests exercise saved legacy hard-budget contracts explicitly.
    kwargs.setdefault("protocol", "legacy")
    kwargs.setdefault("pilot_fraction", .1)
    kwargs.setdefault("pilot_mode", "hybrid")
    kwargs.setdefault("n_min", 1)
    return _SamplingConfig(**kwargs)
