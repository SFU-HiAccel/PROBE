"""Execution contracts: fake experts are isolated unit fixtures, never paper evidence."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paper_reproduce',ROOT/'experiments/reproduce.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def arguments(**overrides):
    values=dict(experiment='main',queries=None,systems=None,seeds=None,
                model='expert',oracle_url='http://localhost/v1/chat/completions',
                model_revision='immutable',ce_revision='ce-immutable',model_config=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_sampling_and_refinement_cover_paper_budget_grid():
    for experiment,variants in [('sampling',4),('refinement',3)]:
        jobs=runner.plan(arguments(experiment=experiment))
        assert {j['point']['budget_pct'] for j in jobs}=={5,10,15,20,25}
        assert len(jobs)==14*5*variants
        assert all(j['executor'] for j in jobs)


def test_experts_have_model_matched_no_proxy_for_each_model(tmp_path):
    path=tmp_path/'models.json'
    experts={name:dict(model=name,oracle_url='http://'+str(i),revision='r'+str(i))
             for i,name in enumerate(['Qwen3-VL-30B-A3B-Instruct','Qwen3.8-27B','Gemma4-26B-A4B','GLM-4.6V-Flash'])}
    path.write_text(json.dumps(dict(experts=experts)))
    jobs=runner.plan(arguments(experiment='experts',model_config=path))
    assert len(jobs)==14*4*2
    for name,config in experts.items():
        part=[j for j in jobs if j['point']['expert']==name]
        assert {j['system'] for j in part}=={'PROBE','no_proxy'}
        assert all(j['model']==config['model'] and j['model_revision']==config['revision'] and
                   j['oracle_url']==config['oracle_url'] for j in part)


def test_bargain_sweep_changes_target_with_fixed_preparation_budget():
    jobs=runner.plan(arguments(experiment='sweep',systems=['BARGAIN-A'],queries=['match113']))
    assert {j['point']['target'] for j in jobs}=={.1,.3,.5,.7,.9}
    assert {j['point']['budget_pct'] for j in jobs}=={10}


def test_low_selectivity_target_baselines_have_one_canonical_point():
    jobs=runner.plan(arguments(experiment='low_selectivity'))
    assert len(jobs)==15
    keys=[json.dumps(j,sort_keys=True) for j in jobs]
    assert len(set(keys))==len(keys)
    for system in ['BARGAIN-A','ScaleDoc','no_proxy']:
        assert len([j for j in jobs if j['system']==system])==3


def test_selection_rejects_wrong_query_and_duplicate_seed():
    with pytest.raises(ValueError,match='Queries outside'):
        runner.plan(arguments(experiment='trace',queries=['match113']))
    with pytest.raises(ValueError,match='Seeds'):
        runner.plan(arguments(seeds=[0,0]))
    assert len(runner.plan(arguments(experiment='all',systems=['PROBE'],queries=['match113'])))>0


class ExpertFixture:
    def __init__(self,n):
        from oracle_budget import OracleLedger
        self.ledger=OracleLedger(n)
        self.requests=[]
    def value(self,key):return int(key%7==0)
    def label_ids(self,ids,*,stage='L'):
        result=[]
        for key in ids:
            key=int(key)
            if key not in self.ledger.cache:
                assert self.ledger.reserve(key,stage)
                label=self.value(key)
                self.ledger.record(key,label)
                self.requests.append((key,stage))
            result.append(self.ledger.cache[key])
        return np.asarray(result)
    def __call__(self,batch):
        return self.label_ids([int(s) for s in batch])


def fixture(tmp_path,monkeypatch,experiment,point=None,system='PROBE'):
    from fixed_ce_sampling import SamplingConfig
    from learned_calibration import paper_calibration_config
    from experiments.paper_pipeline import execute
    n=200
    job=dict(experiment=experiment,query='unit-fixture',paper_N=n,seed=0,model='unit',
             model_revision='unit-revision',oracle_url='unit://',system=system,point=point or {})
    output=tmp_path/experiment/system;output.mkdir(parents=True,exist_ok=True)
    monkeypatch.setenv('PROBE_PAPER_JOB',json.dumps(job))
    monkeypatch.setenv('PROBE_PAPER_OUTPUT',str(output))
    monkeypatch.setenv('PROBE_FROZEN_ROOT',str(tmp_path/'frozen'))
    monkeypatch.delenv('PROBE_ORACLE_LOG',raising=False)
    oracle=ExpertFixture(n)
    features=np.random.default_rng(9).normal(size=(n,5))
    scores=np.linspace(1,0,n)
    result,profile=execute(features,list(map(str,range(n))),lambda:oracle,16,scores,
        SamplingConfig(seed=0),paper_calibration_config(20),query_embedding=np.ones(5))
    assert result.shape==(n,) and result.dtype==bool
    assert len(set(i for i,_ in oracle.requests))==len(oracle.requests)
    assert profile['n_total']==n
    assert all(x>=0 for x in profile['paper_stage_seconds'].values())
    assert (output/'candidate_predictions.npz').is_file()
    return result,profile,oracle,output


@pytest.mark.parametrize('sampler',['activeSampling','random','stratified','top_k'])
def test_sampling_variants_execute_without_graph_or_router(tmp_path,monkeypatch,sampler):
    import cluster_proxy
    def forbidden(*args,**kwargs):raise AssertionError('sampling ablation reached refinement')
    monkeypatch.setattr(cluster_proxy,'_build_knn_graph',forbidden)
    _,profile,oracle,output=fixture(tmp_path,monkeypatch,'sampling',dict(sampler=sampler))
    assert profile['n_labeled']<=16
    assert profile['paper_stage_seconds']['refinement']==0
    assert not any(stage in ('A','D') for _,stage in oracle.requests)


def test_refinement_reuses_identical_actual_acquisition(tmp_path,monkeypatch):
    records=[]
    for stage in ['LR','bias','full']:
        _,profile,oracle,output=fixture(tmp_path,monkeypatch,'refinement',dict(stage=stage))
        record=json.loads((output/'frozen_acquisition.json').read_text())
        records.append(record)
        if stage!='LR':assert not oracle.requests
    assert len({r['labels_sha256'] for r in records})==1
    assert len({json.dumps(r['identity'],sort_keys=True) for r in records})==1


@pytest.mark.parametrize('experiment',['main','sweep','calibration','representation','experts','timing','low_selectivity','trace'])
def test_real_execution_paths_record_predictions_and_timers(tmp_path,monkeypatch,experiment):
    _,profile,oracle,output=fixture(tmp_path,monkeypatch,experiment)
    assert oracle.requests
    if experiment=='trace':
        assert (output/'sampling_trace.json').is_file()
        assert (output/'encoder_ranking.npz').is_file()


def test_csv_and_aiqa_preserve_acquired_labels(tmp_path,monkeypatch):
    for system in ['CSV','AIQA']:
        result,profile,oracle,_=fixture(tmp_path,monkeypatch,'main',dict(budget_pct=10),system)
        for key,label in oracle.ledger.cache.items():assert result[key]==bool(label)
        assert profile['mode']==system


def test_completed_measurement_rejects_failures_missing_gt_and_nonfinite():
    valid=dict(f1='0',delegation_rate='1',duration_s='2',oracle_failures='0',gt_source_sha256='digest')
    runner.validate_measurement(valid)
    for changes in [dict(oracle_failures='1'),dict(gt_source_sha256=''),dict(f1='nan'),dict(duration_s='0')]:
        with pytest.raises(RuntimeError):runner.validate_measurement(dict(valid,**changes))
    assert runner.immutable_revision('a'*40)
    assert not runner.immutable_revision('REPLACE_WITH_SERVED_WEIGHT_COMMIT')


def test_scaledoc_calibration_keeps_exact_count_and_final_edge():
    from experiments.scaledoc_adapter import calibration_indices
    scores=np.ones(200)
    _,bins=np.histogram(scores,bins=64)
    np.random.seed(2)
    ids=calibration_indices(scores,bins,.05,count=13)
    assert len(ids)==13 and len(set(ids))==13
    assert np.all((ids>=0)&(ids<200))
    ids=calibration_indices(np.linspace(0,1,100),np.linspace(0,1,65),.05,count=100)
    assert set(ids)==set(range(100))


def test_lotus_uses_actual_helper_calls_and_upstream_thresholds(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from experiments.paper_baselines import lotus_cascade
    from experiments.paper_pipeline import TimedOracle
    import openai
    oracle=ExpertFixture(80)
    oracle.request_factory=lambda:SimpleNamespace(system_prompt='query-specific prompt')
    requests=[]
    class Client:
        def __init__(self,**kwargs):
            self.chat=SimpleNamespace(completions=self)
        def create(self,**kwargs):
            requests.append(kwargs)
            p=.8 if int(kwargs['messages'][1]['content'])%7==0 else .2
            token=SimpleNamespace(token='True' if p>.5 else 'False',top_logprobs=[
                SimpleNamespace(token='True',logprob=float(np.log(p))),
                SimpleNamespace(token='False',logprob=float(np.log(1-p)))])
            return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[token]))],
                model_dump=lambda:dict(unit_fixture=True))
    monkeypatch.setattr(openai,'OpenAI',Client)
    mask,metadata=lotus_cascade(list(map(str,range(80))),TimedOracle(oracle),
        dict(model='unit',oracle_url='http://unit/v1/chat/completions',revision='unit'),
        target=.9,seed=0,output=tmp_path)
    assert len(requests)==80 and len(mask)==80
    assert all(r['messages'][0]['content']=='query-specific prompt' for r in requests)
    assert len((tmp_path/'proxy_oracle.jsonl').read_text().splitlines())==80
    assert metadata['lower']<=metadata['upper']
    for i,label in oracle.ledger.cache.items():assert mask[i]==bool(label)


def test_no_proxy_checks_universe_before_constructing_expert(monkeypatch):
    from proxy_model import _make_fixed_ce_oracle
    monkeypatch.setenv('PROBE_PAPER_JOB',json.dumps(dict(paper_N=384)))
    with pytest.raises(ValueError,match='Candidate mismatch before expert calls'):
        _make_fixed_ce_oracle(['wrong universe'],'unit','unit://','prompt',16,0,32,1,None,None,None)


def test_match202_uses_high_rated_movies_with_string_ids(tmp_path,monkeypatch):
    import run
    from pyspark.sql import SparkSession
    folder=tmp_path/'movies';folder.mkdir()
    header='ID,Title,Year,Director,Genre,Rating\n'
    (folder/'imdb.csv').write_text(header+'a-low,Low,2000,A,Drama,8.3\na-keep,Keep,2001,B,Drama,8.4\n')
    (folder/'rotten_tomatoes.csv').write_text(header+'b-low,Low,2000,A,Drama,8.5\nb-keep,Keep,2001,B,Drama,8.6\n')
    spark=SparkSession.builder.master('local[1]').appName('paper-lro-loader-contract').getOrCreate()
    seen={}
    monkeypatch.setattr(run,'get_spark',lambda:spark)
    monkeypatch.setattr(run,'LROBENCH_DATABASES_DIR',str(tmp_path))
    monkeypatch.setattr(run,'_write_match_gt_csv',lambda tag,truth:seen.update(gt=truth) or 'unit-gt')
    def capture(tag,left,right,prompt,gt,left_id,right_id):
        seen.update(tag=tag,left=[r[left_id] for r in left.collect()],right=[r[right_id] for r in right.collect()])
    monkeypatch.setattr(run,'_run_match_join',capture)
    try:run.run_match202()
    finally:spark.stop()
    assert seen['tag']=='match202' and seen['left']==['a-keep'] and seen['right']==['b-keep']
    assert len(seen['gt'])==3


@pytest.mark.parametrize('mode',['A','P','R'])
def test_bargain_modes_execute_on_refined_scores_with_shared_expert(tmp_path,monkeypatch,mode):
    monkeypatch.setenv('PROBE_BARGAIN_MODE',mode)
    monkeypatch.setenv('PROBE_BARGAIN_TARGET','.9')
    monkeypatch.setenv('PROBE_BARGAIN_BUDGET_FRAC','1')
    monkeypatch.setenv('PROBE_BARGAIN_R_BUDGET_FRAC','1')
    result,profile,oracle,_=fixture(tmp_path,monkeypatch,'calibration',dict(target=.9),system='BARGAIN-'+mode)
    assert profile['bargain_mode']==mode
    if mode in ('P','R'):assert profile['bargain_budget']==200
    for key,label in oracle.ledger.cache.items():
        if key in np.load(tmp_path/'calibration'/('BARGAIN-'+mode)/'training_labels.npz')['ids']:
            assert result[key]==bool(label)


@pytest.mark.parametrize('system,experiment',[('PROBE','sampling'),('PROBE','refinement'),('AIQA','main')])
def test_sampling_only_single_class_cannot_buy_full_oracle(tmp_path,monkeypatch,system,experiment):
    class NegativeExpert(ExpertFixture):
        def value(self,key):return 0
    monkeypatch.setattr(__import__(__name__,fromlist=['ExpertFixture']),'ExpertFixture',NegativeExpert)
    mask,profile,oracle,_=fixture(tmp_path,monkeypatch,experiment,dict(budget_pct=10,sampler='random'),system)
    assert not mask.any()
    assert len(oracle.requests)<= (20 if system=='AIQA' else 16)
    assert not any(stage in ('fallback','D','A') for _,stage in oracle.requests)
