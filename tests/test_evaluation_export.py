import json
import numpy as np
import pandas as pd
import pytest
from experiments import evaluation_export as export
from experiments.paper_statistics import expected_calibration_error


class Frame:
    def __init__(self, data): self.data = data
    def toPandas(self): return self.data.copy()
    def select(self, *columns): return Frame(self.data[list(columns)])


def setup(monkeypatch, tmp_path, *, experiment='main', query='fixture'):
    monkeypatch.setenv('PROBE_PAPER_JOB', json.dumps(dict(experiment=experiment,seed=0,system='PROBE')))
    monkeypatch.setenv('PROBE_PAPER_OUTPUT',str(tmp_path))
    monkeypatch.setattr(export,'_calibration',None)


def test_retrieval_export_retains_distinct_scorer_units(tmp_path,monkeypatch):
    setup(monkeypatch,tmp_path)
    frame = Frame(pd.DataFrame({'id':['a','a','b','c']}))
    pred = tmp_path/'pred.csv';pd.DataFrame({'id':['a','a','c']}).to_csv(pred,index=False)
    gt = tmp_path/'gt.csv';pd.DataFrame({'id':['a']}).to_csv(gt,index=False)
    export.remember_population(lambda:frame,[1,1,0,1])
    meta=export.export_evaluation('fixture',lambda f:f,pred,gt,2/3)
    assert meta['physical_N']==4 and meta['evaluation_N']==3
    with np.load(tmp_path/'evaluation.npz') as data:
        assert data['truth'].sum()==1 and data['prediction'].sum()==2
    export.remember_population(lambda:frame,[1,1,0,1])
    with pytest.raises(ValueError,match='differs'):
        export.export_evaluation('fixture',lambda f:f,pred,gt,1.)


def test_embedded_multicolumn_reference_needs_no_gt_file(tmp_path,monkeypatch):
    setup(monkeypatch,tmp_path)
    frame=Frame(pd.DataFrame({'ID':[1,2],'image':['a.png','b.png']}))
    pred=tmp_path/'pred.csv';frame.data.iloc[:1].to_csv(pred,index=False)
    export.remember_population(lambda:frame,[1,0])
    meta=export.export_evaluation('mmqa_q2a',lambda f:f,pred,tmp_path/'absent.csv',1.,mmqa_truth={(1,'a.png')})
    assert meta['quality']==1. and 'embedded_reference' in meta['sources']


def test_animals_uses_physical_rows_and_binary_f1(tmp_path,monkeypatch):
    from experiments.workload_ground_truth import animals_q1_ground_truth
    setup(monkeypatch,tmp_path)
    source=tmp_path/'SemBench/files/animals/data/sf_1600';source.mkdir(parents=True)
    pd.DataFrame({'ImagePath':['same','same','other'],'Species':['ZEBRA','ZEBRA','LION']}).to_csv(source/'image_data.csv',index=False)
    gt=animals_q1_ground_truth(tmp_path,1600,tmp_path)
    frame=Frame(pd.DataFrame({'row_id':[0,1,2]}))
    pred=tmp_path/'pred.csv';pd.DataFrame({'row_id':[1,2]}).to_csv(pred,index=False)
    export.remember_population(lambda:frame,[0,1,1])
    meta=export.export_evaluation('animals_q1',lambda f:f,pred,gt,.5)
    assert meta['metric']=='binary_f1' and meta['evaluation_N']==3


def test_calibration_evaluation_excludes_fitting_rows(tmp_path,monkeypatch):
    setup(monkeypatch,tmp_path,experiment='calibration')
    frame=Frame(pd.DataFrame({'id':['a','b','c']}))
    pred=tmp_path/'pred.csv';pd.DataFrame({'id':['a']}).to_csv(pred,index=False)
    gt=tmp_path/'gt.csv';pd.DataFrame({'id':['a']}).to_csv(gt,index=False)
    export.remember_population(lambda:frame,[1,0,0])
    monkeypatch.setattr(export,'_calibration',(np.array([.8,.2,.4]),np.array([0,1,1],bool)))
    meta=export.export_evaluation('fixture',lambda f:f,pred,gt,1.)
    assert meta['ece']==pytest.approx(.3) and meta['ece_N']==2
    assert expected_calibration_error([1,0],[1,0])==0.
