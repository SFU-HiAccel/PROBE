import numpy as np
import pytest
from experiments.paper_statistics import (binary_f1, bootstrap_summary,
                                          paired_stratified_bootstrap)


def test_pairing_and_significance_use_shared_seed_and_class_draws():
    y = np.array([1]*5+[0]*95, dtype=bool)
    good = np.tile(y, (5,1))
    bad = np.zeros_like(good)
    draws = paired_stratified_bootstrap(y, {'PROBE':good,'same':good,'bad':bad},
        scorer=binary_f1,n_boot=100,seed=9)
    np.testing.assert_array_equal(draws['PROBE'],draws['same'])
    summary = bootstrap_summary(draws)
    assert summary['PROBE']['ci95']==[1.,1.]
    assert summary['same']['focus_difference_ci95']==[0.,0.]
    assert not summary['same']['significant'] and summary['bad']['significant']


def test_class_population_and_custom_query_scorer_are_retained():
    y = np.array([1]*2+[0]*18,dtype=bool)
    predictions=np.zeros((5,20),bool);predictions[:,0]=True
    def scorer(truth,prediction,rows):
        assert len(rows)==20 and np.count_nonzero(truth[rows])==2
        # A count-based scorer exercises the callback; binary F1 would differ.
        return 1.-abs(int(prediction[rows].sum())-2)/20
    draws=paired_stratified_bootstrap(y,{'PROBE':predictions},scorer=scorer,n_boot=100)
    assert min(draws['PROBE'])>=.9 and max(draws['PROBE'])==1
    repeat=paired_stratified_bootstrap(y,{'PROBE':predictions},scorer=scorer,n_boot=100)
    np.testing.assert_array_equal(draws['PROBE'],repeat['PROBE'])


def test_seed_resampling_can_represent_run_variability():
    y=np.array([1,0],bool)
    p=np.array([[1,0],[1,0],[1,0],[0,0],[0,0]],bool)
    d=paired_stratified_bootstrap(y,{'PROBE':p},scorer=binary_f1,n_boot=500)
    assert np.std(d['PROBE'])>0


@pytest.mark.parametrize('shape',[(1,20),(4,20),(5,19)])
def test_missing_seeds_or_candidates_are_rejected(shape):
    with pytest.raises(ValueError,match='five measured seeds'):
        paired_stratified_bootstrap(np.zeros(20),{'PROBE':np.zeros(shape)},scorer=binary_f1,n_boot=2)


def test_bootstrap_report_binds_all_runs_and_annotation_arrays(tmp_path):
    import json
    import hashlib
    from experiments.paper_statistics import build_report, validate_report
    manifest={'protocol':{'revision':'fixture'},'experiments':{'main':{
        'queries':['q'],'systems':['PROBE','no_proxy'],'seeds':list(range(5))}}}
    records=[]
    for system in ['PROBE','no_proxy']:
        for seed in range(5):
            folder=tmp_path/f'{system}-{seed}'/'attempt-001';folder.mkdir(parents=True)
            path=folder/'evaluation.npz'
            np.savez_compressed(path,ids=['a','b'],truth=np.array([1,0],bool),
                                prediction=np.array([1,0],bool),metric='binary_f1')
            path.with_suffix('.json').write_text(json.dumps(dict(sources={'fixture':'digest'})))
            completion=folder.parent/'completed.json'
            completion.write_text(json.dumps(dict(outputs={'attempt-001/evaluation.npz':'hash',
                                                           'attempt-001/evaluation.json':'hash'})))
            records.append(dict(experiment='main',query='q',system=system,seed=seed,f1=1.,
                raw_record=str(completion.relative_to(tmp_path)),raw_sha256=hashlib.sha256(completion.read_bytes()).hexdigest()))
    report=build_report(tmp_path,records,manifest,n_boot=20)
    assert report['groups']['all_queries']['summary']['PROBE']['ci95']==[1.,1.]
    validate_report(report,records,manifest)
    changed=[dict(r,raw_sha256='modified') for r in records]
    with pytest.raises(ValueError,match='does not match'):validate_report(report,changed,manifest)
    with pytest.raises(ValueError,match='Missing or duplicate'):build_report(tmp_path,records[:-1],manifest,n_boot=20)


def test_ece_validation_rejects_nonbinary_truth():
    from experiments.paper_statistics import expected_calibration_error
    with pytest.raises(ValueError):expected_calibration_error([2],[.5])
