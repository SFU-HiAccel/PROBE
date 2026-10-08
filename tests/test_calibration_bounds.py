import numpy as np
from itertools import combinations, product
from types import SimpleNamespace
from scipy.stats import hypergeom
from calibration_bounds import hypergeom_upper_total, missed_positive_bound
from learned_calibration import LearnedCalibrationConfig, OofFeatures, run_learned_calibration


def test_no_sample_keeps_full_unknown_population():
    pool=np.arange(10)
    assert missed_positive_bound(pool,None,np.full(10,-1),[],[],np.zeros(10,bool)) == (10,10)
    final=np.zeros(10,bool);final[3]=True
    assert missed_positive_bound(pool,None,np.full(10,-1),[3],[1],final) == (9,10)


def test_exact_one_sided_coverage_all_small_populations():
    for N in range(1,15):
        for n in range(N+1):
            bounds=[hypergeom_upper_total(k,n,N,.05) for k in range(n+1)]
            for K in range(N+1):
                cov=sum(hypergeom.pmf(k,N,K,n) for k,b in enumerate(bounds) if K<=b)
                assert cov >= .95-1e-12
    assert hypergeom_upper_total(3,10,10,.05)==3


def test_population_event_supports_all_post_calibration_output_choices():
    # Exhaust all SRS samples and, for each sample, all possible outputs and
    # zero/one-row routing choices. Counting a sample as failed if ANY such
    # choice violates the bound is stronger than checking one fitted threshold.
    truth=np.array([1,1,0,0,0,0])
    pool=np.arange(len(truth));eta=.25
    samples=list(combinations(pool,3));failed_samples=0
    for sample in samples:
        sample=np.array(sample)
        rest=np.setdiff1d(pool,sample)
        known=np.full(len(pool),-1);known[sample]=truth[sample]
        design=SimpleNamespace(strata=[pool],sampled=[sample])
        any_failure=False
        for route in [[], *[[int(i)] for i in rest]]:
            routed=np.array(route,dtype=int)
            unknown=np.setdiff1d(rest,routed)
            for choices in product([False,True],repeat=len(unknown)):
                final=np.zeros(len(pool),bool)
                final[sample]=truth[sample].astype(bool)
                final[routed]=truth[routed].astype(bool)
                final[unknown]=choices
                upper,_=missed_positive_bound(pool,design,known,routed,
                    truth[routed],final,eta=eta)
                missed=int((truth.astype(bool)&~final).sum())
                any_failure |= missed>upper
        failed_samples += any_failure
    assert failed_samples/len(samples)<=eta


def test_zero_total_budget_is_zero_and_has_nonzero_unknown_bound():
    cfg=LearnedCalibrationConfig(total_budget=0,fixed_calibration=True)
    assert cfg.sample_budget==cfg.calib_budget==cfg.route_budget==0
    def forbidden(_): raise AssertionError('zero budget oracle call')
    result=run_learned_calibration(x=None,h=np.zeros(9),labeled_indices=np.array([],int),
        labeled_values=np.array([],int),oof=OofFeatures(None,None,'empty'),label_rows=forbidden,config=cfg)
    assert result.diagnostics['lc_calls_total']==0
    assert result.diagnostics['lc_upper_missed']==9
    assert result.diagnostics['lc_upper_pool']==9


def test_fixed_audit_does_not_keep_sampling_until_positive():
    calls=[]
    def label(rows): calls.extend(rows);return np.zeros(len(rows),int)
    cfg=LearnedCalibrationConfig(total_budget=20,sample_frac=.5,calib_frac=.2,route_frac=.3,
                                 fallback_unlimited=True,fixed_calibration=True)
    result=run_learned_calibration(x=None,h=np.zeros(100),labeled_indices=np.arange(10),
        labeled_values=np.zeros(10,int),oof=OofFeatures(None,None,'single_class'),label_rows=label,config=cfg)
    assert len(calls)==4
    assert result.diagnostics['lc_fallback_rounds']==0
    assert result.diagnostics['lc_upper_missed']>0
    assert result.diagnostics['lc_bound_scope']=='stationary_oracle_labels_fixed_A_design'
