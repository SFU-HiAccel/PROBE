import json
import pytest
from experiments.paper_plots import best_points, render


def point(quality, delegation, seconds=1):
    return dict(f1_mean=quality, delegation_mean=delegation, latency_mean_s=seconds)


def test_quality_and_cost_operating_points_follow_paper():
    rows = [point(.8,.2), point(.8,.15), point(.72,.1), point(.6,.05)]
    quality, cost, reached = best_points(rows, .8)
    assert quality == rows[1]
    assert cost == rows[2] and reached
    quality, cost, reached = best_points(rows, 1.)
    assert cost == quality == rows[1] and not reached
    zero = point(0.,.1)
    assert best_points([zero], 0.) == (zero, zero, True)


def test_sampling_figure_uses_every_variant_and_budget(tmp_path):
    rows = [dict(point=json.dumps(dict(sampler=s, budget_pct=b)), experiment='sampling',
                 query=q, system='PROBE', complete=True, **point(.5, b/100))
            for q in ['q1','q2'] for s in ['activeSampling','random'] for b in [5,10]]
    render('sampling', rows, {'experiments': {'sampling': {'queries':['q1','q2']}}}, tmp_path)
    assert (tmp_path/'sampling_ablation.pdf').stat().st_size > 1000
    assert (tmp_path/'sampling_ablation.png').stat().st_size > 1000


def test_main_cannot_invent_bootstrap_from_aggregate_means(tmp_path):
    with pytest.raises(ValueError, match='bootstrap'):
        render('main', [], {'experiments': {'main': {'queries':[]}}}, tmp_path)


@pytest.mark.parametrize('experiment',['sweep','calibration','representation','experts','low_selectivity'])
def test_other_paper_renderers_accept_their_declared_grids(tmp_path,experiment):
    from pathlib import Path
    from experiments.reproduce import points_for_system
    manifest=json.loads((Path(__file__).parents[1]/'experiments/paper_manifest.json').read_text())
    spec=manifest['experiments'][experiment]
    spec['queries']=spec['queries'][:2]
    rows=[dict(experiment=experiment,query=q,system=s,point=json.dumps(p),complete=True,
               **point(.7,.2,2 if s=='no_proxy' else 1))
          for q in spec['queries'] for s in spec['systems'] for p in points_for_system(spec,s)]
    render(experiment,rows,manifest,tmp_path)
    assert list(tmp_path.glob('*.pdf')) or list(tmp_path.glob('*.csv'))


def test_trace_figures_use_recorded_batches_and_rank_truth(tmp_path):
    import numpy as np
    run=tmp_path/'run';attempt=run/'attempt-001';attempt.mkdir(parents=True)
    np.savez_compressed(attempt/'encoder_ranking.npz',order=np.arange(8),scores=np.linspace(1,0,8))
    np.savez_compressed(attempt/'candidate_truth.npz',truth=np.array([1,0]*4,bool))
    np.savez_compressed(attempt/'training_labels.npz',ids=np.arange(8),labels=[1,0]*4)
    events=[dict(stage='pilot_audit' if i==0 else 'multi',ids=[2*i,2*i+1]) for i in range(4)]
    (attempt/'sampling_trace.json').write_text(json.dumps(dict(trace=events)))
    (run/'completed.json').write_text(json.dumps(dict(outputs={str(p.relative_to(run)):'hash' for p in attempt.iterdir()})))
    render('trace',[],{'experiments':{'trace':{'queries':['fixture']}}},tmp_path/'plots',
           input_root=tmp_path,records=[dict(experiment='trace',raw_record='run/completed.json')])
    assert (tmp_path/'plots/encoder_rank.pdf').is_file()
    assert (tmp_path/'plots/sas_multi_process.pdf').is_file()
