"""Figures and operating-point tables from validated paper run aggregates."""
import csv
import json
import math
from pathlib import Path
import statistics


def geomean(values):
    values = list(values)
    if not values or any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError('Geometric means require complete finite nonnegative values')
    return 0. if 0 in values else math.exp(statistics.mean(map(math.log, values)))


def best_points(rows, reference):
    """Section 6: maximize quality then minimize delegation, or meet 90% reference."""
    if not rows:
        raise ValueError('An operating-point curve is empty')
    quality = max(r['f1_mean'] for r in rows)
    first = min((r for r in rows if r['f1_mean'] == quality),
                key=lambda r: (r['delegation_mean'], r['latency_mean_s']))
    qualified = [r for r in rows if r['f1_mean'] >= .9 * reference
                 or math.isclose(r['f1_mean'], .9 * reference, rel_tol=0., abs_tol=1e-12)]
    cost = min(qualified, key=lambda r: (r['delegation_mean'], -r['f1_mean'], r['latency_mean_s'])) if qualified else first
    return first, cost, bool(qualified)


def _row(rows, **values):
    matches = [r for r in rows if all(r.get(k) == v for k, v in values.items())]
    if len(matches) != 1:
        raise ValueError(f'Expected one measured configuration for {values}, found {len(matches)}')
    return matches[0]


def _group(rows, keys):
    result = {}
    for r in rows:
        p = json.loads(r['point'])
        key = tuple(p.get(k) for k in keys)
        result.setdefault((r['system'], *key), []).append(r)
    return result


def _save(fig, output, name, plt):
    fig.tight_layout()
    for extension in ('pdf', 'png'):
        fig.savefig(output / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def _table(output, name, rows):
    with (output / f'{name}.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # Escape identifiers without changing their values in the CSV companion.
    def tex(value):
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('_', r'\_').replace('%', r'\%')
    columns = list(rows[0])
    lines = [r'\begin{tabular}{' + 'l' * len(columns) + '}', r'\hline',
             ' & '.join(map(tex, columns)) + r' \\', r'\hline']
    lines += [' & '.join(tex(r[c]) for c in columns) + r' \\' for r in rows]
    lines += [r'\hline', r'\end{tabular}']
    (output / f'{name}.tex').write_text('\n'.join(lines) + '\n')


def render(experiment, rows, manifest, output, *, input_root=None, records=(), bootstrap=None):
    """Render one complete experiment. The caller validates raw hashes and coverage."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    spec = manifest['experiments'][experiment]
    queries = spec['queries']
    rows = [r for r in rows if r['experiment'] == experiment]
    if any(not r['complete'] for r in rows):
        raise ValueError('Plotting requires complete seed coverage for every configuration')
    if experiment == 'main':
        if bootstrap is None:
            raise ValueError('Main quality error bars require measured paired bootstrap results')
        for metric, name, ylabel in [('f1_mean', 'fig_sys_quality', 'Query quality'),
                                      ('speedup', 'fig_sys_speedup', 'Speedup over no proxy')]:
            fig, ax = plt.subplots(figsize=(16, 4))
            systems = spec['systems']; x = np.arange(len(queries)); width = .8 / len(systems)
            for i, system in enumerate(systems):
                arm = [_row(rows, query=q, system=system) for q in queries]
                values = [r[metric] if metric != 'speedup' else
                          _row(rows, query=r['query'], system='no_proxy')['latency_mean_s']/r['latency_mean_s'] for r in arm]
                positions = x - .4 + width*(i+.5)
                ax.bar(positions, values, width, label=system)
                if metric == 'f1_mean':
                    # Percentile intervals need not contain the original point estimate.
                    intervals = [bootstrap['queries'][q][system]['ci95'] for q in queries]
                    ax.vlines(positions, [v[0] for v in intervals], [v[1] for v in intervals], color='black', lw=.6)
            ax.set_xticks(x, queries, rotation=45, ha='right'); ax.set_ylabel(ylabel)
            ax.legend(ncol=7, fontsize=8); _save(fig, output, name, plt)
    elif experiment == 'sweep':
        fig, axes = plt.subplots(4, 4, figsize=(15, 12), squeeze=False)
        table = []
        for ax, query in zip(axes.flat, queries):
            reference = _row(rows, query=query, system='no_proxy')
            ax.axhline(reference['f1_mean'], color='.6', ls='--')
            for system in spec['systems']:
                arm = sorted([r for r in rows if r['query']==query and r['system']==system], key=lambda r:r['delegation_mean'])
                ax.plot([r['delegation_mean'] for r in arm], [r['f1_mean'] for r in arm], 'o-', ms=3, label=system)
                quality, cost, reached = best_points(arm, reference['f1_mean'])
                selected = [('quality_first', quality)] + ([('cost_first', cost)] if system == 'PROBE' else [])
                for policy, r in selected:
                    table.append(dict(query=query,system=system,policy=policy,f1=r['f1_mean'],
                        delegation=r['delegation_mean'],latency_s=r['latency_mean_s'],
                        speedup=reference['latency_mean_s']/r['latency_mean_s'],
                        reached_90pct_reference=reached if policy=='cost_first' else '', point=r['point']))
            ax.set(title=query, xlabel='Delegation fraction', ylabel='Query quality', ylim=(0, 1.03))
        for ax in list(axes.flat)[len(queries):]: ax.set_visible(False)
        axes.flat[0].legend(fontsize=7); _save(fig, output, 'all_queries', plt)
        _table(output, 'best_points', table)
        groups={}
        for row in table:groups.setdefault((row['system'],row['policy']),[]).append(row)
        _table(output,'best_points_summary',[
            dict(system=system,policy=policy,queries=len(arm),
                 quality_geomean=geomean(r['f1'] for r in arm),
                 delegation_median=statistics.median(r['delegation'] for r in arm),
                 latency_geomean_s=geomean(r['latency_s'] for r in arm),
                 speedup_geomean=geomean(r['speedup'] for r in arm))
            for (system,policy),arm in groups.items()])
    elif experiment in ('sampling', 'refinement'):
        key = 'sampler' if experiment == 'sampling' else 'stage'
        groups = _group(rows, [key, 'budget_pct'])
        variants = sorted({k[1] for k in groups})
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for variant in variants:
            budgets = sorted(k[2] for k in groups if k[1] == variant)
            quality = [geomean(r['f1_mean'] for r in groups[('PROBE', variant, b)]) for b in budgets]
            ax.plot(budgets, quality, 'o-', label=variant)
        ax.set(xlabel='Preparation budget (% of N)', ylabel='Geometric-mean quality')
        ax.legend(); _save(fig, output, 'sampling_ablation' if experiment=='sampling' else 'graph_ablation', plt)
    elif experiment == 'calibration':
        groups = _group(rows, ['calibrator', 'alpha', 'target'])
        curves = {}
        for (system, calibrator, alpha, target), arm in groups.items():
            label = calibrator if system == 'PROBE' else system
            curves.setdefault(label, []).append((alpha if system=='PROBE' else target,
                geomean(r['latency_mean_s'] for r in arm),geomean(r['f1_mean'] for r in arm)))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for label, points in sorted(curves.items()):
            points.sort(); ax.plot([p[1] for p in points],[p[2] for p in points],'o-',label=label)
        ax.set(xlabel='Geometric-mean latency (s)',ylabel='Geometric-mean quality')
        ax.legend(); _save(fig, output, 'alpha_tradeoff', plt)
        ece_rows=[]
        for record in records:
            if record['experiment']!='calibration' or record['system']!='PROBE':continue
            completion=Path(input_root)/record['raw_record']
            raw=json.loads(completion.read_text())
            paths=[completion.parent/p for p in raw['outputs'] if p.endswith('/evaluation.json')]
            if len(paths)!=1:raise ValueError(f'Missing calibration evaluation: {completion}')
            metadata=json.loads(paths[0].read_text())
            ece_rows.append(dict(query=record['query'],seed=record['seed'],point=record['point'],
                                 ece=metadata.get('ece'),evaluation_N=metadata.get('ece_N',0)))
        if ece_rows:
            _table(output,'calibration_ece',ece_rows)
            groups={}
            for r in ece_rows:groups.setdefault(r['point'],[]).append(r)
            _table(output,'calibration_ece_geomean',[
                dict(point=p,complete=all(r['ece'] is not None for r in arm),
                     ece_geomean=geomean(r['ece'] for r in arm) if all(r['ece'] is not None for r in arm) else None)
                for p,arm in groups.items()])
    elif experiment == 'representation':
        groups = _group(rows, ['representation'])
        labels = [k[1] for k in groups]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(labels, [geomean(r['f1_mean'] for r in arm) for arm in groups.values()])
        ax.set_ylabel('Geometric-mean quality'); ax.tick_params(axis='x',rotation=20)
        _save(fig, output, 'repr_models', plt)
    elif experiment == 'experts':
        groups = _group(rows, ['expert']); experts = list(dict.fromkeys(k[1] for k in groups))
        fig, ax = plt.subplots(figsize=(6, 3.5)); x=np.arange(len(experts))
        for offset, system in [(-.18, 'no_proxy'), (.18, 'PROBE')]:
            ax.bar(x+offset,[geomean(r['f1_mean'] for r in groups[(system, e)]) for e in experts],.36,label=system)
        ax.set_xticks(x,experts,rotation=20,ha='right'); ax.set_ylabel('Geometric-mean quality'); ax.legend()
        _save(fig,output,'fig_mllm_quality',plt)
        speedups=[]
        for expert in experts:
            probe=groups[('PROBE',expert)]; reference=groups[('no_proxy',expert)]
            speedups.append(geomean(_row(reference,query=r['query'])['latency_mean_s']/r['latency_mean_s'] for r in probe))
        fig,ax=plt.subplots(figsize=(6,3.5));ax.bar(experts,speedups)
        ax.set_ylabel('Geometric-mean speedup');ax.tick_params(axis='x',rotation=20)
        _save(fig,output,'fig_mllm_speedup',plt)
    elif experiment == 'low_selectivity':
        table=[]
        for r in rows:
            table.append(dict(query=r['query'],system=r['system'],point=r['point'],
                              quality=r['f1_mean'],delegation=r['delegation_mean']))
        _table(output,'lowsel_results',table)
    elif experiment == 'timing':
        stages=['embedding','sampling','expert_calls','proxy_training','proxy_prediction','refinement','calibration','client_and_spark_overhead']
        measurements={q:[] for q in queries}
        for record in records:
            if record['experiment'] != experiment: continue
            completion=Path(input_root)/record['raw_record']
            raw=json.loads(completion.read_text())
            paths=[completion.parent/name for name in raw['outputs'] if name.endswith('_paper_timing.json')]
            if len(paths)!=1:raise ValueError(f'Expected one timing record: {completion}')
            timer=json.loads(paths[0].read_text()); total=timer['measured_end_to_end_s']
            measurements[record['query']].append([timer['stages'][s]/total for s in stages[:-1]]+[timer['client_and_spark_overhead_s']/total])
        values=np.array([np.mean(measurements[q],axis=0) for q in queries])
        if values.shape!=(len(queries),len(stages)) or not np.isfinite(values).all():raise ValueError('Missing measured stage times')
        fig,ax=plt.subplots(figsize=(13,4));bottom=np.zeros(len(queries))
        for i,stage in enumerate(stages):
            ax.bar(queries,values[:,i],bottom=bottom,label=stage.replace('_',' '));bottom+=values[:,i]
        ax.set_ylabel('Fraction of measured end-to-end time');ax.tick_params(axis='x',rotation=45)
        ax.legend(ncol=4,fontsize=8);_save(fig,output,'fig_time_breakdown',plt)
    elif experiment == 'trace':
        selected = [r for r in records if r['experiment']=='trace']
        if len(selected)!=1:
            raise ValueError('Trace requires exactly one measured acquisition')
        completion = Path(input_root)/selected[0]['raw_record']
        raw = json.loads(completion.read_text())
        def artifact(name):
            paths = [completion.parent/p for p in raw['outputs'] if p.endswith('/'+name)]
            if len(paths)!=1: raise ValueError(f'Missing trace artifact: {name}')
            return paths[0]
        with np.load(artifact('encoder_ranking.npz')) as data:
            order, scores = data['order'], data['scores']
        with np.load(artifact('candidate_truth.npz')) as data:
            truth = data['truth']
        if len(truth)!=len(order): raise ValueError('Ranked annotations are not aligned')
        with np.load(artifact('training_labels.npz')) as data:
            labels = dict(zip(data['ids'].tolist(), data['labels'].tolist()))
        trace = json.loads(artifact('sampling_trace.json').read_text())['trace']
        fig,ax=plt.subplots(figsize=(9,2.8));rank=np.arange(len(order))
        ax.plot(rank,scores[order],color='#456b87',linewidth=.8,label='Encoder score')
        ax.scatter(rank[truth[order]],scores[order][truth[order]],s=9,color='#be4634',label='Human positive')
        ax.set(xlabel='Encoder rank',ylabel='Score');ax.legend()
        _save(fig,output,'encoder_rank',plt)
        rank_of=np.empty(len(order),int);rank_of[order]=rank
        batches=[];windows=[]
        for event in trace:
            if event['stage']=='windows':windows=event['windows']
            if event['stage'] in ('pilot_audit','multi','stratified') and event.get('ids'):
                batches.append((event['ids'],list(windows)))
        if len(batches)<4:raise ValueError('Trace ended before pilot plus three acquisition batches')
        fig,axes=plt.subplots(4,1,figsize=(10,6),sharex=True)
        known=set()
        for i,(ax,(ids,windows)) in enumerate(zip(axes,batches[:4])):
            for lo,hi in windows:ax.axvspan(lo,hi,alpha=.12,color='#456b87')
            old=sorted(known)
            if old:ax.scatter(rank_of[old],[labels[j] for j in old],s=8,color='#777777',label='Earlier labels')
            ax.scatter(rank_of[ids],[labels[j] for j in ids],s=12,color='#be4634',label='New labels')
            ax.set(yticks=[0,1],ylim=(-.2,1.2),ylabel='Pilot' if i==0 else f'Batch {i}')
            known.update(ids)
        axes[0].legend(loc='upper right');axes[-1].set_xlabel('Encoder rank')
        _save(fig,output,'sas_multi_process',plt)
    else:
        raise ValueError(f'Unknown paper experiment: {experiment}')
