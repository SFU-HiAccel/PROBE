"""Paper Section 6 bootstrap with paired seeds and within-query class strata.

Callers must supply measured predictions aligned with human annotations, and
the benchmark's scorer. Aggregate and semantic-set queries require their own
scorers; a binary-candidate F1 cannot replace them. This module never acquires
labels or supplies missing runs.
"""
import numpy as np


def binary_f1(truth, prediction, rows):
    y = truth[rows]
    p = prediction[rows]
    tp = np.count_nonzero(y & p)
    denominator = np.count_nonzero(y) + np.count_nonzero(p)
    return 2. * tp / denominator if denominator else 0.


def count_quality(truth, prediction, rows):
    actual = int(np.count_nonzero(truth[rows]))
    return 1. - min(abs(int(np.count_nonzero(prediction[rows]))-actual)/max(actual, 1e-12), 1.)


from experiments.evaluation_export import expected_calibration_error


def paired_stratified_bootstrap(truth, predictions, *, scorer, n_boot=10000, seed=0,
                                seed_draws=None):
    """Return draws of each arm's mean quality for one query with five seeds.

    ``predictions`` maps arm names to bool arrays (5, N), with identically
    ordered seeds and candidates. Each replicate shares the seed draws and
    positive/negative row draws across all arms, preserving pairing. Scorers
    receive (truth, one seed's predictions, resampled original row indices).
    """
    y = np.asarray(truth)
    if y.ndim != 1 or not y.size or not np.isin(y, [0, 1]).all():
        raise ValueError('Human truth must be a nonempty aligned binary array')
    if not predictions or n_boot < 2:
        raise ValueError('At least one arm and two bootstrap replicates are required')
    arms = {}
    for name, values in predictions.items():
        p = np.asarray(values)
        if p.shape != (5, y.size) or not np.isin(p, [0, 1]).all():
            raise ValueError('Every arm requires five measured seeds and the same candidate universe')
        arms[name] = p.astype(bool)
    y = y.astype(bool)
    strata = [np.flatnonzero(y), np.flatnonzero(~y)]
    rng = np.random.default_rng(seed)
    if seed_draws is None:
        seed_draws = rng.integers(0, 5, size=(n_boot, 5))
    seed_draws = np.asarray(seed_draws)
    if seed_draws.shape != (n_boot,5) or not np.isin(seed_draws, range(5)).all():
        raise ValueError('Seed draws must be a (replicates,5) array of seed positions')
    draws = {name:np.empty(n_boot) for name in arms}
    if scorer in (binary_f1, count_quality):
        # Equal prediction patterns are interchangeable in the empirical
        # bootstrap. Multinomial pattern counts give the exact row-resampling
        # distribution without materializing N row indices per replicate.
        matrix = np.concatenate(list(arms.values()), axis=0)
        patterns = []
        for indices in strata:
            packed = np.packbits(matrix[:,indices].T, axis=1)
            unique, counts = np.unique(packed, axis=0, return_counts=True)
            bits = np.unpackbits(unique, axis=1)[:,:matrix.shape[0]].astype(float)
            patterns.append((bits, counts, len(indices)))
        for b, seeds in enumerate(seed_draws):
            totals = []
            for bits, counts, size in patterns:
                sampled = rng.multinomial(size, counts/size) if size else np.zeros(len(counts))
                totals.append(sampled @ bits)
            tp, fp = (v.reshape(len(arms),5) for v in totals)
            predicted = tp+fp
            if scorer is binary_f1:
                denominator = y.sum()+predicted
                quality = np.divide(2*tp,denominator,out=np.zeros_like(tp),where=denominator>0)
            else:
                quality = 1-np.minimum(np.abs(predicted-y.sum())/max(int(y.sum()),1e-12),1)
            for i,name in enumerate(arms):draws[name][b]=quality[i,seeds].mean()
        return draws
    for b in range(n_boot):
        seeds = seed_draws[b]
        rows = np.concatenate([rng.choice(s, s.size, replace=True) for s in strata])
        for name, p in arms.items():
            scores = [float(scorer(y, p[s], rows)) for s in seeds]
            if not np.all(np.isfinite(scores)) or min(scores) < 0 or max(scores) > 1:
                raise ValueError('Benchmark scorer must return finite quality in [0,1]')
            draws[name][b] = np.mean(scores)
    return draws


def bootstrap_summary(draws, *, focus='PROBE'):
    """95% percentile CIs; significance means the paired difference excludes zero."""
    if focus not in draws:
        raise ValueError('Focus arm is missing')
    arrays = {name:np.asarray(d, dtype=float) for name,d in draws.items()}
    shape = arrays[focus].shape
    if len(shape) != 1 or shape[0] < 2 or any(
        a.shape != shape or not np.all(np.isfinite(a)) for a in arrays.values()):
        raise ValueError('Paired bootstrap draws must have equal finite nonempty shapes')
    result = {}
    for name, d in arrays.items():
        lo, hi = np.quantile(d, [.025, .975])
        row = dict(ci95=[float(lo), float(hi)], method='paired_seed_class_stratified_bootstrap')
        if name != focus:
            lo, hi = np.quantile(arrays[focus]-d, [.025, .975])
            row.update(focus_difference_ci95=[float(lo), float(hi)],
                       significant=bool(lo > 0 or hi < 0))
        result[name] = row
    return result


def evidence_identity(records, manifest):
    import hashlib
    import json
    main = [r for r in records if r['experiment']=='main']
    return dict(protocol=manifest['protocol'],
                runs={r['raw_record']:r['raw_sha256'] for r in main},
                analysis_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())


def validate_report(report, records, manifest):
    if report.get('evidence') != evidence_identity(records, manifest):
        raise ValueError('Bootstrap report does not match the selected runs, protocol or analysis code')


def build_report(root, records, manifest, *, n_boot=10000, seed=0):
    import json
    from pathlib import Path
    root = Path(root)
    spec = manifest['experiments']['main']
    seeds = spec['seeds']
    if len(seeds)!=5:
        raise ValueError('Main statistics require five seeds')
    seed_draws = np.random.default_rng(seed).integers(0,5,size=(n_boot,5))
    draws, query_reports, rare = {}, {}, []
    for q_index, query in enumerate(spec['queries']):
        predictions = {}
        universe = truth = metric = sources = None
        means = {}
        for system in spec['systems']:
            arm = sorted([r for r in records if r['experiment']=='main' and
                          r['query']==query and r['system']==system],key=lambda r:r['seed'])
            if [r['seed'] for r in arm] != sorted(seeds):
                raise ValueError(f'Missing or duplicate five-seed main evidence: {query}/{system}')
            identities = {json.dumps((r.get('point'),r.get('identity')),sort_keys=True) for r in arm}
            if len(identities)!=1:
                raise ValueError(f'Incompatible five-seed variant identities: {query}/{system}')
            values = []
            for record in arm:
                completion = root/record['raw_record']
                raw = json.loads(completion.read_text())
                paths = [completion.parent/p for p in raw['outputs'] if p.endswith('/evaluation.npz')]
                if len(paths)!=1:
                    raise ValueError(f'Missing evaluation arrays: {completion}')
                path = paths[0]
                meta_name = str(path.with_suffix('.json').relative_to(completion.parent))
                if meta_name not in raw['outputs']:
                    raise ValueError('Evaluation metadata is not hashed in the completed record')
                metadata = json.loads(path.with_suffix('.json').read_text())
                with np.load(path, allow_pickle=False) as data:
                    ids, y, pred, kind = data['ids'], data['truth'], data['prediction'], str(data['metric'])
                if universe is None:
                    universe,truth,metric,sources = ids,y,kind,metadata['sources']
                if not (np.array_equal(universe,ids) and np.array_equal(truth,y)
                        and kind==metric and metadata['sources']==sources):
                    raise ValueError(f'Mismatched candidate or annotation identity: {query}/{system}')
                if kind not in ('binary_f1','count_quality'):
                    raise ValueError(f'Unsupported evaluation metric: {kind}')
                scorer = binary_f1 if kind=='binary_f1' else count_quality
                if not np.isclose(scorer(y,pred,np.arange(len(y))),record['f1'],atol=1e-9,rtol=0):
                    raise ValueError('Evaluation array does not reproduce the measured query scorer')
                values.append(pred)
            predictions[system] = np.asarray(values)
            means[system] = float(np.mean([r['f1'] for r in arm]))
        draws[query] = paired_stratified_bootstrap(truth,predictions,scorer=scorer,
            n_boot=n_boot,seed=seed+q_index+1,seed_draws=seed_draws)
        summary = bootstrap_summary(draws[query])
        for name, row in summary.items():
            row.update(mean=means[name],metric=metric,positive_count=int(truth.sum()),
                       evaluation_N=len(truth))
        query_reports[query] = summary
        if truth.sum()<20:
            rare.append(query)
    groups = {}
    for name, queries in [('all_queries',spec['queries']),('fewer_than_20_positives',rare)]:
        if not queries:
            continue
        values = {system:np.prod([draws[q][system] for q in queries],axis=0)**(1/len(queries))
                  for system in spec['systems']}
        groups[name] = dict(queries=queries,summary=bootstrap_summary(values),
                            metric='geometric mean of benchmark query quality')
        for system in spec['systems']:
            groups[name]['summary'][system]['mean'] = float(np.prod([
                query_reports[q][system]['mean'] for q in queries])**(1/len(queries)))
    return dict(evidence=evidence_identity(records,manifest),replicates=n_boot,seed=seed,
                queries=query_reports,groups=groups,
                method='paired seed and positive/negative evaluation-unit percentile bootstrap')


def main():
    import argparse
    import json
    from pathlib import Path
    from experiments.report import load_runs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/paper'))
    parser.add_argument('--output',type=Path,default=Path('results/figures/bootstrap.json'))
    parser.add_argument('--replicates',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('paper_manifest.json').read_text())
    try:
        records = load_runs(args.input,manifest)
        report = build_report(args.input,records,manifest,n_boot=args.replicates,seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':
    main()
