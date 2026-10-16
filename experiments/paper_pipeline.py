"""Paper experiment execution on the public candidate universe and real oracle.

No ground truth enters this module. Frozen ablations reuse acquired expert labels,
with shared preparation provenance and per-attempt request counts kept distinct.
"""
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
import time

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class TimedOracle:
    def __init__(self, oracle):
        self.oracle = oracle
        self.seconds = 0.

    def label_ids(self, ids, *, stage='L'):
        from oracle_budget import binary_labels_or_raise
        started = time.perf_counter()
        try:
            return binary_labels_or_raise(self.oracle.label_ids(list(map(int, ids)), stage=stage))
        finally:
            self.seconds += time.perf_counter() - started

    def __call__(self, batch):
        started = time.perf_counter()
        try:
            from oracle_budget import binary_labels_or_raise
            return binary_labels_or_raise(self.oracle(batch))
        finally:
            self.seconds += time.perf_counter() - started

    @property
    def ledger(self):
        return self.oracle.ledger


def acquire(features, scores, texts, oracle, budget, sampler, config):
    from fixed_ce_sampling import sample_fixed_ce
    from cluster_proxy import sample_stratified_by_similarity
    n = len(texts)
    if sampler == 'activeSampling':
        result = sample_fixed_ce(range(n), scores, lambda i: oracle.label_ids([i])[0], budget,
                                 config, batch_oracle=lambda ids: oracle.label_ids(ids))
        if result.failed_ids:
            raise RuntimeError(f'Sampling has {len(result.failed_ids)} unresolved expert answers')
        ids = np.asarray(list(result.labels), dtype=int)
        return ids, np.asarray([result.labels[i] for i in ids], dtype=int), dict(
            trace=result.trace, order=result.order, sampler='fixed_ce', config=vars(config),
            pilot_state=vars(result.pilot), stop_reason=result.stop_reason)
    if sampler == 'stratified':
        ids, labels, profile = sample_stratified_by_similarity(
            scores, texts, oracle, budget, 10, .3, 'neyman')
        return ids, labels, profile
    if sampler == 'random':
        ids = np.random.default_rng(config.seed).choice(n, min(n, budget), replace=False)
    elif sampler == 'top_k':
        ids = np.argsort(-np.asarray(scores), kind='stable')[:budget]
    else:
        raise ValueError(f'Unsupported sampler: {sampler}')
    return ids, oracle.label_ids(ids), dict(sampler=sampler, config=vars(config), trace=[])


def frozen_sample(job, features, scores, texts, oracle, budget, config, output):
    """One locked acquisition per exact label-free input/protocol/model identity."""
    request = oracle.oracle.request_factory() if hasattr(oracle.oracle,'request_factory') else None
    request_config = {k:getattr(request,k,None) for k in (
        'model','base_url','system_prompt','max_tokens','temperature','extra_body','timeout')}
    identity = dict(query=job['query'], seed=job['seed'], budget=budget, request=request_config,
                    model=job['model'], revision=job['model_revision'], oracle_url=job['oracle_url'],
                    features=hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest(),
                    scores=hashlib.sha256(np.asarray(scores).tobytes()).hexdigest(),
                    texts=hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest(),
                    config=vars(config), code={name:hashlib.sha256(
                        (Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                        for name in ('experiments/paper_pipeline.py','fixed_ce_sampling.py',
                                     'sampling_strata.py','stratified_calibration.py',
                                     'threshold_search_multi.py','oracle_budget.py','proxy_model.py')})
    identity = json.loads(json.dumps(identity, sort_keys=True))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(os.environ['PROBE_FROZEN_ROOT']) / key
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record_path = root / 'sample.json'
        reused = record_path.exists()
        if reused:
            record = json.loads(record_path.read_text())
            if record['identity'] != identity:
                raise ValueError('Frozen acquisition identity mismatch')
            ids, labels = np.asarray(record['ids'], int), np.asarray(record['labels'], int)
            if len(ids) != len(labels) or len(set(ids.tolist())) != len(ids) or not np.isin(labels, [0, 1]).all() or np.any((ids < 0) | (ids >= len(texts))):
                raise ValueError('Invalid frozen expert labels')
            digest = hashlib.sha256(json.dumps([record['ids'], record['labels']], sort_keys=True).encode()).hexdigest()
            if digest != record['labels_sha256']:
                raise ValueError('Frozen label digest mismatch')
            # These are purchased labels from this exact acquisition, not GT.
            oracle.ledger.cache.update(dict(zip(ids.tolist(), labels.tolist())))
        else:
            begin = time.perf_counter()
            ids, labels, trace = acquire(features, scores, texts, oracle, budget, 'activeSampling', config)
            record = dict(identity=identity, ids=ids.tolist(), labels=labels.tolist(), trace=trace,
                          seconds=time.perf_counter()-begin, source='actual expert acquisition',
                          oracle=oracle.ledger.snapshot(), events=oracle.ledger.events)
            record['labels_sha256'] = hashlib.sha256(json.dumps([record['ids'], record['labels']], sort_keys=True).encode()).hexdigest()
            tmp = record_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(record, default=json_value, indent=2)+'\n')
            tmp.replace(record_path)
        # Include an immutable copy in each completed job's hashed output set.
        (output / 'frozen_acquisition.json').write_text(json.dumps(record, default=json_value, indent=2)+'\n')
        return ids, labels, dict(record['trace'], frozen_id=key, frozen_reused=reused,
                                 shared_preparation_seconds=record['seconds'])


def execute(features, texts, factory, sample_size, scores, config, calibration_config, *,
            query_embedding=None):
    from fixed_ce_sampling import SamplingConfig
    from cluster_proxy import (_build_knn_graph, auto_knn_k, predict_with_score_refinement,
                               _run_learned_calibration_stage)
    job = json.loads(os.environ['PROBE_PAPER_JOB'])
    point = job['point']
    output = Path(os.environ['PROBE_PAPER_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    oracle = TimedOracle(factory())
    n = len(texts)
    if n != job['paper_N']:
        raise ValueError(f"Candidate mismatch before expert calls: {n} != {job['paper_N']}")
    if features.shape[0] != n or np.asarray(scores).shape != (n,):
        raise ValueError('Representation must align with the full candidate universe')
    config = config or SamplingConfig(seed=job['seed'])
    np.random.seed(config.seed)
    random.seed(config.seed)
    budget = int(calibration_config.total_budget)
    training_budget = calibration_config.sample_budget
    prof = dict(n_total=n, random_seed=job['seed'], result_status='returned',
                paper_experiment=job['experiment'], paper_system=job['system'])
    stages = dict(sampling=0., expert_calls=0., proxy_training=0., proxy_prediction=0.,
                  refinement=0., calibration=0.)
    started = time.perf_counter()
    if job['system'] in ('AIQA', 'CSV', 'LOTUS', 'ScaleDoc'):
        from experiments.paper_baselines import execute_baseline
        from oracle_budget import OracleLedger
        # Baselines have their own allocation policy; the PROBE L+A cap must not
        # silently constrain their full-oracle cascade/fallback paths.
        oracle.oracle.ledger = OracleLedger(n)
        mask, metadata = execute_baseline(job, features, query_embedding, texts, oracle, output)
        prof.update(baseline=metadata, mode=job['system'], n_labeled=len(oracle.ledger.cache))
        stages.update(metadata.get('stages', {}))
    else:
        before_calls = oracle.seconds
        begin = time.perf_counter()
        if job['experiment'] == 'refinement':
            ids, labels, trace = frozen_sample(job, features, scores, texts, oracle,
                                               training_budget, config, output)
        else:
            sampler = point.get('sampler', 'activeSampling')
            ids, labels, trace = acquire(features, scores, texts, oracle, training_budget, sampler, config)
        stages['sampling'] = max(0., time.perf_counter()-begin-(oracle.seconds-before_calls))
        prof.update(n_labeled=len(ids), n_pos_labeled=int(labels.sum()),
                    sampling_method=point.get('sampler', 'activeSampling'),
                    activeSampling_profiling=trace, budget_sample_size=training_budget)
        np.savez_compressed(output/'training_labels.npz', ids=ids, labels=labels)
        if len(np.unique(labels)) < 2:
            if job['experiment'] in ('sampling','refinement','trace') and job.get('ablation_single_class','constant') == 'constant':
                begin = time.perf_counter()
                value = int(labels[0]) if len(labels) else 0
                mask = np.full(n, bool(value))
                stages['proxy_prediction'] = time.perf_counter()-begin
                prof.update(mode='constant_proxy', single_class_policy='constant; routing disabled',
                            constant_value=value, whole_query_fallback=False)
                np.savez_compressed(output/'proxy_scores.npz', scores=np.full(n,value,dtype=float), prediction=mask)
            else:
                remaining = np.setdiff1d(np.arange(n), ids)
                mask = np.zeros(n, bool)
                mask[ids] = labels.astype(bool)
                mask[remaining] = oracle.label_ids(remaining, stage='fallback').astype(bool)
                prof.update(whole_query_fallback=True, mode='S3', defer_verify_count=len(remaining))
        else:
            stage = point.get('stage', 'full')
            lr_only = job['experiment'] == 'sampling' or stage == 'LR'
            begin = time.perf_counter()
            if lr_only:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(C=1., class_weight='balanced', max_iter=1000, random_state=job['seed'])
                lr.fit(features[ids], labels)
                stages['proxy_training'] = time.perf_counter()-begin
                begin = time.perf_counter()
                refined = lr.predict_proba(features)[:, list(lr.classes_).index(1)]
                stages['proxy_prediction'] = time.perf_counter()-begin
                mask = refined >= .5
                graph = None
                rp = {}
            else:
                graph = _build_knn_graph(features, auto_knn_k(n), viz_tag=job['query'])
                stages['refinement'] = time.perf_counter()-begin
                mask, refined, rp = predict_with_score_refinement(
                    features, graph, ids, labels, .5, alpha=.8, beta=.8,
                    num_iter_bias=50, num_iter_prop=0 if stage == 'bias' else 50, tol=0.,
                    imbalance_strategy='balanced', lr_C=1., return_logit=True)
                if stage == 'bias':
                    refined = np.asarray(rp['z_corrected'])
                    mask = refined >= .5
                stages['proxy_training'] = rp['lr_fit_time']
                stages['proxy_prediction'] = rp['lr_predict_time']
                stages['refinement'] += sum(rp[k] for k in (
                    'graph_norm_time','bias_diffusion_time','homogeneity_propagation_time'))
            prof.update(mode='lr_only' if lr_only else stage, refinement_profiling=rp)
            # Ablations stop here. Other PROBE/BARGAIN arms retain the actual
            # refined proxy and share the public calibration/transport routines.
            if job['experiment'] not in ('sampling', 'refinement', 'trace'):
                from dataclasses import replace
                cfg = replace(calibration_config, alpha=point.get('alpha', .5),
                              calibrator=point.get('calibrator', 'platt'))
                begin, before_calls = time.perf_counter(), oracle.seconds
                def labels_for_stage(batch):
                    return oracle(batch)
                labels_for_stage.label_ids = oracle.label_ids
                labels_for_stage.oracle_session = oracle.oracle
                if job['system'].startswith('BARGAIN'):
                    from bargain_refined import run_bargain_on_refined
                    mask = run_bargain_on_refined(refined, oracle, ids, labels,
                                                 seed=job['seed'], profiling=prof)
                else:
                    mask = _run_learned_calibration_stage(
                        features, graph, texts, labels_for_stage, ids, labels, mask, rp['lr_logit'],
                        config=cfg, refined_score=refined, confidence_threshold=.5,
                        diffusion_alpha=.8, propagation_beta=.8, refine_num_iter=50,
                        refine_t1=50, refine_t2=50, refine_tol=0., refine_patience=5,
                        imbalance_strategy='balanced', lr_C=1., base_threshold=None, profiling=prof)
                stages['calibration'] = max(0., time.perf_counter()-begin-(oracle.seconds-before_calls))
            mask[ids] = labels.astype(bool)
            np.savez_compressed(output/'proxy_scores.npz', scores=refined, prediction=mask)
        if job['experiment'] == 'trace':
            (output/'sampling_trace.json').write_text(json.dumps(trace, default=json_value, indent=2)+'\n')
            np.savez_compressed(output/'encoder_ranking.npz', ids=np.arange(n),
                                scores=scores, order=np.argsort(-scores, kind='stable'))
    stages['expert_calls'] = oracle.seconds
    prof.update(paper_stage_seconds=stages, vllm_invocation_time=oracle.seconds,
                paper_pipeline_seconds=time.perf_counter()-started)
    (output/'pipeline.json').write_text(json.dumps(prof, default=json_value, indent=2)+'\n')
    np.savez_compressed(output/'candidate_predictions.npz', ids=np.arange(n), prediction=mask)
    return np.asarray(mask, dtype=bool), prof
