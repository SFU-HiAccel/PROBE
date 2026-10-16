"""Post-execution evaluation data for paper statistics, never acquisition labels."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

_population = None
_calibration = None


def remember_calibration(result, scores, h, labeled_indices):
    """Retain scores on rows outside fitting labels for post-run ECE."""
    global _calibration
    raw = os.environ.get('PROBE_PAPER_JOB')
    if not raw or json.loads(raw)['experiment'] != 'calibration': return
    if result.params is None or scores is None: return
    from weighted_calibrator import predict_proba
    params = result.params
    probabilities = params.predict(scores,h) if hasattr(params,'predict') else predict_proba(params,scores,h)
    eligible = np.ones(len(scores),bool)
    eligible[np.asarray(labeled_indices,int)] = False
    eligible[result.calib_idx] = False
    _calibration = np.asarray(probabilities), eligible


def remember_population(factory, prediction):
    """Retain a lazy candidate frame without evaluating annotations during execution."""
    global _population
    raw = os.environ.get('PROBE_PAPER_JOB')
    _population = None
    if raw and json.loads(raw)['experiment'] in ('main', 'trace', 'calibration'):
        _population = (factory, np.asarray(prediction, dtype=bool).copy())


def _keys(frame):
    def value(v):
        text = str(v).strip()
        # Spark numeric IDs can be rendered as floats when reading CSV inputs.
        if text.endswith('.0') and text[:-2].isdigit(): text = text[:-2]
        return text
    return [json.dumps([value(v) for v in row], ensure_ascii=False) for row in frame.itertuples(index=False, name=None)]


def export_evaluation(query, projection, prediction_csv, gt_path, measured_quality, *,
                      mmqa_truth=None, annotation_source=None):
    """Save scorer-aligned predictions and human truth after timing has stopped.

    Set-retrieval scorers operate on distinct projected keys. Animals Q1
    uses physical row IDs. The exact unresampled
    quality must match the query scorer before the export is admitted.
    """
    global _population, _calibration
    if _population is None: return None
    import pandas as pd
    factory, raw_prediction = _population; _population = None
    population = factory()
    frame = projection(population).toPandas()
    if len(frame)!=len(raw_prediction):
        raise ValueError('Evaluation projection must retain every physical candidate row')
    predicted = pd.read_csv(prediction_csv, dtype=str).fillna('')
    sources = {}
    if gt_path and mmqa_truth is None:
        sources[str(Path(gt_path).name)] = hashlib.sha256(Path(gt_path).read_bytes()).hexdigest()
    if annotation_source:
        sources[str(Path(annotation_source).name)] = hashlib.sha256(Path(annotation_source).read_bytes()).hexdigest()
    physical_keys = _keys(frame)
    ids = np.asarray(sorted(set(physical_keys)))
    predicted_keys = set(_keys(predicted))
    if mmqa_truth is not None:
        positive_keys = set(_keys(pd.DataFrame(sorted(mmqa_truth))))
        sources['embedded_reference'] = hashlib.sha256(json.dumps(sorted(mmqa_truth)).encode()).hexdigest()
    else:
        gt = pd.read_csv(gt_path, dtype=str).fillna('')
        # Main retrieval scorers read the first reference column.
        positive_keys = set(_keys(gt.iloc[:, :1]))
    if positive_keys - set(ids):
        raise ValueError(f'{query}: reference keys outside evaluated population')
    if predicted_keys - set(ids):
        raise ValueError(f'{query}: predicted keys outside evaluated population')
    truth = np.asarray([key in positive_keys for key in ids], bool)
    physical_truth = np.asarray([key in positive_keys for key in physical_keys], bool)
    prediction = np.asarray([key in predicted_keys for key in ids], bool)
    metric = 'binary_f1'
    denominator = int(truth.sum()+prediction.sum())
    quality = 2 * np.count_nonzero(truth & prediction)/denominator if denominator else 0.
    if not np.isclose(quality, measured_quality, rtol=0, atol=1e-9):
        raise ValueError(f'{query}: evaluation export quality {quality} differs from benchmark scorer {measured_quality}')
    job = json.loads(os.environ['PROBE_PAPER_JOB'])
    output = Path(os.environ['PROBE_PAPER_OUTPUT'])
    np.savez_compressed(output/'evaluation.npz', ids=ids, truth=truth, prediction=prediction,
                        metric=metric, physical_N=len(raw_prediction))
    if job['experiment'] == 'trace':
        if len(physical_keys) != len(raw_prediction):
            raise ValueError('Trace annotation keys must preserve every physical candidate')
        np.savez_compressed(output/'candidate_truth.npz',
            truth=np.asarray([key in positive_keys for key in physical_keys], bool),
            ids=np.arange(len(physical_keys)))
    metadata = dict(query=query, metric=metric, physical_N=len(raw_prediction), evaluation_N=len(ids),
        seed=job['seed'], quality=quality, sources=sources,
        unit='distinct projected query-result keys (physical row IDs for Animals Q1)')
    if job['experiment']=='calibration' and job['system']=='PROBE':
        if _calibration is not None:
            probabilities, eligible = _calibration
            if len(probabilities)!=len(physical_truth):
                raise ValueError('Calibration probabilities do not align with evaluation population')
            metadata['ece'] = expected_calibration_error(physical_truth[eligible],probabilities[eligible]) if eligible.any() else None
            metadata['ece_N'] = int(eligible.sum())
            metadata['ece_bins'] = 10
            metadata['ece_target'] = 'human annotations on physical rows outside L and A'
            np.savez_compressed(output/'calibration_evaluation.npz',truth=physical_truth,
                                probabilities=probabilities,eligible=eligible)
        else:
            metadata['ece'] = None
            metadata['ece_status'] = 'no fitted calibrator (fallback or fully verified population)'
    _calibration = None
    (output/'evaluation.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def expected_calibration_error(truth, probabilities, bins=10):
    y,p = np.asarray(truth),np.asarray(probabilities)
    if y.shape!=p.shape or y.ndim!=1 or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('ECE needs aligned binary truth and finite probabilities')
    indices = np.minimum((p*bins).astype(int),bins-1)
    return float(sum(abs(y[indices==b].mean()-p[indices==b].mean())*np.count_nonzero(indices==b)/len(y)
                     for b in range(bins) if np.any(indices==b)))
