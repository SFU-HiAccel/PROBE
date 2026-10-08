"""Sampler ablation: with refinement off, every sampler must reach the same LR-only predictor."""
import numpy as np
import pytest
import cluster_proxy as cp
from fixed_ce_sampling import SamplingConfig


@pytest.mark.parametrize('sampler', ['activeSampling', 'random', 'stratified'])
def test_refine_off_is_lr_only_for_every_sampler(monkeypatch, sampler):
    monkeypatch.setattr(cp, '_KNN_CACHE_ENABLED', False)
    rng = np.random.default_rng(3)
    emb = rng.normal(size=(200, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    def label(batch):
        return np.array([int(emb[int(i), 0] > 0.2) for i in batch])

    def forbidden(*args, **kwargs):
        raise AssertionError('graph label propagation reached with refinement off')
    monkeypatch.setattr(cp, 'label_propagation_cutssl', forbidden)

    _, profile = cp.cluster_proxy_pipeline(
        emb, list(map(str, range(200))), lambda: label, 40,
        sampling_method=sampler, similarities=emb[:, 0].copy(),
        fixed_ce_scores=emb[:, 0].copy(), fixed_ce_config=SamplingConfig(seed=3), knn_k=5, refine_enabled=False, defer_enabled=False,
        lr_class_weight_override='balanced',
    )
    assert profile['mode'] == 'lr_only'
    assert profile['propagation_time'] == 0.0
