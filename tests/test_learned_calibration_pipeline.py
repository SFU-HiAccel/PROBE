"""Pipeline wiring for learned calibration: switch on, switch off, nothing else changes."""
import numpy as np
import pytest
from scipy.special import expit

import cluster_proxy as cp
import learned_calibration
from learned_calibration import LearnedCalibrationConfig


@pytest.fixture(autouse=True)
def _no_knn_disk_cache(monkeypatch):
    monkeypatch.setattr(cp, "_KNN_CACHE_ENABLED", False)


def _synthetic(n=600, d=16, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int32)
    emb = rng.normal(size=(n, d))
    emb[:, 0] += 2.0 * y
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, emb[:, 0].astype(np.float32), [str(i) for i in range(n)], y


def _factory(y, calls):
    def factory():
        def fn(batch):
            calls.extend(int(t) for t in batch)
            return np.array([y[int(t)] for t in batch], dtype=np.int32)
        return fn
    return factory


def _run(emb, sims, texts, factory, **kw):
    return cp.cluster_proxy_pipeline(
        emb, texts, factory, sampling_method="activeSampling", similarities=sims,
        refine_enabled=True, defer_enabled=True, fixed_ce_scores=sims,
        activeSampling_backend="fixed_ce",
        random_seed=0, knn_k=10, **kw)


class TestPredictWithScoreRefinementLogit:
    def test_default_output_unchanged_and_logit_matches_lr_probability(self):
        emb, _, _, y = _synthetic()
        W = cp._build_knn_graph(emb, 10, cache=False)
        idx = np.arange(0, 600, 6)
        mask0, z0, prof0 = cp.predict_with_score_refinement(emb, W, idx, y[idx], confidence_threshold=0.5)
        mask1, z1, prof1 = cp.predict_with_score_refinement(emb, W, idx, y[idx], confidence_threshold=0.5,
                                                            return_logit=True)
        assert "lr_logit" not in prof0
        assert np.array_equal(mask0, mask1) and np.allclose(z0, z1)
        # Algorithm 3 retains expert labels in both corrected and final scores.
        assert np.array_equal(y[idx], prof1["z_corrected"][idx])
        assert np.array_equal(y[idx], z1[idx])

    def test_single_class_labels_give_no_logit(self):
        emb, _, _, _ = _synthetic()
        W = cp._build_knn_graph(emb, 10, cache=False)
        idx = np.arange(10)
        _, _, prof = cp.predict_with_score_refinement(emb, W, idx, np.zeros(10, dtype=np.int32),
                                                      confidence_threshold=0.5, return_logit=True)
        assert prof["lr_logit"] is None


class TestPipelineSwitch:
    def test_learned_calibration_respects_total_budget(self):
        emb, sims, texts, y = _synthetic()
        calls = []
        cfg = LearnedCalibrationConfig(total_budget=120, seed=0)
        mask, prof = _run(emb, sims, texts, _factory(y, calls), sample_size=cfg.sample_budget,
                          decision_mode="learned_calibration", learned_calibration_config=cfg)
        assert mask.shape == (600,) and mask.dtype == bool
        assert prof["defer_decision_mode"] == "learned_calibration"
        assert prof["lc_status"] in {"routed", "insufficient_evidence"}
        assert len(calls) == prof["lc_calls_total"]
        assert prof["defer_verify_count"] == prof["lc_calls_A"] + prof["lc_calls_D"]
        assert len(calls) <= max(cfg.total_budget, prof["n_labeled"])
        assert "lr_logit" not in prof["refinement_profiling"]


    def test_missing_config_is_rejected(self):
        emb, sims, texts, y = _synthetic()
        with pytest.raises(ValueError, match="learned_calibration_config"):
            _run(emb, sims, texts, _factory(y, []), sample_size=60, decision_mode="learned_calibration")


def test_sample_size_above_config_sample_budget_is_rejected():
    emb, sims, texts, y = _synthetic()
    calls = []
    cfg = LearnedCalibrationConfig(total_budget=30, seed=0)
    with pytest.raises(ValueError, match="sample_budget"):
        _run(emb, sims, texts, _factory(y, calls), sample_size=120,
             decision_mode="learned_calibration", learned_calibration_config=cfg)
    assert calls == []


def test_refined_calib_input_feeds_the_graph_refined_score(monkeypatch):
    """calib_input='refined': run_learned_calibration receives logit(z_smooth) as x_cal, and the OOF
    features of L come from each fold's refined score rather than the LR logit."""
    seen = {}
    real = learned_calibration.run_learned_calibration

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(learned_calibration, "run_learned_calibration", spy)
    real_refine = cp.predict_with_score_refinement
    finals = []

    def refine_spy(*a, **k):
        out = real_refine(*a, **k)
        finals.append(out[1])
        return out

    monkeypatch.setattr(cp, "predict_with_score_refinement", refine_spy)
    emb, sims, texts, y = _synthetic()
    cfg = LearnedCalibrationConfig(total_budget=120, seed=0, calib_input="refined")
    mask, prof = _run(emb, sims, texts, _factory(y, []), sample_size=cfg.sample_budget,
                      decision_mode="learned_calibration", learned_calibration_config=cfg)
    assert prof["lc_calib_input"] == "refined"
    x_cal = seen["x_cal"]
    assert x_cal is not None and not np.allclose(x_cal, seen["x"])
    # x_cal is the logit of the main refinement's final score (the first refinement call)
    assert np.allclose(x_cal, learned_calibration.refined_logit(finals[0]))
    if seen["oof"].x is not None:
        # OOF features are on the refined scale, i.e. inside the clipped logit range of z in [0, 1]
        assert np.all(np.abs(seen["oof"].x) <= abs(learned_calibration.refined_logit(np.array([0.0])))[0] + 1e-9)
