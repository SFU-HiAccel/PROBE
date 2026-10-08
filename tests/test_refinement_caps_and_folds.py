"""Knobs for the appendix sensitivity study: the two iteration caps T1, T2 and the fold count R.

T1 and T2 cap the two refinement phases independently (they shared one `num_iter` before), and R
is the cross-fitting fold count of the calibration stage, which was pinned to the dataclass default
because run.py never passed it. Each run has to report what it actually used: both phases exit early
on convergence, and R is clipped by the rarer class, so the requested value is not the value that ran.
"""
import inspect
import os

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import cluster_proxy
import learned_calibration as lc


def _ring_graph(n: int = 60) -> tuple[np.ndarray, csr_matrix, np.ndarray, np.ndarray]:
    """A ring so that information needs many hops to cross: the caps bite."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, 4))
    rows = np.repeat(np.arange(n), 2)
    cols = np.concatenate([[(i - 1) % n, (i + 1) % n] for i in range(n)])
    W = csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    labeled = np.array([0, 1, 2, n // 2, n // 2 + 1, n // 2 + 2], dtype=np.int64)
    values = np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)
    return emb, W, labeled, values


def test_caps_are_separate_and_reported():
    emb, W, li, lv = _ring_graph()
    _, _, prof = cluster_proxy.predict_with_score_refinement(
        emb, W, li, lv, confidence_threshold=0.5, num_iter=50,
        num_iter_bias=3, num_iter_prop=7,
    )
    assert prof["bias_diffusion_iters"] == 3
    assert prof["homogeneity_propagation_iters"] == 7
    assert (prof["bias_diffusion_cap"], prof["homogeneity_propagation_cap"]) == (3, 7)


def test_caps_default_to_num_iter():
    emb, W, li, lv = _ring_graph()
    _, _, prof = cluster_proxy.predict_with_score_refinement(
        emb, W, li, lv, confidence_threshold=0.5, num_iter=4,
    )
    assert (prof["bias_diffusion_cap"], prof["homogeneity_propagation_cap"]) == (4, 4)
    assert prof["bias_diffusion_iters"] <= 4 and prof["homogeneity_propagation_iters"] <= 4


def test_early_exit_still_wins_below_the_cap():
    """A high cap must not force iterations: the recorded count is convergence, not the cap."""
    emb, W, li, lv = _ring_graph()
    _, _, prof = cluster_proxy.predict_with_score_refinement(
        emb, W, li, lv, confidence_threshold=0.5, num_iter=500,
    )
    assert prof["bias_diffusion_iters"] < 500
    assert prof["homogeneity_propagation_cap"] == 500


@pytest.mark.parametrize("fn", ["cluster_proxy_pipeline", "_run_learned_calibration_stage",
                                "_compute_oof_z_smooth_and_corrected"])
def test_caps_are_threaded_through_the_pipeline(fn):
    params = inspect.signature(getattr(cluster_proxy, fn)).parameters
    assert "refine_t1" in params and "refine_t2" in params


def test_oof_reports_the_fold_count_it_could_use():
    """R is clipped by the rarer class; the clipped value is what ran."""
    idx = np.arange(20, dtype=np.int64)
    y = np.array([1, 1, 1] + [0] * 17, dtype=np.int32)

    def refine_fn(train_idx, train_y):
        return np.zeros(20, dtype=bool), np.zeros(20, dtype=np.float64)

    oof = lc.compute_oof_features(refine_fn, idx, y, n_folds=5, seed=0)
    assert oof.reason == ""
    assert oof.n_folds_used == 3          # min(5, 3 positives)

    oof10 = lc.compute_oof_features(refine_fn, idx, np.concatenate([np.ones(10, np.int32),
                                                                    np.zeros(10, np.int32)]),
                                    n_folds=10, seed=0)
    assert oof10.n_folds_used == 10

    too_few = lc.compute_oof_features(refine_fn, idx[:4], np.array([1, 0, 0, 0], np.int32),
                                      n_folds=5, seed=0)
    assert too_few.reason == "too_few_labels_per_class" and too_few.n_folds_used == 0


def test_fold_count_is_a_run_knob(monkeypatch):
    import run
    assert "lc_n_folds_used" in lc.LEARNED_CALIBRATION_RESULT_KEYS
    monkeypatch.setattr(run, "LC_FOLDS", 3)
    assert run._learned_calibration_config(10000, 10).n_folds == 3


def test_env_sets_the_caps_and_the_folds():
    env = {"LASO_REFINE_T1": "12", "LASO_REFINE_T2": "4", "LASO_LC_FOLDS": "3"}
    src = open(os.path.join(os.path.dirname(__file__), "..", "run.py")).read()
    for k in env:
        assert k in src, f"{k} is not read by run.py"
