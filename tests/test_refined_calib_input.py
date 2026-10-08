"""Calibrator input = graph-refined score (calib_input="refined") instead of the LR proxy logit."""
import numpy as np
import pytest
from scipy.special import expit, logit

from calibration_audit import run_audit
from learned_calibration import LearnedCalibrationConfig, OofFeatures, refined_logit, run_learned_calibration
from weighted_calibrator import predict_proba


def _world(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 2.0, n)                                   # LR proxy logit
    z = np.clip(expit(0.7 * x - 0.5) + rng.normal(0, 0.05, n), 0, 1)   # refined score, a different ordering
    y = (rng.random(n) < expit(1.3 * x - 1.0)).astype(np.int32)
    h = (z > 0.5).astype(float)
    labeled = np.sort(rng.choice(n, 150, replace=False))
    return x, z, h, y, labeled


def _run(calib_input, calibrator="platt", alpha=0.5):
    x, z, h, y, lab = _world()
    cfg = LearnedCalibrationConfig(total_budget=400, seed=1, calibrator=calibrator, alpha=alpha,
                                   route_unlimited=True, calib_input=calib_input)
    x_cal = refined_logit(z) if calib_input == "refined" else None
    res = run_learned_calibration(x=x, h=h, labeled_indices=lab, labeled_values=y[lab],
                                  oof=OofFeatures(None, None, "skipped"), label_rows=lambda r: y[r],
                                  config=cfg, x_cal=x_cal)
    return res, x, z, h, y, lab, cfg


def test_refined_logit_clips_pinned_scores():
    v = refined_logit(np.array([0.0, 0.5, 1.0]))
    assert np.all(np.isfinite(v)) and v[1] == pytest.approx(0.0) and v[0] < -10 and v[2] > 10


def test_unknown_calib_input_is_rejected():
    with pytest.raises(ValueError):
        LearnedCalibrationConfig(total_budget=100, calib_input="graph")


def test_stratified_sample_is_still_drawn_on_the_proxy_score():
    proxy, *_ = _run("proxy")
    refined, *_ = _run("refined")
    assert np.array_equal(np.sort(proxy.calib_idx), np.sort(refined.calib_idx))


def test_uncalibrated_arm_routes_on_the_refined_score():
    res, x, z, h, *_ = _run("refined", calibrator="none")
    assert np.allclose(predict_proba(res.params, refined_logit(z), h), expit(refined_logit(z)))
    assert res.diagnostics["lc_calib_input"] == "refined"


def test_routing_follows_the_refined_score():
    res, x, z, h, y, lab, _ = _run("refined")
    assert res.status.startswith("routed")
    pool = np.setdiff1d(np.arange(x.size), np.concatenate([lab, res.calib_idx]))
    # among predicted negatives, every routed row has a refined score at least as high as any unrouted one
    neg = pool[h[pool] == 0]
    routed = np.intersect1d(neg, res.verify_idx)
    if routed.size and routed.size < neg.size:
        assert z[routed].min() >= z[np.setdiff1d(neg, routed)].max() - 1e-12


def test_audit_records_the_calibrator_input(tmp_path):
    res, x, z, h, y, lab, cfg = _run("refined")

    class _Request:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def __call__(self, key):
            return int(y[key]), {}

    class _Session:
        request_factory = _Request

    diag = run_audit(result=res, x=x, h=h, labeled_indices=lab, labeled_values=y[lab], session=_Session(),
                     config=cfg, out_dir=str(tmp_path), n=300, x_cal=refined_logit(z))
    d = np.load(diag["lc_audit_path"])
    assert np.allclose(d["x_cal"], refined_logit(z))
    assert np.allclose(d["p_cal"], predict_proba(res.params, refined_logit(z), h))
    assert str(d["meta_calib_input"]) == "refined"
