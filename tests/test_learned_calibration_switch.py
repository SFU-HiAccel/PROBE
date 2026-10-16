"""The learned-calibration switch: plumbing and run.py label mapping."""
import inspect

import pytest

import proxy_model
import sem_ops_register


def test_public_seed_defaults_and_calibration_resolution_agree():
    for fn in (proxy_model.proxy_filter, proxy_model.proxy_join,
               sem_ops_register.sem_filter_impl, sem_ops_register.sem_join_impl):
        assert inspect.signature(fn).parameters["random_seed"].default == 0
    sample, mode, config, defer = proxy_model._resolve_paper_operator(
        100, 10, None, "activeSampling", "fixed_ce", None, None, False, seed=9)
    assert (sample, mode, config.seed, defer) == (6, "learned_calibration", 9, True)


@pytest.mark.parametrize("fn", [
    proxy_model.proxy_filter, proxy_model.proxy_join,
    sem_ops_register.sem_filter_impl, sem_ops_register.sem_join_impl,
])
def test_config_parameter_defaults_to_off_and_is_forwarded(fn):
    param = inspect.signature(fn).parameters["learned_calibration_config"]
    assert param.default is None
    assert "learned_calibration_config=learned_calibration_config" in inspect.getsource(fn)


import run  # noqa: E402  (imports pyspark; ~3 s)


def test_existing_labels_keep_their_decision_modes():
    with pytest.raises(ValueError,match="Retired"):
        run._decision_mode_for_ablation("activeSampling+refine+learned_deferral")
    with pytest.raises(ValueError,match="Retired"):
        run._decision_mode_for_ablation("activeSampling+refine+dasp_topk")
    assert run._decision_mode_for_ablation("activeSampling+refine+trust_proxy") == "none"
    assert run._decision_mode_for_ablation("activeSampling+lr_only") == "none"


def test_default_grid_enables_paper_calibration():
    assert all(run._decision_mode_for_ablation(label) == "learned_calibration"
               for label, _, _ in run.ABLATION_CONFIGS)


def test_new_label_selects_learned_calibration():
    label = "activeSampling+refine+learned_calibration"
    assert run._parse_ablation_label(label) == (label, "activeSampling", True)
    assert run._decision_mode_for_ablation(label) == "learned_calibration"
    assert run._deferral_policy_for_ablation(label) == "learned_calibration"
    assert run._cascade_decision_enabled("learned_calibration") is True


def test_config_uses_total_budget_semantics(monkeypatch):
    monkeypatch.setattr(run, "LC_SAMPLE_FRAC", 0.6)
    monkeypatch.setattr(run, "LC_CALIB_FRAC", 0.1)
    monkeypatch.setattr(run, "RANDOM_SEED", 7)
    cfg = run._learned_calibration_config(10000, 5)
    assert (cfg.total_budget, cfg.sample_budget, cfg.calib_budget, cfg.seed) == (500, 300, 200, 7)


def test_existing_budget_split_is_untouched():
    assert run.split_total_budget(10000, 5, 0.3) == (500, 500, 150)


@pytest.mark.parametrize("fn_name", ["run_experiment_filter", "run_experiment_join"])
def test_experiment_runners_pass_the_config(fn_name):
    src = inspect.getsource(getattr(run, fn_name))
    assert "learned_calibration_config=_lc_cfg" in src
    assert "_learned_calibration_config(" in src


def test_default_routing_is_outside_preparation_budget():
    """B pays for L+A; the current paper allocates them 60/40."""
    cfg = run._learned_calibration_config(10000, 10)
    assert (cfg.total_budget, cfg.sample_budget, cfg.calib_budget) == (1000, 600, 400)
    assert cfg.route_frac == 0.0 and cfg.route_budget == 0
    assert cfg.route_unlimited is True and cfg.route_budget_mult == 0.0
    # Alpha chooses how many additional candidates are routed.
    assert 0.0 < cfg.alpha < 1.0


def test_default_routing_can_exceed_preparation_b():
    """Routing may exceed B while the preparation sample stays within B."""
    import numpy as np
    from learned_calibration import OofFeatures, run_learned_calibration

    rng = np.random.default_rng(0)
    n = 2000
    y_true = (rng.random(n) < 0.2).astype(int)
    x = np.where(y_true == 1, 1.0, -1.0) + rng.normal(0, 1.5, n)   # a weak proxy: much to route
    h = (x > 0).astype(float)
    cfg = run._learned_calibration_config(n, 5)
    cfg = type(cfg)(**{**cfg.__dict__, "alpha": 0.95})
    l_idx = rng.choice(n, cfg.sample_budget, replace=False)
    res = run_learned_calibration(
        x=x, h=h, labeled_indices=l_idx, labeled_values=y_true[l_idx], oof=OofFeatures(None, None, "n/a"),
        label_rows=lambda rows: y_true[rows], config=cfg)
    assert res.status.startswith("routed")
    assert res.diagnostics["lc_calls_total"] > cfg.total_budget
    assert res.diagnostics["lc_calls_L"] + res.diagnostics["lc_calls_A"] <= cfg.total_budget
