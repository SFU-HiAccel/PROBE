"""The learned-calibration switch: plumbing and run.py label mapping."""
import inspect

import pytest

import proxy_model
import sem_ops_register


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
    assert run._decision_mode_for_ablation("activeSampling+refine+learned_deferral") == "learned_deferral"
    assert run._decision_mode_for_ablation("activeSampling+refine+dasp_topk") == "learned_deferral"
    assert run._decision_mode_for_ablation("activeSampling+refine+trust_proxy") == "none"
    assert run._decision_mode_for_ablation("activeSampling+lr_only") == "none"


def test_default_grid_does_not_enable_learned_calibration():
    assert all(run._decision_mode_for_ablation(label) != "learned_calibration"
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
    assert (cfg.total_budget, cfg.sample_budget, cfg.calib_budget, cfg.seed) == (500, 300, 50, 7)


def test_existing_budget_split_is_untouched():
    assert run.split_total_budget(10000, 5, 0.3) == (500, 500, 150)


@pytest.mark.parametrize("fn_name", ["run_experiment_filter", "run_experiment_join"])
def test_experiment_runners_pass_the_config(fn_name):
    src = inspect.getsource(getattr(run, fn_name))
    assert "learned_calibration_config=_lc_cfg" in src
    assert "_learned_calibration_config(" in src
