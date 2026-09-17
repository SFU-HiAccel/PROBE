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
