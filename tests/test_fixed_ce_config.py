import pytest
from fixed_ce_config import runtime_options


def test_runtime_default_and_seed():
    options=runtime_options(18,environ={})
    assert options['activeSampling_backend']=='fixed_ce'
    assert options['fixed_ce_config'].seed==18
    assert options['fixed_ce_config'].batch_size==32
    assert 'activeSampling_n_phases' not in options


def test_json_overrides_and_legacy_rejection():
    options=runtime_options(environ={'PROBE_SAMPLING_CONFIG':'{"policy":"stratified","pilot_fraction":0.2}'})
    assert options['fixed_ce_config'].policy=='stratified'
    assert options['fixed_ce_config'].pilot_fraction==.2
    for key in ('LASO_N_PHASES','LASO_VS_RATIO'):
        with pytest.raises(ValueError,match='Obsolete'):
            runtime_options(environ={key:'1'})
    legacy=runtime_options(environ={'PROBE_SAMPLING_BACKEND':'legacy','LASO_N_PHASES':'3'})
    assert legacy['activeSampling_n_phases']==3
    assert runtime_options(sampling_method='random',environ={'LASO_N_PHASES':'3'})=={}


@pytest.mark.parametrize('key',['ACTIVESAMPLING_N_PHASES','LR_BLEND_SCHEDULE',
                              'COLD_START_PHASES','PHASE_BUDGET_SPLIT'])
def test_retired_environment_request_is_not_silently_ignored(key):
    with pytest.raises(ValueError,match=key):
        runtime_options(environ={key:'1'})


def test_old_json_config_names_give_actionable_rejection():
    with pytest.raises(ValueError,match='Obsolete fixed-CE options: phase_budget_split'):
        runtime_options(environ={'PROBE_SAMPLING_CONFIG':'{"phase_budget_split":"equal"}'})
    with pytest.raises(ValueError,match='PROBE_SAMPLING_CONFIG must be a JSON object'):
        runtime_options(environ={'PROBE_SAMPLING_CONFIG':'[]'})
    with pytest.raises(ValueError,match='PROBE_CE_CONFIG must be a JSON object'):
        runtime_options(environ={'PROBE_CE_CONFIG':'null'})
