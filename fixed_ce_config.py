"""Runtime configuration shared by the experiment entry point and semantic ops."""
import json
import os
from dataclasses import asdict
from fixed_ce_sampling import SamplingConfig
from ce_ranker import CEConfig


# Reject retired spellings as well as the two historical run.py environment
# overrides. A successful invocation must not silently ignore a requested axis.
_RETIRED_AXIS_OPTIONS = (
    'LASO_N_PHASES', 'LASO_VS_RATIO', 'ACTIVESAMPLING_N_PHASES',
    'ACTIVESAMPLING_VS_RATIO', 'ACTIVESAMPLING_LR_BLEND_SCHEDULE',
    'ACTIVESAMPLING_COLD_START_PHASES', 'LR_BLEND_SCHEDULE',
    'COLD_START_PHASES', 'PHASE_BUDGET_SPLIT',
)
_RETIRED_SAMPLING_FIELDS = (
    'n_phases', 'vs_ratio', 'lr_blend_schedule', 'cold_start_phases',
    'phase_budget_split', 'activeSampling_n_phases', 'activeSampling_vs_ratio',
    'activeSampling_lr_blend_schedule', 'activeSampling_cold_start_phases',
    'activeSampling_phase_budget_split',
)


def _reject_retired(keys):
    if keys:
        raise ValueError('Obsolete fixed-CE options: '+', '.join(sorted(keys))+
                         '; remove them or explicitly select PROBE_SAMPLING_BACKEND=legacy')


def runtime_options(seed=0, *, sampling_method='activeSampling', environ=None):
    env = os.environ if environ is None else environ
    if sampling_method != 'activeSampling':
        return {}
    backend = env.get('PROBE_SAMPLING_BACKEND', 'fixed_ce')
    if backend not in ('fixed_ce', 'legacy'):
        raise ValueError('PROBE_SAMPLING_BACKEND must be fixed_ce or legacy')
    if backend == 'legacy':
        return {'activeSampling_backend':'legacy',
                'activeSampling_n_phases':int(env.get('LASO_N_PHASES','5')),
                'activeSampling_vs_ratio':float(env.get('LASO_VS_RATIO','.5')),
                'activeSampling_lr_blend_schedule':'linear',
                'activeSampling_cold_start_phases':1}
    _reject_retired([key for key in _RETIRED_AXIS_OPTIONS if env.get(key)])
    settings = json.loads(env.get('PROBE_SAMPLING_CONFIG','{}'))
    if not isinstance(settings, dict):
        raise ValueError('PROBE_SAMPLING_CONFIG must be a JSON object')
    _reject_retired([key for key in _RETIRED_SAMPLING_FIELDS if key in settings])
    settings.setdefault('seed',int(seed or 0))
    ce = json.loads(env.get('PROBE_CE_CONFIG','{}'))
    if not isinstance(ce, dict):
        raise ValueError('PROBE_CE_CONFIG must be a JSON object')
    return {'activeSampling_backend':'fixed_ce',
            'fixed_ce_config':SamplingConfig(**settings),'ce_config':CEConfig(**ce)}


def defaults():
    return {'sampling':asdict(SamplingConfig()),'ce':asdict(CEConfig())}
