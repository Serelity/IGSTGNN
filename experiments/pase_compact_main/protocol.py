"""One primary model and one seed; no baseline or additional ablation runs."""
from copy import deepcopy
from experiments.pase_compact_v3.protocol import configuration as _configuration
from experiments.pase_compact_v3.protocol import arm_spec as _arm_spec

CORE_ARMS = ARMS = ('FULL',)
PREDECLARED_SEEDS = (2025,)
PROTOCOL_VERSION = 'pase_compact_main_protocol_v1'


def arm_spec(arm):
    if arm != 'FULL':
        raise ValueError('Main results permit FULL only')
    value = _arm_spec('FULL')
    value.update(protocol=PROTOCOL_VERSION, legacy_reuse_arm=None)
    return deepcopy(value)


def configuration(arm='FULL', dataset='Alameda', nodes=None):
    arm_spec(arm)
    if dataset not in ('Alameda', 'Orange'):
        raise ValueError('Reuse existing Contra Costa through its original evaluator')
    return _configuration('FULL', dataset=dataset, nodes=nodes)


def protocol_record():
    return dict(schema=PROTOCOL_VERSION, core_arms=['FULL'], seeds=[2025], epochs=100,
        datasets=['Alameda', 'Orange'], from_scratch=True,
        architecture='unchanged_pase_compact_v3_FULL_96_16_2',
        best_selection='validation_pooled_mae_includes_stage_zero',
        training_order='same_seed_fixed_window_order_per_epoch',
        no_test_based_budget_or_variant_changes=True,
        contra_costa='reuse_existing_completed_FULL_via_original_evaluator',
        engineering_only=dict(epochs=3, cannot_evaluate_test=True))
