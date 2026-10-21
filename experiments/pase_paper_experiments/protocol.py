"""Predeclared PASE-Net mechanism replacements and single-factor ablations.

The numerical/training recipe is the published GW16 recipe. Arm semantics are
recorded separately rather than changing the original GW16 configuration.
"""
from copy import deepcopy

from experiments.state_exchange_multidata.common import load_config, spec


CORE_ARMS = (
    'FULL', 'CONCAT_MLP', 'REGIONAL_SOFTMAX', 'NO_EXCHANGE', 'UP_ONLY',
    'DOWN_ONLY', 'ORIGIN_EXCHANGE', 'NO_POSITION_BRANCH', 'NO_INTERIOR_CENTER',
)
ARMS = CORE_ARMS
SUPPLEMENTARY_ARMS = ('NO_INCIDENT', 'NO_REGION_LOSS')
PREDECLARED_SEEDS = (2025, 2026, 2027)
PROTOCOL_VERSION = 'pase_paper_mechanism_protocol_v1'

_BASE = dict(coefficient='position_state_rank8_tanh', up=True, down=True,
             coefficient_states='current', explicit_position=True,
             interior_center=True, incident=True, region_loss_weight=.2)
_CHANGES = {
    'FULL': dict(),
    'CONCAT_MLP': dict(coefficient='concat_128_16_2_gelu_tanh'),
    'REGIONAL_SOFTMAX': dict(coefficient='position_state_rank8_regional_softmax'),
    'NO_EXCHANGE': dict(up=False, down=False),
    'UP_ONLY': dict(down=False),
    'DOWN_ONLY': dict(up=False),
    'ORIGIN_EXCHANGE': dict(coefficient_states='origin'),
    'NO_POSITION_BRANCH': dict(explicit_position=False),
    'NO_INTERIOR_CENTER': dict(interior_center=False),
    # Registered plans only; these are not in the current training matrix.
    'NO_INCIDENT': dict(incident=False),
    'NO_REGION_LOSS': dict(region_loss_weight=0.),
}


def arm_spec(arm):
    if arm not in _CHANGES:
        raise ValueError('Unknown PASE paper arm: '+str(arm))
    value = dict(_BASE, **_CHANGES[arm])
    value.update(arm=arm, protocol=PROTOCOL_VERSION,
                 implemented=arm in CORE_ARMS,
                 initialization='identical_GW16_constructor_then_isolated_RNG_replacement',
                 messages='unnormalized_current_old_states_synchronous_update',
                 final_local_readout_centered=True,
                 initial_local_state_centered=True)
    return deepcopy(value)


def configuration(arm='FULL', dataset='Contra_Costa', nodes=None):
    """Return the unchanged dataset-specific GW16 recipe for an implemented arm."""
    if arm not in CORE_ARMS:
        raise ValueError('Only the nine predeclared core arms are implemented')
    record = spec(dataset)
    if nodes is not None and (type(nodes) is not int or nodes != record['nodes']):
        raise ValueError('Node count differs from the registered published release')
    return deepcopy(load_config(dataset))


def protocol_record():
    return dict(schema=PROTOCOL_VERSION, core_arms=list(CORE_ARMS),
        predeclared_seeds=list(PREDECLARED_SEEDS), epochs=100,
        initialization='from_scratch_common_tensors_identical_within_seed',
        best_selection='validation_pooled_mae_includes_stage_zero',
        training_order='same_seed_fixed_window_order_per_epoch',
        supplementary_plans=[arm_spec(arm) for arm in SUPPLEMENTARY_ARMS],
        no_inference_ablation_substitution=True,
        no_test_based_budget_or_variant_changes=True,
        old_FULL_reuse_requires_separate_certification=True)
