"""Versioned MLP model and its retrained single-factor ablations.

The model selection followed the completed v1 validation comparison. The frozen
v1 records remain untouched. MLP ablations need their own runs; only FULL and the
functionally identical NO_EXCHANGE arm are candidates for certified reuse.
"""
from copy import deepcopy

from experiments.state_exchange_multidata.common import load_config, spec

CORE_ARMS = (
    'FULL', 'NO_EXCHANGE', 'UP_ONLY', 'DOWN_ONLY', 'ORIGIN_EXCHANGE',
    'NO_POSITION_INPUT', 'NO_INTERIOR_CENTER',
)
ARMS = CORE_ARMS
PREDECLARED_SEEDS = (2025,)
PROTOCOL_VERSION = 'pase_mlp_v2_protocol_v1'
LEGACY_REUSE_ARMS = {'FULL': 'CONCAT_MLP', 'NO_EXCHANGE': 'NO_EXCHANGE'}

_BASE = dict(coefficient='concat_128_16_2_gelu_tanh', up=True, down=True,
             coefficient_states='current', explicit_position=True,
             interior_center=True, incident=True, region_loss_weight=.2)
_CHANGES = {
    'FULL': {},
    'NO_EXCHANGE': dict(up=False, down=False),
    'UP_ONLY': dict(down=False),
    'DOWN_ONLY': dict(up=False),
    'ORIGIN_EXCHANGE': dict(coefficient_states='origin'),
    'NO_POSITION_INPUT': dict(explicit_position=False),
    'NO_INTERIOR_CENTER': dict(interior_center=False),
}


def arm_spec(arm):
    if arm not in CORE_ARMS:
        raise ValueError('Unknown PASE MLP v2 arm: '+str(arm))
    value = dict(_BASE, **_CHANGES[arm])
    value.update(arm=arm, protocol=PROTOCOL_VERSION, implemented=True,
        initialization='identical_legacy_CONCAT_MLP_tensors_and_following_RNG',
        messages='unnormalized_current_old_states_synchronous_update',
        initial_local_state_centered=True, final_local_readout_centered=True,
        position_input_ablation='fixed_zero_p_block_only_other_position_paths_retained',
        legacy_reuse_arm=LEGACY_REUSE_ARMS.get(arm))
    return deepcopy(value)


def configuration(arm='FULL', dataset='Contra_Costa', nodes=None):
    """Return the unmodified dataset-specific GW16 numerical training recipe."""
    arm_spec(arm)
    record = spec(dataset)
    if nodes is not None and (type(nodes) is not int or nodes != record['nodes']):
        raise ValueError('Node count differs from the registered published release')
    return deepcopy(load_config(dataset))


def protocol_record():
    return dict(schema=PROTOCOL_VERSION, core_arms=list(CORE_ARMS),
        predeclared_seeds=list(PREDECLARED_SEEDS), epochs=100,
        initialization='same_seed_legacy_CONCAT_MLP_from_scratch_initialization',
        training_order='same_seed_fixed_window_order_per_epoch',
        best_selection='validation_pooled_mae_includes_stage_zero',
        selected_from='completed_v1_single_seed_development_validation_comparison',
        legacy_reuse_arms=dict(LEGACY_REUSE_ARMS),
        reuse_requires_full_data_config_runtime_source_and_tensor_certification=True,
        old_low_rank_ablation_results_are_not_MLP_ablation_results=True,
        no_inference_ablation_substitution=True,
        no_test_based_budget_or_variant_changes=True,
        engineering_only=dict(epochs=3, cannot_evaluate_test=True))
