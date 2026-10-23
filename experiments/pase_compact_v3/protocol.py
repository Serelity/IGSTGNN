"""Only the compact FULL and its two matched directional controls are enabled.

The removed block is the scorer's explicit position input. Position still enters
the historical encoder, origin condition and fixed spatial regional partition.
No other proposed simplification is combined with this change.
"""
from copy import deepcopy

from experiments.state_exchange_multidata.common import load_config, spec

CORE_ARMS = ('FULL', 'UP_ONLY', 'DOWN_ONLY')
ARMS = CORE_ARMS
PREDECLARED_SEEDS = (2025,)
PROTOCOL_VERSION = 'pase_compact_v3_protocol_v1'
LEGACY_REUSE_ARMS = {'FULL': 'NO_POSITION_INPUT'}

_BASE = dict(coefficient='concat_96_16_2_gelu_tanh', up=True, down=True,
             coefficient_states='current', explicit_position=False,
             interior_center=True, incident=True, region_loss_weight=.2)
_CHANGES = {'FULL': {}, 'UP_ONLY': dict(down=False), 'DOWN_ONLY': dict(up=False)}


def arm_spec(arm):
    if arm not in CORE_ARMS:
        raise ValueError('Unknown PASE compact v3 arm: '+str(arm))
    value = dict(_BASE, **_CHANGES[arm])
    value.update(arm=arm, protocol=PROTOCOL_VERSION, implemented=True,
        initialization='legacy_CONCAT_MLP_initial_tensors_then_remove_first_p_block_preserve_RNG',
        messages='unnormalized_current_old_states_synchronous_update',
        initial_local_state_centered=True, final_local_readout_centered=True,
        position_input_ablation='explicit_scorer_p_block_removed_other_position_paths_retained',
        legacy_reuse_arm=LEGACY_REUSE_ARMS.get(arm))
    return deepcopy(value)


def configuration(arm='FULL', dataset='Contra_Costa', nodes=None):
    """Keep the released per-dataset GW16 numerical training recipe unchanged."""
    arm_spec(arm)
    record = spec(dataset)
    if nodes is not None and (type(nodes) is not int or nodes != record['nodes']):
        raise ValueError('Node count differs from the registered published release')
    return deepcopy(load_config(dataset))


def protocol_record():
    return dict(schema=PROTOCOL_VERSION, core_arms=list(CORE_ARMS),
        predeclared_seeds=list(PREDECLARED_SEEDS), epochs=100,
        initialization='same_seed_legacy_CONCAT_MLP_initialization_then_slice_p_columns',
        training_order='same_seed_fixed_window_order_per_epoch',
        best_selection='validation_pooled_mae_includes_stage_zero',
        selected_from='completed_MLP_v2_single_seed_development_ablation',
        legacy_reuse_arms=dict(LEGACY_REUSE_ARMS),
        reuse_requires_full_data_config_runtime_source_and_tensor_certification=True,
        compact_conversion='remove_first_width_columns_of_first_scorer_linear_only',
        conversion_math_equivalent_but_FP32_bits_not_promised=True,
        existing_explicit_position_directional_results_are_not_matched_controls=True,
        new_training_arms=['UP_ONLY', 'DOWN_ONLY'],
        no_test_based_budget_or_variant_changes=True,
        engineering_only=dict(epochs=3, cannot_evaluate_test=True))
