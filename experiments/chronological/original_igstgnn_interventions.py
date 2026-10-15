"""Temporary fixed-checkpoint interventions for the released IGSTGNN.

The intended model is the isolated 558ab01 implementation, not the later
fixed-A/MIN/B3-R variants.  These interventions measure checkpoint path
dependence.  They neither retrain a reduced model nor measure its runtime.
"""
from contextlib import contextmanager
import math

import torch


CASE_NAMES = (
    'FULL', 'DIF_FORECAST_READOUT_OFF', 'INH_FORECAST_READOUT_OFF',
    'ICSF_ONLY', 'NORM_ONLY', 'TIID_ONLY_ON_NORM', 'RAW_ALL_REPORT_OFF',
    'PRE_BANKS_ZERO', 'DYN_BANKS_ZERO', 'STA_BANKS_ZERO',
    'GRU_TRANSITION_IDENTITY', 'ATTENTION_QUERY_IDENTITY',
)

CASE_DESCRIPTIONS = {
    'FULL': 'Native released model with report; original four-block TIID and head.',
    'DIF_FORECAST_READOUT_OFF': 'Zero every DIF forecast output only; preserve DIF backcasts, INH and all later layers.',
    'INH_FORECAST_READOUT_OFF': 'Zero every INH forecast output only; preserve INH backcasts and all later layers.',
    'ICSF_ONLY': 'Native ICSF including latest-slot LayerNorm; discard the TIID output increment.',
    'NORM_ONLY': 'Replace ICSF-enhanced history by latest-slot LayerNorm without its incident increment; discard TIID increment.',
    'TIID_ONLY_ON_NORM': 'Use normalization-only history; preserve the native incident context computed from the original embedded history for TIID.',
    'RAW_ALL_REPORT_OFF': 'Native incident_data=None: neither latest-slot ICSF LayerNorm nor TIID executes.',
    'PRE_BANKS_ZERO': 'Zero predefined bank feature slices before every gcn_updt; preserve all other banks, bias and graph construction.',
    'DYN_BANKS_ZERO': 'Zero dynamic bank feature slices before every gcn_updt; preserve all other banks, bias and graph construction.',
    'STA_BANKS_ZERO': 'Zero static bank feature slices before every gcn_updt; preserve all other banks, bias and graph construction.',
    'GRU_TRANSITION_IDENTITY': 'Replace every historical and generated GRUCell output by its input; preserve positional encoding and attention.',
    'ATTENTION_QUERY_IDENTITY': 'Replace every historical and generated MultiheadAttention output by its query; preserve GRU, positional encoding and outer dropout.',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_model(model):
    _require(not any(module.training for module in model.modules()), 'Interventions require every module in eval mode')
    _require(all(parameter.grad is None for parameter in model.parameters()), 'Interventions require absent parameter gradients')
    _require(not hasattr(model, '_decode_forecast') and not hasattr(model, 'report_spatial_state'),
             'Use the isolated released model, not fixed-A/MIN/B3-R')
    _require(hasattr(model, 'icsf_module') and hasattr(model, 'tiid_module') and hasattr(model, 'layers'),
             'Missing released IGSTGNN modules')
    _require(len(model.layers) > 0, 'Model must have a decomposition layer')
    for layer in model.layers:
        _require(hasattr(layer, 'dif_layer') and hasattr(layer, 'inh_layer'), 'Missing DIF/INH modules')
        conv = layer.dif_layer.localized_st_conv
        _require(not hasattr(conv, 'incident_exchange'), 'Released convolution cannot contain a latent exchange adapter')


def _bank_layout(conv):
    """Follow released get_graph ordering; do not change registered widths."""
    width = int(conv.hidden_dim)
    offset = 1  # native local/self feature comes first
    predefined = len(conv.pre_defined_graph) if conv.use_predefined_graph else 0
    dynamic = len(conv.pre_defined_graph) if conv.use_dynamic_hidden_graph else 0
    static = int(conv.k_s) if conv.use_static_hidden_graph else 0
    layout = {}
    for name, count in (('PRE_BANKS_ZERO', predefined), ('DYN_BANKS_ZERO', dynamic),
                        ('STA_BANKS_ZERO', static)):
        layout[name] = (offset*width, (offset+count)*width)
        offset += count
    _require(offset == conv.num_matric and conv.gcn_updt.in_features == offset*width,
             'Released graph-bank axis does not match gcn_updt width')
    return layout


@contextmanager
def intervention(model, case, *, report_present=True):
    """Install only local hooks, yield JSON-compatible execution diagnostics."""
    _require(case in CASE_NAMES, 'Unknown original IGSTGNN intervention: '+str(case))
    _validate_model(model)
    _require(report_present or case == 'RAW_ALL_REPORT_OFF', 'This case requires a supplied original report')
    audit = {
        'case': case, 'description': CASE_DESCRIPTIONS[case],
        'scope': 'fixed_checkpoint_path_sensitivity_not_retrained_ablation',
        'native_modules_still_computed_before_output_hooks': True,
        'counts': {}, 'per_module_calls': {}, 'bank_feature_slices': {},
    }
    handles = []

    def count(key, path):
        def record(_module, _args, _output):
            audit['counts'][key] = audit['counts'].get(key, 0)+1
            audit['per_module_calls'][path] = audit['per_module_calls'].get(path, 0)+1
        return record

    def changed(key):
        audit['counts'][key] = audit['counts'].get(key, 0)+1

    def zero_forecast(_module, _args, output):
        _require(isinstance(output, tuple) and len(output) == 2,
                 'DIF/INH must return (backcast_residual, forecast)')
        changed('forecast_outputs_zeroed')
        return output[0], torch.zeros_like(output[1])

    def norm_only(module, args, output):
        _require(isinstance(output, tuple) and len(output) == 2,
                 'Released ICSF must return (enhanced_history, incident_context)')
        history = args[0]
        normalized = history.clone()
        normalized[:, -1] = module.output_norm(history[:, -1])
        changed('icsf_histories_replaced_by_norm_only')
        # The native context is intentionally retained for TIID_ONLY_ON_NORM.
        return normalized, output[1]

    def tiid_identity(_module, args, _output):
        changed('tiid_output_increments_discarded')
        return args[0]

    def gru_identity(_module, args, _output):
        changed('gru_outputs_replaced_by_input')
        return args[0]

    def attention_identity(_module, args, _output):
        changed('attention_outputs_replaced_by_query')
        return args[0], None

    try:
        for key, path, module in (
                ('icsf', 'icsf_module', model.icsf_module),
                ('tiid', 'tiid_module', model.tiid_module)):
            handles.append(module.register_forward_hook(count(key, path)))
        for index, layer in enumerate(model.layers):
            prefix = 'layers.{}'.format(index)
            conv = layer.dif_layer.localized_st_conv
            registrations = (
                ('dif', prefix+'.dif_layer', layer.dif_layer),
                ('inh', prefix+'.inh_layer', layer.inh_layer),
                ('localized_conv', prefix+'.dif_layer.localized_st_conv', conv),
                ('gru_cell', prefix+'.inh_layer.rnn_layer.gru_cell', layer.inh_layer.rnn_layer.gru_cell),
                ('attention', prefix+'.inh_layer.transformer_layer.multi_head_self_attention',
                 layer.inh_layer.transformer_layer.multi_head_self_attention),
            )
            for key, path, module in registrations:
                handles.append(module.register_forward_hook(count(key, path)))
            if case == 'DIF_FORECAST_READOUT_OFF':
                handles.append(layer.dif_layer.register_forward_hook(zero_forecast))
            elif case == 'INH_FORECAST_READOUT_OFF':
                handles.append(layer.inh_layer.register_forward_hook(zero_forecast))
            elif case in ('PRE_BANKS_ZERO', 'DYN_BANKS_ZERO', 'STA_BANKS_ZERO'):
                first, last = _bank_layout(conv)[case]
                _require(last > first, 'Requested graph family is absent: '+case)
                audit['bank_feature_slices'][prefix] = [first, last]

                def mask_bank(_module, args, first=first, last=last):
                    features = args[0].clone()
                    features[..., first:last] = 0.
                    changed('graph_bank_inputs_masked')
                    return (features,)+tuple(args[1:])

                handles.append(conv.gcn_updt.register_forward_pre_hook(mask_bank))
            elif case == 'GRU_TRANSITION_IDENTITY':
                handles.append(layer.inh_layer.rnn_layer.gru_cell.register_forward_hook(gru_identity))
            elif case == 'ATTENTION_QUERY_IDENTITY':
                handles.append(layer.inh_layer.transformer_layer.multi_head_self_attention.register_forward_hook(attention_identity))
        if case in ('NORM_ONLY', 'TIID_ONLY_ON_NORM'):
            handles.append(model.icsf_module.register_forward_hook(norm_only))
        if case in ('ICSF_ONLY', 'NORM_ONLY'):
            handles.append(model.tiid_module.register_forward_hook(tiid_identity))
        with torch.inference_mode():
            yield audit
    finally:
        for handle in reversed(handles):
            handle.remove()


def _check_execution(model, x, case, audit):
    layers = len(model.layers)
    blocks = math.ceil(model._model_args['horizon']/model._model_args['gap'])
    current_length = x.shape[1]
    expected_gru = 0
    expected_modules = {'icsf_module': int(case != 'RAW_ALL_REPORT_OFF'),
                        'tiid_module': int(case != 'RAW_ALL_REPORT_OFF')}
    for index, layer in enumerate(model.layers):
        current_length -= layer.dif_layer.localized_st_conv.k_t-1
        _require(current_length > 0, 'Input history cannot support every released DIF layer')
        inherent_blocks = int(layer.inh_layer.forecast_block.output_seq_len/model._model_args['gap'])
        expected_gru += current_length+inherent_blocks-1
        prefix = 'layers.{}'.format(index)
        expected_modules.update({prefix+'.dif_layer': 1, prefix+'.inh_layer': 1,
                                 prefix+'.dif_layer.localized_st_conv': blocks,
                                 prefix+'.inh_layer.rnn_layer.gru_cell': current_length+inherent_blocks-1,
                                 prefix+'.inh_layer.transformer_layer.multi_head_self_attention': inherent_blocks})
    attention_calls = sum(int(layer.inh_layer.forecast_block.output_seq_len/model._model_args['gap'])
                          for layer in model.layers)
    expected = {'icsf': int(case != 'RAW_ALL_REPORT_OFF'),
                'tiid': int(case != 'RAW_ALL_REPORT_OFF'),
                'dif': layers, 'inh': layers, 'localized_conv': layers*blocks,
                'gru_cell': expected_gru, 'attention': attention_calls}
    for key, count in expected.items():
        _require(audit['counts'].get(key, 0) == count, 'Execution coverage mismatch: '+key)
    if case in ('DIF_FORECAST_READOUT_OFF', 'INH_FORECAST_READOUT_OFF'):
        expected['forecast_outputs_zeroed'] = layers
    elif case in ('PRE_BANKS_ZERO', 'DYN_BANKS_ZERO', 'STA_BANKS_ZERO'):
        expected['graph_bank_inputs_masked'] = layers*blocks
    elif case == 'GRU_TRANSITION_IDENTITY':
        expected['gru_outputs_replaced_by_input'] = expected_gru
    elif case == 'ATTENTION_QUERY_IDENTITY':
        expected['attention_outputs_replaced_by_query'] = attention_calls
    if case in ('NORM_ONLY', 'TIID_ONLY_ON_NORM'):
        expected['icsf_histories_replaced_by_norm_only'] = 1
    if case in ('ICSF_ONLY', 'NORM_ONLY'):
        expected['tiid_output_increments_discarded'] = 1
    for key, count in expected.items():
        _require(audit['counts'].get(key, 0) == count, 'Intervention coverage mismatch: '+key)
    for path, count in expected_modules.items():
        _require(audit['per_module_calls'].get(path, 0) == count, 'Per-module execution coverage mismatch: '+path)
    audit['expected_counts'] = expected
    audit['expected_per_module_calls'] = expected_modules
    audit['execution_coverage_passed'] = True


def run_case_with_audit(model, x, incident, sensors, case):
    """Return (prediction, audit); no synchronization, training or state writes."""
    _require(isinstance(x, torch.Tensor) and x.ndim == 4, 'Expected original traffic X[B,T,N,F]')
    _require(incident is not None or case == 'RAW_ALL_REPORT_OFF', 'A native report is required')
    with intervention(model, case, report_present=incident is not None) as audit:
        with torch.inference_mode():
            prediction = model(x, incident_data=None if case == 'RAW_ALL_REPORT_OFF' else incident,
                               sensor_data=sensors)
        _require(isinstance(prediction, torch.Tensor), 'Released model must return a tensor')
        _check_execution(model, x, case, audit)
    return prediction, audit


def run_case(model, x, incident, sensors, case):
    """Run one temporary intervention and return the native prediction tensor."""
    return run_case_with_audit(model, x, incident, sensors, case)[0]
