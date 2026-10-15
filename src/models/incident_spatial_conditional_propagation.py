"""One report encoder and conditional messages inside DIF, with native INH/readout.

The new spatial operator replaces MIN's localized convolution and latent
exchange. Compatible backbone weights are copied, but old predictions are not
preserved. Report OFF denotes this checkpoint's current HISTORY computation.
"""
import copy

import torch
from torch import nn

from src.models.igstgnn import DifBlock, DifForecast
from src.models.incident_conditional_propagation import ConditionalMessagePropagation
from src.models.incident_spatial import require
from src.models.incident_spatial_event_budget import EventSemanticEncoder
from src.models.incident_spatial_minimal import (
    HistoryCapacityPrior, MinimalIGSTGNN, cutoff_clock_from_history,
)
from src.models.incident_spatial_residual import tensor_state_hash


MAPPING_VERSION = 'minimal_to_conditional_message_propagation_v1'
MODES = ('history', 'mask_only', 'event')
CONTEXT_SCHEMA = 'MIN_conditional_message_propagation_v1'


class NodeReportCondition(nn.Module):
    """Node-local condition; no report pooling across the station axis."""
    def __init__(self, hidden, type_count=6, description_count=29):
        super().__init__()
        self.hidden = hidden
        self.semantic_encoder = EventSemanticEncoder(type_count, description_count)
        self.condition = nn.Sequential(nn.Linear(3*hidden+3+16+1, hidden), nn.SiLU(),
                                       nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, raw, incident, *, mode, enabled=True, semantic_enabled=True):
        require(mode in MODES and type(enabled) is bool and type(semantic_enabled) is bool,
                'Invalid report condition mode/switch')
        batch, _, nodes, hidden = raw.shape
        require(hidden == self.hidden, 'Report history hidden axis mismatch')
        relation = raw.new_zeros(batch, nodes, 3)
        support = torch.zeros(batch, nodes, dtype=torch.bool, device=raw.device)
        available = torch.zeros(batch, dtype=torch.bool, device=raw.device)
        empty = dict(condition=raw.new_zeros(batch, nodes, hidden), support=support,
                     relation=relation, available=available, semantic_used=False, encoder_calls=0)
        if mode == 'history' or not enabled or incident is None:
            return empty
        require(isinstance(incident, dict), 'Conditional propagation event input must be a dictionary')
        value = incident.get('relation')
        require(isinstance(value, torch.Tensor) and value.shape == (batch, nodes, 3)
                and value.dtype == torch.float32 and value.device == raw.device
                and bool(torch.isfinite(value).all()), 'Invalid Float32[B,N,3] event relation')
        flags = []
        for name in ('report_present', 'relation_available'):
            flag = incident.get(name)
            require(isinstance(flag, torch.Tensor) and flag.shape == (batch,)
                    and flag.dtype == torch.bool and flag.device == raw.device,
                    'Invalid Bool[B] '+name)
            flags.append(flag)
        available = flags[0] & flags[1]
        support = (value.abs().sum(-1) > 0) & available[:, None]
        empty.update(support=support, available=available)
        if not bool(support.any()):
            return empty
        # Compute support before masking information. M_ONLY must not lose M.
        if mode == 'event':
            relation = torch.where(support[..., None], value.to(raw.dtype), 0.)
        semantic_used = mode == 'event' and semantic_enabled
        if semantic_used:
            semantic_report = dict(incident)
            row_supported = support.any(-1)
            for name in ('type', 'description'):
                valid = incident.get(name+'_valid')
                require(isinstance(valid, torch.Tensor) and valid.shape == (batch,)
                        and valid.dtype == torch.bool and valid.device == raw.device,
                        'Invalid Bool[B] '+name+'_valid')
                semantic_report[name+'_valid'] = valid & row_supported
            semantic = self.semantic_encoder(semantic_report, batch, raw.device, enabled=True).to(raw.dtype)
        else:
            # M_ONLY has no semantic input, including a learned global UNK.
            semantic = raw.new_zeros(batch, 16)
        semantic = torch.where(support[..., None], semantic[:, None].expand(-1, nodes, -1), 0.)
        summary = torch.cat((raw[:, -1], raw.mean(1), raw[:, -1]-raw[:, 0]), -1)
        features = torch.cat((summary, relation, semantic, support[..., None].to(raw.dtype)), -1)
        condition = self.condition(features)
        condition = torch.where(support[..., None], condition, 0.)
        return dict(condition=condition, support=support, relation=relation, available=available,
                    semantic_used=semantic_used, encoder_calls=1)


def _kernel_call(kernel, history, dynamic, static, context, *, kind=None, block=None, anchors=None):
    collect = context['collect']
    try:
        result = kernel(history, dynamic, static,
                        condition=context['condition'], support=context['support'],
                        available=context['available'], token_kind=kind, token_block=block,
                        report_enabled=context['report_enabled'],
                        logit_enabled=context['logit_enabled'], value_enabled=context['value_enabled'],
                        collect=collect)
        if collect:
            require(isinstance(kernel.last_details, dict), 'Missing kernel diagnostics')
            context['records'].append(dict(kernel.last_details,
                layer=context['layer_index'], source_history_anchors=list(anchors or [])))
    finally:
        # A replay cannot leave diagnostic tensors attached to the model.
        kernel.last_details = None
    return result


class ConditionalDifForecast(DifForecast):
    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph,
                static_graph, spatial_context=None):
        require(isinstance(spatial_context, dict), 'Conditional forecast context required')
        require(self.k_t == 3 and self.output_seq_len == 12 and self.model_args['gap'] == 3,
                'Conditional forecast requires four native gap-three blocks')
        predict = [hidden_states_dif[:, -1:]]
        for block_index in range(1, 4):
            prefix = predict[-self.k_t:]
            history_count = max(0, self.k_t-len(prefix))
            parts = ([gated_history_data[:, -history_count:]] if history_count else [])+prefix
            window = torch.cat(parts, dim=1)
            generated_ids = list(range(max(0, block_index-self.k_t), block_index))
            kind = torch.tensor([0]*history_count+[1]*len(prefix),
                                dtype=torch.long, device=window.device)
            block = torch.tensor([-1]*history_count+generated_ids,
                                 dtype=torch.long, device=window.device)
            if spatial_context['collect']:
                spatial_context['forecast_windows'].append(dict(
                    layer=spatial_context['layer_index'], target_block=block_index,
                    token_kind=kind.detach().cpu().tolist(), token_block=block.detach().cpu().tolist(),
                    history_anchors=(spatial_context['history_anchors'][-history_count:]
                                     if history_count else [])+[-1]*len(prefix)))
            anchors = ((spatial_context['history_anchors'][-history_count:] if history_count else [])
                       +[-1]*len(prefix))
            predict.append(_kernel_call(localized_st_conv, window, dynamic_graph, static_graph,
                                        spatial_context, kind=kind, block=block, anchors=anchors))
        return self.forecast_fc(torch.cat(predict, dim=1))


class ConditionalDifBlock(DifBlock):
    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph,
                spatial_context=None):
        require(isinstance(spatial_context, dict), 'Conditional DIF context required')
        require(len(spatial_context['history_anchors']) == history_data.shape[1]
                and spatial_context['history_anchors'][-1] == 11, 'History anchor alignment changed')
        hidden = _kernel_call(self.localized_st_conv, gated_history_data, dynamic_graph,
                              static_graph, spatial_context, anchors=spatial_context['history_anchors'])
        forecast = self.forecast_branch(gated_history_data, hidden, self.localized_st_conv,
                                        dynamic_graph, static_graph, spatial_context=spatial_context)
        backcast = self.backcast_branch(hidden)
        residual = self.residual_decompose(history_data[:, -backcast.shape[1]:], backcast)
        return residual, forecast


def _summarize(records, reference):
    statistics = {}
    for record in records:
        source = record.get('statistics', record)
        for name, value in source.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                continue
            value = value.detach().to(device=reference.device, dtype=torch.float64).reshape(())
            if name not in statistics:
                statistics[name] = value.clone()
            elif name.endswith('_max'):
                statistics[name] = torch.maximum(statistics[name], value)
            elif name.endswith('_min'):
                statistics[name] = torch.minimum(statistics[name], value)
            else:
                statistics[name] = statistics[name]+value
    return statistics


class ConditionalPropagationIGSTGNN(MinimalIGSTGNN):
    def forward(self, history_data, label=None, incident_data=None, sensor_data=None, forecast_delta=None,
                *, report_enabled=True, logit_enabled=True, value_enabled=True, semantic_enabled=True,
                spatial_report_enabled=True, tiid_report_enabled=True, return_spatial_details=False):
        require(all(type(value) is bool for value in (report_enabled, logit_enabled, value_enabled,
                    semantic_enabled, spatial_report_enabled, tiid_report_enabled, return_spatial_details)),
                'Conditional propagation switches must be bool')
        require(forecast_delta is None, 'Conditional propagation cannot combine an external residual')
        require(isinstance(history_data, torch.Tensor) and history_data.ndim == 4
                and history_data.shape[1:] == (12, self.node_num, 3)
                and history_data.dtype == self.embedding.weight.dtype
                and history_data.device == self.embedding.weight.device
                and bool(torch.isfinite(history_data).all()), 'Invalid conditional propagation history')
        cutoff_clock_from_history(history_data, num_feat=self._model_args['num_feat'],
                                  tpd=self._model_args['tpd'], offset_slots=0)
        history, node_u, node_d, tod_features, dow_features = self._prepare_inputs(history_data)
        raw = self.embedding(history)
        hidden = raw.clone()
        hidden[:, -1] = self.icsf_module.output_norm(raw[:, -1])
        # This report-free graph is computed once, never from incident residuals.
        static, dynamic = self._graph_constructor(node_embedding_u=node_u, node_embedding_d=node_d,
            history_data=hidden, time_in_day_feat=tod_features, day_in_week_feat=dow_features)
        active = report_enabled and spatial_report_enabled and (logit_enabled or value_enabled)
        report = self.report_condition(raw, incident_data, mode=self._conditional_propagation_mode,
                                       enabled=active, semantic_enabled=semantic_enabled)
        context = dict(condition=report['condition'], support=report['support'], available=report['available'],
                       report_enabled=active and self._conditional_propagation_mode != 'history',
                       logit_enabled=logit_enabled, value_enabled=value_enabled,
                       collect=return_spatial_details, records=[], forecast_windows=[],
                       history_anchors=list(range(12)), layer_index=0)
        residual, dif, inh = hidden, [], []
        history_lengths = [hidden.shape[1]]
        history_anchors_by_layer = [list(range(12))]
        for layer_index, layer in enumerate(self.layers):
            context['layer_index'] = layer_index
            residual, df, inf = layer(residual, dynamic, static, node_u, node_d,
                                     tod_features, dow_features, incident_inputs=None, spatial_context=context)
            dif.append(df)
            inh.append(inf)
            history_lengths.append(residual.shape[1])
            context['history_anchors'] = context['history_anchors'][-residual.shape[1]:]
            history_anchors_by_layer.append(list(context['history_anchors']))
        prediction = self._decode_forecast(sum(dif)+sum(inh), None)
        if not return_spatial_details:
            return prediction
        statistics = _summarize(context['records'], raw)
        statistics.update(condition_encoder_calls=raw.new_tensor(report['encoder_calls']),
                          node_positions=raw.new_tensor(raw.shape[0]*self.node_num),
                          supported_node_positions=report['support'].sum().to(raw.dtype),
                          backbone_calls=raw.new_tensor(1), output_head_calls=raw.new_tensor(1))
        propagation = dict(schema=CONTEXT_SCHEMA, mode=self._conditional_propagation_mode,
            support=report['support'].detach(), relation=report['relation'].detach(),
            report_available=report['available'].detach(), semantic_context_used=report['semantic_used'],
            event_context_used=bool(report['support'].any()) and active,
            report_enabled=context['report_enabled'], logit_enabled=logit_enabled, value_enabled=value_enabled,
            statistics=statistics, history_lengths=history_lengths,
            history_anchors_by_layer=history_anchors_by_layer,
            forecast_windows=context['forecast_windows'], kernel_calls=len(context['records']),
            scope='conditional_offline_representation_not_physical_capacity',
            legacy_exchange_removed=True, old_MIN_stage_zero_equivalence=False)
        return dict(prediction=prediction, propagation=propagation)


def from_minimal(model, mode, adjacency, *, receiver_chunk_size=4, checkpoint_messages=True,
                 type_count=6, description_count=29):
    require(mode in MODES, 'Unknown conditional propagation mode')
    require(type(model) is MinimalIGSTGNN and type(model.report_spatial_state) is HistoryCapacityPrior
            and not any(hasattr(model, name) for name in ('forecast_fusion', 'early_event_condition',
                                                        'event_budget_adapter', 'report_condition')),
            'Expected an unadapted MIN architecture')
    require(len(model.layers) == 5 and model.horizon == 12 and model._model_args['gap'] == 3
            and model._model_args['k_t'] == 3 and model._model_args['k_s'] == 2
            and model._model_args['num_feat'] == 1, 'Conditional propagation requires the native 5-layer contract')
    require(type(receiver_chunk_size) is int and receiver_chunk_size > 0
            and type(checkpoint_messages) is bool, 'Invalid conditional resource settings')
    require(torch.as_tensor(adjacency).shape == (model.node_num, model.node_num),
            'Conditional adjacency must match the model station axis')
    before = tensor_state_hash(model.state_dict())
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        result = copy.deepcopy(model)
        result.__class__ = ConditionalPropagationIGSTGNN
        result._conditional_propagation_mode = mode
        reference = next(result.parameters())
        result.report_condition = NodeReportCondition(result._hidden_dim, type_count, description_count)
        result.report_condition.to(device=reference.device, dtype=reference.dtype).train(model.training)
        for layer in result.layers:
            dif = layer.dif_layer
            dif.__class__ = ConditionalDifBlock
            dif.forecast_branch.__class__ = ConditionalDifForecast
            dif.localized_st_conv = ConditionalMessagePropagation(
                dif.localized_st_conv, adjacency, receiver_chunk_size=receiver_chunk_size,
                checkpoint_messages=checkpoint_messages)
            dif.localized_st_conv.to(device=reference.device, dtype=reference.dtype).train(model.training)
        del result.report_spatial_state
        del result.internal_spatial_graph
        result._internal_spatial_mode = 'conditional_message_propagation'
        result._minimal_clock_offset_slots = 0
        result.requires_grad_(True)
    parent, current = model.state_dict(), result.state_dict()
    shared = set(parent) & set(current)
    require(all(parent[name].shape == current[name].shape and torch.equal(parent[name], current[name])
                for name in shared), 'A retained source tensor changed')
    require(tensor_state_hash(model.state_dict()) == before, 'Source MIN was modified')
    removed = sorted(set(parent)-set(current))
    added = sorted(set(current)-set(parent))
    require(removed and added, 'Conditional propagation failed to replace source computation')
    require(not any('incident_exchange' in name or name.startswith(('report_spatial_state.', 'internal_spatial_graph.'))
                    for name in current), 'Unused legacy exchange/capacity state remains')
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    return result, dict(mapping_version=MAPPING_VERSION, architecture_version=CONTEXT_SCHEMA, mode=mode,
        parent_state_sha256=before, initial_state_sha256=tensor_state_hash(current),
        retained_state_exact_verified=True, retained_state_keys=sorted(shared),
        removed_state_keys=removed, added_state_keys=added,
        parent_parameter_count=count(model), parameter_count=count(result),
        parameter_difference=count(result)-count(model), all_parameters_trainable=True,
        report_encoders=1, backbone_forwards=1, output_head_forwards=1,
        original_MIN_prediction_equivalence=False, candidate_own_stage_zero_required=True,
        same_checkpoint_report_OFF='current_HISTORY_computation_not_independent_HISTORY_weights',
        history_capacity_prior_removed=True, latent_exchange_removed=True, final_fusion_absent=True,
        receiver_chunk_size=receiver_chunk_size, checkpoint_messages=checkpoint_messages,
        hidden_dim=result._hidden_dim, formal_contract_width=result._hidden_dim == 32,
        same_registered_parameters_caveat='semantic_parameters_not_active_in_HISTORY_or_M_ONLY',
        graph_scope='fixed_published_support_not_certified_physical_direction',
        event_scope='published_archive_attributes_conditional_offline')
