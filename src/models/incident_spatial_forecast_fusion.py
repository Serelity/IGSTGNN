"""Small state/event-conditioned reweighting of MIN's ten forecast components.

The backbone, its backcasts, graphs and latent exchange budgets run once and
remain unchanged. This is a forecast-feature fusion, not a traffic-capacity
model. Event OFF retains the same current state-conditioned fusion.
"""
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from src.models.incident_spatial import require
from src.models.incident_spatial_event_budget import EventSemanticEncoder
from src.models.incident_spatial_minimal import (
    HistoryCapacityPrior, MinimalIGSTGNN, cutoff_clock_from_history,
)
from src.models.incident_spatial_residual import tensor_state_hash


MAPPING_VERSION = 'minimal_ten_component_state_event_forecast_fusion_v1'
MODES = ('state', 'event')
COMPONENT_ORDER = tuple('DIF'+str(i) for i in range(5))+tuple('INH'+str(i) for i in range(5))
FUSION_SCHEMA = 'MIN_forecast_feature_fusion_v1'


class ForecastFusion(nn.Module):
    """Shared ten-logit scorer; STATE and EVENT register identical tensors."""
    def __init__(self, hidden, horizon, width, max_delta, type_count, description_count):
        super().__init__()
        self.max_delta = float(max_delta)
        self.horizon = horizon
        self.state_norm = nn.LayerNorm(3*hidden)
        self.horizon_embedding = nn.Embedding(horizon, 8)
        self.semantic_encoder = EventSemanticEncoder(type_count, description_count)
        self.condition = nn.Sequential(nn.Linear(3*hidden+8+3+16, width), nn.SiLU(),
                                       nn.Linear(width, len(COMPONENT_ORDER)))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, raw, relation, semantic):
        batch, _, nodes, _ = raw.shape
        summary = torch.cat((raw[:, -1], raw.mean(1), raw[:, -1]-raw[:, 0]), -1)
        state = self.state_norm(summary)[:, None].expand(-1, self.horizon, -1, -1)
        position = self.horizon_embedding.weight[None, :, None].expand(batch, -1, nodes, -1)
        features = torch.cat((state, position, relation[:, None].expand(-1, self.horizon, -1, -1),
                              semantic[:, None].expand(-1, self.horizon, -1, -1)), -1)
        return self.max_delta*self.condition(features).tanh()


def fusion_inputs(model, raw, incident, *, report_enabled, semantic_enabled):
    """Only original report-supported nodes receive relation/semantic inputs.

    State routing always remains available. An absent report means a zero event
    context, rather than a learned UNK context at every station. Explicit UNK
    tokens remain available through the semantic encoder's validity flags.
    """
    batch, _, nodes, _ = raw.shape
    relation = raw.new_zeros(batch, nodes, 3)
    semantic = raw.new_zeros(batch, nodes, 16)
    support = torch.zeros(batch, nodes, 1, dtype=torch.bool, device=raw.device)
    available = torch.zeros(batch, dtype=torch.bool, device=raw.device)
    semantic_used = False
    if model._forecast_fusion_mode == 'state' or not report_enabled or incident is None:
        return relation, semantic, support, available, semantic_used
    require(isinstance(incident, dict), 'Forecast fusion event input must be a dictionary')
    value = incident.get('relation')
    require(isinstance(value, torch.Tensor) and value.shape == (batch, nodes, 3)
            and value.dtype == torch.float32 and value.device == raw.device
            and bool(torch.isfinite(value).all()),
            'Forecast fusion relation must be finite Float32[B,N,3] on model device')
    flags = []
    for name in ('report_present', 'relation_available'):
        flag = incident.get(name)
        require(isinstance(flag, torch.Tensor) and flag.shape == (batch,)
                and flag.dtype == torch.bool and flag.device == raw.device,
                'Forecast fusion '+name+' must be Bool[B] on model device')
        flags.append(flag)
    available = flags[0] & flags[1]
    support = (value.abs().sum(-1, keepdim=True) > 0) & available[:, None, None]
    relation = torch.where(support, value.to(raw.dtype), 0.)
    semantic_used = semantic_enabled and bool(support.any())
    if semantic_used:
        semantic_report = dict(incident)
        row_supported = support.any(1).squeeze(-1)
        for name in ('type', 'description'):
            valid = incident.get(name+'_valid')
            require(isinstance(valid, torch.Tensor) and valid.shape == (batch,)
                    and valid.dtype == torch.bool and valid.device == raw.device,
                    'Forecast fusion '+name+'_valid must be Bool[B] on model device')
            semantic_report[name+'_valid'] = valid & row_supported
        encoded = model.forecast_fusion.semantic_encoder(semantic_report, batch, raw.device, enabled=True).to(raw.dtype)
        semantic = torch.where(support, encoded[:, None].expand(-1, nodes, -1), 0.)
    return relation, semantic, support, available, semantic_used


def fuse_components(dif, inh, delta, *, horizon, gap, collect=False):
    """Expand the native block sum once, then add weighted component increments.

    Components stay in block form for multiplication. No tensor with axes
    [B,H,N,10,F] is stacked or retained. The sum order of the zero-delta base
    exactly matches MIN, rather than reassociating its ten summands.
    """
    require(len(dif) == len(inh) == 5, 'Forecast fusion requires five DIF and five INH components')
    base = sum(dif)+sum(inh)
    batch, blocks, nodes, features = base.shape
    require(horizon == blocks*gap and delta.shape == (batch, horizon, nodes, 10),
            'Forecast fusion block/horizon/component axes mismatch')
    require(all(value.shape == base.shape and value.dtype == base.dtype and value.device == base.device
                for value in (*dif, *inh)) and delta.dtype == base.dtype and delta.device == base.device,
            'Forecast fusion component dtype/device/shape mismatch')
    expanded = base.repeat_interleave(gap, dim=1)
    increment = torch.zeros_like(expanded)
    component_means = []
    for index, value in enumerate((*dif, *inh)):
        # Broadcasting the block tensor avoids ten expanded component copies.
        adjustment = (value[:, :, None]*delta[..., index].reshape(batch, blocks, gap, nodes, 1))
        adjustment = adjustment.reshape(batch, horizon, nodes, features)
        increment = increment+adjustment
        if collect:
            component_means.append(adjustment.detach().abs().mean((0, 2, 3)))
    details = None
    if collect:
        details = dict(
            component_increment_abs_mean_by_horizon_component=torch.stack(component_means, -1),
            weighted_hidden_change_abs_mean_by_horizon=increment.detach().abs().mean((0, 2, 3)),
            weighted_hidden_change_abs_max=increment.detach().abs().max(),
            hidden_increment_abs_sum=increment.detach().abs().sum())
    return expanded+increment, details


class ForecastFusionIGSTGNN(MinimalIGSTGNN):
    def forward(self, history_data, label=None, incident_data=None, sensor_data=None,
                forecast_delta=None, *, spatial_report_enabled=True, tiid_report_enabled=True,
                fusion_enabled=True, report_enabled=True, semantic_enabled=True,
                uniform_weights=False, return_spatial_details=False):
        require(all(type(value) is bool for value in (spatial_report_enabled, tiid_report_enabled,
                    fusion_enabled, report_enabled, semantic_enabled, uniform_weights, return_spatial_details)),
                'Forecast fusion switches must be bool')
        require(forecast_delta is None, 'Forecast fusion cannot combine an external forecast residual')
        tod, dow = cutoff_clock_from_history(history_data, num_feat=self._model_args['num_feat'],
            tpd=self._model_args['tpd'], offset_slots=getattr(self, '_minimal_clock_offset_slots', 2))
        history, node_u, node_d, tod_features, dow_features = self._prepare_inputs(history_data)
        raw = self.embedding(history)
        context = self.report_spatial_state(raw, self.T_i_D_emb[tod], self.D_i_W_emb[dow],
                                            self.internal_spatial_graph)
        context['collect'] = return_spatial_details
        hidden = raw.clone()
        hidden[:, -1] = self.icsf_module.output_norm(raw[:, -1])
        static, dynamic = self._graph_constructor(node_embedding_u=node_u, node_embedding_d=node_d,
            history_data=hidden, time_in_day_feat=tod_features, day_in_week_feat=dow_features)
        dif, inh, residual = [], [], hidden
        for layer in self.layers:
            residual, df, inf = layer(residual, dynamic, static, node_u, node_d, tod_features, dow_features,
                                     incident_inputs=None, spatial_context=context)
            dif.append(df)
            inh.append(inf)
        relation, semantic, support, available, semantic_used = fusion_inputs(
            self, raw, incident_data, report_enabled=report_enabled and fusion_enabled,
            semantic_enabled=semantic_enabled)
        batch, _, nodes, _ = raw.shape
        delta = (self.forecast_fusion(raw, relation, semantic) if fusion_enabled
                 else raw.new_zeros(batch, self.horizon, nodes, 10))
        if uniform_weights:
            delta = delta.mean(-1, keepdim=True).expand_as(delta)
        gap = self._model_args['gap']
        if fusion_enabled:
            steps, fusion_details = fuse_components(dif, inh, delta, horizon=self.horizon, gap=gap,
                                                   collect=return_spatial_details)
            # The features are already real-step indexed. Calling _decode_forecast
            # here would repeat them a second time and corrupt the horizon axis.
            channels = torch.arange(self.horizon, device=steps.device).remainder(gap)
            channels = channels.view(1, -1, 1, 1).expand(batch, -1, nodes, 1)
            forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(steps))))
            prediction = forecast.gather(-1, channels)
        else:
            # OFF is the current MIN head, with no ten zero multiplications.
            prediction = self._decode_forecast(sum(dif)+sum(inh), None)
            fusion_details = dict(
                component_increment_abs_mean_by_horizon_component=raw.new_zeros(self.horizon, 10),
                weighted_hidden_change_abs_mean_by_horizon=raw.new_zeros(self.horizon),
                weighted_hidden_change_abs_max=raw.new_zeros(()), hidden_increment_abs_sum=raw.new_zeros(()))
        if not return_spatial_details:
            return prediction
        detached = delta.detach()
        statistics = dict(
            fusion_calls=raw.new_tensor(int(fusion_enabled)), node_positions=raw.new_tensor(batch*nodes),
            supported_node_positions=support.sum().to(raw.dtype),
            horizon_component_positions=raw.new_tensor(delta.numel()),
            weight_delta_abs_sum=detached.abs().sum(), weight_delta_abs_max=detached.abs().max(),
            weight_delta_signed_sum=detached.sum(),
            hidden_increment_abs_sum=fusion_details['hidden_increment_abs_sum'],
            hidden_increment_abs_max=fusion_details['weighted_hidden_change_abs_max'])
        fusion = dict(schema=FUSION_SCHEMA, mode=self._forecast_fusion_mode,
            enabled=fusion_enabled, uniform_weights=uniform_weights,
            component_order=list(COMPONENT_ORDER), weights=1+detached, support=support.detach(),
            relation=relation.detach(), report_available=available.detach(),
            event_context_used=bool(support.any()), semantic_context_used=semantic_used,
            weight_delta_by_horizon_component=detached.mean((0, 2)),
            weight_delta_abs_mean_by_horizon_component=detached.abs().mean((0, 2)),
            statistics=statistics, **fusion_details)
        return dict(prediction=prediction, spatial=context, fusion=fusion)


def from_minimal(model, mode='state', width=32, max_delta=.1, *, type_count=6, description_count=29):
    require(mode in MODES, 'Unknown forecast fusion mode')
    require(type(width) is int and width > 0, 'Forecast fusion width must be a positive integer')
    require(type(max_delta) in (int, float) and math.isfinite(max_delta) and 0 < max_delta < 1,
            'Forecast fusion max_delta must be finite and between zero and one')
    require(all(type(value) is int and value > 0 for value in (type_count, description_count)),
            'Forecast fusion category counts must be positive integers')
    require(type(model) is MinimalIGSTGNN and type(model.report_spatial_state) is HistoryCapacityPrior
            and not hasattr(model, 'forecast_fusion') and not hasattr(model, 'early_event_condition')
            and not hasattr(model, 'event_budget_adapter'), 'Expected an unadapted MIN architecture')
    require(len(model.layers) == 5 and model.horizon == 12 and model._model_args['gap'] == 3,
            'Forecast fusion requires the native five-layer 12-step gap-three architecture')
    before = tensor_state_hash(model.state_dict())
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        result = copy.deepcopy(model)
        result.__class__ = ForecastFusionIGSTGNN
        result._forecast_fusion_mode = mode
        result.forecast_fusion = ForecastFusion(model._hidden_dim, model.horizon, width, max_delta,
                                                type_count, description_count)
        reference = next(result.parameters())
        result.forecast_fusion.to(device=reference.device, dtype=reference.dtype).train(model.training)
        result.requires_grad_(True)
    parent, current = model.state_dict(), result.state_dict()
    retained = parent.keys() <= current.keys() and all(torch.equal(value, current[name]) for name, value in parent.items())
    require(retained, 'Forecast fusion changed retained MIN state')
    added = set(current)-set(parent)
    require(added and all(name.startswith('forecast_fusion.') for name in added), 'Unexpected added forecast fusion state')
    require(tensor_state_hash(model.state_dict()) == before, 'Parent MIN was modified')
    count = lambda module: sum(value.numel() for value in module.parameters())
    return result, dict(mapping_version=MAPPING_VERSION, mode=mode, width=width, max_delta=float(max_delta),
        type_count=type_count, description_count=description_count,
        retained_state_exact_verified=retained, parent_state_sha256=before,
        fusion_initial_sha256=tensor_state_hash(result.forecast_fusion.state_dict()),
        parent_parameter_count=count(model), parameter_count=count(result), added_parameters=count(result)-count(model),
        added_state_keys=sorted(added), all_parameters_trainable=all(value.requires_grad for value in result.parameters()),
        component_order=list(COMPONENT_ORDER), real_horizon_weights=True, gap=3,
        prediction_equivalence='zero_initialization_or_fusion_OFF_equals_same_weights_MIN',
        report_OFF_equivalence='same_current_state_conditioned_fusion_not_historical_MIN',
        graph_and_capacity_inputs='unchanged_original_MIN_history', backbone_forwards=1,
        backcasts_and_native_forecast_head='unchanged', new_capacity_path=False, new_output_bias=False,
        same_registered_parameters_all_modes=True, model_scope='published_archive_attributes_conditional_offline')
