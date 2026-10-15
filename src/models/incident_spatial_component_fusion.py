"""Content-aware relative reweighting of MIN's native forecast features.

The scorer sees compact representations of the ten features it reweights.
It changes only the final fusion: graphs, backcasts, latent exchange and the
original nonlinear output head remain intact. Relative coefficients have zero
mean up to floating-point roundoff; this does not imply zero output change.
"""
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from src.models.incident_spatial import require
from src.models.incident_spatial_event_budget import EventSemanticEncoder
from src.models.incident_spatial_forecast_fusion import COMPONENT_ORDER, MODES, fusion_inputs, fuse_components
from src.models.incident_spatial_minimal import HistoryCapacityPrior, MinimalIGSTGNN, cutoff_clock_from_history
from src.models.incident_spatial_residual import tensor_state_hash


MAPPING_VERSION = 'minimal_content_aware_relative_forecast_fusion_v1'
FUSION_SCHEMA = 'MIN_component_content_relative_fusion_v1'


def centered_bounded_delta(scores, max_delta):
    """Center bounded scores and shrink each ten-component vector uniformly.

    No future targets or output errors enter this transformation. A single
    shared scale preserves relative directions rather than clipping entries.
    """
    raw = float(max_delta)*scores.tanh()
    common = raw.mean(-1, keepdim=True)
    centered = raw-common
    # clamp_min avoids dividing by zero without adding an output offset.
    scale = (centered.abs().amax(-1, keepdim=True)/float(max_delta)).clamp_min(1.)
    return centered/scale, raw, common, scale


class ComponentFusion(nn.Module):
    """Shared component scorer with compact block-content and common context."""
    def __init__(self, hidden, forecast_features, horizon, width, component_dim, context_dim, max_delta,
                 type_count, description_count):
        super().__init__()
        self.max_delta = float(max_delta)
        self.horizon, self.component_dim, self.context_dim = horizon, component_dim, context_dim
        self.forecast_features = forecast_features
        self.component_norm = nn.LayerNorm(forecast_features)
        self.component_projection = nn.Linear(forecast_features, component_dim)
        self.state_norm = nn.LayerNorm(3*hidden)
        self.state_projection = nn.Linear(3*hidden, context_dim)
        self.horizon_embedding = nn.Embedding(horizon, 4)
        self.component_embedding = nn.Embedding(len(COMPONENT_ORDER), 4)
        self.semantic_encoder = EventSemanticEncoder(type_count, description_count)
        # Feature order is content, content contrast, log RMS, common context,
        # horizon, relation, semantic encoding, then component identity.
        self.content_width = 2*component_dim+1
        self.condition = nn.Sequential(nn.Linear(self.content_width+context_dim+4+3+16+4, width),
                                       nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, raw, relation, semantic, dif, inh, *, gap=3, content_enabled=True, collect=False):
        require(type(content_enabled) is bool and type(collect) is bool, 'Component fusion switches must be bool')
        require(len(dif) == len(inh) == 5, 'Component fusion requires five DIF and five INH components')
        components = (*dif, *inh)
        batch, _, nodes, _ = raw.shape
        shape = components[0].shape
        require(len(shape) == 4 and shape[0] == batch and shape[2:] == (nodes, self.forecast_features)
                and shape[1]*gap == self.horizon, 'Component fusion block/horizon axes mismatch')
        require(all(value.shape == shape and value.dtype == raw.dtype and value.device == raw.device
                    for value in components), 'Component fusion component dtype/device/shape mismatch')
        require(relation.shape == (batch, nodes, 3) and semantic.shape == (batch, nodes, 16),
                'Component fusion context axes mismatch')
        if content_enabled:
            # Stack only compact encodings, never [B,H,N,10,hidden].
            encoded = torch.stack([F.silu(self.component_projection(self.component_norm(value)))
                                   for value in components], dim=-2)
            contrast = encoded-encoded.mean(-2, keepdim=True)
            log_rms = torch.stack([torch.log1p(torch.linalg.vector_norm(value, dim=-1)/math.sqrt(self.forecast_features))
                                   for value in components], dim=-1)
        else:
            encoded = raw.new_zeros(batch, shape[1], nodes, 10, self.component_dim)
            contrast = torch.zeros_like(encoded)
            log_rms = raw.new_zeros(batch, shape[1], nodes, 10)
        summary = torch.cat((raw[:, -1], raw.mean(1), raw[:, -1]-raw[:, 0]), -1)
        state = F.silu(self.state_projection(self.state_norm(summary)))[:, None].expand(-1, self.horizon, -1, -1)
        position = self.horizon_embedding.weight[None, :, None].expand(batch, -1, nodes, -1)
        common_features = torch.cat((state, position, relation[:, None].expand(-1, self.horizon, -1, -1),
                                     semantic[:, None].expand(-1, self.horizon, -1, -1)), -1)
        affine, activation, output = self.condition
        start = self.content_width
        # Algebraically the same shared Linear(concatenated features). Splitting
        # the affine avoids ten copies of the common expanded context.
        common_hidden = F.linear(common_features, affine.weight[:, start:-4], affine.bias)
        id_hidden = F.linear(self.component_embedding.weight, affine.weight[:, -4:])
        scores = []
        for index in range(10):
            content = torch.cat((encoded[..., index, :], contrast[..., index, :],
                                 log_rms[..., index:index+1]), -1)
            content_hidden = F.linear(content, affine.weight[:, :start]).repeat_interleave(gap, 1)
            scores.append(output(activation(content_hidden+common_hidden+id_hidden[index])))
        scores = torch.cat(scores, -1)
        delta, uncentered, common, scale = centered_bounded_delta(scores, self.max_delta)
        details = None
        if collect:
            with torch.no_grad():
                details = dict(
                    component_encoding_abs_mean_by_component=encoded.detach().abs().mean((0, 1, 2, 4)),
                    component_contrast_abs_mean_by_component=contrast.detach().abs().mean((0, 1, 2, 4)),
                    component_log_rms_mean_by_component=log_rms.detach().mean((0, 1, 2)),
                    router_statistics=dict(
                        component_content_calls=raw.new_tensor(int(content_enabled)),
                        component_encoding_positions=raw.new_tensor(encoded.numel()),
                        component_encoding_abs_sum=encoded.detach().abs().sum(),
                        component_encoding_abs_max=encoded.detach().abs().max(),
                        component_contrast_abs_sum=contrast.detach().abs().sum(),
                        component_contrast_abs_max=contrast.detach().abs().max(),
                        component_log_rms_sum=log_rms.detach().sum(),
                        component_log_rms_max=log_rms.detach().max(),
                        raw_common_delta_signed_sum=common.detach().sum(),
                        raw_common_delta_abs_sum=common.detach().abs().sum(),
                        raw_common_delta_abs_max=common.detach().abs().max(),
                        raw_delta_abs_max=uncentered.detach().abs().max(),
                        output_delta_component_sum_abs_max=delta.detach().sum(-1).abs().max(),
                        bounded_rescaled_positions=(scale.detach() > 1).sum().to(raw.dtype),
                        bounded_scale_min=scale.detach().reciprocal().min()))
        return delta, details


class ComponentFusionIGSTGNN(MinimalIGSTGNN):
    def forward(self, history_data, label=None, incident_data=None, sensor_data=None,
                forecast_delta=None, *, spatial_report_enabled=True, tiid_report_enabled=True,
                fusion_enabled=True, report_enabled=True, semantic_enabled=True,
                uniform_weights=False, content_enabled=True, return_spatial_details=False):
        require(all(type(value) is bool for value in (spatial_report_enabled, tiid_report_enabled,
                    fusion_enabled, report_enabled, semantic_enabled, uniform_weights, content_enabled,
                    return_spatial_details)), 'Component fusion switches must be bool')
        require(forecast_delta is None, 'Component fusion cannot combine an external forecast residual')
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
        router_details = None
        if fusion_enabled:
            delta, router_details = self.forecast_fusion(raw, relation, semantic, dif, inh,
                gap=self._model_args['gap'], content_enabled=content_enabled, collect=return_spatial_details)
        else:
            delta = raw.new_zeros(batch, self.horizon, nodes, 10)
        if uniform_weights:
            delta = delta.mean(-1, keepdim=True).expand_as(delta)
        gap = self._model_args['gap']
        if fusion_enabled:
            steps, fusion_details = fuse_components(dif, inh, delta, horizon=self.horizon, gap=gap,
                                                   collect=return_spatial_details)
            channels = torch.arange(self.horizon, device=steps.device).remainder(gap)
            channels = channels.view(1, -1, 1, 1).expand(batch, -1, nodes, 1)
            forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(steps))))
            prediction = forecast.gather(-1, channels)
        else:
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
        if router_details is not None:
            statistics.update(router_details.pop('router_statistics'))
        fusion = dict(schema=FUSION_SCHEMA, mode=self._forecast_fusion_mode,
            enabled=fusion_enabled, uniform_weights=uniform_weights,
            component_content_enabled=fusion_enabled and content_enabled,
            coefficient_policy='centered_relative_uniformly_bounded', max_delta=self.forecast_fusion.max_delta,
            component_order=list(COMPONENT_ORDER), weights=1+detached, support=support.detach(),
            relation=relation.detach(), report_available=available.detach(),
            event_context_used=bool(support.any()), semantic_context_used=semantic_used,
            weight_delta_by_horizon_component=detached.mean((0, 2)),
            weight_delta_abs_mean_by_horizon_component=detached.abs().mean((0, 2)),
            statistics=statistics, **fusion_details, **(router_details or {}))
        return dict(prediction=prediction, spatial=context, fusion=fusion)


def from_minimal(model, mode='state', width=16, component_dim=8, context_dim=8, max_delta=.1,
                 *, type_count=6, description_count=29):
    require(mode in MODES, 'Unknown component fusion mode')
    require(all(type(value) is int and value > 0 for value in (width, component_dim, context_dim)),
            'Component fusion dimensions must be positive integers')
    require(type(max_delta) in (int, float) and math.isfinite(max_delta) and 0 < max_delta < 1,
            'Component fusion max_delta must be finite and between zero and one')
    require(all(type(value) is int and value > 0 for value in (type_count, description_count)),
            'Component fusion category counts must be positive integers')
    require(type(model) is MinimalIGSTGNN and type(model.report_spatial_state) is HistoryCapacityPrior
            and not hasattr(model, 'forecast_fusion') and not hasattr(model, 'early_event_condition')
            and not hasattr(model, 'event_budget_adapter'), 'Expected an unadapted MIN architecture')
    require(len(model.layers) == 5 and model.horizon == 12 and model._model_args['gap'] == 3,
            'Component fusion requires the native five-layer 12-step gap-three architecture')
    before = tensor_state_hash(model.state_dict())
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        result = copy.deepcopy(model)
        result.__class__ = ComponentFusionIGSTGNN
        result._forecast_fusion_mode = mode
        result.forecast_fusion = ComponentFusion(model._hidden_dim, model.out_fc_1.in_features,
                                                model.horizon, width, component_dim,
                                                context_dim, max_delta, type_count, description_count)
        reference = next(result.parameters())
        result.forecast_fusion.to(device=reference.device, dtype=reference.dtype).train(model.training)
        result.requires_grad_(True)
    parent, current = model.state_dict(), result.state_dict()
    retained = parent.keys() <= current.keys() and all(torch.equal(value, current[name]) for name, value in parent.items())
    require(retained, 'Component fusion changed retained MIN state')
    added = set(current)-set(parent)
    require(added and all(name.startswith('forecast_fusion.') for name in added), 'Unexpected added component fusion state')
    require(tensor_state_hash(model.state_dict()) == before, 'Parent MIN was modified')
    count = lambda module: sum(value.numel() for value in module.parameters())
    return result, dict(mapping_version=MAPPING_VERSION, mode=mode, width=width,
        component_dim=component_dim, context_dim=context_dim, max_delta=float(max_delta),
        type_count=type_count, description_count=description_count,
        retained_state_exact_verified=retained, parent_state_sha256=before,
        fusion_initial_sha256=tensor_state_hash(result.forecast_fusion.state_dict()),
        parent_parameter_count=count(model), parameter_count=count(result), added_parameters=count(result)-count(model),
        added_state_keys=sorted(added), all_parameters_trainable=all(value.requires_grad for value in result.parameters()),
        component_order=list(COMPONENT_ORDER), real_horizon_weights=True, gap=3,
        prediction_equivalence='zero_initialization_or_fusion_OFF_equals_same_weights_MIN',
        report_OFF_equivalence='same_current_content_and_state_conditioned_fusion_not_historical_MIN',
        content_OFF_equivalence='same_weights_router_without_component_content_retains_history_id_horizon_report',
        graph_and_capacity_inputs='unchanged_original_MIN_history', backbone_forwards=1,
        backcasts_and_native_forecast_head='unchanged', new_capacity_path=False, new_output_bias=False,
        coefficient_policy='centered_relative_uniformly_bounded', component_gradients_detached=False,
        full_hidden_component_stack=False, same_registered_parameters_all_modes=True,
        model_scope='published_archive_attributes_conditional_offline')
