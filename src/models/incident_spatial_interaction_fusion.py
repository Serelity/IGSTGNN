"""Prototype score interactions on top of a retained, possibly trained F2.

The native MIN backbone runs once. The fixed-sum prediction is decoded with
the existing head, in the same normalization as historical_flow; it is not a
label or an individual component prediction. Existing F2 report scores remain
active. The new report interaction is an additional, independently switchable
score residual. This prototype has no training/evaluation runner.
"""
import copy
import math

import torch
import torch.nn.functional as F

from src.models.incident_component_interaction import InteractionScoreResidual
from src.models.incident_spatial import require
from src.models.incident_spatial_component_fusion import (
    ComponentFusion, ComponentFusionIGSTGNN, centered_bounded_delta,
)
from src.models.incident_spatial_forecast_fusion import (
    COMPONENT_ORDER, MODES, fusion_inputs, fuse_components,
)
from src.models.incident_spatial_minimal import cutoff_clock_from_history
from src.models.incident_spatial_residual import tensor_state_hash


MAPPING_VERSION = 'trained_F2_to_separable_score_interaction_prototype_v1'
FUSION_SCHEMA = 'MIN_component_score_interaction_prototype_v1'


def _center_added_scores(scores):
    """Center one new branch, making constant vectors exactly inactive.

    Subtracting an anchor first is algebraically ordinary mean centering. It
    also avoids a mean-reduction rounding error for a constant score vector.
    Original F2 scores are never centered by this function.
    """
    shifted = scores-scores[..., :1]
    return shifted-shifted.mean(-1, keepdim=True)


def _component_scores(router, raw, relation, semantic, dif, inh, *, gap, content_enabled):
    """Reuse F2's compact encodings and exact split-affine arithmetic order.

    Return pre-tanh scores, not an already bounded delta. No full forecast
    feature tensor with a ten-component axis is constructed.
    """
    require(len(dif) == len(inh) == 5, 'Component fusion requires five DIF and five INH components')
    components = (*dif, *inh)
    batch, _, nodes, _ = raw.shape
    shape = components[0].shape
    require(len(shape) == 4 and shape[0] == batch and shape[2:] == (nodes, router.forecast_features)
            and shape[1]*gap == router.horizon, 'Component fusion block/horizon axes mismatch')
    require(all(value.shape == shape and value.dtype == raw.dtype and value.device == raw.device
                for value in components), 'Component fusion component dtype/device/shape mismatch')
    require(relation.shape == (batch, nodes, 3) and semantic.shape == (batch, nodes, 16),
            'Component fusion context axes mismatch')
    if content_enabled:
        encoded = torch.stack([F.silu(router.component_projection(router.component_norm(value)))
                               for value in components], dim=-2)
        contrast = encoded-encoded.mean(-2, keepdim=True)
        log_rms = torch.stack([torch.log1p(torch.linalg.vector_norm(value, dim=-1)/math.sqrt(router.forecast_features))
                               for value in components], dim=-1)
    else:
        encoded = raw.new_zeros(batch, shape[1], nodes, 10, router.component_dim)
        contrast = torch.zeros_like(encoded)
        log_rms = raw.new_zeros(batch, shape[1], nodes, 10)
    summary = torch.cat((raw[:, -1], raw.mean(1), raw[:, -1]-raw[:, 0]), -1)
    state = F.silu(router.state_projection(router.state_norm(summary)))[:, None].expand(-1, router.horizon, -1, -1)
    position = router.horizon_embedding.weight[None, :, None].expand(batch, -1, nodes, -1)
    common_features = torch.cat((state, position, relation[:, None].expand(-1, router.horizon, -1, -1),
                                 semantic[:, None].expand(-1, router.horizon, -1, -1)), -1)
    affine, activation, output = router.condition
    start = router.content_width
    common_hidden = F.linear(common_features, affine.weight[:, start:-4], affine.bias)
    id_hidden = F.linear(router.component_embedding.weight, affine.weight[:, -4:])
    scores = []
    for index in range(10):
        content = torch.cat((encoded[..., index, :], contrast[..., index, :],
                             log_rms[..., index:index+1]), -1)
        content_hidden = F.linear(content, affine.weight[:, :start]).repeat_interleave(gap, 1)
        scores.append(output(activation(content_hidden+common_hidden+id_hidden[index])))
    return torch.cat(scores, -1), encoded, contrast, log_rms


class InteractionFusionIGSTGNN(ComponentFusionIGSTGNN):
    def forward(self, history_data, label=None, incident_data=None, sensor_data=None,
                forecast_delta=None, *, spatial_report_enabled=True, tiid_report_enabled=True,
                fusion_enabled=True, report_enabled=True, semantic_enabled=True,
                uniform_weights=False, content_enabled=True, return_spatial_details=False,
                history_interaction_enabled=True, report_interaction_enabled=True):
        require(all(type(value) is bool for value in (spatial_report_enabled, tiid_report_enabled,
                    fusion_enabled, report_enabled, semantic_enabled, uniform_weights, content_enabled,
                    return_spatial_details, history_interaction_enabled, report_interaction_enabled)),
                'Interaction fusion switches must be bool')
        require(forecast_delta is None, 'Interaction fusion cannot combine an external forecast residual')
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
        gap = self._model_args['gap']
        zero_scores = raw.new_zeros(batch, self.horizon, nodes, 10)
        history_scores, report_scores = zero_scores, zero_scores
        centered_history, centered_report = zero_scores, zero_scores
        interaction_diagnostics = None
        router_details = None
        history_active = fusion_enabled and history_interaction_enabled
        report_active = (fusion_enabled and report_interaction_enabled and report_enabled
                         and self._forecast_fusion_mode == 'event' and bool(support.any()))
        if fusion_enabled:
            scores, encoded, contrast, log_rms = _component_scores(
                self.forecast_fusion, raw, relation, semantic, dif, inh,
                gap=gap, content_enabled=content_enabled)
            if history_active or report_active:
                # A complete native prediction, not ten separate decoded
                # contributions. This adds only one call of the existing head.
                fixedsum_prediction = self._decode_forecast(sum(dif)+sum(inh), None)
                interaction = self.forecast_interaction(
                    component_encodings=encoded, historical_flow=history_data[..., :1],
                    fixed_sum_prediction=fixedsum_prediction,
                    horizon_features=self.forecast_fusion.horizon_embedding.weight,
                    report_features=torch.cat((relation, semantic), -1) if report_active else None,
                    support=support if report_active else None, gap=gap,
                    history_enabled=history_active, report_enabled=report_active,
                    collect=return_spatial_details)
                history_scores, report_scores = interaction['history_scores'], interaction['report_scores']
                centered_history = _center_added_scores(history_scores)
                centered_report = _center_added_scores(report_scores)
                # Keep the trained F2 scores exactly as they are. Center each
                # new branch independently before the one original transform.
                scores = scores+centered_history+centered_report
                interaction_diagnostics = interaction['diagnostics']
            delta, uncentered, common, scale = centered_bounded_delta(scores, self.forecast_fusion.max_delta)
            if return_spatial_details:
                with torch.no_grad():
                    router_details = dict(
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
        else:
            delta = zero_scores
        if uniform_weights:
            delta = delta.mean(-1, keepdim=True).expand_as(delta)
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
        with torch.no_grad():
            interaction_details = dict(
                history_enabled=history_active, report_enabled=report_active,
                report_is_additional_to_legacy_F2=True,
                score_policy='independently_center_new_branches_before_original_single_bounded_transform',
                history_score_abs_mean_by_horizon_component=history_scores.detach().abs().mean((0, 2)),
                report_score_abs_mean_by_horizon_component=report_scores.detach().abs().mean((0, 2)),
                centered_history_score_abs_mean_by_horizon_component=centered_history.detach().abs().mean((0, 2)),
                centered_report_score_abs_mean_by_horizon_component=centered_report.detach().abs().mean((0, 2)),
                diagnostics=interaction_diagnostics)
            for name, value in (('history', history_scores), ('report', report_scores),
                                ('centered_history', centered_history), ('centered_report', centered_report)):
                statistics[name+'_interaction_score_abs_sum'] = value.detach().abs().sum()
                statistics[name+'_interaction_score_abs_max'] = value.detach().abs().max()
        fusion = dict(schema=FUSION_SCHEMA, prototype=True, mode=self._forecast_fusion_mode,
            enabled=fusion_enabled, uniform_weights=uniform_weights,
            component_content_enabled=fusion_enabled and content_enabled,
            coefficient_policy='centered_relative_uniformly_bounded', max_delta=self.forecast_fusion.max_delta,
            component_order=list(COMPONENT_ORDER), weights=1+detached, support=support.detach(),
            relation=relation.detach(), report_available=available.detach(),
            event_context_used=bool(support.any()), semantic_context_used=semantic_used,
            weight_delta_by_horizon_component=detached.mean((0, 2)),
            weight_delta_abs_mean_by_horizon_component=detached.abs().mean((0, 2)),
            statistics=statistics, interaction=interaction_details,
            **fusion_details, **(router_details or {}))
        return dict(prediction=prediction, spatial=context, fusion=fusion)


def from_component_fusion(parent):
    """Independently extend an existing F2 without consuming its RNG/state."""
    require(type(parent) is ComponentFusionIGSTGNN and type(parent.forecast_fusion) is ComponentFusion
            and not hasattr(parent, 'forecast_interaction'), 'Expected an unadapted ComponentFusionIGSTGNN')
    require(parent._forecast_fusion_mode in MODES and len(parent.layers) == 5
            and parent.horizon == 12 and parent._model_args['gap'] == 3
            and parent._model_args['num_feat'] == 1,
            'Interaction fusion requires native five-layer 12-step gap-three flow-only F2')
    before = tensor_state_hash(parent.state_dict())
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        result = copy.deepcopy(parent)
        result.__class__ = InteractionFusionIGSTGNN
        result.forecast_interaction = InteractionScoreResidual(
            component_dim=parent.forecast_fusion.component_dim, rank=8, horizon_dim=4, report_dim=19)
        reference = next(result.parameters())
        result.forecast_interaction.to(device=reference.device, dtype=reference.dtype).train(parent.training)
        result.requires_grad_(True)
    old, current = parent.state_dict(), result.state_dict()
    retained = old.keys() <= current.keys() and all(torch.equal(value, current[name]) for name, value in old.items())
    require(retained, 'Interaction fusion changed retained F2 state')
    added = set(current)-set(old)
    require(added and all(name.startswith('forecast_interaction.') for name in added),
            'Unexpected added interaction fusion state')
    require(tensor_state_hash(parent.state_dict()) == before, 'Parent F2 was modified')
    count = lambda module: sum(value.numel() for value in module.parameters())
    return result, dict(mapping_version=MAPPING_VERSION, prototype=True, mode=parent._forecast_fusion_mode,
        rank=8, horizon_dim=4, report_dim=19, component_dim=parent.forecast_fusion.component_dim,
        retained_state_exact_verified=retained, parent_state_sha256=before,
        interaction_initial_sha256=tensor_state_hash(result.forecast_interaction.state_dict()),
        parent_parameter_count=count(parent), parameter_count=count(result),
        added_parameters=count(result)-count(parent), added_state_keys=sorted(added),
        all_parameters_trainable=all(value.requires_grad for value in result.parameters()),
        backbone_forwards=1, additional_native_head_forwards_when_interaction_active=1,
        component_gradients_detached=False, full_hidden_component_stack=False,
        legacy_report_scoring_retained=True, new_report_is_separable_residual=True,
        score_policy='independently_center_new_branches_before_original_single_bounded_transform',
        prediction_equivalence='zero_new_readouts_or_both_new_OFF_equals_same_weights_parent_F2',
        fusion_OFF_equivalence='same_current_weights_MIN',
        report_OFF_equivalence='all_legacy_and_new_report_inputs_disabled_history_interaction_retained',
        new_report_OFF_equivalence='only_new_report_interaction_disabled_legacy_F2_report_path_retained',
        model_scope='prototype_without_training_or_evaluation_runner')
