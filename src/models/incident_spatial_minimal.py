"""Parameter-pruned history-only B3-R control with its constrained exchange.

The model keeps the original backbone and P capacity prior, but registers no
incident encoder, TIID, report residual D or strength gate. Capacities and
exchange states are dimensionless latent features, not measured traffic units.
"""
import copy

import torch
from torch import nn

from src.models.igstgnn import IGSTGNN
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import (
    CONTEXT_SCHEMA, EPSILON, ReportCapacityResidual, node_capacity_to_edges,
    tensor_state_hash,
)


MAPPING_VERSION = 'b3r_to_history_prior_pruned_minimal_v1'
CUTOFF_CLOCK_POLICY = 'last_history_slot_plus_two_5minute_slots_with_day_rollover'


def cutoff_clock_from_history(history_data, *, num_feat=1, tpd=288):
    """Recover t0 from X's nominal clock, independently of incident contents.

    The frozen package has x_end=t0-10 minutes. X contains 12 consecutive
    five-minute slots. Floating-point roundoff in normalized clock channels is
    removed only after checking that they represent integer nominal slots.
    """
    require(tpd == 288 and history_data.ndim == 4 and history_data.shape[1] == 12
            and history_data.shape[2] > 0 and history_data.shape[3] == num_feat+2,
            'Minimal model requires the 12-slot five-minute history clock contract')
    clocks = history_data[:, -1, :, num_feat:num_feat+2]
    require(bool(torch.isfinite(clocks).all()), 'Non-finite history cutoff clock')
    require(torch.equal(clocks, clocks[:, :1].expand_as(clocks)),
            'History clock must agree across nodes')
    nominal = clocks[:, 0]*clocks.new_tensor([tpd, 7])
    rounded = nominal.round()
    require(bool((nominal-rounded).abs().le(3e-5).all()),
            'History clock must lie on the nominal five-minute/day grid')
    slot, day = rounded.long().unbind(-1)
    require(bool(((slot >= 0) & (slot < tpd) & (day >= 0) & (day < 7)).all()),
            'History cutoff clock index is out of range')
    future = slot+2
    return future.remainder(tpd), (day+torch.div(future, tpd, rounding_mode='floor')).remainder(7)


class LatestHistoryNormalization(nn.Module):
    """Keep the original state-mode ICSF normalization and no report weights."""
    def __init__(self, parent):
        super().__init__()
        self.output_norm = copy.deepcopy(parent.output_norm)
        self.train(parent.training)


class HistoryCapacityPrior(nn.Module):
    """P, inlet and exit copied with their original state-dictionary names."""
    def __init__(self, parent):
        super().__init__()
        self.latent = parent.latent
        self.report_path_enabled = False
        for name in ('P', 'inlet', 'exit'):
            setattr(self, name, copy.deepcopy(getattr(parent, name)))
        self.train(parent.training)

    def forward(self, history, tod, dow, graph):
        b, _, n, _ = history.shape
        require(n == graph.nodes, 'Capacity graph/node mismatch')
        clock = torch.cat((tod, dow), -1)[:, None].expand(-1, n, -1)
        summary = torch.cat((history[:, -1], history.mean(1), history[:, -1]-history[:, 0], clock), -1)
        base_logit = self.P(summary)
        node = EPSILON+(1-EPSILON)*base_logit.sigmoid()
        edge = node_capacity_to_edges(node, graph)
        src, dst = graph.edge_index
        boundary = torch.where(
            (src < 0)[None, :, None],
            self.inlet(history[:, -1]).sigmoid()[:, dst.clamp_min(0)],
            self.exit(history[:, -1]).sigmoid()[:, src.clamp_min(0)])
        boundary = torch.where(((src < 0) | (dst < 0))[None, :, None], boundary, 0.)
        # The compatibility diagnostic mask is always empty. It does not mean
        # this validation package contains naturally accident-free samples.
        return dict(schema=CONTEXT_SCHEMA, graph=graph, capacity=edge,
                    base_edge_capacity=edge, boundary=boundary,
                    base_node_capacity=node, node_capacity=node,
                    base_logit=base_logit, updated_logit=base_logit,
                    report_logit_delta=None, effective_logit_delta=torch.zeros_like(base_logit),
                    report_mask=history.new_zeros(b, n, 1), report_path_executed=False,
                    report_mask_scope='removed_report_path_no_model_report_input', records=[])


class MinimalIGSTGNN(IGSTGNN):
    """History-only forward; the removed incident paths cannot be re-enabled.

    Legacy keyword arguments remain accepted for the frozen training loop.
    ``incident_data`` and ``sensor_data`` are ignored, including their clocks.
    Report switches are type-checked but cannot recreate removed modules.
    """
    def forward(self, history_data, label=None, incident_data=None, sensor_data=None,
                forecast_delta=None, *, spatial_report_enabled=True,
                tiid_report_enabled=True, return_spatial_details=False):
        require(type(spatial_report_enabled) is bool and type(tiid_report_enabled) is bool,
                'Report switches must be bool')
        require(forecast_delta is None, 'Minimal model cannot combine with an external forecast residual')
        cutoff_tod, cutoff_dow = cutoff_clock_from_history(
            history_data, num_feat=self._model_args['num_feat'], tpd=self._model_args['tpd'])
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat = self._prepare_inputs(history_data)
        report_free_history = self.embedding(history_data)
        spatial_context = self.report_spatial_state(
            report_free_history, self.T_i_D_emb[cutoff_tod], self.D_i_W_emb[cutoff_dow],
            self.internal_spatial_graph)
        spatial_context['collect'] = return_spatial_details
        history_data = report_free_history.clone()
        history_data[:, -1] = self.icsf_module.output_norm(report_free_history[:, -1])
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u, node_embedding_d=node_embedding_d,
            history_data=history_data, time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)
        dif_forecasts, inh_forecasts = [], []
        residual = history_data
        for layer in self.layers:
            residual, dif_forecast, inh_forecast = layer(
                residual, dynamic_graph, static_graph, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, incident_inputs=None,
                spatial_context=spatial_context)
            dif_forecasts.append(dif_forecast)
            inh_forecasts.append(inh_forecast)
        prediction = self._decode_forecast(sum(dif_forecasts)+sum(inh_forecasts), None)
        return {'prediction': prediction, 'spatial': spatial_context} if return_spatial_details else prediction


def from_residual(model):
    """Independently prune B3-R while exactly retaining every active tensor.

    Conversion is valid for fresh initialization or trained weights. It does
    not imply that training the smaller model reproduces full B3-R training.
    Reference forward equality is B3-R with both report paths disabled, when
    its supplied cutoff clocks agree with X's frozen chronological contract.
    """
    require(getattr(model, '_internal_spatial_mode', None) == 'state_restricted'
            and isinstance(model.report_spatial_state, ReportCapacityResidual)
            and getattr(model.icsf_module, 'state_spatial_mode', False)
            and model._incident_routing == 'none' and model._time_response_mode == 'fixed'
            and model._incident_schema == 'report_location_v1'
            and not model.icsf_module.use_sensor_info,
            'Expected restricted report-only fixed-TIID B3-R architecture')
    parent_state = model.state_dict()
    parent_hash = tensor_state_hash(parent_state)
    # No construction/initialization draws are needed for a pure subset copy.
    with torch.inference_mode(False):
        result = copy.deepcopy(model)
        result.__class__ = MinimalIGSTGNN
        result.icsf_module = LatestHistoryNormalization(result.icsf_module)
        result.report_spatial_state = HistoryCapacityPrior(result.report_spatial_state)
        del result.tiid_module
    result._minimal_variant = 'P_only_no_TIID_no_D'
    copied = result.state_dict()
    require(set(copied) < set(parent_state), 'Minimal conversion failed to prune registered tensors')
    require(all(torch.equal(value, parent_state[name]) for name, value in copied.items()),
            'Retained parameters or buffers changed during minimal conversion')
    expected_prefixes = ('icsf_module.report_encoder.', 'icsf_module.k_proj.', 'tiid_module.',
                         'report_spatial_state.D.', 'report_spatial_state.gate.')
    removed = {name: value for name, value in parent_state.items() if name not in copied}
    require(all(name.startswith(expected_prefixes) for name in removed),
            'Unexpected active parameter or buffer removed')
    require(not any(name.startswith(expected_prefixes) for name in copied),
            'Unused incident parameters remain registered')
    require(tensor_state_hash(model.state_dict()) == parent_hash, 'Input parent model was modified')
    parameter_count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    groups = {}
    for name, parameter in model.named_parameters():
        if name not in copied:
            group = next(prefix.rstrip('.') for prefix in expected_prefixes if name.startswith(prefix))
            groups[group] = groups.get(group, 0)+parameter.numel()
    certificate = dict(
        mapping_version=MAPPING_VERSION, mode='P_only_no_TIID_no_D',
        cutoff_clock_policy=CUTOFF_CLOCK_POLICY,
        prediction_inputs=['historical_flow', 'historical_nominal_tod_dow', 'candidate_directed_graph'],
        prediction_equivalence='same_weights_B3R_spatial_report_enabled_false_tiid_report_enabled_false',
        equivalence_clock_condition='legacy_report_cutoff_clock_equals_last_history_clock_plus_two_slots',
        parent_state_sha256=parent_hash, retained_shared_state_sha256=tensor_state_hash(copied),
        retained_shared_state_specification={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                                             for name, value in copied.items()},
        parent_parameter_count=parameter_count(model), parameter_count=parameter_count(result),
        removed_parameter_count=parameter_count(model)-parameter_count(result),
        removed_parameter_groups=groups, removed_state_keys=sorted(removed),
        all_parameters_trainable=all(parameter.requires_grad for parameter in result.parameters()),
        report_inputs_consumed=False, native_graph_convolution_retained=True,
        history_capacity_prior_retained=True, constrained_exchange_retained=True,
        mapping={'retained_state': 'exact independent subset of parent state',
                 'latest_history_normalization': 'icsf_module.output_norm retained',
                 'P': 'unchanged', 'inlet': 'unchanged', 'exit': 'unchanged',
                 'TIID': 'deleted', 'D': 'deleted', 'report_encoder_and_K': 'deleted',
                 'strength_gate': 'deleted if present'})
    return result, certificate
