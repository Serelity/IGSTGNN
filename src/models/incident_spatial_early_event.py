"""One bounded report condition at MIN's latest historical hidden slot.

The original history prior, boundary budgets and graph construction consume
their original inputs. Only the representation sent into the five native
decomposition layers is modulated. Computational depth is not physical time.
"""
import copy
import math

import torch
from torch import nn

from src.models.incident_spatial import require
from src.models.incident_spatial_event_budget import EventSemanticEncoder
from src.models.incident_spatial_minimal import (
    HistoryCapacityPrior, MinimalIGSTGNN, cutoff_clock_from_history,
)
from src.models.incident_spatial_residual import tensor_state_hash


MAPPING_VERSION = 'minimal_latest_slot_multiplicative_event_v1'
MODES = ('state', 'location', 'event')
DIAGNOSTIC_SCOPE = 'single_latest_history_slot_direct_condition_not_physical_capacity'


class EarlyEventCondition(nn.Module):
    """Identical registered parameters in all three conditioning controls."""
    def __init__(self, hidden, width, max_scale, type_count, description_count):
        super().__init__()
        self.hidden, self.max_scale = hidden, float(max_scale)
        self.state_norm = nn.LayerNorm(3*hidden)
        self.semantic_encoder = EventSemanticEncoder(type_count, description_count)
        self.condition = nn.Sequential(nn.Linear(3*hidden+3+16, width), nn.SiLU(),
                                       nn.Linear(width, hidden))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, raw_history, relation, semantic):
        summary = torch.cat((raw_history[:, -1], raw_history.mean(1),
                             raw_history[:, -1]-raw_history[:, 0]), -1)
        features = torch.cat((self.state_norm(summary), relation,
                             semantic[:, None].expand(-1, raw_history.shape[2], -1)), -1)
        return self.max_scale*self.condition(features).tanh()


def condition_inputs(model, raw, incident, *, enabled, semantic_enabled):
    """Missing relation evidence disables the report arm, rather than asserting no accident."""
    batch, _, nodes, _ = raw.shape
    relation = raw.new_zeros(batch, nodes, 3)
    support = torch.zeros(batch, nodes, 1, device=raw.device, dtype=torch.bool)
    available = torch.zeros(batch, device=raw.device, dtype=torch.bool)
    encoder = model.early_event_condition.semantic_encoder
    read_semantics = False
    if enabled and model._early_event_mode == 'state':
        # A state-only control uses all stations and consumes no incident value.
        support.fill_(True)
    elif enabled and incident is not None:
        require(isinstance(incident, dict), 'Early event input must be a dictionary')
        value = incident.get('relation')
        require(isinstance(value, torch.Tensor) and value.shape == (batch, nodes, 3)
                and value.dtype == torch.float32 and value.device == raw.device
                and bool(torch.isfinite(value).all()),
                'Early event relation must be finite Float32[B,N,3] on model device')
        flags = []
        for name in ('report_present', 'relation_available'):
            flag = incident.get(name)
            require(isinstance(flag, torch.Tensor) and flag.shape == (batch,)
                    and flag.dtype == torch.bool and flag.device == raw.device,
                    'Early event '+name+' must be Bool[B] on model device')
            flags.append(flag)
        available = flags[0] & flags[1]
        support = (value.abs().sum(-1, keepdim=True) > 0) & available[:, None, None]
        relation = value.to(raw.dtype)
        # An unavailable row must not affect the feature input of an active row.
        relation = torch.where(support, relation, 0.)
        read_semantics = bool(support.any()) and model._early_event_mode == 'event' and semantic_enabled
    semantic = encoder(incident if read_semantics else {}, batch, raw.device,
                       enabled=read_semantics).to(raw.dtype)
    return relation, semantic, support, available, read_semantics


class EarlyEventIGSTGNN(MinimalIGSTGNN):
    def forward(self, history_data, label=None, incident_data=None, sensor_data=None,
                forecast_delta=None, *, spatial_report_enabled=True,
                tiid_report_enabled=True, report_enabled=True, semantic_enabled=True,
                return_spatial_details=False):
        require(all(type(value) is bool for value in (spatial_report_enabled, tiid_report_enabled,
                    report_enabled, semantic_enabled, return_spatial_details)), 'Early event switches must be bool')
        require(forecast_delta is None, 'Early event cannot combine an external forecast residual')
        tod, dow = cutoff_clock_from_history(history_data, num_feat=self._model_args['num_feat'],
            tpd=self._model_args['tpd'], offset_slots=getattr(self, '_minimal_clock_offset_slots', 2))
        history, node_u, node_d, tod_features, dow_features = self._prepare_inputs(history_data)
        raw = self.embedding(history)
        batch, _, nodes, _ = raw.shape
        context = self.report_spatial_state(raw, self.T_i_D_emb[tod], self.D_i_W_emb[dow],
                                            self.internal_spatial_graph)
        context['collect'] = return_spatial_details
        base_hidden = raw.clone()
        base_hidden[:, -1] = self.icsf_module.output_norm(raw[:, -1])
        # Keep graph inputs identical to MIN: no second report injection route.
        static, dynamic = self._graph_constructor(node_embedding_u=node_u, node_embedding_d=node_d,
            history_data=base_hidden, time_in_day_feat=tod_features, day_in_week_feat=dow_features)
        relation, semantic, support, available, read_semantics = condition_inputs(
            self, raw, incident_data, enabled=report_enabled, semantic_enabled=semantic_enabled)
        executed = bool(support.any())
        gamma = self.early_event_condition(raw, relation, semantic) if executed else torch.zeros_like(raw[:, -1])
        effective_gamma = gamma*support.to(raw.dtype)
        hidden = base_hidden.clone()
        # Addition is chosen so an exactly zero gamma preserves every input bit.
        change = base_hidden[:, -1]*effective_gamma
        hidden[:, -1] = base_hidden[:, -1]+change
        context.update(early_event_mode=self._early_event_mode,
                       early_event_diagnostic_scope=DIAGNOSTIC_SCOPE,
                       early_event_support=support.detach(), early_event_relation=relation.detach(),
                       early_event_semantic=semantic.detach(), early_event_gamma=effective_gamma.detach(),
                       early_event_direct_change=change.detach(), early_event_report_available=available.detach(),
                       early_event_semantic_enabled=read_semantics)
        context['early_event_diagnostics'] = dict(
            injection_calls=raw.new_tensor(int(executed)), node_positions=raw.new_tensor(batch*nodes),
            supported_node_positions=support.sum().to(raw.dtype),
            gamma_positions=raw.new_tensor(effective_gamma.numel()),
            gamma_abs_sum=effective_gamma.detach().abs().sum(), gamma_abs_max=effective_gamma.detach().abs().max(),
            gamma_saturation_count=(effective_gamma.detach().abs() >= .99*self.early_event_condition.max_scale).sum().to(raw.dtype),
            change_abs_sum=change.detach().abs().sum(), change_signed_sum=change.detach().sum(),
            change_abs_max=change.detach().abs().max())
        dif, inh, residual = [], [], hidden
        for layer in self.layers:
            residual, df, inf = layer(residual, dynamic, static, node_u, node_d, tod_features, dow_features,
                                     incident_inputs=None, spatial_context=context)
            dif.append(df)
            inh.append(inf)
        prediction = self._decode_forecast(sum(dif)+sum(inh), None)
        return dict(prediction=prediction, spatial=context) if return_spatial_details else prediction


def from_minimal(model, mode='event', width=32, max_scale=.25, *, type_count=6, description_count=29):
    require(mode in MODES, 'Unknown early event mode')
    require(type(width) is int and width > 0, 'Early event width must be a positive integer')
    require(type(max_scale) in (int, float) and math.isfinite(max_scale) and 0 < max_scale < 1,
            'Early event max_scale must be finite and between zero and one')
    require(all(type(value) is int and value > 0 for value in (type_count, description_count)),
            'Early event category counts must be positive integers')
    require(type(model) is MinimalIGSTGNN and type(model.report_spatial_state) is HistoryCapacityPrior
            and not hasattr(model, 'early_event_condition') and not hasattr(model, 'event_budget_adapter'),
            'Expected an unadapted MIN architecture')
    before = tensor_state_hash(model.state_dict())
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        result = copy.deepcopy(model)
        result.__class__ = EarlyEventIGSTGNN
        result._early_event_mode = mode
        result.early_event_condition = EarlyEventCondition(model._hidden_dim, width, max_scale,
                                                          type_count, description_count)
        reference = next(result.parameters())
        result.early_event_condition.to(device=reference.device, dtype=reference.dtype).train(model.training)
        result.requires_grad_(True)
    parent, current = model.state_dict(), result.state_dict()
    retained = parent.keys() <= current.keys() and all(torch.equal(value, current[name]) for name, value in parent.items())
    require(retained, 'Early event condition changed retained MIN state')
    added = set(current)-set(parent)
    require(added and all(name.startswith('early_event_condition.') for name in added), 'Unexpected added early event state')
    require(tensor_state_hash(model.state_dict()) == before, 'Parent MIN was modified')
    count = lambda module: sum(value.numel() for value in module.parameters())
    return result, dict(mapping_version=MAPPING_VERSION, mode=mode, width=width, max_scale=float(max_scale),
        type_count=type_count, description_count=description_count,
        retained_state_exact_verified=retained, parent_state_sha256=before,
        condition_initial_sha256=tensor_state_hash(result.early_event_condition.state_dict()),
        parent_parameter_count=count(model), parameter_count=count(result), added_parameters=count(result)-count(model),
        added_state_keys=sorted(added), all_parameters_trainable=all(value.requires_grad for value in result.parameters()),
        prediction_equivalence='zero_initialization_or_report_OFF_equals_same_weights_MIN',
        injection_site='latest_normalized_history_slot_before_native_decomposition_layers',
        graph_inputs='unchanged_original_MIN_normalized_history',
        prior_and_boundary_inputs='unchanged_original_MIN_embedded_history',
        support_policy='all_stations_no_report_inputs' if mode == 'state' else 'report_present_and_relation_available_and_nonzero_relation',
        model_scope='published_archive_attributes_conditional_offline',
        same_registered_parameters_all_modes=True, computational_injections_per_forward=1,
        new_capacity_path=False, new_output_bias=False)
