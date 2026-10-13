"""A single shared event adapter changes only internal latent edge budgets.

Each exchange substep conditions its original P budget on current edge states.
Layers and substeps are computational positions, not physical time. Four latent
budgets have no identified traffic units or accident-capacity causal meaning.
"""
import copy

import torch
from torch import nn

from src.models.incident_spatial import SpatialFeatureExchange, exchange_step, require, safe_scale
from src.models.incident_spatial_minimal import (
    MinimalIGSTGNN, HistoryCapacityPrior, cutoff_clock_from_history,
)
from src.models.incident_spatial_residual import EPSILON, tensor_state_hash


MAPPING_VERSION = 'minimal_to_shared_event_edge_budget_v1'
MODES = ('location', 'semantic')
DIAGNOSTIC_SCOPE = 'same_current_state_C_vs_C0_with_each_own_proposal_not_full_OFF_trajectory'


def signed_budget_update(base, adjustment, mask):
    """Zero is exactly identity; both signs remain within [epsilon, one]."""
    require(base.shape == adjustment.shape and mask.shape == base.shape[:-1]+(1,),
            'Event budget/mask axes mismatch')
    room = torch.where(adjustment >= 0, 1-base, base-EPSILON)
    return base+mask.to(base.dtype)*adjustment*room


class EventSemanticEncoder(nn.Module):
    def __init__(self, type_count, description_count):
        super().__init__()
        self.type_count, self.description_count = type_count, description_count
        self.type_embedding = nn.Embedding(type_count+1, 4)
        self.description_embedding = nn.Embedding(description_count+1, 8)
        self.projection = nn.Sequential(nn.Linear(14, 16), nn.ReLU())

    def forward(self, report, batch, device, *, enabled=True):
        if enabled:
            values, flags = [], []
            for field, count in (('type', self.type_count), ('description', self.description_count)):
                value, valid = report.get(field), report.get(field+'_valid')
                require(isinstance(value, torch.Tensor) and value.shape == (batch,)
                        and value.dtype == torch.long and value.device == device,
                        'Event '+field+' must be Long[B] on the model device')
                require(isinstance(valid, torch.Tensor) and valid.shape == (batch,)
                        and valid.dtype == torch.bool and valid.device == device,
                        'Event '+field+'_valid must be Bool[B] on the model device')
                require(bool(((value[valid] >= 0) & (value[valid] < count)).all()),
                        'Event '+field+' valid category is out of range')
                values.append(torch.where(valid, value, count))
                flags.append(valid)
        else:
            values = [torch.full((batch,), count, device=device, dtype=torch.long)
                      for count in (self.type_count, self.description_count)]
            flags = [torch.zeros(batch, device=device, dtype=torch.bool)]*2
        embedding = torch.cat((self.type_embedding(values[0]), self.description_embedding(values[1]),
                               torch.stack(flags, -1).to(self.type_embedding.weight.dtype)), -1)
        return self.projection(embedding)


class SharedEventBudgetAdapter(nn.Module):
    def __init__(self, hidden, latent, mode, type_count, description_count):
        super().__init__()
        self.mode, self.hidden, self.latent = mode, hidden, latent
        # Identical A dimensions and initialization in both modes. L's E is zero.
        self.A = nn.Sequential(nn.Linear(2*hidden+3*latent+6+16, 16), nn.ReLU(), nn.Linear(16, latent))
        nn.init.zeros_(self.A[-1].weight)
        nn.init.zeros_(self.A[-1].bias)
        if mode == 'semantic':
            with torch.random.fork_rng(devices=[]):
                self.semantic_encoder = EventSemanticEncoder(type_count, description_count)

    def forward(self, hidden, state, base, relation, embedding, graph):
        src, dst = graph.edge_index
        features = torch.cat((hidden[:, src.clamp_min(0)], hidden[:, dst.clamp_min(0)],
                              state[:, src.clamp_min(0)], state[:, dst.clamp_min(0)], base,
                              relation[:, src.clamp_min(0)], relation[:, dst.clamp_min(0)],
                              embedding[:, None].expand(-1, graph.edges, -1)), -1)
        return self.A(features).tanh()


def event_context(model, history, incident, *, report_enabled, semantic_enabled):
    """Build forward-local values; unavailable evidence does not become no event."""
    batch, _, nodes, _ = history.shape
    graph, adapter = model.internal_spatial_graph, model.event_budget_adapter
    relation = history.new_zeros(batch, nodes, 3)
    node_support = torch.zeros(batch, nodes, 1, device=history.device, dtype=torch.bool)
    available = torch.zeros(batch, device=history.device, dtype=torch.bool)
    embedding = history.new_zeros(batch, 16)
    if report_enabled and incident is not None:
        require(isinstance(incident, dict), 'Event input must be a dictionary')
        raw_relation = incident.get('relation')
        require(isinstance(raw_relation, torch.Tensor) and raw_relation.shape == (batch, nodes, 3)
                and raw_relation.dtype == torch.float32 and raw_relation.device == history.device
                and bool(torch.isfinite(raw_relation).all()),
                'Event relation must be finite Float32[B,N,3] on the model device')
        flags = []
        for field in ('relation_available', 'report_present'):
            value = incident.get(field)
            require(isinstance(value, torch.Tensor) and value.shape == (batch,)
                    and value.dtype == torch.bool and value.device == history.device,
                    'Event '+field+' must be Bool[B] on the model device')
            flags.append(value)
        available = flags[0] & flags[1]
        node_support = (raw_relation.abs().sum(-1, keepdim=True) > 0) & available[:, None, None]
        relation = raw_relation.to(history.dtype)
        if adapter.mode == 'semantic':
            embedding = adapter.semantic_encoder(incident, batch, history.device, enabled=semantic_enabled)
    src, dst = graph.edge_index
    internal = (src >= 0) & (dst >= 0)
    edge_support = (node_support[:, src.clamp_min(0)] | node_support[:, dst.clamp_min(0)]) & internal[None, :, None]
    return dict(event_budget_adapter=adapter, event_budget_enabled=bool(edge_support.any()),
                event_relation=relation, event_embedding=embedding, event_edge_support=edge_support,
                event_budget_records=[], event_budget_diagnostic_scope=DIAGNOSTIC_SCOPE,
                event_semantic_enabled=bool(adapter.mode == 'semantic' and semantic_enabled and edge_support.any()),
                report_mask=node_support.to(history.dtype), report_path_executed=bool(edge_support.any()),
                event_report_available=available)


def spatial_statistics(state, initial, proposal, capacity, record, graph, internal, eta):
    """Original collect ordering, denominators and balance checks stay intact."""
    eligible = graph.operator_mask[None, :, None].expand_as(state)
    displacement = record['state']-state
    sending = graph.aggregate(torch.minimum(proposal, capacity), incoming=False) > state+1e-6
    capped = torch.minimum(proposal, capacity)
    src, dst = graph.edge_index
    scaled = capped*torch.where((src >= 0)[None, :, None],
        safe_scale(state, graph.aggregate(capped, incoming=False))[:, src.clamp_min(0)], 1.)
    receiving = graph.aggregate(scaled, incoming=True) > 1-state+1e-6
    chosen, clip = record['state'][eligible], record['clip'][eligible]
    flux64 = record['flux'].double()
    external64 = flux64[:, src < 0].sum(1)-flux64[:, dst < 0].sum(1)
    balance64 = (record['state'].double()-state.double()).sum(1)-eta*external64
    scale64 = (record['inflow'].double().abs()+record['outflow'].double().abs()).sum(1).max()
    return torch.stack((eligible.sum().to(state.dtype), (clip.abs()>1e-6).sum().to(state.dtype),
        clip.abs().sum(), clip.abs().max() if clip.numel() else state.new_zeros(()),
        chosen.sum(), chosen.min() if chosen.numel() else state.new_zeros(()),
        chosen.max() if chosen.numel() else state.new_zeros(()), displacement[eligible].abs().sum(),
        record['internal_flux'].sum(), record['inlet_flux'].sum(), record['exit_flux'].sum(),
        (proposal>capacity+1e-6).sum().to(state.dtype), sending[eligible].sum().to(state.dtype),
        receiving[eligible].sum().to(state.dtype), state.new_tensor(proposal.numel()),
        record['balance'].abs().max(), proposal.sum(), balance64.abs().max().to(state.dtype), scale64.to(state.dtype),
        (record['flux']-capacity).clamp_min(0).max() if capacity.numel() else state.new_zeros(()),
        (record['outflow']-state).clamp_min(0).max(),
        (record['inflow']-(1-state)).clamp_min(0).max())).detach()


class EventBudgetExchange(SpatialFeatureExchange):
    """No registered adapter here: every call uses the one top-level module."""
    def forward(self, hidden, context):
        graph = context['graph']
        batch, windows, nodes, width = hidden.shape
        require(nodes == graph.nodes, 'Spatial hidden/node mismatch')
        flat = hidden.reshape(batch*windows, nodes, width)
        initial = self.encode(flat).sigmoid()
        state = initial
        src, dst = graph.edge_index
        base = context['capacity'][:, None].expand(-1, windows, -1, -1).reshape(batch*windows, graph.edges, initial.shape[-1])
        boundary = context['boundary'][:, None].expand(-1, windows, -1, -1).reshape_as(base)
        internal = (src >= 0) & (dst >= 0)
        mask = context['event_edge_support'][:, None].expand(-1, windows, -1, -1).reshape(batch*windows, graph.edges, 1)
        enabled = context['event_budget_enabled']
        if enabled:
            relation = context['event_relation'][:, None].expand(-1, windows, -1, -1).reshape(batch*windows, nodes, 3)
            embedding = context['event_embedding'][:, None].expand(-1, windows, -1).reshape(batch*windows, 16)
        in_sum, out_sum, stats = torch.zeros_like(state), torch.zeros_like(state), []

        def proposal_for(capacity):
            features = torch.cat((flat[:, src.clamp_min(0)], flat[:, dst.clamp_min(0)],
                                  state[:, src.clamp_min(0)], state[:, dst.clamp_min(0)], capacity,
                                  state[:, src.clamp_min(0)], 1-state[:, dst.clamp_min(0)]), -1)
            return torch.where(internal[None, :, None], self.proposal(features).sigmoid(), boundary)

        for step in range(self.steps):
            capacity = (signed_budget_update(base, context['event_budget_adapter'](
                flat, state, base, relation, embedding, graph), mask) if enabled else base)
            proposal = proposal_for(capacity)
            record = exchange_step(state, proposal, capacity, graph, restricted=self.restricted,
                                   eta=self.eta, validate=False)
            in_sum, out_sum = in_sum+record['inflow'], out_sum+record['outflow']
            if context.get('collect', False):
                with torch.no_grad():
                    stats.append(spatial_statistics(state, initial, proposal, capacity, record, graph, internal, self.eta))
                    if not self.training:
                        baseline = (exchange_step(state, proposal_for(base), base, graph, restricted=self.restricted,
                                                  eta=self.eta, validate=False) if enabled else record)
                        change, flux_change = (capacity-base).abs(), (record['flux']-baseline['flux']).abs()
                        zero = state.new_zeros(())
                        context['event_budget_records'].append(dict(
                            substep=step, edge_positions=base.new_tensor(base.numel()),
                            eligible_edge_positions=mask.sum().to(base.dtype)*base.shape[-1],
                            capacity_change_abs_sum=change.sum(),
                            capacity_change_abs_max=change.max() if change.numel() else zero,
                            flux_change_abs_sum=flux_change.sum(),
                            flux_change_abs_max=flux_change.max() if flux_change.numel() else zero,
                            capacity_limited_count=(proposal>capacity+1e-6).sum().to(base.dtype),
                            balance_abs_max=record['balance'].abs().max()))
            state = record['state']
        exchanged = torch.cat((state-initial, in_sum/self.steps, out_sum/self.steps), -1)
        require(bool(torch.isfinite(exchanged).all()), 'Non-finite spatial exchange')
        if stats:
            context['records'].append(torch.stack(stats))
        return self.readout(exchanged).reshape(batch, windows, nodes, width)


class EventBudgetIGSTGNN(MinimalIGSTGNN):
    def forward(self, history_data, label=None, incident_data=None, sensor_data=None,
                forecast_delta=None, *, spatial_report_enabled=True, tiid_report_enabled=True,
                report_enabled=True, semantic_enabled=True, return_spatial_details=False):
        require(all(type(value) is bool for value in (spatial_report_enabled, tiid_report_enabled,
                    report_enabled, semantic_enabled)), 'Report switches must be bool')
        require(forecast_delta is None, 'Event budget model cannot combine with an external forecast residual')
        tod, dow = cutoff_clock_from_history(history_data, num_feat=self._model_args['num_feat'],
            tpd=self._model_args['tpd'], offset_slots=getattr(self, '_minimal_clock_offset_slots', 2))
        history_data, node_u, node_d, tod_features, dow_features = self._prepare_inputs(history_data)
        raw = self.embedding(history_data)
        context = self.report_spatial_state(raw, self.T_i_D_emb[tod], self.D_i_W_emb[dow], self.internal_spatial_graph)
        context.update(event_context(self, raw, incident_data, report_enabled=report_enabled,
                                     semantic_enabled=semantic_enabled))
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
        prediction = self._decode_forecast(sum(dif)+sum(inh), None)
        return dict(prediction=prediction, spatial=context) if return_spatial_details else prediction


def from_minimal(model, mode, *, type_count=6, description_count=29):
    require(mode in MODES, 'Unknown event budget mode')
    require(all(type(value) is int and value > 0 for value in (type_count, description_count)),
            'Category counts must be positive integers')
    require(type(model) is MinimalIGSTGNN and type(model.report_spatial_state) is HistoryCapacityPrior
            and getattr(model, '_internal_spatial_mode', None) == 'state_restricted'
            and not hasattr(model, 'tiid_module') and not hasattr(model, 'event_budget_adapter')
            and all(type(layer.dif_layer.localized_st_conv.incident_exchange) is SpatialFeatureExchange
                    and layer.dif_layer.localized_st_conv.incident_exchange.restricted for layer in model.layers),
            'Expected an unadapted restricted MIN architecture')
    before = tensor_state_hash(model.state_dict())
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        result = copy.deepcopy(model)
        result.__class__ = EventBudgetIGSTGNN
        latent = result.report_spatial_state.latent
        result.event_budget_adapter = SharedEventBudgetAdapter(model._hidden_dim, latent, mode,
                                                               type_count, description_count)
        reference = next(result.parameters())
        result.event_budget_adapter.to(device=reference.device, dtype=reference.dtype).train(model.training)
        for layer in result.layers:
            layer.dif_layer.localized_st_conv.incident_exchange.__class__ = EventBudgetExchange
    parent, current = model.state_dict(), result.state_dict()
    exact = parent.keys() <= current.keys() and all(torch.equal(value, current[name]) for name, value in parent.items())
    require(exact, 'Event adapter changed retained MIN parameters or buffers')
    added = set(current)-set(parent)
    require(added and all(name.startswith('event_budget_adapter.') for name in added), 'Unexpected added state')
    require(tensor_state_hash(model.state_dict()) == before, 'Parent MIN was modified')
    count = lambda module: sum(value.numel() for value in module.parameters())
    certificate = dict(mapping_version=MAPPING_VERSION, mode=mode, retained_state_exact_verified=exact,
        parent_state_sha256=before, retained_shared_state_sha256=tensor_state_hash({name: current[name] for name in parent}),
        retained_shared_state_specification={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in parent.items()},
        parent_parameter_count=count(model), shared_parameter_count=count(model), parameter_count=count(result),
        added_parameters=count(result)-count(model), added_state_keys=sorted(added),
        all_parameters_trainable=all(value.requires_grad for value in result.parameters()),
        adapter_shared_across_layers_and_substeps=True, initial_budget_update_abs_max=0.,
        adapter_A_initial_sha256=tensor_state_hash(result.event_budget_adapter.A.state_dict()),
        type_count=type_count, description_count=description_count,
        event_inputs='published_relation_and_optional_type_description_only',
        budget_scope='internal_edge_dimensionless_four_channel_per_computational_substep',
        clock_policy='unchanged_from_parent_MIN', boundary_budget_changed=False,
        prediction_equivalence='zero_initialization_or_report_OFF_equals_same_weights_MIN',
        flux_diagnostic_scope=DIAGNOSTIC_SCOPE)
    return result, certificate
