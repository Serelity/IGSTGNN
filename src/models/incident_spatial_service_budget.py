"""Frozen MIN with separate exchange proposals and report service limits.

The exchange proposal always reads the baseline latent budget C0. A compact
adapter changes only the service upper bound passed to ``exchange_step``.
These are dimensionless latent quantities, not identified road capacities.
The state control retains the same report-derived eligible-edge mask but never
reads the numeric relation vectors in its learned computation.
"""
import math

import torch
from torch import nn

from src.models.incident_spatial import SpatialFeatureExchange, exchange_step, require
from src.models.incident_spatial_event_budget import (
    from_minimal, signed_budget_update, spatial_statistics,
)
from src.models.incident_spatial_frozen_budget import (
    ADAPTER_PREFIX, assert_frozen, frozen_state_hash, set_frozen_train_mode,
    trainable_named_parameters,
)
from src.models.incident_spatial_residual import tensor_state_hash


SERVICE_SCHEMA = 'frozen_min_separate_proposal_service_budget_v1'
MODES = ('location', 'state')
DIAGNOSTIC_SCOPE = 'same_current_state_same_baseline_proposal_C_vs_C0_not_full_OFF_trajectory'
PROPOSAL_BUDGET_SOURCE = 'unchanged_baseline_C0'


def adapter_parameter_count(inputs, width, latent):
    return width*(inputs+latent+1)+latent


def matched_state_width(hidden, latent, location_width=16):
    """Closest active parameter count; no zero relation or semantic columns."""
    state_inputs = 2*hidden+3*latent
    target = adapter_parameter_count(state_inputs+6, location_width, latent)
    ideal = (target-latent)/(state_inputs+latent+1)
    candidates = {max(1, math.floor(ideal)), max(1, math.ceil(ideal))}
    return min(candidates, key=lambda width: (abs(adapter_parameter_count(state_inputs, width, latent)-target), width))


class SharedServiceBudgetAdapter(nn.Module):
    def __init__(self, hidden, latent, conditioning_mode, hidden_width):
        super().__init__()
        require(conditioning_mode in MODES, 'Unknown service budget conditioning mode')
        require(type(hidden_width) is int and hidden_width > 0, 'Adapter width must be a positive integer')
        # Compatibility tag for immutable EventBudgetIGSTGNN/event_context and
        # the original frozen allowlist. Scientific mode is conditioning_mode.
        self.mode = 'location'
        self.conditioning_mode = conditioning_mode
        self.hidden, self.latent, self.hidden_width = hidden, latent, hidden_width
        self.input_dimension = 2*hidden+3*latent+(6 if conditioning_mode == 'location' else 0)
        self.A = nn.Sequential(nn.Linear(self.input_dimension, hidden_width), nn.ReLU(), nn.Linear(hidden_width, latent))
        nn.init.zeros_(self.A[-1].weight)
        nn.init.zeros_(self.A[-1].bias)

    def forward(self, hidden, state, base, relation, embedding, graph):
        src, dst = graph.edge_index
        features = [hidden[:, src.clamp_min(0)], hidden[:, dst.clamp_min(0)],
                    state[:, src.clamp_min(0)], state[:, dst.clamp_min(0)], base]
        if self.conditioning_mode == 'location':
            features.extend((relation[:, src.clamp_min(0)], relation[:, dst.clamp_min(0)]))
        # No semantic embedding, numeric D padding, or forward-local cache.
        return self.A(torch.cat(features, -1)).tanh()

    def initialize_from_legacy_location(self, legacy):
        """Retain all active old location weights; state shares common rows.

        Legacy location has sixteen identically zero semantic input columns.
        Removing those columns preserves its function, including after arbitrary
        changes of the final projection. Extra state rows retain their seeded
        constructor initialization, and their final output weights start at zero.
        """
        require(legacy.mode == 'location' and not hasattr(legacy, 'semantic_encoder'),
                'Expected the original location adapter')
        shared_rows = min(self.hidden_width, legacy.A[0].out_features)
        with torch.no_grad():
            self.A[0].weight[:shared_rows].copy_(legacy.A[0].weight[:shared_rows, :self.input_dimension])
            self.A[0].bias[:shared_rows].copy_(legacy.A[0].bias[:shared_rows])
            self.A[-1].weight[:, :shared_rows].copy_(legacy.A[-1].weight[:, :shared_rows])
            self.A[-1].bias.copy_(legacy.A[-1].bias)
        return shared_rows


class ServiceBudgetExchange(SpatialFeatureExchange):
    """One shared adapter; C_event never enters the proposal MLP directly."""
    def proposal_from_base(self, hidden, state, base, boundary, graph):
        src, dst = graph.edge_index
        internal = (src >= 0) & (dst >= 0)
        features = torch.cat((hidden[:, src.clamp_min(0)], hidden[:, dst.clamp_min(0)],
                              state[:, src.clamp_min(0)], state[:, dst.clamp_min(0)], base,
                              state[:, src.clamp_min(0)], 1-state[:, dst.clamp_min(0)]), -1)
        return torch.where(internal[None, :, None], self.proposal(features).sigmoid(), boundary)

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
        adapter = context['event_budget_adapter']
        context['event_budget_diagnostic_scope'] = DIAGNOSTIC_SCOPE
        context['proposal_budget_source'] = PROPOSAL_BUDGET_SOURCE
        context['service_conditioning_mode'] = adapter.conditioning_mode
        context['service_budget_schema'] = SERVICE_SCHEMA
        if enabled:
            relation = context['event_relation'][:, None].expand(-1, windows, -1, -1).reshape(batch*windows, nodes, 3)
            embedding = context['event_embedding'][:, None].expand(-1, windows, -1).reshape(batch*windows, 16)
        in_sum, out_sum, stats = torch.zeros_like(state), torch.zeros_like(state), []
        for step in range(self.steps):
            # The proposal is independent of the adapted service limit at this
            # same state. Later states can change through previous exchange.
            proposal = self.proposal_from_base(flat, state, base, boundary, graph)
            capacity = (signed_budget_update(base, adapter(flat, state, base, relation, embedding, graph), mask)
                        if enabled else base)
            record = exchange_step(state, proposal, capacity, graph, restricted=self.restricted,
                                   eta=self.eta, validate=False)
            in_sum, out_sum = in_sum+record['inflow'], out_sum+record['outflow']
            if context.get('collect', False):
                with torch.no_grad():
                    stats.append(spatial_statistics(state, initial, proposal, capacity, record, graph, internal, self.eta))
                    if not self.training:
                        baseline = (exchange_step(state, proposal, base, graph, restricted=self.restricted,
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
                            base_capacity_limited_count=(proposal>base+1e-6).sum().to(base.dtype),
                            proposal_change_abs_max=zero,
                            balance_abs_max=record['balance'].abs().max()))
            state = record['state']
        exchanged = torch.cat((state-initial, in_sum/self.steps, out_sum/self.steps), -1)
        require(bool(torch.isfinite(exchanged).all()), 'Non-finite spatial exchange')
        if stats:
            context['records'].append(torch.stack(stats))
        return self.readout(exchanged).reshape(batch, windows, nodes, width)


def build_frozen_service(base, mode='location', hidden_width=None):
    """Copy source MIN, preserve its RNG/state, and freeze all but the adapter.

    Default location width is 16; default state width minimizes the difference
    in active trainable parameters. Both candidates retain exactly the original
    report-supported internal-edge mask. State is therefore a support-matched
    state-conditioning control, not a no-incident model.
    """
    require(mode in MODES, 'Unknown service budget mode')
    before = tensor_state_hash(base.state_dict())
    model, legacy_conversion = from_minimal(base, 'location')
    hidden, latent = model._hidden_dim, model.report_spatial_state.latent
    if hidden_width is None:
        hidden_width = 16 if mode == 'location' else matched_state_width(hidden, latent)
    legacy = model.event_budget_adapter
    legacy_adapter_hash = tensor_state_hash(legacy.state_dict())
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        adapter = SharedServiceBudgetAdapter(hidden, latent, mode, hidden_width)
        reference = next(model.parameters())
        adapter.to(device=reference.device, dtype=reference.dtype)
        shared_rows = adapter.initialize_from_legacy_location(legacy)
        model.event_budget_adapter = adapter
        for layer in model.layers:
            layer.dif_layer.localized_st_conv.incident_exchange.__class__ = ServiceBudgetExchange
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(ADAPTER_PREFIX))
        parameter.grad = None
    set_frozen_train_mode(model)
    digest = frozen_state_hash(model)
    require(digest == before == tensor_state_hash(base.state_dict()), 'Frozen copy differs from source MIN')
    assert_frozen(model, digest)
    trainable = trainable_named_parameters(model)
    count = lambda module: sum(value.numel() for value in module.parameters())
    conversion = dict(legacy_conversion)
    conversion.update(mapping_version=SERVICE_SCHEMA, mode=mode,
        parameter_count=count(model), added_parameters=count(adapter),
        all_parameters_trainable=False, adapter_A_initial_sha256=tensor_state_hash(adapter.A.state_dict()),
        event_inputs=('published_relation_values_and_support' if mode == 'location' else 'published_relation_support_only'),
        flux_diagnostic_scope=DIAGNOSTIC_SCOPE, proposal_budget_source=PROPOSAL_BUDGET_SOURCE,
        original_location_adapter_initial_sha256=legacy_adapter_hash)
    certificate = dict(
        freeze_version=SERVICE_SCHEMA, service_schema=SERVICE_SCHEMA, service_mode=mode,
        conditioning_mode=mode, adapter_compatibility_mode=adapter.mode,
        conversion=conversion, frozen_state_sha256=digest,
        adapter_initial_sha256=tensor_state_hash(adapter.state_dict()),
        frozen_parameters=count(base), trainable_parameters=count(adapter),
        trainable_names=[name for name, _ in trainable],
        input_dimension=adapter.input_dimension, hidden_width=hidden_width,
        location_values_used=mode == 'location', semantic_values_used=False,
        support_mask_source='unchanged_original_report_relation_nonzero_internal_endpoint_support',
        support_mask_equal_across_modes=True, proposal_budget_source=PROPOSAL_BUDGET_SOURCE,
        proposal_direct_service_budget_input=False,
        baseline_and_adapted_flux_share_current_state_and_proposal=True,
        shared_initial_active_rows=shared_rows,
        additional_seeded_hidden_rows=max(0, hidden_width-shared_rows),
        location_initialization='legacy_location_active_columns_exact_copy',
        location_legacy_function_equivalent=(mode == 'location' and hidden_width == legacy.A[0].out_features),
        learned_input_fields=(['source_hidden', 'destination_hidden', 'source_state', 'destination_state', 'baseline_budget']
                              +(['source_relation', 'destination_relation'] if mode == 'location' else [])),
        parameter_matching='closest_active_parameter_count_not_exact_architecture_or_hash',
        default_location_parameters=adapter_parameter_count(2*hidden+3*latent+6, 16, latent),
        state_parameters_at_selected_width=adapter_parameter_count(2*hidden+3*latent, hidden_width, latent),
        backbone_evaluation_mode=True, adapter_only_training_mode=True,
        backbone_gradients_disabled=True, downstream_autograd_required=True,
        semantic_encoder_present=False, optimizer_parameter_scope='event_budget_adapter_only',
        identity_reference='same_state_MIN_with_frozen_parameters_and_evaluation_mode',
        original_trainable_parameter_replay='fixed_tolerance_required_before_freezing',
        budget_scope='dimensionless_latent_exchange_service_upper_bound',
        flux_diagnostic_scope=DIAGNOSTIC_SCOPE, boundary_budget_changed=False,
    )
    return model, certificate
