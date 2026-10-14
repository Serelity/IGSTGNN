"""A shared, persistent incident response beside a MIN forecast.

MIN is evaluated once. Its activated, repeated forecast features drive two
small trajectories with identical initial state and transition parameters.
Only one trajectory receives the report-conditioned budget update. Their
state/flux difference is read out in normalized flow units. Latent states and
budgets have no identified density, queue, capacity or physical-time units.
The default phase freezes MIN; an explicit joint phase also trains its
backbone and changes OFF's reference to the current joint checkpoint.
"""
import copy

import torch
from torch import nn

from src.models.incident_spatial import exchange_step, require
from src.models.incident_spatial_event_budget import signed_budget_update
from src.models.incident_spatial_minimal import MinimalIGSTGNN
from src.models.incident_spatial_residual import EPSILON, node_capacity_to_edges, tensor_state_hash


RESPONSE_SCHEMA = 'frozen_min_shared_persistent_response_v1'
MODES = ('location', 'state')
TIME_SCOPE = 'ordered_published_target_slots_not_certified_cutoff_relative_minutes'
UNIT_SCOPE = 'dimensionless_latent_state_and_exchange_budget_normalized_flow_readout'


def _mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, outputs))


def response_common_hash(response):
    return tensor_state_hash({name: value for name, value in response.state_dict().items()
                              if not name.startswith('relation_projection.')})


class PersistentResponse(nn.Module):
    """One state coordinate system and transition shared across target slots."""
    def __init__(self, forecast_width, mode, *, latent=4, feature_width=16,
                 interaction_rank=8, steps=2, eta=.5):
        super().__init__()
        require(mode in MODES, 'Unknown response mode')
        for value in (forecast_width, latent, feature_width, interaction_rank, steps):
            require(type(value) is int and value > 0, 'Response dimensions and steps must be positive integers')
        require(isinstance(eta, (int, float)) and not isinstance(eta, bool) and 0 < eta <= 1,
                'Response eta must be in (0,1]')
        self.mode, self.latent = mode, latent
        self.feature_width, self.interaction_rank = feature_width, interaction_rank
        self.steps, self.eta = steps, float(eta)
        # last/mean/change of standardized historical flow, last nominal tod/dow.
        self.origin_encoder = _mlp(5, latent, feature_width)
        self.feature_projection = nn.Linear(forecast_width, feature_width)
        self.capacity_prior = _mlp(latent, latent, feature_width)
        inputs = 2*feature_width+3*latent+1
        self.proposal = _mlp(inputs, latent, feature_width)
        self.condition_state = _mlp(inputs, interaction_rank, feature_width)
        self.condition_output = nn.Linear(interaction_rank, latent)
        nn.init.zeros_(self.condition_output.weight)
        nn.init.zeros_(self.condition_output.bias)
        self.readout = nn.Linear(3*latent, 1, bias=False)
        nn.init.normal_(self.readout.weight, std=.02)
        # Construct this last: location/state have exactly the same common
        # seeded parameters, while the extra 6*rank parameters remain active.
        if mode == 'location':
            self.relation_projection = nn.Linear(6, interaction_rank, bias=False)

    def origin(self, history):
        flow = history[..., :1]
        features = torch.cat((flow[:, -1], flow.mean(1), flow[:, -1]-flow[:, 0],
                              history[:, -1, :, 1:3]), -1)
        return self.origin_encoder(features).sigmoid()

    def edge_features(self, hidden, state, baseline_budget, graph, slot):
        src, dst = graph.edge_index.clamp_min(0)
        position = baseline_budget.new_full(baseline_budget.shape[:-1]+(1,), slot)
        return torch.cat((hidden[:, src], hidden[:, dst], state[:, src], state[:, dst],
                          baseline_budget, position), -1)

    def condition(self, features, relation, graph):
        interaction = self.condition_state(features)
        if self.mode == 'location':
            src, dst = graph.edge_index.clamp_min(0)
            detail = torch.cat((relation[:, src], relation[:, dst]), -1)
            interaction = interaction*(1+self.relation_projection(detail).tanh())
        return self.condition_output(interaction).tanh()

    def forward(self, history, forecast_features, relation, edge_support, graph, *, collect=False):
        batch, horizon, nodes, _ = forecast_features.shape
        initial = self.origin(history)
        reference_state, event_state = initial, initial
        src, dst = graph.edge_index
        internal = (src >= 0) & (dst >= 0)
        c0 = node_capacity_to_edges(EPSILON+(1-EPSILON)*self.capacity_prior(initial).sigmoid(), graph)
        # Boundary conditions are deliberately not learned in this candidate.
        c0 = torch.where(internal[None, :, None], c0, 0.)
        outputs, diagnostic_rows = [], []
        max_balance = initial.new_zeros(())
        max_capacity = initial.new_zeros(())
        max_sending = initial.new_zeros(())
        max_receiving = initial.new_zeros(())
        for slot in range(horizon):
            hidden = self.feature_projection(forecast_features[:, slot])
            reference_in, reference_out = torch.zeros_like(initial), torch.zeros_like(initial)
            event_in, event_out = torch.zeros_like(initial), torch.zeros_like(initial)
            capacity_changes, flux_changes = [], []
            proposal_over_all, proposal_over_eligible, final_at_budget_eligible = [], [], []
            for _ in range(self.steps):
                fraction = (slot+1)/horizon
                ref_features = self.edge_features(hidden, reference_state, c0, graph, fraction)
                event_features = self.edge_features(hidden, event_state, c0, graph, fraction)
                reference_proposal = torch.where(internal[None, :, None], self.proposal(ref_features).sigmoid(), 0.)
                event_proposal = torch.where(internal[None, :, None], self.proposal(event_features).sigmoid(), 0.)
                adjusted = signed_budget_update(c0, self.condition(event_features, relation, graph), edge_support)
                capacity = torch.where(internal[None, :, None], adjusted, 0.)
                ref = exchange_step(reference_state, reference_proposal, c0, graph,
                                    restricted=True, eta=self.eta, validate=False)
                event = exchange_step(event_state, event_proposal, capacity, graph,
                                      restricted=True, eta=self.eta, validate=False)
                reference_in, reference_out = reference_in+ref['inflow'], reference_out+ref['outflow']
                event_in, event_out = event_in+event['inflow'], event_out+event['outflow']
                if collect:
                    with torch.no_grad():
                        capacity_changes.append((capacity-c0).abs().mean())
                        flux_changes.append((event['flux']-ref['flux']).abs().mean())
                        exceeds = event_proposal > capacity+1e-6
                        eligible = edge_support.expand_as(capacity)
                        eligible_count = eligible.sum()
                        denominator = eligible_count.clamp_min(1).to(capacity.dtype)
                        # This first statistic is a proposal threshold test,
                        # with all B*E*latent entries as its denominator. It is
                        # not evidence that the final flow is budget-bound.
                        proposal_over_all.append((exceeds & internal[None, :, None]).float().mean())
                        proposal_over_eligible.append((exceeds & eligible).sum().to(capacity.dtype)/denominator)
                        at_budget = torch.isclose(event['flux'], capacity, rtol=1e-5, atol=1e-6)
                        final_at_budget_eligible.append((exceeds & at_budget & eligible).sum().to(capacity.dtype)/denominator)
                        for old, record, bound in ((reference_state, ref, c0), (event_state, event, capacity)):
                            max_balance = torch.maximum(max_balance, record['balance'].abs().max())
                            max_capacity = torch.maximum(max_capacity, (record['flux']-bound).clamp_min(0).max())
                            max_sending = torch.maximum(max_sending, (record['outflow']-old).clamp_min(0).max())
                            max_receiving = torch.maximum(max_receiving, (record['inflow']-(1-old)).clamp_min(0).max())
                reference_state, event_state = ref['state'], event['state']
            response = torch.cat((event_state-reference_state, (event_in-reference_in)/self.steps,
                                   (event_out-reference_out)/self.steps), -1)
            output = self.readout(response)
            output = torch.where(graph.operator_mask[None, :, None], output, 0.)
            outputs.append(output)
            if collect:
                diagnostic_rows.append(torch.stack(((event_state-reference_state).abs().mean().detach(),
                    torch.stack(flux_changes).mean(), torch.stack(capacity_changes).mean(),
                    torch.stack(proposal_over_all).mean(), torch.stack(proposal_over_eligible).mean(),
                    torch.stack(final_at_budget_eligible).mean(), output.mean().detach(), output.abs().mean().detach())))
        delta = torch.stack(outputs, 1)
        require(bool(torch.isfinite(delta).all()), 'Non-finite response prediction')
        diagnostics = {}
        if collect:
            rows = torch.stack(diagnostic_rows).detach().cpu().tolist()
            names = ('state_delta_abs_mean', 'flux_change_abs_mean', 'capacity_change_abs_mean',
                     'internal_proposal_exceeds_budget_fraction_all_edges',
                     'eligible_proposal_exceeds_budget_fraction', 'eligible_final_flux_at_budget_fraction',
                     'delta_prediction_signed_mean', 'delta_prediction_abs_mean')
            diagnostics = {name: sum(row[index] for row in rows)/horizon for index, name in enumerate(names)}
            diagnostics['by_slot'] = [dict(slot=index+1, **dict(zip(names, row))) for index, row in enumerate(rows)]
            violations = torch.stack((max_balance, max_capacity, max_sending, max_receiving)).detach().cpu().tolist()
            diagnostics.update(zip(('balance_abs_max', 'capacity_violation_max', 'sending_violation_max',
                                    'receiving_violation_max'), violations))
        return delta, diagnostics


class FrozenMINResponse(nn.Module):
    def __init__(self, backbone, response):
        super().__init__()
        self.backbone, self.response = backbone, response
        self._frozen_digest = tensor_state_hash(backbone.state_dict())
        self.joint_training = False
        self.train(True)

    def train(self, mode=True):
        require(type(mode) is bool, 'Training mode must be bool')
        super().train(mode)
        if not self.joint_training:
            self.backbone.eval()
        return self

    def set_joint_training(self, enabled=True):
        """Explicit phase change; joint OFF refers to this updated backbone.

        Joint optimization updates this independent candidate backbone copy.
        Callers must create a separately identified run. The original MIN is
        reused read-only; matched state/location candidates receive the same
        new joint-training budget. Evaluation disables dropout normally.
        """
        require(type(enabled) is bool, 'Joint training switch must be bool')
        self.joint_training = enabled
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(enabled)
            parameter.grad = None
        self.train(self.training)
        return self

    def assert_backbone_frozen(self, expected_digest=None):
        require(not self.joint_training, 'Frozen-source assertion is unavailable in joint training mode')
        for name, parameter in self.backbone.named_parameters():
            require(not parameter.requires_grad and parameter.grad is None, 'Frozen MIN gradient or flag changed: '+name)
        require(all(not module.training for module in self.backbone.modules()), 'Frozen MIN must remain in evaluation mode')
        digest = tensor_state_hash(self.backbone.state_dict())
        require(digest == (expected_digest or self._frozen_digest), 'Frozen MIN parameters or buffers changed')
        return True

    def _context(self, history, incident, enabled, override):
        batch, _, nodes, _ = history.shape
        relation = history.new_zeros(batch, nodes, 3)
        support = torch.zeros(batch, nodes, 1, device=history.device, dtype=torch.bool)
        if override is not None:
            require(isinstance(override, torch.Tensor) and override.shape == (batch, nodes, 1)
                    and override.dtype == torch.bool and override.device == history.device,
                    'Support override must be Bool[B,N,1] on the model device')
        if enabled and incident is not None:
            require(isinstance(incident, dict), 'Event input must be a dictionary')
            raw = incident.get('relation')
            require(isinstance(raw, torch.Tensor) and raw.shape == (batch, nodes, 3)
                    and raw.dtype == torch.float32 and raw.device == history.device
                    and bool(torch.isfinite(raw).all()),
                    'Event relation must be finite Float32[B,N,3] on the model device')
            flags = []
            for field in ('relation_available', 'report_present'):
                flag = incident.get(field)
                require(isinstance(flag, torch.Tensor) and flag.shape == (batch,)
                        and flag.dtype == torch.bool and flag.device == history.device,
                        'Event '+field+' must be Bool[B] on the model device')
                flags.append(flag)
            available = flags[0] & flags[1]
            support = ((raw.abs().sum(-1, keepdim=True)>0) if override is None else override)
            support = support & available[:, None, None]
            relation = raw.to(history.dtype)
        graph = self.backbone.internal_spatial_graph
        src, dst = graph.edge_index
        internal = (src >= 0) & (dst >= 0)
        edge_support = (support[:, src.clamp_min(0)] | support[:, dst.clamp_min(0)]) & internal[None, :, None]
        return relation, support, edge_support

    def forward(self, history_data, incident_data=None, *, report_enabled=True,
                return_details=False, support_override=None):
        require(type(report_enabled) is bool and type(return_details) is bool, 'Response switches must be bool')
        require(isinstance(history_data, torch.Tensor) and history_data.ndim == 4
                and history_data.shape[-1] == 3 and bool(torch.isfinite(history_data).all()),
                'Response history must be finite [B,T,N,3]')
        relation, support, edge_support = self._context(history_data, incident_data, report_enabled, support_override)
        captured = []
        def capture(module, args):
            captured.append(args[0] if self.joint_training else args[0].detach())
        handle = self.backbone.out_fc_1.register_forward_pre_hook(capture)
        try:
            if not self.joint_training:
                self.backbone.eval()
            with torch.set_grad_enabled(self.joint_training and torch.is_grad_enabled()):
                baseline = self.backbone(history_data)
        finally:
            handle.remove()
        require(len(captured) == 1 and captured[0].shape[:3] == baseline.shape[:3],
                'Expected one repeated forecast feature capture with target-slot axes')
        graph = self.backbone.internal_spatial_graph
        enabled = bool(edge_support.any())
        if enabled:
            delta, diagnostics = self.response(history_data, captured[0], relation, edge_support, graph,
                                               collect=return_details)
            prediction = baseline+delta
        else:
            prediction = baseline
            diagnostics = dict(by_slot=[], state_delta_abs_mean=0., flux_change_abs_mean=0.,
                capacity_change_abs_mean=0., internal_proposal_exceeds_budget_fraction_all_edges=0.,
                eligible_proposal_exceeds_budget_fraction=0., eligible_final_flux_at_budget_fraction=0.,
                delta_prediction_signed_mean=0.,
                delta_prediction_abs_mean=0., balance_abs_max=0., capacity_violation_max=0.,
                sending_violation_max=0., receiving_violation_max=0.)
        if return_details:
            diagnostics.update(schema=RESPONSE_SCHEMA, mode=self.response.mode, path_executed=enabled,
                target_slots=int(baseline.shape[1]), spatial_substeps=self.response.steps,
                edge_positions=int(edge_support.shape[0]*graph.edges*self.response.latent),
                eligible_edge_positions=int(edge_support.sum().detach().cpu())*self.response.latent,
                supported_node_count=int(support.sum().detach().cpu()), time_scope=TIME_SCOPE, unit_scope=UNIT_SCOPE)
            diagnostics['optimization_phase'] = 'joint_backbone_and_response' if self.joint_training else 'frozen_MIN_complete_response'
            diagnostics['off_reference'] = 'same_joint_checkpoint_MIN' if self.joint_training else 'immutable_source_MIN'
            diagnostics['proposal_fraction_scope'] = 'proposal_threshold_not_final_binding_or_causal_sensitivity'
            diagnostics['eligible_final_flux_scope'] = 'proposal_exceeds_budget_and_final_flux_close_to_budget_not_unique_binding_cause'
            diagnostics['eligible_final_flux_tolerance'] = dict(rtol=1e-5, atol=1e-6)
            return dict(prediction=prediction, baseline_prediction=baseline.detach(), response=diagnostics)
        return prediction


def build_frozen_response(base, mode='location', latent=4, feature_width=16,
                          interaction_rank=8, steps=2, eta=.5):
    """Independent source copy; preserve caller RNG and exact common initialization."""
    require(type(base) is MinimalIGSTGNN, 'Expected a pruned MinimalIGSTGNN source')
    before = tensor_state_hash(base.state_dict())
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        backbone = copy.deepcopy(base).requires_grad_(False).eval()
        for parameter in backbone.parameters():
            parameter.grad = None
        response = PersistentResponse(base.out_fc_1.in_features, mode, latent=latent,
            feature_width=feature_width, interaction_rank=interaction_rank, steps=steps, eta=eta)
        reference = next(backbone.parameters())
        response.to(device=reference.device, dtype=reference.dtype)
        model = FrozenMINResponse(backbone, response)
    require(before == tensor_state_hash(base.state_dict()), 'Source MIN was modified')
    model.assert_backbone_frozen(before)
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    certificate = dict(schema=RESPONSE_SCHEMA, mode=mode, frozen_state_sha256=before,
        response_initial_sha256=tensor_state_hash(response.state_dict()), common_initial_sha256=response_common_hash(response),
        frozen_parameters=sum(value.numel() for value in backbone.parameters()),
        trainable_parameters=sum(value.numel() for _, value in trainable), trainable_names=[name for name, _ in trainable],
        latent=latent, feature_width=feature_width, interaction_rank=interaction_rank, steps=steps, eta=float(eta),
        location_values_used=mode == 'location', semantic_values_used=False, support_mask_equal_across_modes=True,
        support_scope='same_original_nonzero_relation_internal_endpoint_mask',
        state_control_scope='report_support_matched_state_control_not_no_incident',
        forecast_feature_scope='one_MIN_out_fc_1_pre_hook_activated_repeated_target_slot_features',
        feature_capture_width=base.out_fc_1.in_features, backbone_forward_calls=1,
        optimizer_parameter_scope='complete_response_branch_only', boundary_proposal_and_budget_zero=True,
        zero_condition_exact_MIN=True, time_scope=TIME_SCOPE, unit_scope=UNIT_SCOPE,
        reference_scope='shared_latent_trajectory_without_added_condition_not_real_no_incident_counterfactual')
    return model, certificate
