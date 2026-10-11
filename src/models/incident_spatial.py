"""IS1.1 incident-conditioned latent spatial exchange, with no minute/vehicle units.

Reports enter exchange ONLY through capacity. Native statistical graphs remain
outside these budgets. Call context is local to a single model forward.
"""
import torch
from torch import nn

from src.models.incident_strength_gate import IncidentStrengthGate


MODES = ('native', 'strength', 'state_ordinary', 'state_restricted')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mlp(inputs, outputs, width=16):
    return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, outputs))


def safe_scale(budget, total):
    positive = total > 0
    denominator = torch.where(positive, total, torch.ones_like(total))
    return torch.where(positive, torch.minimum(torch.ones_like(total), budget / denominator), 1.)


def state_capacity(state, graph):
    node_capacity = 1e-3+(1-1e-3)*(1-state)
    src, dst = graph.edge_index
    left, right = node_capacity[:, src.clamp_min(0)], node_capacity[:, dst.clamp_min(0)]
    return torch.where((src < 0)[None, :, None], right,
                       torch.where((dst < 0)[None, :, None], left, torch.minimum(left, right)))


def exchange_step(state, proposal, capacity, graph, *, restricted, eta=.5, validate=True):
    """One simultaneous step [batch,nodes,channels]; external endpoints are -1."""
    require(0 < eta <= 1, 'Spatial eta must be in (0,1]')
    require(state.ndim == 3 and state.shape[1] == graph.nodes, 'State axes mismatch')
    require(proposal.shape == capacity.shape == (state.shape[0], graph.edges, state.shape[2]),
            'Exchange edge axes mismatch')
    require(all(v.device == state.device and v.dtype == state.dtype for v in (proposal, capacity)),
            'Exchange dtype/device mismatch')
    require(graph.edge_index.device == state.device, 'Graph device mismatch')
    require(type(restricted) is bool, 'restricted must be bool')
    if validate:
        require(bool(torch.isfinite(state).all() & torch.isfinite(proposal).all() & torch.isfinite(capacity).all()
                     & ((state >= 0) & (state <= 1)).all() & (proposal >= 0).all() & (capacity >= 0).all()),
                'Invalid latent state/proposal/capacity values')
    src, dst = graph.edge_index
    if restricted:
        bid = torch.minimum(proposal, capacity)
        sending = safe_scale(state, graph.aggregate(bid, incoming=False))
        bid = bid * torch.where((src >= 0)[None, :, None], sending[:, src.clamp_min(0)], 1.)
        receiving = safe_scale(1-state, graph.aggregate(bid, incoming=True))
        flux = bid * torch.where((dst >= 0)[None, :, None], receiving[:, dst.clamp_min(0)], 1.)
    else:
        flux = proposal
    inflow = graph.aggregate(flux, incoming=True)
    outflow = graph.aggregate(flux, incoming=False)
    raw = state + eta * (inflow-outflow)
    updated = raw if restricted else raw.clamp(0, 1)
    if restricted:
        require(bool(((updated >= -1e-5) & (updated <= 1+1e-5)).all()), 'Restricted state escaped bounds')
    internal = (src >= 0) & (dst >= 0)
    external = flux[:, src < 0].sum(1)-flux[:, dst < 0].sum(1)
    balance = (updated-state).sum(1)-eta*external
    return dict(state=updated, flux=flux, inflow=inflow, outflow=outflow,
                raw=raw, clip=(updated-raw), balance=balance,
                internal_flux=flux[:, internal], inlet_flux=flux[:, src < 0],
                exit_flux=flux[:, dst < 0])


class ReportSpatialState(nn.Module):
    def __init__(self, hidden, time_dim, latent=4):
        super().__init__()
        self.latent = latent
        summary = 3*hidden+2*time_dim
        conditioned = summary+hidden+4
        self.prior = mlp(summary, latent)
        self.proposal = mlp(conditioned, latent)
        self.adoption = mlp(conditioned, latent)
        self.inlet = nn.Linear(hidden, latent)
        self.exit = nn.Linear(hidden, latent)

    def forward(self, history, tod, dow, embedding, incident, graph, *, report_enabled=True):
        require(type(report_enabled) is bool, 'report_enabled must be bool')
        b, _, n, _ = history.shape
        require(n == graph.nodes, 'State graph/node mismatch')
        clock = torch.cat((tod, dow), -1)[:, None].expand(-1, n, -1)
        summary = torch.cat((history[:, -1], history.mean(1), history[:, -1]-history[:, 0], clock), -1)
        prior = self.prior(summary).sigmoid()
        posterior, adoption = prior, torch.zeros_like(prior)
        mask = history.new_zeros(b, n, 1)
        if report_enabled and incident is not None:
            require(embedding is not None, 'Report encoding required')
            distance = incident['distances']
            require(distance.shape == (b, n, 3), 'Report distance axes mismatch')
            age = incident['report_age_minutes'][:, None, None].expand(-1, n, 1)/5.
            evidence = torch.cat((embedding[:, None].expand(-1, n, -1), distance, age), -1)
            conditioned = torch.cat((summary, evidence), -1)
            mask = (distance.abs().sum(-1, keepdim=True) > 0).to(history.dtype)
            adoption = self.adoption(conditioned).sigmoid()*mask
            candidate = self.proposal(conditioned).sigmoid()
            posterior = prior+adoption*(candidate-prior)
        src, dst = graph.edge_index
        capacity = state_capacity(posterior, graph)
        boundary = torch.where((src < 0)[None, :, None], self.inlet(history[:, -1]).sigmoid()[:, dst.clamp_min(0)],
                               self.exit(history[:, -1]).sigmoid()[:, src.clamp_min(0)])
        boundary = torch.where(((src < 0) | (dst < 0))[None, :, None], boundary, 0.)
        return dict(graph=graph, capacity=capacity, boundary=boundary, prior=prior,
                    posterior=posterior, adoption=adoption, report_mask=mask, records=[])


STAT_NAMES = ('positions', 'clip_count', 'clip_abs_sum', 'clip_abs_max',
              'state_sum', 'state_min', 'state_max', 'displacement_abs_sum',
              'internal_flux_sum', 'inlet_flux_sum', 'exit_flux_sum',
              'capacity_limited_count', 'sending_limited_count', 'receiving_limited_count',
              'edge_positions', 'balance_abs_max', 'proposal_sum',
              'float64_balance_abs_max', 'balance_accumulation_scale_max',
              'capacity_violation_max', 'sending_violation_max', 'receiving_violation_max')


class SpatialFeatureExchange(nn.Module):
    def __init__(self, hidden, *, restricted, latent=4, steps=2, eta=.5):
        super().__init__()
        require(type(restricted) is bool and steps >= 1 and 0 < eta <= 1, 'Invalid spatial configuration')
        self.restricted, self.steps, self.eta = restricted, steps, eta
        self.encode = nn.Linear(hidden, latent)
        self.proposal = mlp(2*hidden+5*latent, latent)
        self.readout = nn.Linear(3*latent, hidden, bias=False)
        nn.init.xavier_uniform_(self.readout.weight)

    def forward(self, hidden, context):
        graph = context['graph']
        b, w, n, h = hidden.shape
        require(n == graph.nodes, 'Spatial hidden/node mismatch')
        flat = hidden.reshape(b*w, n, h)
        initial = self.encode(flat).sigmoid()
        state = initial
        src, dst = graph.edge_index
        capacity = context['capacity'][:, None].expand(-1, w, -1, -1).reshape(b*w, graph.edges, initial.shape[-1])
        boundary = context['boundary'][:, None].expand(-1, w, -1, -1).reshape_as(capacity)
        internal = (src >= 0) & (dst >= 0)
        in_sum, out_sum = torch.zeros_like(state), torch.zeros_like(state)
        stats = []
        for _ in range(self.steps):
            features = torch.cat((flat[:, src.clamp_min(0)], flat[:, dst.clamp_min(0)],
                                  state[:, src.clamp_min(0)], state[:, dst.clamp_min(0)], capacity,
                                  state[:, src.clamp_min(0)], 1-state[:, dst.clamp_min(0)]), -1)
            proposal = self.proposal(features).sigmoid()
            proposal = torch.where(internal[None, :, None], proposal, boundary)
            record = exchange_step(state, proposal, capacity, graph, restricted=self.restricted, eta=self.eta,
                                   validate=False)  # sigmoid outputs and previous bounded step satisfy input contract
            in_sum, out_sum = in_sum+record['inflow'], out_sum+record['outflow']
            if context.get('collect', False):
                with torch.no_grad():
                    eligible = graph.operator_mask[None, :, None].expand_as(state)
                    displacement = record['state']-state
                    sending = graph.aggregate(torch.minimum(proposal, capacity), incoming=False) > state+1e-6
                    capped = torch.minimum(proposal, capacity)
                    scaled = capped*torch.where((src >= 0)[None, :, None],
                        safe_scale(state, graph.aggregate(capped, incoming=False))[:, src.clamp_min(0)], 1.)
                    receiving = graph.aggregate(scaled, incoming=True) > 1-state+1e-6
                    chosen = record['state'][eligible]
                    clip = record['clip'][eligible]
                    flux64 = record['flux'].double()
                    external64 = flux64[:, src < 0].sum(1)-flux64[:, dst < 0].sum(1)
                    balance64 = (record['state'].double()-state.double()).sum(1)-self.eta*external64
                    scale64 = (record['inflow'].double().abs()+record['outflow'].double().abs()).sum(1).max()
                    stats.append(torch.stack((eligible.sum().to(state.dtype), (clip.abs()>1e-6).sum().to(state.dtype),
                        clip.abs().sum(), clip.abs().max() if clip.numel() else state.new_zeros(()),
                        chosen.sum(), chosen.min() if chosen.numel() else state.new_zeros(()),
                        chosen.max() if chosen.numel() else state.new_zeros(()), displacement[eligible].abs().sum(),
                        record['internal_flux'].sum(), record['inlet_flux'].sum(), record['exit_flux'].sum(),
                        (proposal>capacity+1e-6).sum().to(state.dtype), sending[eligible].sum().to(state.dtype),
                        receiving[eligible].sum().to(state.dtype), state.new_tensor(proposal.numel()),
                        record['balance'].abs().max(), proposal.sum(), balance64.abs().max().to(state.dtype),
                        scale64.to(state.dtype),
                        (record['flux']-capacity).clamp_min(0).max() if capacity.numel() else state.new_zeros(()),
                        (record['outflow']-state).clamp_min(0).max(),
                        (record['inflow']-(1-state)).clamp_min(0).max())).detach())
            state = record['state']
        exchanged = torch.cat((state-initial, in_sum/self.steps, out_sum/self.steps), -1)
        require(bool(torch.isfinite(exchanged).all()), 'Non-finite spatial exchange')
        if stats:
            context['records'].append(torch.stack(stats))
        return self.readout(exchanged).reshape(b, w, n, h)


def configure_spatial_mode(model, mode, graph=None):
    """Configure AFTER native initialization; no hooks, freezing or output residual."""
    require(mode in MODES, 'Unknown internal spatial mode')
    require(getattr(model, '_internal_spatial_mode', 'native') == 'native', 'Spatial mode already configured')
    require(not hasattr(model.icsf_module, 'strength_gate'), 'Spatial adapter already configured')
    if mode == 'native':
        return model
    require(model._incident_schema == 'report_location_v1' and not model.icsf_module.use_sensor_info
            and model._incident_routing == 'none' and model._time_response_mode == 'fixed',
            'IS1.1 requires original report-only fixed-TIID backbone without ACDG')
    if mode == 'strength':
        require(graph is None, 'B1 must not consume candidate road graph')
    else:
        require(graph is not None and graph.nodes == model.node_num, 'Candidate graph required')
    model._internal_spatial_mode = mode
    if mode == 'strength':
        model.icsf_module.strength_gate = IncidentStrengthGate('node', model._hidden_dim)
    else:
        model.internal_spatial_graph = graph
        model.report_spatial_state = ReportSpatialState(model._hidden_dim, model._model_args['time_emb_dim'])
        for name in ('q_proj', 'v_proj', 'icsf_fusion_mlp'):
            delattr(model.icsf_module, name)
        model.icsf_module.state_spatial_mode = True
        for layer in model.layers:
            layer.dif_layer.localized_st_conv.incident_exchange = SpatialFeatureExchange(
                model._hidden_dim, restricted=mode == 'state_restricted')
    return model.to(next(model.embedding.parameters()).device)
