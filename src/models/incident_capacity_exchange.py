"""M3.0 bounded latent exchange, not a vehicle-unit traffic solver.

The caller supplies qualified edges, a cutoff-state estimate and future boundary
predictions. No traffic targets or reports enter this operator. Incident inputs
reach either recurrence ONLY through the same edge-capacity tensor.
"""
import math

import torch
from torch import nn

from src.models.incident_relative_capacity import finite_float, require


def compatible(value, reference, name):
    finite_float(value, name)
    require(value.device == reference.device and value.dtype == reference.dtype,
            name + ' device/dtype mismatch')


def bernstein_loss(coefficients, elapsed_minutes, end_minutes):
    """[B,E,4] coefficients -> [B,K,E] loss at explicit cutoff-relative times."""
    finite_float(coefficients, 'Coefficients')
    require(coefficients.ndim == 3 and coefficients.shape[-1] == 4,
            'Expected coefficients [B,E,4]')
    compatible(elapsed_minutes, coefficients, 'Elapsed minutes')
    require(isinstance(end_minutes, (int, float)) and not isinstance(end_minutes, bool)
            and math.isfinite(end_minutes) and end_minutes > 0, 'Invalid trajectory end')
    require(elapsed_minutes.ndim == 1 and elapsed_minutes.numel() > 0
            and (elapsed_minutes >= 0).all() and (elapsed_minutes <= end_minutes).all()
            and (torch.diff(elapsed_minutes) > 0).all(), 'Invalid trajectory time axis')
    require(((coefficients >= 0) & (coefficients <= .95)).all(), 'Coefficients outside [0,.95]')
    u = elapsed_minutes / end_minutes
    basis = torch.stack(((1-u)**3, 3*u*(1-u)**2, 3*u*u*(1-u), u**3), -1)
    return torch.einsum('ber,kr->bke', coefficients, basis)


def condition_coefficients(history_coefficients, report_coefficients, association, *, incident_enabled=True):
    """No association gives EXACT history coefficients, including on/off replay."""
    require(isinstance(incident_enabled, bool), 'incident_enabled must be bool')
    finite_float(history_coefficients, 'History coefficients')
    require(history_coefficients.ndim == 3 and history_coefficients.shape[-1] == 4,
            'Expected coefficients [B,E,4]')
    compatible(report_coefficients, history_coefficients, 'Report coefficients')
    compatible(association, history_coefficients, 'Association')
    require(report_coefficients.shape == history_coefficients.shape
            and association.shape == history_coefficients.shape[:2], 'Condition axes mismatch')
    for value in (history_coefficients, report_coefficients):
        require(((value >= 0) & (value <= .95)).all(), 'Coefficients outside [0,.95]')
    require(((association >= 0) & (association <= 1)).all(), 'Association outside [0,1]')
    if not incident_enabled:
        return history_coefficients
    return history_coefficients + association[..., None] * (report_coefficients-history_coefficients)


class ExchangeGraph(nn.Module):
    """Static directed graph; -1 denotes an explicitly marked external endpoint.

    operator_mask is supplied eligibility, NOT an inferred qualification result.
    No graph is constructed from correlation, station order or report support.
    """
    def __init__(self, nodes, edge_index, operator_mask, boundary_nodes, *, evidence_scope):
        super().__init__()
        require(isinstance(nodes, int) and not isinstance(nodes, bool) and nodes > 0, 'Invalid node count')
        require(evidence_scope in ('synthetic_only', 'candidate_unverified', 'verified_latent_operator'),
                'Explicit graph evidence scope required')
        require(edge_index.dtype == torch.long and edge_index.ndim == 2 and edge_index.shape[0] == 2,
                'Expected int64 edge_index [2,E]')
        for value in (operator_mask, boundary_nodes):
            require(value.shape == (nodes,) and value.dtype == torch.bool
                    and value.device == edge_index.device, 'Graph mask mismatch')
        src, dst = edge_index
        require(((edge_index >= -1) & (edge_index < nodes)).all(), 'Unknown graph endpoint')
        require(~((src == -1) & (dst == -1)).any(), 'External-to-external edge forbidden')
        require(~((src == dst) & (src >= 0)).any(), 'Self exchange forbidden')
        internal = (src >= 0) & (dst >= 0)
        pairs = edge_index[:, internal].T
        require(torch.unique(pairs, dim=0).shape[0] == pairs.shape[0], 'Duplicate internal edge')
        require(operator_mask[src[src >= 0]].all() and operator_mask[dst[dst >= 0]].all(),
                'Edge touches an ineligible node')
        require(boundary_nodes[dst[src == -1]].all() and boundary_nodes[src[dst == -1]].all(),
                'External exchange requires marked boundary nodes')
        self.nodes, self.evidence_scope = nodes, evidence_scope
        self.register_buffer('edge_index', edge_index.clone())
        self.register_buffer('operator_mask', operator_mask.clone())
        self.register_buffer('boundary_nodes', boundary_nodes.clone())

    @property
    def edges(self):
        return self.edge_index.shape[1]

    def aggregate(self, values, *, incoming):
        indices = self.edge_index[1 if incoming else 0]
        selected = indices >= 0
        return values.new_zeros(values.shape[0], self.nodes, values.shape[-1]).index_add(
            1, indices[selected], values[:, selected])


def edge_capacities(coefficients, elapsed_minutes, end_minutes, graph, alpha, outgoing_weights):
    """Fixed kappa=1 reference. Per-station traffic normalizers never enter here."""
    loss = bernstein_loss(coefficients, elapsed_minutes, end_minutes)
    require(coefficients.shape[1] == graph.edges and alpha.ndim == 2
            and alpha.shape[0] == graph.nodes and outgoing_weights.shape == (graph.edges,),
            'Capacity graph axes mismatch')
    compatible(alpha, coefficients, 'Alpha')
    compatible(outgoing_weights, coefficients, 'Outgoing weights')
    require(graph.edge_index.device == coefficients.device, 'Graph device mismatch')
    src = graph.edge_index[0]
    base = outgoing_weights[:, None] * alpha[src.clamp_min(0)]
    base = torch.where((src >= 0)[:, None], base, 0)
    return (1-loss[..., None]) * base[None, None]


def validate_inputs(initial, capacity, boundary, times, weights, graph, alpha, beta, interval_minutes):
    finite_float(initial, 'Initial state')
    require(initial.ndim == 3 and initial.shape[0] > 0 and initial.shape[1] == graph.nodes
            and initial.shape[2] > 0, 'Expected state [B,N,d]')
    b, _, d = initial.shape
    for name, value in (('Capacity', capacity), ('Boundary', boundary), ('Times', times),
                        ('Weights', weights), ('Alpha', alpha), ('Beta', beta)):
        compatible(value, initial, name)
    require(graph.edge_index.device == initial.device, 'Graph device mismatch')
    require(times.ndim == 1 and times.numel() >= 2 and times[0] == 0
            and (torch.diff(times) > 0).all(), 'Times must start at cutoff zero and increase')
    require(isinstance(interval_minutes, (float, int)) and not isinstance(interval_minutes, bool)
            and math.isfinite(interval_minutes) and interval_minutes > 0, 'Invalid observation interval')
    require(capacity.shape == boundary.shape == (b, times.numel()-1, graph.edges, d),
            'Expected capacity/boundary [B,K,E,d]')
    require(alpha.shape == beta.shape == (graph.nodes, d) and weights.shape == (graph.edges,),
            'Rate/weight axes mismatch')
    require(((initial >= 0) & (initial <= 1)).all(), 'Initial state outside [0,1]')
    for value in (alpha, beta):
        require(((value >= .05) & (value <= 2)).all(), 'Rates outside [.05,2]')
    require((capacity >= 0).all() and ((boundary >= 0) & (boundary <= 2)).all()
            and (weights >= 0).all(), 'Negative capacity/weight or invalid boundary demand')
    h = torch.diff(times) / interval_minutes
    require((h[:, None, None] * alpha <= 1).all() and (h[:, None, None] * beta <= 1).all(),
            'State-invariance step condition violated')
    src, dst = graph.edge_index
    inlet = src < 0
    require((weights[inlet] == 0).all() and (capacity[:, :, inlet] == 0).all()
            and (boundary[:, :, ~inlet] == 0).all(), 'Boundary channels used at wrong edges')
    sums = weights.new_zeros(graph.nodes).index_add(0, src[~inlet], weights[~inlet])
    has_out = torch.bincount(src[~inlet], minlength=graph.nodes) > 0
    tol = 32 * torch.finfo(initial.dtype).eps
    require(torch.allclose(sums[has_out], torch.ones_like(sums[has_out]), atol=tol, rtol=0),
            'Internal and exit edges must share one outgoing allocation budget')
    base = weights[:, None] * alpha[src.clamp_min(0)]
    require((capacity[:, :, ~inlet] <= base[None, None, ~inlet] + tol).all(),
            'Capacity exceeds fixed-kappa sending-rate reference')
    return h


def exchange_step(state, capacity, boundary, weights, graph, alpha, beta, h):
    """Private checked-rollout kernel. All bids use the SAME old state."""
    src, dst = graph.edge_index
    sending, receiving = alpha * state, beta * (1-state)
    sending_bid = weights[None, :, None] * sending[:, src.clamp_min(0)]
    bid = torch.where((src < 0)[None, :, None], boundary, torch.minimum(sending_bid, capacity))
    incoming_bid = graph.aggregate(bid, incoming=True)
    positive = incoming_bid > 0
    denominator = torch.where(positive, incoming_bid, torch.ones_like(incoming_bid))
    scale = torch.where(positive, torch.minimum(torch.ones_like(receiving), receiving/denominator), 1)
    edge_scale = torch.where((dst < 0)[None, :, None], 1, scale[:, dst.clamp_min(0)])
    flux = bid * edge_scale
    inflow, outflow = graph.aggregate(flux, incoming=True), graph.aggregate(flux, incoming=False)
    updated = state + h * (inflow-outflow)
    tol = 32 * torch.finfo(state.dtype).eps
    flags = dict(capacity_limited=(src >= 0)[None, :, None] & (capacity < sending_bid-tol),
                 receiving_limited=(dst >= 0)[None, :, None] & (bid > tol) & (edge_scale < 1-tol),
                 sending_bid_limited=(src >= 0)[None, :, None] & (sending_bid < capacity-tol))
    external = flux[:, src < 0].sum(1)-flux[:, dst < 0].sum(1)
    residual = (updated-state).sum(1)-h*external
    return dict(state=updated, edge_flux=flux, inflow=inflow, outflow=outflow,
                receiving_scale=scale, balance_residual=residual, **flags)


class CapacityLimitedExchange(nn.Module):
    def __init__(self, graph, alpha, beta, interval_minutes=5.):
        super().__init__()
        self.graph, self.interval_minutes = graph, interval_minutes
        self.register_buffer('alpha', alpha.clone())
        self.register_buffer('beta', beta.clone())

    def forward(self, initial, capacity, boundary, times, outgoing_weights):
        h = validate_inputs(initial, capacity, boundary, times, outgoing_weights,
                            self.graph, self.alpha, self.beta, self.interval_minutes)
        state, states, records = initial, [initial], []
        for k in range(h.numel()):
            record = exchange_step(state, capacity[:, k], boundary[:, k], outgoing_weights,
                                   self.graph, self.alpha, self.beta, h[k])
            state = record['state']
            states.append(state)
            records.append(record)
        result = {key: torch.stack([r[key] for r in records], 1) for key in records[0] if key != 'state'}
        return dict(states=torch.stack(states, 1), times=times, interval_minutes=self.interval_minutes,
                    operator_mask=self.graph.operator_mask, kind='capacity_limited', **result)


class OrdinaryDirectedRecurrence(CapacityLimitedExchange):
    """Same capacity-only report entrance; no hard flow or receive budget.

    This is an operator reference, NOT yet a parameter-matched full IGSTGNN arm.
    Its additional learned parameters are explicitly counted by the checker.
    """
    def __init__(self, graph, alpha, beta, interval_minutes=5., hidden=8):
        super().__init__(graph, alpha, beta, interval_minutes)
        require(isinstance(hidden, int) and not isinstance(hidden, bool) and hidden > 0, 'Invalid hidden size')
        d = alpha.shape[-1]
        self.message = nn.Sequential(nn.Linear(3*d, hidden), nn.Tanh(), nn.Linear(hidden, d), nn.Tanh())
        self.transition = nn.Linear(3*d, d)
        self.gate = nn.Linear(3*d, d)

    def forward(self, initial, capacity, boundary, times, outgoing_weights):
        h = validate_inputs(initial, capacity, boundary, times, outgoing_weights,
                            self.graph, self.alpha, self.beta, self.interval_minutes)
        require(self.transition.weight.device == initial.device and self.transition.weight.dtype == initial.dtype,
                'Ordinary model device/dtype mismatch')
        src, dst = self.graph.edge_index
        state, states, messages, ins, outs = initial, [initial], [], [], []
        for k in range(h.numel()):
            left = torch.where((src >= 0)[None, :, None], state[:, src.clamp_min(0)], 0)
            right = torch.where((dst >= 0)[None, :, None], state[:, dst.clamp_min(0)], 0)
            message = self.message(torch.cat((left, right, capacity[:, k]), -1))
            message = torch.where((src < 0)[None, :, None], boundary[:, k], message)
            incoming, outgoing = self.graph.aggregate(message, incoming=True), self.graph.aggregate(message, incoming=False)
            features = torch.cat((state, incoming, outgoing), -1)
            # A time-scaled gate avoids silently doubling the gate's time unit
            # when the internal step size changes.
            gamma = -torch.expm1(-h[k] * torch.nn.functional.softplus(self.gate(features)))
            updated = (1-gamma)*state + gamma*torch.sigmoid(self.transition(features))
            state = torch.where(self.graph.operator_mask[None, :, None], updated, state)
            states.append(state)
            messages.append(message)
            ins.append(incoming)
            outs.append(outgoing)
        return dict(states=torch.stack(states, 1), edge_flux=torch.stack(messages, 1),
                    inflow=torch.stack(ins, 1), outflow=torch.stack(outs, 1), times=times,
                    interval_minutes=self.interval_minutes, operator_mask=self.graph.operator_mask,
                    kind='ordinary_messages_NOT_physical_flux')


class LocalCapacityRecurrence(CapacityLimitedExchange):
    """L1: independent future cells with history-predicted incoming demand.

    An internal edge has separate arrival and departure estimates. They are
    deliberately NOT a shared interface flux, so global balance is not claimed.
    No evolving neighbour state enters either estimate. Local sending/receiving
    budgets still preserve [0,1] without clipping.
    """
    def forward(self, initial, capacity, boundary, times, outgoing_weights, *, local_demand):
        h = validate_inputs(initial, capacity, boundary, times, outgoing_weights,
                            self.graph, self.alpha, self.beta, self.interval_minutes)
        compatible(local_demand, initial, 'History-predicted local demand')
        require(local_demand.shape == capacity.shape and (local_demand >= 0).all(),
                'Invalid local incoming demand')
        src, dst = self.graph.edge_index
        internal = (src >= 0) & (dst >= 0)
        require((local_demand[:, :, ~internal] == 0).all(), 'Local demand belongs only to internal edges')
        state, states, records = initial, [initial], []
        for k in range(h.numel()):
            sending = outgoing_weights[None, :, None] * self.alpha[src.clamp_min(0)] * state[:, src.clamp_min(0)]
            departure = torch.where((src >= 0)[None, :, None], torch.minimum(sending, capacity[:, k]), 0)
            arrival_bid = torch.where((src < 0)[None, :, None], boundary[:, k],
                                      torch.minimum(local_demand[:, k], capacity[:, k]))
            incoming_bid = self.graph.aggregate(arrival_bid, incoming=True)
            denominator = torch.where(incoming_bid > 0, incoming_bid, 1)
            scale = torch.minimum(torch.ones_like(state), self.beta*(1-state)/denominator)
            arrival = arrival_bid * scale[:, dst.clamp_min(0)]
            inflow = self.graph.aggregate(arrival, incoming=True)
            outflow = self.graph.aggregate(departure, incoming=False)
            state = state + h[k]*(inflow-outflow)
            states.append(state)
            records.append(dict(inflow=inflow, outflow=outflow, edge_inflow=arrival,
                                edge_outflow=departure, receiving_scale=scale))
        return dict(states=torch.stack(states, 1), times=times,
                    interval_minutes=self.interval_minutes, operator_mask=self.graph.operator_mask,
                    kind='local_independent_arrivals_NOT_shared_flux',
                    **{key: torch.stack([r[key] for r in records], 1) for key in records[0]})


def target_window_features(rollout, windows):
    """[B,H,N,4d]: end state, mean in/out rates, within-window state change.

    Windows are explicit [H,2] cutoff-relative minutes, NOT horizon labels.
    They must tile whole internal steps. No interpolation/target traffic enters.
    """
    states, times = rollout['states'], rollout['times']
    compatible(windows, times, 'Target windows')
    require(windows.ndim == 2 and windows.shape[1] == 2 and windows.shape[0] > 0
            and (windows[:, 1] > windows[:, 0]).all()
            and (windows[1:, 0] >= windows[:-1, 1]).all(), 'Invalid target windows')
    h = torch.diff(times) / rollout['interval_minutes']
    result = []
    tol = 32 * torch.finfo(times.dtype).eps
    for start, end in windows:
        starts = torch.where(torch.isclose(times, start, atol=tol, rtol=0))[0]
        ends = torch.where(torch.isclose(times, end, atol=tol, rtol=0))[0]
        require(starts.numel() == ends.numel() == 1, 'Target endpoint is off the internal time grid')
        a, b = int(starts[0]), int(ends[0])
        duration = h[a:b].sum()
        aggregate = lambda name: (rollout[name][:, a:b] * h[None, a:b, None, None]).sum(1)/duration
        result.append(torch.cat((states[:, b], aggregate('inflow'), aggregate('outflow'),
                                 states[:, b]-states[:, a]), -1))
    features = torch.stack(result, 1)
    return torch.where(rollout['operator_mask'][None, None, :, None], features, 0)
