"""M4.0 history -> cutoff state -> exchange -> pre-head residual.

All states/rates are latent. Graph qualification is external and explicit.
Reports enter ONLY the shared edge coefficient network. No hooks or targets.
"""
import torch
from torch import nn

from src.models.incident_relative_capacity import finite_float, require
from src.models.incident_capacity_exchange import (
    CapacityLimitedExchange, OrdinaryDirectedRecurrence, LocalCapacityRecurrence, compatible,
    condition_coefficients, edge_capacities, target_window_features,
)


def report_summary(weights, ages, present, distance, confidence):
    """Deduplicated reports [B,R,E] -> eight fixed edge features + association.

    Entity IDs and cutoff-visible versions must be checked by the data adapter.
    Ages are cutoff-relative, not eventual duration. No report may cover >1 edge
    in total weight. Empty collections are supported without invented reports.
    """
    finite_float(weights, 'Report weights')
    require(weights.ndim == 3, 'Expected report weights [B,R,E]')
    b, r, e = weights.shape
    for name, v in (('Report age', ages), ('Report distance', distance), ('Confidence', confidence)):
        compatible(v, weights, name)
    require(ages.shape == confidence.shape == present.shape == (b, r)
            and distance.shape == weights.shape and present.dtype == torch.bool
            and present.device == weights.device, 'Report axes mismatch')
    require(((weights >= 0) & (weights <= 1)).all()
            and (weights.sum(-1) <= 1 + 1e-6).all(), 'Report weights exceed local edge allocation')
    require((ages >= 0).all() and (distance >= 0).all()
            and ((confidence >= 0) & (confidence <= 1)).all(), 'Invalid report age/distance/confidence')
    require((weights[~present] == 0).all(), 'Absent report has association')
    if r == 0:
        return weights.new_zeros(b, e, 8), weights.new_zeros(b, e)
    supported = weights > 0
    count = supported.sum(1).to(weights.dtype)
    total = weights.sum(1)
    denominator = total.clamp_min(torch.finfo(weights.dtype).tiny)
    weighted = lambda v: (weights * v).sum(1) / denominator
    age = torch.log1p(ages)[..., None] / 5.
    minimum = torch.where(supported, age, torch.inf).amin(1)
    minimum = torch.where(count > 0, minimum, 0)
    maximum = torch.where(supported, age, 0).amax(1)
    association = 1 - (1-weights).prod(1)
    features = torch.stack(((count > 0).to(weights.dtype), torch.log1p(count), minimum,
                            weighted(age), maximum, weighted(torch.log1p(distance)),
                            weighted(confidence[..., None]), association), -1)
    return features, association


def history_summary(normalized, valid):
    """Eight per-node values; endpoint identity is retained separately per edge."""
    b, t, n, _ = normalized.shape
    counts = valid.sum(1)
    mean = normalized.sum(1) / counts.clamp_min(1)
    slots = torch.arange(t, device=valid.device)[None, :, None, None]
    last_index = torch.where(valid, slots, -1).amax(1)
    first_index = torch.where(valid, slots, t).amin(1)
    last = normalized.gather(1, last_index.clamp_min(0)[:, None]).squeeze(1)
    first = normalized.gather(1, first_index.clamp_max(t-1)[:, None]).squeeze(1)
    last = torch.where(counts > 0, last, 0)
    speed_trend = (last[..., 2] - first[..., 2]) / (last_index[..., 2]-first_index[..., 2]).clamp_min(1)
    speed_trend = torch.where(counts[..., 2] > 1, speed_trend, 0)
    fraction = valid.to(normalized.dtype).mean((1, 3))
    return torch.cat((mean, last, speed_trend[..., None], fraction[..., None]), -1)


class CutoffHistoryEncoder(nn.Module):
    def __init__(self, hidden=16, channels=4):
        super().__init__()
        self.hidden, self.channels = hidden, channels
        self.gru = nn.GRU(8, hidden, batch_first=True)
        self.cutoff_head = nn.Linear(hidden + 9, channels)

    def forward(self, history, valid, references, labels):
        require(history.ndim == 4 and history.shape[1] == 12 and history.shape[-1] == 3,
                'Expected legal history [B,12,N,3]')
        b, t, n, _ = history.shape
        require(b > 0 and valid.shape == history.shape and valid.dtype == torch.bool
                and valid.device == history.device, 'Observation mask mismatch')
        require(history.is_floating_point(), 'History must be floating point')
        compatible(references, history, 'Training references')
        compatible(labels, history, 'History labels')
        require(references.shape == (n, 3) and (references > 0).all(), 'Invalid training reference axes')
        expected = torch.arange(-65., -5., 5., dtype=history.dtype, device=history.device)
        require(labels.shape == (12,) and torch.equal(labels, expected), 'History must preserve cutoff-relative X clock')
        observed = history[valid]
        require(torch.isfinite(observed).all() and (observed >= 0).all(), 'Invalid usable observation')
        # Invalid values, including NaN, never enter normalization or the GRU.
        normalized = torch.asinh(torch.where(valid, history, 0) / references[None, None])
        require(torch.isfinite(normalized).all(), 'History normalization overflow')
        clocks = torch.stack((labels/65., (labels+5.)/65.), -1)
        clocks = clocks[None, :, None].expand(b, t, n, 2)
        encoded = torch.cat((normalized, valid.to(history.dtype), clocks), -1)
        _, hidden = self.gru(encoded.permute(0, 2, 1, 3).reshape(b*n, t, 8))
        hidden = hidden[0].reshape(b, n, self.hidden)
        summary = history_summary(normalized, valid)
        available = valid.any(1).any(-1)
        # Infer t0 from history ending at -5, explicitly exposing the blind lag.
        interval_ends = (labels+5.)[None, :, None, None]
        last_end = torch.where(valid, interval_ends, -torch.inf).amax((1, 3))
        lag = torch.where(available, -last_end/5., 0)
        hidden = torch.where(available[..., None], hidden, 0)
        initial = torch.sigmoid(self.cutoff_head(torch.cat((hidden, summary, lag[..., None]), -1)))
        initial = torch.where(available[..., None], initial, .5)
        return dict(initial=initial, hidden=hidden, summary=summary, available=available, lag_intervals=lag)


class IncidentCapacityBranch(nn.Module):
    def __init__(self, graph, outgoing_weights, *, mode='capacity', hidden=16, channels=4,
                 forecast_dim=256, substeps=4):
        super().__init__()
        require(mode in ('capacity', 'ordinary', 'local'), 'Unknown recurrence mode')
        require(isinstance(substeps, int) and substeps >= 2, 'At least two substeps required')
        require(outgoing_weights.shape == (graph.edges,), 'Outgoing weight axes mismatch')
        self.mode, self.hidden, self.channels, self.substeps = mode, hidden, channels, substeps
        self.history = CutoffHistoryEncoder(hidden, channels)
        # Left/right hidden + left/right eight-value summary + endpoint presence,
        # three edge kinds and outgoing allocation. Conditions share one MLP.
        common = 2*hidden + 16 + 2 + 3 + 1
        self.coefficients = nn.Sequential(nn.Linear(common+8, hidden), nn.Tanh(), nn.Linear(hidden, 4))
        nn.init.normal_(self.coefficients[-1].weight, std=.01)
        nn.init.constant_(self.coefficients[-1].bias, .15)
        self.boundary = nn.Sequential(nn.Linear(hidden+9, hidden), nn.Tanh(), nn.Linear(hidden, 4*channels))
        self.observation = nn.Linear(4*channels, 1)
        self.projection = nn.Linear(4*channels, forecast_dim, bias=False)
        nn.init.zeros_(self.projection.weight)
        rates = outgoing_weights.new_ones(graph.nodes, channels)
        recurrence = dict(capacity=CapacityLimitedExchange, ordinary=OrdinaryDirectedRecurrence,
                          local=LocalCapacityRecurrence)[mode]
        self.operator = recurrence(graph, rates, rates)
        self.register_buffer('outgoing_weights', outgoing_weights.clone())

    @property
    def graph(self):
        return self.operator.graph

    def prepare(self, history, valid, references, labels, reports=None, *, incident_enabled=True):
        require(isinstance(incident_enabled, bool), 'incident_enabled must be bool')
        h = self.history(history, valid, references, labels)
        require(history.shape[2] == self.graph.nodes, 'History/graph node mismatch')
        src, dst = self.graph.edge_index
        def endpoint(value, index):
            return torch.where((index >= 0)[None, :, None], value[:, index.clamp_min(0)], 0)
        left, right = endpoint(h['summary'], src), endpoint(h['summary'], dst)
        types = torch.stack((src < 0, dst < 0, (src >= 0) & (dst >= 0)), -1).to(history.dtype)
        presence = torch.stack((src >= 0, dst >= 0), -1).to(history.dtype)
        static = torch.cat((presence, types, self.outgoing_weights[:, None]), -1)
        common = torch.cat((endpoint(h['hidden'], src), endpoint(h['hidden'], dst), left, right,
                            static[None].expand(history.shape[0], -1, -1)), -1)
        denom = presence.sum(-1).clamp_min(1)[None, :, None]
        historical_condition = (left+right) / denom
        b_history = self.coefficients(torch.cat((common, historical_condition), -1)).clamp(0, .95)
        if reports is None:
            summary, association = history.new_zeros(history.shape[0], self.graph.edges, 8), history.new_zeros(history.shape[0], self.graph.edges)
        else:
            require(set(reports) == {'weights', 'ages', 'present', 'distance', 'confidence'}, 'Unknown report inputs')
            summary, association = report_summary(**reports)
            require(summary.shape[:2] == b_history.shape[:2], 'Report/history axes mismatch')
            require((association[:, src < 0] == 0).all(), 'Reports cannot change external inlet demand')
        b_report = self.coefficients(torch.cat((common, summary), -1)).clamp(0, .95)
        coefficients = condition_coefficients(b_history, b_report, association, incident_enabled=incident_enabled)
        # Report-free, four-control-point boundary predictor at marked inlets.
        boundary_input = torch.cat((h['hidden'], h['summary'], h['lag_intervals'][..., None]), -1)
        boundary_coefficients = 2*torch.sigmoid(self.boundary(boundary_input)).reshape(
            history.shape[0], self.graph.nodes, 4, self.channels)
        return dict(**h, coefficients=coefficients, history_coefficients=b_history,
                    report_association=association, boundary_coefficients=boundary_coefficients)

    def forward(self, history, valid, references, labels, reports=None, *, incident_enabled=True):
        prepared = self.prepare(history, valid, references, labels, reports, incident_enabled=incident_enabled)
        return self.rollout_prepared(prepared)

    def rollout_prepared(self, prepared, windows=None):
        """Same recurrence/readout for the main task and a shorter prefix task."""
        reference = prepared['initial']
        if windows is None:
            starts = torch.arange(5., 65., 5., device=reference.device, dtype=reference.dtype)
            windows = torch.stack((starts, starts+5), -1)
        compatible(windows, reference, 'Forecast windows')
        require(windows.ndim == 2 and windows.shape[1] == 2 and windows.numel() > 0,
                'Expected forecast windows [H,2]')
        end = float(windows[-1, 1])
        require(0 < end <= 65 and end*self.substeps/5 == int(end*self.substeps/5),
                'Forecast end must lie on the existing 65-minute grid')
        times = torch.arange(int(end*self.substeps/5)+1, device=reference.device,
                             dtype=reference.dtype)*(5./self.substeps)
        u = times[:-1]/65.
        basis = torch.stack(((1-u)**3, 3*u*(1-u)**2, 3*u*u*(1-u), u**3), -1)
        capacity = edge_capacities(prepared['coefficients'], times[:-1], 65., self.graph,
                                   self.operator.alpha, self.outgoing_weights)
        src, dst = self.graph.edge_index
        node_demand = torch.einsum('bnrd,kr->bknd', prepared['boundary_coefficients'], basis)
        boundary = torch.where((src < 0)[None, None, :, None], node_demand[:, :, dst.clamp_min(0)], 0)
        extra = {}
        if self.mode == 'local':
            internal = (src >= 0) & (dst >= 0)
            demand = self.outgoing_weights[None, None, :, None] * node_demand[:, :, src.clamp_min(0)]
            extra['local_demand'] = torch.where(internal[None, None, :, None], demand, 0)
        rollout = self.operator(prepared['initial'], capacity, boundary, times, self.outgoing_weights, **extra)
        features = target_window_features(rollout, windows)
        mask = self.graph.operator_mask[None, None, :, None]
        delta = torch.where(mask, self.projection(features), 0)
        observation = torch.where(mask, self.observation(features), 0)
        return dict(forecast_delta=delta, auxiliary_prediction=observation, features=features,
                    rollout=rollout, capacity=capacity, boundary_demand=boundary, **prepared)


class CapacityAugmentedIGSTGNN(nn.Module):
    """Explicit composition; existing ICSF/TIID receive the original trigger."""
    def __init__(self, backbone, branch):
        super().__init__()
        require(backbone._time_response_mode == 'fixed' and backbone._incident_routing == 'none',
                'M4.0 requires the complete fixed backbone')
        require(backbone.node_num == branch.graph.nodes
                and backbone._forecast_dim == branch.projection.out_features, 'Backbone/branch axes mismatch')
        self.backbone, self.branch = backbone, branch

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None, *,
                capacity_inputs, incident_enabled=True, return_details=False):
        require(set(capacity_inputs) in ({'history', 'valid', 'references', 'labels'},
                                        {'history', 'valid', 'references', 'labels', 'reports'}),
                'Only legal auxiliary history/metadata allowed; no targets')
        result = self.branch(**capacity_inputs, incident_enabled=incident_enabled)
        prediction = self.backbone(history_data, label, incident_data, sensor_data,
                                   forecast_delta=result['forecast_delta'])
        return dict(prediction=prediction, **result) if return_details else prediction
