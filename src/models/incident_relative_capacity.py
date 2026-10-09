"""M2.1 shared relative capacity state; no absolute capacity or spatial solver.

See docs/incident_relative_capacity_m21.md for the frozen interpretation.
Only explicit historical inputs enter the encoder. Report conditioning affects
the added state only; this module never imports or modifies native IGSTGNN.
"""
import torch
from torch import nn


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_float(value, name):
    require(value.is_floating_point() and torch.isfinite(value).all(),
            name + ' must be finite floating point')


def supply_limited_flow(demand, baseline_capacity, retention, downstream_supply):
    """Independent interfaces, all rates in ONE caller-declared common scale.

    This is not a recurrence. Never compare per-station normalized rates across
    edges without first converting them to a common physical flow scale.
    """
    values = (demand, baseline_capacity, retention, downstream_supply)
    for value in values:
        require(value.shape == demand.shape and value.device == demand.device
                and value.dtype == demand.dtype, 'Supply/capacity axes, device or dtype mismatch')
        finite_float(value, 'Supply/capacity input')
        require((value >= 0).all(), 'Supply/capacity inputs must be nonnegative')
    require((retention <= 1).all(), 'Retention must be in [0,1]')
    return torch.minimum(torch.minimum(demand, baseline_capacity * retention), downstream_supply)


class SharedIncidentState(nn.Module):
    """Matched capacity/ordinary modes with no learned node-specific parameters.

    state: [B,K,N], history_state: [B,N,H]. Missing all history gives a neutral
    state and an explicit false history_available mask; it is not certification.
    """
    def __init__(self, hidden=16, mode='capacity'):
        super().__init__()
        require(isinstance(hidden, int) and not isinstance(hidden, bool) and hidden > 0,
                'hidden must be a positive integer')
        require(mode in ('capacity', 'ordinary'), 'Unknown state mode')
        self.hidden, self.mode = hidden, mode
        self.history_encoder = nn.GRU(6, hidden, batch_first=True)
        self.condition_head = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        # Keep the initial amplitude inside the differentiable clamp interval.
        nn.init.normal_(self.condition_head[-1].weight, std=.01)
        nn.init.zeros_(self.condition_head[-1].bias)

    def configuration(self):
        return {'hidden': self.hidden, 'mode': self.mode}

    def forward(self, history, valid, references, report_features, report_age_minutes,
                elapsed_minutes, *, incident_enabled=True):
        require(isinstance(incident_enabled, bool), 'incident_enabled must be a bool')
        require(history.ndim == 4 and history.shape[1] == 12 and history.shape[-1] == 3
                and history.shape[0] > 0 and history.shape[2] > 0, 'Expected history [B,12,N,3]')
        b, _, nodes, _ = history.shape
        require(valid.shape == history.shape and valid.dtype == torch.bool
                and valid.device == history.device, 'History mask mismatch')
        require(references.shape == (nodes, 3) and report_features.shape == (b, nodes, 3)
                and report_age_minutes.shape == (b,), 'Reference/report axes mismatch')
        require(elapsed_minutes.ndim == 1 and elapsed_minutes.numel() > 0, 'Expected nonempty elapsed time axis')
        for value in (references, report_features, report_age_minutes, elapsed_minutes):
            finite_float(value, 'State input')
            require(value.device == history.device and value.dtype == history.dtype,
                    'State input device/dtype mismatch')
        require(history.is_floating_point() and history.dtype == self.condition_head[0].weight.dtype
                and history.device == self.condition_head[0].weight.device, 'History/model device/dtype mismatch')
        require(torch.isfinite(history[valid]).all() and (history[valid] >= 0).all(),
                'Usable history must be finite and nonnegative')
        require((references > 0).all(), 'References must be positive, never capacities')
        require((report_age_minutes >= 0).all(), 'Future report age is forbidden')
        require((elapsed_minutes >= 0).all() and (torch.diff(elapsed_minutes) > 0).all(),
                'Elapsed times must be nonnegative and strictly increasing')
        require((report_features[..., 0] == 0).all()
                and ((report_features[..., 1] >= 0) & (report_features[..., 1] <= 1)).all()
                and ((report_features[..., 2] == 0) | (report_features[..., 2] == 1)).all(),
                'Expected report_location_v1 features (zero, score, source-PM-side)')

        normalized = torch.where(valid, history, 0) / references[None, None]
        require(torch.isfinite(normalized).all(), 'History/reference ratio overflow')
        encoded = torch.cat([torch.asinh(normalized), valid.to(history.dtype)], -1)
        encoded = encoded.permute(0, 2, 1, 3).reshape(b * nodes, 12, 6)
        _, hidden = self.history_encoder(encoded)
        hidden = hidden[0].reshape(b, nodes, self.hidden)
        available = valid.any(dim=1).any(dim=-1)
        support = report_features.ne(0).any(dim=-1) & available
        effective_support = support if incident_enabled else torch.zeros_like(support)
        # Reports arrive after much of X; no report is injected into the past GRU.
        event = torch.cat([report_features[..., 1:],
                           torch.log1p(report_age_minutes[:, None, None] / 60).expand(b, nodes, 1)], -1)
        raw = self.condition_head(torch.cat([hidden, event], -1))
        time = elapsed_minutes[None, :, None]
        result = {'history_state': hidden, 'history_available': available,
                  'report_support': support, 'effective_support': effective_support}
        return self._decode_state(raw, time, effective_support, result)

    def _decode_state(self, raw, time, effective_support, result):
        """Separate time expansion so new curves retain the identical input path."""
        active = effective_support[:, None]
        if self.mode == 'capacity':
            amplitude = (.10 + .20 * raw[..., 0]).clamp(0, .95)
            tau = 5 + 175 * torch.sigmoid(raw[..., 1])
            loss = torch.where(active, amplitude[:, None] * torch.exp(-time / tau[:, None]), 0)
            state = 1 - loss
            result.update(capacity_loss=loss, amplitude_at_cutoff=torch.where(effective_support, amplitude, 0),
                          recovery_minutes=tau)
        else:
            delta = raw[:, None, :, 0] + raw[:, None, :, 1] * torch.tanh(time / 60)
            state = 1 + torch.where(active, delta, 0)
        result['state'] = state
        return result
