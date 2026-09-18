"""Conditional research inputs and residual time weights; no physical effect claim."""
import torch
from torch import nn


def history_state(history_data):
    """Per-node state from standardized historical flow only, never future labels."""
    flow = history_data[..., 0]
    return torch.stack([flow.mean(dim=1), flow[:, -1],
                        flow[:, -3:].mean(dim=1) - flow[:, :3].mean(dim=1)], dim=-1)


def forecast_clock_embeddings(incident_data, tod_table, dow_table):
    features = []
    for field, table in (('forecast_tod', tod_table), ('forecast_dow', dow_table)):
        index = incident_data[field]
        if (index.ndim != 1 or index.dtype not in (torch.int32, torch.int64)
                or torch.any(index < 0) or torch.any(index >= table.shape[0])):
            raise ValueError(f'{field} must be an in-range integer vector')
        features.append(table[index])
    return features


class ReportLocationEncoder(nn.Module):
    """Report age and forecast clock; location reaches ICSF through distances.

    Availability of the supplied report location is a conditional development
    assumption. No description, type, holiday, duration or old position is read.
    """
    def __init__(self, hidden_dim, time_emb_dim):
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear(1 + 2 * time_emb_dim, 64), nn.ReLU(),
                                    nn.Linear(64, hidden_dim))

    def forward(self, report_age_minutes, tod_features, dow_features):
        if (report_age_minutes.ndim != 1 or report_age_minutes.shape[0] != tod_features.shape[0]
                or not torch.isfinite(report_age_minutes).all() or torch.any(report_age_minutes < 0)):
            raise ValueError('report_age_minutes must be a finite nonnegative batch vector')
        age = report_age_minutes.to(tod_features).unsqueeze(-1) / 5
        return self.fusion(torch.cat([age, tod_features, dow_features], dim=-1))


class IncidentTimeResponse(nn.Module):
    """g(h) + tanh(offset): signed prediction-residual weights, not physical decay.

    The learned offset is added outside the Gaussian so long-horizon gradients
    do not inherit its near-zero scale. Instantiate after the common model.
    """
    def __init__(self, mode, horizon=12):
        super().__init__()
        if mode not in ('shared', 'conditioned'):
            raise ValueError('Learned time response must be shared or conditioned')
        self.mode = mode
        self.b = nn.Parameter(torch.zeros(horizon))
        if mode == 'conditioned':
            self.mlp = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, horizon, bias=False))
            nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, gaussian, state=None, connected_mask=None, report_age_minutes=None):
        offset = self.b.view(1, -1, 1, 1)
        if self.mode == 'conditioned':
            if state is None or state.ndim != 3 or state.shape[-1] != 3:
                raise ValueError('Conditioned response requires historical state [batch,node,3]')
            offset = offset + self.mlp(state).permute(0, 2, 1).unsqueeze(-1)
        return gaussian + torch.tanh(offset)


class ConstrainedPhaseResponse(nn.Module):
    """Event-level mixture of smooth nonnegative residual-response bases.

    The coefficients remain prediction-residual weights, not identified causal
    effects.  Traffic state is pooled over connected nodes so it can choose one
    event curve, while the existing TIID context retains spatial heterogeneity.
    """
    def __init__(self, horizon=12, hidden_dim=16, initial_gate_logit=-8.0):
        super().__init__()
        if horizon < 2 or hidden_dim < 1:
            raise ValueError('Phase response requires horizon >= 2 and a positive hidden dimension')
        self.horizon = int(horizon)
        self.phase_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4, bias=False),
        )
        nn.init.zeros_(self.phase_encoder[-1].weight)

        steps = torch.arange(1, horizon + 1, dtype=torch.float32)
        peak_scale = torch.exp(torch.tensor(-0.5))
        immediate = torch.exp(-steps.square() / 2)
        sustained = peak_scale * torch.exp(-(steps - 1) / 6)
        delayed = peak_scale * torch.exp(-(steps - 6).square() / (2 * 2.5 ** 2))
        self.register_buffer('phase_bases', torch.stack([immediate, sustained, delayed]))
        self.register_buffer('initial_gate_logit', torch.tensor(float(initial_gate_logit)))

    def forward(self, gaussian, state=None, connected_mask=None, report_age_minutes=None):
        if gaussian.ndim != 4 or gaussian.shape[1] != self.horizon:
            raise ValueError('Phase response requires gaussian [1,horizon,1,1]')
        if state is None or state.ndim != 3 or state.shape[-1] != 3:
            raise ValueError('Phase response requires historical state [batch,node,3]')
        if connected_mask is None:
            raise ValueError('Phase response requires a connected-node mask')
        if connected_mask.ndim == 3 and connected_mask.shape[-1] == 1:
            connected_mask = connected_mask.squeeze(-1)
        if connected_mask.shape != state.shape[:2]:
            raise ValueError('Connected-node mask must match the history state node axis')
        if (report_age_minutes is None or report_age_minutes.ndim != 1
                or report_age_minutes.shape[0] != state.shape[0]
                or not torch.isfinite(report_age_minutes).all()
                or torch.any(report_age_minutes < 0)):
            raise ValueError('Phase response requires finite nonnegative report age [batch]')

        weights = connected_mask.to(state)
        counts = weights.sum(dim=1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError('Each phase-response sample requires at least one connected node')
        event_state = (state * weights.unsqueeze(-1)).sum(dim=1) / counts
        age = report_age_minutes.to(state).unsqueeze(-1) / 5
        phase_parameters = self.phase_encoder(torch.cat([event_state, age], dim=-1))

        mixture_weights = torch.softmax(phase_parameters[:, :3], dim=-1)
        mixture = mixture_weights @ self.phase_bases.to(state)
        gate = torch.sigmoid(self.initial_gate_logit.to(state) + phase_parameters[:, 3:4])
        baseline = gaussian.to(state).reshape(1, self.horizon)
        response = (1 - gate) * baseline + gate * mixture
        return response.unsqueeze(-1).unsqueeze(-1)
