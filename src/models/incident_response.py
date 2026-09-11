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

    def forward(self, gaussian, state=None):
        offset = self.b.view(1, -1, 1, 1)
        if self.mode == 'conditioned':
            if state is None or state.ndim != 3 or state.shape[-1] != 3:
                raise ValueError('Conditioned response requires historical state [batch,node,3]')
            offset = offset + self.mlp(state).permute(0, 2, 1).unsqueeze(-1)
        return gaussian + torch.tanh(offset)
