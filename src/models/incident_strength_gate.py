"""Identity-initialized ICSF adapters for the frozen, single-report v12c experiment."""

import torch
from torch import nn


class IncidentStrengthGate(nn.Module):
    def __init__(self, variant, hidden_dim, width=16):
        super().__init__()
        self.variant = variant
        if variant == 'scalar':
            self.logit = nn.Parameter(torch.zeros(()))
        elif variant == 'node':
            self.network = nn.Sequential(nn.Linear(3 * hidden_dim + 4, width), nn.Tanh(),
                                         nn.Linear(width, 1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        else:
            raise ValueError('Expected scalar or node gate')

    def forward(self, history, incident):
        b, _, n, _ = history.shape
        if self.variant == 'scalar':
            logits = self.logit.expand(b, n, 1)
        else:
            age = incident['report_age_minutes'].reshape(b, 1, 1).expand(b, n, 1) / 5.
            features = torch.cat((history[:, -1], history.mean(1),
                                  history[:, -1] - history[:, 0],
                                  incident['distances'], age), dim=-1)
            logits = self.network(features)
        return 2 * torch.sigmoid(logits)


class GatedICSF(nn.Module):
    """Use the exact M=1 injection; keep native normalization and TIID context.

    The frozen native ICSF has two singleton softmax axes, so injection=M*V.
    Restrict this adapter to the report_location_v1/no-sensor configuration.
    The surrounding native backbone, graph constructor and TIID are unmodified.
    """

    def __init__(self, base, variant, width=16):
        super().__init__()
        if base.incident_schema != 'report_location_v1' or base.use_sensor_info:
            raise ValueError('v12c requires one report and no sensor feature branch')
        self.base = base.requires_grad_(False)
        self.gate = IncidentStrengthGate(variant, base.q_proj.in_features, width)
        self.last_gate = None

    def forward(self, history, incident, sensor_data=None, incident_tod_feat=None,
                incident_day_feat=None):
        embedding = self.base.embed_incident_features(incident, incident_tod_feat, incident_day_feat)
        distances = incident['distances']
        mask = (distances.abs().sum(-1, keepdim=True) > 0).to(history.dtype)
        strength = self.gate(history, incident)
        self.last_gate = strength
        injection = mask * self.base.v_proj(embedding)[:, None, :]
        enhanced = history.clone()
        enhanced[:, -1] = self.base.output_norm(history[:, -1] + strength * injection)
        return enhanced, {'incident_key': self.base.k_proj(embedding),
                          'sensor_features': history.new_empty(history.shape[0], history.shape[2], 0),
                          'distances': distances}


def attach_gate(model, variant, width=16):
    """Load the original checkpoint strictly BEFORE attaching an adapter."""
    if isinstance(model.icsf_module, GatedICSF):
        raise ValueError('Gate already attached')
    model.requires_grad_(False).eval()
    original = model.icsf_module
    adapter = GatedICSF(original, variant, width).to(next(model.parameters()).device)
    model.icsf_module = adapter
    return adapter.gate
