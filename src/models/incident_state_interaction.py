"""v12f: bounded, identity-initialized changes to the frozen ICSF value vector."""

import torch
from torch import nn

from src.models.incident_strength_gate import GatedICSF, IncidentStrengthGate

ARMS = ('strength', 'state_vector', 'interaction_vector')


class IncidentStateVector(nn.Module):
    def __init__(self, hidden_dim, interaction):
        super().__init__()
        self.interaction = interaction
        self.state = nn.Linear(3 * hidden_dim + 4, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history, incident, value):
        b, _, n, _ = history.shape
        age = incident['report_age_minutes'].reshape(b, 1, 1).expand(b, n, 1) / 5.
        features = torch.cat((history[:, -1], history.mean(1),
                              history[:, -1] - history[:, 0], incident['distances'], age), -1)
        state = torch.tanh(self.state(features))
        if self.interaction:
            state = state * (1 + torch.tanh(value))
        return torch.tanh(self.output(state))


class StateInteractionICSF(nn.Module):
    def __init__(self, base, arm, width=16):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown v12f arm')
        if base.incident_schema != 'report_location_v1' or base.use_sensor_info:
            raise ValueError('v12f requires one report and no sensor feature branch')
        self.base = base.requires_grad_(False)
        self.arm = arm
        hidden = base.q_proj.in_features
        self.gate = (IncidentStrengthGate('node', hidden, width) if arm == 'strength'
                     else IncidentStateVector(hidden, arm == 'interaction_vector'))
        self.clear_observations()

    def clear_observations(self):
        self.last_gate = None
        self.last_unit_residual = None
        self.last_delta = None
        self.last_injection = None
        self.last_postnorm_delta = None
        self.last_context_modifier = None

    def forward(self, history, incident, sensor_data=None, incident_tod_feat=None,
                incident_day_feat=None):
        embedding = self.base.embed_incident_features(incident, incident_tod_feat, incident_day_feat)
        distances = incident['distances']
        mask = (distances.abs().sum(-1, keepdim=True) > 0).to(history.dtype)
        value = self.base.v_proj(embedding)[:, None, :]
        injection = mask * value
        if self.arm == 'strength':
            strength = self.gate(history, incident)
            self.last_gate = strength
            unit = strength - 1
            delta = unit * injection
            # Keep the exact original v12c arithmetic, including operation order.
            pre_norm = history[:, -1] + strength * injection
        else:
            unit = self.gate(history, incident, value)
            # Do not clamp this scale: V=0 must give delta=0 exactly.
            delta = mask * value.square().mean(-1, keepdim=True).sqrt() * unit
            pre_norm = history[:, -1] + injection + delta
        enhanced = history.clone()
        enhanced[:, -1] = self.base.output_norm(pre_norm)
        self.last_unit_residual = unit
        # Diagnostics never retain the backbone autograd graph.
        self.last_delta = delta.detach()
        self.last_injection = injection.detach()
        with torch.no_grad():
            self.last_postnorm_delta = (enhanced[:, -1].detach()
                - self.base.output_norm(history[:, -1] + injection))
        self.last_context_modifier = (1 + value.detach().tanh()).expand_as(injection)
        return enhanced, {'incident_key': self.base.k_proj(embedding),
                          'sensor_features': history.new_empty(history.shape[0], history.shape[2], 0),
                          'distances': distances}


def attach_adapter(model, arm, width=16):
    if isinstance(model.icsf_module, (GatedICSF, StateInteractionICSF)):
        raise ValueError('Adapter already attached')
    model.requires_grad_(False).eval()
    adapter = StateInteractionICSF(model.icsf_module, arm, width).to(next(model.parameters()).device)
    model.icsf_module = adapter
    return adapter.gate
