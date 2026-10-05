"""Opt-in inference audit of the current single-report ICSF; no production replacement."""

import torch
from torch import nn


class SingleIncidentICSF(nn.Module):
    """Reuse the original parameters, bypassing only singleton attention arithmetic.

    Training is deliberately unsupported: skipping the fusion MLP's dropout changes
    RNG consumption even when its singleton attention cannot change the output.
    K is retained because the separate TIID context uses it.
    """

    def __init__(self, base):
        super().__init__()
        if base.incident_schema != 'report_location_v1' or base.use_sensor_info:
            raise ValueError('Requires report_location_v1 without static sensor features')
        self.base = base

    def forward(self, history_data, incident_data, sensor_data=None,
                incident_tod_feat=None, incident_day_feat=None):
        if self.training or self.base.training or torch.is_grad_enabled():
            raise ValueError('SingleIncidentICSF is eval/inference-only')
        if history_data.dtype != torch.float32 or history_data.ndim != 4:
            raise ValueError('Expected float32 history [batch,time,node,hidden]')
        if sensor_data is not None:
            raise ValueError('Static sensor input is outside this equivalence scope')
        distances = incident_data['distances']
        if distances.shape != (history_data.shape[0], history_data.shape[2], 3):
            raise ValueError('Expected one report with distances [batch,node,3]')
        if (not torch.isfinite(history_data).all() or not torch.isfinite(distances).all()
                or incident_data['report_age_minutes'].shape != (history_data.shape[0],)):
            raise ValueError('Nonfinite history/distance or multiple reports')
        embedding = self.base.embed_incident_features(
            incident_data, incident_tod_feat, incident_day_feat)
        key = self.base.k_proj(embedding)
        value = self.base.v_proj(embedding).unsqueeze(1)
        mask = (distances.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
        enhanced = history_data.clone()
        enhanced[:, -1] = self.base.output_norm(history_data[:, -1] + mask * value)
        return enhanced, {
            'incident_key': key,
            'sensor_features': history_data.new_empty(history_data.shape[0], history_data.shape[2], 0),
            'distances': distances,
        }


@torch.inference_mode()
def compare_single_incident(model, x, incident, atol=1e-6, rtol=1e-6):
    """Compare the real forward path twice and restore the module even on failure."""
    if (model.training or any(module.training for module in model.modules())
            or model._incident_schema != 'report_location_v1'
            or model._time_response_mode != 'fixed'):
        raise ValueError('Requires eval-mode fixed report_location_v1 model')
    base = model.icsf_module
    simplified = SingleIncidentICSF(base).eval()
    captured = []

    def capture(_module, _args, output):
        history, context = output
        captured.append({'enhanced_history': history.clone(),
                         **{name: tensor.clone() for name, tensor in context.items()}})

    handle = base.register_forward_hook(capture)
    try:
        native = model(x, incident_data=incident)
    finally:
        handle.remove()
    handle = simplified.register_forward_hook(capture)
    try:
        model.icsf_module = simplified
        candidate = model(x, incident_data=incident)
    finally:
        model.icsf_module = base
        handle.remove()
    if len(captured) != 2:
        raise ValueError('Expected exactly one ICSF invocation in each forward')
    captured[0]['forecast'], captured[1]['forecast'] = native, candidate
    result = {}
    for name, reference in captured[0].items():
        actual = captured[1][name]
        if reference.shape != actual.shape or not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
            raise ValueError(f'Nonfinite or mismatched equivalence tensor: {name}')
        difference = (reference.double() - actual.double()).abs()
        if not torch.allclose(reference, actual, atol=atol, rtol=rtol):
            raise ValueError(f'Single-incident equivalence failed: {name}')
        result[name] = {
            'cells': reference.numel(),
            'abs_sum': float(difference.sum()),
            'abs_max': float(difference.max()) if difference.numel() else 0.,
            'exactly_equal': bool(torch.equal(reference, actual)),
        }
    return result
