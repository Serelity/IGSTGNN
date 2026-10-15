"""Attenuate only the current E1 report response by forecast horizon.

Both evaluations share every core parameter. Both paths differentiate through
the loss. This is not a second independent predictor or a free output residual.
Only twelve projected scalar weights are added. Training recomputes activations
to avoid retaining two complete MIN activation sets at once.
"""
from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.models.incident_spatial import require
from src.models.incident_spatial_early_event import EarlyEventIGSTGNN


MAPPING_VERSION = 'early_event_shared_core_horizon_attenuation_v1'
MODES = ('control', 'horizon')


@contextmanager
def shadow_batchnorm_buffers(model):
    """Give an auxiliary/recomputed pass private running buffers.

Swapping buffers, rather than copying them in-place afterwards, also protects
the versions of tensors already saved by the main training pass for backward.
Only the original ON forward updates persistent BatchNorm statistics.
"""
    saved = []
    try:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                    value = module._buffers.get(name)
                    if value is not None:
                        saved.append((module, name, value))
                        module._buffers[name] = value.detach().clone()
        yield
    finally:
        for module, name, value in reversed(saved):
            module._buffers[name] = value


class EventHorizonIGSTGNN(nn.Module):
    """y = ON + (g_h - 1) * (ON - OFF), with 0 <= g_h <= 1.

Initial g=1 returns the existing ON prediction bit-for-bit. Projecting the
offset after each optimizer step is mandatory; forward clamping is a guard.
CONTROL registers the same offsets but freezes them at zero and runs ON once.
"""
    def __init__(self, core, mode='horizon', checkpoint_branches=True):
        super().__init__()
        require(isinstance(core, EarlyEventIGSTGNN) and core._early_event_mode == 'event',
                'Expected an EVENT-mode early-event core')
        require(mode in MODES and type(checkpoint_branches) is bool, 'Invalid horizon mode/checkpoint flag')
        require(all(parameter.requires_grad for parameter in core.parameters()),
                'The complete current EVENT backbone and conditioner must train jointly')
        self.core, self.mode, self.checkpoint_branches = core, mode, checkpoint_branches
        reference = next(core.parameters())
        self.horizon_offset = nn.Parameter(reference.new_zeros(12), requires_grad=mode == 'horizon')
        self.train(core.training)

    def param_num(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def horizon_weights(self):
        require(bool(torch.isfinite(self.horizon_offset).all()), 'Nonfinite horizon offsets')
        if self.mode == 'control':
            require(bool((self.horizon_offset == 0).all()), 'CONTROL offsets must remain zero')
        return 1.+self.horizon_offset.clamp(-1., 0.)

    @torch.no_grad()
    def project_horizon_weights_(self):
        require(bool(torch.isfinite(self.horizon_offset).all()), 'Nonfinite horizon offsets')
        self.horizon_offset.clamp_(-1., 0.)
        if self.mode == 'control':
            self.horizon_offset.zero_()

    def _branch(self, history, label, incident, sensor, *, enabled, semantic, details, auxiliary):
        kwargs = dict(incident_data=incident, sensor_data=sensor, report_enabled=enabled,
                      semantic_enabled=semantic, return_spatial_details=details)
        def function(value):
            return self.core(value, label, **kwargs)
        use_checkpoint = self.training and torch.is_grad_enabled() and self.checkpoint_branches
        if use_checkpoint:
            require(not details, 'Training diagnostics must not retain full paired activations')
            def contexts():
                return (shadow_batchnorm_buffers(self.core) if auxiliary else nullcontext(),
                        shadow_batchnorm_buffers(self.core))
            return checkpoint(function, history, use_reentrant=False,
                              preserve_rng_state=True, context_fn=contexts)
        with shadow_batchnorm_buffers(self.core) if auxiliary and self.training else nullcontext():
            return function(history)

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None, *,
                report_enabled=True, semantic_enabled=True, return_spatial_details=False,
                horizon_enabled=True):
        require(all(type(value) is bool for value in (report_enabled, semantic_enabled,
                return_spatial_details, horizon_enabled)), 'Horizon switches must be bool')
        require(not (self.training and return_spatial_details), 'Collect paired diagnostics in evaluation mode')
        require(history_data.device == next(self.parameters()).device, 'History/model device mismatch')
        weights = self.horizon_weights()
        # OFF and raw-ON controls need only one core call.
        paired = report_enabled and horizon_enabled and self.mode == 'horizon'
        cpu_start = torch.get_rng_state().clone() if paired and self.training else None
        devices = [history_data.device.index if history_data.device.index is not None
                   else torch.cuda.current_device()] if history_data.device.type == 'cuda' else []
        cuda_start = torch.cuda.get_rng_state(devices[0]).clone() if cpu_start is not None and devices else None
        on = self._branch(history_data, label, incident_data, sensor_data, enabled=report_enabled,
                          semantic=semantic_enabled, details=return_spatial_details, auxiliary=False)
        on_prediction = on['prediction'] if return_spatial_details else on
        require(on_prediction.ndim == 4 and on_prediction.shape[1] == 12,
                'Horizon model requires twelve forecast slots')
        if not paired:
            if not return_spatial_details:
                return on_prediction
            return dict(prediction=on_prediction, spatial=on['spatial'], horizon=dict(
                mode=self.mode, weights=weights.detach(), paired_core_calls=1, response_measured=False,
                raw_on=on_prediction.detach() if report_enabled else None,
                raw_off=on_prediction.detach() if not report_enabled else None,
                raw_delta=None, applied_delta=None))
        # An OFF pass receives the identical stochastic stream. It does not
        # advance the caller's stream or change persistent normalization state.
        with torch.random.fork_rng(devices=devices, enabled=cpu_start is not None):
            if cpu_start is not None:
                torch.set_rng_state(cpu_start)
                if cuda_start is not None:
                    torch.cuda.set_rng_state(cuda_start, devices[0])
            off = self._branch(history_data, label, incident_data, sensor_data, enabled=False,
                               semantic=semantic_enabled, details=return_spatial_details, auxiliary=True)
        off_prediction = off['prediction'] if return_spatial_details else off
        delta = on_prediction-off_prediction
        gate = weights.view(1, 12, 1, 1)
        prediction = on_prediction+(gate-1.)*delta
        if not return_spatial_details:
            return prediction
        return dict(prediction=prediction, spatial=on['spatial'], horizon=dict(
            mode=self.mode, weights=weights.detach(), paired_core_calls=2, response_measured=True,
            raw_on=on_prediction.detach(), raw_off=off_prediction.detach(),
            raw_delta=delta.detach(), applied_delta=(gate*delta).detach()))
