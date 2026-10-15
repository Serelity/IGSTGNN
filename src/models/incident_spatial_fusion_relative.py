"""Temporary inference-only relative-coefficient probes for existing MIN-F1.

The old model and its checkpoint schema are unchanged. A hook transforms only
the ten learned coefficient increments, before the existing feature fusion.
Zero-mean coefficients do not imply an unchanged forecast mean or physical
conservation. The bounded case also rescales the relative coefficients.
"""
from contextlib import contextmanager
import math

import torch

from src.models.incident_spatial import require
from src.models.incident_spatial_forecast_fusion import ForecastFusion, ForecastFusionIGSTGNN


MODES = ('exact', 'bounded')
SCHEMA = 'MIN_F1_relative_coefficient_intervention_v1'


def transform_delta(delta, mode, max_delta=.1):
    """Return centered coefficient increments and compact input-only statistics.

    Exact: r = delta - mean(delta). Bounded: r * min(1, a/max(abs(r))).
    All ten components of one sample/horizon/node share the bounded factor.
    Float64 arithmetic avoids introducing a residual for constant float32 inputs;
    the result retains the model's dtype and device.
    """
    require(mode in MODES, 'Unknown relative fusion mode')
    require(type(max_delta) in (int, float) and math.isfinite(max_delta) and 0 < max_delta < 1,
            'Relative fusion max_delta must be finite and between zero and one')
    require(isinstance(delta, torch.Tensor) and delta.ndim == 4 and delta.shape[1] == 12
            and delta.shape[-1] == 10 and delta.shape[0] > 0 and delta.shape[2] > 0
            and delta.dtype in (torch.float32, torch.float64) and bool(torch.isfinite(delta).all()),
            'Relative fusion delta must be finite float32/float64 [batch,12,node,10]')
    require(not delta.requires_grad, 'Relative fusion transform is inference only')
    a, epsilon = float(max_delta), torch.finfo(delta.dtype).eps
    tolerance = 32*epsilon*a
    require(float(delta.detach().abs().max()) <= a+tolerance, 'Original fusion delta exceeds its registered bound')
    wide = delta.to(torch.float64)
    common = wide.mean(-1, keepdim=True)
    relative = wide-common
    peak = relative.abs().amax(-1, keepdim=True)
    scale = a/peak.clamp_min(a) if mode == 'bounded' else torch.ones_like(peak)
    result = (relative*scale).to(dtype=delta.dtype)
    sum_residual = result.to(torch.float64).sum(-1).abs().max()
    require(float(sum_residual) <= 64*epsilon*a, 'Centered coefficient sum is not numerically zero')
    if mode == 'bounded':
        require(float(result.abs().max()) <= a+tolerance, 'Bounded relative fusion exceeds original bound')
    positions = int(peak.numel())
    stats = dict(mode=mode, max_delta=a, positions=positions, component_positions=int(delta.numel()),
        raw_common_delta_signed_sum=float(common.sum()), raw_common_delta_abs_sum=float(common.abs().sum()),
        raw_common_delta_abs_max=float(common.abs().max()), raw_delta_abs_max=float(wide.abs().max()),
        uncapped_relative_abs_max=float(peak.max()), output_delta_abs_max=float(result.abs().max()),
        output_delta_component_sum_abs_max=float(sum_residual),
        uncapped_exceeds_bound_positions=int((peak > a).sum()),
        bounded_rescaled_positions=int((scale < 1.).sum()),
        bounded_scale_min=float(scale.min()), bounded_scale_sum=float(scale.sum()))
    return result, stats


@contextmanager
def relative_fusion(model, mode='exact', expected_calls=1):
    """Intercept one existing EVENT fusion module, then always remove the hook.

    The caller still runs the original model forward under inference/no_grad.
    Successful use must cover exactly expected_calls. An exception removes the
    hook without replacing the original exception with a coverage error.
    """
    require(mode in MODES, 'Unknown relative fusion mode')
    require(type(expected_calls) is int and expected_calls > 0, 'Expected relative hook calls must be positive')
    require(type(model) is ForecastFusionIGSTGNN and getattr(model, '_forecast_fusion_mode', None) == 'event'
            and type(getattr(model, 'forecast_fusion', None)) is ForecastFusion,
            'Relative fusion requires an existing EVENT MIN-F1 architecture')
    require(not model.training and not model.forecast_fusion.training, 'Relative fusion requires evaluation mode')
    module = model.forecast_fusion
    require(not module._forward_hooks and not module._forward_pre_hooks,
            'Relative fusion rejects pre-existing fusion hooks')
    require(getattr(module.forward, '__func__', None) is ForecastFusion.forward,
            'Relative fusion requires the original coefficient forward')
    audit = dict(schema=SCHEMA, mode=mode, max_delta=float(module.max_delta), hook_calls=0)

    def replace(current, inputs, output):
        require(current is module and not model.training and not current.training,
                'Relative fusion hook coverage or evaluation mode changed')
        require(not torch.is_grad_enabled(), 'Relative fusion forward must run under inference/no_grad')
        replacement, stats = transform_delta(output, mode, module.max_delta)
        audit['hook_calls'] += 1
        for name, value in stats.items():
            if name in ('mode', 'max_delta'):
                require(audit[name] == value, 'Relative fusion static audit fields changed')
            elif name.endswith('_max'):
                audit[name] = max(audit.get(name, 0.), value)
            elif name.endswith('_min'):
                audit[name] = min(audit.get(name, 1.), value)
            else:
                audit[name] = audit.get(name, 0)+value
        return replacement

    handle = module.register_forward_hook(replace)
    try:
        yield audit
    except BaseException:
        raise
    else:
        require(audit['hook_calls'] == expected_calls, 'Incomplete relative fusion hook coverage')
    finally:
        handle.remove()
