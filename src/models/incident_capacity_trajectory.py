"""M2.2 bounded transient capacity loss and a parameter-matched monotone control.

Time scales are minutes; outputs are relative states, not measured capacity.
The single Gamma-shaped pulse is a design assumption, not an incident label.
"""
import torch
from torch import nn

from src.models.incident_relative_capacity import SharedIncidentState, finite_float, require


def pulse_kernel(time, peak_minutes):
    """Nonnegative time / positive peak, checked at the module input boundary."""
    u = time / peak_minutes
    return u * torch.exp(1 - u)


def capacity_curve(amplitude, recovery_minutes, strength, peak_minutes, elapsed_minutes, trajectory):
    """[B,N] parameters + [K] minutes -> [B,K,N] relative capacity loss."""
    require(trajectory in ('pulse', 'mixture'), 'Expected pulse or monotone mixture')
    require(amplitude.ndim == 2, 'Capacity parameters must be [B,N]')
    for value in (amplitude, recovery_minutes, strength, peak_minutes, elapsed_minutes):
        finite_float(value, 'Trajectory input')
        require(value.device == amplitude.device and value.dtype == amplitude.dtype, 'Trajectory device/dtype mismatch')
    require(all(v.shape == amplitude.shape for v in (recovery_minutes, strength, peak_minutes)), 'Trajectory parameter axes mismatch')
    require(((amplitude >= 0) & (amplitude <= .95)).all() and ((strength >= 0) & (strength <= 1)).all(),
            'Amplitude or pulse strength out of range')
    require(((recovery_minutes >= 5) & (recovery_minutes <= 180)).all()
            and ((peak_minutes >= 15) & (peak_minutes <= 90)).all(), 'Time scales outside the declared engineering range')
    require(elapsed_minutes.ndim == 1 and elapsed_minutes.numel() > 0 and (elapsed_minutes >= 0).all()
            and (torch.diff(elapsed_minutes) > 0).all(), 'Invalid elapsed minutes')
    time = elapsed_minutes[None, :, None]
    base = amplitude[:, None] * torch.exp(-time / recovery_minutes[:, None])
    if trajectory == 'pulse':
        return base + (.95 - base) * strength[:, None] * pulse_kernel(time, peak_minutes[:, None])
    # Equivalent to a convex mixture; this form returns base exactly at p=0.
    second = amplitude[:, None] * torch.exp(-time / peak_minutes[:, None])
    return base + strength[:, None] * (second - base)


class TrajectoryIncidentState(SharedIncidentState):
    def __init__(self, hidden=16, trajectory='pulse'):
        require(trajectory in ('pulse', 'mixture', 'ordinary'), 'Unknown trajectory')
        super().__init__(hidden=hidden, mode='ordinary' if trajectory == 'ordinary' else 'capacity')
        self.trajectory = trajectory
        original = self.condition_head[-1]
        expanded = nn.Linear(hidden, 4)
        nn.init.normal_(expanded.weight, std=.01)
        nn.init.zeros_(expanded.bias)
        with torch.no_grad():
            expanded.weight[:2].copy_(original.weight)
            expanded.bias[:2].copy_(original.bias)
        self.condition_head[-1] = expanded

    def configuration(self):
        return {'hidden': self.hidden, 'trajectory': self.trajectory}

    def initialize_common_from(self, legacy):
        """Copy all M2.1 weights; preserve the two new shape output rows."""
        require(type(legacy) is SharedIncidentState and legacy.mode == 'capacity' and legacy.hidden == self.hidden,
                'Expected matching M2.1 capacity model')
        current = self.state_dict()
        for key, value in legacy.state_dict().items():
            if key in ('condition_head.2.weight', 'condition_head.2.bias'):
                current[key][:2].copy_(value)
            else:
                current[key].copy_(value)
        self.load_state_dict(current)

    def _decode_state(self, raw, time, effective_support, result):
        active = effective_support[:, None]
        peak = 15 + 75 * torch.sigmoid(raw[..., 3])
        if self.trajectory == 'ordinary':
            delta = (raw[:, None, :, 0] + raw[:, None, :, 1] * torch.tanh(time / 60)
                     + raw[:, None, :, 2] * pulse_kernel(time, peak[:, None]))
            result['state'] = 1 + torch.where(active, delta, 0)
            return result
        amplitude = (.10 + .20 * raw[..., 0]).clamp(0, .95)
        tau = 5 + 175 * torch.sigmoid(raw[..., 1])
        strength = (.10 + .20 * raw[..., 2]).clamp(0, 1)
        loss = capacity_curve(amplitude, tau, strength, peak, time[0, :, 0], self.trajectory)
        loss = torch.where(active, loss, 0)
        result.update(state=1-loss, capacity_loss=loss,
                      amplitude_at_cutoff=torch.where(effective_support, amplitude, 0),
                      recovery_minutes=tau, shape_strength=torch.where(effective_support, strength, 0),
                      shape_time_minutes=peak,
                      amplitude_saturated=effective_support & ((amplitude == 0) | (amplitude == .95)),
                      strength_saturated=effective_support & ((strength == 0) | (strength == 1)))
        return result
