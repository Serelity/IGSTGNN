"""Differentiable traffic transitions in explicit vehicle/hour/km units.

Point queues have no spatial storage or spillback. The CTM kernel is restricted
to a directed corridor without ramps; it is not a solver for an arbitrary GNN.
"""

import math

import torch


def positive_scalar(value, name):
    if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def nonnegative(*values):
    for value in values:
        if not value.is_floating_point() or not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError('Physical tensors must be finite, floating and nonnegative')


def point_queue_step(queue, arrival, capacity, dt_hours, *, validate=True):
    """queue [vehicles], arrival/capacity/departure [vehicles/hour]."""
    dt = positive_scalar(dt_hours, 'dt_hours')
    if queue.shape != arrival.shape or queue.shape != capacity.shape:
        raise ValueError('Point queue state and rates must have identical shapes')
    if validate:
        nonnegative(queue, arrival, capacity)
    departure = torch.minimum(capacity, arrival + queue / dt)
    # Clamp only cancellation-level negative roundoff, not excess storage.
    updated = (queue + dt * (arrival - departure)).clamp_min(0)
    return updated, departure


def rollout_point_queue(initial_queue, arrival, capacity, dt_hours):
    """Rates [batch, internal_step, bottleneck]; states include the initial one."""
    if arrival.ndim != 3 or arrival.shape != capacity.shape or initial_queue.shape != arrival[:, 0].shape:
        raise ValueError('Expected initial [B,Q] and rates [B,K,Q]')
    nonnegative(initial_queue, arrival, capacity)
    queue, states, departures = initial_queue, [initial_queue], []
    for k in range(arrival.shape[1]):
        queue, outflow = point_queue_step(queue, arrival[:, k], capacity[:, k], dt_hours, validate=False)
        states.append(queue)
        departures.append(outflow)
    return torch.stack(states, 1), torch.stack(departures, 1)


def ctm_corridor_step(density, capacity, lengths_km, free_speed_kmh,
                      wave_speed_kmh, jam_density, upstream_demand,
                      downstream_supply, dt_hours):
    """One no-ramp CTM step; density and rates are totals across all lanes.

    density/capacity: [B,C], road parameters: [C], boundary rates: [B].
    Returns [B,C] density and [B,C+1] shared interface fluxes. Report-specific
    capacity changes must leave storage fixed. Caller supplies future boundary
    *predictions*, never future observed traffic at forecast time.
    """
    dt = positive_scalar(dt_hours, 'dt_hours')
    if density.ndim != 2 or density.shape[1] < 1 or capacity.shape != density.shape:
        raise ValueError('CTM expects density and capacity [B,C]')
    b, cells = density.shape
    road = (lengths_km, free_speed_kmh, wave_speed_kmh, jam_density)
    if any(x.shape != (cells,) for x in road) or upstream_demand.shape != (b,) or downstream_supply.shape != (b,):
        raise ValueError('CTM road/boundary dimensions do not match')
    nonnegative(density, capacity, *road, upstream_demand, downstream_supply)
    if any((x <= 0).any() for x in road):
        raise ValueError('CTM road lengths, speeds and jam densities must be positive')
    tol = 32 * torch.finfo(density.dtype).eps
    if (density > jam_density).any():
        raise ValueError('Initial density exceeds fixed jam storage')
    peak = free_speed_kmh * wave_speed_kmh * jam_density / (free_speed_kmh + wave_speed_kmh)
    if (capacity > peak * (1 + tol)).any():
        raise ValueError('Capacity exceeds the triangular fundamental-diagram envelope')
    if (dt * torch.maximum(free_speed_kmh, wave_speed_kmh) > lengths_km * (1 + tol)).any():
        raise ValueError('CTM CFL condition violated: use shorter internal time steps')
    sending = torch.minimum(free_speed_kmh * density, capacity)
    receiving = torch.minimum(wave_speed_kmh * (jam_density - density), capacity)
    flux = torch.cat([
        torch.minimum(upstream_demand, receiving[:, 0])[:, None],
        torch.minimum(sending[:, :-1], receiving[:, 1:]),
        torch.minimum(sending[:, -1], downstream_supply)[:, None],
    ], dim=1)
    updated = density + dt / lengths_km * (flux[:, :-1] - flux[:, 1:])
    return updated, flux
