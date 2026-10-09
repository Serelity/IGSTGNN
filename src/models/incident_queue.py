"""H1 physical-state auxiliary branch and an unconstrained recurrence control.

The caller must resolve bottlenecks, units, history masks and target windows via
a validated data contract. No detector is automatically treated as a queue.
These models preserve the native IGSTGNN file and all existing checkpoint keys.
"""

import torch
from torch import nn
from torch.nn import functional as F

from src.models.igstgnn import IGSTGNN
from src.models.traffic_physics import point_queue_step, positive_scalar


class IncidentQueueBranch(nn.Module):
    def __init__(self, capacities, queue_scales, readout_weights, *,
                 mode='queue', hidden=8, forecast_dim=256,
                 step_minutes=1, gap_minutes=10, interval_minutes=5, horizon=12):
        super().__init__()
        if mode not in ('queue', 'recurrent'):
            raise ValueError('Expected queue or recurrent transition')
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError('hidden must be a positive integer')
        step = positive_scalar(step_minutes, 'step_minutes')
        interval = positive_scalar(interval_minutes, 'interval_minutes')
        if gap_minutes < 0 or horizon != 12:
            raise ValueError('Require nonnegative gap and 12 horizons')
        values = (gap_minutes / step, interval / step)
        if any(abs(x - round(x)) > 1e-9 for x in values):
            raise ValueError('Internal steps must exactly tile gap and observation intervals')
        self.gap_steps, self.bin_steps = map(lambda x: int(round(x)), values)
        self.steps = self.gap_steps + horizon * self.bin_steps
        self.dt_hours, self.mode, self.horizon = step / 60, mode, horizon
        cap = torch.as_tensor(capacities, dtype=torch.float32)
        scale = torch.as_tensor(queue_scales, dtype=torch.float32)
        weights = torch.as_tensor(readout_weights, dtype=torch.float32)
        if (cap.ndim != 1 or not len(cap) or scale.shape != cap.shape
                or not torch.isfinite(cap).all() or not torch.isfinite(scale).all()
                or (cap <= 0).any() or (scale <= 0).any()):
            raise ValueError('Positive finite bottleneck capacities and queue scales required')
        if (weights.ndim != 2 or weights.shape[0] != len(cap) or not torch.isfinite(weights).all()
                or (weights < 0).any() or not torch.equal(weights.sum(1), torch.ones_like(cap))
                or (weights.gt(0).sum(0) > 1).any()):
            raise ValueError('Use normalized nonoverlapping bottleneck-to-node readouts')
        self.register_buffer('base_capacity', cap)
        self.register_buffer('queue_scale', scale)
        self.register_buffer('readout_weights', weights)
        # Both arms share exactly these encoders, constrained forcing and readout.
        self.history_encoder = nn.GRU(4, hidden, batch_first=True)
        self.traffic_head = nn.Linear(hidden, 3)
        self.event_head = nn.Sequential(nn.Linear(hidden + 4, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.projection = nn.Linear(4, forecast_dim, bias=False)
        nn.init.zeros_(self.projection.weight)
        # The reference has extra recurrent parameters, explicitly reported.
        self.recurrence = nn.GRUCell(3, 4) if mode == 'recurrent' else None

    def forward(self, physical):
        history, valid = physical['history_rates'], physical['history_valid']
        event, support = physical['event_features'], physical['report_support']
        b, steps, queues, channels = history.shape
        if (steps != 12 or channels != 2 or queues != self.base_capacity.numel()
                or valid.shape != history.shape or valid.dtype != torch.bool
                or event.shape != (b, queues, 4) or support.shape != (b, queues)
                or support.dtype != torch.bool):
            raise ValueError('Incorrect queue history, masks or report dimensions')
        if (not torch.isfinite(history).all() or (history < 0).any()
                or not torch.isfinite(event).all()):
            raise ValueError('Queue inputs must be finite; rate placeholders must be nonnegative')
        cap, scale = self.base_capacity, self.queue_scale
        rates = torch.where(valid, history, 0) / cap[None, None, :, None]
        encoded = torch.cat([torch.asinh(rates), valid.to(history.dtype)], -1)
        encoded = encoded.permute(0, 2, 1, 3).reshape(b * queues, 12, 4)
        _, hidden = self.history_encoder(encoded)
        hidden = hidden[0].reshape(b, queues, -1)
        traffic = self.traffic_head(hidden)
        q0 = torch.sigmoid(traffic[..., 0]) * scale
        arrival_base = 3 * torch.sigmoid(traffic[..., 1]) * cap
        slope = .5 * torch.tanh(traffic[..., 2])
        event_state = self.event_head(torch.cat([hidden, event], -1))
        amplitude = .95 * torch.sigmoid(event_state[..., 0]) * support
        recovery_minutes = 5 + 175 * torch.sigmoid(event_state[..., 1])
        queue = q0
        state = torch.stack([q0 / scale, torch.zeros_like(q0), torch.ones_like(q0), arrival_base / cap], -1)
        features, arrivals, capacities, departures, queue_states = [], [], [], [], [q0]
        for k in range(self.steps):
            elapsed = (k + .5) * self.dt_hours * 60
            arrival = arrival_base * (1 + slope * min(elapsed / 60, 1))
            capacity = cap * (1 - amplitude * torch.exp(-elapsed / recovery_minutes))
            if self.mode == 'queue':
                queue, departure = point_queue_step(queue, arrival, capacity, self.dt_hours, validate=False)
                state = torch.stack([queue / scale, departure / cap, capacity / cap, arrival / cap], -1)
                departures.append(departure)
                queue_states.append(queue)
            else:
                forcing = torch.stack([arrival / cap, capacity / cap, torch.full_like(capacity, self.dt_hours)], -1)
                state = self.recurrence(forcing.reshape(b * queues, 3), state.reshape(b * queues, 4)).reshape(b, queues, 4)
            features.append(state)
            arrivals.append(arrival)
            capacities.append(capacity)
        # Average hidden physical features on target intervals, after the gap.
        features = torch.stack(features, 1)[:, self.gap_steps:]
        features = features.reshape(b, self.horizon, self.bin_steps, queues, 4).mean(2)
        # A report-supported candidate is not an observed causal impact label.
        features = features * support[:, None, :, None]
        delta_local = self.projection(features)
        delta = torch.einsum('bhqf,qn->bhnf', delta_local, self.readout_weights)
        trace = {'initial_queue': q0, 'arrival': torch.stack(arrivals, 1),
                 'capacity': torch.stack(capacities, 1), 'features': features,
                 'capacity_loss_amplitude': amplitude, 'recovery_minutes': recovery_minutes}
        if self.mode == 'queue':
            trace.update(queue=torch.stack(queue_states, 1), departure=torch.stack(departures, 1))
        return delta, trace


class QueueAugmentedIGSTGNN(IGSTGNN):
    """Native architecture plus an explicit, scoped TIID-to-head extension.

    forward requires X-only physical inputs. Labels are not routed to the
    branch. The short-lived context is removed even when the backbone raises.
    A module instance cannot serve concurrent/reentrant forwards.
    """
    def __init__(self, model_args, queue_config, **args):
        if (model_args.get('incident_schema') != 'report_location_v1'
                or model_args.get('incident_routing', 'none') != 'none'
                or model_args.get('time_response', 'fixed') != 'fixed'):
            raise ValueError('H1 starts from complete fixed, without P2 or other time-response candidates')
        super().__init__(model_args, **args)
        self.queue_branch = IncidentQueueBranch(**queue_config, forecast_dim=self._forecast_dim)
        if self.queue_branch.readout_weights.shape[1] != self.node_num:
            raise ValueError('Queue readout and backbone node axis disagree')

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None, *, physical_inputs=None):
        if physical_inputs is None or incident_data is None:
            raise ValueError('H1 requires explicit history-only physical inputs and report context')
        if hasattr(self, '_queue_delta'):
            raise RuntimeError('Reentrant H1 forward is unsupported')
        delta, trace = self.queue_branch(physical_inputs)
        self._queue_delta = delta
        try:
            output = super().forward(history_data, label=label, incident_data=incident_data, sensor_data=sensor_data)
        finally:
            del self._queue_delta
        self.last_queue_trace = {key: value.detach() for key, value in trace.items()}
        return output

    def _decode_forecast(self, forecast_hidden, incident_inputs):
        # Mirrors the frozen native decoder; only the marked addition is new.
        gap = self._model_args['gap']
        step_hidden = forecast_hidden.repeat_interleave(gap, dim=1)[:, :self.horizon]
        if incident_inputs is not None:
            step_hidden = self.tiid_module(
                step_hidden, incident_key=incident_inputs['incident_key'],
                sensor_features=incident_inputs['sensor_features'],
                distances=incident_inputs['distances'],
                history_state=incident_inputs.get('history_state'),
                report_age_minutes=incident_inputs.get('report_age_minutes'))
        if step_hidden.shape != self._queue_delta.shape:
            raise ValueError('Queue delta and native TIID output have different axes')
        step_hidden = step_hidden + self._queue_delta
        forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(step_hidden))))
        channels = torch.arange(self.horizon, device=forecast.device) % gap
        channels = channels.view(1, -1, 1, 1).expand(forecast.shape[0], -1, self.node_num, 1)
        return forecast.gather(-1, channels)
