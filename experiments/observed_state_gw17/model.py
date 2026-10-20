"""Causal observation updates for one shared regional/local transition.

Both arms instantiate exactly the same trainable modules. They differ solely in
the scalar observation supplied after a history prediction: observed change, or
observed change minus the model's predicted change. Incident reports enter only
the future rollout, after the complete traffic history has been assimilated.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from experiments.position_state_gw16.geometry import RegionGeometry, pool_contrast, lift_contrast
from experiments.position_state_gw16.model import (
    GraphLinear, PositionEncoder, IncidentEncoder, PositionStateExchange, parameter_count)


ARMS = ('OBSERVATION_CONTROL', 'INNOVATION_STATE')
SENSOR_KEYS = ('sensor_type', 'surface', 'roadway_use')


def calendar_indices(values):
    """Decode the same normalized five-minute TOD and weekday channels as GW16."""
    tod = (values[..., 1]*288).round().long().clamp(0, 287)
    dow = (values[..., 2]*7).round().long().clamp(0, 6)
    return tod, dow


def advance_calendar(tod, dow, steps):
    """Advance known clock features, including midnight and weekday rollover."""
    total = tod+steps
    return total.remainder(288), (dow+torch.div(total, 288, rounding_mode='floor')).remainder(7)


def _step_logits(initial, count):
    initial = float(initial)
    if not 0 < initial < 1:
        raise ValueError('State update initial strength must be strictly between zero and one')
    return nn.Parameter(torch.full((count,), math.log(initial/(1-initial))))


class InitialStateEncoder(nn.Module):
    """Static/calendar encoding and a first-observation-only state initializer."""
    def __init__(self, c, sensors, sizes):
        super().__init__()
        d = c['hidden']
        self.flow = nn.Linear(1, d)
        self.node = nn.Embedding(c['nodes'], d)
        self.tod = nn.Embedding(288, d)
        self.dow = nn.Embedding(7, d)
        self.sensor_embeddings = nn.ModuleList(nn.Embedding(int(size), d) for size in sizes)
        for key in SENSOR_KEYS:
            self.register_buffer(key, torch.as_tensor(sensors[key], dtype=torch.long))
        self.register_buffer('sensor_numeric', torch.stack([
            torch.as_tensor(sensors[key], dtype=torch.float32)
            for key in ('road_width', 'speed_limit')], -1)/100.)
        self.sensor_numbers = nn.Linear(2, d)
        self.static_norm = nn.LayerNorm(d)
        self.condition_norm = nn.LayerNorm(d)
        self.initial_norm = nn.LayerNorm(d)
        self.graph_layers = nn.ModuleList(GraphLinear(d) for _ in range(c['spatial_layers']))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(c['spatial_layers']))
        self.dropout = nn.Dropout(c['dropout'])

    def static(self, position):
        sensor = self.sensor_numbers(self.sensor_numeric)
        for embedding, key in zip(self.sensor_embeddings, SENSOR_KEYS):
            sensor = sensor+embedding(getattr(self, key))
        return self.static_norm((position+self.node.weight+sensor)/math.sqrt(3.))

    def clock(self, tod, dow):
        return (self.tod(tod)+self.dow(dow))/math.sqrt(2.)

    def condition(self, static, tod, dow, vector=None, relation=None):
        value = static[None]+self.clock(tod, dow)
        if vector is not None:
            value = value+vector[:, None]+relation
        # With no report the future context is exactly the history context at
        # the same clock; introducing the optional event never rescales it.
        return self.condition_norm(value/math.sqrt(2.))

    def forward(self, first, static, forward, backward):
        tod, dow = calendar_indices(first)
        hidden = self.initial_norm((self.flow(first[..., :1])+static[None]
                                    +self.clock(tod, dow))/math.sqrt(3.))
        for layer, norm in zip(self.graph_layers, self.norms):
            hidden = norm(hidden+self.dropout(F.gelu(layer(hidden, forward, backward))))
        return hidden


class CoupledTransition(nn.Module):
    """One synchronous transition, shared by all historical and future steps."""
    def __init__(self, c):
        super().__init__()
        d, width = c['hidden'], c.get('update_hidden', 3*c['hidden'])
        self.coarse_norm = nn.LayerNorm(d)
        self.local_norm = nn.LayerNorm(d)
        self.coarse_graph = GraphLinear(d)
        self.local_graph = GraphLinear(d)
        self.coarse_update = nn.Sequential(nn.Linear(2*d, width), nn.GELU(), nn.Linear(width, d))
        self.local_update = nn.Sequential(nn.Linear(2*d, width), nn.GELU(), nn.Linear(width, d))
        self.exchange = PositionStateExchange(d, c.get('exchange_rank', 8))
        self.pool_map = nn.Linear(d, d, bias=False)
        self.lift_map = nn.Linear(d, d, bias=False)
        self.source_gate = nn.Linear(d, 1)
        self.source_projection = nn.Linear(d, d, bias=False)
        self.coarse_readout = nn.Linear(d, 1, bias=False)
        self.local_readout = nn.Linear(d, 1, bias=False)
        # Identical exact persistence at stage zero, without detaching states.
        nn.init.zeros_(self.coarse_readout.weight)
        nn.init.zeros_(self.local_readout.weight)
        self.step_logits = _step_logits(c.get('step_initial', .1), 2)
        self.dropout = nn.Dropout(c['dropout'])

    def incident_source(self, condition, vector, support):
        return (support[..., None].to(condition.dtype)*torch.sigmoid(self.source_gate(condition))
                *self.source_projection(vector)[:, None])

    def forward(self, coarse, local, condition, source, position, geometry, graphs):
        forward, backward, regional, regional_backward = graphs
        cv, lv = self.coarse_norm(coarse), self.local_norm(local)
        weights = self.exchange(position, condition, geometry.lift(cv), lv, geometry)
        up = self.pool_map(pool_contrast(geometry, local, weights[..., :1]))
        down = self.lift_map(lift_contrast(geometry, coarse, weights[..., 1:2]))
        coarse_delta = self.coarse_update(torch.cat([
            self.coarse_graph(cv, regional, regional_backward), geometry.pool(condition)], -1))
        local_delta = self.local_update(torch.cat([
            self.local_graph(lv, forward, backward), condition], -1))
        step = self.step_logits.sigmoid()
        # Both right-hand sides read only the old C/L state.
        next_coarse = coarse+step[0]*(self.dropout(coarse_delta)+up+geometry.pool(source))
        next_local = geometry.center(local+step[1]*(self.dropout(local_delta)+down+geometry.center(source)))
        return next_coarse, next_local

    def read_delta(self, coarse, local, previous_coarse, previous_local, geometry):
        alpha = self.coarse_readout(coarse-previous_coarse)
        beta = geometry.center(self.local_readout(local-previous_local))
        return alpha, beta, geometry.lift(alpha)+beta


class ObservationCorrection(nn.Module):
    """P/Q observation residuals correct C/L with position x current-state gates."""
    def __init__(self, c):
        super().__init__()
        d = c['hidden']
        self.coarse_norm = nn.LayerNorm(d)
        self.local_norm = nn.LayerNorm(d)
        self.gate = PositionStateExchange(d, c.get('exchange_rank', 8))
        self.coarse_map = nn.Linear(1, d, bias=False)
        self.local_map = nn.Linear(1, d, bias=False)
        self.step_logits = _step_logits(c.get('observation_step_initial', .1), 2)

    def forward(self, coarse, local, residual, condition, position, geometry):
        weights = self.gate(position, condition, geometry.lift(self.coarse_norm(coarse)),
                            self.local_norm(local), geometry, centered=False)
        coarse_delta = self.coarse_map(geometry.pool(residual))*geometry.pool(weights[..., :1])
        local_delta = self.local_map(geometry.center(residual))*weights[..., 1:2]
        step = self.step_logits.sigmoid()
        return coarse+step[0]*coarse_delta, geometry.center(local+step[1]*local_delta)


class ObservedStateModel(nn.Module):
    def __init__(self, data, c, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown GW17 arm: '+str(arm))
        self.config, self.arm, self.stage = dict(c), arm, 'joint'
        self.geometry = RegionGeometry(data.assignment)
        self.register_buffer('mean', torch.tensor(float(data.mean)))
        self.register_buffer('std', torch.tensor(float(data.std)))
        for key, value in zip(('forward_graph', 'backward_graph', 'regional_graph'), data.graphs):
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))
        reverse = self.regional_graph.T
        self.register_buffer('regional_backward', reverse/reverse.sum(-1, keepdim=True).clamp_min(1e-12))
        # No arm-dependent initialization or inactive capacity padding.
        self.position = PositionEncoder(data.position_features, data.position_categories,
                                        data.position_sizes, c['hidden'])
        self.encoder = InitialStateEncoder(c, data.sensors, data.sensor_sizes)
        self.incident = IncidentEncoder(c['hidden'])
        self.core = CoupledTransition(c)
        self.correction = ObservationCorrection(c)
        active = parameter_count(self.core)+parameter_count(self.correction)
        self.capacity = dict(target_head_parameters=active, actual_head_parameters=active,
            match_fraction=0., internal_width=c.get('update_hidden', 3*c['hidden']),
            matching='identical_all_modules_observation_residual_rule_only',
            actual_parameters=parameter_count(self), history_future_transition_shared=True)

    def set_stage(self, stage):
        if stage != 'joint':
            raise ValueError('GW17 only supports joint training')
        self.stage = stage

    @staticmethod
    def _rms(coarse, local):
        return torch.stack([coarse.detach().square().mean().sqrt(), local.detach().square().mean().sqrt()])

    def forward(self, history, event):
        c, geometry = self.config, self.geometry
        if history.ndim != 4 or tuple(history.shape[1:]) != (c['history'], c['nodes'], 3):
            raise ValueError('Expected history [batch,history,nodes,flow_TOD_DOW]')
        if c['history'] < 2:
            raise ValueError('GW17 requires at least two historical observations')
        position = self.position()
        static = self.encoder.static(position)
        graphs = self.forward_graph, self.backward_graph, self.regional_graph, self.regional_backward
        hidden = self.encoder(history[:, 0], static, self.forward_graph, self.backward_graph)
        coarse, local = geometry.pool(hidden), geometry.center(hidden)
        tod, dow = calendar_indices(history)
        history_prediction, history_norms, innovation_rms, history_valid = [], [], [], []
        for t in range(1, c['history']):
            # The next clock is known; the next observed flow is read only below.
            condition = self.encoder.condition(static, tod[:, t], dow[:, t])
            source = torch.zeros_like(local)
            next_coarse, next_local = self.core(coarse, local, condition, source,
                                               position, geometry, graphs)
            _, _, predicted_delta = self.core.read_delta(next_coarse, next_local, coarse, local, geometry)
            history_prediction.append(self.mean+self.std*(history[:, t-1, :, :1]+predicted_delta))
            observed_delta = history[:, t, :, :1]-history[:, t-1, :, :1]
            innovation = observed_delta-predicted_delta
            residual = observed_delta if self.arm == 'OBSERVATION_CONTROL' else innovation
            # A missing/nonpositive slot is not an observed traffic change. The
            # same pair-validity mask applies in both arms, after prediction.
            previous_raw = self.mean+self.std*history[:, t-1, :, :1]
            current_raw = self.mean+self.std*history[:, t, :, :1]
            minimum = c.get('target_minimum', .001)
            valid = (torch.isfinite(previous_raw) & torch.isfinite(current_raw)
                     & (previous_raw.double() > float(minimum))
                     & (current_raw.double() > float(minimum)))
            residual = torch.where(valid, residual, torch.zeros_like(residual))
            coarse, local = self.correction(next_coarse, next_local, residual,
                                             condition, position, geometry)
            history_norms.append(self._rms(coarse, local))
            innovation_rms.append(torch.where(valid, innovation.detach(),
                                              torch.zeros_like(innovation)).square().mean().sqrt())
            history_valid.append(valid)

        # Reports are not accessed until all history predictions and updates end.
        if c.get('event_ablation', 'none') == 'no_report':
            event = dict(event, report_present=torch.zeros_like(event['report_present']))
        vector, relation, support, available = self.incident(event)
        c0, l0 = coarse, local
        alpha, beta, norms = [], [], []
        for h in range(c['horizon']):
            future_tod, future_dow = advance_calendar(tod[:, -1], dow[:, -1], h+1)
            condition = self.encoder.condition(static, future_tod, future_dow, vector, relation)
            source = self.core.incident_source(condition, vector, support)
            coarse, local = self.core(coarse, local, condition, source, position, geometry, graphs)
            a, b, _ = self.core.read_delta(coarse, local, c0, l0, geometry)
            alpha.append(a)
            beta.append(b)
            norms.append(self._rms(coarse, local))
        alpha, beta = torch.stack(alpha, 1), torch.stack(beta, 1)
        reference = (self.mean+self.std*history[:, -1:, :, :1]).expand(-1, c['horizon'], -1, -1)
        prediction = reference+self.std*(geometry.lift(alpha)+beta)
        return dict(prediction=prediction, reference=reference,
            regional=geometry.pool(reference)+self.std*alpha, alpha=alpha, beta=beta,
            state_rms=torch.stack(norms), support=support, available=available,
            history_prediction=torch.stack(history_prediction, 1),
            history_target=(self.mean+self.std*history[:, 1:, :, :1]).detach(),
            history_valid=torch.stack(history_valid, 1),
            history_state_rms=torch.stack(history_norms), innovation_rms=torch.stack(innovation_rms))


def build_model(data, config, arm, device):
    return ObservedStateModel(data, config, arm).to(device)
