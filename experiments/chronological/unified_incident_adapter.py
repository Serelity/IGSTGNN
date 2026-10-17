"""Report-free original traffic core with one incident-state adapter (v0.2).

ICSF attention/injection and Gaussian TIID are absent from the unified model.
The latest-slot LayerNorm is always applied, including when no report exists.
Only STACKED, an explicit redundancy control, retains the old report paths.
Graphs use [receiver, source]; states/weights are predictive, not physical flux.
"""
import copy
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from experiments.chronological.original_isp import ISPIGSTGNN, IncidentGraph


ARMS = ('ORIGINAL', 'TRAFFIC_CORE', 'TIME', 'GRAPH', 'BOTH', 'STACKED')
ADAPTER_CONFIG = dict(state_dim=16, graph_rank=8, max_graph_mix=.25, max_state_delta=.1)


class TrafficCore(nn.Module):
    """Original traffic operators and unconditional cutoff normalization only."""
    def __init__(self, original):
        super().__init__()
        for name in ('embedding', 'layers', 'dynamic_graph_constructor', 'out_fc_1', 'out_fc_2',
                     'node_emb_u', 'node_emb_d', 'T_i_D_emb', 'D_i_W_emb'):
            setattr(self, name, getattr(original, name))
        self.latest_norm = original.icsf_module.output_norm
        self._model_args = dict(original._model_args)
        self._prepare = type(original)._prepare_inputs
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon',
                     '_hidden_dim', '_forecast_dim'):
            setattr(self, name, getattr(original, name))

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def prepare(self, history):
        flow, u, d, tod, dow = self._prepare(self, history)
        encoded = self.embedding(flow)
        encoded = torch.cat((encoded[:, :-1], self.latest_norm(encoded[:, -1:])), dim=1)
        return encoded, u, d, tod, dow

    def graphs(self, history, u, d, tod, dow):
        static = [F.softmax(F.relu(torch.mm(u, d.T)), dim=1)] if self._model_args['sta_graph'] else []
        constructor = self.dynamic_graph_constructor
        bases = constructor.normalizer(constructor.mask(constructor.distance_function(history, d, u, tod, dow)))
        dynamic = constructor.st_localization(constructor.multi_order(bases))
        return static, dynamic, bases

    def features(self, history, u, d, tod, dow, static, dynamic, schedule=None):
        residual, dif_outputs, inh_outputs = history, [], []
        for layer in self.layers:
            if schedule is None:
                residual, dif, inh = layer(residual, dynamic, static, u, d, tod, dow)
            else:
                gated = layer.estimation_gate(u, d, tod, dow, residual)
                remaining, dif = ISPIGSTGNN._forecast_dif(layer, (gated, residual), dynamic, static, schedule)
                residual, inh = layer.inh_layer(remaining)
            dif_outputs.append(dif)
            inh_outputs.append(inh)
        return sum(dif_outputs) + sum(inh_outputs)

    def decode(self, hidden):
        prediction = self.out_fc_2(F.relu(self.out_fc_1(F.relu(hidden))))
        prediction = prediction.transpose(1, 2).contiguous().view(prediction.shape[0], prediction.shape[2], -1)
        return prediction.transpose(1, 2).unsqueeze(-1)

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        h, u, d, tod, dow = self.prepare(history_data)
        static, dynamic, _ = self.graphs(h, u, d, tod, dow)
        return self.decode(self.features(h, u, d, tod, dow, static, dynamic))

    def diagnostics(self):
        return dict(schema='unified_incident_adapter_v02', mode='TRAFFIC_CORE',
                    report_read_by_traffic_core=False, native_ICSF_TIID_retained=False,
                    latest_norm_unconditional=True, physics_certified=False)


def embedding_with_unknown(source):
    result = nn.Embedding(source.num_embeddings + 1, source.embedding_dim)
    with torch.no_grad():
        result.weight[:-1].copy_(source.weight.detach().cpu())
        result.weight[-1].copy_(source.weight.detach().cpu().mean(0))
    return result


class ReportEncoder(nn.Module):
    """One report encoder, reusing original valid-category initial tensors.

Unknown rows are explicitly appended, distinct from category zero. Sensor
context keeps the original parser/embeddings, with no extra dropout RNG.
No ICSF attention, distance softmax, injection or TIID is registered here.
"""
    def __init__(self, icsf):
        super().__init__()
        self.position_embedding = embedding_with_unknown(icsf.position_embedding)
        self.desc_embedding = embedding_with_unknown(icsf.desc_embedding)
        self.incident_type_embedding = embedding_with_unknown(icsf.incident_type_embedding)
        self.holiday_embedding = embedding_with_unknown(icsf.holiday_embedding)
        self.incident_fusion = copy.deepcopy(icsf.incident_fusion)
        self.use_sensor_info = icsf.use_sensor_info
        if self.use_sensor_info:
            for name in ('sensor_type_embedding', 'surface_embedding', 'roadway_use_embedding',
                         'road_width_linear', 'speed_limit_linear'):
                setattr(self, name, copy.deepcopy(getattr(icsf, name)))
            self.sensor_fusion = nn.Sequential(*[
                copy.deepcopy(layer) for layer in icsf.sensor_fusion if not isinstance(layer, nn.Dropout)])

    @staticmethod
    def lookup(layer, value):
        indices = value.long()
        unknown = layer.num_embeddings - 1
        indices = torch.where((indices >= 0) & (indices < unknown), indices, torch.full_like(indices, unknown))
        return layer(indices)

    def forward(self, report, tod, dow):
        values = report['incident']
        parts = [self.lookup(self.position_embedding, report['position']),
                 self.lookup(self.desc_embedding, values[:, 1]),
                 self.lookup(self.incident_type_embedding, values[:, 2]),
                 self.lookup(self.holiday_embedding, values[:, 3]), tod, dow]
        return self.incident_fusion(torch.cat(parts, -1))

    def sensors(self, sensor_data, nodes):
        if not self.use_sensor_info or sensor_data is None:
            return nodes.new_zeros((*nodes.shape[:2], 0 if not self.use_sensor_info else self.incident_fusion[-1].out_features))
        parts = [self.sensor_type_embedding(sensor_data['sensor_type'].long().clamp(0, self.sensor_type_embedding.num_embeddings - 1)),
                 self.surface_embedding(sensor_data['surface'].long().clamp(0, self.surface_embedding.num_embeddings - 1)),
                 self.roadway_use_embedding(sensor_data['roadway_use'].long().clamp(0, self.roadway_use_embedding.num_embeddings - 1)),
                 self.road_width_linear(sensor_data['road_width'].float().unsqueeze(-1)),
                 self.speed_limit_linear(sensor_data['speed_limit'].float().unsqueeze(-1))]
        return self.sensor_fusion(torch.cat(parts, -1))


class ReportStateAdapter(nn.Module):
    """Single report entry with shared state and graph/time output heads."""
    def __init__(self, original, state_dim=16, graph_rank=8, max_graph_mix=.25, max_state_delta=.1):
        super().__init__()
        if state_dim < 1 or graph_rank < 1 or not 0 < max_graph_mix <= 1 or not 0 < max_state_delta <= 1:
            raise ValueError('Invalid unified adapter configuration')
        if (not original._model_args['dy_graph'] or len(original._model_args['adjs']) != 2
                or original.seq_len != original.horizon or original.horizon % original._model_args['gap']):
            raise ValueError('Unified adapter requires the original two-direction, complete gap-block contract')
        c, f = original._hidden_dim, original._forecast_dim
        self.encoder = ReportEncoder(original.icsf_module)
        self.summary = nn.Sequential(nn.Linear(3 * c, state_dim), nn.Tanh())
        self.initial = nn.Linear(c + state_dim, state_dim)
        self.transition_input = nn.Linear(c + 2 * state_dim, state_dim)
        self.transition = nn.GRUCell(state_dim, state_dim)
        self.blocks = original.horizon // original._model_args['gap']
        self.step = nn.Embedding(self.blocks, state_dim)
        sensor_dim = c if original.icsf_module.use_sensor_info else 0
        self.local_gate = nn.Linear(2 * state_dim + 3 + sensor_dim, 1)
        self.intensity = nn.Linear(state_dim, 1)
        self.output_state = nn.Linear(state_dim, f, bias=False)
        self.output_node = nn.Linear(state_dim, f, bias=False)
        nn.init.zeros_(self.output_state.weight)
        nn.init.zeros_(self.output_node.weight)
        self.graph = IncidentGraph(state_dim, graph_rank)
        self.max_graph_mix, self.max_state_delta = max_graph_mix, max_state_delta
        self.state_dim, self.graph_rank = state_dim, graph_rank
        eye = torch.eye(original.node_num, dtype=torch.bool)
        roads = torch.stack([g.detach().cpu() > 0 for g in original._model_args['adjs']]) & ~eye
        self.register_buffer('road_support', roads)
        self._trace = None

    def forward(self, history, report, sensors, tod, dow):
        b, _, n, _ = history.shape
        nodes = self.summary(torch.cat((history[:, -1], history.mean(1), history[:, -1] - history[:, 0]), -1))
        available = report.get('available', torch.ones(b, dtype=torch.bool, device=history.device))
        available = torch.as_tensor(available, device=history.device)
        if available.shape != (b,) or not torch.isfinite(available).all() or not ((available == 0) | (available == 1)).all():
            raise ValueError('available must be a per-sample boolean/0-1 vector')
        support = (report['distances'].abs().sum(-1) > 0) & available.bool().unsqueeze(-1)
        active = support.any(-1).to(history.dtype).unsqueeze(-1)
        e = self.encoder(report, tod[:, -1, 0], dow[:, -1, 0])
        pooled = (nodes * support.unsqueeze(-1)).sum(1) / support.sum(1, keepdim=True).clamp_min(1)
        state = torch.tanh(self.initial(torch.cat((e, pooled), -1))) * active
        states = []
        for k in range(self.blocks):
            position = self.step.weight[k].expand(b, -1)
            value = torch.tanh(self.transition_input(torch.cat((e, pooled, position), -1)))
            state = self.transition(value, state) * active
            states.append(state)
        states = torch.stack(states, 1)
        sensor_context = self.encoder.sensors(sensors, nodes)
        # Only [B,K,N,C] local features, never receiver-source feature pairs.
        context = torch.cat((nodes, report['distances'].to(history.dtype), sensor_context), -1)
        local = torch.cat((context[:, None].expand(-1, self.blocks, -1, -1),
                           states[:, :, None].expand(-1, -1, n, -1)), -1)
        gates = torch.sigmoid(self.local_gate(local)).squeeze(-1)
        intensity = torch.sigmoid(self.intensity(states)).squeeze(-1) * active
        signed = self.output_state(states)[:, :, None] + self.output_node(nodes)[:, None]
        delta = self.max_state_delta * torch.tanh(signed) * gates.unsqueeze(-1) * support[:, None, :, None]
        self._trace = dict(state_abs_mean=states.detach().abs().mean((0, 2)),
                           intensity_mean=intensity.detach().mean(0),
                           direct_support=support.detach().sum(-1), delta_abs_max=delta.detach().abs().amax())
        return nodes, states, intensity, gates, support, delta

    def graph_schedule(self, core, values, bases):
        nodes, states, intensity, gates, support, _ = values
        reachable, schedules = support, [None]
        union = self.road_support.any(0).to(nodes.dtype)
        for k in range(1, self.blocks):
            reachable = reachable | (reachable.to(nodes.dtype) @ union.T > 0)
            corrected = self.graph(nodes, states[:, k], bases, self.road_support,
                self.max_graph_mix * intensity[:, k], reachable * gates[:, k])
            schedules.append([g for direction in core.dynamic_graph_constructor.multi_order(corrected) for g in direction])
        return schedules


class UnifiedIncidentIGSTGNN(nn.Module):
    def __init__(self, original, mode='BOTH', **config):
        super().__init__()
        if mode not in ('TIME', 'GRAPH', 'BOTH', 'STACKED'):
            raise ValueError('Unknown unified adapter mode')
        self.mode = mode
        reference = next(original.parameters())
        devices = [reference.device.index] if reference.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            self.adapter = ReportStateAdapter(original, **config).to(reference.device, reference.dtype)
        if mode == 'STACKED':
            self.original = original
        else:
            self.core = TrafficCore(original)
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon'):
            setattr(self, name, getattr(original, name))
        self._time_enabled = mode in ('TIME', 'BOTH', 'STACKED')
        self._graph_enabled = mode in ('GRAPH', 'BOTH', 'STACKED')
        self._report_enabled = True

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    @contextmanager
    def intervention(self, *, time=True, graph=True, report=True):
        previous = self._time_enabled, self._graph_enabled, self._report_enabled
        self._time_enabled, self._graph_enabled = previous[0] and time, previous[1] and graph
        self._report_enabled = previous[2] and report
        try:
            yield self
        finally:
            self._time_enabled, self._graph_enabled, self._report_enabled = previous

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        stacked = self.mode == 'STACKED'
        if stacked and incident_data is None:
            self.adapter._trace = None
            return self.original(history_data, incident_data=None, sensor_data=sensor_data)
        core = TrafficCore(self.original) if stacked else self.core
        h, u, d, tod, dow = core.prepare(history_data)
        enhanced, legacy_context = h, None
        if stacked and incident_data is not None:
            raw = self.original.embedding(self.original._prepare_inputs(history_data)[0])
            enhanced, legacy_context = self.original.icsf_module(raw, incident_data, sensor_data,
                                                               tod[:, -1, 0], dow[:, -1, 0])
        static, dynamic, bases = core.graphs(enhanced, u, d, tod, dow)
        values = None
        if incident_data is not None and self._report_enabled:
            values = self.adapter(h, incident_data, sensor_data, tod, dow)
        else:
            self.adapter._trace = None
        schedule = self.adapter.graph_schedule(core, values, bases) if values is not None and self._graph_enabled else None
        hidden = core.features(enhanced, u, d, tod, dow, static, dynamic, schedule)
        if stacked and legacy_context is not None:
            hidden = self.original.tiid_module(hidden, legacy_context)
        if values is not None and self._time_enabled:
            hidden = hidden + values[-1]
        return core.decode(hidden)

    def diagnostics(self):
        result = dict(schema='unified_incident_adapter_v02', mode=self.mode,
            aggregation_scope='last_forward_batch_only', report_entry_count=2 if self.mode == 'STACKED' else 1,
            native_ICSF_TIID_retained=self.mode == 'STACKED', latest_norm_unconditional=self.mode != 'STACKED',
            zero_adapter_reference='complete_original' if self.mode == 'STACKED' else 'report_free_traffic_core',
            forecast_blocks=self.adapter.blocks, state_time_unit='gap_sized_latent_block',
            physics_certified=False, persistent_cross_sample_state=False,
            graph_scope='future_DIF_blocks_2_to_4', registered_parameters=self.param_num())
        if self.adapter._trace is not None:
            result.update({k: v.cpu().tolist() for k, v in self.adapter._trace.items()})
        return result


def build_variant(original, arm):
    if arm not in ARMS:
        raise ValueError('Unknown unified experiment arm')
    if arm == 'ORIGINAL':
        return original
    if arm == 'TRAFFIC_CORE':
        return TrafficCore(original)
    return UnifiedIncidentIGSTGNN(original, mode=arm, **ADAPTER_CONFIG)


def common_traffic_state(model):
    if isinstance(model, TrafficCore):
        return model.state_dict()
    if isinstance(model, UnifiedIncidentIGSTGNN) and model.mode != 'STACKED':
        return model.core.state_dict()
    original = model.original if isinstance(model, UnifiedIncidentIGSTGNN) else model
    return TrafficCore(original).state_dict()
