"""Traffic core plus one report-state adapter, revision v0.3.

The frozen v0.2 and original source remain untouched. Relation values are
predictive features, not distances with certified units or physical capacity.
The open-loop state has a direct-MLP control; no clearance labels are invented.
"""
import copy
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from experiments.chronological.unified_incident_adapter import (
    TrafficCore as _TrafficCore, ReportEncoder as _ReportEncoder)


DEFAULT_ARMS = ('ORIGINAL', 'TRAFFIC_CORE', 'H_ONLY', 'R_ONLY', 'BOTH', 'DIRECT')
ARMS = DEFAULT_ARMS + ('TIME', 'GRAPH', 'SEPARATE', 'STACKED')
ADAPTER_CONFIG = dict(state_dim=16, graph_rank=8, max_log_weight=2.,
                      max_log_strength=2., max_prediction_delta=.1)


def arm_spec(arm):
    if arm not in ARMS:
        raise ValueError('Unknown unified v03 arm')
    return dict(mode=arm if arm in ('ORIGINAL', 'TRAFFIC_CORE', 'TIME', 'GRAPH', 'STACKED') else 'BOTH',
                information='H' if arm == 'H_ONLY' else 'R' if arm == 'R_ONLY' else 'RC',
                state_mode='direct' if arm == 'DIRECT' else 'separate' if arm == 'SEPARATE' else 'recurrent')


class TrafficCore(_TrafficCore):
    """Exactly the v02 traffic core, with one-pass pre-normalization access."""
    def prepare_with_summary(self, history):
        flow, u, d, tod, dow = self._prepare(self, history)
        raw = self.embedding(flow)
        encoded = torch.cat((raw[:, :-1], self.latest_norm(raw[:, -1:])), dim=1)
        return encoded, raw, u, d, tod, dow

    def prepare(self, history):
        h, _, u, d, tod, dow = self.prepare_with_summary(history)
        return h, u, d, tod, dow

    def diagnostics(self):
        return dict(super().diagnostics(), schema='unified_incident_adapter_v03')


class ReportEncoder(_ReportEncoder):
    @staticmethod
    def lookup(layer, value):
        # Check before conversion: NaN and fractional categories are UNKNOWN,
        # never silently truncated into a valid category such as zero.
        unknown = layer.num_embeddings - 1
        valid = torch.isfinite(value) & (value == value.trunc()) & (value >= 0) & (value < unknown)
        safe = torch.where(valid, value, torch.full_like(value, unknown)).long()
        return layer(safe)

    def forward(self, report, tod, dow, semantic=True):
        values = report['incident']
        if semantic:
            position, desc, kind = report['position'], values[:, 1], values[:, 2]
        else:
            # Same registered path and shared holiday/clock; only incident
            # position/type/description are removed by the information control.
            position = torch.full_like(report['position'], -1)
            desc = torch.full_like(values[:, 1], -1)
            kind = torch.full_like(values[:, 2], -1)
        parts = [self.lookup(self.position_embedding, position),
                 self.lookup(self.desc_embedding, desc),
                 self.lookup(self.incident_type_embedding, kind),
                 self.lookup(self.holiday_embedding, values[:, 3]), tod, dow]
        return self.incident_fusion(torch.cat(parts, -1))


def _boolean_vector(value, batch, device, name):
    result = torch.as_tensor(value, device=device)
    if result.shape != (batch,):
        raise ValueError(name + ' must be a per-sample boolean/0-1 vector')
    if result.dtype != torch.bool and not bool((torch.isfinite(result) & ((result == 0) | (result == 1))).all()):
        raise ValueError(name + ' must be a per-sample boolean/0-1 vector')
    return result.bool()


def sanitize_report(report, raw_history):
    """Drop unavailable placeholders before arithmetic, preserving UNK identity.

    Legacy callers without availability bits remain supported, but diagnostics
    expose that fallback. Nonfinite *available* relations fail explicitly.
    """
    b, _, n, _ = raw_history.shape
    device, dtype = raw_history.device, raw_history.dtype
    if report is None:
        return None, torch.zeros(b, dtype=torch.bool, device=device), False
    available = _boolean_vector(report.get('available', torch.ones(b, device=device)), b, device, 'available')
    if 'report_present' in report:
        available = available & _boolean_vector(report['report_present'], b, device, 'report_present')
    relation_available = _boolean_vector(report.get('relation_available', torch.ones(b, device=device)),
                                         b, device, 'relation_available')
    available = available & relation_available
    relations = report['distances'].to(device=device, dtype=dtype)
    if relations.shape != (b, n, 3):
        raise ValueError('distances must have shape [batch, nodes, 3]')
    finite = torch.isfinite(relations).all((1, 2))
    if (available & ~finite).any():
        raise ValueError('Available relation contains nonfinite values')
    relations = torch.where(available[:, None, None], relations, torch.zeros_like(relations))
    incident = report['incident'].to(device=device, dtype=dtype)
    position = report['position'].to(device=device, dtype=dtype)
    if incident.shape != (b, 4) or position.shape != (b,):
        raise ValueError('incident/position shape does not match the published contract')
    # Unknown category tokens safely encode missing semantic values. Holiday is
    # retained as shared context, including when report content is switched off.
    incident = torch.where(torch.isfinite(incident), incident, torch.full_like(incident, -1))
    position = torch.where(torch.isfinite(position), position, torch.full_like(position, -1))
    safe = dict(incident=incident, position=position, distances=relations, available=available)
    return safe, available, 'available' in report and 'relation_available' in report


class OrderedTrafficSummary(nn.Module):
    """A short ordered convolution; every history slot uses the same scale."""
    def __init__(self, hidden_dim, state_dim, seq_len):
        super().__init__()
        self.temporal = nn.Conv1d(hidden_dim, state_dim, 3, padding=1)
        self.project = nn.Linear(seq_len * state_dim + 3 * hidden_dim, state_dim)
        self.seq_len = seq_len

    @staticmethod
    def statistics(raw_history):
        return torch.cat((raw_history[:, -1], raw_history[:, -1] - raw_history[:, 0],
                          raw_history.std(1, unbiased=False)), -1)

    def forward(self, raw_history):
        b, t, n, c = raw_history.shape
        ordered = raw_history.permute(0, 2, 3, 1).reshape(b*n, c, t)
        ordered = F.silu(self.temporal(ordered)).reshape(b, n, -1)
        return torch.tanh(self.project(torch.cat((ordered, self.statistics(raw_history)), -1)))


class IncidentGraph(nn.Module):
    """Score explicit road edges, then scatter into native dense graph banks.

    Both source and receiver features include relation values. Redistribution
    changes neighbor choice; an independent signed log-strength changes total
    legal message weight, including rows with one legal neighbor. Neither is a
    measured traffic capacity. No [B,N,N,C] pair features are constructed.
    """
    def __init__(self, state_dim, rank, road_support, max_log_weight=2., max_log_strength=2.):
        super().__init__()
        self.query = nn.ModuleList([nn.Linear(2*state_dim+3, rank, bias=False) for _ in range(2)])
        self.key = nn.ModuleList([nn.Linear(2*state_dim+3, rank, bias=False) for _ in range(2)])
        self.strength = nn.ModuleList([nn.Linear(2*state_dim+3, 1) for _ in range(2)])
        for layer in list(self.query) + list(self.strength):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for direction in range(2):
            receiver, source = road_support[direction].nonzero(as_tuple=True)
            self.register_buffer('receivers_' + str(direction), receiver)
            self.register_buffer('sources_' + str(direction), source)
        self.rank = rank
        self.max_log_weight, self.max_log_strength = max_log_weight, max_log_strength
        self._trace = None

    def forward(self, nodes, state, relations, bases, receivers, gates):
        b, n, _ = nodes.shape
        local = torch.cat((nodes, state[:, None].expand(-1, n, -1), relations), -1)
        corrected, changes, strengths, edge_elements = [], [], [], 0
        for direction, base in enumerate(bases):
            rows = getattr(self, 'receivers_' + str(direction))
            cols = getattr(self, 'sources_' + str(direction))
            edge_elements += b * rows.numel()
            if rows.numel() == 0:
                corrected.append(base)
                continue
            query, key = self.query[direction](local), self.key[direction](local)
            scores = (query[:, rows] * key[:, cols]).sum(-1) / math.sqrt(self.rank)
            legal = base[:, rows, cols]
            weighted = legal * torch.exp(self.max_log_weight * torch.tanh(scores))
            indices = rows[None].expand(b, -1)
            mass = base.new_zeros(b, n).scatter_add(1, indices, legal)
            denominator = base.new_zeros(b, n).scatter_add(1, indices, weighted)
            safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
            redistributed = weighted * (mass / safe_denominator)[:, rows]
            log_strength = self.max_log_strength * torch.tanh(self.strength[direction](local).squeeze(-1))
            candidate = redistributed * torch.exp(log_strength[:, rows])
            mix = receivers.to(base.dtype) * gates
            change = mix[:, rows] * (candidate - legal)
            output = base.clone()
            output[:, rows, cols] = legal + change
            corrected.append(output)
            changes.append(change.detach().abs().sum())
            strengths.append(log_strength.detach().abs().amax())
        self._trace = dict(edge_change_abs_sum=sum(changes, nodes.new_zeros(())),
                           edge_elements=edge_elements,
                           log_strength_abs_max=torch.stack(strengths).amax() if strengths else nodes.new_zeros(()))
        return corrected


class ReportStateAdapter(nn.Module):
    def __init__(self, original, information='RC', state_mode='recurrent', state_dim=16,
                 graph_rank=8, max_log_weight=2., max_log_strength=2., max_prediction_delta=.1):
        super().__init__()
        if information not in ('H', 'R', 'RC') or state_mode not in ('recurrent', 'direct', 'separate'):
            raise ValueError('Invalid information or state mode')
        if (state_dim < 1 or graph_rank < 1 or not 0 < max_prediction_delta <= 1
                or not 0 < max_log_weight <= 3 or not 0 < max_log_strength <= 3):
            raise ValueError('Invalid v03 adapter bounds or dimensions')
        if (not original._model_args['dy_graph'] or len(original._model_args['adjs']) != 2
                or original.seq_len != original.horizon or original.horizon % original._model_args['gap']
                or original.output_dim != 1):
            raise ValueError('v03 requires original two-direction complete gap blocks and scalar output')
        c = original._hidden_dim
        self.encoder = ReportEncoder(original.icsf_module)
        self.summary = OrderedTrafficSummary(c, state_dim, original.seq_len)
        self.pool = nn.Linear(2*state_dim+7, state_dim)
        self.initial = nn.Linear(c+state_dim, state_dim)
        self.transition_input = nn.Linear(c+2*state_dim, state_dim)
        self.transition = nn.GRUCell(state_dim, state_dim)
        self.direct = nn.Sequential(nn.Linear(c+2*state_dim, state_dim), nn.Tanh(), nn.Linear(state_dim, state_dim))
        self.separate_transition = copy.deepcopy(self.transition)
        self.step = nn.Embedding(original.horizon // original._model_args['gap'], state_dim)
        self.blocks, self.gap = self.step.num_embeddings, original._model_args['gap']
        sensor_dim = c if original.icsf_module.use_sensor_info else 0
        self.local_gate = nn.Linear(2*state_dim+3+sensor_dim, 1)
        self.time_interaction = nn.Sequential(nn.Linear(2*state_dim+3+sensor_dim, state_dim), nn.SiLU())
        self.output = nn.Linear(state_dim, self.gap)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        eye = torch.eye(original.node_num, dtype=torch.bool)
        roads = torch.stack([g.detach().cpu() > 0 for g in original._model_args['adjs']]) & ~eye
        self.register_buffer('road_support', roads)
        self.graph = IncidentGraph(state_dim, graph_rank, roads, max_log_weight, max_log_strength)
        self.information, self.state_mode = information, state_mode
        self.max_prediction_delta = max_prediction_delta
        self._trace = None

    def forward(self, raw_history, report, sensors, tod, dow, *, semantic=True, need_time=True, sanitized=None):
        b, _, n, _ = raw_history.shape
        nodes = self.summary(raw_history)
        if self.information == 'H':
            # Do not even validate actual relation/category content in this arm:
            # it is an information control, not a report-masked route.
            shared = nodes.new_full((b, 4), -1)
            if report is not None:
                shared[:, 3] = report['incident'][:, 3].to(shared)
            report = dict(incident=shared, position=nodes.new_full((b,), -1),
                          distances=nodes.new_zeros(b, n, 3), available=torch.ones(b, device=nodes.device))
        safe, available, explicit = sanitize_report(report, raw_history) if sanitized is None else sanitized
        relations = nodes.new_zeros(b, n, 3) if safe is None else safe['distances']
        support = relations.abs().sum(-1) > 0
        if self.information == 'H':
            # Crucially independent of actual report presence and R support.
            relations = torch.zeros_like(relations)
            support = torch.ones_like(support)
            active = torch.ones(b, dtype=torch.bool, device=nodes.device)
        else:
            active = available & support.any(-1)
        if safe is None:
            safe = dict(incident=nodes.new_full((b, 4), -1), position=nodes.new_full((b,), -1), distances=relations)
        semantic = semantic and self.information == 'RC'
        encoding_report = dict(safe)
        if self.information == 'H':
            # Calendar is derived from history; holiday is a shared, separately
            # supplied covariate if present. No incident content enters H.
            encoding_report['position'] = nodes.new_full((b,), -1)
            encoding_report['incident'] = safe['incident'].clone()
            encoding_report['incident'][:, :3] = -1
        e = self.encoder(encoding_report, tod[:, -1, 0], dow[:, -1, 0], semantic=semantic)
        weights = support.to(nodes.dtype) * (1 + relations.abs().sum(-1))
        denominator = weights.sum(1, keepdim=True).clamp_min(1)
        mean = (nodes * weights[..., None]).sum(1) / denominator
        variance = ((nodes-mean[:, None]).square() * weights[..., None]).sum(1) / denominator
        relation_mean = relations.mean(1)
        relation_std = relations.std(1, unbiased=False)
        pooled = torch.tanh(self.pool(torch.cat((mean, variance, relation_mean, relation_std,
                                                 support.to(nodes.dtype).mean(1, keepdim=True)), -1)))
        state = torch.tanh(self.initial(torch.cat((e, pooled), -1)))
        other = state
        states, graph_states = [], []
        for k in range(self.blocks):
            step = self.step.weight[k].expand(b, -1)
            inputs = torch.cat((e, pooled, step), -1)
            if self.state_mode == 'direct':
                state = torch.tanh(self.direct(inputs))
            else:
                value = torch.tanh(self.transition_input(inputs))
                state = self.transition(value, state)
                if self.state_mode == 'separate':
                    other = self.separate_transition(value, other)
            states.append(torch.where(active[:, None], state, torch.zeros_like(state)))
            graph_states.append(torch.where(active[:, None], other if self.state_mode == 'separate' else state,
                                            torch.zeros_like(state)))
        states, graph_states = torch.stack(states, 1), torch.stack(graph_states, 1)
        sensor = self.encoder.sensors(sensors, nodes)
        local = torch.cat((nodes[:, None].expand(-1, self.blocks, -1, -1),
                           states[:, :, None].expand(-1, -1, n, -1),
                           relations[:, None].expand(-1, self.blocks, -1, -1),
                           sensor[:, None].expand(-1, self.blocks, -1, -1)), -1)
        gates = torch.sigmoid(self.local_gate(local)).squeeze(-1)
        graph_gates = gates
        if self.state_mode == 'separate':
            graph_local = torch.cat((local[..., :nodes.shape[-1]],
                                     graph_states[:, :, None].expand(-1, -1, n, -1),
                                     local[..., 2*nodes.shape[-1]:]), -1)
            graph_gates = torch.sigmoid(self.local_gate(graph_local)).squeeze(-1)
        delta = None
        if need_time:
            signed = self.output(self.time_interaction(local))
            change = self.max_prediction_delta * torch.tanh(signed) * gates[..., None]
            change = torch.where(support[:, None, :, None] & active[:, None, None, None], change, torch.zeros_like(change))
            delta = change.permute(0, 1, 3, 2).reshape(b, self.blocks*self.gap, n, 1)
        self._trace = dict(state_abs_sum=states.detach().abs().sum(), state_elements=states.numel(),
                           supported_positions=support.detach().sum(), node_positions=b*n,
                           active_samples=active.detach().sum(), availability_explicit=explicit,
                           output_elements=delta.numel() if delta is not None else 0,
                           output_delta_abs_sum=delta.detach().abs().sum() if delta is not None else nodes.new_zeros(()),
                           output_delta_abs_max=delta.detach().abs().amax() if delta is not None else nodes.new_zeros(()),
                           local_gate_sum=gates.detach().sum(), local_gate_elements=gates.numel(),
                           local_gate_saturated=((gates.detach() < .01) | (gates.detach() > .99)).sum())
        return dict(nodes=nodes, states=states, graph_states=graph_states, gates=gates, graph_gates=graph_gates,
                    support=support, active=active, relations=relations, delta=delta)

    def graph_schedule(self, core, values, bases):
        # One explicit operator envelope; no assumed one-hop-per-time-block law.
        support = values['support']
        union = self.road_support.any(0).to(values['nodes'].dtype)
        receivers = support | (support.to(union.dtype) @ union.T > 0)
        schedules, traces = [None], []
        for k in range(1, self.blocks):
            corrected = self.graph(values['nodes'], values['graph_states'][:, k], values['relations'], bases,
                                   receivers & values['active'][:, None], values['graph_gates'][:, k])
            traces.append(self.graph._trace)
            schedules.append([g for direction in core.dynamic_graph_constructor.multi_order(corrected) for g in direction])
        self.graph._trace = dict(edge_change_abs_sum=sum(t['edge_change_abs_sum'] for t in traces),
                                 edge_elements=sum(t['edge_elements'] for t in traces),
                                 log_strength_abs_max=torch.stack([t['log_strength_abs_max'] for t in traces]).amax())
        return schedules


class UnifiedIncidentIGSTGNN(nn.Module):
    def __init__(self, original, mode='BOTH', information='RC', state_mode='recurrent', **config):
        super().__init__()
        if mode not in ('TIME', 'GRAPH', 'BOTH', 'STACKED'):
            raise ValueError('Unknown unified v03 adapter mode')
        self.mode = mode
        reference = next(original.parameters())
        devices = [reference.device.index] if reference.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            self.adapter = ReportStateAdapter(original, information, state_mode, **config).to(reference.device, reference.dtype)
        if mode == 'STACKED':
            self.original = original
        else:
            self.core = TrafficCore(original)
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon'):
            setattr(self, name, getattr(original, name))
        self._time_enabled = mode in ('TIME', 'BOTH', 'STACKED')
        self._graph_enabled = mode in ('GRAPH', 'BOTH', 'STACKED')
        self._report_enabled = self._semantic_enabled = True
        self._all_reports_off = False

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    @contextmanager
    def intervention(self, *, time=True, graph=True, report=True, semantic=True, all_reports=False):
        if self.mode == 'STACKED' and not report and not all_reports:
            raise ValueError('STACKED report=False is ambiguous; use all_reports=True to disable every report path')
        previous = (self._time_enabled, self._graph_enabled, self._report_enabled,
                    self._semantic_enabled, self._all_reports_off)
        self._time_enabled, self._graph_enabled = previous[0] and time, previous[1] and graph
        self._report_enabled, self._semantic_enabled = previous[2] and report and not all_reports, previous[3] and semantic
        self._all_reports_off = previous[4] or all_reports
        try:
            yield self
        finally:
            (self._time_enabled, self._graph_enabled, self._report_enabled,
             self._semantic_enabled, self._all_reports_off) = previous

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        stacked = self.mode == 'STACKED'
        report = None if self._all_reports_off else incident_data
        if stacked and report is None:
            self.adapter._trace = None
            return self.original(history_data, incident_data=None, sensor_data=sensor_data)
        core = TrafficCore(self.original) if stacked else self.core
        h, raw, u, d, tod, dow = core.prepare_with_summary(history_data)
        enhanced, legacy_context = h, None
        if stacked:
            # The optional redundancy control retains the unmodified legacy
            # embedding contract. Its lack of missing/UNK support is explicit;
            # never send NaN placeholders into an original report path.
            safe, available, _ = sanitize_report(report, raw)
            if not bool(available.all()):
                raise ValueError('STACKED diagnostic requires complete available legacy reports; missing reports are unsupported')
            source = self.original.icsf_module
            for category, layer in ((report['position'], source.position_embedding),
                                    (report['incident'][:, 1], source.desc_embedding),
                                    (report['incident'][:, 2], source.incident_type_embedding),
                                    (report['incident'][:, 3], source.holiday_embedding)):
                valid = torch.isfinite(category) & (category == category.trunc()) & (category >= 0) & (category < layer.num_embeddings)
                if not bool(valid.all()):
                    raise ValueError('STACKED diagnostic requires complete legal legacy categories; UNK is unsupported')
            enhanced, legacy_context = self.original.icsf_module(raw, report, sensor_data, tod[:, -1, 0], dow[:, -1, 0])
        static, dynamic, bases = core.graphs(enhanced, u, d, tod, dow)
        values = None
        sanitized = None
        if self._report_enabled and (report is not None or self.adapter.information == 'H'):
            # Entire unavailable batches avoid state and graph allocations. Mixed
            # batches retain exact per-row fallback through where masks.
            if self.adapter.information == 'H':
                usable = True
            else:
                sanitized = sanitize_report(report, raw)
                safe, available, _ = sanitized
                usable = bool((available & (safe['distances'].abs().sum((1, 2)) > 0)).any())
            if usable:
                values = self.adapter(raw, report, sensor_data, tod, dow,
                                      semantic=self._semantic_enabled, need_time=self._time_enabled, sanitized=sanitized)
        if values is None:
            self.adapter._trace = None
        if values is None or not self._graph_enabled:
            self.adapter.graph._trace = None
        schedule = self.adapter.graph_schedule(core, values, bases) if values is not None and self._graph_enabled else None
        hidden = core.features(enhanced, u, d, tod, dow, static, dynamic, schedule)
        if stacked and legacy_context is not None:
            hidden = self.original.tiid_module(hidden, legacy_context)
        prediction = core.decode(hidden)
        if values is not None and self._time_enabled:
            prediction = prediction + values['delta']
        return prediction

    def diagnostics(self):
        result = dict(schema='unified_incident_adapter_v03', mode=self.mode,
                      information=self.adapter.information, state_mode=self.adapter.state_mode,
                      aggregation_scope='last_forward_batch_only_collect_each_batch_for_dataset_totals',
                      report_entry_count=2 if self.mode == 'STACKED' else 1,
                      native_ICSF_TIID_retained=self.mode == 'STACKED', latest_norm_unconditional=self.mode != 'STACKED',
                      zero_adapter_reference='complete_original' if self.mode == 'STACKED' else 'report_free_traffic_core',
                      forecast_blocks=self.adapter.blocks, persistent_cross_sample_state=False,
                      graph_scope='future_DIF_blocks_2_to_4_first_block_native_high_orders_may_leave_envelope',
                      graph_row_mass_preserved=False, output_delta_space='normalized_flow_after_native_decoder',
                      state_rollout='open_loop_not_observed_clearance_or_traffic_feedback', physics_certified=False,
                      registered_parameters=self.param_num())
        if self.adapter._trace is not None:
            result.update({k: v.detach().cpu().tolist() if torch.is_tensor(v) else v for k, v in self.adapter._trace.items()})
        if self.adapter.graph._trace is not None:
            result.update({k: v.detach().cpu().tolist() if torch.is_tensor(v) else v for k, v in self.adapter.graph._trace.items()})
        return result

    def batch_diagnostics(self):
        """Raw batch sums/counts/maxima; runner must aggregate every batch."""
        return self.diagnostics()


def build_variant(original, arm, **overrides):
    spec = arm_spec(arm)
    if arm == 'ORIGINAL':
        return original
    if arm == 'TRAFFIC_CORE':
        return TrafficCore(original)
    return UnifiedIncidentIGSTGNN(original, **spec, **dict(ADAPTER_CONFIG, **overrides))


def common_traffic_state(model):
    if isinstance(model, TrafficCore):
        return model.state_dict()
    if isinstance(model, UnifiedIncidentIGSTGNN) and model.mode != 'STACKED':
        return model.core.state_dict()
    original = model.original if isinstance(model, UnifiedIncidentIGSTGNN) else model
    return TrafficCore(original).state_dict()
