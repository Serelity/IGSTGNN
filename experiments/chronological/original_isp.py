"""ISP-Lite around the hash-checked original IGSTGNN, without physical claims.

History is encoded without report content. A cutoff report initializes a local
latent state, which evolves autonomously over the original gap-sized forecast
blocks. Its residual augments TIID; optional incident-conditioned graphs change
only the future DIF calls. All original historical/backcast/INH paths remain.

Every adjacency uses [receiver, source]. Learned states, intensities and edges
are predictive representations, not identified incident effects or vehicle flux.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = ('STATE', 'STATE_PROPAGATION')


def supported_reweight(base, scores, road_support, alpha, receivers):
    """Convex, row-mass-preserving correction on explicit first-order road edges.

Inherited off-road epsilon edges stay exactly unchanged. Empty legal rows fall
back to the base. The high-order banks produced later need not have unit sums.
Zero scores give bitwise-identical first-order matrices, without a dead gate.
"""
    legal = base * road_support.to(base.dtype)
    mass = legal.sum(-1, keepdim=True)
    weighted = legal * torch.exp(.5 * torch.tanh(scores))
    denominator = weighted.sum(-1, keepdim=True)
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    candidate = weighted * (mass / safe_denominator)
    difference = candidate - legal
    mix = alpha.reshape(-1, 1, 1) * receivers.to(base.dtype).unsqueeze(-1)
    return base + mix * difference


class IncidentState(nn.Module):
    """Traffic-only historical encoder, cutoff injection, then autonomous rollout."""
    def __init__(self, hidden_dim, state_dim, forecast_dim, blocks, max_delta):
        super().__init__()
        self.history = nn.GRU(hidden_dim, state_dim, batch_first=True)
        self.initial = nn.Linear(hidden_dim + state_dim, state_dim)
        self.step = nn.Embedding(blocks, state_dim)
        self.transition_input = nn.Linear(hidden_dim + 2 * state_dim, state_dim)
        self.transition = nn.GRUCell(state_dim, state_dim)
        self.intensity = nn.Linear(state_dim, 1)
        self.output = nn.Linear(state_dim, forecast_dim, bias=False)
        nn.init.zeros_(self.output.weight)
        self.blocks = blocks
        self.max_delta = max_delta

    def forward(self, history, report, support):
        b, t, n, c = history.shape
        # No report is fed to historical time steps; state starts fresh per sample.
        sequence = history.permute(0, 2, 1, 3).reshape(b * n, t, c)
        _, last = self.history(sequence)
        nodes = last[0].reshape(b, n, -1)
        weights = support.to(history.dtype)
        pooled = (nodes * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        present = support.any(-1).to(history.dtype).unsqueeze(-1)
        state = torch.tanh(self.initial(torch.cat((report, pooled), -1))) * present
        states = []
        for k in range(self.blocks):
            step = self.step.weight[k].unsqueeze(0).expand(b, -1)
            transition_input = torch.tanh(self.transition_input(torch.cat((report, pooled, step), -1)))
            state = self.transition(transition_input, state) * present
            states.append(state)
        states = torch.stack(states, 1)
        intensity = torch.sigmoid(self.intensity(states)).squeeze(-1) * present
        delta = self.max_delta * torch.tanh(self.output(states))
        delta = delta.unsqueeze(2) * weights[:, None, :, None]
        return nodes, states, intensity, delta


class IncidentGraph(nn.Module):
    """Low-rank receiver/source scoring; no [B,N,N,hidden] edge tensor."""
    def __init__(self, state_dim, rank, directions=2):
        super().__init__()
        self.query_node = nn.ModuleList([nn.Linear(state_dim, rank, bias=False) for _ in range(directions)])
        self.query_state = nn.ModuleList([nn.Linear(state_dim, rank, bias=False) for _ in range(directions)])
        self.key_node = nn.ModuleList([nn.Linear(state_dim, rank, bias=False) for _ in range(directions)])
        self.key_state = nn.ModuleList([nn.Linear(state_dim, rank, bias=False) for _ in range(directions)])
        for layer in list(self.query_node) + list(self.query_state):
            nn.init.zeros_(layer.weight)
        self.rank = rank

    def forward(self, nodes, state, base, road_support, alpha, receivers):
        result = []
        for direction, matrix in enumerate(base):
            query = self.query_node[direction](nodes) + self.query_state[direction](state).unsqueeze(1)
            key = self.key_node[direction](nodes) + self.key_state[direction](state).unsqueeze(1)
            scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.rank)
            result.append(supported_reweight(matrix, scores, road_support[direction], alpha, receivers))
        return result


class ISPIGSTGNN(nn.Module):
    """Original + state, or original + state + forecast graph evolution.

Both candidate modes register identical new tensors, but STATE does not use the
graph scorer. This does not imply equal active capacity. Original parameters
are neither reinitialized nor frozen. Construction preserves the global RNG.
"""
    def __init__(self, original, mode='STATE_PROPAGATION', state_dim=16, graph_rank=8,
                 max_graph_mix=.25, max_state_delta=.1):
        super().__init__()
        if mode not in MODES or state_dim < 1 or graph_rank < 1:
            raise ValueError('Invalid ISP mode or hidden dimensions')
        if not 0 < max_graph_mix <= 1 or not 0 < max_state_delta <= 1:
            raise ValueError('ISP mixing/residual bounds must be in (0, 1]')
        if not original._model_args['dy_graph'] or len(original._model_args['adjs']) != 2:
            raise ValueError('ISP-Lite requires original two-direction dynamic graphs')
        if original.seq_len != original.horizon or original.horizon % original._model_args['gap']:
            raise ValueError('ISP-Lite requires complete original gap-sized forecast blocks')
        self.original = original
        self.mode = mode
        self.node_num = original.node_num
        self.input_dim = original.input_dim
        self.output_dim = original.output_dim
        self.seq_len = original.seq_len
        self.horizon = original.horizon
        self.blocks = self.horizon // original._model_args['gap']
        self.state_dim, self.graph_rank = state_dim, graph_rank
        self.max_graph_mix, self.max_state_delta = max_graph_mix, max_state_delta
        reference = next(original.parameters())
        # Extensions are reproducible without consuming the original dropout RNG.
        devices = [reference.device.index] if reference.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            self.state = IncidentState(original._hidden_dim, state_dim, original._forecast_dim,
                                       self.blocks, max_state_delta).to(reference.device, reference.dtype)
            self.graph = IncidentGraph(state_dim, graph_rank).to(reference.device, reference.dtype)
        eye = torch.eye(self.node_num, dtype=torch.bool, device=reference.device)
        roads = torch.stack([g.to(reference.device) > 0 for g in original._model_args['adjs']]) & ~eye
        self.register_buffer('road_support', roads)
        self._state_enabled = self._graph_enabled = self._new_report_enabled = True
        self._trace = None

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    @contextmanager
    def intervention(self, *, state=True, propagation=True, new_report=True):
        """Same-checkpoint path probes; new_report leaves original ICSF/TIID on."""
        previous = self._state_enabled, self._graph_enabled, self._new_report_enabled
        self._state_enabled, self._graph_enabled, self._new_report_enabled = state, propagation, new_report
        try:
            yield self
        finally:
            self._state_enabled, self._graph_enabled, self._new_report_enabled = previous

    def _original_graphs(self, history, node_u, node_d, tod, dow):
        # The same original operations, additionally retaining first-order bases
        # before diagonal removal so mixing happens before multi-order expansion.
        static = [F.softmax(F.relu(torch.mm(node_u, node_d.T)), dim=1)] if self.original._model_args['sta_graph'] else []
        constructor = self.original.dynamic_graph_constructor
        bases = constructor.normalizer(constructor.mask(constructor.distance_function(
            history, node_d, node_u, tod, dow)))
        dynamic = constructor.st_localization(constructor.multi_order(bases))
        return static, dynamic, bases

    def _schedule(self, nodes, states, intensity, support, bases, native_dynamic):
        constructor = self.original.dynamic_graph_constructor
        reachable = support
        union = self.road_support.any(0).to(nodes.dtype)
        graphs, counts = [], []
        for k in range(self.blocks):
            if not k:
                # Block zero is the native historical DIF readout, so no future
                # graph is evaluated or expanded for this unused schedule slot.
                graphs.append(None)
                counts.append(reachable.sum(-1))
                continue
            reachable = reachable | (torch.matmul(reachable.to(nodes.dtype), union.T) > 0)
            corrected = self.graph(nodes, states[:, k], bases, self.road_support,
                                   self.max_graph_mix * intensity[:, k], reachable)
            # Keep [B,N,N] banks, not three replicated time slices per future
            # block. Future convolution applies their compact message residual.
            graphs.append([matrix for direction in constructor.multi_order(corrected) for matrix in direction])
            counts.append(reachable.sum(-1))
        return graphs, torch.stack(counts, 1)

    @staticmethod
    def _future_convolution(convolution, window, dynamic, static, corrected):
        """Native messages plus compact graph deltas before the native dropout.

Original graphs repeat the same spatial matrix over k_t slots. Therefore a
graph difference contracts against the SUM of the transformed slots. Keeping
the native contraction avoids changing FP reduction order at initialization.
No future [B,N,k_t*N] graph copies or [B,N,N,C] messages are materialized.
"""
        x = window.unfold(1, convolution.k_t, 1).permute(0, 1, 2, 4, 3)
        b, t, n, k, c = x.shape
        encoded = convolution.activation(convolution.fc_list_updt(x.reshape(b, t, n, k * c)))
        encoded = encoded.view(b, t, n, k, c)
        values = encoded.transpose(-3, -2).reshape(b, t, k * n, c)
        out = [encoded.mean(-2)]
        if convolution.use_predefined_graph:
            out += [torch.einsum('nm,btmf->btnf', graph, values) for graph in convolution.pre_defined_graph]
        summed = encoded.sum(-2)
        if convolution.use_dynamic_hidden_graph:
            for old, new in zip(dynamic, corrected):
                native = torch.einsum('bnm,btmf->btnf', old, values)
                compact_old = old.reshape(b, n, k, n)[:, :, 0, :]
                change = torch.einsum('bnm,btmf->btnf', new - compact_old, summed)
                out.append(native + change)
        if convolution.use_static_hidden_graph:
            out += [torch.einsum('nm,btmf->btnf', graph, values) for graph in convolution.get_graph(static)]
        return convolution.dropout(convolution.gcn_updt(torch.cat(out, -1)))

    @staticmethod
    def _forecast_dif(layer, history, dynamic, static, schedule):
        gated, original_history = history
        dif = layer.dif_layer
        convolution = dif.localized_st_conv
        encoded = convolution(gated, dynamic, static)
        predict = [encoded[:, -1:].contiguous()]
        window_size = convolution.k_t
        for k in range(1, len(schedule)):
            window = predict[-window_size:]
            if len(window) < window_size:
                window = torch.cat([gated[:, -(window_size - len(window)):]] + window, dim=1)
            else:
                window = torch.cat(window, dim=1)
            predict.append(ISPIGSTGNN._future_convolution(convolution, window, dynamic, static, schedule[k]))
        forecast = dif.forecast_branch.forecast_fc(torch.cat(predict, dim=1))
        backcast = dif.backcast_branch(encoded)
        residual = dif.residual_decompose(original_history[:, -backcast.shape[1]:], backcast)
        return residual, forecast

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        # Native no-report behavior, including its original normalization rules.
        if incident_data is None:
            self._trace = None
            return self.original(history_data, label, incident_data=None, sensor_data=sensor_data)
        native = self.original
        traffic, node_u, node_d, tod, dow = native._prepare_inputs(history_data)
        traffic = native.embedding(traffic)
        enhanced, context = native.icsf_module(traffic, incident_data, sensor_data, tod[:, -1, 0], dow[:, -1, 0])
        static, dynamic, bases = self._original_graphs(enhanced, node_u, node_d, tod, dow)
        support = incident_data['distances'].abs().sum(-1) > 0
        if not self._new_report_enabled:
            support = torch.zeros_like(support)
        report = native.icsf_module.embed_incident_features(incident_data, tod[:, -1, 0], dow[:, -1, 0])
        nodes, states, intensity, delta = self.state(traffic, report, support)
        schedule = None
        reachable = support.sum(-1)[:, None].expand(-1, self.blocks)
        if self.mode == 'STATE_PROPAGATION' and self._graph_enabled:
            schedule, reachable = self._schedule(nodes, states, intensity, support, bases, dynamic)
        residual, forecasts_dif, forecasts_inh = enhanced, [], []
        for layer in native.layers:
            if schedule is None:
                residual, dif, inh = layer(residual, dynamic, static, node_u, node_d, tod, dow)
            else:
                gated = layer.estimation_gate(node_u, node_d, tod, dow, residual)
                dif_residual, dif = self._forecast_dif(layer, (gated, residual), dynamic, static, schedule)
                residual, inh = layer.inh_layer(dif_residual)
            forecasts_dif.append(dif)
            forecasts_inh.append(inh)
        hidden = sum(forecasts_dif) + sum(forecasts_inh)
        hidden = native.tiid_module(hidden, context)
        if self._state_enabled:
            hidden = hidden + delta
        prediction = native.out_fc_2(F.relu(native.out_fc_1(F.relu(hidden))))
        prediction = prediction.transpose(1, 2).contiguous().view(prediction.shape[0], prediction.shape[2], -1)
        self._trace = dict(states=states.detach(), intensity=intensity.detach(),
                           direct_support=support.sum(-1).detach(), reachable=reachable.detach(),
                           state_delta_abs_max=delta.detach().abs().amax())
        return prediction.transpose(1, 2).unsqueeze(-1)

    def diagnostics(self):
        result = dict(schema='original_isp_lite_v1', mode=self.mode, state_dim=self.state_dim,
            graph_rank=self.graph_rank, max_graph_mix=self.max_graph_mix, max_state_delta=self.max_state_delta,
            forecast_blocks=self.blocks, state_time_unit='original_gap_sized_latent_block',
            new_state_historical_report_injection=False, persistent_cross_sample_state=False,
            original_ICSF_TIID_retained=True, native_latest_slot_ICSF_retained=True, physics_certified=False,
            graph_convention='receiver_source', graph_scope='future_DIF_calls_after_first_forecast_block')
        result['aggregation_scope'] = 'last_forward_batch_only_not_complete_validation'
        if self._trace is not None:
            result.update(intensity_mean_by_block=self._trace['intensity'].mean(0).cpu().tolist(),
                state_abs_mean_by_block=self._trace['states'].abs().mean((0, 2)).cpu().tolist(),
                supported_nodes_per_sample=self._trace['direct_support'].cpu().tolist(),
                reachable_nodes_mean_by_block=self._trace['reachable'].float().mean(0).cpu().tolist(),
                state_delta_abs_max=float(self._trace['state_delta_abs_max'].cpu()))
        return result
