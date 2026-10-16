"""First-layer historical road alignment with path-level boundary fallback.

The original local-window encoder and native banks are evaluated once. Only
road-bank messages receive a bounded, zero-initialized residual before the
original GCN projection/dropout. Forecast calls and deeper DIFs stay native.
"""
from contextlib import contextmanager

import torch
from torch import nn

from experiments.chronological.original_edge_delay import (
    lag_kernels, mass_only_messages, road_messages)

MODES = ('FIXED', 'DYNAMIC', 'NO_ALIGN', 'SHUFFLE', 'ZERO_BOUNDARY')


class ResidualEdgeDelay(nn.Module):
    def __init__(self, adjacency, hidden_dim, max_delay=2, score_dim=8):
        super().__init__()
        if (adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
                or not torch.isfinite(adjacency).all() or (adjacency < 0).any()):
            raise ValueError('Expected finite nonnegative square road matrix')
        if type(max_delay) is not int or max_delay < 1 or score_dim < 1:
            raise ValueError('Invalid delay bounds or scorer width')
        self.max_delay = max_delay
        self.register_buffer('adjacency', adjacency.detach().clone())
        receiver, source = torch.where(adjacency != 0)
        self.register_buffer('receiver', receiver)
        self.register_buffer('source', source)
        # L2 regularizes deviation from the reference, not the reference itself.
        self.register_buffer('base_logit', adjacency.new_tensor(-2.))
        self.edge_deviation = nn.Parameter(adjacency.new_zeros(len(source)))
        self.source_encoder = nn.Linear(2 * hidden_dim, score_dim)
        self.receiver_encoder = nn.Linear(2 * hidden_dim, score_dim)
        self.score = nn.Linear(score_dim, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self.mode = 'DYNAMIC'

    def delays(self, history):
        if self.mode not in MODES:
            raise ValueError('Unknown delay mode')
        score = self.base_logit + self.edge_deviation.unsqueeze(0).expand(len(history), -1)
        if self.mode != 'FIXED':
            latest = history[:, -1]
            context = torch.cat((latest, latest - history[:, -2]), dim=-1)
            source = self.source_encoder(context)[:, self.source]
            receiver = self.receiver_encoder(context)[:, self.receiver]
            score = score + self.score(torch.tanh(source + receiver)).squeeze(-1)
        delay = self.max_delay * torch.sigmoid(score)
        if self.mode == 'SHUFFLE':
            delay = delay.roll(1, -1)
        self.last_delay = delay.detach()
        return delay

    def kernels(self, history):
        delay = self.delays(history)
        dense = history.new_zeros((len(history), *self.adjacency.shape))
        dense[:, self.receiver, self.source] = delay
        return lag_kernels(self.adjacency, dense, self.max_delay)


def temporal_residuals(summary, kernels, orders=2):
    """Available delayed values minus the same coefficients at current time.

Adding this residual to the native off-diagonal A^k message replaces only
available paths. Missing path coefficients retain the native current value.
Return paths are removed at their accumulated lag, as in both v1 operators.
For any sequence constant over time, these residuals are zero (roundoff aside).
"""
    aligned = road_messages(summary, kernels, orders)
    current = mass_only_messages(summary, kernels, orders)
    return [a - c for a, c in zip(aligned, current)]


def historical_forward(conv, history, dynamic_graph, static_graph, estimators, scales,
                       zero_boundary=False):
    windows = history.unfold(1, conv.k_t, 1).permute(0, 1, 2, 4, 3)
    batch, time, nodes, slots, channels = windows.shape
    encoded = conv.activation(conv.fc_list_updt(windows.reshape(batch, time, nodes, slots * channels)))
    encoded = encoded.view(batch, time, nodes, slots, channels)
    expanded = encoded.transpose(-3, -2).reshape(batch, time, slots * nodes, channels)
    support = list(conv.pre_defined_graph)
    if conv.use_dynamic_hidden_graph:
        support += list(dynamic_graph)
    if conv.use_static_hidden_graph:
        support += conv.get_graph(static_graph)
    # Exactly the original order and contractions; no double dropout or extra RNG.
    messages = [encoded.mean(-2)]
    for graph in support:
        equation = 'nm,btmf->btnf' if graph.ndim == 2 else 'bnm,btmf->btnf'
        messages.append(torch.einsum(equation, graph, expanded))
    summary = encoded.sum(-2)
    for graph_index, estimator in enumerate(estimators):
        kernels = estimator.kernels(history)
        if zero_boundary:
            aligned = road_messages(summary, kernels, conv.k_s)
            residuals = [value - messages[1 + graph_index * conv.k_s + order]
                         for order, value in enumerate(aligned)]
        else:
            residuals = temporal_residuals(summary, kernels, conv.k_s)
        for order, residual in enumerate(residuals):
            index = 1 + graph_index * conv.k_s + order
            messages[index] = messages[index] + scales[graph_index, order] * residual
    return conv.dropout(conv.gcn_updt(torch.cat(messages, -1)))


class FirstLayerDelayDif(nn.Module):
    def __init__(self, original, mode='DYNAMIC', max_delay=2, score_dim=8):
        super().__init__()
        conv = original.localized_st_conv
        if (not conv.use_predefined_graph or conv.k_s not in (1, 2)
                or len(original.pre_defined_graph) != 2):
            raise ValueError('Expected two original road directions and k_s=1 or 2')
        self.original = original
        self.delays = nn.ModuleList([ResidualEdgeDelay(graph, conv.hidden_dim, max_delay, score_dim)
                                    for graph in original.pre_defined_graph])
        self.alignment_scale = nn.Parameter(torch.zeros(len(self.delays), conv.k_s))
        self.set_mode(mode)

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError('Unknown delay mode')
        self.mode = mode
        for estimator in self.delays:
            estimator.mode = mode

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        if self.mode == 'NO_ALIGN':
            return self.original(history_data, gated_history_data, dynamic_graph, static_graph)
        conv = self.original.localized_st_conv
        hidden = historical_forward(conv, gated_history_data, dynamic_graph, static_graph,
            self.delays, self.alignment_scale.tanh(), self.mode == 'ZERO_BOUNDARY')
        forecast = self.original.forecast_branch(gated_history_data, hidden, conv, dynamic_graph, static_graph)
        backcast = self.original.backcast_branch(hidden)
        residual = self.original.residual_decompose(history_data[:, -backcast.shape[1]:], backcast)
        return residual, forecast


def install_delay(model, mode='DYNAMIC', max_delay=2, score_dim=8, seed=2025):
    if any(isinstance(layer.dif_layer, FirstLayerDelayDif) or hasattr(layer.dif_layer, 'original')
           for layer in model.layers):
        raise ValueError('Expected an unwrapped original model')
    device = next(model.parameters()).device
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model.layers[0].dif_layer = FirstLayerDelayDif(model.layers[0].dif_layer, mode,
                                                     max_delay, score_dim).to(device)
    model.train(model.training)
    return model


@contextmanager
def delay_mode(model, mode):
    wrapper = model.layers[0].dif_layer
    if not isinstance(wrapper, FirstLayerDelayDif):
        raise ValueError('Expected a v2 delay model')
    previous = wrapper.mode
    try:
        wrapper.set_mode(mode)
        yield
    finally:
        wrapper.set_mode(previous)


def diagnostics(model):
    wrapper = model.layers[0].dif_layer
    if not isinstance(wrapper, FirstLayerDelayDif):
        return {}
    lags = []
    for number, estimator in enumerate(wrapper.delays):
        delay = getattr(estimator, 'last_delay', None)
        if delay is not None and delay.numel() and wrapper.mode != 'NO_ALIGN':
            lags.append(dict(graph=number, positions=delay.numel(), mean=float(delay.mean()),
                min=float(delay.min()), max=float(delay.max()),
                edge_std_mean=float(delay.std(-1, unbiased=False).mean()),
                sample_std_mean=float(delay.std(0, unbiased=False).mean())))
    return dict(scales=wrapper.alignment_scale.detach().tanh().cpu().tolist(), lags=lags)


def description(model):
    wrapper = model.layers[0].dif_layer
    return dict(schema='original_first_layer_boundary_delay_v2',
        modified_layers=[0] if isinstance(wrapper, FirstLayerDelayDif) else [],
        registered_parameters=sum(p.numel() for p in model.parameters()),
        original_equivalent_initialization=True, scales='tanh, per road direction/order, initially zero',
        delay='max_delay * sigmoid(fixed -2 buffer + zero-initialized edge deviation + state score)',
        boundary='missing complete path contribution falls back to native same-time message',
        temporal_axis='ending time of complete nonlinear observed-window summary',
        forecast_calls='native original gap=3, unchanged',
        fixed_active_parameters='state encoders/scorer registered but not called in FIXED',
        physical_delay_identified=False, causal_claim=False)
