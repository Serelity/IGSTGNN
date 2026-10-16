"""Historical road-bank delay alignment around the hash-checked original model.

Only historical DIF road messages change. Native temporal projection, self,
dynamic/static banks, backcast, INH, and all gap=3 forecast calls are retained.
Delays are constant within each supplied historical window, in its sampling
steps; they are predictive lags, not certified travel times or causal effects.
"""
from contextlib import contextmanager
import torch
from torch import nn


MODES = ('FIXED', 'DYNAMIC', 'NO_ALIGN', 'SHUFFLE', 'MASS_ONLY')


def lag_kernels(adjacency, delay, max_delay):
    """Piecewise-linear sampling weights [B, lag, receiver, source].

At integer delays the right-hand interpolation derivative is retained. A
single delayed sample has at most two nonzero kernels. No lag is negative.
"""
    lower = torch.floor(delay).to(torch.long)
    fraction = delay - lower.to(delay.dtype)
    weights = []
    for lag in range(max_delay + 1):
        weight = (lower == lag).to(delay.dtype) * (1 - fraction)
        weight = weight + (lower + 1 == lag).to(delay.dtype) * fraction
        weights.append(adjacency.unsqueeze(0) * weight)
    return torch.stack(weights, dim=1)


def shifted(sequence, lag):
    """Shift [B,T,N,C] into the past, with unavailable contributions zeroed."""
    if lag == 0:
        return sequence
    if lag >= sequence.shape[1]:
        return torch.zeros_like(sequence)
    return torch.cat((torch.zeros_like(sequence[:, :lag]), sequence[:, :-lag]), dim=1)


def apply_kernels(sequence, kernels):
    result = torch.zeros_like(sequence)
    for lag in range(min(sequence.shape[1], kernels.shape[1])):
        # No [B,T,E,C] messages or receiver Python loops are materialized.
        past = shifted(sequence, lag)
        result = result + torch.einsum('bnm,btmc->btnc', kernels[:, lag], past)
    return result


def road_messages(sequence, kernels, orders=2, *, aligned=True):
    """One/two-hop lag messages, removing only final bank diagonals.

Composition uses actual delayed intermediate states. A two-edge path has
the sum of its (window-constant) edge lags. Return paths i->j->i are removed
at their own accumulated delay, matching the original diagonal bank mask.
Unavailable temporal mass is dropped, never clamped to the first sample or
renormalized onto available observations.
"""
    if orders not in (1, 2):
        raise ValueError('Round one supports one or two spatial orders')
    if not aligned:
        return mass_only_messages(sequence, kernels, orders)
    first = apply_kernels(sequence, kernels)
    diagonal = torch.zeros_like(sequence)
    for lag in range(min(sequence.shape[1], kernels.shape[1])):
        diagonal = diagonal + kernels[:, lag].diagonal(dim1=-2, dim2=-1)[:, None, :, None] * shifted(sequence, lag)
    results = [first - diagonal]
    if orders == 2:
        second = apply_kernels(first, kernels)
        returns = torch.zeros_like(sequence)
        for a in range(kernels.shape[1]):
            for b in range(min(kernels.shape[1], max(0, sequence.shape[1] - a))):
                # diag(K_a @ K_b); no dense N^3 matrix multiplication.
                coefficient = (kernels[:, a] * kernels[:, b].transpose(-1, -2)).sum(-1)
                returns = returns + coefficient[:, None, :, None] * shifted(sequence, a + b)
        results.append(second - returns)
    return results


def mass_only_messages(sequence, kernels, orders=2):
    """Same available path mass as alignment, but all values read at time t.

This inference control separates temporal resampling from the attenuation
caused by missing early history. Delays, supports and path weights stay put.
"""
    time = torch.arange(sequence.shape[1], device=sequence.device)
    values = [torch.einsum('bnm,btmc->btnc', matrix, sequence) for matrix in kernels.unbind(1)]
    first = torch.zeros_like(sequence)
    second = torch.zeros_like(sequence)
    for a, matrix in enumerate(kernels.unbind(1)):
        valid = (time >= a)[None, :, None, None]
        diagonal = matrix.diagonal(dim1=-2, dim2=-1)[:, None, :, None]
        first = first + (values[a] - diagonal * sequence) * valid
        if orders == 2:
            for b, other in enumerate(kernels.unbind(1)):
                if a+b >= sequence.shape[1]:
                    continue
                diagonal = (matrix * other.transpose(-1, -2)).sum(-1)[:, None, :, None]
                second = second + (torch.einsum('bnm,btmc->btnc', matrix, values[b]) - diagonal * sequence) * (time >= a+b)[None, :, None, None]
    return [first, second] if orders == 2 else [first]


class EdgeDelay(nn.Module):
    """A learned static edge lag plus a shared state-dependent correction."""

    def __init__(self, adjacency, hidden_dim, max_delay=2, score_dim=8):
        super().__init__()
        if (adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
                or not torch.isfinite(adjacency).all() or (adjacency < 0).any()):
            raise ValueError('Expected a finite nonnegative square road matrix')
        if type(max_delay) is not int or max_delay < 1 or score_dim < 1:
            raise ValueError('Invalid delay bounds/scorer width')
        self.max_delay = max_delay
        self.register_buffer('adjacency', adjacency.detach().clone())
        receiver, source = torch.where(adjacency != 0)
        self.register_buffer('receiver', receiver)
        self.register_buffer('source', source)
        self.edge_logits = nn.Parameter(adjacency.new_full((len(source),), -2.0))
        self.source_encoder = nn.Linear(2 * hidden_dim, score_dim)
        self.receiver_encoder = nn.Linear(2 * hidden_dim, score_dim)
        self.score = nn.Linear(score_dim, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self.mode = 'DYNAMIC'

    def delays(self, history):
        if self.mode not in MODES:
            raise ValueError('Unknown delay mode: ' + self.mode)
        # State here is observed input history (or the original deeper-layer
        # residual), never a target or a gap=3 generated forecast block.
        latest = history[:, -1]
        change = latest - history[:, -2]
        context = torch.cat((latest, change), dim=-1)
        score = self.edge_logits.unsqueeze(0).expand(history.shape[0], -1)
        if self.mode in ('DYNAMIC', 'SHUFFLE', 'MASS_ONLY'):
            source = self.source_encoder(context)[:, self.source]
            receiver = self.receiver_encoder(context)[:, self.receiver]
            score = score + self.score(torch.tanh(source + receiver)).squeeze(-1)
        delay = self.max_delay * torch.sigmoid(score)
        if self.mode == 'SHUFFLE':
            # Fixed cyclic edge permutation within this directed graph. No RNG
            # draws, targets, graph changes, or cross-direction mixing.
            delay = delay.roll(1, dims=-1)
        self.last_delay = delay.detach()
        return delay

    def kernels(self, history):
        sparse_delay = self.delays(history)
        dense = history.new_zeros((history.shape[0], *self.adjacency.shape))
        dense[:, self.receiver, self.source] = sparse_delay
        return lag_kernels(self.adjacency, dense, self.max_delay)


def historical_forward(conv, history, dynamic_graph, static_graph, delays):
    """Shift complete local-window summaries on their real ending-time axis.

The original 3C->3C nonlinearity is preserved. P(t) summarizes the original
complete window ending at t. Thus resampling P(t-tau) shifts the whole
window before spatial mixing; it never labels mixed subslots as timestamps.
"""
    windows = history.unfold(1, conv.k_t, 1).permute(0, 1, 2, 4, 3)
    batch, time, nodes, slots, channels = windows.shape
    encoded = conv.activation(conv.fc_list_updt(windows.reshape(batch, time, nodes, slots * channels)))
    encoded = encoded.reshape(batch, time, nodes, slots, channels)
    messages = [encoded.mean(-2)]
    summary = encoded.sum(-2)
    for estimator in delays:
        messages.extend(road_messages(summary, estimator.kernels(history), conv.k_s,
            aligned=estimator.mode != 'MASS_ONLY'))
    expanded = encoded.transpose(-3, -2).reshape(batch, time, slots * nodes, channels)
    remaining = []
    if conv.use_dynamic_hidden_graph:
        remaining.extend(dynamic_graph)
    if conv.use_static_hidden_graph:
        remaining.extend(conv.get_graph(static_graph))
    for graph in remaining:
        equation = 'nm,btmf->btnf' if graph.ndim == 2 else 'bnm,btmf->btnf'
        messages.append(torch.einsum(equation, graph, expanded))
    return conv.dropout(conv.gcn_updt(torch.cat(messages, dim=-1)))


class HistoricalDelayDif(nn.Module):
    """Original DIF with one historical road-message replacement."""

    def __init__(self, original, mode='DYNAMIC', max_delay=2, score_dim=8):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown delay mode')
        conv = original.localized_st_conv
        if not conv.use_predefined_graph or conv.k_s not in (1, 2):
            raise ValueError('Expected original predefined road banks, k_s=1 or 2')
        self.original = original
        self.delays = nn.ModuleList([EdgeDelay(graph, conv.hidden_dim, max_delay, score_dim)
            for graph in original.pre_defined_graph])
        self.set_mode(mode)

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError('Unknown delay mode')
        self.mode = mode
        for estimator in self.delays:
            estimator.mode = mode

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        conv = self.original.localized_st_conv
        if self.mode == 'NO_ALIGN':
            return self.original(history_data, gated_history_data, dynamic_graph, static_graph)
        hidden = historical_forward(conv, gated_history_data, dynamic_graph, static_graph, self.delays)
        # Pass the ORIGINAL convolution: generated forecast blocks do not have
        # a one-observation-step clock and must never use this lag estimator.
        forecast = self.original.forecast_branch(gated_history_data, hidden, conv, dynamic_graph, static_graph)
        backcast = self.original.backcast_branch(hidden)
        residual = self.original.residual_decompose(history_data[:, -backcast.shape[1]:], backcast)
        return residual, forecast


def install_delay(model, mode='DYNAMIC', max_delay=2, score_dim=8, seed=2025):
    """Install on an already loaded original model without advancing its RNG."""
    if any(isinstance(layer.dif_layer, HistoricalDelayDif) for layer in model.layers):
        raise ValueError('Delay module already installed')
    device = next(model.parameters()).device
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for layer in model.layers:
            layer.dif_layer = HistoricalDelayDif(layer.dif_layer, mode, max_delay, score_dim).to(device)
    # Keep model training/eval mode coherent after replacing submodules.
    model.train(model.training)
    return model


@contextmanager
def delay_mode(model, mode):
    layers = [layer.dif_layer for layer in model.layers if isinstance(layer.dif_layer, HistoricalDelayDif)]
    if not layers:
        raise ValueError('Expected a delay model')
    previous = [layer.mode for layer in layers]
    try:
        for layer in layers:
            layer.set_mode(mode)
        yield
    finally:
        for layer, old in zip(layers, previous):
            layer.set_mode(old)


def description(model):
    layers = [layer.dif_layer for layer in model.layers if isinstance(layer.dif_layer, HistoricalDelayDif)]
    return dict(schema='original_igstgnn_historical_edge_delay_v1',
        modified_history_layers=len(layers), registered_parameters=sum(p.numel() for p in model.parameters()),
        road_edges_by_layer=[[len(est.source) for est in layer.delays] for layer in layers],
        max_delay_by_layer=[[est.max_delay for est in layer.delays] for layer in layers],
        initial_delay_steps='max_delay * sigmoid(-2), not zero',
        source_window_projection='unchanged original nonlinear 3C->3C, summed over mixed subslots',
        delay_axis='ending time of complete observed windows, not mixed subslot or gap=3 forecast index',
        history_boundary='drop unavailable temporal mass; no first-sample clamping or renormalization',
        multi_hop='compose delayed operators, remove delayed return paths only at final bank',
        delay_scope='constant across each layer input window, state-dependent across samples',
        forecast_convolution='original unchanged',
        direction='original adjacency receiver/source convention and its transpose; physical direction not certified',
        fixed_scope='learned edge-specific constants shared across samples, not data-calibrated physical delays',
        causal_claim=False, physical_delay_identified=False)


def delay_diagnostics(model):
    """Descriptive lag statistics after forward; outside its timing boundary."""
    rows = []
    for number, layer in enumerate(model.layers):
        if not isinstance(layer.dif_layer, HistoricalDelayDif):
            continue
        for graph, estimator in enumerate(layer.dif_layer.delays):
            delay = getattr(estimator, 'last_delay', None)
            if delay is None or not delay.numel() or layer.dif_layer.mode == 'NO_ALIGN':
                continue
            rows.append(dict(layer=number, graph=graph, mode=estimator.mode, positions=delay.numel(),
                mean=float(delay.mean()), min=float(delay.min()), max=float(delay.max()),
                edge_std_mean=float(delay.std(-1, unbiased=False).mean()),
                sample_std_mean=float(delay.std(0, unbiased=False).mean())))
    return rows
