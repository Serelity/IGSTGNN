"""GW1: graph-source/content selection of native IGSTGNN local window slots.

Only the first DIF STLocalizedConv is wrapped. Its shared window transform,
self path, graph banks, bank order, projection and dropout remain native.
Slots are *mixed latent slots* after the native dense temporal transform;
their weights are neither physical delays nor incident importance.
"""
from contextlib import contextmanager

import torch
from torch import nn


ARMS = ('ORIGINAL', 'STATIC', 'SHARED_DYNAMIC', 'GRAPH_CONTENT')
ROUTER_MODES = ARMS[1:]
DESIGN = dict(version='GW1', selected_layer=0, router_hidden=16,
    placement='after_native_shared_window_transform_before_graph_contraction',
    source_order='predefined_directions_then_dynamic_directions_then_adaptive',
    share_across_graph_orders=True, self_path='unchanged',
    content='current_native_gated_hidden_window_including_inherited_ICSF_context',
    forecast_scope='historical_and_autoregressive_DIF_calls_in_selected_layer',
    separate_report_inputs=False, target_inputs=False)


class WindowRouter(nn.Module):
    """All candidate arms register the same tensors; active capacity differs.

    STATIC: graph-specific constants. SHARED_DYNAMIC: content-dependent weights
    shared by all graph sources. GRAPH_CONTENT: static + shared content +
    source-specific content interaction (centered across sources).
    Zero score heads recover unit coefficients, without a detached zero branch.
    """
    def __init__(self, channels, kernel, sources, mode, hidden=16):
        super().__init__()
        if mode not in ROUTER_MODES or min(channels, kernel, sources, hidden) < 1:
            raise ValueError('Invalid GW1 router dimensions or mode')
        self.kernel, self.sources, self.mode = kernel, sources, mode
        self.encoder = nn.Linear(kernel * channels, hidden)
        self.shared_head = nn.Linear(hidden, kernel)
        self.interaction_head = nn.Linear(hidden, sources * kernel)
        self.static_logits = nn.Parameter(torch.zeros(sources, kernel))
        nn.init.zeros_(self.shared_head.weight)
        nn.init.zeros_(self.shared_head.bias)
        nn.init.zeros_(self.interaction_head.weight)
        nn.init.zeros_(self.interaction_head.bias)

    def forward(self, window, mode=None):
        # window [B,T,N,K,C]; output [B,T,N,G,K]. No future targets/labels.
        mode = self.mode if mode is None else mode
        if mode not in (*ROUTER_MODES, 'UNIFORM'):
            raise ValueError('Unknown GW1 intervention')
        prefix = window.shape[:-2]
        if mode == 'UNIFORM':
            return window.new_ones((*prefix, self.sources, self.kernel))
        if mode == 'STATIC':
            logits = self.static_logits.expand(*prefix, -1, -1)
        else:
            encoded = torch.tanh(self.encoder(window.flatten(-2)))
            shared = self.shared_head(encoded).unsqueeze(-2)
            if mode == 'SHARED_DYNAMIC':
                logits = shared.expand(*prefix, self.sources, -1)
            else:
                interaction = self.interaction_head(encoded).reshape(*prefix, self.sources, self.kernel)
                interaction = interaction - interaction.mean(-2, keepdim=True)
                logits = self.static_logits + shared + interaction
        probability = torch.softmax(logits, dim=-1)
        # Algebraically K*softmax; subtraction guarantees exact unit recovery
        # at equal logits, also for kernels whose reciprocal is rounded.
        return 1. + self.kernel * (probability - probability.new_tensor(1. / self.kernel))


class GraphWindowConv(nn.Module):
    def __init__(self, native, mode, hidden=16):
        super().__init__()
        self.native = native
        k = native.k_s
        if k < 1 or len(native.pre_defined_graph) % k:
            raise ValueError('Native predefined graph order is invalid')
        directions = len(native.pre_defined_graph) // k
        labels = []
        if native.use_predefined_graph:
            labels += ['PREDEFINED_'+str(i) for i in range(directions)]
        if native.use_dynamic_hidden_graph:
            labels += ['DYNAMIC_'+str(i) for i in range(directions)]
        if native.use_static_hidden_graph:
            labels += ['ADAPTIVE_0']
        if not labels or native.num_matric != 1 + k * len(labels):
            raise ValueError('GW1 requires native source-major, order-minor graph banks')
        self.source_labels, self.directions = tuple(labels), directions
        # Preserve native model/training RNG, including all target CUDA states.
        device = next(native.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.router = WindowRouter(native.hidden_dim, native.k_t, len(labels), mode, hidden).to(device)
        self.override_mode = None
        self._stats = None

    def _supports(self, dynamic, static):
        native = self.native
        supports = []
        if native.use_predefined_graph:
            supports += native.pre_defined_graph
        if native.use_dynamic_hidden_graph:
            if len(dynamic) != self.directions * native.k_s:
                raise ValueError('Dynamic graph count/order differs from native source contract')
            supports += dynamic
        if native.use_static_hidden_graph:
            if len(static) != 1:
                raise ValueError('Expected the single native adaptive graph')
            supports += native.get_graph(static)
        return supports

    def forward(self, x, dynamic_graph, static_graph):
        native = self.native
        if x.ndim != 4 or x.shape[1] < native.k_t or x.shape[-1] != native.hidden_dim:
            raise ValueError('GW1 input must be [batch,time>=kernel,node,native_channels]')
        if self.override_mode == 'UNIFORM':
            return native(x, dynamic_graph, static_graph)
        window = x.unfold(1, native.k_t, 1).permute(0, 1, 2, 4, 3)
        b, t, n, k, c = window.shape
        transformed = native.activation(native.fc_list_updt(window.reshape(b, t, n, k*c)))
        transformed = transformed.view(b, t, n, k, c)
        weights = self.router(window, self.override_mode)
        supports = self._supports(dynamic_graph, static_graph)
        out = [transformed.mean(-2)]  # native self path never gated
        for source in range(len(self.source_labels)):
            selected = transformed * weights[..., source, :, None]
            # Match native [lag,source_node] flattening, not [node,lag].
            selected = selected.transpose(-3, -2).reshape(b, t, k*n, c)
            for graph in supports[source*native.k_s:(source+1)*native.k_s]:
                if graph.ndim not in (2, 3) or tuple(graph.shape[-2:]) != (n, k*n):
                    raise ValueError('Graph must use native [receiver,lag*source_node] axes')
                out.append(torch.einsum('nm,btmf->btnf', graph, selected) if graph.ndim == 2
                           else torch.einsum('bnm,btmf->btnf', graph, selected))
        if self._stats is not None:
            self._accumulate(weights)
        return native.dropout(native.gcn_updt(torch.cat(out, dim=-1)))

    def _accumulate(self, weights):
        # Detach on device, no .item()/.cpu() or per-graph host synchronization.
        w = weights.detach().double()
        p = (w / self.native.k_t).clamp_min(torch.finfo(w.dtype).tiny)
        row = dict(weight_sum=w.sum((0, 1, 2)),
            weight_square_sum=w.square().sum((0, 1, 2)),
            entropy_sum=(-(p * p.log()).sum(-1)).sum((0, 1, 2)),
            deviation_abs_sum=(w-1).abs().sum(), deviation_abs_max=(w-1).abs().max(),
            coefficient_mass_error_abs_max=(w.sum(-1)-self.native.k_t).abs().max())
        stats = self._stats
        for key, value in row.items():
            stats[key] = (torch.maximum(stats[key], value) if key.endswith('_max') else stats[key]+value) if key in stats else value
        stats['positions'] = stats.get('positions', 0) + weights.shape[0]*weights.shape[1]*weights.shape[2]
        stats['calls'] = stats.get('calls', 0) + 1

    def start_diagnostics(self):
        self._stats = {}

    def finish_diagnostics(self):
        raw, self._stats = self._stats, None
        if not raw:
            return dict(calls=0, source_labels=list(self.source_labels))
        count = raw['positions']
        converted = {key: value.cpu().tolist() if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
        converted.update(source_labels=list(self.source_labels),
            weight_mean=(raw['weight_sum']/count).cpu().tolist(),
            weight_std=(raw['weight_square_sum']/count-(raw['weight_sum']/count).square()).clamp_min(0).sqrt().cpu().tolist(),
            entropy_mean=(raw['entropy_sum']/count).cpu().tolist(),
            scope='all_selected_layer_calls_in_one_validation_pass_including_autoregressive_windows',
            interpretation='latent_slot_selection_not_physical_lag_or_module_importance')
        return converted


class GraphWindowIGSTGNN(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        if mode not in ROUTER_MODES:
            raise ValueError('Unknown GW1 training arm')
        self.original, self.mode = original, mode
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon'):
            setattr(self, name, getattr(original, name))
        conv = original.layers[DESIGN['selected_layer']].dif_layer.localized_st_conv
        if isinstance(conv, GraphWindowConv):
            raise ValueError('GW1 cannot be installed twice')
        original.layers[DESIGN['selected_layer']].dif_layer.localized_st_conv = GraphWindowConv(conv, mode, DESIGN['router_hidden'])

    @property
    def selector(self):
        return self.original.layers[DESIGN['selected_layer']].dif_layer.localized_st_conv

    def param_num(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        # label is deliberately ignored, including in the native engine path.
        return self.original(history_data, incident_data=incident_data, sensor_data=sensor_data)

    @contextmanager
    def intervention(self, mode):
        if mode not in ('UNIFORM', 'STATIC'):
            raise ValueError('Only predefined same-checkpoint GW1 interventions are allowed')
        previous = self.selector.override_mode
        self.selector.override_mode = mode
        try:
            yield self
        finally:
            self.selector.override_mode = previous

    def start_diagnostics(self):
        self.selector.start_diagnostics()

    def diagnostics(self):
        return self.selector.finish_diagnostics()


def build_variant(original, arm):
    if arm not in ARMS:
        raise ValueError('Unknown GW1 arm')
    return original if arm == 'ORIGINAL' else GraphWindowIGSTGNN(original, arm)


def split_state(model):
    """Recover native key names for initialization/identity evidence, not loading."""
    if not isinstance(model, GraphWindowIGSTGNN):
        return dict(model.state_dict()), {}
    prefix = 'layers.0.dif_layer.localized_st_conv.'
    native, added = {}, {}
    for key, value in model.original.state_dict().items():
        if key.startswith(prefix+'router.'):
            added[key[len(prefix+'router.'):]] = value
        else:
            native[key.replace(prefix+'native.', prefix)] = value
    return native, added
