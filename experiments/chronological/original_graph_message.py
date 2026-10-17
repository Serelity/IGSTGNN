"""GW2: receiver-side graph-message selection in the first native DIF layer.

The matched POST_LOCAL/POST_MESSAGE arms differ only in scoring evidence.
Native graph banks/orders, mixed latent slots, self path and readout remain.
No target, new report input, physical delay or conservation claim is added.
"""
from contextlib import contextmanager

import torch
from torch import nn


ARMS = ('ORIGINAL', 'STATIC', 'POST_LOCAL', 'POST_MESSAGE')
ROUTER_MODES = ARMS[1:]
DESIGN = dict(version='GW2', selected_layer=0, router_hidden=16,
    placement='receiver_after_graph_propagation_before_slot_sum',
    source_order='predefined_directions_then_dynamic_directions_then_adaptive',
    context='slot_value_and_within_window_centered_contrast',
    message_order_for_scoring=1, share_scorer_across_sources=True,
    propagation_kernel='pack_slots_as_channels_on_native_repeated_adjacency',
    share_across_graph_orders=True, self_path='unchanged',
    forecast_scope='historical_and_autoregressive_DIF_calls_in_selected_layer',
    separate_report_inputs=False, target_inputs=False,
    recovery='floating_tolerance_due_to_changed_reduction_order')


class MessageRouter(nn.Module):
    def __init__(self, channels, kernel, sources, hidden=16):
        super().__init__()
        if min(channels, kernel, sources, hidden) < 1:
            raise ValueError('Invalid GW2 router dimensions')
        self.kernel, self.sources = kernel, sources
        self.static_logits = nn.Parameter(torch.zeros(sources, kernel))
        self.encoder = nn.Linear(2*channels, hidden)
        # A common scalar head bias would cancel in the slot softmax.
        self.score_head = nn.Linear(hidden, 1, bias=False)
        nn.init.zeros_(self.score_head.weight)

    def forward(self, content, source, mode):
        # content [B,T,receiver,K,C]; no graph/source-specific dynamic head.
        if mode not in (*ROUTER_MODES, 'UNIFORM') or not 0 <= source < self.sources:
            raise ValueError('Unknown GW2 mode or graph source')
        if content.ndim != 5 or content.shape[-2] != self.kernel:
            raise ValueError('GW2 content must be [batch,window,node,slot,channel]')
        if mode == 'UNIFORM':
            return content.new_ones(content.shape[:-1])
        logits = self.static_logits[source].expand(content.shape[:-1])
        if mode != 'STATIC':
            contrast = content-content.mean(-2, keepdim=True)
            features = torch.cat((content, contrast), -1)
            logits = logits + self.score_head(torch.tanh(self.encoder(features))).squeeze(-1)
        probability = torch.softmax(logits, -1)
        return 1. + self.kernel*(probability-probability.new_tensor(1./self.kernel))


class GraphMessageConv(nn.Module):
    def __init__(self, native, mode, hidden=16):
        super().__init__()
        if mode not in ROUTER_MODES:
            raise ValueError('Unknown GW2 training arm')
        self.native, self.mode = native, mode
        if native.k_s < 1 or len(native.pre_defined_graph) % native.k_s:
            raise ValueError('Native graph source/order contract differs')
        self.directions = len(native.pre_defined_graph)//native.k_s
        labels = []
        if native.use_predefined_graph:
            labels += ['PREDEFINED_'+str(i) for i in range(self.directions)]
        if native.use_dynamic_hidden_graph:
            labels += ['DYNAMIC_'+str(i) for i in range(self.directions)]
        if native.use_static_hidden_graph:
            labels += ['ADAPTIVE_0']
        if not labels or native.num_matric != 1+native.k_s*len(labels):
            raise ValueError('Native source-major/order-minor banks required')
        self.source_labels = tuple(labels)
        device = next(native.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.router = MessageRouter(native.hidden_dim, native.k_t, len(labels), hidden).to(device)
        self.override_mode = None
        self._stats = None
        self.call_index = 0

    def _supports(self, dynamic, static):
        native, supports = self.native, []
        if native.use_predefined_graph:
            supports += native.pre_defined_graph
        if native.use_dynamic_hidden_graph:
            if len(dynamic) != self.directions*native.k_s:
                raise ValueError('Dynamic graph count/order differs')
            supports += dynamic
        if native.use_static_hidden_graph:
            if len(static) != 1:
                raise ValueError('Expected the single native adaptive graph')
            supports += native.get_graph(static)
        return supports

    @staticmethod
    def slot_messages(graph, transformed, native_repeated=False):
        # Native columns are [lag*sender], rows are receivers. Preserve lag
        # rather than summing it in the native contraction. No extra graph hop.
        b, t, n, k, c = transformed.shape
        if native_repeated:
            # Frozen STLocalizedConv.get_graph / DynamicGraphConstructor.
            # st_localization repeat the SAME adjacency for every slot. Pack
            # slots as feature channels to avoid K graph batches/copies. This
            # branch is valid only for that authenticated native contract.
            if graph.ndim not in (2, 3) or tuple(graph.shape[-2:]) != (n, k*n) or (graph.ndim == 3 and graph.shape[0] != b):
                raise ValueError('Graph must use native [receiver,lag*sender] axes and batch')
            packed = transformed.reshape(b, t, n, k*c)
            base = graph[..., :n]
            result = (torch.einsum('nm,btmf->btnf', base, packed) if graph.ndim == 2
                      else torch.einsum('bnm,btmf->btnf', base, packed))
            return result.reshape(b, t, n, k, c)
        if graph.ndim == 2 and tuple(graph.shape) == (n, k*n):
            return torch.einsum('nkm,btmkc->btnkc', graph.reshape(n, k, n), transformed)
        if graph.ndim == 3 and tuple(graph.shape) == (b, n, k*n):
            return torch.einsum('bnkm,btmkc->btnkc', graph.reshape(b, n, k, n), transformed)
        raise ValueError('Graph must use native [receiver,lag*sender] axes and batch')

    def forward(self, x, dynamic_graph, static_graph):
        native = self.native
        if x.ndim != 4 or x.shape[1] < native.k_t or x.shape[-1] != native.hidden_dim:
            raise ValueError('GW2 input must be [batch,time>=kernel,node,native_channels]')
        phase = 'history' if self.call_index == 0 else 'forecast_'+str(self.call_index)
        self.call_index += 1
        window = x.unfold(1, native.k_t, 1).permute(0, 1, 2, 4, 3)
        b, t, n, k, c = window.shape
        transformed = native.activation(native.fc_list_updt(window.reshape(b, t, n, k*c))).view(b, t, n, k, c)
        supports = self._supports(dynamic_graph, static_graph)
        mode = self.override_mode or self.mode
        out = [transformed.mean(-2)]
        for source in range(len(self.source_labels)):
            banks = supports[source*native.k_s:(source+1)*native.k_s]
            first = self.slot_messages(banks[0], transformed, native_repeated=True)
            content = first if self.mode == 'POST_MESSAGE' else transformed
            weights = self.router(content, source, mode)
            out.append((first*weights[..., None]).sum(-2))
            # Reuse the first-order score across higher graph orders.
            for graph in banks[1:]:
                message = self.slot_messages(graph, transformed, native_repeated=True)
                out.append((message*weights[..., None]).sum(-2))
            if self._stats is not None:
                self._accumulate(weights, source, phase)
        return native.dropout(native.gcn_updt(torch.cat(out, -1)))

    def start_diagnostics(self):
        self._stats = {}

    def _accumulate(self, weights, source, phase):
        w = weights.detach().double()
        p = (w/self.native.k_t).clamp_min(torch.finfo(w.dtype).tiny)
        key = (phase, source)
        # Keep node and local-window index fixed, aggregate only batch/sample.
        # This separates across-sample variation from fixed node/call effects.
        row = dict(sum=w.sum(0), square=w.square().sum(0),
            entropy=(-(p*p.log()).sum(-1)).sum(),
            coefficient_mass_error_abs_max=(w.sum(-1)-self.native.k_t).abs().max())
        if key not in self._stats:
            self._stats[key] = dict(samples=0, calls=0,
                **{name: torch.zeros_like(value) for name, value in row.items()})
        target = self._stats[key]
        if target['sum'].shape != row['sum'].shape:
            raise ValueError('GW2 diagnostic node/window axes changed within a call phase')
        for name, value in row.items():
            target[name] = torch.maximum(target[name], value) if name.endswith('_max') else target[name]+value
        target['samples'] += len(w)
        target['calls'] += 1

    def finish_diagnostics(self):
        raw, self._stats = self._stats, None
        result = dict(source_labels=list(self.source_labels), calls=0, by_phase={},
            scope='fixed_node_local_window_index_and_call_phase_across_samples',
            interpretation='across_sample_variation_is_not_predictive_utility_or_physical_delay')
        if not raw:
            return result
        source_sums, source_squares, source_counts, source_entropy = [], [], [], []
        for source in range(len(self.source_labels)):
            rows = [row for (phase, s), row in raw.items() if s == source]
            source_sums.append(sum(row['sum'].sum((0, 1)) for row in rows))
            source_squares.append(sum(row['square'].sum((0, 1)) for row in rows))
            source_counts.append(sum(row['samples']*row['sum'].shape[0]*row['sum'].shape[1] for row in rows))
            source_entropy.append(sum(row['entropy'] for row in rows))
        count = torch.tensor(source_counts, device=source_sums[0].device)[:, None]
        mean = torch.stack(source_sums)/count
        result.update(weight_mean=mean.cpu().tolist(),
            weight_std=(torch.stack(source_squares)/count-mean.square()).clamp_min(0).sqrt().cpu().tolist(),
            entropy_mean=(torch.stack(source_entropy)/count.squeeze(-1)).cpu().tolist(),
            calls=sum(row['calls'] for (phase, source), row in raw.items() if source == 0),
            coefficient_mass_error_abs_max=max(float(row['coefficient_mass_error_abs_max']) for row in raw.values()))
        for (phase, source), row in raw.items():
            conditional_mean = row['sum']/row['samples']
            variance = (row['square']/row['samples']-conditional_mean.square()).clamp_min(0)
            result['by_phase'].setdefault(phase, {})[self.source_labels[source]] = dict(
                samples=row['samples'], calls=row['calls'], local_window_positions=row['sum'].shape[0],
                weight_mean=conditional_mean.mean((0, 1)).cpu().tolist(),
                across_sample_std_rms=variance.mean((0, 1)).sqrt().cpu().tolist())
        return result


class GraphMessageIGSTGNN(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        if mode not in ROUTER_MODES:
            raise ValueError('Unknown GW2 training arm')
        self.original, self.mode = original, mode
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon'):
            setattr(self, name, getattr(original, name))
        conv = original.layers[0].dif_layer.localized_st_conv
        if not hasattr(conv, 'k_t'):
            raise ValueError('GW2 requires an unwrapped original first DIF convolution')
        original.layers[0].dif_layer.localized_st_conv = GraphMessageConv(conv, mode, DESIGN['router_hidden'])

    @property
    def selector(self):
        return self.original.layers[0].dif_layer.localized_st_conv

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        self.selector.call_index = 0
        return self.original(history_data, incident_data=incident_data, sensor_data=sensor_data)

    @contextmanager
    def intervention(self, mode):
        if mode not in ('UNIFORM', 'STATIC'):
            raise ValueError('Only predefined GW2 interventions allowed')
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
        raise ValueError('Unknown GW2 arm')
    return original if arm == 'ORIGINAL' else GraphMessageIGSTGNN(original, arm)


def split_state(model):
    if not isinstance(model, GraphMessageIGSTGNN):
        return dict(model.state_dict()), {}
    prefix = 'layers.0.dif_layer.localized_st_conv.'
    native, added = {}, {}
    for key, value in model.original.state_dict().items():
        if key.startswith(prefix+'router.'):
            added[key[len(prefix+'router.'):]] = value
        else:
            native[key.replace(prefix+'native.', prefix)] = value
    return native, added
