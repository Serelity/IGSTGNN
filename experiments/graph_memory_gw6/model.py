"""GW6 observed graph-message memory and direct multi-horizon DIF decoding.

The original GW4 module is retained under ``anchor`` with its tensor names
unchanged.  Joint arms replace DIF *forecasting*, not its historical backcast
or the INH branch.  A probe is a separate, frozen-anchor residual experiment.
These implementations are engineering candidates, not measured improvements.
"""
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from experiments.chronological.original_graph_message import GraphMessageConv
from experiments.chronological.original_receiver_residual import (
    ReceiverResidualConv, ReceiverResidualIGSTGNN)


DEFAULT_ARMS = ('RECURRENT_CONTROL', 'POOLED_MEMORY', 'ADDITIVE_MEMORY', 'JOINT_MEMORY')
ARMS = DEFAULT_ARMS + ('FACTORIZED_MEMORY', 'LEAD_ONLY', 'NO_LEAD',
                       'TIME_POOLED', 'SOURCE_POOLED')
DESIGN = dict(version='GW6', stage='joint_or_frozen_probe',
    primary_stage='joint', history_source='same_trained_GW4_RECEIVER_RESIDUAL',
    joint_dif='one_history_graph_propagation_then_direct_actual_horizon_readout_per_layer',
    source_order='native_source_major_order_minor',
    memory_axes='batch_node_graph_source_graph_order_historical_window_channel',
    memory_value='GW4_weighted_slot_sum_in_layer0_native_slot_sum_in_other_layers',
    receiver='last_historical_transformed_receiver_window_mean',
    query_inputs='historical_receiver_origin_available_ICSF_context_known_lead_index',
    joint_query='projected_receiver_context_times_lead_embedding',
    additive_query='projected_receiver_context_plus_lead_embedding',
    no_lead_ablation='same_read_weights_at_all_leads_lead_embedding_only_after_readout',
    factorization='source_times_window_conditional_on_order_times_order_probability',
    preserved='DIF_historical_backcast_INH_native_four_block_TIID_gap_channel_semantics',
    native_inh_and_tiid='compute_native_blocks_then_repeat_each_block_gap_times',
    output_channel='lead_index_mod_gap',
    probe='first_layer_only_zero_output_head_frozen_GW4_exact_recovery',
    node_chunk='exact_softmax_over_all_source_order_window_tokens_for_each_node_chunk',
    joint_freeze='unused_native_DIF_forecast_branches_only',
    cache='per_complete_forward_cleared_on_success_and_exception',
    target_inputs=False, additional_report_inputs=False,
    interpretation='graph_axes_and_slots_are_not_physical_propagation_delays',
    evidence_status='server_verification_and_predictive_training_required')


def _native(conv):
    return conv.native if isinstance(conv, ReceiverResidualConv) else conv


def _source_labels(conv):
    if isinstance(conv, ReceiverResidualConv):
        return tuple(conv.source_labels)
    n = _native(conv)
    if len(n.pre_defined_graph) % n.k_s:
        raise ValueError('Native graph banks must be source-major/order-minor')
    directions = len(n.pre_defined_graph)//n.k_s
    labels = []
    if n.use_predefined_graph:
        labels += ['PREDEFINED_'+str(i) for i in range(directions)]
    if n.use_dynamic_hidden_graph:
        labels += ['DYNAMIC_'+str(i) for i in range(directions)]
    if n.use_static_hidden_graph:
        labels += ['ADAPTIVE_0']
    if not labels or n.num_matric != 1+len(labels)*n.k_s:
        raise ValueError('Unsupported native source-count contract')
    return tuple(labels)


def history_graph_memory(conv, x, dynamic_graph, static_graph):
    """Compute each message once, reuse it for native history and memory.

    Return historical hidden states, bank [B,N,G,O,W,C], receiver [B,N,C].
    For native layers the slotwise contraction changes floating-point reduction
    order relative to the original flattened contraction; tolerances apply.
    The GW4 first-layer operation and reduction order are unchanged.
    """
    native = _native(conv)
    if x.ndim != 4 or x.shape[1] < native.k_t or x.shape[-1] != native.hidden_dim:
        raise ValueError('Historical DIF input must be [batch,time>=kernel,node,channel]')
    labels = _source_labels(conv)
    if isinstance(conv, ReceiverResidualConv):
        conv._phase = 'history'
        conv.call_index += 1
        supports = conv._supports(dynamic_graph, static_graph)
    else:
        supports = []
        if native.use_predefined_graph:
            supports += native.pre_defined_graph
        if native.use_dynamic_hidden_graph:
            supports += dynamic_graph
        if native.use_static_hidden_graph:
            supports += native.get_graph(static_graph)
    if len(supports) != len(labels)*native.k_s:
        raise ValueError('Historical graph count differs from decoder source/order axes')
    window = x.unfold(1, native.k_t, 1).permute(0, 1, 2, 4, 3)
    b, w, n, k, c = window.shape
    transformed = native.activation(native.fc_list_updt(window.reshape(b, w, n, k*c)))
    transformed = transformed.view(b, w, n, k, c)
    receiver = transformed.mean(-2)
    parts, source_banks = [receiver], []
    for source in range(len(labels)):
        banks = supports[source*native.k_s:(source+1)*native.k_s]
        first = GraphMessageConv.slot_messages(banks[0], transformed, native_repeated=True)
        weights = (conv.weights(first, receiver, source)
                   if isinstance(conv, ReceiverResidualConv) else None)
        by_order = []
        for order, graph in enumerate(banks):
            message = first if order == 0 else GraphMessageConv.slot_messages(
                graph, transformed, native_repeated=True)
            value = ((message*weights[..., None]).sum(-2)
                     if weights is not None else message.sum(-2))
            parts.append(value)
            by_order.append(value)
        source_banks.append(torch.stack(by_order, dim=3))  # B,W,N,O,C
        if isinstance(conv, ReceiverResidualConv) and conv._stats is not None:
            conv._accumulate(weights, source, 'history')
    memory = torch.stack(source_banks, dim=3).permute(0, 2, 3, 4, 1, 5)
    hidden = native.dropout(native.gcn_updt(torch.cat(parts, -1)))
    return hidden, memory, receiver[:, -1]


class GraphMemoryDecoder(nn.Module):
    """A node-local decoder; every softmax retains all memory tokens."""
    def __init__(self, channels, sources, orders, windows, horizon, forecast_dim,
                 arm, hidden=32, node_chunk=32, zero_output=False):
        super().__init__()
        if arm not in ARMS[1:] or min(channels, sources, orders, windows, horizon,
                                      forecast_dim, hidden, node_chunk) < 1:
            raise ValueError('Invalid graph-memory decoder arm or dimensions')
        self.arm, self.hidden, self.node_chunk = arm, hidden, node_chunk
        self.channels, self.horizon = channels, horizon
        self.sources, self.orders, self.windows = sources, orders, windows
        self.context = nn.Linear(2*channels, hidden)
        self.key = nn.Linear(channels, hidden, bias=False)
        self.value = nn.Linear(channels, hidden)
        self.lead = nn.Parameter(torch.empty(horizon, hidden))
        self.source_embedding = nn.Parameter(torch.empty(sources, hidden))
        self.order_embedding = nn.Parameter(torch.empty(orders, hidden))
        self.window_embedding = nn.Parameter(torch.empty(windows, hidden))
        # Capacity control: a pooled input, receiver and actual target index
        # pass through an MLP. Unused attention tensors are explicitly frozen.
        self.pooled_mix = nn.Linear(2*hidden, hidden)
        self.output = nn.Linear(hidden, forecast_dim)
        nn.init.normal_(self.lead, mean=1., std=.1)
        for p in (self.source_embedding, self.order_embedding, self.window_embedding):
            nn.init.normal_(p, std=.02)
        if zero_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        if arm == 'POOLED_MEMORY':
            self.key.requires_grad_(False)
            for p in (self.source_embedding, self.order_embedding, self.window_embedding):
                p.requires_grad_(False)
        else:
            self.pooled_mix.requires_grad_(False)
        if arm == 'LEAD_ONLY':
            self.context.requires_grad_(False)
        if arm == 'TIME_POOLED':
            self.window_embedding.requires_grad_(False)
        if arm == 'SOURCE_POOLED':
            self.source_embedding.requires_grad_(False)

    def _query(self, receiver, incident_context):
        context = self.context(torch.cat((receiver, incident_context), -1)).unsqueeze(-2)
        if self.arm == 'LEAD_ONLY':
            return self.lead[None, None].expand(receiver.shape[0], receiver.shape[1], -1, -1)
        if self.arm == 'NO_LEAD':
            return context.expand(-1, -1, self.horizon, -1)
        lead = self.lead[None, None]
        return context+lead if self.arm == 'ADDITIVE_MEMORY' else context*lead

    def _tokens(self, memory):
        if self.arm == 'TIME_POOLED':
            memory = memory.mean(4, keepdim=True)
        if self.arm == 'SOURCE_POOLED':
            memory = memory.mean(2, keepdim=True)
        key = self.key(memory)
        if self.arm != 'SOURCE_POOLED':
            key = key+self.source_embedding[None, None, :, None, None]
        key = key+self.order_embedding[None, None, None, :, None]
        if self.arm != 'TIME_POOLED':
            key = key+self.window_embedding[None, None, None, None, :]
        return key, self.value(memory)

    def _weights(self, query, key):
        scores = torch.einsum('bnhd,bngowd->bnhgow', query, key)/sqrt(self.hidden)
        if self.arm == 'FACTORIZED_MEMORY':
            # alpha(G,W|O)=alpha(G|O)*alpha(W|O), with a separate alpha(O).
            source = torch.softmax(scores.mean(-1), dim=-2).unsqueeze(-1)
            window = torch.softmax(scores.mean(-3), dim=-1).unsqueeze(-3)
            order = torch.softmax(scores.mean((-3, -1)), dim=-1).unsqueeze(-2).unsqueeze(-1)
            return source*window*order
        return torch.softmax(scores.flatten(-3), dim=-1).view_as(scores)

    def attention_weights(self, memory, receiver, incident_context=None):
        """Unchunked weights for small engineering/property tests only."""
        if self.arm == 'POOLED_MEMORY':
            raise ValueError('The pooled MLP has no attention weights')
        context = torch.zeros_like(receiver) if incident_context is None else incident_context
        key, _ = self._tokens(memory)
        return self._weights(self._query(receiver, context), key)

    def forward(self, memory, receiver, incident_context=None, diagnostic=None):
        if memory.ndim != 6 or tuple(memory.shape[2:]) != (
                self.sources, self.orders, self.windows, self.channels):
            raise ValueError('Memory must be [batch,node,source,order,window,channel]')
        if receiver.shape != memory.shape[:2]+(self.channels,):
            raise ValueError('Receiver state axes differ from graph memory')
        incident_context = torch.zeros_like(receiver) if incident_context is None else incident_context
        if incident_context.shape != receiver.shape:
            raise ValueError('ICSF context must match historical receiver state')
        chunks = []
        for start in range(0, memory.shape[1], self.node_chunk):
            stop = min(start+self.node_chunk, memory.shape[1])
            bank = memory[:, start:stop]
            query = self._query(receiver[:, start:stop], incident_context[:, start:stop])
            if self.arm == 'POOLED_MEMORY':
                pooled = self.value(bank.mean((2, 3, 4))).unsqueeze(-2).expand_as(query)
                read = torch.tanh(self.pooled_mix(torch.cat((pooled, query), -1)))
            else:
                key, value = self._tokens(bank)
                probability = self._weights(query, key)
                read = torch.einsum('bnhgow,bngowd->bnhd', probability, value)
                if self.arm == 'NO_LEAD':
                    # Isolate lead-conditioned *selection*: preserve a learned
                    # 12-lead readout even when all lead weights are identical.
                    read = read+self.lead[None, None]
                if diagnostic is not None:
                    with torch.no_grad():
                        p = probability.detach().float().flatten(-3)
                        diagnostic['reads'] += p.numel()//p.shape[-1]
                        diagnostic['entropy_sum'] += float((-(p*p.clamp_min(1e-30).log()).sum(-1)).sum())
                        diagnostic['mass_error_abs_max'] = max(diagnostic['mass_error_abs_max'],
                            float((p.sum(-1)-1.).abs().max()))
            chunks.append(self.output(read))
        return torch.cat(chunks, dim=1).permute(0, 2, 1, 3)


class GraphMemoryIGSTGNN(nn.Module):
    def __init__(self, anchor, arm, stage='joint', hidden=32, node_chunk=32):
        super().__init__()
        if not isinstance(anchor, ReceiverResidualIGSTGNN) or anchor.arm != 'RECEIVER_RESIDUAL':
            raise ValueError('GW6 requires the trained GW4 RECEIVER_RESIDUAL anchor')
        if arm not in ARMS or stage not in ('joint', 'probe'):
            raise ValueError('Unknown GW6 arm or stage')
        if stage == 'probe' and arm == 'RECURRENT_CONTROL':
            raise ValueError('Frozen probe control has no trainable parameters; evaluate GW4 directly')
        if hidden < 1 or node_chunk < 1:
            raise ValueError('Hidden dimension and node chunk must be positive')
        self.anchor, self.arm, self.stage = anchor, arm, stage
        self.hidden, self.node_chunk = hidden, node_chunk
        self._cache, self._diagnostics = {}, None
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon'):
            setattr(self, name, getattr(anchor, name))
        original = self.original
        self.gap = int(original._model_args['gap'])
        if self.horizon != self.seq_len or self.horizon % self.gap:
            raise ValueError('GW6 requires native horizon=sequence_length divisible by gap')
        self.blocks = self.horizon//self.gap
        self.decoders = nn.ModuleList()
        # No new modules for the exact recurrent control.
        if arm != 'RECURRENT_CONTROL':
            device = next(anchor.parameters()).device
            devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                       if device.type == 'cuda' else [])
            with torch.random.fork_rng(devices=devices):
                remaining = self.seq_len
                for index, layer in enumerate(original.layers):
                    conv = layer.dif_layer.localized_st_conv
                    native = _native(conv)
                    windows = remaining-native.k_t+1
                    if windows < 1:
                        raise ValueError('A DIF layer has no complete historical window')
                    forecast_dim = layer.dif_layer.forecast_branch.forecast_fc.out_features
                    self.decoders.append(GraphMemoryDecoder(native.hidden_dim, len(_source_labels(conv)),
                        native.k_s, windows, self.horizon, forecast_dim if stage == 'joint' else self.output_dim,
                        arm, hidden, node_chunk, zero_output=stage == 'probe').to(device))
                    if stage == 'probe':
                        break
                    remaining = windows
        # Restore joint training explicitly: GW4's own train() deliberately
        # freezes/evaluates its base. Do not call that override in joint mode.
        self.anchor.requires_grad_(stage == 'joint')
        if stage == 'joint' and arm != 'RECURRENT_CONTROL':
            for layer in original.layers:
                layer.dif_layer.forecast_branch.requires_grad_(False)
        self.train(False)

    @property
    def original(self):
        return self.anchor.base.original

    @property
    def cache_is_clear(self):
        return not self._cache

    def clear_cache(self):
        self._cache.clear()

    def assert_cache_clear(self):
        if self._cache:
            raise AssertionError('GW6 retained batch-specific graph memory')
        return True

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_counts(self):
        return dict(total=self.param_num(), active=sum(p.numel() for p in self.parameters() if p.requires_grad),
            anchor=sum(p.numel() for p in self.anchor.parameters() if p.requires_grad),
            decoder=sum(p.numel() for p in self.decoders.parameters() if p.requires_grad),
            inactive_native_dif_forecast=sum(p.numel() for layer in self.original.layers
                for p in layer.dif_layer.forecast_branch.parameters() if not p.requires_grad))

    def train(self, mode=True):
        # nn.Module.train recurses into GW4's freezing override; then explicitly
        # set the original backbone to the stage's intended mode.
        super().train(mode)
        if self.stage == 'joint':
            self.anchor.training = mode
            self.anchor.base.train(mode)
        else:
            self.anchor.eval()
        self.decoders.train(mode)
        return self

    def start_diagnostics(self):
        self._diagnostics = dict(version='GW6', arm=self.arm, stage=self.stage,
            complete_forwards=0, failed_forwards=0, layers={})
        self.anchor.start_diagnostics()

    def diagnostics(self):
        result, self._diagnostics = self._diagnostics, None
        result = result or dict(version='GW6', arm=self.arm, stage=self.stage,
            complete_forwards=0, failed_forwards=0, layers={})
        for row in result['layers'].values():
            row['entropy_mean'] = row['entropy_sum']/row['reads'] if row['reads'] else None
        result['anchor'] = self.anchor.diagnostics()
        result['cache_clear'] = self.cache_is_clear
        result['utility_caveat'] = 'Attention entropy and variation do not establish predictive utility'
        return result

    def _row(self, index, memory):
        if self._diagnostics is None:
            return None
        row = self._diagnostics['layers'].setdefault(str(index), dict(reads=0, entropy_sum=0.,
            mass_error_abs_max=0., history_calls=0, memory_shape=list(memory.shape),
            source_labels=list(_source_labels(self.original.layers[index].dif_layer.localized_st_conv))))
        row['history_calls'] += 1
        return row

    def _joint(self, history_data, incident_data, sensor_data):
        original = self.original
        x, up, down, tod, dow = original._prepare_inputs(history_data)
        x = original.embedding(x)
        incident_context = None
        if incident_data is not None:
            x, incident_context = original.icsf_module(x, incident_data, sensor_data,
                tod[:, -1, 0], dow[:, -1, 0])
        static_graph, dynamic_graph = original._graph_constructor(node_embedding_u=up,
            node_embedding_d=down, history_data=x, time_in_day_feat=tod, day_in_week_feat=dow)
        dif_values, inh_values = [], []
        self.anchor.selector.call_index = 0
        for index, layer in enumerate(original.layers):
            gated = layer.estimation_gate(up, down, tod, dow, x)
            dif = layer.dif_layer
            hidden, memory, receiver = history_graph_memory(dif.localized_st_conv,
                gated, dynamic_graph, static_graph)
            self._cache[index] = memory
            direct = self.decoders[index](memory, receiver, incident_context, self._row(index, memory))
            backcast = dif.backcast_branch(hidden)
            residual = dif.residual_decompose(x[:, -backcast.shape[1]:], backcast)
            x, inherent = layer.inh_layer(residual)
            if inherent.shape[1] != self.blocks:
                raise ValueError('Native INH output block count changed')
            dif_values.append(direct)
            inh_values.append(inherent)
            del self._cache[index]
        # Preserve the original four-block Gaussian decay before expansion.
        inherent = sum(inh_values)
        if incident_context is not None:
            inherent = original.tiid_module(inherent, incident_context)
        combined = sum(dif_values)+inherent.repeat_interleave(self.gap, dim=1)
        channels = original.out_fc_2(F.relu(original.out_fc_1(F.relu(combined))))
        return self.select_gap_channels(channels, self.gap)

    @staticmethod
    def select_gap_channels(channels, gap):
        """Lead h uses original within-block output channel h % gap."""
        if channels.ndim != 4 or channels.shape[-1] != gap:
            raise ValueError('Output channels must equal the original native gap')
        channel = torch.arange(channels.shape[1], device=channels.device).remainder(gap)
        index = channel.view(1, -1, 1, 1).expand(*channels.shape[:-1], 1)
        return channels.gather(-1, index)

    def _probe(self, history_data, incident_data, sensor_data):
        conv = self.anchor.selector
        had_instance_forward = 'forward' in conv.__dict__
        previous = conv.__dict__.get('forward')
        ordinary_forward = conv.forward
        hook = None

        def capture(x, dynamic, static):
            if conv.call_index != 0:
                return ordinary_forward(x, dynamic, static)
            result, memory, receiver = history_graph_memory(conv, x, dynamic, static)
            self._cache['memory'] = memory.detach()
            self._cache['receiver'] = receiver.detach()
            return result

        def context_hook(module, inputs, output):
            self._cache['context'] = output[1].detach()

        try:
            conv.forward = capture
            if incident_data is not None:
                hook = self.original.icsf_module.register_forward_hook(context_hook)
            with torch.no_grad():
                baseline = self.anchor(history_data, incident_data=incident_data, sensor_data=sensor_data)
            if 'memory' not in self._cache:
                raise RuntimeError('Probe failed to capture the first historical DIF call')
            correction = self.decoders[0](self._cache['memory'], self._cache['receiver'],
                self._cache.get('context'), self._row(0, self._cache['memory']))
            return baseline+correction
        finally:
            if had_instance_forward:
                conv.forward = previous
            else:
                delattr(conv, 'forward')
            if hook is not None:
                hook.remove()

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        # label is accepted solely for author-engine compatibility. It is never
        # read by any query, message, graph, context or prediction computation.
        if self._cache:
            self.clear_cache()
            raise RuntimeError('Stale or reentrant graph-memory forward')
        try:
            if self.arm == 'RECURRENT_CONTROL':
                result = self.anchor(history_data, incident_data=incident_data, sensor_data=sensor_data)
            elif self.stage == 'probe':
                result = self._probe(history_data, incident_data, sensor_data)
            else:
                result = self._joint(history_data, incident_data, sensor_data)
            if self._diagnostics is not None:
                self._diagnostics['complete_forwards'] += 1
            return result
        except BaseException:
            if self._diagnostics is not None:
                self._diagnostics['failed_forwards'] += 1
            raise
        finally:
            self.clear_cache()
