"""GW8: joint receiver--sender--slot transport inside the first native DIF.

The new value update enters the executed native ``gcn_updt`` result before its
existing dropout, and therefore participates in historical backcast and native
forecast recurrence. No prediction head, memory decoder, trend correction, or
extra dropout is added. Raw slots are gated hidden states; generated forecast
slots are not observed timestamps or physical propagation delays.
"""
from math import sqrt

import torch
from torch import nn

from experiments.chronological.original_receiver_residual import ReceiverResidualConv
from experiments.graph_memory_gw6.model import GraphMemoryIGSTGNN


ARMS = ('R_CONTINUE', 'R_LOCAL_KERNEL', 'R_SEPARABLE_KERNEL', 'R_JOINT_KERNEL')
DESIGN = dict(version='GW8', anchor='certified_GW7_R_CONTINUE_BEST',
    stage='joint_native_recurrent_hidden_transport', selected_layer=0,
    placement='native_gcn_updt_output_before_existing_native_dropout',
    query='receiver_latest_raw_gated_hidden_slot',
    key_value='sender_each_raw_pre_fc_gated_hidden_slot',
    scope='all_first_layer_DIF_history_and_native_forecast_calls',
    prior='mean_of_valid_source_first_order_nonnegative_offdiagonal_row_priors',
    sparsity='receiver_topk_prior_then_retained_mass_renormalization',
    joint='softmax_over_sender_edge_times_slot',
    separable='same_joint_sender_marginal_times_same_joint_slot_marginal',
    local='same_parameters_attention_over_receiver_own_raw_slots',
    output='bias_free_zero_initialized_hidden_value_projection',
    all_native_parameters_trainable=True, target_inputs=False, future_inputs=False,
    additional_report_inputs=False, parameter_initialization_preserves_rng=True,
    physical_delay_claim=False, novelty_status='unestablished',
    evidence_status='server_tests_and_predeclared_equal_budget_screen_required')


def factorize_joint(probability):
    """Retain sender and slot marginals while removing their coupling."""
    if probability.ndim < 2:
        raise ValueError('Joint probabilities require final [edge,slot] axes')
    return probability.sum(-1, keepdim=True)*probability.sum(-2, keepdim=True)


def build_transport_prior(supports, batch, nodes, kernel, sources, orders, edges):
    """Return top sender IDs, differentiable retained prior, and tensor stats.

    Supports use the authenticated native [receiver,slot*sender] layout. Each
    first-order source repeats one adjacency across slots. Graph-order banks
    after first order are deliberately not part of this new transport prior.
    Nonfinite graphs fail explicitly; negative weights are clipped to zero.
    """
    if min(batch, nodes, kernel, sources, orders, edges) < 1:
        raise ValueError('Transport prior dimensions must be positive')
    if len(supports) != sources*orders:
        raise ValueError('Native source-major/order-minor support count changed')
    combined, valid_sources = None, None
    for source in range(sources):
        graph = supports[source*orders]
        if (not torch.is_tensor(graph) or graph.ndim not in (2, 3)
                or tuple(graph.shape[-2:]) != (nodes, kernel*nodes)
                or (graph.ndim == 3 and graph.shape[0] != batch)):
            raise ValueError('Native first-order graph shape differs')
        if not graph.is_floating_point() or not bool(torch.isfinite(graph).all()):
            raise ValueError('Transport graph must be finite floating point')
        base = graph[..., :nodes]
        for slot in range(1, kernel):
            if not torch.equal(base, graph[..., slot*nodes:(slot+1)*nodes]):
                raise ValueError('First-order native adjacency is not repeated across slots')
        if base.ndim == 2:
            base = base.unsqueeze(0).expand(batch, -1, -1)
        diagonal = torch.eye(nodes, device=base.device, dtype=torch.bool)
        base = base.clamp_min(0).masked_fill(diagonal, 0.)
        mass = base.sum(-1, keepdim=True)
        if not bool(torch.isfinite(mass).all()):
            raise FloatingPointError('Transport graph row normalization overflowed')
        valid = mass > 0
        safe_mass = torch.where(valid, mass, torch.ones_like(mass))
        normalized = torch.where(valid, base/safe_mass, torch.zeros_like(base))
        combined = normalized if combined is None else combined+normalized
        valid_sources = valid.to(base.dtype) if valid_sources is None else valid_sources+valid
    combined = combined/valid_sources.clamp_min(1.)
    count = min(edges, nodes-1)
    values, indices = torch.topk(combined, count, dim=-1, sorted=True)
    retained = values.sum(-1, keepdim=True)
    retained_valid = retained > 0
    safe_retained = torch.where(retained_valid, retained, torch.ones_like(retained))
    prior = torch.where(retained_valid, values/safe_retained, torch.zeros_like(values))
    stats = dict(receiver_rows=batch*nodes,
        nonempty_rows=(valid_sources.squeeze(-1) > 0).sum(),
        valid_source_rows=valid_sources.sum(),
        retained_mass_sum=retained.sum(),
        retained_edges=(values > 0).sum(), selected_edges=count,
        retained_mass=retained.squeeze(-1),
        nonempty_row_mask=valid_sources.squeeze(-1) > 0,
        zero_neighbor_rows=(valid_sources.squeeze(-1) == 0).sum())
    return indices, prior, stats


class RawLagKernel(nn.Module):
    """Equal-capacity local, separable, and joint hidden value transport."""
    def __init__(self, channels, kernel, arm, hidden=16, node_chunk=32):
        super().__init__()
        if arm not in ARMS[1:] or min(channels, kernel, hidden, node_chunk) < 1:
            raise ValueError('Invalid raw-slot kernel arm or dimensions')
        self.arm, self.channels, self.kernel = arm, channels, kernel
        self.hidden, self.node_chunk = hidden, node_chunk
        self.query = nn.Linear(channels, hidden, bias=False)
        self.key = nn.Linear(channels, hidden, bias=False)
        self.value = nn.Linear(channels, hidden, bias=False)
        self.lag_bias = nn.Parameter(torch.zeros(kernel))
        self.output = nn.Linear(hidden, channels, bias=False)
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def _probability(scores, valid):
        # Softmax of an entirely masked row would be NaN with -inf. A finite
        # sentinel plus explicit masking makes its probability exactly zero.
        masked = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        probability = torch.softmax(masked.flatten(-2), -1).view_as(scores)
        probability = probability*valid.to(probability.dtype)
        mass = probability.sum((-2, -1), keepdim=True)
        safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
        return probability/safe_mass

    def attention_weights(self, window, prior=None):
        """Unchunked actual-arm [B,W,N,E,K] probabilities for small tests only."""
        if (not torch.is_tensor(window) or window.ndim != 5
                or tuple(window.shape[-2:]) != (self.kernel, self.channels)
                or min(window.shape[:3]) < 1 or not bool(torch.isfinite(window).all())):
            raise ValueError('Raw slots must be finite [batch,window,node,kernel,channel]')
        b, w, n, _, _ = window.shape
        q, key = self.query(window[..., -1, :]), self.key(window)
        if self.arm == 'R_LOCAL_KERNEL':
            keys = key.unsqueeze(3)
            logits = (q[..., None, None, :]*keys).sum(-1)/sqrt(self.hidden)+self.lag_bias
            valid = torch.ones_like(logits, dtype=torch.bool)
        else:
            if not isinstance(prior, (tuple, list)) or len(prior) != 2:
                raise ValueError('Graph kernels require (sender_indices, prior)')
            indices, mass = prior
            if (indices.ndim != 3 or indices.shape[:2] != (b, n) or mass.shape != indices.shape
                    or indices.dtype != torch.long or indices.device != window.device
                    or mass.device != window.device or not bool(torch.isfinite(mass).all())
                    or bool((mass < 0).any()) or bool((indices < 0).any())
                    or bool((indices >= n).any())):
                raise ValueError('Invalid receiver-specific sender indices or prior')
            if indices.shape[-1] == 0:
                return window.new_zeros((b, w, n, 0, self.kernel))
            batches = torch.arange(b, device=window.device)[:, None, None]
            keys = key.transpose(1, 2)[batches, indices].permute(0, 3, 1, 2, 4, 5)
            p = mass[:, None, :, :, None]
            valid = (p > 0).expand(b, w, n, indices.shape[-1], self.kernel)
            logits = ((q[..., None, None, :]*keys).sum(-1)/sqrt(self.hidden)
                +self.lag_bias+p.clamp_min(torch.finfo(p.dtype).tiny).log())
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Raw-slot attention logits became nonfinite')
        joint = self._probability(logits, valid)
        return factorize_joint(joint) if self.arm == 'R_SEPARABLE_KERNEL' else joint

    def forward(self, window, prior=None, diagnostic=None):
        if (not torch.is_tensor(window) or window.ndim != 5
                or tuple(window.shape[-2:]) != (self.kernel, self.channels)
                or min(window.shape[:3]) < 1 or not bool(torch.isfinite(window).all())):
            raise ValueError('Raw slots must be finite [batch,window,node,kernel,channel]')
        b, w, n, _, _ = window.shape
        query = self.query(window[..., -1, :])
        key, value = self.key(window), self.value(window)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Raw-slot value projection became nonfinite')
        outputs = []
        if self.arm != 'R_LOCAL_KERNEL':
            if not isinstance(prior, (tuple, list)) or len(prior) != 2:
                raise ValueError('Graph kernels require (sender_indices, prior)')
            indices, mass = prior
            if (indices.ndim != 3 or indices.shape[:2] != (b, n) or mass.shape != indices.shape
                    or indices.dtype != torch.long or indices.device != window.device
                    or mass.device != window.device or not bool(torch.isfinite(mass).all())
                    or bool((mass < 0).any()) or bool((indices < 0).any())
                    or bool((indices >= n).any())):
                raise ValueError('Invalid receiver-specific sender indices or prior')
            if indices.shape[-1] == 0:
                return self.output(value.sum(-2)*0.)
            by_node_key, by_node_value = key.transpose(1, 2), value.transpose(1, 2)
            batches = torch.arange(b, device=window.device)[:, None, None]
        for start in range(0, n, self.node_chunk):
            stop = min(start+self.node_chunk, n)
            q = query[:, :, start:stop, None, None]
            if self.arm == 'R_LOCAL_KERNEL':
                keys = key[:, :, start:stop].unsqueeze(3)
                values = value[:, :, start:stop].unsqueeze(3)
                valid = torch.ones((*keys.shape[:-1],), device=window.device, dtype=torch.bool)
                logits = (q*keys).sum(-1)/sqrt(self.hidden)+self.lag_bias
            else:
                selected = indices[:, start:stop]
                # B,Q,E,W,K,D -> B,W,Q,E,K,D. No N-by-N attention tensor.
                keys = by_node_key[batches, selected].permute(0, 3, 1, 2, 4, 5)
                values = by_node_value[batches, selected].permute(0, 3, 1, 2, 4, 5)
                p = mass[:, None, start:stop, :, None]
                valid = (p > 0).expand(b, w, stop-start, selected.shape[-1], self.kernel)
                logits = ((q*keys).sum(-1)/sqrt(self.hidden)+self.lag_bias
                    +p.clamp_min(torch.finfo(p.dtype).tiny).log())
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Raw-slot attention logits became nonfinite')
            joint = self._probability(logits, valid)
            probability = factorize_joint(joint) if self.arm == 'R_SEPARABLE_KERNEL' else joint
            read = (probability[..., None]*values).sum((-3, -2))
            outputs.append(self.output(read))
            if diagnostic is not None:
                with torch.no_grad():
                    positive = probability > 0
                    entropy = -(probability*probability.clamp_min(
                        torch.finfo(probability.dtype).tiny).log()).sum((-2, -1))
                    active = positive.any((-2, -1))
                    diagnostic['attention_rows'] += int(active.sum())
                    diagnostic['entropy_sum'] += float(entropy.sum())
                    if bool(active.any()):
                        error = (probability.sum((-2, -1))[active]-1.).abs().max()
                        diagnostic['probability_mass_error_abs_max'] = max(
                            diagnostic['probability_mass_error_abs_max'], float(error))
        return torch.cat(outputs, dim=2)


class LagTransportIGSTGNN(nn.Module):
    def __init__(self, anchor, arm, hidden=16, edges=8, node_chunk=32):
        super().__init__()
        if (not isinstance(anchor, GraphMemoryIGSTGNN) or anchor.arm != 'RECURRENT_CONTROL'
                or anchor.stage != 'joint'):
            raise ValueError('GW8 requires a joint GW6 RECURRENT_CONTROL anchor')
        if arm not in ARMS or min(hidden, edges, node_chunk) < 1:
            raise ValueError('Invalid GW8 arm or dimensions')
        if not isinstance(anchor.anchor.selector, ReceiverResidualConv):
            raise ValueError('GW8 requires the authenticated first GW4 receiver convolution')
        self.anchor, self.arm = anchor, arm
        self.hidden, self.edges, self.node_chunk = hidden, edges, node_chunk
        self._cache, self._handles = {}, []
        self._in_forward, self._diagnostics = False, None
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon'):
            setattr(self, name, getattr(anchor, name))
        native = self.selector.native
        self.sources, self.orders = len(self.selector.source_labels), native.k_s
        self.channels, self.slots = native.hidden_dim, native.k_t
        self.kernel = None
        mode = anchor.training
        if arm != 'R_CONTINUE':
            parameter = next(anchor.parameters())
            device, dtype = parameter.device, parameter.dtype
            devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == 'cuda' else [])
            with torch.random.fork_rng(devices=devices):
                self.kernel = RawLagKernel(self.channels, self.slots, arm, hidden,
                    node_chunk).to(device=device, dtype=dtype)
        self.anchor.requires_grad_(True)
        self.train(mode)

    @property
    def selector(self):
        return self.anchor.anchor.selector

    @property
    def original(self):
        return self.anchor.original

    @property
    def cache_is_clear(self):
        return not self._cache and not self._handles and not self._in_forward and self.anchor.cache_is_clear

    def assert_cache_clear(self):
        if not self.cache_is_clear:
            raise AssertionError('GW8 retained batch tensors, hooks, or an active forward')
        return True

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_counts(self):
        added = 0 if self.kernel is None else sum(p.numel() for p in self.kernel.parameters() if p.requires_grad)
        return dict(total=self.param_num(), active=sum(p.numel() for p in self.parameters() if p.requires_grad),
            anchor=sum(p.numel() for p in self.anchor.parameters() if p.requires_grad),
            added=added, decoder=0, inactive_native_dif_forecast=0)

    def train(self, mode=True):
        super().train(mode)
        self.anchor.train(mode)
        return self

    def start_diagnostics(self):
        self._diagnostics = dict(version='GW8', arm=self.arm, complete_forwards=0,
            failed_forwards=0, transport_calls=0, history_calls=0, forecast_calls=0,
            attention_rows=0, entropy_sum=0., probability_mass_error_abs_max=0.,
            receiver_rows=0, nonempty_rows=0, valid_source_rows=0.,
            retained_mass_sum=0., retained_edges=0, hidden_update_square_sum=0.,
            hidden_update_elements=0, hidden_update_abs_max=0.)
        self.anchor.start_diagnostics()

    def diagnostics(self):
        result, self._diagnostics = self._diagnostics, None
        result = result or dict(version='GW8', arm=self.arm, complete_forwards=0, failed_forwards=0)
        if 'attention_rows' in result:
            rows, active = result['receiver_rows'], result['nonempty_rows']
            result['entropy_mean'] = result['entropy_sum']/result['attention_rows'] if result['attention_rows'] else None
            result['retained_mass_mean_nonempty'] = result['retained_mass_sum']/active if active else None
            result['neighbor_coverage_fraction'] = active/rows if rows else None
            elements = result['hidden_update_elements']
            result['hidden_update_rms'] = sqrt(result['hidden_update_square_sum']/elements) if elements else None
        result['anchor'] = self.anchor.diagnostics()
        result['cache_clear'] = self.cache_is_clear
        result['utility_caveat'] = 'Kernel coupling and support coverage do not establish utility or physical delay'
        return result

    def _before_selector(self, module, inputs):
        if self._cache:
            raise RuntimeError('GW8 transport update was not consumed by native gcn_updt')
        if len(inputs) != 3 or not torch.is_tensor(inputs[0]):
            raise RuntimeError('First DIF convolution input contract changed')
        x, dynamic, static = inputs
        if x.ndim != 4 or x.shape[1] < self.slots or x.shape[2:] != (self.node_num, self.channels):
            raise RuntimeError('First DIF raw hidden input shape changed')
        window = x.unfold(1, self.slots, 1).permute(0, 1, 2, 4, 3)
        prior, stats = None, None
        if self.arm != 'R_LOCAL_KERNEL':
            indices, mass, stats = build_transport_prior(self.selector._supports(dynamic, static),
                x.shape[0], self.node_num, self.slots, self.sources, self.orders, self.edges)
            prior = (indices, mass)
        delta = self.kernel(window, prior, self._diagnostics)
        self._cache['delta'] = delta
        if self._diagnostics is not None:
            row = self._diagnostics
            row['transport_calls'] += 1
            row['history_calls' if self.selector.call_index == 0 else 'forecast_calls'] += 1
            if stats is not None:
                for name in ('receiver_rows', 'nonempty_rows', 'valid_source_rows', 'retained_mass_sum', 'retained_edges'):
                    row[name] += float(stats[name]) if torch.is_tensor(stats[name]) else stats[name]
            with torch.no_grad():
                observed = delta.detach().double()
                row['hidden_update_square_sum'] += float(observed.square().sum())
                row['hidden_update_elements'] += observed.numel()
                row['hidden_update_abs_max'] = max(row['hidden_update_abs_max'], float(observed.abs().max()))

    def _after_gcn(self, module, inputs, output):
        if 'delta' not in self._cache:
            raise RuntimeError('Native gcn_updt executed without its corresponding GW8 update')
        delta = self._cache.pop('delta')
        if not torch.is_tensor(output) or output.shape != delta.shape:
            raise RuntimeError('Native gcn_updt output shape differs from raw-slot update')
        return output+delta

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        # Reject a nested invocation without disturbing the enclosing forward.
        if self._in_forward:
            raise RuntimeError('Reentrant GW8 forward is not supported')
        if self._cache or self._handles:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._cache.clear()
            raise RuntimeError('Stale GW8 forward cache or hooks')
        self._in_forward = True
        try:
            if self.kernel is not None:
                self._handles.append(self.selector.register_forward_pre_hook(self._before_selector))
                self._handles.append(self.selector.native.gcn_updt.register_forward_hook(self._after_gcn))
            result = self.anchor(history_data, incident_data=incident_data, sensor_data=sensor_data)
            if self._cache:
                raise RuntimeError('The last GW8 update was not consumed by native convolution')
            if self._diagnostics is not None:
                self._diagnostics['complete_forwards'] += 1
            return result
        except BaseException:
            if self._diagnostics is not None:
                self._diagnostics['failed_forwards'] += 1
            raise
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._cache.clear()
            self._in_forward = False
