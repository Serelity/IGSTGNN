"""GW4: three frozen-GW2 logit-residual controls, with no new target inputs.

The receiver/message arms have equal active capacity. STATIC has fewer active
parameters. Slots are mixed latent features, not observed propagation delays.
"""
from contextlib import contextmanager
from math import sqrt

import torch
from torch import nn

from experiments.chronological.original_graph_message import GraphMessageConv, GraphMessageIGSTGNN


ARMS = ('STATIC_RESIDUAL', 'MESSAGE_RESIDUAL', 'RECEIVER_RESIDUAL')
DESIGN = dict(version='GW4', stage='frozen_anchor_logit_residual', selected_layer=0,
    anchor='certified_GW2_POST_MESSAGE_best', hidden=16,
    score='old_GW2_logits_plus_source_slot_bias_plus_query_dot_message_key',
    receiver_query='mean_of_native_transformed_receiver_slots',
    message_query='mean_of_first_order_source_messages',
    key='first_order_source_message', share_across_sources=True,
    share_across_graph_orders=True, initialization='zero_bias_zero_key_random_query',
    source_gating=False, self_path='unchanged', old_parameters_frozen=True,
    old_modules_eval=True, additional_report_inputs=False, target_inputs=False,
    forecast_scope='all_first_layer_DIF_calls_including_autoregression',
    recovery='same_GW2_reduction_path_floating_tolerance',
    capacity='MESSAGE_and_RECEIVER_active_match_STATIC_only_bias_active')
RESIDUAL_PREFIX = 'original.layers.0.dif_layer.localized_st_conv.residual.'


class LogitResidual(nn.Module):
    def __init__(self, channels, kernel, sources, arm, hidden=16):
        super().__init__()
        if arm not in ARMS or min(channels, kernel, sources, hidden) < 1:
            raise ValueError('Invalid GW4 arm or dimensions')
        self.arm, self.hidden = arm, hidden
        self.bias = nn.Parameter(torch.zeros(sources, kernel))
        self.query = nn.Linear(channels, hidden)
        # A common linear-key bias cancels in the slot softmax.
        self.key = nn.Linear(channels, hidden, bias=False)
        nn.init.zeros_(self.key.weight)
        if arm == 'STATIC_RESIDUAL':
            self.query.requires_grad_(False)
            self.key.requires_grad_(False)

    def forward(self, message, receiver, source, setting=None):
        delta = self.bias[source].expand(message.shape[:-1])
        if self.arm == 'STATIC_RESIDUAL' or setting == 'STATIC_ONLY':
            return delta, None, None
        context = (receiver if self.arm == 'RECEIVER_RESIDUAL' and setting != 'QUERY_SELF'
                   else message.mean(-2))
        query, key = self.query(context), self.key(message)
        return delta + (query.unsqueeze(-2)*key).sum(-1)/sqrt(self.hidden), query, key


class ReceiverResidualConv(GraphMessageConv):
    def __init__(self, anchor_conv, arm):
        # Keep the authenticated GW2 objects and their state-dict names.
        nn.Module.__init__(self)
        if not isinstance(anchor_conv, GraphMessageConv) or anchor_conv.mode != 'POST_MESSAGE':
            raise ValueError('GW4 requires a GW2 POST_MESSAGE convolution')
        self.native, self.router = anchor_conv.native, anchor_conv.router
        self.mode, self.directions = anchor_conv.mode, anchor_conv.directions
        self.source_labels = anchor_conv.source_labels
        self.arm, self.override_mode = arm, None
        self._stats, self._residual_stats, self.call_index = None, None, 0
        device = next(self.native.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.residual = LogitResidual(self.native.hidden_dim, self.native.k_t,
                len(self.source_labels), arm, DESIGN['hidden']).to(device)

    def weights(self, message, receiver, source):
        if self.override_mode == 'BASE':
            return self.router(message, source, 'POST_MESSAGE')
        # This is the exact old scorer, including its centered feature input.
        contrast = message-message.mean(-2, keepdim=True)
        features = torch.cat((message, contrast), -1)
        logits = self.router.static_logits[source].expand(message.shape[:-1])
        logits = logits + self.router.score_head(torch.tanh(self.router.encoder(features))).squeeze(-1)
        delta, query, key = self.residual(message, receiver, source, self.override_mode)
        probability = torch.softmax(logits+delta, -1)
        weights = 1. + self.native.k_t*(probability-probability.new_tensor(1./self.native.k_t))
        if self._residual_stats is not None:
            self._record_residual(weights, self.router(message, source, 'POST_MESSAGE'),
                delta, query, key, source)
        return weights

    def forward(self, x, dynamic_graph, static_graph):
        native = self.native
        if x.ndim != 4 or x.shape[1] < native.k_t or x.shape[-1] != native.hidden_dim:
            raise ValueError('GW4 input must be [batch,time>=kernel,node,native_channels]')
        self._phase = 'history' if self.call_index == 0 else 'forecast_'+str(self.call_index)
        self.call_index += 1
        window = x.unfold(1, native.k_t, 1).permute(0, 1, 2, 4, 3)
        b, t, n, k, c = window.shape
        transformed = native.activation(native.fc_list_updt(window.reshape(b, t, n, k*c))).view(b, t, n, k, c)
        receiver = transformed.mean(-2)
        supports, out = self._supports(dynamic_graph, static_graph), [receiver]
        for source in range(len(self.source_labels)):
            banks = supports[source*native.k_s:(source+1)*native.k_s]
            first = self.slot_messages(banks[0], transformed, native_repeated=True)
            weights = self.weights(first, receiver, source)
            out.append((first*weights[..., None]).sum(-2))
            for graph in banks[1:]:
                message = self.slot_messages(graph, transformed, native_repeated=True)
                out.append((message*weights[..., None]).sum(-2))
            if self._stats is not None:
                self._accumulate(weights, source, self._phase)
        return native.dropout(native.gcn_updt(torch.cat(out, -1)))

    def start_diagnostics(self):
        super().start_diagnostics()
        self._residual_stats = {}

    def _record_residual(self, weights, old_weights, delta, query, key, source):
        w, old, d = (v.detach().double() for v in (weights, old_weights, delta))
        p = w/self.native.k_t
        tag = (self._phase, source)
        row = self._residual_stats.setdefault(tag, dict(count=0, delta_abs_sum=0.,
            weight_change_abs_sum=0., delta_abs_max=0., probability_max=0.,
            near_one_hot_count=0, slots=0, query_norm_sum=0., query_count=0,
            key_norm_sum=0., key_count=0))
        row['count'] += d.numel()
        row['slots'] += p[..., 0].numel()
        row['delta_abs_sum'] += float(d.abs().sum())
        row['weight_change_abs_sum'] += float((w-old).abs().sum())
        row['delta_abs_max'] = max(row['delta_abs_max'], float(d.abs().max()))
        row['probability_max'] = max(row['probability_max'], float(p.max()))
        row['near_one_hot_count'] += int((p.max(-1).values >= .99).sum())
        if query is not None:
            qnorm, knorm = query.detach().double().norm(dim=-1), key.detach().double().norm(dim=-1)
            row['query_norm_sum'] += float(qnorm.sum())
            row['query_count'] += qnorm.numel()
            row['key_norm_sum'] += float(knorm.sum())
            row['key_count'] += knorm.numel()

    def finish_diagnostics(self):
        result = super().finish_diagnostics()
        raw, self._residual_stats = self._residual_stats or {}, None
        result.update(version='GW4', anchor_frozen=True, residual_by_phase={},
            call_phase_caveat='forecast_call_index_is_not_actual_forecast_horizon',
            utility_caveat='weight_variation_norm_or_entropy_is_not_predictive_utility')
        for (phase, source), row in raw.items():
            result['residual_by_phase'].setdefault(phase, {})[self.source_labels[source]] = dict(
                logit_delta_abs_mean=row['delta_abs_sum']/row['count'],
                logit_delta_abs_max=row['delta_abs_max'],
                weight_change_abs_mean=row['weight_change_abs_sum']/row['count'],
                slot_probability_max=row['probability_max'],
                near_one_hot_fraction=row['near_one_hot_count']/row['slots'],
                query_norm_mean=row['query_norm_sum']/row['query_count'] if row['query_count'] else None,
                key_norm_mean=row['key_norm_sum']/row['key_count'] if row['key_count'] else None)
        return result


class ReceiverResidualIGSTGNN(nn.Module):
    def __init__(self, anchor, arm):
        super().__init__()
        if arm not in ARMS or not isinstance(anchor, GraphMessageIGSTGNN) or anchor.mode != 'POST_MESSAGE':
            raise ValueError('GW4 needs an unmodified GW2 POST_MESSAGE model and a declared arm')
        self.base, self.arm = anchor, arm
        self.base.requires_grad_(False)
        conv = ReceiverResidualConv(anchor.selector, arm)
        self.base.original.layers[0].dif_layer.localized_st_conv = conv
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon'):
            setattr(self, name, getattr(anchor, name))
        self.train(False)

    @property
    def selector(self):
        return self.base.selector

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def train(self, mode=True):
        super().train(mode)
        # The author engine calls model.train() each epoch. Preserve frozen
        # dropout/buffers while keeping autograd through all downstream ops.
        self.base.eval()
        self.selector.residual.train(mode)
        return self

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        return self.base(history_data, incident_data=incident_data, sensor_data=sensor_data)

    @contextmanager
    def intervention(self, setting):
        if setting not in ('BASE', 'STATIC_ONLY', 'QUERY_SELF') or (
                setting == 'QUERY_SELF' and self.arm != 'RECEIVER_RESIDUAL'):
            raise ValueError('Unknown or inapplicable GW4 intervention')
        previous = self.selector.override_mode
        self.selector.override_mode = setting
        try:
            yield self
        finally:
            self.selector.override_mode = previous

    def start_diagnostics(self):
        self.selector.start_diagnostics()

    def diagnostics(self):
        return self.selector.finish_diagnostics()


def split_state(state):
    """Map unchanged tensors back to the certified GW2 state-dict identity."""
    frozen, added = {}, {}
    prefix = 'base.'+RESIDUAL_PREFIX
    for name, value in state.items():
        if name.startswith(prefix):
            added[name[len(prefix):]] = value
        elif name.startswith('base.'):
            frozen[name[len('base.'):]] = value
        else:
            raise ValueError('Unexpected GW4 state entry: '+name)
    return frozen, added
