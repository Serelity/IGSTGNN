"""GW3: receiver-query / graph-message-key matching in the first DIF block.

Slots are mixed hidden features, not physical lags. The GW2 propagation kernel,
graph orders, self path and native readout are kept. No targets enter routing.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn

from experiments.chronological.original_graph_message import (
    GraphMessageConv, GraphMessageIGSTGNN, split_state)


ARMS = ('FIXED_UNIFORM', 'MESSAGE_SELF', 'RECEIVER_MATCH')
DESIGN = dict(version='GW3', selected_layer=0, matching_dim=16,
    query_receiver='mean_transformed_local_slots',
    query_control='mean_first_order_graph_message_slots',
    key='first_order_graph_message_per_slot',
    score='static_prior_plus_query_dot_key_div_sqrt_dim',
    graph_kernel='unchanged_GW2_packed_repeated_adjacency',
    graph_orders='original_shared_first_order_weights',
    self_path='unchanged', native_layers='only_first_DIF_aggregation_wrapped',
    zero_initialization='key_projection_weight',
    target_inputs=False, new_incident_inputs=False,
    coefficient_sum='K_not_output_or_vehicle_conservation',
    recovery='floating_tolerance_not_bitwise_native_trajectory')


class MatchingRouter(nn.Module):
    def __init__(self, channels, kernel, sources, hidden=16):
        super().__init__()
        if min(channels, kernel, sources, hidden) < 1:
            raise ValueError('Positive GW3 dimensions required')
        self.kernel, self.sources, self.hidden = kernel, sources, hidden
        self.static_logits = nn.Parameter(torch.zeros(sources, kernel))
        self.query = nn.Linear(channels, hidden)
        # A key bias cancels in slot softmax when the query is shared across
        # slots, so do not register an unidentifiable extra parameter.
        self.key = nn.Linear(channels, hidden, bias=False)
        # Zero only one projection: two zero projections would block learning.
        nn.init.zeros_(self.key.weight)

    def forward(self, local, message, source, mode):
        if mode not in (*ARMS, 'UNIFORM', 'STATIC', 'QUERY_SELF'):
            raise ValueError('Unknown GW3 router mode')
        if (local.ndim != 5 or local.shape != message.shape
                or local.shape[-2] != self.kernel or not 0 <= source < self.sources):
            raise ValueError('GW3 local/message must match [B,T,N,K,C]')
        if mode in ('FIXED_UNIFORM', 'UNIFORM'):
            return message.new_ones(message.shape[:-1])
        logits = self.static_logits[source].expand(message.shape[:-1])
        if mode != 'STATIC':
            content = local if mode == 'RECEIVER_MATCH' else message
            query = self.query(content.mean(-2)).unsqueeze(-2)
            keys = self.key(message)
            logits = logits + (query*keys).sum(-1)/math.sqrt(self.hidden)
        p = torch.softmax(logits, -1)
        return 1. + self.kernel*(p-p.new_tensor(1./self.kernel))


class ReceiverMatchConv(GraphMessageConv):
    def __init__(self, native, mode):
        if mode not in ARMS:
            raise ValueError('Unknown GW3 training arm')
        # Inherit only GW2 support/propagation/diagnostic machinery.
        super().__init__(native, 'POST_MESSAGE', DESIGN['matching_dim'])
        device = next(native.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.router = MatchingRouter(native.hidden_dim, native.k_t,
                len(self.source_labels), DESIGN['matching_dim']).to(device)
        self.mode = mode
        if mode == 'FIXED_UNIFORM':
            # Registered for initialization/capacity accounting only; no scoring
            # or router gradient/Adam updates in the fixed training control.
            self.router.requires_grad_(False)

    def forward(self, x, dynamic_graph, static_graph):
        native = self.native
        if x.ndim != 4 or x.shape[1] < native.k_t or x.shape[-1] != native.hidden_dim:
            raise ValueError('GW3 requires native [batch,time,node,channel] input')
        phase = 'history' if self.call_index == 0 else 'forecast_'+str(self.call_index)
        self.call_index += 1
        window = x.unfold(1, native.k_t, 1).permute(0, 1, 2, 4, 3)
        b, t, n, k, c = window.shape
        local = native.activation(native.fc_list_updt(window.reshape(b, t, n, k*c))).view(b, t, n, k, c)
        supports = self._supports(dynamic_graph, static_graph)
        mode = self.override_mode or self.mode
        out = [local.mean(-2)]
        for source in range(len(self.source_labels)):
            banks = supports[source*native.k_s:(source+1)*native.k_s]
            message = self.slot_messages(banks[0], local, native_repeated=True)
            weights = self.router(local, message, source, mode)
            out.append((message*weights[..., None]).sum(-2))
            for graph in banks[1:]:
                higher = self.slot_messages(graph, local, native_repeated=True)
                out.append((higher*weights[..., None]).sum(-2))
            if self._stats is not None:
                self._accumulate(weights, source, phase)
        return native.dropout(native.gcn_updt(torch.cat(out, -1)))


class ReceiverMatchIGSTGNN(GraphMessageIGSTGNN):
    def __init__(self, original, mode):
        nn.Module.__init__(self)
        self.original, self.mode = original, mode
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon'):
            setattr(self, name, getattr(original, name))
        conv = original.layers[0].dif_layer.localized_st_conv
        if not hasattr(conv, 'k_t'):
            raise ValueError('GW3 requires an unwrapped native first DIF block')
        original.layers[0].dif_layer.localized_st_conv = ReceiverMatchConv(conv, mode)

    @contextmanager
    def intervention(self, mode):
        if self.mode == 'FIXED_UNIFORM' or mode not in ('UNIFORM', 'STATIC', 'QUERY_SELF'):
            raise ValueError('Invalid GW3 intervention')
        previous = self.selector.override_mode
        self.selector.override_mode = mode
        try:
            yield self
        finally:
            self.selector.override_mode = previous


def build_variant(original, arm):
    return ReceiverMatchIGSTGNN(original, arm)
