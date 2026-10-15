"""Sparse, incident-conditioned replacement for MIN's localized convolution.

This operator follows the 2026-10-15 data/theory contract.  It is not an
equivalent reparameterization of STLocalizedConv and makes no physical flow
conservation claim.  Graph priors remain report independent.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ConditionalMessagePropagation(nn.Module):
    """Eleven graph/order banks with joint source-position message selection.

    ``condition`` is an already encoded, node-local report tensor [B,N,C].
    ``support`` and ``available`` must be boolean [B,N] and [B].  A disabled
    report path does not inspect any of these inputs.  Token metadata is long
    [T]: kind 0/1 denotes history/generated and block -1/0..3 denotes history
    or generated block.  ``collect`` records only detached scalar diagnostics
    in ``last_details``; the return value is always the hidden tensor.
    """

    def __init__(self, old_conv, adjacency, receiver_chunk_size=4,
                 checkpoint_messages=True):
        super().__init__()
        self.hidden_dim = int(old_conv.hidden_dim)
        self.k_t = int(old_conv.k_t)
        self.k_s = int(old_conv.k_s)
        if self.k_t != 3 or self.k_s != 2:
            raise ValueError('Conditional propagation requires k_t=3 and k_s=2')
        if not (old_conv.use_predefined_graph and old_conv.use_dynamic_hidden_graph
                and old_conv.use_static_hidden_graph and old_conv.num_matric == 11
                and len(old_conv.pre_defined_graph) == 4):
            raise ValueError('Conditional propagation requires the original eleven banks')
        if isinstance(receiver_chunk_size, bool) or not isinstance(receiver_chunk_size, int) or receiver_chunk_size < 1:
            raise ValueError('receiver_chunk_size must be a positive integer')
        if not isinstance(checkpoint_messages, bool):
            raise TypeError('checkpoint_messages must be bool')
        self.receiver_chunk_size = receiver_chunk_size
        self.checkpoint_messages = checkpoint_messages
        adjacency = torch.as_tensor(adjacency).detach().cpu()
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError('adjacency must be square')
        if not adjacency.is_floating_point() or not bool(torch.isfinite(adjacency).all()) or bool((adjacency < 0).any()):
            raise ValueError('adjacency must be finite, floating and nonnegative')
        self.node_num = adjacency.shape[0]
        if not self.node_num:
            raise ValueError('adjacency must not be empty')
        directed = adjacency > 0
        identity = torch.eye(self.node_num, dtype=torch.bool)
        directional = []
        for first in (directed, directed.T):
            second = (first.to(torch.int64) @ first.to(torch.int64)) > 0
            directional.extend((first & ~identity, second & ~identity))
        masks = [identity] + directional + directional + [
            (directional[0] | directional[2]),
            (directional[1] | directional[3])]
        hard_masks = torch.stack(masks)
        rows, sources = torch.where(hard_masks.any(0))
        self.register_buffer('hard_masks', hard_masks)
        self.register_buffer('edge_receivers', rows)
        self.register_buffer('edge_sources', sources)
        self.register_buffer('bank_edge_mask', hard_masks[:, rows, sources])
        counts = torch.bincount(rows, minlength=self.node_num)
        self.register_buffer('edge_offsets', torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0))))
        # Python offsets avoid a GPU synchronization for each receiver chunk.
        self._edge_offsets_cpu = tuple(self.edge_offsets.tolist())
        self.register_load_state_dict_post_hook(self._refresh_edge_offsets)
        for index, graph in enumerate(old_conv.pre_defined_graph):
            self.register_buffer('predefined_{}'.format(index), graph.detach().clone())
        c = self.hidden_dim
        self.source_value = nn.ModuleList(nn.Linear(c, c) for _ in range(self.k_t))
        self.query = nn.Linear(c, c, bias=False)
        self.key = nn.Linear(c, c, bias=False)
        self.position = nn.Linear(11, c, bias=False)
        self.time_factor = nn.Linear(11, c, bias=False)
        self.relation = nn.Linear(2*c, c)
        self.kind_embedding = nn.Embedding(2, 4)
        self.block_embedding = nn.Embedding(5, 4)
        self.event_logit = nn.Linear(c, 1)
        self.event_value = nn.Linear(2*c, c)
        nn.init.zeros_(self.event_logit.weight)
        nn.init.zeros_(self.event_logit.bias)
        nn.init.zeros_(self.event_value.weight)
        nn.init.zeros_(self.event_value.bias)
        self.gcn_updt = deepcopy(old_conv.gcn_updt)
        self.dropout = deepcopy(old_conv.dropout)
        self.beta = 1.0
        self.eta = 1.0
        self.last_details = None
        # Respect the source module's dtype/device without keeping its modules.
        self.to(device=self.gcn_updt.weight.device, dtype=self.gcn_updt.weight.dtype)

    def _refresh_edge_offsets(self, module, incompatible_keys):
        # Topology is serialized, so loading a same-size but different graph
        # must also refresh the Python-only chunk boundaries.
        self._edge_offsets_cpu = tuple(self.edge_offsets.detach().cpu().tolist())

    def _restore_graph(self, graph, batch, x, name, localized):
        if not isinstance(graph, torch.Tensor) or graph.device != x.device or graph.dtype != x.dtype:
            raise ValueError('{} must share X dtype and device'.format(name))
        n = self.node_num
        if localized:
            if graph.ndim == 2 and graph.shape == (n, self.k_t*n):
                restored = graph.reshape(n, self.k_t, n)
                if not torch.equal(restored, restored[:, :1].expand_as(restored)):
                    raise ValueError('{} localized positions must repeat the same graph'.format(name))
                graph = restored[:, 0]
            elif graph.ndim == 3 and graph.shape == (batch, n, self.k_t*n):
                restored = graph.reshape(batch, n, self.k_t, n)
                if not torch.equal(restored, restored[:, :, :1].expand_as(restored)):
                    raise ValueError('{} localized positions must repeat the same graph'.format(name))
                graph = restored[:, :, 0]
            else:
                raise ValueError('{} has an invalid localized graph shape'.format(name))
        elif graph.shape not in ((n, n), (batch, n, n)):
            raise ValueError('{} has an invalid graph shape'.format(name))
        if not bool(torch.isfinite(graph).all()) or bool((graph < 0).any()):
            raise ValueError('{} must be finite and nonnegative'.format(name))
        return graph

    def _graph_weights(self, dynamic_graph, static_graph, batch, x):
        if not isinstance(dynamic_graph, (list, tuple)) or len(dynamic_graph) != 4:
            raise ValueError('dynamic_graph must contain four original localized banks')
        if not isinstance(static_graph, (list, tuple)) or len(static_graph) != 1:
            raise ValueError('static_graph must contain one original base graph')
        graphs = [torch.eye(self.node_num, device=x.device, dtype=x.dtype)]
        graphs += [self._restore_graph(getattr(self, 'predefined_{}'.format(i)), batch, x,
                                      'predefined', True) for i in range(4)]
        graphs += [self._restore_graph(g, batch, x, 'dynamic', True) for g in dynamic_graph]
        base = self._restore_graph(static_graph[0], batch, x, 'static', False)
        graphs += [base, base @ base]
        weights = []
        for i, graph in enumerate(graphs):
            selected = graph[..., self.edge_receivers, self.edge_sources]
            if selected.ndim == 1:
                selected = selected.unsqueeze(0).expand(batch, -1)
            weights.append(selected * self.bank_edge_mask[i].to(x.dtype))
        return torch.stack(weights)

    def _metadata(self, x, token_kind, token_block):
        t = x.shape[1]
        if token_kind is None and token_block is None:
            token_kind = torch.zeros(t, device=x.device, dtype=torch.long)
            token_block = torch.full((t,), -1, device=x.device, dtype=torch.long)
        for name, value in (('token_kind', token_kind), ('token_block', token_block)):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.device != x.device or value.shape != (t,):
                raise ValueError('{} must be long[T] on X device'.format(name))
        if bool(((token_kind < 0) | (token_kind > 1)).any()):
            raise ValueError('token_kind must be HISTORY=0 or GENERATED=1')
        valid = ((token_kind == 0) & (token_block == -1)) | ((token_kind == 1) & (token_block >= 0) & (token_block < 4))
        if not bool(valid.all()):
            raise ValueError('history block must be -1; generated block must be 0..3')
        kinds = token_kind.unfold(0, self.k_t, 1)
        blocks = token_block.unfold(0, self.k_t, 1) + 1
        local = torch.eye(self.k_t, dtype=x.dtype, device=x.device).expand(kinds.shape[0], -1, -1)
        return torch.cat((local, self.kind_embedding(kinds), self.block_embedding(blocks)), -1)

    @staticmethod
    def _bank_reduce(logits, values, prior, receiver_index, receiver_count, temporal_size, self_bank):
        """Grouped softmax and weighted sum; an empty receiver row is zero."""
        b, q, e, kt = logits.shape
        index = receiver_index.repeat_interleave(kt)
        flat_logits = logits.reshape(b, q, e*kt)
        expanded = index.view(1, 1, -1).expand(b, q, -1)
        maxima = logits.new_full((b, q, receiver_count), -torch.inf)
        maxima = maxima.scatter_reduce(2, expanded, flat_logits, reduce='amax', include_self=True)
        safe_maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
        numerator = torch.exp(flat_logits - safe_maxima.gather(2, expanded))
        denominator = logits.new_zeros((b, q, receiver_count)).scatter_add(2, expanded, numerator)
        coefficients = numerator / denominator.gather(2, expanded).clamp_min(torch.finfo(logits.dtype).tiny)
        # Graph priors remain float32 under autocast.  Accumulate in the score
        # dtype instead of scattering float32 products into float16 storage.
        aggregate_values = values.to(logits.dtype)
        weighted = aggregate_values.reshape(b, q, e*kt, -1) * coefficients.unsqueeze(-1)
        result = aggregate_values.new_zeros((b, q, receiver_count, values.shape[-1]))
        result = result.scatter_add(2, expanded.unsqueeze(-1).expand_as(weighted), weighted)
        if self_bank:
            rho = prior.new_ones((b, receiver_count))
        else:
            rho = prior.new_zeros((b, receiver_count)).scatter_add(
                1, receiver_index.unsqueeze(0).expand(b, -1), prior) * temporal_size
        return result * rho[:, None, :, None]

    def _chunk(self, values, queries, keys, times, condition, support, available,
               weights, start, stop, edge_start, edge_stop, logit_enabled, value_enabled):
        receivers = self.edge_receivers[edge_start:edge_stop]
        sources = self.edge_sources[edge_start:edge_stop]
        local_receivers = receivers-start
        u = queries[:, :, receivers, None, :]
        z = keys[:, :, sources, :, :]
        base_score = (u*z).sum(-1) / math.sqrt(self.hidden_dim)
        message_values = values[:, :, sources, :, :]
        e = torch.zeros_like(base_score)
        dv = torch.zeros_like(message_values)
        active_count = torch.zeros((), dtype=torch.int64, device=values.device)
        if condition is not None and (logit_enabled or value_enabled):
            relation = torch.tanh(self.relation(torch.cat((condition[:, receivers], condition[:, sources]), -1)))
            active = available[:, None] & (support[:, receivers] | support[:, sources])
            gamma = active[:, None, :, None, None].to(values.dtype)
            interaction = u*z*times[None, :, None, :, :]*relation[:, None, :, None, :]
            if logit_enabled:
                e = (self.beta * torch.tanh(self.event_logit(interaction)) * gamma).squeeze(-1)
            if value_enabled:
                pair_condition = relation[:, None, :, None, :].expand_as(interaction)
                dv = self.eta * torch.tanh(self.event_value(torch.cat((interaction, pair_condition), -1))) * gamma
            active_count = active.sum(dtype=torch.int64)
        result = []
        selected_values = message_values + dv
        for bank in range(11):
            prior = weights[bank, :, edge_start:edge_stop]
            positive = prior > 0
            # Safe log keeps all strictly positive (including subnormal)
            # priors intact, and gives exactly zero derivatives to zero edges.
            log_prior = prior.masked_fill(~positive, 1).log().masked_fill(~positive, -torch.inf)
            logits = base_score + e + log_prior[:, None, :, None]
            result.append(self._bank_reduce(logits, selected_values, prior,
                                            local_receivers, stop-start, self.k_t, bank == 0))
        projected = self.gcn_updt(torch.cat(result, -1))
        diagnostics = torch.stack((e.abs().amax().to(torch.float64),
                                   dv.abs().amax().to(torch.float64),
                                   active_count.to(torch.float64)))
        return projected, diagnostics

    def forward(self, X, dynamic_graph, static_graph, *, condition=None, support=None,
                available=None, token_kind=None, token_block=None, report_enabled=True,
                logit_enabled=True, value_enabled=True, collect=False):
        self.last_details = None
        for name, flag in (('report_enabled', report_enabled), ('logit_enabled', logit_enabled),
                           ('value_enabled', value_enabled), ('collect', collect)):
            if not isinstance(flag, bool):
                raise TypeError('{} must be bool'.format(name))
        if not isinstance(X, torch.Tensor) or X.ndim != 4 or X.shape[1] < self.k_t or X.shape[2:] != (self.node_num, self.hidden_dim):
            raise ValueError('X must have shape [B,T>=3,N,C]')
        if X.dtype != self.gcn_updt.weight.dtype or X.device != self.gcn_updt.weight.device or not X.is_floating_point():
            raise ValueError('X must share model dtype/device')
        if not bool(torch.isfinite(X).all()):
            raise ValueError('X must be finite')
        b = X.shape[0]
        if not b:
            raise ValueError('X batch must not be empty')
        if not report_enabled or (condition is None and support is None and available is None):
            condition = support = available = None
        else:
            if not isinstance(condition, torch.Tensor) or condition.shape != (b, self.node_num, self.hidden_dim) or condition.dtype != X.dtype or condition.device != X.device or not bool(torch.isfinite(condition).all()):
                raise ValueError('condition must be finite [B,N,C] with X dtype/device')
            if not isinstance(support, torch.Tensor) or support.shape != (b, self.node_num) or support.dtype != torch.bool or support.device != X.device:
                raise ValueError('support must be bool[B,N] on X device')
            if not isinstance(available, torch.Tensor) or available.shape != (b,) or available.dtype != torch.bool or available.device != X.device:
                raise ValueError('available must be bool[B] on X device')
        weights = self._graph_weights(dynamic_graph, static_graph, b, X)
        metadata = self._metadata(X, token_kind, token_block)
        windows = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        values = torch.stack([layer(windows[..., i, :]) for i, layer in enumerate(self.source_value)], -2)
        queries = torch.tanh(self.query(windows[..., -1, :]))
        keys = torch.tanh(self.key(windows) + self.position(metadata)[None, :, None, :, :])
        times = torch.tanh(self.time_factor(metadata))
        chunks, stats = [], []
        for start in range(0, self.node_num, self.receiver_chunk_size):
            stop = min(start+self.receiver_chunk_size, self.node_num)
            edge_start, edge_stop = self._edge_offsets_cpu[start], self._edge_offsets_cpu[stop]
            # Bind ranges now: backward checkpoint recomputation must not use
            # the final loop iteration's receiver indices.
            def compute(v, u, z, t, r, m, a, w, start=start, stop=stop,
                        edge_start=edge_start, edge_stop=edge_stop):
                return self._chunk(v, u, z, t, r, m, a, w, start, stop,
                                   edge_start, edge_stop, logit_enabled, value_enabled)
            args = (values, queries, keys, times, condition, support, available, weights)
            if self.training and self.checkpoint_messages and torch.is_grad_enabled():
                hidden, diagnostic = checkpoint(compute, *args, use_reentrant=False)
            else:
                hidden, diagnostic = compute(*args)
            chunks.append(hidden)
            if collect:
                stats.append(diagnostic.detach())
        hidden = self.dropout(torch.cat(chunks, 2))
        if collect:
            summaries = torch.stack(stats)
            self.last_details = {
                'banks': 11, 'windows': X.shape[1]-self.k_t+1,
                'union_edges': self.edge_sources.numel(),
                'receiver_chunk_size': self.receiver_chunk_size,
                'checkpoint_messages': self.training and self.checkpoint_messages and torch.is_grad_enabled(),
                'statistics': {
                    'event_logit_abs_max': summaries[:, 0].max(),
                    'event_value_abs_max': summaries[:, 1].max(),
                    'active_report_edges': summaries[:, 2].sum().to(torch.int64),
                },
            }
        return hidden
