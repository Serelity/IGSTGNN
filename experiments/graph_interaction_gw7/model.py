"""GW7: retain the native recurrent predictor and learn a graph-memory increment.

Only inputs to an already executed native first-layer ``gcn_updt`` are observed.
No native contraction, return value, dropout, or forecast recurrence is replaced.
The interaction projection is a latent feature restriction, not conservation or
a guarantee against bias drift. Predictive claims require server experiments.
"""
from math import sqrt

import torch
from torch import nn

from experiments.chronological.original_receiver_residual import ReceiverResidualConv
from experiments.graph_memory_gw6.model import GraphMemoryIGSTGNN


ARMS = ('R_CONTINUE', 'R_MLP', 'R_MEMORY', 'R_INTERACTION_MEMORY')
DESIGN = dict(version='GW7', stage='joint_recurrent_plus_output_increment',
    anchor='same_certified_GW6_RECURRENT_CONTROL_epoch20',
    preserved='all_native_DIF_recurrences_INH_TIID_output_head',
    history_source='first_GW4_gcn_updt_input_observed_without_recomputation',
    memory_axes='batch_node_source_order_window_channel',
    ordinary_banks='P_W_M_and_(I-P_W)_M',
    interaction_banks='(I-P_G)(I-P_O)_applied_to_both_ordinary_banks',
    stable_bank_windows=1, query='last_historical_receiver_ICSF_times_known_lead',
    reader='two_independent_all_token_softmax_readers_no_new_dropout',
    value_bias=False, output_bias=False, output_initialization='zero_weight',
    mlp='pooled_M_and_mean_absolute_window_change_plus_same_query_two_layer_MLP',
    mlp_capacity='nearest_positive_integer_hidden_width_to_two_reader_parameter_count',
    residual_domain='same_normalized_output_domain_as_anchor',
    all_anchor_parameters_trainable=True, target_inputs=False,
    future_inputs=False, additional_report_inputs=False,
    cache='hooks_and_batch_tensors_cleared_on_success_and_exception',
    caveat='latent_interaction_projection_is_not_output_mean_conservation',
    evidence_status='server_tests_and_controlled_training_required')


def memory_banks(memory, interaction=False):
    """Return window-mean and window-change banks, optionally centered on G/O.

    Axes are [B,N,G,O,W,C]. The stable bank retains a singleton W axis. No
    broadcast mean is materialized across W, and both arms have identical shapes.
    """
    if memory.ndim != 6 or min(memory.shape) < 1:
        raise ValueError('Memory must have nonempty [B,N,G,O,W,C] axes')
    stable = memory.mean(4, keepdim=True)
    changing = memory-stable
    if interaction:
        stable = stable-stable.mean(2, keepdim=True)
        stable = stable-stable.mean(3, keepdim=True)
        changing = changing-changing.mean(2, keepdim=True)
        changing = changing-changing.mean(3, keepdim=True)
    return stable, changing


class MemoryReader(nn.Module):
    def __init__(self, channels, sources, orders, windows, hidden):
        super().__init__()
        self.hidden = hidden
        self.key = nn.Linear(channels, hidden, bias=False)
        self.value = nn.Linear(channels, hidden, bias=False)
        self.source_embedding = nn.Parameter(torch.empty(sources, hidden))
        self.order_embedding = nn.Parameter(torch.empty(orders, hidden))
        # A singleton window embedding shifts every token key equally and
        # cancels from softmax; do not register mathematically unused weights.
        if windows > 1:
            self.window_embedding = nn.Parameter(torch.empty(windows, hidden))
        else:
            self.register_parameter('window_embedding', None)
        for p in (self.source_embedding, self.order_embedding, self.window_embedding):
            if p is not None:
                nn.init.normal_(p, std=.02)

    def forward(self, bank, query, diagnostic=None):
        key = (self.key(bank)+self.source_embedding[None, None, :, None, None]
            +self.order_embedding[None, None, None, :, None])
        if self.window_embedding is not None:
            key = key+self.window_embedding[None, None, None, None, :]
        scores = torch.einsum('bnhd,bngowd->bnhgow', query, key)/sqrt(self.hidden)
        probability = torch.softmax(scores.flatten(-3), -1).view_as(scores)
        value = self.value(bank)
        if diagnostic is not None:
            with torch.no_grad():
                p = probability.detach().float().flatten(-3)
                diagnostic['reads'] += p.numel()//p.shape[-1]
                diagnostic['entropy_sum'] += float((-(p*p.clamp_min(1e-30).log()).sum(-1)).sum())
                diagnostic['mass_error_abs_max'] = max(diagnostic['mass_error_abs_max'],
                    float((p.sum(-1)-1.).abs().max()))
        return torch.einsum('bnhgow,bngowd->bnhd', probability, value)


class ResidualMemory(nn.Module):
    def __init__(self, channels, sources, orders, windows, horizon, output_dim,
                 arm, hidden=32, node_chunk=32):
        super().__init__()
        if arm not in ARMS[1:] or min(channels, sources, orders, windows,
                horizon, output_dim, hidden, node_chunk) < 1:
            raise ValueError('Invalid residual memory arm or dimensions')
        self.arm, self.channels, self.horizon = arm, channels, horizon
        self.sources, self.orders, self.windows = sources, orders, windows
        self.hidden, self.node_chunk = hidden, node_chunk
        self.context = nn.Linear(2*channels, hidden)
        self.lead = nn.Parameter(torch.empty(horizon, hidden))
        nn.init.normal_(self.lead, mean=1., std=.1)
        self.readers = nn.ModuleList()
        self.mlp = None
        self.mlp_width = None
        # Every parameter in the capacity control participates in its forward.
        # A signed full mean of the changing bank would be identically zero;
        # its mean absolute value instead retains historical variation.
        reader_parameters = 4*channels*hidden+hidden*(2*sources+2*orders+(windows if windows > 1 else 0))
        if arm == 'R_MLP':
            denominator = 2*channels+3*hidden+1
            self.mlp_width = max(1, int((reader_parameters-2*hidden)/denominator+.5))
            self.mlp = nn.Sequential(nn.Linear(2*channels+hidden, self.mlp_width),
                nn.Tanh(), nn.Linear(self.mlp_width, 2*hidden), nn.Tanh())
        else:
            self.readers.extend([MemoryReader(channels, sources, orders, 1, hidden),
                MemoryReader(channels, sources, orders, windows, hidden)])
        self.output = nn.Linear(2*hidden, output_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, memory, receiver, incident_context=None, diagnostic=None):
        if memory.ndim != 6 or tuple(memory.shape[2:]) != (
                self.sources, self.orders, self.windows, self.channels):
            raise ValueError('Memory differs from declared source/order/window/channel axes')
        if receiver.shape != memory.shape[:2]+(self.channels,):
            raise ValueError('Receiver axes differ from memory')
        context = torch.zeros_like(receiver) if incident_context is None else incident_context
        if context.shape != receiver.shape:
            raise ValueError('ICSF context must match historical receiver state')
        stable, changing = memory_banks(memory, self.arm == 'R_INTERACTION_MEMORY')
        outputs = []
        for start in range(0, memory.shape[1], self.node_chunk):
            stop = min(start+self.node_chunk, memory.shape[1])
            if diagnostic is not None:
                # Preserve float64 sum-of-squares semantics without materializing
                # a float64 copy of the complete batch/source/window bank.
                with torch.no_grad():
                    for name, bank in (('raw', memory), ('stable', stable), ('changing', changing)):
                        value = bank[:, start:stop]
                        diagnostic[name+'_square_sum'] += float(value.detach().double().square().sum())
                        diagnostic[name+'_elements'] += value.numel()
            query = self.context(torch.cat((receiver[:, start:stop], context[:, start:stop]), -1))
            query = query.unsqueeze(-2)*self.lead[None, None]
            if self.arm == 'R_MLP':
                level = stable[:, start:stop].mean((2, 3, 4))
                variation = changing[:, start:stop].abs().mean((2, 3, 4))
                pooled = torch.cat((level, variation), -1).unsqueeze(-2)
                pooled = pooled.expand(-1, -1, self.horizon, -1)
                read = self.mlp(torch.cat((pooled, query), -1))
            else:
                read = torch.cat((self.readers[0](stable[:, start:stop], query, diagnostic),
                    self.readers[1](changing[:, start:stop], query, diagnostic)), -1)
            outputs.append(self.output(read))
        return torch.cat(outputs, 1).permute(0, 2, 1, 3)


class InteractionMemoryIGSTGNN(nn.Module):
    def __init__(self, anchor, arm, hidden=32, node_chunk=32):
        super().__init__()
        if (not isinstance(anchor, GraphMemoryIGSTGNN) or anchor.arm != 'RECURRENT_CONTROL'
                or anchor.stage != 'joint'):
            raise ValueError('GW7 requires a joint GW6 RECURRENT_CONTROL anchor')
        if arm not in ARMS or hidden < 1 or node_chunk < 1:
            raise ValueError('Invalid GW7 arm or dimensions')
        selector = anchor.anchor.selector
        if not isinstance(selector, ReceiverResidualConv):
            raise ValueError('The authenticated first GW4 receiver convolution is required')
        self.anchor, self.arm = anchor, arm
        self.hidden, self.node_chunk = hidden, node_chunk
        self._cache, self._handles = {}, []
        self._in_forward, self._diagnostics = False, None
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon'):
            setattr(self, name, getattr(anchor, name))
        native = selector.native
        self.sources, self.orders = len(selector.source_labels), native.k_s
        self.windows, self.channels = self.seq_len-native.k_t+1, native.hidden_dim
        if self.windows < 1 or native.num_matric != 1+self.sources*self.orders:
            raise ValueError('Unexpected first historical gcn_updt memory layout')
        self.residual = None
        if arm != 'R_CONTINUE':
            parameter = next(anchor.parameters())
            device, dtype = parameter.device, parameter.dtype
            devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == 'cuda' else [])
            with torch.random.fork_rng(devices=devices):
                self.residual = ResidualMemory(self.channels, self.sources, self.orders,
                    self.windows, self.horizon, self.output_dim, arm, hidden, node_chunk
                    ).to(device=device, dtype=dtype)
        self.anchor.requires_grad_(True)
        self.train(False)

    @property
    def original(self):
        return self.anchor.original

    @property
    def cache_is_clear(self):
        return not self._cache and not self._handles and not self._in_forward and self.anchor.cache_is_clear

    def assert_cache_clear(self):
        if not self.cache_is_clear:
            raise AssertionError('GW7 retained batch tensors, hooks, or an active forward')
        return True

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_counts(self):
        added = 0 if self.residual is None else sum(p.numel() for p in self.residual.parameters() if p.requires_grad)
        return dict(total=self.param_num(), active=sum(p.numel() for p in self.parameters() if p.requires_grad),
            anchor=sum(p.numel() for p in self.anchor.parameters() if p.requires_grad),
            decoder=added, added=added, inactive_native_dif_forecast=0,
            mlp_width=None if self.residual is None else self.residual.mlp_width)

    def train(self, mode=True):
        super().train(mode)
        # GW6 joint control explicitly bypasses GW4's frozen-backbone override.
        self.anchor.train(mode)
        return self

    def start_diagnostics(self):
        self._diagnostics = dict(version='GW7', arm=self.arm, complete_forwards=0,
            failed_forwards=0, history_captures=0, reads=0, entropy_sum=0., mass_error_abs_max=0.,
            raw_square_sum=0., raw_elements=0, stable_square_sum=0., stable_elements=0,
            changing_square_sum=0., changing_elements=0, components={})
        self.anchor.start_diagnostics()

    def diagnostics(self):
        result, self._diagnostics = self._diagnostics, None
        result = result or dict(version='GW7', arm=self.arm, complete_forwards=0, failed_forwards=0)
        if 'reads' in result:
            result['entropy_mean'] = result['entropy_sum']/result['reads'] if result['reads'] else None
            for name in ('raw', 'stable', 'changing'):
                n = result[name+'_elements']
                result[name+'_rms'] = sqrt(result[name+'_square_sum']/n) if n else None
            for row in result['components'].values():
                row['mean'] = row['sum']/row['elements']
                row['rms'] = sqrt(row['square_sum']/row['elements'])
        result['anchor'] = self.anchor.diagnostics()
        result['cache_clear'] = self.cache_is_clear
        result['utility_caveat'] = 'Latent interaction energy and attention do not establish predictive utility'
        return result

    def _capture_history(self, module, inputs):
        if 'memory' in self._cache:
            return
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError('First gcn_updt input is not a tensor')
        parts = inputs[0]
        p = 1+self.sources*self.orders
        if parts.ndim != 4 or tuple(parts.shape[1:]) != (
                self.windows, self.node_num, p*self.channels):
            raise RuntimeError('First gcn_updt call is not the expected historical layout')
        b, w, n, _ = parts.shape
        parts = parts.reshape(b, w, n, p, self.channels)
        self._cache['receiver'] = parts[:, -1, :, 0]
        self._cache['memory'] = parts[..., 1:, :].reshape(
            b, w, n, self.sources, self.orders, self.channels).permute(0, 2, 3, 4, 1, 5)
        if self._diagnostics is not None:
            self._diagnostics['history_captures'] += 1

    def _capture_context(self, module, inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError('ICSF no longer returns its historical context')
        self._cache['context'] = output[1]

    def _component_diagnostics(self, components):
        if self._diagnostics is None:
            return
        with torch.no_grad():
            for name, value in components.items():
                value = value.detach().double()
                row = self._diagnostics['components'].setdefault(name,
                    dict(sum=0., square_sum=0., elements=0, negative=0))
                row['sum'] += float(value.sum())
                row['square_sum'] += float(value.square().sum())
                row['elements'] += value.numel()
                row['negative'] += int((value < 0).sum())

    def _execute(self, history_data, incident_data, sensor_data, components):
        # A rejected nested call must not erase the enclosing call's cache/hooks.
        if self._in_forward:
            raise RuntimeError('Reentrant GW7 forward is not supported')
        if self._cache or self._handles:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._cache.clear()
            raise RuntimeError('Stale GW7 forward cache or hooks')
        self._in_forward = True
        try:
            if self.residual is not None:
                self._handles.append(self.anchor.anchor.selector.native.gcn_updt.register_forward_pre_hook(
                    self._capture_history))
                self._handles.append(self.original.icsf_module.register_forward_hook(self._capture_context))
            base = self.anchor(history_data, incident_data=incident_data, sensor_data=sensor_data)
            if self.residual is None:
                total = base
                correction = torch.zeros_like(base) if components or self._diagnostics is not None else None
            else:
                if 'memory' not in self._cache:
                    raise RuntimeError('The first historical graph memory was not observed')
                correction = self.residual(self._cache['memory'], self._cache['receiver'],
                    self._cache.get('context'), self._diagnostics)
                if correction.shape != base.shape:
                    raise RuntimeError('Residual prediction shape differs from native output')
                total = base+correction
            result = dict(base=base, correction=correction, total=total)
            self._component_diagnostics(result)
            if self._diagnostics is not None:
                self._diagnostics['complete_forwards'] += 1
            return result if components else total
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

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        # Accepted for author-engine compatibility; label never enters prediction.
        return self._execute(history_data, incident_data, sensor_data, False)

    def capture_components(self, history_data, label=None, incident_data=None, sensor_data=None):
        """One native forward; returned tensors retain autograd, caches do not."""
        return self._execute(history_data, incident_data, sensor_data, True)
