"""GW11: origin-state residuals at native pre-sigmoid estimation gates.

The complete trained GW9 SUM model is retained. Temporary hooks change only
the output of each executed gate's second Linear; the native sigmoid and
history-length slice still execute once in their original positions.
"""
from math import pi, sqrt

import torch
from torch import nn

from experiments.branch_readout_gw9.model import BranchReadoutIGSTGNN


ARMS = ('SUM_CONTINUE', 'STATIC_GATE', 'STATE_READOUT', 'STATE_GATE')
ADAPTER_ARMS = ARMS[1:]
GATE_ARMS = ('STATIC_GATE', 'STATE_GATE')
LAYERS = 5
HISTORY = 12
FEATURES = 7
DESIGN = dict(version='GW11', source='certified_trained_GW9_SUM_CONTROL_BEST',
    candidate='STATE_GATE', source_training='joint_all_original_parameters',
    added_parameters=725, adapter_count=5, hidden=16,
    placement='executed_native_estimation_gate_fc2_output_before_sigmoid_and_native_crop',
    state_inputs=['normalized_last_flow', 'log1p_mean_absolute_normalized_history_change',
        'origin_tod_sin', 'origin_tod_cos', 'origin_dow_sin', 'origin_dow_cos',
        'position_0_to_11_over_11'],
    static_inputs='node_embedding_u_mean,node_embedding_d_mean,same_clocks_and_position',
    readout='mean_of_five_same_state_adapters_added_after_complete_GW9_SUM_forward',
    zero_output_recovers='trained_GW9_SUM_CONTROL', target_inputs=False,
    novelty_status='unestablished', evidence_status='requires_server_tests_and_equal_budget_screen')


class StateAdapter(nn.Module):
    """One 145-parameter adapter, with an exactly zero initial output."""
    def __init__(self, hidden=16):
        super().__init__()
        if hidden != 16:
            raise ValueError('GW11 fixes each adapter hidden dimension to 16')
        self.input = nn.Linear(FEATURES, hidden)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        return self.output(self.activation(self.input(features)))


class StateGateIGSTGNN(nn.Module):
    """Nested trained SUM model with same-capacity state/location controls."""
    def __init__(self, source, arm, hidden=16):
        super().__init__()
        if (not isinstance(source, BranchReadoutIGSTGNN)
                or source.arm != 'SUM_CONTROL' or source.stage != 'joint'
                or source.head is None):
            raise ValueError('GW11 requires the complete GW9 SUM_CONTROL joint source with its trained head')
        if arm not in ARMS or hidden != 16:
            raise ValueError('Invalid GW11 arm or fixed hidden dimension')
        source.assert_cache_clear()
        if not all(parameter.requires_grad for parameter in source.parameters()):
            raise ValueError('All GW11 source parameters must retain joint-training flags')
        original = source.original
        if (source.seq_len != HISTORY or source.horizon != HISTORY
                or source.output_dim != 1 or len(original.layers) != LAYERS
                or int(original._model_args['num_feat']) != 1):
            raise ValueError('GW11 requires five native layers, 12 history/forecast positions and one traffic feature')
        for layer in original.layers:
            gate = layer.estimation_gate.fully_connected_layer_2
            if not isinstance(gate, nn.Linear) or gate.out_features != 1:
                raise ValueError('Native estimation gate fc2 contract changed')
        for embedding in (original.node_emb_u, original.node_emb_d):
            if embedding.ndim != 2 or embedding.shape[0] != source.node_num:
                raise ValueError('Native node embedding axis differs')
        self.source, self.arm, self.hidden = source, arm, hidden
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon'):
            setattr(self, name, getattr(source, name))
        self.adapters = nn.ModuleList()
        self._cache, self._handles = {}, []
        self._in_forward, self._diagnostics = False, None
        # Preserve source tensors, parameter flags and heterogeneous module modes.
        # The fork protects the caller's CPU RNG and the source device's CUDA RNG.
        mode = source.training
        if arm in ADAPTER_ARMS:
            parameter = next(source.parameters())
            device, dtype = parameter.device, parameter.dtype
            devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == 'cuda' else [])
            with torch.random.fork_rng(devices=devices):
                self.adapters.extend(StateAdapter(hidden).to(device=device, dtype=dtype)
                    for _ in range(LAYERS))
            self.adapters.train(mode)
        self.training = mode

    @property
    def original(self):
        return self.source.original

    @property
    def cache_is_clear(self):
        return (not self._cache and not self._handles and not self._in_forward
            and self.source.cache_is_clear)

    def assert_cache_clear(self):
        if not self.cache_is_clear:
            raise AssertionError('GW11 retained hook handles, features, active forward or source cache')
        self.source.assert_cache_clear()
        return True

    def param_num(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_counts(self):
        added = sum(parameter.numel() for parameter in self.adapters.parameters())
        active_added = sum(parameter.numel() for parameter in self.adapters.parameters()
            if parameter.requires_grad)
        return dict(total=self.param_num(), active=sum(parameter.numel() for parameter
            in self.parameters() if parameter.requires_grad),
            source=sum(parameter.numel() for parameter in self.source.parameters()
                if parameter.requires_grad), added=added, active_added=active_added,
            gate_adapters=added if self.arm in GATE_ARMS else 0,
            readout_adapters=added if self.arm == 'STATE_READOUT' else 0, decoder=0)

    def state_features(self, history_data):
        """Seven origin-available features, with no label or future input."""
        parameter = next(self.source.parameters())
        if (not torch.is_tensor(history_data) or history_data.ndim != 4
                or tuple(history_data.shape[1:]) != (HISTORY, self.node_num, 3)
                or history_data.shape[0] < 1 or history_data.device != parameter.device
                or history_data.dtype != parameter.dtype):
            raise ValueError('GW11 expects source-dtype [batch,12,node,3] history on the source device')
        batch, _, nodes, _ = history_data.shape
        if self.arm == 'STATIC_GATE':
            first = self.original.node_emb_u.mean(dim=-1).unsqueeze(0).expand(batch, nodes)
            second = self.original.node_emb_d.mean(dim=-1).unsqueeze(0).expand(batch, nodes)
        else:
            traffic = history_data[..., 0]
            first = traffic[:, -1]
            second = torch.log1p((traffic[:, 1:]-traffic[:, :-1]).abs().mean(dim=1))
        tod, dow = history_data[:, -1, :, 1], history_data[:, -1, :, 2]
        origin = (first, second, torch.sin((2*pi)*tod), torch.cos((2*pi)*tod),
            torch.sin((2*pi)*dow), torch.cos((2*pi)*dow))
        expanded = [value[:, None, :].expand(batch, HISTORY, nodes) for value in origin]
        coordinate = (torch.arange(HISTORY, dtype=history_data.dtype,
            device=history_data.device)/(HISTORY-1))[None, :, None].expand(batch, HISTORY, nodes)
        return torch.stack([*expanded, coordinate], dim=-1)

    def start_diagnostics(self):
        self._diagnostics = dict(version='GW11', arm=self.arm, complete_forwards=0,
            failed_forwards=0, source_forward_calls=0, adapter_calls=[0]*LAYERS,
            gate_hook_calls=[0]*LAYERS, added_parameters=self.trainable_parameter_counts()['added'],
            layer_logit_delta=[dict(square_sum=0., elements=0, abs_max=0.) for _ in range(LAYERS)],
            readout_delta=dict(square_sum=0., elements=0, abs_max=0.))
        self.source.start_diagnostics()

    @staticmethod
    def _accumulate(record, value):
        observed = value.detach().double()
        record['square_sum'] += float(observed.square().sum())
        record['elements'] += observed.numel()
        record['abs_max'] = max(record['abs_max'], float(observed.abs().max()))

    def diagnostics(self):
        result, self._diagnostics = self._diagnostics, None
        result = result or dict(version='GW11', arm=self.arm, complete_forwards=0, failed_forwards=0)
        for record in (*result.get('layer_logit_delta', []),
                *([result['readout_delta']] if 'readout_delta' in result else [])):
            record['rms'] = sqrt(record['square_sum']/record['elements']) if record['elements'] else None
        result['source'] = self.source.diagnostics()
        result['cache_clear'] = self.cache_is_clear
        result['utility_caveat'] = 'State dependence and nonzero logit changes do not establish predictive utility'
        return result

    def _gate_delta(self, index, output):
        order = self._cache['gate_order']
        if index != len(order):
            raise RuntimeError('GW11 estimation gates must execute exactly once in native layer order')
        features = self._cache['features']
        expected = (*features.shape[:3], 1)
        if (not torch.is_tensor(output) or tuple(output.shape) != expected
                or output.device != features.device or output.dtype != features.dtype):
            raise RuntimeError('Native gate fc2 must return the unchanged [batch,12,node,1] logit axis')
        order.append(index)
        delta = self.adapters[index](features)
        if delta.shape != output.shape:
            raise RuntimeError('GW11 gate adapter output axis differs from native logits')
        if self._diagnostics is not None:
            self._diagnostics['adapter_calls'][index] += 1
            self._diagnostics['gate_hook_calls'][index] += 1
            self._accumulate(self._diagnostics['layer_logit_delta'][index], delta)
        return output+delta

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        # A rejected nested call must leave the enclosing call's hooks intact.
        if self._in_forward:
            raise RuntimeError('Reentrant GW11 forward is not supported')
        if self._cache or self._handles:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._cache.clear()
            raise RuntimeError('Stale GW11 feature cache or hooks')
        self._in_forward = True
        try:
            if self.arm in ADAPTER_ARMS:
                self._cache['features'] = self.state_features(history_data)
            if self.arm in GATE_ARMS:
                self._cache['gate_order'] = []
                for index, layer in enumerate(self.original.layers):
                    def observe(module, inputs, output, index=index):
                        return self._gate_delta(index, output)
                    self._handles.append(layer.estimation_gate.fully_connected_layer_2.register_forward_hook(observe))
            if self._diagnostics is not None:
                self._diagnostics['source_forward_calls'] += 1
            result = self.source(history_data, incident_data=incident_data, sensor_data=sensor_data)
            if self.arm in GATE_ARMS and self._cache['gate_order'] != list(range(LAYERS)):
                raise RuntimeError('GW11 did not observe all five native estimation gates')
            if self.arm == 'STATE_READOUT':
                deltas = [adapter(self._cache['features']) for adapter in self.adapters]
                delta = sum(deltas)/LAYERS
                if delta.shape != result.shape:
                    raise RuntimeError('GW11 readout and complete source forecast axes differ')
                if self._diagnostics is not None:
                    for index in range(LAYERS):
                        self._diagnostics['adapter_calls'][index] += 1
                    self._accumulate(self._diagnostics['readout_delta'], delta)
                result = result+delta
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
