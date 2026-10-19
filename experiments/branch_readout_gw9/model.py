"""GW9 preserves the complete R forecast and tests information in D - I.

Temporary observation hooks read outputs from the one executed native forward.
The new readout receives the actual pre-ReLU TIID output as S when TIID runs;
otherwise S is the native ordered sum of DIF and INH forecasts. No native
graphs, incident modules, dropout calls, or forecast recurrences are repeated.
"""
from math import sqrt

import torch
from torch import nn

from experiments.lag_transport_gw8.model import LagTransportIGSTGNN


ARMS = ('R_CONTINUE', 'SUM_CONTROL', 'BRANCH_READOUT')
PROBE_ARMS = ARMS[1:]
DESIGN = dict(version='GW9', anchor='certified_GW8_R_CONTINUE_BEST',
    question='predictive_information_in_forecast_branch_difference_beyond_sum',
    base_prediction='complete_unchanged_native_R_forward',
    placement='additive_normalized_forecast_readout_after_native_output',
    S='executed_pre_relu_TIID_output_or_ordered_DIF_sum_plus_INH_sum',
    Q='ordered_DIF_sum_minus_ordered_INH_sum',
    SUM_CONTROL='Linear(concat(S,tanh(S)))->Tanh->bias_free_zero_Linear',
    BRANCH_READOUT='Linear(concat(S,tanh(Q)))->Tanh->bias_free_zero_Linear',
    probe='frozen_eval_anchor_shared_features_two_identical_capacity_heads',
    joint='all_anchor_parameters_trainable_with_same_capacity_head_controls',
    target_inputs=False, additional_report_inputs=False,
    parameter_initialization_preserves_rng=True, novelty_status='unestablished',
    evidence_status='server_tests_and_predeclared_equal_budget_screen_required')


def reshape_blocks(blocks):
    """Map native [B,block,N,gap] channels to [B,block*gap,N,1]."""
    if not torch.is_tensor(blocks) or blocks.ndim != 4 or min(blocks.shape) < 1:
        raise ValueError('Readout blocks require nonempty [batch,block,node,gap]')
    batch, _, nodes, _ = blocks.shape
    flat = blocks.transpose(1, 2).contiguous().view(batch, nodes, -1)
    return flat.transpose(1, 2).unsqueeze(-1)


class ReadoutHead(nn.Module):
    """Equal-capacity, zero-output readout for sum or branch information."""
    def __init__(self, channels, gap, arm, hidden=16):
        super().__init__()
        if arm not in PROBE_ARMS or min(channels, gap, hidden) < 1:
            raise ValueError('Invalid GW9 readout arm or dimensions')
        self.channels, self.gap, self.arm, self.hidden = channels, gap, arm, hidden
        self.input = nn.Linear(2*channels, hidden, bias=True)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden, gap, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, S, Q):
        if (not torch.is_tensor(S) or not torch.is_tensor(Q) or S.ndim != 4
                or S.shape != Q.shape or S.shape[-1] != self.channels
                or min(S.shape) < 1 or S.device != Q.device or S.dtype != Q.dtype):
            raise ValueError('GW9 features must be matching [batch,block,node,channel]')
        secondary = S if self.arm == 'SUM_CONTROL' else Q
        joined = torch.cat((S, torch.tanh(secondary)), dim=-1)
        return reshape_blocks(self.output(self.activation(self.input(joined))))


class BranchReadoutIGSTGNN(nn.Module):
    """Nested R model with temporary branch observations and optional readout."""
    def __init__(self, anchor, arm, stage='joint', hidden=16):
        super().__init__()
        if (not isinstance(anchor, LagTransportIGSTGNN) or anchor.arm != 'R_CONTINUE'
                or anchor.kernel is not None):
            raise ValueError('GW9 requires the unchanged GW8 R_CONTINUE anchor')
        if arm not in ARMS or stage not in ('probe', 'joint') or hidden < 1:
            raise ValueError('Invalid GW9 arm, stage, or hidden dimension')
        anchor.assert_cache_clear()
        self.anchor, self.arm, self.stage, self.hidden = anchor, arm, stage, hidden
        for name in ('node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon'):
            setattr(self, name, getattr(anchor, name))
        self.channels = int(self.original._forecast_dim)
        self.gap = int(self.original._model_args['gap'])
        if self.horizon % self.gap:
            raise ValueError('GW9 requires native horizon divisible by gap')
        self.blocks = self.horizon//self.gap
        self._cache, self._handles = {}, []
        self._in_forward, self._diagnostics = False, None
        self.head = None
        mode = anchor.training
        if arm != 'R_CONTINUE':
            parameter = next(anchor.parameters())
            device, dtype = parameter.device, parameter.dtype
            devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == 'cuda' else [])
            with torch.random.fork_rng(devices=devices):
                self.head = ReadoutHead(self.channels, self.gap, arm, hidden).to(
                    device=device, dtype=dtype)
            self.head.train(mode)
        # Preserve heterogeneous native module modes in the joint constructor.
        # Probe freezing intentionally overrides anchor modes and trainability.
        self.training = mode
        self.anchor.requires_grad_(stage == 'joint')
        if stage == 'probe':
            self.anchor.eval()

    @property
    def original(self):
        return self.anchor.original

    @property
    def cache_is_clear(self):
        return (not self._cache and not self._handles and not self._in_forward
            and self.anchor.cache_is_clear)

    def assert_cache_clear(self):
        if not self.cache_is_clear:
            raise AssertionError('GW9 retained batch tensors, observation hooks, or active forward')
        return True

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_counts(self):
        added = 0 if self.head is None else sum(p.numel() for p in self.head.parameters()
            if p.requires_grad)
        return dict(total=self.param_num(), active=sum(p.numel() for p in self.parameters()
            if p.requires_grad), anchor=sum(p.numel() for p in self.anchor.parameters()
            if p.requires_grad), added=added, decoder=0, inactive_native_dif_forecast=0)

    def train(self, mode=True):
        super().train(mode)
        self.anchor.train(mode if self.stage == 'joint' else False)
        return self

    def start_diagnostics(self):
        self._diagnostics = dict(version='GW9', arm=self.arm, stage=self.stage,
            complete_forwards=0, failed_forwards=0, captured_forwards=0,
            layer_observations=0, tiid_observations=0, readout_calls=0,
            correction_square_sum=0., correction_elements=0, correction_abs_max=0.)
        self.anchor.start_diagnostics()

    def diagnostics(self):
        result, self._diagnostics = self._diagnostics, None
        result = result or dict(version='GW9', arm=self.arm, stage=self.stage,
            complete_forwards=0, failed_forwards=0)
        if 'correction_elements' in result:
            count = result['correction_elements']
            result['correction_rms'] = sqrt(result['correction_square_sum']/count) if count else None
        result['anchor'] = self.anchor.diagnostics()
        result['cache_clear'] = self.cache_is_clear
        result['utility_caveat'] = 'Branch differences and readout size do not establish predictive utility'
        return result

    def _observe_layer(self, index, module, inputs, output):
        observed = self._cache['layers']
        if index != len(observed):
            raise RuntimeError('GW9 native layers did not execute exactly once in native order')
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise RuntimeError('Native decoupling layer output contract changed')
        dif, inh = output[1:]
        wanted = (inputs[0].shape[0], self.blocks, self.node_num, self.channels)
        if (not torch.is_tensor(dif) or not torch.is_tensor(inh)
                or tuple(dif.shape) != wanted or tuple(inh.shape) != wanted):
            raise RuntimeError('Native DIF/INH forecast shape changed')
        observed.append((dif, inh))
        if self._diagnostics is not None:
            self._diagnostics['layer_observations'] += 1

    def _observe_tiid(self, module, inputs, output):
        if 'S' in self._cache:
            raise RuntimeError('GW9 observed multiple native TIID calls')
        if (not torch.is_tensor(output) or len(inputs) != 2
                or output.shape != inputs[0].shape):
            raise RuntimeError('Native TIID output contract changed')
        self._cache['S'] = output
        if self._diagnostics is not None:
            self._diagnostics['tiid_observations'] += 1

    def _features(self, base):
        layers = self._cache['layers']
        if len(layers) != len(self.original.layers):
            raise RuntimeError('GW9 did not observe every native forecast layer')
        # Match original Python sum order including the initial integer zero.
        dif = sum([pair[0] for pair in layers])
        inh = sum([pair[1] for pair in layers])
        S = self._cache['S'] if 'S' in self._cache else dif+inh
        features = dict(base=base, S=S, Q=dif-inh)
        if self.stage == 'probe':
            features = {name: value.detach() for name, value in features.items()}
        return features

    def _run(self, history_data, incident_data, sensor_data, capture, apply_head):
        # A rejected nested invocation must not destroy the enclosing capture.
        if self._in_forward:
            raise RuntimeError('Reentrant GW9 forward or feature capture is not supported')
        if self._cache or self._handles:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._cache.clear()
            raise RuntimeError('Stale GW9 observation cache or hooks')
        self._in_forward = True
        try:
            if capture:
                self._cache['layers'] = []
                for index, layer in enumerate(self.original.layers):
                    # Bind the current index instead of closing over the loop variable.
                    def observe(module, inputs, output, index=index):
                        return self._observe_layer(index, module, inputs, output)
                    self._handles.append(layer.register_forward_hook(observe))
                self._handles.append(self.original.tiid_module.register_forward_hook(self._observe_tiid))
            base = self.anchor(history_data, incident_data=incident_data, sensor_data=sensor_data)
            if capture:
                features = self._features(base)
                if self._diagnostics is not None:
                    self._diagnostics['captured_forwards'] += 1
                if not apply_head:
                    result = features
                else:
                    delta = self.head(features['S'], features['Q'])
                    if delta.shape != base.shape:
                        raise RuntimeError('GW9 correction and native output shapes differ')
                    result = base+delta
                    if self._diagnostics is not None:
                        row = self._diagnostics
                        row['readout_calls'] += 1
                        with torch.no_grad():
                            observed = delta.detach().double()
                            row['correction_square_sum'] += float(observed.square().sum())
                            row['correction_elements'] += observed.numel()
                            row['correction_abs_max'] = max(row['correction_abs_max'],
                                float(observed.abs().max()))
            else:
                result = base
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

    def capture_features(self, history_data, label=None, incident_data=None, sensor_data=None):
        """Return base/S/Q from one native pass; labels are intentionally unused."""
        return self._run(history_data, incident_data, sensor_data, capture=True, apply_head=False)

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        return self._run(history_data, incident_data, sensor_data,
            capture=self.head is not None, apply_head=self.head is not None)
