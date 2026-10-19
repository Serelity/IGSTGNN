"""GW15 history-change residual inside the existing IGSTGNN embedding.

The sorted control retains the same eleven observed changes and normalization
but removes their order. The adapter owns no source module or input cache.
Server-only contracts and experiments are required before interpreting utility.
"""
from contextlib import contextmanager
from math import sqrt

import torch
from torch import nn


ARMS = ('SORTED_DELTA', 'ORDERED_DELTA')
HIDDEN, HISTORY, INPUTS, RESIDUAL_SCALE = 16, 12, 11, .1
_GUARD = '_gw15_encoding_context_active'
DESIGN = dict(version='GW15', arms=list(ARMS), history=HISTORY,
    inputs=INPUTS, adapter_hidden=HIDDEN, residual_scale=RESIDUAL_SCALE,
    source='certified_trained_GW9_SUM_CONTROL_BEST', source_mode='eval',
    source_updates=0, source_requires_grad_flags='preserved_for_certified_replay',
    source_forward='autograd_enabled_when_fitting_adapter',
    gradient_request='autograd.grad_loss_with_respect_to_adapter_parameters_only',
    placement='native_embedding_output_before_incident_fusion_and_native_spatiotemporal_layers',
    features='observed_traffic_first_differences_divided_by_1_plus_nodewise_mean_absolute_difference',
    ordered_inputs='eleven_changes_in_observed_temporal_order',
    sorted_control='same_eleven_normalized_changes_sorted_ascending_only_in_adapter_input',
    native_history='unchanged_for_both_arms',
    adapter='Linear_11_16_bias_Tanh_Linear_16_12H_no_bias',
    output='0.1_tanh_adapter_output_reshaped_batch_history_node_source_hidden',
    added_parameter_formula='16*12+16*12*source_hidden',
    source_hidden=32, added_parameters=6336,
    zero_output_recovers='complete_trained_GW9_SUM_CONTROL_forward',
    feature_gradients=False, future_inputs=False, lookup_corrections=False,
    novelty_status='unestablished', predictive_utility='requires_server_equal_budget_screen')


def history_delta(history, arm):
    """Detached, scale-bounded observed changes as [batch,node,11].

    The first channel is source-normalized traffic. The other two observed
    channels are checked for finite values but do not enter the adapter. There
    is no future observation, forecast, label, lookup cell or fitted threshold.
    """
    if arm not in ARMS:
        raise ValueError('Unknown GW15 arm')
    if (not torch.is_tensor(history) or history.ndim != 4
            or history.shape[0] < 1 or history.shape[1] != HISTORY
            or history.shape[2] < 1 or history.shape[3] != 3
            or not history.is_floating_point()):
        raise ValueError('GW15 expects floating [batch,12,node,3] observed history')
    if not torch.isfinite(history).all():
        raise ValueError('GW15 history contains non-finite values')
    with torch.no_grad():
        traffic = history[..., 0]
        change = (traffic[:, 1:]-traffic[:, :-1]).permute(0, 2, 1)
        change = change/(1+change.abs().mean(dim=-1, keepdim=True))
        if not torch.isfinite(change).all():
            raise ValueError('GW15 history-change normalization produced non-finite values')
        if arm == 'SORTED_DELTA':
            change = change.sort(dim=-1).values
        return change.detach()


class TrajectoryEncoder(nn.Module):
    """Zero-initialized bounded residual with identical paired initialization."""
    def __init__(self, arm, source_hidden, hidden=HIDDEN):
        super().__init__()
        if (arm not in ARMS or hidden != HIDDEN or isinstance(source_hidden, bool)
                or not isinstance(source_hidden, int) or source_hidden < 1):
            raise ValueError('GW15 fixes its arm, positive source width and hidden width 16')
        self.arm, self.source_hidden, self.hidden = arm, source_hidden, hidden
        self._diagnostics = None
        # Construction is on CPU. Preserve CPU and every initialized CUDA RNG,
        # giving both arms identical tensors without advancing the caller RNG.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            self.input = nn.Linear(INPUTS, HIDDEN, bias=True)
            self.activation = nn.Tanh()
            self.output = nn.Linear(HIDDEN, HISTORY*source_hidden, bias=False)
            nn.init.zeros_(self.output.weight)

    def param_num(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def start_diagnostics(self):
        self._diagnostics = dict(version='GW15', arm=self.arm,
            complete_forwards=0, failed_forwards=0, adapter_calls=0,
            embedding_hook_calls=0, failed_contexts=0, added_parameters=self.param_num(),
            units='native_embedding', square_sum=0., sum=0., elements=0, abs_max=0.)

    reset_diagnostics = start_diagnostics

    def diagnostics(self):
        if self._diagnostics is None:
            self.start_diagnostics()
        result, self._diagnostics = self._diagnostics, None
        elements = result['elements']
        result['delta_rms'] = sqrt(result['square_sum']/elements) if elements else None
        result['delta_mean'] = result['sum']/elements if elements else None
        result['delta_abs_max'] = result['abs_max']
        return result

    def forward(self, features):
        try:
            parameter = self.input.weight
            if (not torch.is_tensor(features) or features.ndim != 3
                    or features.shape[0] < 1 or features.shape[1] < 1
                    or features.shape[2] != INPUTS or features.dtype != parameter.dtype
                    or features.device != parameter.device):
                raise ValueError('GW15 encoder expects matching-device/dtype [batch,node,11] features')
            if not torch.isfinite(features).all():
                raise ValueError('GW15 encoder features contain non-finite values')
            batch, nodes, _ = features.shape
            residual = RESIDUAL_SCALE*torch.tanh(self.output(self.activation(self.input(features))))
            residual = residual.reshape(batch, nodes, HISTORY, self.source_hidden).permute(0, 2, 1, 3)
            if not torch.isfinite(residual).all():
                raise ValueError('GW15 encoder produced non-finite residuals')
            if self._diagnostics is not None:
                record, observed = self._diagnostics, residual.detach().double()
                record['complete_forwards'] += 1
                record['adapter_calls'] += 1
                record['square_sum'] += float(observed.square().sum())
                record['sum'] += float(observed.sum())
                record['elements'] += observed.numel()
                record['abs_max'] = max(record['abs_max'], float(observed.abs().max()))
            return residual
        except BaseException:
            if self._diagnostics is not None:
                self._diagnostics['failed_forwards'] += 1
            raise


@contextmanager
def encoding_context(source, encoder, history):
    """Insert one residual at native embedding output, removing hooks on exit.

    Source parameters and heterogeneous module modes remain untouched. The
    caller must run one complete source forward inside this context with
    autograd enabled when fitting the adapter; frozen source updates alone do
    not permit no_grad around that forward. Gradients remain connected from
    the forecast, through all native operations, into the added residual.
    """
    if not isinstance(source, nn.Module) or not isinstance(encoder, TrajectoryEncoder):
        raise ValueError('GW15 requires a source module and TrajectoryEncoder')
    original = getattr(source, 'original', None)
    embedding = getattr(original, 'embedding', None)
    if (not isinstance(embedding, nn.Linear) or embedding.in_features != 1
            or embedding.out_features != encoder.source_hidden):
        raise ValueError('GW15 requires native Linear(1, source_hidden) embedding')
    if hasattr(source, _GUARD) or hasattr(embedding, _GUARD):
        raise RuntimeError('GW15 encoding context is not reentrant and cannot share an active embedding')
    token, handle, calls = object(), None, 0
    setattr(source, _GUARD, token)
    setattr(embedding, _GUARD, token)
    try:
        features = history_delta(history, encoder.arm)
        parameter = encoder.input.weight
        if (history.dtype != parameter.dtype or history.device != parameter.device
                or embedding.weight.dtype != parameter.dtype
                or embedding.weight.device != parameter.device):
            raise ValueError('GW15 history, encoder and native embedding dtype/device must agree')
        residual = encoder(features)
        expected_input = (history.shape[0], HISTORY, history.shape[2], 1)
        expected_output = (*expected_input[:-1], encoder.source_hidden)

        def inject(module, inputs, output):
            nonlocal calls
            calls += 1
            if calls != 1:
                raise RuntimeError('GW15 native embedding must execute exactly once')
            if (len(inputs) != 1 or not torch.is_tensor(inputs[0])
                    or tuple(inputs[0].shape) != expected_input
                    or inputs[0].dtype != parameter.dtype or inputs[0].device != parameter.device
                    or not torch.is_tensor(output) or tuple(output.shape) != expected_output
                    or output.dtype != parameter.dtype or output.device != parameter.device):
                raise ValueError('GW15 native embedding input/output contract changed')
            if not torch.isfinite(output).all():
                raise ValueError('GW15 native embedding output contains non-finite values')
            if encoder._diagnostics is not None:
                encoder._diagnostics['embedding_hook_calls'] += 1
            return output+residual

        handle = embedding.register_forward_hook(inject)
        yield encoder
        if calls != 1:
            raise RuntimeError('GW15 native embedding must execute exactly once')
    except BaseException:
        if encoder._diagnostics is not None:
            encoder._diagnostics['failed_contexts'] += 1
        raise
    finally:
        if handle is not None:
            handle.remove()
        if getattr(embedding, _GUARD, None) is token:
            delattr(embedding, _GUARD)
        if getattr(source, _GUARD, None) is token:
            delattr(source, _GUARD)
