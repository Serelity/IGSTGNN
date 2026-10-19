"""GW12 heads on detached origin features, separate from the frozen SUM source.

The head owns only its 725 added parameters. The engine must keep the complete
certified source in eval mode, call it under no_grad, and verify its tensors and
parameter flags remain unchanged. Freezing source updates does not require
changing the source requires_grad flags used for certified cache replay.
"""
from math import pi, sqrt

import torch
from torch import nn

from experiments.state_gate_gw11.model import StateAdapter


ARMS = ('STATIC_READOUT', 'STATE_READOUT')
HISTORY, FEATURES, ADAPTERS, ADDED_PARAMETERS = 12, 7, 5, 725
DESIGN = dict(version='GW12', placement='after_complete_certified_GW9_SUM_BEST',
    source_updates=False, source_mode='eval', source_forward='no_grad',
    source_requires_grad_flags='preserved_for_certified_replay',
    head_count=5, hidden=16, added_parameters=725,
    output='mean_of_five_adapters_in_normalized_forecast_units',
    state_inputs=['normalized_last_flow', 'log1p_mean_absolute_normalized_history_change',
        'origin_tod_sin', 'origin_tod_cos', 'origin_dow_sin', 'origin_dow_cos',
        'position_0_to_11_over_11'],
    static_replacements=['node_embedding_u_mean', 'node_embedding_d_mean'],
    feature_gradients=False, future_inputs=False, refitted_normalization=False,
    novelty_status='unestablished', predictive_utility='requires_server_screen')


def origin_features(history, node_emb_u, node_emb_d, arm):
    """Return detached [batch,12,node,7] inputs using only observed history.

    Clock values retain the source preprocessing convention: origin TOD and DOW
    are the last observed history entries. Position is a known output coordinate,
    not a future observation. No label, incident, neighbor or target is accepted.
    """
    if arm not in ARMS:
        raise ValueError('Unknown GW12 arm')
    if (not torch.is_tensor(history) or history.ndim != 4
            or history.shape[0] < 1 or history.shape[1] != HISTORY
            or history.shape[2] < 1 or history.shape[3] != 3
            or not history.is_floating_point()):
        raise ValueError('GW12 expects floating [batch,12,node,3] source-normalized history')
    for embedding in (node_emb_u, node_emb_d):
        if (not torch.is_tensor(embedding) or embedding.ndim != 2
                or embedding.shape[0] != history.shape[2] or embedding.shape[1] < 1
                or embedding.device != history.device or embedding.dtype != history.dtype):
            raise ValueError('GW12 node embeddings must match history node, dtype and device axes')
    if not torch.isfinite(history).all():
        raise ValueError('GW12 history contains non-finite values')
    # Features never establish a differentiable path into the source or input.
    with torch.no_grad():
        batch, _, nodes, _ = history.shape
        if arm == 'STATIC_READOUT':
            first = node_emb_u.detach().mean(dim=-1)[None].expand(batch, nodes)
            second = node_emb_d.detach().mean(dim=-1)[None].expand(batch, nodes)
        else:
            traffic = history[..., 0].detach()
            first = traffic[:, -1]
            second = torch.log1p((traffic[:, 1:]-traffic[:, :-1]).abs().mean(dim=1))
        tod, dow = history[:, -1, :, 1], history[:, -1, :, 2]
        origin = (first, second, torch.sin((2*pi)*tod), torch.cos((2*pi)*tod),
            torch.sin((2*pi)*dow), torch.cos((2*pi)*dow))
        values = [value[:, None, :].expand(batch, HISTORY, nodes) for value in origin]
        coordinate = (torch.arange(HISTORY, dtype=history.dtype,
            device=history.device)/(HISTORY-1))[None, :, None].expand(batch, HISTORY, nodes)
        result = torch.stack([*values, coordinate], dim=-1)
        if not torch.isfinite(result).all():
            raise ValueError('GW12 origin features contain non-finite values')
        return result.detach()


class FrozenReadoutHead(nn.Module):
    """Five zero-output adapters; no source model is registered in this head."""
    def __init__(self, arm, hidden=16):
        super().__init__()
        if arm not in ARMS or hidden != 16:
            raise ValueError('GW12 fixes the registered arm and adapter hidden dimension to 16')
        self.arm, self.hidden = arm, hidden
        self._diagnostics = None
        # Linear construction is on CPU. Preserve all already-initialized RNGs
        # as well, so constructing the paired heads cannot change data/dropout RNG.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            self.adapters = nn.ModuleList(StateAdapter(hidden) for _ in range(ADAPTERS))

    def param_num(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def start_diagnostics(self):
        self._diagnostics = dict(version='GW12', arm=self.arm, complete_forwards=0,
            failed_forwards=0, adapter_calls=[0]*ADAPTERS,
            units='normalized_forecast', readout_delta=dict(
                square_sum=0., sum=0., elements=0, abs_max=0.))

    def diagnostics(self):
        result, self._diagnostics = self._diagnostics, None
        result = result or dict(version='GW12', arm=self.arm, complete_forwards=0,
            failed_forwards=0, adapter_calls=[0]*ADAPTERS,
            units='normalized_forecast', readout_delta=dict(
                square_sum=0., sum=0., elements=0, abs_max=0.))
        delta = result['readout_delta']
        delta['rms'] = sqrt(delta['square_sum']/delta['elements']) if delta['elements'] else None
        delta['mean'] = delta['sum']/delta['elements'] if delta['elements'] else None
        result['delta_rms'], result['delta_mean'] = delta['rms'], delta['mean']
        return result

    def forward(self, features):
        try:
            parameter = next(self.parameters())
            if (not torch.is_tensor(features) or features.ndim != 4
                    or features.shape[0] < 1 or features.shape[1] != HISTORY
                    or features.shape[2] < 1 or features.shape[3] != FEATURES
                    or features.dtype != parameter.dtype or features.device != parameter.device):
                raise ValueError('GW12 head expects matching-device/dtype [batch,12,node,7] features')
            if not torch.isfinite(features).all():
                raise ValueError('GW12 head features contain non-finite values')
            detached = features.detach()
            delta = sum(adapter(detached) for adapter in self.adapters)/ADAPTERS
            if self._diagnostics is not None:
                record = self._diagnostics
                record['adapter_calls'] = [calls+1 for calls in record['adapter_calls']]
                record['complete_forwards'] += 1
                value = delta.detach().double()
                record['readout_delta']['square_sum'] += float(value.square().sum())
                record['readout_delta']['sum'] += float(value.sum())
                record['readout_delta']['elements'] += value.numel()
                record['readout_delta']['abs_max'] = max(
                    record['readout_delta']['abs_max'], float(value.abs().max()))
            return delta
        except BaseException:
            if self._diagnostics is not None:
                self._diagnostics['failed_forwards'] += 1
            raise
