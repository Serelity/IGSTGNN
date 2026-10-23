"""PASE compact scorer: physically remove the already zeroed explicit p block.

The inherited synchronous regional/local equations, incident conditions and
persistent centering are unchanged. Every arm starts from the same legacy
128-input initialization before slicing, rather than redrawing shared tensors.
"""
from collections.abc import Mapping

import torch
from torch import nn

from experiments.pase_mlp_v2.model import (
    MLPCoupledHead, capture_exchange_diagnostics,
)
from experiments.pase_paper_experiments.model import build_model as build_legacy_model
from .protocol import arm_spec

FIRST_WEIGHT = 'head.exchange.network.0.weight'
SCORER_KEYS = frozenset((FIRST_WEIGHT, 'head.exchange.network.0.bias',
                        'head.exchange.network.2.weight', 'head.exchange.network.2.bias'))


def compact_state_dict(weights, width=32):
    """Copy a complete legacy 128-input model state, removing only p columns.

This function accepts model tensors, never a checkpoint/optimizer envelope. Its
source must contain exactly the four expected scorer tensors and their original
shapes. The caller additionally loads the result into its constructed model with
``strict=True`` to certify all other keys and shapes against the actual dataset.
No selected epoch, optimizer state, training RNG or budget is synthesized.
"""
    if type(width) is not int or width < 1:
        raise ValueError('Expected a positive integer hidden width')
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError('Expected a nonempty complete model state dictionary')
    if any(type(key) is not str or not isinstance(value, torch.Tensor)
           for key, value in weights.items()):
        raise ValueError('Model state must contain only named tensors, not checkpoint envelopes')
    scorer = {key for key in weights if key.startswith('head.exchange.')}
    if scorer != SCORER_KEYS:
        raise ValueError('Legacy MLP scorer keys differ from the four expected tensors')
    shapes = {FIRST_WEIGHT: (16, 4*width), 'head.exchange.network.0.bias': (16,),
              'head.exchange.network.2.weight': (2, 16), 'head.exchange.network.2.bias': (2,)}
    dtype, device = weights[FIRST_WEIGHT].dtype, weights[FIRST_WEIGHT].device
    for key, shape in shapes.items():
        value = weights[key]
        if tuple(value.shape) != shape or not value.is_floating_point():
            raise ValueError('Legacy MLP scorer tensor has an unexpected shape or dtype: '+key)
        if value.dtype != dtype or value.device != device:
            raise ValueError('Legacy MLP scorer tensors must share dtype and device')
    if not any(key.startswith('encoder.') for key in weights):
        raise ValueError('Expected a complete PASE model state, not a head-only artifact')
    result = {key: value.detach().clone() for key, value in weights.items()}
    result[FIRST_WEIGHT] = weights[FIRST_WEIGHT][:, width:].detach().clone().contiguous()
    return result


class CompactExchange(nn.Module):
    """96 -> 16 -> 2 at width 32, without any registered explicit p weights."""
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(3*width, 16), nn.GELU(), nn.Linear(16, 2))

    def forward(self, position, condition, coarse_view, local_view, geometry):
        # Keep the inherited call signature; position is used elsewhere in the
        # complete model but is deliberately absent from this scorer's inputs.
        score = self.network(torch.cat([condition, coarse_view, local_view], -1))
        return geometry.center(torch.tanh(score))


class CompactCoupledHead(MLPCoupledHead):
    """Reuse the unchanged current-state synchronous update implementation."""
    def __init__(self, config, arm):
        super().__init__(config, arm)
        self.options = arm_spec(arm)
        self.exchange = CompactExchange(config['hidden'])


def build_model(data, config, arm='FULL', device='cpu'):
    options = arm_spec(arm)
    # This same construction consumes the same RNG sequence as the completed
    # MLP-v2 experiments. Replacement constructors run inside an isolated fork.
    model = build_legacy_model(data, config, 'CONCAT_MLP', 'cpu')
    selected = compact_state_dict(model.state_dict(), int(config['hidden']))
    with torch.random.fork_rng(devices=[]):
        head = CompactCoupledHead(config, arm)
    model.head = head
    model.load_state_dict(selected, strict=True)
    model.arm = arm
    model.paper_arm, model.paper_protocol = arm, options
    if not options['up']:
        model.head.pool_map.requires_grad_(False)
    if not options['down']:
        model.head.lift_map.requires_grad_(False)
    actual = sum(parameter.numel() for parameter in model.head.parameters())
    target = model.capacity['target_head_parameters']
    model.capacity = dict(model.capacity, actual_head_parameters=actual,
        match_fraction=abs(actual-target)/target,
        matching='compact_scorer_actual_parameters_relative_to_GW16_template')
    return model.to(device)


def parameter_accounting(model):
    """Count registered tensors and the exact active shared-scorer dependencies."""
    options = arm_spec(model.paper_arm)
    registered = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    coefficient = sum(parameter.numel() for parameter in model.head.exchange.parameters())
    directions = int(options['up'])+int(options['down'])
    final = model.head.exchange.network[-1]
    inactive_coefficient = (2-directions)*(final.in_features+1)
    map_inactive = 0
    if not options['up']:
        map_inactive += sum(parameter.numel() for parameter in model.head.pool_map.parameters())
    if not options['down']:
        map_inactive += sum(parameter.numel() for parameter in model.head.lift_map.parameters())
    active = registered-inactive_coefficient-map_inactive
    return dict(registered=registered, trainable=trainable, structurally_active=active,
        coefficient_registered=coefficient, coefficient_active=coefficient-inactive_coefficient,
        inactive_registered=registered-active,
        parameter_bytes=sum(parameter.numel()*parameter.element_size() for parameter in model.parameters()),
        buffer_bytes=sum(buffer.numel()*buffer.element_size() for buffer in model.buffers()),
        active_rule='compact_MLP_shared_hidden_partial_output_rows_no_explicit_p_columns',
        one_way_unused_coefficient_rows_retained=directions == 1,
        all_registered_parameters_included_when_frozen=True)
