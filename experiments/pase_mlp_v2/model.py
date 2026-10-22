"""PASE MLP v2: preserve the selected legacy model, version the ablations.

FULL returns the legacy CONCAT_MLP model without replacing any numerical module
or its forward. Consequently old state dictionaries and training RNG streams are
compatible. Non-FULL arms copy those same tensors into a head with explicit
single-factor switches. No v1 experiment or original GW16 source is modified.
"""
from contextlib import contextmanager

import torch

from experiments.position_state_gw16.geometry import pool_contrast, lift_contrast
from experiments.position_state_gw16.model import CoupledHead
from experiments.pase_paper_experiments.model import (
    ConcatMLPExchange, build_model as build_legacy_model,
)
from .protocol import CORE_ARMS, arm_spec


class MLPExchange(ConcatMLPExchange):
    """Identical registered scorer; optionally zero only its explicit p block."""
    def __init__(self, width, *, explicit_position=True):
        super().__init__(width)
        self.explicit_position = bool(explicit_position)

    def forward(self, position, condition, coarse_view, local_view, geometry):
        if self.explicit_position:
            return super().forward(position, condition, coarse_view, local_view, geometry)
        return super().forward(torch.zeros_like(position), condition,
                               coarse_view, local_view, geometry)


class MLPCoupledHead(CoupledHead):
    """Same synchronous equations as v1, with the MLP scorer in every arm."""
    def __init__(self, config, arm):
        super().__init__(config)
        self.options = arm_spec(arm)
        self.exchange = MLPExchange(config['hidden'],
            explicit_position=self.options['explicit_position'])

    def forward(self, hidden, origin, vector, support, position, geometry, graphs, c, arm):
        options = self.options
        forward, backward, regional, regional_backward = graphs
        c0, l0 = geometry.pool(hidden), geometry.center(hidden)
        coarse, local = c0, l0
        alpha, beta, rms = [], [], []
        for h in range(c['horizon']):
            condition, source = self.condition_source(origin, vector, support, h)
            cv, lv = self.coarse_norm(coarse), self.local_norm(local)
            up, down = torch.zeros_like(coarse), torch.zeros_like(local)
            if options['up'] or options['down']:
                if options['coefficient_states'] == 'origin':
                    gate_c, gate_l = self.coarse_norm(c0), self.local_norm(l0)
                else:
                    gate_c, gate_l = cv, lv
                weights = self.exchange(position, condition, geometry.lift(gate_c), gate_l, geometry)
                if options['up']:
                    up = self.pool_map(pool_contrast(geometry, local, weights[..., :1]))
                if options['down']:
                    down = self.lift_map(lift_contrast(geometry, coarse, weights[..., 1:2]))
            coarse_delta = self.coarse_update(torch.cat([
                self.coarse_graph(cv, regional, regional_backward), geometry.pool(condition)], -1))
            local_delta = self.local_update(torch.cat([
                self.local_graph(lv, forward, backward), condition], -1))
            step = self.step_logits.sigmoid()
            next_coarse = coarse+step[0]*(self.dropout(coarse_delta)+up+geometry.pool(source))
            next_local = local+step[1]*(self.dropout(local_delta)+down+geometry.center(source))
            if options['interior_center']:
                next_local = geometry.center(next_local)
            coarse, local = next_coarse, next_local
            alpha.append(self.coarse_readout(coarse-c0))
            beta.append(geometry.center(self.local_readout(local-l0)))
            rms.append(torch.stack([coarse.detach().square().mean().sqrt(),
                                    local.detach().square().mean().sqrt()]))
        return torch.stack(alpha, 1), torch.stack(beta, 1), torch.stack(rms)


def build_model(data, config, arm='FULL', device='cpu'):
    options = arm_spec(arm)
    # Construct on CPU so isolated replacement construction never advances the
    # CUDA RNG stream. This is exactly the legacy CONCAT_MLP construction path.
    model = build_legacy_model(data, config, 'CONCAT_MLP', 'cpu')
    if arm != 'FULL':
        with torch.random.fork_rng(devices=[]):
            head = MLPCoupledHead(config, arm)
            head.load_state_dict(model.head.state_dict(), strict=True)
        model.head = head
        model.arm = arm
        if not options['up']:
            model.head.pool_map.requires_grad_(False)
        if not options['down']:
            model.head.lift_map.requires_grad_(False)
        if not options['up'] and not options['down']:
            model.head.exchange.requires_grad_(False)
    model.paper_arm = arm
    model.paper_protocol = options
    return model.to(device)


def parameter_accounting(model):
    """Count numerical dependencies, including partially inactive linear blocks.

    Both MLP outputs share its hidden layer. Removing one direction disables
    only 16 final weights and one final bias, not half the whole scorer.
    Position-zeroing leaves 16*d first-layer entries structurally inactive.
    """
    options = arm_spec(model.paper_arm)
    registered = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    coefficient = sum(p.numel() for p in model.head.exchange.parameters())
    directions = int(options['up'])+int(options['down'])
    inactive_coefficient = 0
    if directions == 0:
        inactive_coefficient = coefficient
    else:
        final = model.head.exchange.network[-1]
        inactive_coefficient += (2-directions)*(final.in_features+1)
        if not options['explicit_position']:
            first = model.head.exchange.network[0]
            inactive_coefficient += first.out_features*int(model.config['hidden'])
    map_inactive = sum(p.numel() for p in model.head.pool_map.parameters()) if not options['up'] else 0
    if not options['down']:
        map_inactive += sum(p.numel() for p in model.head.lift_map.parameters())
    active = registered-inactive_coefficient-map_inactive
    return dict(registered=registered, trainable=trainable, structurally_active=active,
        coefficient_registered=coefficient, coefficient_active=coefficient-inactive_coefficient,
        inactive_registered=registered-active,
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
        active_rule='MLP_shared_hidden_layer_partial_output_rows_and_zero_input_columns',
        one_way_unused_coefficient_rows_retained=directions == 1,
        all_registered_parameters_included_when_frozen=True)


@contextmanager
def capture_exchange_diagnostics(model):
    """Detached, same-forward diagnostics; exclude hooks from speed benchmarks."""
    options = arm_spec(model.paper_arm)
    weights, ups, downs, handles, record = [], [], [], [], {}

    def coefficient_hook(module, args, output):
        weights.append(output.detach().square().mean(dim=(0, 1)).sqrt().cpu())

    def message_hook(destination):
        def hook(module, args, output):
            destination.append(output.detach().square().mean().sqrt().cpu())
        return hook

    try:
        handles.append(model.head.exchange.register_forward_hook(coefficient_hook))
        handles.append(model.head.pool_map.register_forward_hook(message_hook(ups)))
        handles.append(model.head.lift_map.register_forward_hook(message_hook(downs)))
        yield record
    finally:
        for handle in handles:
            handle.remove()
    horizon = int(model.config['horizon'])
    if any(values and len(values) != horizon for values in (weights, ups, downs)):
        raise ValueError('Diagnostics must wrap exactly one complete forward')
    coefficient = torch.stack(weights) if weights else torch.zeros(horizon, 2)
    if not options['up']:
        coefficient[:, 0] = 0
    if not options['down']:
        coefficient[:, 1] = 0
    up = torch.stack(ups) if ups else torch.zeros(horizon)
    down = torch.stack(downs) if downs else torch.zeros(horizon)
    record.update(coefficient_rms=coefficient.tolist(),
        message_rms=torch.stack([up, down], -1).tolist(),
        active_directions=[options['up'], options['down']],
        measurement='per_batch_detached_RMS_not_population_RMS')
