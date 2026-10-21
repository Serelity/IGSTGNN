"""Isolated paper variants of the unchanged GW16 PASE-Net dataflow.

FULL executes the original GW16 forward. Other arms replace only a coefficient
generator or one switch in the coupled head. The initial complete GW16 model is
constructed first in every arm; replacement constructors run in an isolated RNG
context and common tensors are copied exactly. No existing GW16 source changes.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn

from experiments.position_state_gw16.geometry import pool_contrast, lift_contrast
from experiments.position_state_gw16.model import (
    CoupledHead, PositionStateModel, PositionStateExchange,
)
from experiments.state_exchange_multidata.model import DatasetStateExchange
from .protocol import CORE_ARMS, arm_spec


class ConcatMLPExchange(nn.Module):
    """Same p/U/C/L inputs, two scalar outputs, 2098 parameters at d=32."""
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4*width, 16), nn.GELU(), nn.Linear(16, 2))

    def forward(self, position, condition, coarse_view, local_view, geometry):
        p = position[None].expand(condition.shape[0], -1, -1)
        score = self.network(torch.cat([p, condition, coarse_view, local_view], -1))
        return geometry.center(torch.tanh(score))


class RegionalSoftmaxExchange(PositionStateExchange):
    """Replace only tanh by n_r*softmax within each fixed region, then Q."""
    def forward(self, position, condition, coarse_view, local_view, geometry):
        phi = self.position(position).reshape(position.shape[0], 2, self.rank)
        psi = self.dynamic(torch.cat([condition, coarse_view, local_view], -1))
        psi = psi.reshape(*condition.shape[:-1], 2, self.rank)
        score = (phi[None]*psi).sum(-1)/math.sqrt(self.rank)
        normalized = torch.zeros_like(score)
        for region in range(geometry.regions):
            indices = torch.nonzero(geometry.assignment == region, as_tuple=False).flatten()
            values = score.index_select(-2, indices)
            values = torch.softmax(values, dim=-2)*geometry.counts[region].to(score.dtype)
            normalized = normalized.index_copy(-2, indices, values)
        return geometry.center(normalized)


class NoPositionExchange(nn.Module):
    """Remove only the coefficient generator's explicit position projection.

    Position remains in history/origin conditions and in the fixed partition.
    The two learned rank-vectors are shared across all nodes.
    """
    def __init__(self, width, rank):
        super().__init__()
        self.rank = rank
        self.shared_position = nn.Parameter(torch.full((2, rank), 1/math.sqrt(rank)))
        self.dynamic = nn.Linear(3*width, 2*rank)

    def forward(self, position, condition, coarse_view, local_view, geometry):
        psi = self.dynamic(torch.cat([condition, coarse_view, local_view], -1))
        psi = psi.reshape(*condition.shape[:-1], 2, self.rank)
        raw = torch.tanh((self.shared_position[None, None]*psi).sum(-1)/math.sqrt(self.rank))
        return geometry.center(raw)


class PaperCoupledHead(CoupledHead):
    """The GW16 update equations, with explicitly declared single-factor switches."""
    def forward(self, hidden, origin, vector, support, position, geometry, graphs, c, arm):
        options = arm_spec(arm)
        forward, backward, regional, regional_backward = graphs
        c0, l0 = geometry.pool(hidden), geometry.center(hidden)
        coarse, local = c0, l0
        alpha, beta, rms = [], [], []
        for h in range(c['horizon']):
            condition, source = self.condition_source(origin, vector, support, h)
            # All messages and own-state deltas read the same old state.
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


def _replacement_head(original, config, arm):
    # CPU construction only, so no CUDA RNG state or common dropout stream moves.
    with torch.random.fork_rng(devices=[]):
        replacement = PaperCoupledHead(config)
        replacement.load_state_dict(original.state_dict(), strict=True)
        d, rank = config['hidden'], config.get('exchange_rank', 8)
        if arm == 'CONCAT_MLP':
            replacement.exchange = ConcatMLPExchange(d)
        elif arm == 'REGIONAL_SOFTMAX':
            exchange = RegionalSoftmaxExchange(d, rank)
            exchange.load_state_dict(original.exchange.state_dict(), strict=True)
            replacement.exchange = exchange
        elif arm == 'NO_POSITION_BRANCH':
            exchange = NoPositionExchange(d, rank)
            exchange.dynamic.load_state_dict(original.exchange.dynamic.state_dict(), strict=True)
            replacement.exchange = exchange
    return replacement


def build_model(data, config, arm, device):
    if arm not in CORE_ARMS:
        raise ValueError('Unknown or unimplemented PASE paper arm: '+str(arm))
    if config.get('coupling', 'both') != 'both' or config.get('event_ablation', 'none') != 'none':
        raise ValueError('Core paper arms require the unchanged full GW16 base configuration')
    # The dataset extension inherits the original forward verbatim. Its only
    # extension is the released vocabulary cardinality for Alameda and Orange.
    if hasattr(data, 'event_counts'):
        model = DatasetStateExchange(data, config)
    else:
        model = PositionStateModel(data, config, 'STATE_EXCHANGE')
    model.paper_arm = arm
    if arm != 'FULL':
        model.head = _replacement_head(model.head, config, arm)
        model.arm = arm
        options = arm_spec(arm)
        if not options['up']:
            model.head.pool_map.requires_grad_(False)
        if not options['down']:
            model.head.lift_map.requires_grad_(False)
        if not options['up'] and not options['down']:
            model.head.exchange.requires_grad_(False)
        # The inherited GW16 capacity record describes the original head.
        # Keep its template reference but report the replacement's actual size.
        actual = sum(parameter.numel() for parameter in model.head.parameters())
        target = model.capacity['target_head_parameters']
        model.capacity = dict(model.capacity, actual_head_parameters=actual,
            match_fraction=abs(actual-target)/target,
            matching='paper_variant_actual_parameters_relative_to_GW16_template')
    model.paper_protocol = arm_spec(arm)
    return model.to(device)


def parameter_accounting(model):
    """Count stored parameters separately from parameters in the active function.

    A one-way arm still registers a two-output scorer. Only half its output rows
    can affect predictions. They are counted exactly rather than counting an
    entire Parameter as active because one of its entries receives gradients.
    """
    arm = getattr(model, 'paper_arm', 'FULL')
    options = arm_spec(arm)
    registered = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    coefficient = sum(p.numel() for p in model.head.exchange.parameters())
    directions = int(options['up'])+int(options['down'])
    coefficient_active = coefficient if directions == 2 else coefficient*directions//2
    map_inactive = sum(p.numel() for p in model.head.pool_map.parameters()) if not options['up'] else 0
    map_inactive += sum(p.numel() for p in model.head.lift_map.parameters()) if not options['down'] else 0
    active = registered-(coefficient-coefficient_active)-map_inactive
    return dict(registered=registered, trainable=trainable, structurally_active=active,
        coefficient_registered=coefficient, coefficient_active=coefficient_active,
        inactive_registered=registered-active,
        buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        active_rule='structural_prediction_dependencies_including_partial_linear_rows',
        one_way_unused_coefficient_rows_retained=directions == 1,
        all_registered_parameters_included_when_frozen=True)


@contextmanager
def capture_exchange_diagnostics(model):
    """Optional detached diagnostics; never use hooks for latency measurements.

    The original FULL forward is preserved, including module construction and
    arithmetic. Hooks return None and collect the executed coefficient/message
    RMS. Disabled message directions are reported as zero.
    """
    arm = getattr(model, 'paper_arm', 'FULL')
    options = arm_spec(arm)
    weights, ups, downs, handles = [], [], [], []
    record = {}

    def coefficient_hook(module, args, result):
        weights.append(result.detach().square().mean(dim=(0, 1)).sqrt().cpu())

    def message_hook(destination):
        def hook(module, args, result):
            destination.append(result.detach().square().mean().sqrt().cpu())
        return hook

    try:
        handles.append(model.head.exchange.register_forward_hook(coefficient_hook))
        handles.append(model.head.pool_map.register_forward_hook(message_hook(ups)))
        handles.append(model.head.lift_map.register_forward_hook(message_hook(downs)))
        yield record
    finally:
        for handle in handles:
            handle.remove()
    # Keep postprocessing outside finally: a failed forward must retain its
    # original error (for example CUDA OOM), not an incomplete-diagnostics error.
    horizon = int(model.config['horizon'])
    if weights and len(weights) != horizon:
        raise ValueError('Diagnostics context must wrap exactly one complete forward')
    if (ups and len(ups) != horizon) or (downs and len(downs) != horizon):
        raise ValueError('Incomplete message diagnostics')
    if weights:
        coefficient = torch.stack(weights)
    else:
        coefficient = torch.zeros(horizon, 2)
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
