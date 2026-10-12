"""Detached B3-R accumulators; capacity is counted once, exchange per substep."""
import numpy as np
import torch

from src.models.incident_spatial import STAT_NAMES, require
from src.models.incident_spatial_residual import CONTEXT_SCHEMA


SCHEMA = 'b3r_diagnostics_v1'
BOTTLENECK_ATOL = 1e-6


class CapacityDiagnostics:
    def __init__(self):
        self.calls, self.values, self.scalars = 0, None, {}
        self.delta_computed_calls = 0
        self.bottlenecks = {kind: dict(positions=0, report_endpoint_positions=0, report_strict_min=0, ties=0)
                            for kind in ('internal', 'inlet', 'exit')}

    def add(self, name, tensor, mask=None):
        value = tensor.detach().double()
        if mask is not None:
            value = value[mask.expand_as(value)]
        else:
            value = value.flatten()
        old = self.scalars.setdefault(name, dict(count=0, sum=0., minimum=None, maximum=None))
        if value.numel():
            low, high = float(value.min()), float(value.max())
            old['count'] += value.numel()
            old['sum'] += float(value.sum())
            old['minimum'] = low if old['minimum'] is None else min(low, old['minimum'])
            old['maximum'] = high if old['maximum'] is None else max(high, old['maximum'])

    @torch.no_grad()
    def update(self, context):
        require(context['schema'] == CONTEXT_SCHEMA, 'Unsupported capacity context schema')
        values = torch.stack(context['records']).detach().double().cpu().numpy()
        require(values.shape == (20, 2, len(STAT_NAMES)), 'Incomplete five-layer spatial coverage')
        if self.values is None:
            self.values = values.copy()
        else:
            for i, name in enumerate(STAT_NAMES):
                if name.endswith('_max') or name == 'state_max':
                    self.values[..., i] = np.maximum(self.values[..., i], values[..., i])
                elif name == 'state_min':
                    self.values[..., i] = np.minimum(self.values[..., i], values[..., i])
                else:
                    self.values[..., i] += values[..., i]
        self.calls += 1
        mask = context['report_mask'].bool()
        graph, node = context['graph'], context['node_capacity'].detach()
        src, dst = graph.edge_index
        edge_mask = mask[:, src.clamp_min(0)] | mask[:, dst.clamp_min(0)]
        edge_mask = torch.where((src < 0)[None, :, None], mask[:, dst.clamp_min(0)],
                               torch.where((dst < 0)[None, :, None], mask[:, src.clamp_min(0)], edge_mask))
        fields = dict(base_logit=context['base_logit'], updated_logit=context['updated_logit'],
                      effective_logit_delta=context['effective_logit_delta'],
                      node_capacity=node, base_node_capacity=context['base_node_capacity'],
                      node_capacity_abs_change=(node-context['base_node_capacity']).abs(),
                      sigmoid_slope=context['updated_logit'].sigmoid()*(1-context['updated_logit'].sigmoid()),
                      near_saturation=((node < .01) | (node > .99)).to(node.dtype))
        if context['report_logit_delta'] is not None:
            self.delta_computed_calls += 1
            fields['report_logit_delta'] = context['report_logit_delta']
        for name, value in fields.items():
            self.add(name+':all_nodes', value)
            self.add(name+':report_nodes', value, mask)
        for name, value in (('edge_capacity', context['capacity']),
                            ('base_edge_capacity', context['base_edge_capacity']),
                            ('edge_capacity_abs_change', (context['capacity']-context['base_edge_capacity']).abs())):
            self.add(name+':all_edges', value)
            self.add(name+':report_edges', value, edge_mask)
        left, right = node[:, src.clamp_min(0)], node[:, dst.clamp_min(0)]
        for kind, selected in (('internal', (src >= 0) & (dst >= 0)), ('inlet', src < 0), ('exit', dst < 0)):
            old = self.bottlenecks[kind]
            old['positions'] += node.shape[0]*int(selected.sum())*node.shape[-1]
            if kind == 'internal':
                # Denominator counts report-associated endpoint/channel incidences.
                lm, rm = mask[:, src[selected]].expand_as(left[:, selected]), mask[:, dst[selected]].expand_as(right[:, selected])
                old['report_endpoint_positions'] += int(lm.sum()+rm.sum())
                old['report_strict_min'] += int(((left[:, selected] < right[:, selected]-BOTTLENECK_ATOL) & lm).sum()
                                              + ((right[:, selected] < left[:, selected]-BOTTLENECK_ATOL) & rm).sum())
                old['ties'] += int(((left[:, selected]-right[:, selected]).abs() <= BOTTLENECK_ATOL).sum())
            else:
                count = int(edge_mask[:, selected].expand_as(left[:, selected]).sum())
                old['report_endpoint_positions'] += count
                old['report_strict_min'] += count  # open edge has exactly one real endpoint

    def result(self):
        if self.values is None:
            return None
        rows = []
        for call, steps in enumerate(self.values):
            for step, value in enumerate(steps):
                row = dict(zip(STAT_NAMES, map(float, value)))
                row.update(layer=call//4, call_in_layer=call%4, substep=step,
                           clip_fraction=row['clip_count']/max(row['positions'], 1),
                           potential_capacity_limited_fraction=row['capacity_limited_count']/max(row['edge_positions'], 1))
                rows.append(row)
        scalars = {k: dict(count=v['count'], mean=v['sum']/v['count'] if v['count'] else None,
                           minimum=v['minimum'], maximum=v['maximum'], status='available' if v['count'] else 'unavailable')
                   for k, v in self.scalars.items()}
        return dict(schema=SCHEMA, batches=self.calls, capacity_evaluations=self.calls,
                    raw_delta_computed_batches=self.delta_computed_calls,
                    raw_delta_status='not_computed' if not self.delta_computed_calls else 'computed',
                    spatial_call_count_per_forward=20, rows=rows, capacity_statistics=scalars,
                    bottlenecks=self.bottlenecks, bottleneck_atol=BOTTLENECK_ATOL,
                    bottleneck_denominator='report endpoint/channel incidences; ties use edge/channel positions',
                    budget_counts_semantics='restrictions applied; counts do not measure final flux sensitivity',
                    flux_units='dimensionless_per_call_latent_channel',
                    restricted_stability=dict(clip_fraction=sum(r['clip_count'] for r in rows)/max(sum(r['positions'] for r in rows), 1),
                                              clip_abs_max=max(r['clip_abs_max'] for r in rows)),
                    inference_gain_claim=False)
