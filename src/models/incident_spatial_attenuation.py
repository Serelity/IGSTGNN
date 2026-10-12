"""Attenuate only the B3-R D logit correction with a scalar node strength.

P, the signed D proposal, open-boundary heads, latent exchange, readout and TIID
are copied intact. Capacities and exchanges retain their original latent units.
"""
import copy

import torch
from torch import nn

from src.models.incident_spatial import mlp, require
from src.models.incident_spatial_residual import (
    CONTEXT_SCHEMA, EPSILON, ReportCapacityResidual, node_capacity_to_edges, tensor_state_hash,
)


MAPPING_VERSION = 'b3r_to_scalar_d_attenuation_v1'
MODES = ('fixed_half', 'state_gate')


class ReportCapacityAttenuation(ReportCapacityResidual):
    """Copy the parent report module and add optional causal D attenuation.

The gate observes the same available history/report evidence as D. One alpha
per node scales all latent channels; it never scales P or the exchange readout.
"""
    def __init__(self, parent, mode):
        nn.Module.__init__(self)
        require(type(parent) is ReportCapacityResidual and mode in MODES,
                'Expected an original B3R residual module and a supported attenuation mode')
        self.latent, self.report_path_enabled = parent.latent, parent.report_path_enabled
        self.attenuation_mode = mode
        for name in ('P', 'D', 'inlet', 'exit'):
            setattr(self, name, copy.deepcopy(getattr(parent, name)))
        self.gate = None
        if mode == 'state_gate':
            # The gate is constructed on CPU, so this preserves the caller's RNG
            # without initializing CUDA or consuming later backbone draws.
            with torch.random.fork_rng(devices=[]):
                self.gate = mlp(parent.D[0].in_features, 1)
                nn.init.zeros_(self.gate[-1].weight)
                nn.init.zeros_(self.gate[-1].bias)
            reference = parent.P[0].weight
            self.gate.to(device=reference.device, dtype=reference.dtype)
        self.train(parent.training)

    def forward(self, history, tod, dow, embedding, incident, graph, *, report_enabled=True):
        require(type(report_enabled) is bool, 'report_enabled must be bool')
        b, _, n, _ = history.shape
        require(n == graph.nodes, 'Capacity graph/node mismatch')
        clock = torch.cat((tod, dow), -1)[:, None].expand(-1, n, -1)
        summary = torch.cat((history[:, -1], history.mean(1), history[:, -1]-history[:, 0], clock), -1)
        base_logit = self.P(summary)
        mask = history.new_zeros(b, n, 1)
        if incident is not None:
            distance = incident['distances']
            require(distance.shape == (b, n, 3), 'Report distance axes mismatch')
            mask = (distance.abs().sum(-1, keepdim=True) > 0).to(history.dtype)
        enabled = self.report_path_enabled and report_enabled and incident is not None
        delta, strength = None, None
        raw_effective, effective = torch.zeros_like(base_logit), torch.zeros_like(base_logit)
        if enabled:
            require(embedding is not None, 'Report encoding required')
            age = incident['report_age_minutes'][:, None, None].expand(-1, n, 1)/5.
            evidence = torch.cat((embedding[:, None].expand(-1, n, -1), distance, age), -1)
            conditioned = torch.cat((summary, evidence), -1)
            delta = self.D(conditioned)
            strength = (history.new_full((b, n, 1), .5) if self.gate is None
                        else self.gate(conditioned).sigmoid())
            raw_effective = mask*delta
            effective = raw_effective*strength
        updated_logit = base_logit+effective
        base_node = EPSILON+(1-EPSILON)*base_logit.sigmoid()
        node = EPSILON+(1-EPSILON)*updated_logit.sigmoid()
        src, dst = graph.edge_index
        boundary = torch.where((src < 0)[None, :, None], self.inlet(history[:, -1]).sigmoid()[:, dst.clamp_min(0)],
                               self.exit(history[:, -1]).sigmoid()[:, src.clamp_min(0)])
        boundary = torch.where(((src < 0) | (dst < 0))[None, :, None], boundary, 0.)
        return dict(schema=CONTEXT_SCHEMA, graph=graph, capacity=node_capacity_to_edges(node, graph),
                    base_edge_capacity=node_capacity_to_edges(base_node, graph), boundary=boundary,
                    base_node_capacity=base_node, node_capacity=node, base_logit=base_logit,
                    updated_logit=updated_logit, report_logit_delta=delta,
                    raw_effective_logit_delta=raw_effective, effective_logit_delta=effective,
                    d_strength=strength, report_mask=mask, report_path_executed=enabled, records=[])


def from_residual(model, mode):
    """Return an independent B3-R variant with an auditable exact shared copy."""
    require(mode in MODES, 'Unknown D attenuation mode')
    require(getattr(model, '_internal_spatial_mode', None) == 'state_restricted'
            and type(model.report_spatial_state) is ReportCapacityResidual,
            'Expected original B3R architecture')
    parent_state = model.state_dict()
    parent_hash = tensor_state_hash(parent_state)
    # Converted weights must remain ordinary trainable tensors even when a
    # caller constructs a comparison variant inside an inference context.
    with torch.inference_mode(False):
        result = copy.deepcopy(model)
        result.report_spatial_state = ReportCapacityAttenuation(result.report_spatial_state, mode)
    copied = result.state_dict()
    shared = {name: copied[name] for name in parent_state}
    require(all(torch.equal(parent_state[name], shared[name]) for name in parent_state),
            'Shared parent parameters or buffers changed during attenuation conversion')
    added = set(copied)-set(parent_state)
    require(added == ({'report_spatial_state.gate.'+name for name in result.report_spatial_state.gate.state_dict()}
                      if mode == 'state_gate' else set()), 'Unexpected non-gate parameters or buffers')
    require(tensor_state_hash(model.state_dict()) == parent_hash, 'Input parent model was modified')
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    certificate = dict(mapping_version=MAPPING_VERSION, mode=mode, initial_alpha=.5,
                       report_path_enabled=model.report_spatial_state.report_path_enabled,
                       parent_state_sha256=parent_hash,
                       retained_shared_state_sha256=tensor_state_hash(shared),
                       retained_shared_state_specification={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                                                            for name, value in shared.items()},
                       parent_parameter_count=count(model), parameter_count=count(result),
                       shared_parameter_count=count(model), added_parameters=count(result)-count(model),
                       gate_parameter_count=count(result.report_spatial_state.gate) if mode == 'state_gate' else 0,
                       report_module_parameter_count=count(result.report_spatial_state),
                       all_parameters_trainable=all(parameter.requires_grad for parameter in result.parameters()),
                       mapping={'parent_model_state': 'exact independent copy of all parameters and buffers',
                                'P': 'unchanged', 'D': 'unchanged signed proposal',
                                'inlet': 'unchanged', 'exit': 'unchanged', 'TIID': 'unchanged',
                                'exchange_readout': 'unchanged',
                                'alpha': 'fixed 0.5' if mode == 'fixed_half' else 'sigmoid gate with zero final weight/bias',
                                'effective_logit_delta': 'report_mask * alpha * D'})
    return result, certificate
