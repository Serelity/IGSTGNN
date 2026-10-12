"""B3-R: history capacity logits plus a masked report-conditioned residual.

Only the shared report update changes. The canonical B3 exchange, backbone and
TIID are copied intact, including buffers. Capacities have no vehicle units.
"""
import copy
import hashlib

import torch
from torch import nn

from src.models.incident_spatial import ReportSpatialState, mlp, require


MAPPING_VERSION = 'b3_to_pd_negate_a_zero_d_v1'
CONTEXT_SCHEMA = 'b3r_capacity_context_v1'
EPSILON = .001


def node_capacity_to_edges(capacity, graph):
    """Map capacity directly; do not apply the old state-to-capacity transform."""
    require(capacity.ndim == 3 and capacity.shape[1] == graph.nodes, 'Node capacity axes mismatch')
    require(capacity.device == graph.edge_index.device, 'Capacity graph device mismatch')
    src, dst = graph.edge_index
    left, right = capacity[:, src.clamp_min(0)], capacity[:, dst.clamp_min(0)]
    return torch.where((src < 0)[None, :, None], right,
                       torch.where((dst < 0)[None, :, None], left, torch.minimum(left, right)))


class ReportCapacityResidual(nn.Module):
    def __init__(self, hidden, time_dim, latent=4, *, report_path_enabled=True):
        super().__init__()
        require(type(report_path_enabled) is bool, 'report_path_enabled must be bool')
        self.latent, self.report_path_enabled = latent, report_path_enabled
        summary = 3*hidden+2*time_dim
        self.P, self.D = mlp(summary, latent), mlp(summary+hidden+4, latent)
        nn.init.zeros_(self.D[-1].weight)
        nn.init.zeros_(self.D[-1].bias)
        self.inlet, self.exit = nn.Linear(hidden, latent), nn.Linear(hidden, latent)

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
        delta, effective = None, torch.zeros_like(base_logit)
        if enabled:
            require(embedding is not None, 'Report encoding required')
            age = incident['report_age_minutes'][:, None, None].expand(-1, n, 1)/5.
            evidence = torch.cat((embedding[:, None].expand(-1, n, -1), distance, age), -1)
            delta = self.D(torch.cat((summary, evidence), -1))
            effective = mask*delta
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
                    updated_logit=updated_logit, report_logit_delta=delta, effective_logit_delta=effective,
                    report_mask=mask, report_path_executed=enabled, records=[])


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def retained_state(model):
    return {name: value for name, value in model.state_dict().items()
            if not name.startswith(('report_spatial_state.prior.', 'report_spatial_state.proposal.',
                                    'report_spatial_state.adoption.', 'report_spatial_state.P.',
                                    'report_spatial_state.D.'))}


def from_canonical_b3(canonical, *, report_path_enabled=True):
    """Return an independent model and a checkable initialization certificate.

    Call on a freshly initialized B3, before creating either optimizer. Same seed
    alone is insufficient: deleting G changes later random initialization draws.
    """
    require(getattr(canonical, '_internal_spatial_mode', None) == 'state_restricted'
            and type(canonical.report_spatial_state) is ReportSpatialState, 'Expected canonical B3 architecture')
    require(all(p.requires_grad for p in canonical.parameters()), 'Canonical training parameters must not be frozen')
    result = copy.deepcopy(canonical)
    old = canonical.report_spatial_state
    new = ReportCapacityResidual(canonical._hidden_dim, canonical._model_args['time_emb_dim'], old.latent,
                                 report_path_enabled=report_path_enabled).to(device=old.prior[0].weight.device,
                                                                          dtype=old.prior[0].weight.dtype)
    with torch.no_grad():
        new.P.load_state_dict(old.prior.state_dict())
        new.P[-1].weight.neg_()
        new.P[-1].bias.neg_()
        new.D[0].load_state_dict(old.proposal[0].state_dict())
        new.inlet.load_state_dict(old.inlet.state_dict())
        new.exit.load_state_dict(old.exit.state_dict())
    new.train(old.training)
    result.report_spatial_state = new
    common, copied = retained_state(canonical), retained_state(result)
    require(common.keys() == copied.keys() and all(torch.equal(common[k], copied[k]) for k in common),
            'Retained parameters or buffers differ from canonical B3')
    require(not any(name.startswith(('report_spatial_state.proposal.', 'report_spatial_state.adoption.'))
                    for name, _ in result.named_parameters()), 'Old U/G parameters remain registered')
    manifest = dict(mapping_version=MAPPING_VERSION, report_path_enabled=report_path_enabled,
                    retained_state_sha256=tensor_state_hash(copied),
                    retained_state_specification={k: dict(shape=list(v.shape), dtype=str(v.dtype))
                                                  for k, v in copied.items()},
                    parameter_count=sum(p.numel() for p in result.parameters()),
                    report_module_parameter_count=sum(p.numel() for p in new.parameters()),
                    all_parameters_trainable=all(p.requires_grad for p in result.parameters()),
                    mapping={'P.0': 'prior.0', 'P.2': 'negative prior.2', 'D.0': 'proposal.0',
                             'D.2': 'zero weight and bias', 'inlet': 'inlet', 'exit': 'exit'})
    return result, manifest
