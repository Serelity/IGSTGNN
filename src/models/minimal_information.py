"""v13b: common trainable traffic backbone with explicit information interfaces."""

import torch
from torch import nn

from src.models.igstgnn import IGSTGNN


class _CommonModel(IGSTGNN):
    def __init__(self, adjacency, architecture):
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if (adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
                or not torch.isfinite(adjacency).all() or (adjacency < 0).any()):
            raise ValueError('Invalid static adjacency')
        supports = []
        for matrix in (adjacency, adjacency.T):
            degree = matrix.sum(1, keepdim=True)
            supports.append(matrix / degree.clamp_min(torch.finfo(matrix.dtype).tiny))
        super().__init__(model_args=dict(
            num_feat=1, num_hidden=architecture['hidden'], node_hidden=architecture['node_hidden'],
            time_emb_dim=architecture['time_hidden'], layer=architecture['layers'],
            k_s=2, k_t=3, tpd=288, dropout=architecture['dropout'], gap=3,
            adjs=supports, incident_schema='report_location_v1', time_response='fixed',
            sigma_t=1., use_sensor_info=False), node_num=len(adjacency), input_dim=3,
            output_dim=1, seq_len=12, horizon=12, dataset='Contra_Costa')
        # These unused incident paths must not enter the optimizer or checkpoint.
        del self.icsf_module
        del self.tiid_module
        hidden, time_hidden = architecture['hidden'], architecture['time_hidden']
        self.context_projection = nn.Sequential(
            nn.Linear(hidden + 2 * time_hidden + 4, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.context_norm = nn.LayerNorm(hidden)
        # Original graph supports live in a Python list. Register and rebind them
        # on every forward so .to(device) really moves all graph operands.
        self.register_buffer('road_forward', supports[0])
        self.register_buffer('road_backward', supports[1])

    def _apply(self, fn):
        super()._apply(fn)
        # Released ST convolutions cache graph powers outside registered buffers.
        for layer in self.layers:
            conv = layer.dif_layer.localized_st_conv
            conv.pre_defined_graph = [fn(graph) for graph in conv.pre_defined_graph]
        self._model_args['adjs'][:] = [self.road_forward, self.road_backward]
        return self

    def _predict(self, history, clock, distances=None, age=None):
        if (history.ndim != 4 or history.shape[1:] != (12, self.node_num, 3)
                or history.dtype != torch.float32 or not torch.isfinite(history).all()):
            raise ValueError('Expected finite FP32 history [batch,12,nodes,3]')
        batch = len(history)
        if (clock.shape != (batch, 2) or clock.dtype != torch.int64
                or (clock < 0).any() or (clock[:, 0] >= 288).any() or (clock[:, 1] >= 7).any()):
            raise ValueError('Expected independent integer forecast clock [batch,2]')
        if distances is None:
            distances = history.new_zeros(batch, self.node_num, 3)
        if (distances.shape != (batch, self.node_num, 3) or distances.dtype != history.dtype
                or not torch.isfinite(distances).all()):
            raise ValueError('Invalid report distances')
        if age is None:
            age = history.new_zeros(batch)
        if age.shape != (batch,) or not torch.isfinite(age).all() or (age < 0).any() or (age > 5).any():
            raise ValueError('Invalid report age')
        # Disentangle information availability from normalization / output support.
        raw, node_u, node_d, tod, dow = self._prepare_inputs(history)
        embedded = self.embedding(raw)
        calendar = torch.cat((self.T_i_D_emb[clock[:, 0]], self.D_i_W_emb[clock[:, 1]]), -1)
        context = torch.cat((embedded[:, -1], calendar[:, None].expand(-1, self.node_num, -1),
                             distances, (age / 5)[:, None, None].expand(-1, self.node_num, 1)), -1)
        last = self.context_norm(embedded[:, -1] + self.context_projection(context))
        enhanced = torch.cat((embedded[:, :-1], last[:, None]), 1)
        self._model_args['adjs'][:] = [self.road_forward, self.road_backward]
        static, dynamic = self._graph_constructor(node_embedding_u=node_u, node_embedding_d=node_d,
            history_data=enhanced, time_in_day_feat=tod, day_in_week_feat=dow)
        residual, diffusion, inherent = enhanced, [], []
        for layer in self.layers:
            residual, d, i = layer(residual, dynamic, static, node_u, node_d, tod, dow)
            diffusion.append(d)
            inherent.append(i)
        return self._decode_forecast(sum(diffusion) + sum(inherent), None)

    def capacity(self, arm):
        total = sum(p.numel() for p in self.parameters())
        # In this frozen package D0 is identically zero. Remaining zero columns
        # are structural controls, not proof of equal effective capacity.
        zero_columns = {'M0': 4, 'M1': 2, 'M2': 1}[arm]
        inactive = zero_columns * self.context_projection[0].out_features
        return {'nominal_trainable_parameters': total,
                'known_zero_input_weights': inactive,
                'parameters_minus_known_zero_input_weights': total - inactive,
                'effective_capacity_equality_claimed': False}


class TrafficClockModel(_CommonModel):
    def forward(self, history, clock):
        return self._predict(history, clock)


class LocationModel(_CommonModel):
    def forward(self, history, clock, distances):
        return self._predict(history, clock, distances)


class LocationAgeModel(_CommonModel):
    def forward(self, history, clock, distances, age):
        return self._predict(history, clock, distances, age)


MODELS = {'M0': TrafficClockModel, 'M1': LocationModel, 'M2': LocationAgeModel}


def forward_inputs(arm, batch, device):
    """Only whitelisted tensors cross the forward boundary; never labels/masks/IDs."""
    names = ['x', 'clock']
    if arm in ('M1', 'M2'):
        names.append('distances')
    if arm == 'M2':
        names.append('age')
    if arm not in MODELS:
        raise ValueError('Unknown information arm')
    return tuple(torch.as_tensor(batch[name], device=device) for name in names)
