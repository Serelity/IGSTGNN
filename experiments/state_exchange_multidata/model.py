"""Unchanged GW16 STATE_EXCHANGE with release-specific vocabulary dimensions.

The old forward, history encoder, regional partition, exchange and forecast head
are reused directly. Only the published event embedding cardinalities and their
explicit unknown tokens differ between counties. No category-ID remapping.
"""
import torch
from torch import nn

from experiments.position_state_gw16.geometry import RegionGeometry
from experiments.position_state_gw16.model import (
    CoupledHead, HistoryEncoder, OriginCondition, PositionEncoder,
    PositionStateModel, parameter_count)


class DatasetIncidentEncoder(nn.Module):
    def __init__(self, width, counts):
        super().__init__()
        self.type_count = int(counts['type'])
        self.description_count = int(counts['description'])
        if min(self.type_count, self.description_count) < 1:
            raise ValueError('Invalid published vocabulary size')
        # Constructor order and all non-vocabulary dimensions match GW16.
        self.type_embedding = nn.Embedding(self.type_count+1, 8)
        self.description_embedding = nn.Embedding(self.description_count+1, 16)
        self.fusion = nn.Sequential(nn.Linear(26, width), nn.GELU(), nn.Linear(width, width))
        self.relation = nn.Sequential(nn.Linear(3, width), nn.Tanh())

    def forward(self, event):
        raw = event['relation'].float()
        support = raw.abs().sum(-1) > 0
        available = (event['report_present'].bool() & event['relation_available'].bool()
                     & support.any(-1))
        event_type = torch.where(event['type_valid'].bool(), event['type'].long(), self.type_count)
        description = torch.where(event['description_valid'].bool(), event['description'].long(),
                                  self.description_count)
        flags = torch.stack([event['type_valid'], event['description_valid']], -1).float()
        vector = self.fusion(torch.cat([self.type_embedding(event_type),
            self.description_embedding(description), flags], -1))*available[:, None]
        relation = self.relation(raw)*available[:, None, None]
        return vector, relation, support & available[:, None], available


class DatasetStateExchange(PositionStateModel):
    def __init__(self, data, c):
        # Avoid constructing and discarding a Contra-sized incident encoder,
        # which would consume RNG and break the same-shape initialization check.
        nn.Module.__init__(self)
        self.config, self.arm, self.stage = dict(c), 'STATE_EXCHANGE', 'joint'
        self.geometry = RegionGeometry(data.assignment)
        self.register_buffer('mean', torch.tensor(float(data.mean)))
        self.register_buffer('std', torch.tensor(float(data.std)))
        for key, value in zip(('forward_graph', 'backward_graph', 'regional_graph'), data.graphs):
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))
        reverse = self.regional_graph.T
        self.register_buffer('regional_backward', reverse/reverse.sum(-1, keepdim=True).clamp_min(1e-12))
        self.position = PositionEncoder(data.position_features, data.position_categories,
                                       data.position_sizes, c['hidden'])
        self.encoder = HistoryEncoder(c, data.sensors, data.sensor_sizes)
        self.incident = DatasetIncidentEncoder(c['hidden'], data.event_counts)
        self.origin_condition = OriginCondition(c['hidden'])
        self.head = CoupledHead(c)
        count = parameter_count(self.head)
        self.capacity = dict(target_head_parameters=count, actual_head_parameters=count,
            match_fraction=0., internal_width=c.get('update_hidden', 3*c['hidden']),
            matching='identical_origin_state_architecture')


def build_model(data, config, arm, device):
    if arm != 'STATE_EXCHANGE':
        raise ValueError('This experiment fixes the selected GW16 STATE_EXCHANGE architecture')
    return DatasetStateExchange(data, config).to(device)
