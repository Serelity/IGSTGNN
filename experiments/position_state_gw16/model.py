"""Position-dependent exchange in history-initialized regional/local dynamics.

All four mechanism arms share input modules and active input fields. ORIGIN and
STATE differ only in the states read by the exchange conditioner. The augmented
native arm is a fresh end-to-end comparison, not a replay of an old checkpoint.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from . import ARMS
from .geometry import RegionGeometry, pool_contrast, lift_contrast


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class GraphLinear(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_map = nn.Linear(width, width, bias=False)
        self.forward_map = nn.Linear(width, width, bias=False)
        self.backward_map = nn.Linear(width, width, bias=False)

    def forward(self, value, forward, backward):
        return (self.self_map(value)+self.forward_map(forward@value)
                +self.backward_map(backward@value))/3.


class PositionEncoder(nn.Module):
    def __init__(self, features, categories, sizes, width):
        super().__init__()
        self.register_buffer('features', torch.as_tensor(features, dtype=torch.float32))
        self.register_buffer('categories', torch.as_tensor(categories, dtype=torch.long))
        self.road = nn.Embedding(int(sizes[0]), width)
        self.direction = nn.Embedding(int(sizes[1]), width)
        self.numeric = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self):
        return self.norm((self.road(self.categories[:, 0])+self.direction(self.categories[:, 1])
                          +self.numeric(self.features))/math.sqrt(3.))


class HistoryEncoder(nn.Module):
    def __init__(self, c, sensors, sizes):
        super().__init__()
        d = c['hidden']
        self.flow = nn.Linear(1, d)
        self.node = nn.Embedding(c['nodes'], d)
        self.tod = nn.Embedding(288, d)
        self.dow = nn.Embedding(7, d)
        self.sensor_embeddings = nn.ModuleList(nn.Embedding(int(size), d) for size in sizes)
        for key in ('sensor_type', 'surface', 'roadway_use'):
            self.register_buffer(key, torch.as_tensor(sensors[key], dtype=torch.long))
        self.register_buffer('sensor_numeric', torch.stack([
            torch.as_tensor(sensors[key], dtype=torch.float32)
            for key in ('road_width', 'speed_limit')], -1)/100.)
        self.sensor_numbers = nn.Linear(2, d)
        self.history_norm = nn.LayerNorm(d)
        self.temporal = nn.GRU(d, d, batch_first=True)
        self.graph_layers = nn.ModuleList(GraphLinear(d) for _ in range(c['spatial_layers']))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(c['spatial_layers']))
        self.dropout = nn.Dropout(c['dropout'])

    def forward(self, x, position, forward, backward):
        b, t, n, _ = x.shape
        sensor = self.sensor_numbers(self.sensor_numeric)
        for embedding, key in zip(self.sensor_embeddings, ('sensor_type', 'surface', 'roadway_use')):
            sensor = sensor+embedding(getattr(self, key))
        tod = (x[..., 1]*288).round().long().clamp(0, 287)
        dow = (x[..., 2]*7).round().long().clamp(0, 6)
        values = (self.flow(x[..., :1])+self.tod(tod)+self.dow(dow)
                  +self.node.weight[None, None]+sensor[None, None]+position[None, None])/math.sqrt(6.)
        values = self.history_norm(values).permute(0, 2, 1, 3).reshape(b*n, t, -1)
        _, hidden = self.temporal(values)
        hidden = hidden[-1].reshape(b, n, -1)
        for layer, norm in zip(self.graph_layers, self.norms):
            hidden = norm(hidden+self.dropout(F.gelu(layer(hidden, forward, backward))))
        return hidden


class IncidentEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.type_embedding = nn.Embedding(7, 8)
        self.description_embedding = nn.Embedding(30, 16)
        self.fusion = nn.Sequential(nn.Linear(26, width), nn.GELU(), nn.Linear(width, width))
        self.relation = nn.Sequential(nn.Linear(3, width), nn.Tanh())

    def forward(self, event):
        raw = event['relation'].float()
        support = raw.abs().sum(-1) > 0
        available = (event['report_present'].bool() & event['relation_available'].bool()
                     & support.any(-1))
        event_type = torch.where(event['type_valid'].bool(), event['type'].long(), 6)
        description = torch.where(event['description_valid'].bool(), event['description'].long(), 29)
        flags = torch.stack([event['type_valid'], event['description_valid']], -1).float()
        vector = self.fusion(torch.cat([self.type_embedding(event_type),
            self.description_embedding(description), flags], -1))*available[:, None]
        relation = self.relation(raw)*available[:, None, None]
        return vector, relation, support & available[:, None], available


class OriginCondition(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.summary = nn.Linear(2, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, history, hidden, position, vector, relation):
        last = history[:, -1, :, 0]
        change = (history[:, 1:, :, 0]-history[:, :-1, :, 0]).abs().mean(1)
        summary = self.summary(torch.stack([last, change], -1))
        return self.norm((hidden+position[None]+vector[:, None]+relation)/2.)+summary


class HeadInputs(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c['hidden']
        self.lead = nn.Embedding(c['horizon'], d)
        self.source_gate = nn.Linear(d, 1)
        self.source_projection = nn.Linear(d, d, bias=False)
        self.coarse_readout = nn.Linear(d, 1, bias=False)
        self.local_readout = nn.Linear(d, 1, bias=False)
        nn.init.normal_(self.coarse_readout.weight, std=.02)
        nn.init.normal_(self.local_readout.weight, std=.02)

    def condition_source(self, origin, vector, support, step):
        condition = origin+self.lead.weight[step]
        source = (support[..., None].to(origin.dtype)*torch.sigmoid(self.source_gate(condition))
                  *self.source_projection(vector)[:, None])
        return condition, source


class PositionStateExchange(nn.Module):
    """Rank-limited explicit position x dynamic-state interaction, two directions."""
    def __init__(self, width, rank):
        super().__init__()
        self.rank = rank
        self.position = nn.Linear(width, 2*rank, bias=False)
        self.dynamic = nn.Linear(3*width, 2*rank)

    def forward(self, position, condition, coarse_view, local_view, geometry, *, centered=True):
        phi = self.position(position).reshape(position.shape[0], 2, self.rank)
        psi = self.dynamic(torch.cat([condition, coarse_view, local_view], -1))
        psi = psi.reshape(*condition.shape[:-1], 2, self.rank)
        raw = torch.tanh((phi[None]*psi).sum(-1)/math.sqrt(self.rank))
        return geometry.center(raw) if centered else .5*(1+raw)


class CoupledHead(HeadInputs):
    def __init__(self, c):
        super().__init__(c)
        d = c['hidden']
        self.coarse_norm = nn.LayerNorm(d)
        self.local_norm = nn.LayerNorm(d)
        self.coarse_graph = GraphLinear(d)
        self.local_graph = GraphLinear(d)
        width = c.get('update_hidden', 3*d)
        self.coarse_update = nn.Sequential(nn.Linear(2*d, width), nn.GELU(), nn.Linear(width, d))
        self.local_update = nn.Sequential(nn.Linear(2*d, width), nn.GELU(), nn.Linear(width, d))
        self.exchange = PositionStateExchange(d, c.get('exchange_rank', 8))
        self.pool_map = nn.Linear(d, d, bias=False)
        self.lift_map = nn.Linear(d, d, bias=False)
        initial = float(c.get('step_initial', .1))
        self.step_logits = nn.Parameter(torch.full((2,), math.log(initial/(1-initial))))
        self.dropout = nn.Dropout(c['dropout'])

    def forward(self, hidden, origin, vector, support, position, geometry, graphs, c, arm):
        forward, backward, regional, regional_backward = graphs
        c0, l0 = geometry.pool(hidden), geometry.center(hidden)
        coarse, local = c0, l0
        alpha, beta, rms = [], [], []
        for h in range(c['horizon']):
            condition, source = self.condition_source(origin, vector, support, h)
            # All right-hand terms read the old state; C/L updates are simultaneous.
            cv, lv = self.coarse_norm(coarse), self.local_norm(local)
            if arm == 'ORIGIN_EXCHANGE':
                gate_c, gate_l = self.coarse_norm(c0), self.local_norm(l0)
            else:
                gate_c, gate_l = cv, lv
            weights = self.exchange(position, condition, geometry.lift(gate_c), gate_l, geometry)
            up = self.pool_map(pool_contrast(geometry, local, weights[..., :1]))
            down = self.lift_map(lift_contrast(geometry, coarse, weights[..., 1:2]))
            if c.get('coupling', 'both') == 'none':
                up, down = torch.zeros_like(up), torch.zeros_like(down)
            coarse_delta = self.coarse_update(torch.cat([
                self.coarse_graph(cv, regional, regional_backward), geometry.pool(condition)], -1))
            local_delta = self.local_update(torch.cat([
                self.local_graph(lv, forward, backward), condition], -1))
            step = self.step_logits.sigmoid()
            next_coarse = coarse+step[0]*(self.dropout(coarse_delta)+up+geometry.pool(source))
            next_local = geometry.center(local+step[1]*(self.dropout(local_delta)+down+geometry.center(source)))
            coarse, local = next_coarse, next_local
            alpha.append(self.coarse_readout(coarse-c0))
            beta.append(geometry.center(self.local_readout(local-l0)))
            rms.append(torch.stack([coarse.detach().square().mean().sqrt(), local.detach().square().mean().sqrt()]))
        return torch.stack(alpha, 1), torch.stack(beta, 1), torch.stack(rms)


class DirectHead(HeadInputs):
    def __init__(self, c, width):
        super().__init__(c)
        d = c['hidden']
        self.coarse = nn.Sequential(nn.Linear(2*d, width), nn.GELU(), nn.Dropout(c['dropout']), nn.Linear(width, d))
        self.local = nn.Sequential(nn.Linear(4*d, width), nn.GELU(), nn.Dropout(c['dropout']), nn.Linear(width, d))

    def forward(self, hidden, origin, vector, support, position, geometry, graphs, c, arm):
        c0, l0 = geometry.pool(hidden), geometry.center(hidden)
        alpha, beta = [], []
        for h in range(c['horizon']):
            condition, source = self.condition_source(origin, vector, support, h)
            condition = condition+source
            coarse = self.coarse(torch.cat([c0, geometry.pool(condition)], -1))
            local = self.local(torch.cat([hidden, condition, geometry.lift(c0), l0], -1))
            alpha.append(self.coarse_readout(coarse))
            beta.append(geometry.center(self.local_readout(local)))
        return torch.stack(alpha, 1), torch.stack(beta, 1), hidden.new_zeros((c['horizon'], 2))


class GraphGRUUpdate(nn.Module):
    """Ordinary free hidden state; graph and cross-grain inputs enter GRU gates."""
    def __init__(self, d, width, dropout):
        super().__init__()
        self.graph = GraphLinear(d)
        self.norm = nn.LayerNorm(d)
        self.gates = nn.Sequential(nn.Linear(4*d, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 3*d))
        self.hidden_gates = nn.Linear(d, 3*d)

    def forward(self, state, other, condition, forward, backward):
        view = self.norm(state)
        input_r, input_z, input_n = self.gates(torch.cat([
            view, self.graph(view, forward, backward), condition, other], -1)).chunk(3, -1)
        hidden_r, hidden_z, hidden_n = self.hidden_gates(state).chunk(3, -1)
        reset = torch.sigmoid(input_r+hidden_r)
        update = torch.sigmoid(input_z+hidden_z)
        proposed = torch.tanh(input_n+reset*hidden_n)
        return update*state+(1-update)*proposed


class FreeHead(HeadInputs):
    def __init__(self, c, width):
        super().__init__(c)
        d = c['hidden']
        self.coarse = GraphGRUUpdate(d, width, c['dropout'])
        self.local = GraphGRUUpdate(d, width, c['dropout'])
        self.exchange = PositionStateExchange(d, c.get('exchange_rank', 8))

    def forward(self, hidden, origin, vector, support, position, geometry, graphs, c, arm):
        forward, backward, regional, regional_backward = graphs
        c0, l0 = geometry.pool(hidden), geometry.center(hidden)
        coarse, local = c0, l0
        alpha, beta, rms = [], [], []
        for h in range(c['horizon']):
            condition, source = self.condition_source(origin, vector, support, h)
            weights = self.exchange(position, condition, geometry.lift(self.coarse.norm(coarse)),
                self.local.norm(local), geometry, centered=False)
            up = geometry.pool(weights[..., :1]*local)
            down = weights[..., 1:2]*geometry.lift(coarse)
            next_coarse = self.coarse(coarse, up, geometry.pool(condition+source), regional, regional_backward)
            next_local = self.local(local, down, condition+source, forward, backward)
            coarse, local = next_coarse, next_local  # no interior zero-mean constraint
            alpha.append(self.coarse_readout(coarse-c0))
            beta.append(geometry.center(self.local_readout(local-l0)))
            rms.append(torch.stack([coarse.detach().square().mean().sqrt(), local.detach().square().mean().sqrt()]))
        return torch.stack(alpha, 1), torch.stack(beta, 1), torch.stack(rms)


def _matched_head(c, arm):
    # Counting templates are isolated from the actual shared/head initialization.
    # Parameter counts are affine in internal width; no fitting/training search.
    with torch.random.fork_rng(devices=[]):
        target = parameter_count(CoupledHead(c))
        cls = DirectHead if arm == 'DIRECT' else FreeHead
        one, two = parameter_count(cls(c, 1)), parameter_count(cls(c, 2))
    key = 'direct_head_width' if arm == 'DIRECT' else 'free_transition_width'
    requested = c.get(key)
    width = int(requested) if requested is not None else max(1, int(round(1+(target-one)/(two-one))))
    head = cls(c, width)
    actual = parameter_count(head)
    fraction = abs(actual-target)/target
    if fraction > c.get('capacity_tolerance', .02):
        raise ValueError('Active head capacity mismatch exceeds locked tolerance')
    return head, dict(target_head_parameters=target, actual_head_parameters=actual,
        match_fraction=fraction, internal_width=width, matching='active_parameter_count_affine_width_only')


class PositionStateModel(nn.Module):
    def __init__(self, data, c, arm):
        super().__init__()
        self.config, self.arm, self.stage = dict(c), arm, 'joint'
        self.geometry = RegionGeometry(data.assignment)
        self.register_buffer('mean', torch.tensor(float(data.mean)))
        self.register_buffer('std', torch.tensor(float(data.std)))
        for key, value in zip(('forward_graph', 'backward_graph', 'regional_graph'), data.graphs):
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))
        reverse = self.regional_graph.T
        self.register_buffer('regional_backward', reverse/reverse.sum(-1, keepdim=True).clamp_min(1e-12))
        # Keep this constructor sequence identical in every mechanism arm.
        self.position = PositionEncoder(data.position_features, data.position_categories, data.position_sizes, c['hidden'])
        self.encoder = HistoryEncoder(c, data.sensors, data.sensor_sizes)
        self.incident = IncidentEncoder(c['hidden'])
        self.origin_condition = OriginCondition(c['hidden'])
        if arm in ('ORIGIN_EXCHANGE', 'STATE_EXCHANGE'):
            self.head = CoupledHead(c)
            count = parameter_count(self.head)
            self.capacity = dict(target_head_parameters=count, actual_head_parameters=count,
                match_fraction=0., internal_width=c.get('update_hidden', 3*c['hidden']), matching='identical_origin_state_architecture')
        else:
            self.head, self.capacity = _matched_head(c, arm)

    def set_stage(self, stage):
        if stage != 'joint':
            raise ValueError('GW16 only supports joint training')
        self.stage = stage

    def forward(self, history, event):
        c, geometry = self.config, self.geometry
        if history.ndim != 4 or tuple(history.shape[1:]) != (c['history'], c['nodes'], 3):
            raise ValueError('Expected history [batch,history,nodes,flow_TOD_DOW]')
        if c.get('event_ablation', 'none') == 'no_report':
            event = dict(event, report_present=torch.zeros_like(event['report_present']))
        position = self.position()
        hidden = self.encoder(history, position, self.forward_graph, self.backward_graph)
        vector, relation, support, available = self.incident(event)
        origin = self.origin_condition(history, hidden, position, vector, relation)
        graphs = self.forward_graph, self.backward_graph, self.regional_graph, self.regional_backward
        alpha, beta, norms = self.head(hidden, origin, vector, support, position, geometry, graphs, c, self.arm)
        reference = (self.mean+self.std*history[:, -1:, :, :1]).expand(-1, c['horizon'], -1, -1)
        prediction = reference+self.std*(geometry.lift(alpha)+beta)
        return dict(prediction=prediction, reference=reference, regional=geometry.pool(reference)+self.std*alpha,
            alpha=alpha, beta=beta, state_rms=norms, support=support, available=available)


class NativePositionModel(nn.Module):
    """Original complete backbone with explicit position and certified event boundary."""
    def __init__(self, data, c, device):
        super().__init__()
        from experiments.chronological.original_igstgnn_runtime import CONFIG
        from experiments.chronological.original_igstgnn_558ab01.model import IGSTGNN
        from experiments.chronological.original_igstgnn_558ab01.graph_algo import normalize_adj_mx
        self.config, self.arm, self.stage = dict(c), 'NATIVE_POS', 'joint'
        self.geometry = RegionGeometry(data.assignment)
        self.register_buffer('mean', torch.tensor(float(data.mean)))
        self.register_buffer('std', torch.tensor(float(data.std)))
        arguments = dict(CONFIG, seq_len=c['history'], horizon=c['horizon'], data_path=str(data.directory),
            sensor_type_size=data.sensor_sizes[0], surface_size=data.sensor_sizes[1],
            roadway_use_size=data.sensor_sizes[2], device=str(device),
            adjs=[torch.as_tensor(value, dtype=torch.float32, device=device)
                  for value in normalize_adj_mx(data.native_adjacency, 'doubletransition')])
        self.position = PositionEncoder(data.position_features, data.position_categories,
                                        data.position_sizes, arguments['num_hidden'])
        self.native = IGSTGNN(arguments, node_num=c['nodes'], input_dim=3, output_dim=1,
                             seq_len=c['history'], horizon=c['horizon'])
        # Release category zero is real. Missing fields receive independent rows.
        for name in ('desc_embedding', 'incident_type_embedding'):
            previous = getattr(self.native.icsf_module, name)
            extended = nn.Embedding(previous.num_embeddings+1, previous.embedding_dim)
            with torch.no_grad():
                extended.weight[:-1].copy_(previous.weight)
            setattr(self.native.icsf_module, name, extended)
        for key, value in data.sensors.items():
            self.register_buffer('sensor_'+key, torch.as_tensor(value))
        self.capacity = dict(native_augmented=True, matched_to_mechanism_arms=False,
            position_entry='after_flow_embedding_before_ICSF', faithful_original_replay=False)

    def set_stage(self, stage):
        if stage != 'joint':
            raise ValueError('GW16 only supports joint training')
        self.stage = stage

    def _event(self, event):
        support = event['relation'].abs().sum(-1) > 0
        available = event['report_present'].bool() & event['relation_available'].bool() & support.any(-1)
        if self.config.get('event_ablation', 'none') == 'no_report':
            available = torch.zeros_like(available)
        event_type = torch.where(event['type_valid'].bool(), event['type'].long(), 6)
        description = torch.where(event['description_valid'].bool(), event['description'].long(), 29)
        zero = torch.zeros_like(event_type)
        # Incident Time, Holiday and incident_position are never read.
        payload = dict(incident=torch.stack([zero, description, event_type, zero], -1),
                       position=zero, distances=event['relation'].float()*available[:, None, None])
        return payload, support & available[:, None], available

    def forward(self, history, event):
        model = self.native
        flow, eu, ed, tod, dow = model._prepare_inputs(history)
        hidden = model.embedding(flow)+self.position()[None, None]
        payload, support, available = self._event(event)
        sensors = {key:getattr(self, 'sensor_'+key)[None].expand(len(history), -1)
                   for key in ('sensor_type', 'surface', 'roadway_use', 'road_width', 'speed_limit')}
        enhanced, incident_context = model.icsf_module(hidden, payload, sensors, tod[:, -1, 0], dow[:, -1, 0])
        hidden = torch.where(available[:, None, None, None], enhanced, hidden)
        incident_context = incident_context*available[:, None, None]
        static, dynamic = model._graph_constructor(node_embedding_u=eu, node_embedding_d=ed,
            history_data=hidden, time_in_day_feat=tod, day_in_week_feat=dow)
        residual, forecasts = hidden, []
        for layer in model.layers:
            residual, diffusion, inherent = layer(residual, dynamic, static, eu, ed, tod, dow)
            forecasts.append((diffusion, inherent))
        forecast = sum(pair[0] for pair in forecasts)+sum(pair[1] for pair in forecasts)
        forecast = model.tiid_module(forecast, incident_context)
        forecast = model.out_fc_2(F.relu(model.out_fc_1(F.relu(forecast))))
        forecast = forecast.transpose(1, 2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        normalized = forecast.transpose(1, 2).unsqueeze(-1)
        prediction = self.mean+self.std*normalized
        reference = (self.mean+self.std*history[:, -1:, :, :1]).expand(-1, self.config['horizon'], -1, -1)
        alpha, beta = self.geometry.additive_coordinates(reference, prediction, self.std)
        return dict(prediction=prediction, reference=reference, regional=self.geometry.pool(reference)+self.std*alpha,
            alpha=alpha, beta=beta, support=support, available=available,
            state_rms=history.new_zeros((self.config['horizon'], 2)))


def build_model(data, config, arm, device):
    if arm not in ARMS:
        raise ValueError('Unknown GW16 arm: '+str(arm))
    model = NativePositionModel(data, config, device) if arm == 'NATIVE_POS' else PositionStateModel(data, config, arm)
    return model.to(device)
