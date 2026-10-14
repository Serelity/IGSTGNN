"""A pruned MIN statistical encoder with one open stock--flow output path.

The old per-layer latent exchanges and history budget are absent.  The native
forecast head supplies requested outflow; it is never added to decoder output.
Stocks use author-flow units times nominal slots, not certified vehicle counts.
No real road-capacity, cell geometry, event-time or CFL claim is made.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.incident_spatial import require
from src.models.incident_spatial_minimal import MinimalIGSTGNN
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.igstgnn import InhBlock
from src.models.open_stock_flow import step


SCHEMA = 'MIN_single_open_stock_flow_v1'
MODES = ('state', 'location')
UNIT_SCOPE = 'author_flow_scale_times_nominal_slot_not_certified_vehicles'
TIME_SCOPE = 'ordered_published_target_slots_not_certified_event_relative_minutes'


def _mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, outputs))


class StatisticalMIN(MinimalIGSTGNN):
    """The compatible statistical computation, with every old exchange pruned."""
    def forward(self, history_data):
        history_data, node_u, node_d, tod, dow = self._prepare_inputs(history_data)
        encoded = self.embedding(history_data)
        history = encoded.clone()
        history[:, -1] = self.icsf_module.output_norm(encoded[:, -1])
        static, dynamic = self._graph_constructor(node_embedding_u=node_u,
            node_embedding_d=node_d, history_data=history,
            time_in_day_feat=tod, day_in_week_feat=dow)
        dif, inh, residual = [], [], history
        for layer in self.layers:
            residual, left, right = layer(residual, dynamic, static, node_u, node_d,
                                         tod, dow, incident_inputs=None, spatial_context=None)
            dif.append(left)
            inh.append(right)
        return self._decode_forecast(sum(dif)+sum(inh), None)


class ForecastOnlyInhBlock(InhBlock):
    """The final layer has no consumer for its otherwise computed backcast."""
    def forward(self, hidden_inherent_signal):
        rnn = self.pos_encoder(self.rnn_layer(hidden_inherent_signal))
        hidden = self.transformer_layer(rnn, rnn, rnn)
        forecast = self.forecast_block(hidden_inherent_signal, rnn, hidden,
                                      self.transformer_layer, self.rnn_layer, self.pos_encoder)
        return hidden_inherent_signal, forecast


def prune_statistical_MIN(base):
    require(type(base) is MinimalIGSTGNN, 'Expected an existing MinimalIGSTGNN source')
    before = tensor_state_hash(base.state_dict())
    with torch.inference_mode(False):
        result = copy.deepcopy(base)
        result.__class__ = StatisticalMIN
        del result.report_spatial_state
        del result.internal_spatial_graph
        for layer in result.layers:
            exchange = layer.dif_layer.localized_st_conv
            require(hasattr(exchange, 'incident_exchange'), 'Source MIN lacks its expected old exchange')
            del exchange.incident_exchange
            # Constructed in the released code, but never read by its forward.
            del exchange.bn
        del result.dynamic_graph_constructor.distance_function.time_slot_embedding
        final_inherent = result.layers[-1].inh_layer
        final_inherent.__class__ = ForecastOnlyInhBlock
        del final_inherent.backcast_fc
        del final_inherent.residual_decompose
        result._internal_spatial_mode = 'native'
        result._minimal_variant = 'statistical_encoder_only_stock_flow_demand'
    parent, retained = base.state_dict(), result.state_dict()
    removed = sorted(set(parent)-set(retained))
    require(bool(removed) and set(retained) < set(parent), 'Old physical path was not pruned')
    require(all(torch.equal(value, parent[name]) for name, value in retained.items()),
            'Compatible warm-start tensor changed during pruning')
    final_prefix = 'layers.'+str(len(result.layers)-1)+'.inh_layer.'
    dead_suffixes = ('.localized_st_conv.bn.',)
    require(all(name.startswith(('report_spatial_state.', 'internal_spatial_graph.',
                    'dynamic_graph_constructor.distance_function.time_slot_embedding.'))
                or '.incident_exchange.' in name or any(value in name for value in dead_suffixes)
                or name.startswith((final_prefix+'backcast_fc.', final_prefix+'residual_decompose.'))
                for name in removed), 'Unexpected source parameter was pruned')
    require(before == tensor_state_hash(base.state_dict()), 'Source MIN was modified')
    return result, dict(source_state_sha256=before,
        retained_state_sha256=tensor_state_hash(retained), retained_state_exact_verified=True,
        removed_state_keys=removed, compatible_warm_start=True,
        exact_source_prediction_initialization=False,
        initialization_scope='compatible_statistical_tensors_not_complete_source_function')


def complete_open_boundaries(edge_index, nodes):
    """Keep candidate edges; add explicit exterior links where a side is absent."""
    require(edge_index.dtype == torch.long and edge_index.ndim == 2 and edge_index.shape[0] == 2,
            'Candidate edge index must be Long[2,E]')
    require(bool(((edge_index >= -1) & (edge_index < nodes)).all()), 'Candidate graph endpoint out of range')
    pairs = [tuple(pair) for pair in edge_index.detach().cpu().t().tolist()]
    require(len(set(pairs)) == len(pairs) and all(a != b for a, b in pairs),
            'Duplicate or self candidate edge')
    require(all((a >= 0 or b >= 0) for a, b in pairs), 'Exterior-to-exterior candidate edge')
    incoming = {b for _, b in pairs if b >= 0}
    outgoing = {a for a, _ in pairs if a >= 0}
    additions = [(-1, node) for node in range(nodes) if node not in incoming]
    additions += [(node, -1) for node in range(nodes) if node not in outgoing]
    result = torch.tensor(pairs+additions, dtype=torch.long, device=edge_index.device).t().contiguous()
    return result, additions


def stock_flow_common_hash(decoder):
    return tensor_state_hash({name: value for name, value in decoder.state_dict().items()
                              if not name.startswith('relation_projection.')})


class OpenStockFlowDecoder(nn.Module):
    def __init__(self, width, mode, flow_scale, edge_index, recipe):
        super().__init__()
        require(mode in MODES, 'Unknown stock-flow conditioning mode')
        self.mode = mode
        self.steps = recipe.get('steps', 2)
        self.dt = float(recipe.get('dt', .5))
        self.storage_slots = float(recipe.get('storage_slots', 4.))
        self.service_multiplier = float(recipe.get('service_multiplier', 1.))
        self.boundary_max_factor = float(recipe.get('boundary_max_factor', 4.))
        self.condition_log_bound = float(recipe.get('condition_log_bound', .5))
        feature_width = recipe.get('feature_width', 16)
        rank = recipe.get('interaction_rank', 8)
        require(all(type(value) is int and value > 0 for value in (width, self.steps, feature_width, rank)),
                'Stock-flow feature dimensions and steps must be positive integers')
        require(self.steps > 0 and math.isclose(self.steps*self.dt, 1., rel_tol=0., abs_tol=1e-12),
                'Substeps must cover exactly one nominal output slot')
        require(all(math.isfinite(v) and v > 0 for v in (self.dt, self.storage_slots,
                self.service_multiplier, self.boundary_max_factor, self.condition_log_bound)),
                'Stock-flow recipe bounds must be finite and positive')
        self.register_buffer('flow_scale', flow_scale.detach().clone())
        self.register_buffer('edge_index', edge_index.detach().clone())
        src, dst = edge_index
        nodes = flow_scale.numel()
        outgoing = torch.bincount(src[src >= 0], minlength=nodes).to(flow_scale.dtype)
        entries = torch.bincount(dst[src < 0], minlength=nodes).to(flow_scale.dtype)
        self.register_buffer('outgoing_count', outgoing)
        self.register_buffer('entry_count', entries)
        self.initial_stock = _mlp(5, 1, feature_width)
        self.feature_projection = nn.Linear(width, feature_width)
        # Current stock fraction and ordered target-slot position are explicit.
        self.service_prior = _mlp(feature_width+2, 1, feature_width)
        # Start around one training-median flow unit, while retaining the wider
        # [.25,4] model range for observed peaks.  This is an initialization
        # hypothesis, not capacity calibration or a binding guarantee.
        nn.init.constant_(self.service_prior[-1].bias, math.log(.75/3.))
        self.boundary_arrival = _mlp(feature_width+7, 1, feature_width)
        self.condition_state = _mlp(feature_width+3, rank, feature_width)
        self.condition_output = nn.Linear(rank, 1)
        nn.init.zeros_(self.condition_output.weight)
        nn.init.zeros_(self.condition_output.bias)
        # Instantiate D-specific weights LAST to preserve seeded common state.
        if mode == 'location':
            self.relation_projection = nn.Linear(3, rank, bias=False)

    def history_summary(self, history, mean, std):
        flow = (history[..., :1]*std+mean)/self.flow_scale[None, None, :, None]
        return torch.cat((flow[:, -1], flow.mean(1), flow[:, -1]-flow[:, 0],
                          history[:, -1, :, 1:3]), -1)

    def forward(self, history, features, proposal, relation, support, *, mean, std, collect=False):
        batch, horizon, nodes, _ = proposal.shape
        scale = self.flow_scale[None, :, None]
        storage = (self.storage_slots*scale).expand(batch, -1, -1)
        summary = self.history_summary(history, mean, std)
        stock = storage*self.initial_stock(summary).sigmoid()
        src, dst = self.edge_index
        entry = src < 0
        backlog = history.new_zeros(batch, src.numel(), 1)
        outputs, rows = [], []
        max_stats, sum_stats = {}, {}
        for slot in range(horizon):
            hidden = self.feature_projection(features[:, slot])
            position = stock.new_full(stock.shape, (slot+1)/horizon)
            # Arrival process is history/H/role conditioned; D never enters it.
            role = (self.entry_count > 0).to(history.dtype)[None, :, None].expand(batch, -1, -1)
            arrivals = self.boundary_max_factor*scale*self.boundary_arrival(
                torch.cat((hidden, summary, role, position), -1)).sigmoid()
            native_demand = proposal[:, slot]
            slot_outflow = torch.zeros_like(stock)
            slot_service_change, slot_service_active = [], []
            for _ in range(self.steps):
                stock_fraction = stock/storage
                prior_features = torch.cat((hidden, stock_fraction, position), -1)
                base_service = self.service_multiplier*scale*(.25+3.75*self.service_prior(prior_features).sigmoid())
                condition_features = torch.cat((prior_features, base_service/scale), -1)
                interaction = self.condition_state(condition_features)
                if self.mode == 'location':
                    interaction = interaction*(1+self.relation_projection(relation).tanh())
                gamma = self.condition_output(interaction)
                factor = torch.exp(self.condition_log_bound*gamma.tanh())
                service = torch.where(support, base_service*factor, base_service)
                internal_demand = native_demand[:, src.clamp_min(0)]/self.outgoing_count[src.clamp_min(0)][None, :, None]
                entry_demand = arrivals[:, dst.clamp_min(0)]/self.entry_count[dst.clamp_min(0)].clamp_min(1)[None, :, None]
                demand = torch.where(entry[None, :, None], entry_demand, internal_demand)
                record = step(stock, storage, service, demand, backlog, self.edge_index, self.dt)
                slot_outflow = slot_outflow+self.dt*record['outflow']
                stock, backlog = record['stock'], record['external_backlog']
                if collect:
                    slot_service_change.append((service-base_service).abs().mean().detach())
                    slot_service_active.append(record['stats']['service_active_count'])
                    for name, value in record['stats'].items():
                        if name.endswith('_max'):
                            max_stats[name] = torch.maximum(max_stats.get(name, torch.zeros_like(value)), value)
                        else:
                            sum_stats[name] = sum_stats.get(name, torch.zeros_like(value))+value
            outputs.append(slot_outflow)
            if collect:
                rows.append(dict(slot=slot+1,
                    service_change_abs_mean=float(torch.stack(slot_service_change).mean().cpu()),
                    service_active_count=float(torch.stack(slot_service_active).sum().cpu()),
                    stock_mean=float(stock.detach().mean().cpu()),
                    external_backlog_sum=float(backlog.detach().sum().cpu()),
                    output_flow_mean=float(slot_outflow.detach().mean().cpu())))
        flow = torch.stack(outputs, 1)
        diagnostics = {}
        if collect:
            diagnostics = {name: float(value.detach().cpu()) for name, value in {**max_stats, **sum_stats}.items()}
            diagnostics.update(by_slot=rows,
                final_stock_sum=float(stock.detach().sum().cpu()),
                final_external_backlog_sum=float(backlog.detach().sum().cpu()),
                service_change_abs_mean=sum(row['service_change_abs_mean'] for row in rows)/horizon)
        return flow, diagnostics


class IncidentStockFlow(nn.Module):
    def __init__(self, backbone, decoder, mean, std):
        super().__init__()
        self.backbone, self.decoder = backbone, decoder
        self.scaler_mean, self.scaler_std = float(mean), float(std)

    def report_context(self, history, incident, enabled):
        batch, _, nodes, _ = history.shape
        relation = history.new_zeros(batch, nodes, 3)
        support = torch.zeros(batch, nodes, 1, device=history.device, dtype=torch.bool)
        if enabled and incident is not None:
            require(isinstance(incident, dict), 'Incident input must be a dictionary')
            relation = incident.get('relation')
            require(isinstance(relation, torch.Tensor) and relation.shape == (batch, nodes, 3)
                    and relation.device == history.device and relation.dtype == history.dtype
                    and bool(torch.isfinite(relation).all()), 'Incident relation axes/dtype/device mismatch')
            available = torch.ones(batch, dtype=torch.bool, device=history.device)
            for name in ('relation_available', 'report_present'):
                flag = incident.get(name)
                require(isinstance(flag, torch.Tensor) and flag.shape == (batch,)
                        and flag.dtype == torch.bool and flag.device == history.device,
                        'Incident '+name+' must be Bool[B]')
                available = available & flag
            support = (relation.abs().sum(-1, keepdim=True)>0) & available[:, None, None]
        return relation, support

    def forward(self, history_data, incident_data=None, *, report_enabled=True, return_details=False):
        require(type(report_enabled) is bool and type(return_details) is bool, 'Stock-flow switches must be bool')
        require(history_data.ndim == 4 and history_data.shape[-1] == 3
                and bool(torch.isfinite(history_data).all()), 'Expected finite [B,T,N,3] history')
        relation, support = self.report_context(history_data, incident_data, report_enabled)
        captured = []
        def capture(module, args):
            captured.append(args[0])
        handle = self.backbone.out_fc_1.register_forward_pre_hook(capture)
        try:
            native_prediction = self.backbone(history_data)
        finally:
            handle.remove()
        require(len(captured) == 1 and captured[0].shape[:3] == native_prediction.shape[:3],
                'Expected one activated target-slot feature capture')
        # A smooth positive request, in author flow scale; no output bypass.
        proposal = F.softplus(native_prediction*self.scaler_std+self.scaler_mean)
        flow, diagnostics = self.decoder(history_data, captured[0], proposal, relation, support,
            mean=self.scaler_mean, std=self.scaler_std, collect=return_details)
        prediction = (flow-self.scaler_mean)/self.scaler_std
        require(bool(torch.isfinite(prediction).all()), 'Stock-flow prediction became non-finite')
        if return_details:
            diagnostics.update(schema=SCHEMA, mode=self.decoder.mode,
                unit_scope=UNIT_SCOPE, time_scope=TIME_SCOPE, physical_certified=False,
                report_enabled=report_enabled, supported_node_count=int(support.sum().detach().cpu()),
                output_scope='sum_actual_node_outflow_dt_no_native_prediction_residual',
                off_reference='same_candidate_single_decoder_condition_disabled_not_source_MIN',
                steps=self.decoder.steps, dt=self.decoder.dt,
                conditional_service_only=True, initial_stock_history_only=True,
                boundary_arrivals_read_numeric_D=False)
            return dict(prediction=prediction, flow_prediction=flow,
                        demand_proposal=proposal, stock_flow=diagnostics, mechanisms=diagnostics)
        return prediction


def build_stock_flow(base, mode='location', flow_scale=None, recipe=None):
    """Transfer compatible source tensors; jointly train a new independent model."""
    recipe = {} if recipe is None else dict(recipe)
    require(mode in MODES, 'Unknown stock-flow conditioning mode')
    require(not recipe.get('strict_physical', False) and not recipe.get('strict_physics', False)
            and recipe.get('physics_mode') != 'strict',
            'Strict real-traffic physics requires a certified observation/grid contract')
    mean, std = recipe.get('scaler_mean'), recipe.get('scaler_std')
    require(isinstance(mean, (float, int)) and not isinstance(mean, bool) and math.isfinite(float(mean))
            and isinstance(std, (float, int)) and not isinstance(std, bool)
            and math.isfinite(float(std)) and std > 0, 'Recipe must provide finite source scaler_mean/std')
    require(isinstance(flow_scale, torch.Tensor) and flow_scale.shape == (base.node_num,)
            and flow_scale.dtype in (torch.float32, torch.float64) and bool(torch.isfinite(flow_scale).all())
            and bool((flow_scale > 0).all()), 'flow_scale must be finite positive Tensor[N] fitted on train only')
    source_hash = tensor_state_hash(base.state_dict())
    parameter = next(base.parameters())
    require(parameter.dtype in (torch.float32, torch.float64), 'Stock-flow source must use float32 or float64')
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        backbone, transfer = prune_statistical_MIN(base)
        edge_index, additions = complete_open_boundaries(base.internal_spatial_graph.edge_index, base.node_num)
        decoder = OpenStockFlowDecoder(base.out_fc_1.in_features, mode,
            flow_scale.to(device=parameter.device, dtype=parameter.dtype), edge_index, recipe)
        decoder.to(device=parameter.device, dtype=parameter.dtype)
        model = IncidentStockFlow(backbone, decoder, mean, std).requires_grad_(True)
        for value in model.parameters():
            value.grad = None
    require(source_hash == tensor_state_hash(base.state_dict()), 'Source MIN was modified')
    src, dst = edge_index
    count = lambda module: sum(value.numel() for value in module.parameters())
    certificate = dict(schema=SCHEMA, mode=mode, **transfer,
        common_initial_sha256=stock_flow_common_hash(decoder),
        decoder_initial_sha256=tensor_state_hash(decoder.state_dict()),
        trainable_parameters=count(model), parameters=count(model),
        backbone_parameters=count(backbone), decoder_parameters=count(decoder),
        trainable_names=[name for name, value in model.named_parameters() if value.requires_grad],
        pruned_names=transfer['removed_state_keys'], physical_paths=1, trajectories=1,
        old_history_P_removed=True, old_layer_exchanges_removed=True,
        native_q_head_scope='positive_requested_outflow_only_not_output_bypass',
        compatible_tensors_exact=True, source_preserved=True,
        output_scope='sum_actual_node_outflow_dt_no_native_prediction_residual',
        location_values_used=mode == 'location', support_mask_equal_across_modes=True,
        support_scope='unchanged_original_nonzero_relation_node_mask',
        state_control_scope='support_matched_state_conditioning_not_no_incident',
        added_boundary_edges=[list(pair) for pair in additions], added_boundary_count=len(additions),
        internal_edge_count=int(((src >= 0)&(dst >= 0)).sum()),
        entry_edge_count=int((src < 0).sum()), exit_edge_count=int((dst < 0).sum()),
        graph_scope='candidate_metadata_connections_plus_explicit_missing_side_boundaries_not_certified_cells',
        routing_scope='fixed_equal_requested_flow_split_over_each_nodes_outgoing_edges',
        service_factor_range=[.25, 4.], service_multiplier=decoder.service_multiplier,
        conditional_service_factor_range=[math.exp(-decoder.condition_log_bound), math.exp(decoder.condition_log_bound)],
        boundary_max_factor=decoder.boundary_max_factor, storage_slots=decoder.storage_slots,
        steps=decoder.steps, dt=decoder.dt, physical_certified=False,
        unit_scope=UNIT_SCOPE, time_scope=TIME_SCOPE,
        capacity_scope='effective_model_service_not_identified_real_road_capacity',
        service_initial_bias_scope='approximately_one_train_median_not_capacity_calibration',
        additional_dead_parameter_cleanup=['unused_localized_batchnorm', 'unused_time_slot_linear',
                                           'unconsumed_final_inherent_backcast'],
        optimizer_parameter_scope='complete_independent_candidate_joint_training')
    return model, certificate
