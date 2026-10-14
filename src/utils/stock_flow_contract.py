"""Evidence and units for the experimental output-domain stock-flow model.

This contract deliberately does not relax traffic_physics_contract.py. Candidate
connections may define a conditional model, but are never self-certified as a
vehicle-conservation network. No targets are needed to fit the model's scale.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'open_stock_flow_model_units_contract_r2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_graph_contract(graph_path, source_summary_path, station_ids=None, *, strict_physical=False):
    """Audit the current candidate graph without inventing a physical mask.

    A stricter traffic model needs an independently supplied, versioned physical
    certificate. Setting Boolean fields in this exploratory graph cannot create
    that certificate; strict mode is deliberately unavailable through this API.
    """
    require(type(strict_physical) is bool, 'strict_physical must be Boolean')
    graph_path, source_summary_path = Path(graph_path), Path(source_summary_path)
    graph = json.loads(graph_path.read_text(encoding='utf-8'))
    source = json.loads(source_summary_path.read_text(encoding='utf-8'))
    ids = graph.get('station_ids')
    require(isinstance(ids, list) and ids and all(type(x) is int for x in ids)
            and len(set(ids)) == len(ids), 'Invalid candidate station axis')
    if station_ids is not None:
        require(np.array_equal(np.asarray(ids), np.asarray(station_ids)), 'Candidate station axis differs')
    n = len(ids)
    require(source.get('nodes') == n, 'Source summary node count differs')
    mask = graph.get('operator_mask')
    require(isinstance(mask, list) and len(mask) == n
            and all(type(x) is bool for x in mask), 'Expected explicit Boolean candidate coverage mask')
    incoming, outgoing = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    internal, entries, exits = 0, 0, 0
    signatures = set()
    edges = graph.get('edges')
    require(isinstance(edges, list), 'Expected candidate edge list')
    for e, edge in enumerate(edges):
        a, b = edge.get('source'), edge.get('destination')
        require(type(a) is int and type(b) is int and -1 <= a < n and -1 <= b < n
                and (a >= 0 or b >= 0) and a != b, 'Invalid directed candidate edge')
        require(edge.get('index') == e, 'Candidate edge order/index mismatch')
        signature = (a, b)
        require(signature not in signatures, 'Duplicate numeric candidate connection')
        signatures.add(signature)
        if a >= 0:
            outgoing[a] += 1
        if b >= 0:
            incoming[b] += 1
        internal += int(a >= 0 and b >= 0)
        entries += int(a == -1)
        exits += int(b == -1)
    coverage = int(sum(mask))
    require(coverage == source.get('metadata_chain_nodes')
            and internal == source.get('metadata_internal_edges')
            and entries+exits == source.get('metadata_boundary_edges'),
            'Candidate graph counts disagree with recorded evidence')
    joint = source.get('jointly_certified_nodes')
    require(type(joint) is int and 0 <= joint <= n, 'Missing joint physical certification count')
    units = source.get('source_provenance', {})
    gaps = ['detector_to_storage_cell_observation_mapping', 'flow_unit_and_lane_aggregation',
            'physical_link_lengths_and_storage', 'boundary_and_ramp_flow_time_series',
            'turning_or_routing_fractions', 'density_or_speed_occupancy_mapping',
            'as_of_incident_attributes_and_active_duration']
    result = dict(schema=SCHEMA, evidence_scope='conditional_model_assumptions_not_vehicle_certification',
        graph_sha256=_sha256(graph_path), source_summary_sha256=_sha256(source_summary_path),
        nodes=n, candidate_covered_nodes=coverage, jointly_certified_nodes=joint,
        internal_edges=internal, observed_metadata_entry_edges=entries,
        observed_metadata_exit_edges=exits,
        open_storage_entry_completions=np.flatnonzero(incoming == 0).tolist(),
        open_storage_exit_completions=np.flatnonzero(outgoing == 0).tolist(),
        boundary_completion_scope='explicit_model_fallback_no_new_claimed_road_connection',
        direction_rule=graph.get('direction_rule'), routing_rule=graph.get('outgoing_allocation'),
        physical_grid_certified=graph.get('physical_grid_certified') is True,
        direct_topology_certified=graph.get('direct_topology_certified') is True,
        historical_multichannel_semantics=units,
        published_model_inputs=['flow', 'nominal_time_of_day', 'nominal_day_of_week', 'report_relation'],
        time_unit='one_ordered_published_target_slot_not_certified_minutes',
        rate_unit='author_flow_scale_per_nominal_target_slot',
        stock_unit='author_flow_scale_times_nominal_target_slot',
        observed_flow_mapping='model_assumption_total_node_outflow_per_target_slot',
        physical_capacity_identified=False, physical_queue_identified=False,
        real_spillback_identified=False,
        neural_requested_outflow_is_additional_limit=True,
        work_conserving_classical_queue=False,
        represented_open_boundaries_learned_not_zero=True,
        unrepresented_model_sources_zero_by_model_definition=True,
        absence_of_real_unrepresented_sources_certified=False,
        strict_physical_ready=False, missing_physical_evidence=gaps,
        test_targets_read=False)
    require(not strict_physical,
            'Strict physical mode requires an independent observation/unit/topology/boundary certificate; '
            f'this candidate graph has {joint}/{n} jointly certified nodes')
    return result


def fit_training_flow_scale(history_normalized, mean, std, *, peak_margin=1.25, max_rate_factor=4.):
    """Train-X-only numerical scale with a historical peak envelope.

    Repeated released sliding-window observations retain their training-window
    multiplicities. Scale is max(positive median or global fallback,
    peak_margin * positive historical max / max_rate_factor).  The ceiling
    max_rate_factor*scale therefore covers known TRAIN-X peaks without needing
    an incident factor.  This is a numerical envelope, not a physical capacity
    estimate, unit conversion, or guarantee about future flow peaks. Zero-only
    stations use the global positive median, or one if the full train X is zero.
    """
    x = np.asarray(history_normalized)
    require(x.ndim in (3, 4) and x.shape[0] > 0 and x.shape[1] == 12 and x.shape[2] > 0
            and (x.ndim == 3 or x.shape[3] == 3) and np.issubdtype(x.dtype, np.floating)
            and np.isfinite(x).all(), 'Expected finite training flow[sample,12,station] or X[sample,12,station,3]')
    require(np.ndim(mean) == np.ndim(std) == 0 and np.isfinite(mean) and np.isfinite(std)
            and std > 0, 'Invalid source flow scaler')
    require(not isinstance(peak_margin, (bool, np.bool_)) and np.ndim(peak_margin) == 0
            and np.isfinite(peak_margin) and peak_margin >= 1,
            'peak_margin must be finite and at least one')
    require(not isinstance(max_rate_factor, (bool, np.bool_)) and np.ndim(max_rate_factor) == 0
            and np.isfinite(max_rate_factor) and max_rate_factor > 0,
            'max_rate_factor must be finite and positive')
    peak_margin, max_rate_factor = float(peak_margin), float(max_rate_factor)
    historical_flow = x[..., 0] if x.ndim == 4 else x
    with np.errstate(over='ignore', invalid='ignore'):
        flow = historical_flow.astype(np.float64)*float(std)+float(mean)
    require(np.isfinite(flow).all(), 'Unscaled training flow overflowed')
    flat = flow.reshape(-1, flow.shape[-1])
    positive = flat[flat > 0]
    global_scale = float(np.median(positive)) if positive.size else 1.
    fallback, medians, maxima, stations = [], [], [], []
    for i in range(flat.shape[1]):
        observed = flat[:, i][flat[:, i] > 0]
        if observed.size:
            median = float(np.median(observed))
            maximum = float(observed.max())
            p95, p99 = (float(value) for value in np.percentile(observed, [95, 99]))
        else:
            median, maximum, p95, p99 = None, None, None, None
            fallback.append(i)
        medians.append(median if median is not None else global_scale)
        maxima.append(maximum if maximum is not None else 0.)
        stations.append(dict(station_index=i,
            positive_observation_status='available' if observed.size else 'no_positive_training_observations',
            positive_observation_count=int(observed.size), raw_positive_median=median,
            positive_max=maximum, positive_p95=p95, positive_p99=p99,
            median_or_global_fallback=medians[-1]))
    medians, maxima = np.asarray(medians, dtype=np.float64), np.asarray(maxima, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        floor = peak_margin*maxima/max_rate_factor
        desired = np.maximum(medians, floor)
        scale = desired.astype(np.float32)
    require(np.isfinite(scale).all() and np.all(scale > 0), 'Training scales cannot be represented as positive float32')
    # Round upward if float32 conversion would violate the numerical envelope.
    scale = np.where(scale.astype(np.float64) < desired,
                     np.nextafter(scale, np.float32(np.inf)), scale).astype(np.float32)
    require(np.isfinite(scale).all(), 'Rounded training scales overflow float32')
    expanded = np.flatnonzero(floor > medians).tolist()
    before_scale = medians.astype(np.float32)
    require(np.isfinite(before_scale).all() and (before_scale > 0).all(),
            'Median fallback scale cannot be represented as positive float32')
    before_ceiling = max_rate_factor*before_scale.astype(np.float64)
    after_ceiling = max_rate_factor*scale.astype(np.float64)
    before_counts = (flat > before_ceiling[None, :]).sum(0)
    after_counts = (flat > after_ceiling[None, :]).sum(0)
    for row in stations:
        i = row['station_index']
        row.update(peak_floor=float(floor[i]), expanded_scale=float(scale[i]),
                   expanded_by_peak_floor=i in expanded,
                   base_service_ceiling_before=float(before_ceiling[i]),
                   base_service_ceiling_after=float(after_ceiling[i]),
                   above_ceiling_history_points_before=int(before_counts[i]),
                   above_ceiling_history_points_after=int(after_counts[i]))
    def envelope_audit(counts, ceiling):
        p95_count = sum(row['positive_p95'] is not None and row['positive_p95'] > ceiling[row['station_index']]
                        for row in stations)
        p99_count = sum(row['positive_p99'] is not None and row['positive_p99'] > ceiling[row['station_index']]
                        for row in stations)
        exceed_count = int(counts.sum())
        return dict(history_points=int(flat.size), positive_history_points=int(positive.size),
            above_ceiling_history_points=exceed_count,
            fraction_all_history_points=exceed_count/flat.size,
            fraction_positive_history_points=exceed_count/positive.size if positive.size else None,
            stations_with_any_exceedance=int((counts > 0).sum()),
            stations_positive_p95_above_ceiling=int(p95_count),
            stations_positive_p99_above_ceiling=int(p99_count),
            maximum_positive_flow_over_ceiling=float(np.max(maxima/ceiling)))
    selected = np.ascontiguousarray(historical_flow)
    provenance = dict(schema='training_history_positive_median_peak_envelope_scale_v2',
        fit_scope='training_X_only_unscaled_historical_flow', samples=x.shape[0],
        history_slots=x.shape[1], nodes=x.shape[2], source_mean=float(mean), source_std=float(std),
        historical_flow_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
        historical_flow_dtype=str(selected.dtype),
        scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest(),
        zero_only_fallback_nodes=fallback, global_fallback_scale=global_scale,
        scale_policy='max_positive_median_or_globalfallback_and_trainX_positive_max_times_margin_divided_by_max_rate_factor',
        envelope_scope='numeric_envelope_not_physical_capacity_or_future_peak_guarantee',
        peak_margin=peak_margin, max_rate_factor=max_rate_factor,
        before_ceiling_scale_scope='float32_positive_median_or_global_fallback_as_prior_v1',
        expanded_nodes=expanded, expanded_node_count=len(expanded), stations=stations,
        base_service_ceiling_audit_before=envelope_audit(before_counts, before_ceiling),
        base_service_ceiling_audit_after=envelope_audit(after_counts, after_ceiling),
        conditional_D_factor_required_to_cover_trainX_peaks=False,
        repeated_window_observations='training_window_multiplicities_retained',
        validation_values_used=False, future_targets_used=False, physical_capacity_estimate=False)
    return scale, provenance
