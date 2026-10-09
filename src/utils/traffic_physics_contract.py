"""Explicit H1 units, bottleneck identities and history-only observation mapping."""

import copy
from datetime import datetime, timedelta
import math

import numpy as np
import torch


SCHEMA = 'incident_point_queue_v1'
EVIDENCE_FIELDS = ('flow_units', 'aggregation', 'bottleneck_boundaries', 'capacity_calibration')


def contract_template(station_ids, package_hashes):
    return {
        'schema': SCHEMA, 'scope': 'conditional_offline_screening',
        'status': 'unresolved', 'main_training_ready': False,
        'station_ids': [int(x) for x in station_ids],
        'package_sha256': dict(package_hashes),
        'flow_unit': None, 'lane_basis': None,
        'interval_minutes': 5, 'interval_label': None,
        'history_steps': 12, 'gap_minutes': 10, 'horizon': 12, 'step_minutes': 1,
        'event_assignment': 'candidate_support_at_departure_not_true_impact',
        'bottlenecks': [],
        'evidence': {key: None for key in EVIDENCE_FIELDS},
    }


def validate_contract(contract, station_ids, package_hashes):
    """Validate declared evidence, not independently certify physical truth."""
    c = copy.deepcopy(contract)
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(c.get('schema') == SCHEMA and c.get('scope') == 'conditional_offline_screening', 'Unexpected physics contract schema/scope')
    require(c.get('status') == 'declared_for_conditional_screening' and c.get('main_training_ready') is False,
            'Physical units/boundaries remain unresolved; no real-data model run')
    ids = [int(x) for x in station_ids]
    require(len(ids) == len(set(ids)) and c.get('station_ids') == ids, 'Physical contract station order mismatch')
    require(c.get('package_sha256') == package_hashes, 'Physical contract package fingerprints mismatch')
    require(c.get('flow_unit') in ('vehicles_per_5min', 'vehicles_per_hour'), 'Unsupported or unconfirmed flow unit')
    require(c.get('lane_basis') == 'all_lanes_total', 'H1 requires all-lane total vehicle rates')
    require(c.get('interval_label') in ('start', 'end'), 'Aggregation label must have an explicit start/end meaning')
    require([c.get(k) for k in ('history_steps', 'gap_minutes', 'horizon', 'interval_minutes')] == [12, 10, 12, 5],
            'Contract must preserve 12 history steps, two gap slots and 12 targets')
    require(c.get('step_minutes') in (1, .5, .25), 'Unsupported internal step')
    require(c.get('event_assignment') == 'candidate_support_at_departure_not_true_impact', 'Unknown report assignment policy')
    evidence = c.get('evidence', {})
    for key in EVIDENCE_FIELDS:
        require(isinstance(evidence.get(key), str) and len(evidence[key].strip()) >= 12,
                f'Missing evidence reference: {key}')
    bottlenecks = c.get('bottlenecks')
    require(isinstance(bottlenecks, list) and bool(bottlenecks), 'At least one documented bottleneck is required')
    names, readouts = set(), set()
    for item in bottlenecks:
        require(isinstance(item, dict), 'Invalid bottleneck record')
        name = item.get('id')
        require(isinstance(name, str) and name and name not in names, 'Bottleneck IDs must be unique')
        names.add(name)
        arrival, departure = item.get('arrival_station_id'), item.get('departure_station_id')
        require(arrival in ids and departure in ids and arrival != departure, 'Bottleneck boundaries must be distinct known stations')
        require(item.get('readout_station_id') == departure and departure not in readouts,
                'First H1 readout is one unique departure detector per bottleneck')
        readouts.add(departure)
        for key in ('capacity_veh_per_hour', 'queue_scale_vehicles'):
            value = item.get(key)
            require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0,
                    f'Missing positive calibrated parameter: {key}')
        require(isinstance(item.get('boundary_evidence'), str) and len(item['boundary_evidence'].strip()) >= 12,
                'Bottleneck needs a boundary/observation evidence reference')
    return c


def check_manifest_timing(rows, interval_label=None):
    """Check nominal offsets; do not certify time zones, source latency or DST."""
    for row in rows:
        t = datetime.fromisoformat(row['t0'])
        for field, minutes in (('x_start', -65), ('x_end', -10), ('y_start', 5), ('y_end', 60)):
            if datetime.fromisoformat(row[field]) != t + timedelta(minutes=minutes):
                raise ValueError(f'Unexpected chronological offset: {field}')
    return {'nominal_offsets_checked': True, 'interval_label': interval_label,
            'unobserved_gap_minutes_if_common_5min_bins': 10,
            'first_target_label_minus_last_history_label_minutes': 15,
            'online_semantics_certified': False}


def queue_config(contract, mode):
    ids, queues = contract['station_ids'], contract['bottlenecks']
    weights = np.zeros((len(queues), len(ids)), dtype=np.float32)
    for i, item in enumerate(queues):
        weights[i, ids.index(item['readout_station_id'])] = 1
    return {'capacities': [x['capacity_veh_per_hour'] for x in queues],
            'queue_scales': [x['queue_scale_vehicles'] for x in queues],
            'readout_weights': weights, 'mode': mode,
            'step_minutes': contract['step_minutes'], 'gap_minutes': contract['gap_minutes'],
            'interval_minutes': contract['interval_minutes'], 'horizon': contract['horizon']}


def build_physical_inputs(history_flow, history_valid, incident, contract):
    """Accept only raw X [B,12,N], never a 26-slot window or a whole batch."""
    if (history_flow.ndim != 3 or history_flow.shape[1:] != (12, len(contract['station_ids']))
            or history_valid.shape != history_flow.shape or history_valid.dtype != torch.bool):
        raise ValueError('Physical mapping accepts only history [B,12,N] and bool mask')
    if not torch.isfinite(history_flow[history_valid]).all() or (history_flow[history_valid] < 0).any():
        raise ValueError('Invalid values marked as observed history')
    ids, queues = contract['station_ids'], contract['bottlenecks']
    up = [ids.index(x['arrival_station_id']) for x in queues]
    down = [ids.index(x['departure_station_id']) for x in queues]
    rate = torch.stack([history_flow[:, :, up], history_flow[:, :, down]], -1)
    mask = torch.stack([history_valid[:, :, up], history_valid[:, :, down]], -1)
    factors = {'vehicles_per_5min': 12., 'vehicles_per_hour': 1.}
    if contract['flow_unit'] not in factors:
        raise ValueError('Unconfirmed flow unit cannot be passed to a physical branch')
    rate = torch.where(mask, rate, 0) * factors[contract['flow_unit']]
    d = incident['distances'][:, down]
    age = incident['report_age_minutes'].reshape(-1, 1, 1).expand(-1, len(queues), 1)
    if not torch.isfinite(age).all() or ((age <= 0) | (age > 5)).any():
        raise ValueError('Expected report age in (0,5] minutes, not incident duration')
    features = torch.cat([age / 5, d], -1).to(rate.dtype)
    return {'history_rates': rate, 'history_valid': mask,
            'event_features': features, 'report_support': d.abs().sum(-1) > 0}
