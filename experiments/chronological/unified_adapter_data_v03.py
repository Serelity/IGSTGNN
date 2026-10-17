"""Auditable train/validation report batches for the v0.3 candidate models.

The frozen original loader remains the ORIGINAL arm's parser.  Candidate
loaders read its *raw samples*, before missing categories become category zero.
They do not mutate the source samples/loaders, and do not inspect test targets.
Availability inferred from a published field's existence is an explicit
assumption, never proof that the report was published at the forecast cutoff.
"""
from collections import Counter
from collections.abc import Mapping

import numpy as np
import torch


SCHEMA = 'unified_adapter_data_contract_v03'
CATEGORY_SIZES = {'Description': 29, 'Type': 6, 'Holiday': 2}
INFORMATION_MODES = ('H', 'R', 'RC')
IDENTITY_FIELDS = ('incident_id', 'event_id', 'window_start', 'window_end',
    'forecast_cutoff', 'report_published_at', 'report_updated_at',
    'incident_start', 'incident_end', 'clearance_time', 'absolute_timestamp')


def category(value, count):
    """Preserve legal integer zero; route missing/noninteger/invalid to -1."""
    if value is None:
        return -1, 'missing'
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return -1, 'invalid_type'
    if not np.isfinite(value):
        return -1, 'nonfinite'
    if value != int(value):
        return -1, 'noninteger'
    if not 0 <= int(value) < count:
        return -1, 'out_of_range'
    return int(value), 'valid'


def _flag(sample, names, default):
    values = [(name, sample[name]) for name in names if name in sample]
    if not values:
        return bool(default), 'inferred_from_field_presence_not_cutoff_certified'
    parsed = []
    for name, value in values:
        if not isinstance(value, (bool, np.bool_, int, np.integer)) or value not in (0, 1):
            raise ValueError(name + ' must be a Boolean or integer 0/1')
        parsed.append(bool(value))
    if len(set(parsed)) != 1:
        raise ValueError('Conflicting availability flags: ' + ', '.join(names))
    return parsed[0], '+'.join(name for name, _ in values)


def _features(sample):
    raw = sample.get('incident_features')
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return raw
    raw = np.asarray(raw, dtype=object)
    if raw.shape != (4,):
        raise ValueError('incident_features must be a dictionary or a four-value vector')
    return dict(zip(('Incident Time', 'Description', 'Type', 'Holiday'), raw))


def encode_report(sample, nodes):
    """Normalize a raw sample without consuming x/y values or future duration."""
    available, available_source = _flag(sample,
        ('available', 'report_available', 'report_present'),
        'incident_features' in sample or 'incident_distances' in sample)
    relation_available, relation_source = _flag(sample, ('relation_available',),
        'incident_distances' in sample)
    if available and relation_available:
        if 'incident_distances' not in sample:
            raise ValueError('relation_available=True but incident_distances is absent')
        relation = np.asarray(sample['incident_distances'], dtype=np.float32)
        if relation.shape != (nodes, 3) or not np.isfinite(relation).all():
            raise ValueError('Available incident relation must be finite with shape [nodes,3]')
        relation = relation.copy()
    else:
        # Do not multiply NaN/inf placeholders by zero: replace before any math.
        relation = np.zeros((nodes, 3), dtype=np.float32)
    try:
        features = _features(sample)
    except (TypeError, ValueError):
        if available:
            raise
        # An absent report may use a scalar NaN placeholder for the whole record.
        features = {}
    reasons = {}
    values = np.zeros(4, dtype=np.float32)
    for index, name in ((1, 'Description'), (2, 'Type'), (3, 'Holiday')):
        values[index], reasons[name] = category(features.get(name), CATEGORY_SIZES[name])
    position, reasons['position'] = category(sample.get('incident_position'), 12)
    return dict(incident=values, position=position, distances=relation,
        available=available, relation_available=relation_available,
        category_valid=np.asarray([reasons[name] == 'valid'
            for name in ('Description', 'Type', 'Holiday', 'position')], dtype=np.bool_),
        category_reasons=reasons, available_source=available_source,
        relation_available_source=relation_source)


def make_loaders(data, modules, logger):
    """Return independent candidate train/val loaders; retain original objects."""
    class CandidateLoader(modules.loader.IncidentDataLoader):
        def _validate_sample(self, sample):
            missing = [key for key in ('x_data', 'y_data') if key not in sample]
            if missing:
                raise KeyError('Sample is missing required keys: ' + repr(missing))

        def _build_batch(self, samples):
            x, y = self._stack_xy(samples)
            reports = [encode_report(sample, self.num_nodes) for sample in samples]
            batch = dict(x_data=x, y_data=y,
                incident_features=np.stack([r['incident'] for r in reports]),
                incident_position=np.asarray([r['position'] for r in reports], dtype=np.int64),
                incident_distances=np.stack([r['distances'] for r in reports]),
                available=np.asarray([r['available'] for r in reports], dtype=np.bool_),
                relation_available=np.asarray([r['relation_available'] for r in reports], dtype=np.bool_),
                category_valid=np.stack([r['category_valid'] for r in reports]))
            sensors = self._sensor_batch(len(samples))
            if sensors is not None:
                batch['sensor_data'] = sensors
            return batch

    result = {}
    for name in ('train_loader', 'val_loader'):
        if name not in data.loaders:
            continue
        old = data.loaders[name]
        result[name] = CandidateLoader(list(old.samples), True, old.bs, logger,
            input_dim=old.input_dim, sensor_info=old.sensor_info)
    if not result:
        raise ValueError('Candidate batches require a train or validation loader')
    return result


def device_batch_v03(batch, device, information='RC'):
    """Convert candidate batches with strict H / H+R / H+R+C isolation.

    Holiday, clocks inside x, and sensor metadata are public common context.
    Report position belongs to report content, so R and H cannot read it.
    Incident Time is deliberately zero: its semantics are uncertified.
    """
    if information not in INFORMATION_MODES:
        raise ValueError('information must be H, R, or RC')
    x = torch.as_tensor(batch['x_data'], dtype=torch.float32, device=device)
    y = torch.as_tensor(batch['y_data'], dtype=torch.float32, device=device)
    values = torch.as_tensor(batch['incident_features'], dtype=torch.float32, device=device).clone()
    relation = torch.as_tensor(batch['incident_distances'], dtype=torch.float32, device=device).clone()
    available = torch.as_tensor(batch['available'], dtype=torch.bool, device=device).clone()
    relation_available = torch.as_tensor(batch['relation_available'], dtype=torch.bool, device=device).clone()
    position = torch.as_tensor(batch['incident_position'], dtype=torch.long, device=device).clone()
    valid = torch.as_tensor(batch['category_valid'], dtype=torch.bool, device=device).clone()
    if values.shape != (x.shape[0], 4) or relation.shape != (x.shape[0], x.shape[2], 3):
        raise ValueError('Candidate report tensor shape mismatch')
    if available.shape != (x.shape[0],) or relation_available.shape != available.shape:
        raise ValueError('Candidate availability shape mismatch')
    effective = available & relation_available
    relation = torch.where(effective[:, None, None], relation, torch.zeros_like(relation))
    if not torch.isfinite(relation).all():
        raise ValueError('Available incident relation contains nonfinite values')
    values[:, 0] = 0
    values[:, 1:3] = torch.where(available[:, None], values[:, 1:3], -1.)
    position = torch.where(available, position, -1)
    valid[:, :2] &= available[:, None]
    valid[:, 3] &= available
    if information in ('H', 'R'):
        values[:, 1:3] = -1
        position.fill_(-1)
        valid[:, :2] = False
        valid[:, 3] = False
    if information == 'H':
        relation.zero_()
        available.zero_()
        relation_available.zero_()
    sensors = {key: torch.as_tensor(value, device=device)
        for key, value in batch.get('sensor_data', {}).items()} or None
    incident = dict(incident=values, position=position, distances=relation,
        available=available, relation_available=relation_available, category_valid=valid)
    return x, y, incident, sensors


def audit_data(data):
    """Inspect train/val metadata only; do not read any sample y_data field."""
    splits = {}
    for name in ('train_loader', 'val_loader'):
        if name not in data.loaders:
            continue
        loader = data.loaders[name]
        counts, categories, reasons = Counter(), {k: Counter() for k in (*CATEGORY_SIZES, 'position')}, {
            k: Counter() for k in (*CATEGORY_SIZES, 'position')}
        flag_sources = {'available': Counter(), 'relation_available': Counter()}
        fields = Counter()
        for sample in loader.samples:
            row = encode_report(sample, loader.num_nodes)
            counts['rows'] += 1
            counts['reports_available'] += row['available']
            counts['relations_available'] += row['relation_available']
            counts['missing_relations_with_report'] += row['available'] and not row['relation_available']
            support = np.abs(row['distances']).sum(-1) > 0
            counts['supported_node_positions'] += int(support.sum())
            counts['zero_support_rows'] += not bool(support.any())
            counts['available_zero_support_rows'] += row['available'] and row['relation_available'] and not bool(support.any())
            for index, key in ((1, 'Description'), (2, 'Type'), (3, 'Holiday')):
                categories[key][str(int(row['incident'][index]))] += 1
            categories['position'][str(row['position'])] += 1
            for key, reason in row['category_reasons'].items():
                reasons[key][reason] += 1
            for key in flag_sources:
                flag_sources[key][row[key + '_source']] += 1
            for key in IDENTITY_FIELDS:
                if key in sample and sample[key] is not None:
                    fields[key] += 1
        splits[name.removesuffix('_loader')] = dict(counts=dict(counts),
            category_histograms={k: dict(v) for k, v in categories.items()},
            category_validity={k: dict(v) for k, v in reasons.items()},
            flag_sources={k: dict(v) for k, v in flag_sources.items()},
            optional_identity_field_rows={k: fields[k] for k in IDENTITY_FIELDS})
    return dict(schema=SCHEMA, splits=splits, test_targets_read=False,
        candidate_data_mutates_original=False,
        unknown_category=-1, legal_zero_preserved=True,
        original_parser='unchanged_frozen_loader_only_for_ORIGINAL',
        public_context=['history_flow', 'history_clock', 'Holiday', 'sensor_metadata'],
        report_content=['Description', 'Type', 'incident_position'],
        unconsumed_fields=['Incident Time', 'duration', 'durations', 'future_clearance'],
        evidence_boundaries=dict(availability_cutoff_certified=False,
            end_time_semantics_certified=False, absolute_order_certified=False,
            event_independence_certified=False,
            field_existence_does_not_certify_publication_or_event_time=True,
            original_load_data_may_deserialize_whole_published_array=True))
