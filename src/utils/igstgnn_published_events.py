"""Explicit event metadata on the unchanged author-published traffic windows.

Missing fields have separate unknown tokens and validity flags. Neither a zero
category nor a zero relation is interpreted as absence of an event. Incident
Time, position, Holiday and durations are deliberately not model inputs here.
"""
import json
from pathlib import Path

import numpy as np

from src.utils.igstgnn_published import PublishedData, PublishedLoader, require


TYPE_COUNT = 6
DESCRIPTION_COUNT = 29
TYPE_UNKNOWN = TYPE_COUNT
DESCRIPTION_UNKNOWN = DESCRIPTION_COUNT
EVENT_KEYS = ('type', 'description', 'type_valid', 'description_valid',
              'relation', 'relation_available', 'report_present')


def _category(features, key, count):
    if key not in features:
        return count, False
    value = features[key]
    require(not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating)),
            f'Invalid published event {key}: expected a numeric category')
    require(np.isfinite(value) and value == int(value) and 0 <= value < count,
            f'Invalid published event {key}: expected integer in [0, {count})')
    return int(value), True


def encode_event(sample, nodes):
    """Validate one raw record without looking at traffic targets or time fields.

    A present (possibly incomplete) event dictionary or relation payload denotes
    a reported event unless an explicit Boolean ``report_present`` says otherwise.
    This payload convention is not certification of a natural no-event label.
    """
    has_features = 'incident_features' in sample
    features = sample['incident_features'] if has_features else {}
    require(isinstance(features, dict), 'Published incident_features must be a dictionary')
    event_type, type_valid = _category(features, 'Type', TYPE_COUNT)
    description, description_valid = _category(features, 'Description', DESCRIPTION_COUNT)
    relation_available = 'incident_distances' in sample
    if relation_available:
        raw = np.asarray(sample['incident_distances'])
        require(raw.shape == (nodes, 3) and np.issubdtype(raw.dtype, np.number)
                and not np.issubdtype(raw.dtype, np.complexfloating),
                'Invalid published incident_distances shape or dtype')
        require(np.isfinite(raw).all(), 'Non-finite published incident_distances')
        with np.errstate(over='ignore', invalid='ignore'):
            relation = raw.astype(np.float32, copy=True)
        require(np.isfinite(relation).all(), 'Published incident_distances overflow float32')
    else:
        relation = np.zeros((nodes, 3), dtype=np.float32)
    report_present = sample.get('report_present', has_features or relation_available)
    require(isinstance(report_present, (bool, np.bool_)), 'report_present must be Boolean')
    return dict(type=event_type, description=description, type_valid=type_valid,
                description_valid=description_valid, relation=relation,
                relation_available=relation_available, report_present=bool(report_present))


def verify_category_mappings(directory):
    """Validate the frozen release vocabulary axes; hashes belong to the runner."""
    for name, count in (('type_mapping.json', TYPE_COUNT), ('desc_mapping.json', DESCRIPTION_COUNT)):
        with (Path(directory)/name).open(encoding='utf-8-sig') as stream:
            mapping = json.load(stream)
        require(isinstance(mapping, dict) and len(mapping) == count,
                f'Published category vocabulary size mismatch: {name}')
        values = list(mapping.values())
        require(all(type(value) is int for value in values)
                and sorted(values) == list(range(count)),
                f'Published category vocabulary IDs must be unique contiguous integers: {name}')


class EventPublishedLoader(PublishedLoader):
    """Same X/Y batching and resumable ordering, with an explicit event payload."""
    def __init__(self, samples, batch_size, logger):
        super().__init__(samples, batch_size, logger)
        # Validate before training, rather than discovering malformed metadata
        # only after its sample is reached by a shuffled iterator.
        for sample in self.original_samples:
            encode_event(sample, self.num_nodes)

    def _build_batch(self, batch_samples):
        x, y = self._stack_xy(batch_samples)
        rows = [encode_event(sample, self.num_nodes) for sample in batch_samples]
        events = {}
        for key in EVENT_KEYS:
            dtype = np.float32 if key == 'relation' else np.int64 if key in ('type', 'description') else np.bool_
            events[key] = np.asarray([row[key] for row in rows], dtype=dtype)
        return dict(x_data=x, y_data=y, event_data=events)

    def event_audit(self):
        counts = dict(rows=self.size, report_present=0, report_absent_payload=0,
                      explicit_report_flag=0, missing_type=0, missing_description=0,
                      missing_relation=0, zero_relation_support=0)
        support_counts = []
        for sample in self.original_samples:
            row = encode_event(sample, self.num_nodes)
            counts['report_present'] += int(row['report_present'])
            counts['report_absent_payload'] += int(not row['report_present'])
            counts['explicit_report_flag'] += int('report_present' in sample)
            counts['missing_type'] += int(not row['type_valid'])
            counts['missing_description'] += int(not row['description_valid'])
            counts['missing_relation'] += int(not row['relation_available'])
            support = int((np.abs(row['relation']).sum(-1) > 0).sum())
            support_counts.append(support)
            counts['zero_relation_support'] += int(support == 0)
        counts['relation_support_min'] = min(support_counts)
        counts['relation_support_max'] = max(support_counts)
        return counts


class EventPublishedData(PublishedData):
    """Reuse the original release read, split, checks, scaler and station axis.

    The superclass deserializes the all-sample object array once. Test metadata
    and targets are not validated or batched until an explicit test_loader call.
    """
    def __init__(self, directory, protocol, logger, *, check=False):
        verify_category_mappings(directory)
        super().__init__(directory, protocol, logger, check=check)
        self.loaders = {key: EventPublishedLoader(loader.original_samples, loader.bs, logger)
                        for key, loader in self.loaders.items()}
        split_audits = {key: loader.event_audit() for key, loader in self.loaders.items()}
        self.audit.update(
            incident_inputs_used=True, event_input_schema='published_type_description_relation_v1',
            event_validation_scope='used_train_validation_samples_only',
            event_split_audits=split_audits,
            no_natural_no_event_observations_verified=False,
            all_used_samples_have_report_payload=all(row['report_absent_payload'] == 0 for row in split_audits.values()),
            category_counts=dict(type=TYPE_COUNT, description=DESCRIPTION_COUNT),
            unknown_tokens=dict(type=TYPE_UNKNOWN, description=DESCRIPTION_UNKNOWN),
            unknown_policy='separate_token_and_validity_flag_not_class_zero',
            report_presence_policy='explicit_boolean_or_metadata_payload_presence_not_relation_support',
            relation_mask_policy='abs_sum_positive_after_original_float32_conversion',
            ignored_event_fields=['Incident Time', 'incident_position', 'Holiday', 'durations'],
            event_attribute_first_report_availability_verified=False,
            report_age_available=False)

    def test_loader(self, batch_size, logger):
        return EventPublishedLoader(self._test_samples, batch_size, logger)
