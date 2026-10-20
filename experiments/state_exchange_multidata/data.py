"""Authenticated author releases, with dataset-specific event vocabulary axes."""
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.utils.dataloader import _load_sensor_info
from src.utils.igstgnn_published import PublishedData, PublishedLoader
from src.utils.igstgnn_published_events import EVENT_KEYS, _category
from experiments.position_state_gw16.data import (
    aligned_rows, build_partition, build_position_metadata, graph_support, normalize_rows)
from .common import dataset_name, read_json, require, sha256, spec, validate_config


def event_counts(record):
    return dict(type=record['type_count'], description=record['description_count'])


def verify_category_mappings(directory, counts):
    mappings = {}
    for name, key in (('type_mapping.json', 'type'), ('desc_mapping.json', 'description')):
        mapping = read_json(Path(directory)/name)
        require(isinstance(mapping, dict) and len(mapping) == counts[key],
                'Published category vocabulary size mismatch: '+name)
        require(all(type(value) is int for value in mapping.values())
                and sorted(mapping.values()) == list(range(counts[key])),
                'Published category IDs must be unique contiguous integers: '+name)
        mappings[key] = mapping
    return mappings


def encode_event(sample, nodes, counts):
    """Same GW16 event boundary, with each release's unchanged local ID space."""
    has_features = 'incident_features' in sample
    features = sample['incident_features'] if has_features else {}
    require(isinstance(features, dict), 'Published incident_features must be a dictionary')
    incident_type, type_valid = _category(features, 'Type', counts['type'])
    description, description_valid = _category(features, 'Description', counts['description'])
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
    return dict(type=incident_type, description=description, type_valid=type_valid,
        description_valid=description_valid, relation=relation,
        relation_available=relation_available, report_present=bool(report_present))


class EventPublishedLoader(PublishedLoader):
    def __init__(self, samples, batch_size, logger, counts):
        self.event_counts = dict(counts)
        super().__init__(samples, batch_size, logger)
        for sample in self.original_samples:
            encode_event(sample, self.num_nodes, self.event_counts)

    def _build_batch(self, batch_samples):
        x, y = self._stack_xy(batch_samples)
        rows = [encode_event(sample, self.num_nodes, self.event_counts) for sample in batch_samples]
        events = {}
        for key in EVENT_KEYS:
            dtype = (np.float32 if key == 'relation' else np.int64
                     if key in ('type', 'description') else np.bool_)
            events[key] = np.asarray([row[key] for row in rows], dtype=dtype)
        return dict(x_data=x, y_data=y, event_data=events)

    def event_audit(self):
        counts = dict(rows=self.size, report_present=0, report_absent_payload=0,
            explicit_report_flag=0, missing_type=0, missing_description=0,
            missing_relation=0, zero_relation_support=0)
        support_counts = []
        for sample in self.original_samples:
            row = encode_event(sample, self.num_nodes, self.event_counts)
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
    """Only used training/validation events are read before explicit test access."""
    def __init__(self, directory, protocol, logger, counts, *, check=False):
        self.event_counts = dict(counts)
        self.category_mappings = verify_category_mappings(directory, counts)
        super().__init__(directory, protocol, logger, check=check)
        self.loaders = {key: EventPublishedLoader(loader.original_samples, loader.bs, logger, counts)
                        for key, loader in self.loaders.items()}
        split_audits = {key: loader.event_audit() for key, loader in self.loaders.items()}
        self.audit.update(incident_inputs_used=True,
            event_input_schema='published_type_description_relation_v1',
            event_validation_scope='used_train_validation_samples_only',
            event_split_audits=split_audits, no_natural_no_event_observations_verified=False,
            all_used_samples_have_report_payload=all(row['report_absent_payload'] == 0
                for row in split_audits.values()), category_counts=dict(counts),
            unknown_tokens=dict(counts), unknown_policy='separate_token_and_validity_flag_not_class_zero',
            report_presence_policy='explicit_boolean_or_metadata_payload_presence_not_relation_support',
            relation_mask_policy='abs_sum_positive_after_original_float32_conversion',
            ignored_event_fields=['Incident Time', 'incident_position', 'Holiday', 'durations'],
            event_attribute_first_report_availability_verified=False, report_age_available=False)

    def test_loader(self, batch_size, logger):
        return EventPublishedLoader(self._test_samples, batch_size, logger, self.event_counts)


def authenticate_release(directory, dataset):
    directory = Path(directory).resolve()
    record = spec(dataset)
    hashes = {}
    for name, expected in record['hashes'].items():
        actual = sha256(directory/name)
        require(actual == expected, 'Published dataset fingerprint mismatch: '+str(directory/name))
        hashes[name] = actual
    return record, hashes


def load_data(directory, config, logger, *, engineering=False):
    validate_config(config, engineering=engineering)
    directory = Path(directory).resolve()
    name = dataset_name(config)
    record, hashes = authenticate_release(directory, name)
    nodes = record['nodes']
    counts = event_counts(record)
    protocol = {key:record[key] for key in ('total_samples', 'train_samples', 'val_samples', 'test_samples')}
    protocol.update(stations=nodes, batch_size=config['batch_size'])
    published = EventPublishedData(directory, protocol, logger, counts, check=engineering)
    with (directory/'sensors.csv').open(encoding='utf-8-sig', newline='') as stream:
        raw_rows = list(csv.DictReader(stream))
    rows = aligned_rows(raw_rows, published.station_ids)
    require([int(row['station_id']) for row in raw_rows] == list(published.station_ids),
            'Existing sensor parser row axis differs from published station axis')
    sensor_args = SimpleNamespace(use_sensor_info=True, dataset=name)
    sensors = _load_sensor_info(str(directory), nodes, sensor_args, logger)
    require(all(np.isfinite(value).all() for value in sensors.values()), 'Invalid sensor metadata')
    position_features, position_categories, position_sizes, position_identity = build_position_metadata(
        rows, published.station_ids)
    assignment, partition = build_partition(rows, config['max_region_size'], config['partition'])
    adjacency = np.load(directory/'adj_matrix.npy', allow_pickle=False)
    require(adjacency.shape == (nodes, nodes), 'Author adjacency axis mismatch')
    forward = graph_support(adjacency, config['graph_top_k'])
    backward = graph_support(adjacency.T, config['graph_top_k'])
    indicator = np.eye(len(partition['names']), dtype=np.float32)[assignment]
    regional = normalize_rows(indicator.T @ (forward-np.diag(np.diag(forward))) @ indicator)
    identity = dict(dataset=name, hashes=hashes, published=published.audit,
        loader_batching={key:dict(batch_size=value.bs, batches=value.num_batch, samples=value.size)
                         for key, value in published.loaders.items()},
        station_ids=published.station_ids, partition=partition, position=position_identity,
        graph=dict(top_k=config['graph_top_k'], self_loops=True,
            orientation='row_receives_column; forward_and_transpose',
            native_predefined='unchanged_author_graph_doubletransition_without_new_topk',
            kernel='dense_deterministic_GEMM_on_fixed_sparse_support', physical_direction_certified=False),
        sensor_parser='existing_src_utils_dataloader_no_new_unit_parsing',
        incident_fields=['Type', 'Description', 'incident_distances'],
        ignored_incident_fields=['Incident Time', 'incident_position', 'Holiday', 'durations'],
        event_vocabulary=dict(counts=counts, mappings=published.category_mappings,
            mapping_hashes={key:hashes[key] for key in ('type_mapping.json', 'desc_mapping.json')},
            id_policy='unchanged_dataset_local_author_IDs'),
        native_contract=dict(position_entry='after_flow_embedding_before_ICSF',
            type_unknown=counts['type'], description_unknown=counts['description'],
            incident_position_constant=0, holiday_constant=0,
            unusable_report='per_sample_ICSF_bypass_and_zero_TIID_context', faithful_original_replay=False),
        first_report_availability_verified=False, engineering_only=engineering)
    return SimpleNamespace(published=published, loaders=published.loaders, sensors=sensors,
        sensor_sizes=[sensor_args.sensor_type_size, sensor_args.surface_size, sensor_args.roadway_use_size],
        station_ids=published.station_ids, assignment=assignment, partition=partition,
        position_features=position_features, position_categories=position_categories,
        position_sizes=position_sizes, position_identity=position_identity,
        native_adjacency=adjacency, graphs=(forward, backward, regional), event_counts=counts,
        mean=float(published.audit['mean']), std=float(published.audit['std']),
        identity=identity, directory=directory)


def device_batch(raw, data, device):
    x = torch.as_tensor(raw['x_data'], dtype=torch.float32, device=device)
    target = torch.as_tensor(raw['y_data'], dtype=torch.float32, device=device)*data.std+data.mean
    event = {key:torch.as_tensor(value, device=device) for key, value in raw['event_data'].items()}
    return x, target, event


def validation_loader(data, split, batch_size, logger):
    require(split in ('val', 'test'), 'Evaluation split must be val or test')
    record = spec(data.identity['dataset'])
    if split == 'val':
        return data.loaders['val_loader'], record['train_samples']
    return data.published.test_loader(batch_size, logger), record['train_samples']+record['val_samples']
