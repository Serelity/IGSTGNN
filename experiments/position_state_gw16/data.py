"""Authenticated published windows and target-free fixed spatial metadata."""
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.utils.dataloader import _load_sensor_info
from src.utils.igstgnn_published_events import EventPublishedData
from .common import RELEASE, require, sha256


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    require(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
            and np.isfinite(matrix).all() and np.all(matrix >= 0), 'Invalid graph weights')
    matrix = matrix.copy()
    matrix[np.diag_indices_from(matrix)] += 1.0
    return matrix/np.maximum(matrix.sum(1, keepdims=True), 1e-12)


def graph_support(adjacency, top_k):
    """Row receives column; author graph is not certified physical flow direction."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    require(adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1]
            and np.isfinite(adjacency).all() and np.all(adjacency >= 0), 'Invalid author adjacency')
    require(type(top_k) is int and top_k > 0, 'top_k must be positive')
    result = np.zeros_like(adjacency)
    values = adjacency.copy()
    np.fill_diagonal(values, 0)
    for i in range(len(values)):
        indices = np.argsort(-values[i], kind='stable')[:min(top_k, len(values)-1)]
        result[i, indices] = values[i, indices]
    return normalize_rows(result)


def aligned_rows(rows, station_ids):
    """Explicit station-ID join; duplicates/missing/extra IDs cannot silently align."""
    try:
        ids = [int(row['station_id']) for row in rows]
        expected = [int(value) for value in station_ids]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Invalid sensor station_id') from error
    require(len(set(ids)) == len(ids) and len(set(expected)) == len(expected),
            'Duplicate station_id in spatial metadata or release axis')
    require(set(ids) == set(expected), 'Spatial station IDs differ from published traffic axis')
    lookup = dict(zip(ids, rows))
    return [lookup[value] for value in expected]


def _text(row, key):
    value = str(row.get(key, '')).strip()
    return '' if value.lower() in ('', 'nan', 'none', 'unknown', 'na') else value


def _number(row, key):
    try:
        value = float(row.get(key, 'nan'))
    except (TypeError, ValueError):
        return float('nan')
    return value if np.isfinite(value) else float('nan')


def build_position_metadata(rows, station_ids):
    rows = aligned_rows(rows, station_ids)
    vocabularies, encoded = {}, []
    for key in ('Fwy', 'Direction'):
        values = [_text(row, key) for row in rows]
        vocabulary = {'<UNKNOWN>': 0}
        vocabulary.update({value: i+1 for i, value in enumerate(sorted(set(values)-{''}))})
        vocabularies[key] = vocabulary
        encoded.append([vocabulary.get(value, 0) for value in values])
    categories = np.asarray(encoded, dtype=np.int64).T
    numeric = np.asarray([[_number(row, key) for key in ('Lat', 'Lng', 'Abs PM')]
                          for row in rows], dtype=np.float64)
    valid = np.isfinite(numeric)
    features = np.zeros((len(rows), 6), dtype=np.float32)
    normalization = {}
    for j, key in enumerate(('Lat', 'Lng')):
        values = numeric[valid[:, j], j]
        center = float(values.mean()) if len(values) else 0.
        scale = float(values.std()) if len(values) else 1.
        scale = scale if scale > 1e-12 else 1.
        features[valid[:, j], j] = (numeric[valid[:, j], j]-center)/scale
        normalization[key] = dict(mean=center, std=scale, fitted_to='fixed_sensor_metadata_only')
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((_text(row, 'Fwy'), _text(row, 'Direction')), []).append(i)
    postmile_groups = []
    for group, members in sorted(groups.items()):
        members = np.asarray(members, dtype=np.int64)
        available = members[valid[members, 2]]
        values = numeric[available, 2]
        lower, upper = (float(values.min()), float(values.max())) if len(values) else (0., 0.)
        denominator = upper-lower if upper-lower > 1e-12 else 1.
        features[available, 2] = (values-lower)/denominator
        postmile_groups.append(dict(road=group[0], direction=group[1], minimum=lower,
            maximum=upper, denominator=denominator, available_count=len(available)))
    features[:, 3:] = valid.astype(np.float32)
    require(np.isfinite(features).all(), 'Nonfinite encoded position metadata')
    identity = dict(schema='gw16_position_metadata_v1', station_ids=[int(v) for v in station_ids],
        feature_names=['Lat_z', 'Lng_z', 'relative_Abs_PM', 'Lat_valid', 'Lng_valid', 'Abs_PM_valid'],
        categorical_names=['Fwy', 'Direction'], vocabularies=vocabularies,
        unknown_category=0, normalization=normalization, postmile_groups=postmile_groups,
        missing_numeric_fill=0., numeric_features=features.tolist(), categories=categories.tolist(),
        fitting_scope='fixed_metadata_no_traffic_no_targets', physical_direction_certified=False)
    return features, categories, [len(vocabularies[key]) for key in ('Fwy', 'Direction')], identity


def build_partition(rows, max_size, mode='road_direction_postmile'):
    require(type(max_size) is int and max_size > 0, 'Invalid maximum region size')
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((_text(row, 'Fwy'), _text(row, 'Direction')), []).append(i)
    assignment = np.empty(len(rows), dtype=np.int64)
    names, members = [], []
    for group in sorted(groups):
        def key(i):
            mile = _number(rows[i], 'Abs PM')
            return (not np.isfinite(mile), mile if np.isfinite(mile) else 0., int(rows[i]['station_id']))
        ordered = sorted(groups[group], key=key)
        chunks = np.array_split(ordered, int(np.ceil(len(ordered)/max_size)))
        for j, chunk in enumerate(chunks):
            indices = [int(value) for value in chunk]
            assignment[indices] = len(names)
            members.append(indices)
            names.append('/'.join(group)+':'+str(j))
    require(mode in ('road_direction_postmile', 'random_matched'), 'Unknown partition')
    if mode == 'random_matched':
        assignment = assignment[np.random.RandomState(1701).permutation(len(rows))]
        members = [np.flatnonzero(assignment == r).tolist() for r in range(len(names))]
        names = ['random:'+str(r) for r in range(len(names))]
    require(len(names) > 0 and all(members), 'Empty region')
    return assignment, dict(policy=mode, max_region_size=max_size, random_partition_seed=1701,
        sorting='Fwy/Direction then finite numeric Abs PM then station_id; no traffic targets',
        names=names, members=members, assignment=assignment.tolist(),
        sizes=[len(value) for value in members], physical_conservation_certified=False)


def load_data(directory, config, logger, *, engineering=False):
    directory = Path(directory).resolve()
    require((config['nodes'], config['history'], config['horizon']) == (496, 12, 12),
            'The certified release adapter requires 496 nodes and 12-to-12')
    hashes = {}
    for name, expected in RELEASE.items():
        actual = sha256(directory/name)
        require(actual == expected, 'Published dataset fingerprint mismatch: '+str(directory/name))
        hashes[name] = actual
    protocol = dict(total_samples=5585, train_samples=3909, val_samples=837,
                    test_samples=839, stations=496, batch_size=config['batch_size'])
    published = EventPublishedData(directory, protocol, logger, check=engineering)
    with (directory/'sensors.csv').open(encoding='utf-8-sig', newline='') as stream:
        raw_rows = list(csv.DictReader(stream))
    rows = aligned_rows(raw_rows, published.station_ids)
    # Existing sensor parser returns CSV-row arrays, so certify its exact axis.
    require([int(row['station_id']) for row in raw_rows] == list(published.station_ids),
            'Existing sensor parser row axis differs from published station axis')
    sensor_args = SimpleNamespace(use_sensor_info=True, dataset='Contra_Costa')
    sensors = _load_sensor_info(str(directory), 496, sensor_args, logger)
    require(all(np.isfinite(value).all() for value in sensors.values()), 'Invalid sensor metadata')
    position_features, position_categories, position_sizes, position_identity = build_position_metadata(
        rows, published.station_ids)
    assignment, partition = build_partition(rows, config['max_region_size'], config['partition'])
    adjacency = np.load(directory/'adj_matrix.npy', allow_pickle=False)
    require(adjacency.shape == (496, 496), 'Author adjacency axis mismatch')
    forward = graph_support(adjacency, config['graph_top_k'])
    backward = graph_support(adjacency.T, config['graph_top_k'])
    s = np.eye(len(partition['names']), dtype=np.float32)[assignment]
    regional = normalize_rows(s.T @ (forward-np.diag(np.diag(forward))) @ s)
    identity = dict(hashes=hashes, published=published.audit,
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
        native_contract=dict(position_entry='after_flow_embedding_before_ICSF',
            type_unknown=6, description_unknown=29, incident_position_constant=0,
            holiday_constant=0, unusable_report='per_sample_ICSF_bypass_and_zero_TIID_context',
            faithful_original_replay=False),
        first_report_availability_verified=False, engineering_only=engineering)
    return SimpleNamespace(published=published, loaders=published.loaders, sensors=sensors,
        sensor_sizes=[sensor_args.sensor_type_size, sensor_args.surface_size, sensor_args.roadway_use_size],
        station_ids=published.station_ids, assignment=assignment, partition=partition,
        position_features=position_features, position_categories=position_categories,
        position_sizes=position_sizes, position_identity=position_identity,
        native_adjacency=adjacency, graphs=(forward, backward, regional),
        mean=float(published.audit['mean']), std=float(published.audit['std']),
        identity=identity, directory=directory)


def device_batch(raw, data, device):
    x = torch.as_tensor(raw['x_data'], dtype=torch.float32, device=device)
    target = torch.as_tensor(raw['y_data'], dtype=torch.float32, device=device)*data.std+data.mean
    event = {key:torch.as_tensor(value, device=device) for key, value in raw['event_data'].items()}
    return x, target, event


def validation_loader(data, split, batch_size, logger):
    require(split in ('val', 'test'), 'Evaluation split must be val or test')
    if split == 'val':
        return data.loaders['val_loader'], 3909
    return data.published.test_loader(batch_size, logger), 4746
