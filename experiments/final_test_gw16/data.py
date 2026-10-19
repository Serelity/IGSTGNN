"""Read the authenticated author-order test slice after the caller freezes models.

This module never selects checkpoints, fits a transform, or performs a forward.
The caller must finish its freeze/completed-budget checks before calling here.
Author array order is the split contract, not independently verified event time.
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.chronological.original_igstgnn_runtime import RELEASE, sha256


COUNTS = dict(total_samples=5585, train_samples=3909, val_samples=837, test_samples=839)
TEST_START, TEST_STOP, NODES, STEPS, BATCH_SIZE = 4746, 5585, 496, 12, 48


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_sample(sample, index):
    require(isinstance(sample, dict), 'Published test sample is not a dictionary: '+str(index))
    for key in ('x_data', 'y_data'):
        value = np.asarray(sample[key])
        require(value.shape == (STEPS, NODES, 3) and value.dtype == np.float32
                and np.isfinite(value).all(), 'Invalid published test '+key+': '+str(index))
    clocks = sample['x_data'][..., 1:]
    require(np.array_equal(clocks, np.broadcast_to(clocks[:, :1], clocks.shape)),
            'Published test history clocks disagree across stations: '+str(index))
    nominal = clocks[:, 0]*np.array([288, 7], dtype=np.float32)
    require(np.all(np.abs(nominal-np.round(nominal)) <= 3e-5)
            and np.all((nominal >= 0) & (nominal < [288, 7])),
            'Published test history clocks are invalid: '+str(index))
    require(all(key in sample for key in
                ('incident_features', 'incident_position', 'incident_distances')),
            'Original test incident payload is incomplete: '+str(index))


def load_test_data(directory, original_config, gw16_data, modules, logger):
    """Return paired frozen-parser loaders without re-reading the all-sample array.

    ``gw16_data`` must be the unchanged result of the existing GW16 ``load_data``.
    Both loaders see identical raw object references and preserve release order.
    Category axes/position transforms/partition are reused from train/validation;
    the candidate retains its existing data object and its existing identity.
    """
    directory = Path(directory).resolve()
    require(directory == Path(gw16_data.directory).resolve(),
            'Final test directory differs from frozen GW16 data directory')
    require(gw16_data.identity.get('engineering_only') is False,
            'Engineering GW16 data cannot authorize a formal test')
    require(gw16_data.identity.get('hashes') == RELEASE,
            'GW16 data identity is not the certified six-file author release')
    hashes = {}
    for name, expected in RELEASE.items():
        actual = sha256(directory/name)
        require(actual == expected, 'Published final-test fingerprint mismatch: '+name)
        hashes[name] = actual

    published = gw16_data.published
    audit = published.audit
    require(audit.get('counts') == COUNTS, 'Published final-test split counts differ')
    require(audit.get('split_policy') == 'author_order_70_15_15'
            and audit.get('scaler_refitted') is False,
            'Expected unchanged author-order split and published scaler')
    require(audit.get('category_counts') == dict(type=6, description=29)
            and audit.get('unknown_tokens') == dict(type=6, description=29)
            and audit.get('event_input_schema') == 'published_type_description_relation_v1',
            'Candidate event schema differs from its frozen release vocabulary')
    station_ids = np.asarray(gw16_data.station_ids, dtype=np.int64)
    with (directory/'sensors.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_ids = np.asarray([int(row['station_id']) for row in csv.DictReader(stream)], dtype=np.int64)
    require(station_ids.shape == (NODES,) and len(np.unique(station_ids)) == NODES
            and np.array_equal(csv_ids, station_ids)
            and np.array_equal(np.asarray(published.station_ids), station_ids),
            'Final-test station axis differs from the frozen author release')

    with np.load(directory/'incident_stats.npz', allow_pickle=False) as stats:
        mean, std = float(stats['mean']), float(stats['std'])
        require(bool(stats['normalized']) and np.isfinite([mean, std]).all() and std > 0,
                'Invalid unchanged published final-test normalization')
    require(mean == float(gw16_data.mean) == float(audit['mean'])
            and std == float(gw16_data.std) == float(audit['std']),
            'Final-test scaler differs from frozen GW16 normalization')

    config = dict(original_config)
    require(config.get('dataset') == 'Contra_Costa'
            and config.get('seq_len') == config.get('horizon') == STEPS
            and config.get('input_dim') == 3 and config.get('output_dim') == 1
            and config.get('incident') is True and config.get('use_sensor_info') is True
            and config.get('bs') == BATCH_SIZE,
            'Original final-test model configuration differs from certified axes/batching')
    args = SimpleNamespace(**dict(config, node_num=NODES))
    sensors = modules.loader._load_sensor_info(str(directory), NODES, args, logger)
    require(bool(sensors) and all(np.asarray(value).shape == (NODES,)
            and np.isfinite(value).all() for value in sensors.values()),
            'Frozen original parser returned invalid final-test sensor metadata')
    scaler = modules.loader.StandardScaler(mean, std)

    # PublishedData constructed this exact [4746:5585] slice without testing it.
    # Do not reload arrays, sort samples, or change train/validation loaders.
    samples = tuple(published._test_samples)
    require(len(samples) == TEST_STOP-TEST_START, 'Expected all 839 final-test windows')
    for index, sample in enumerate(samples, TEST_START):
        _validate_sample(sample, index)
    native_loader = modules.loader.IncidentDataLoader(samples, True, BATCH_SIZE, logger,
        input_dim=3, sensor_info=sensors)
    candidate_loader = published.test_loader(BATCH_SIZE, logger)
    require(candidate_loader.size == native_loader.size == TEST_STOP-TEST_START
            and candidate_loader.bs == native_loader.bs == BATCH_SIZE
            and candidate_loader.num_batch == native_loader.num_batch == 18,
            'Final-test paired loaders have different sample or batch axes')
    require(all(actual is expected for actual, expected in
                zip(candidate_loader.original_samples, samples)),
            'Candidate final-test loader changed published sample order')
    require(len(candidate_loader.original_samples) == len(samples)
            and len(native_loader.samples) == len(samples)
            and all(actual is expected for actual, expected in zip(native_loader.samples, samples)),
            'Original final-test loader changed published sample order')
    sample_ids = np.arange(TEST_START, TEST_STOP, dtype=np.int64)
    partition = [min(BATCH_SIZE, len(samples)-start) for start in range(0, len(samples), BATCH_SIZE)]
    identity = dict(schema='final_test_gw16_author_release_v1', hashes=hashes,
        data_source='author_model_ready_release', split_policy='author_order_70_15_15',
        counts=COUNTS.copy(), bounds_half_open=[TEST_START, TEST_STOP],
        selected_indices=sample_ids.tolist(), count=len(samples), station_ids=station_ids.tolist(),
        batch_size=BATCH_SIZE, batch_partition=partition,
        normalization=dict(mean=mean, std=std, refitted=False),
        history=STEPS, horizon=STEPS, input_channels=3, target_channels=1,
        native_sensor_parser='unchanged_558ab01', native_source_fingerprint=modules.source_fingerprint,
        candidate_event_schema=audit['event_input_schema'],
        category_counts=dict(type=6, description=29), unknown_tokens=dict(type=6, description=29),
        candidate_transforms='reuse_frozen_gw16_data_no_fit_no_repartition',
        traffic_rebuilt=False, scaler_refitted=False, test_payload_deserialized=True,
        test_evaluated=False, absolute_calendar_order_independently_verified=False,
        first_report_availability_verified=False)
    native_data = SimpleNamespace(directory=directory, config=vars(args), sensors=sensors,
        scaler=scaler, station_ids=station_ids.copy(), loaders={'test_loader': native_loader},
        identity=identity)
    return SimpleNamespace(native_data=native_data, native_loader=native_loader,
        candidate_loader=candidate_loader, sample_ids=sample_ids,
        station_ids=station_ids.copy(), identity=identity)
