"""Package fixed candidate corridors from verified v11a training history only."""

import argparse
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.incident_corridor import (
    CHANNELS, REPORT_FIELDS, SCHEMA, CorridorHistory, history_diagnostics,
    read_json, read_rows, require, select_corridors, sha256, validate_train_clock,
    write_json, write_rows,
)


def verified(path, expected):
    require(isinstance(expected, str) and sha256(path) == expected,
            'Input checksum mismatch: ' + str(path))
    return expected


def discover_history(roots, data_summary_hash):
    """Bounded search for reusable v11a packages, never choose different contents."""
    candidates = {}
    for root in roots:
        root = Path(root)
        paths = [root / 'summary.json']
        for pattern in ('*/summary.json', '*/multichannel/summary.json'):
            paths.extend(root.glob(pattern))
        for path in paths:
            if not path.is_file():
                continue
            try:
                meta = read_json(path)
                if (meta.get('status') != 'MULTICHANNEL_HISTORY_MATERIALIZATION_COMPLETE'
                        or meta.get('engineering_check') is not False
                        or meta.get('acceptance', {}).get('gate_passed') is not True
                        or meta.get('inputs', {}).get('data_summary_sha256') != data_summary_hash):
                    continue
                digest = meta['outputs']['train_history.npy']['sha256']
                if not all((path.parent / name).is_file() for name in
                           ('train_history.npy', 'train_multichannel_scaler.json')):
                    continue
                candidates[str(path.parent.absolute())] = digest
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
    require(len(set(candidates.values())) <= 1,
            'Multiple different v11a histories found; specify one explicitly')
    return Path(sorted(candidates)[0]) if candidates else None


def load_inputs(data_dir, history_dir, sensors, metadata_dir, config):
    """Read only whitelisted train files; do not open raw flow/gap/Y or val arrays."""
    fingerprints = {}
    for filename, key in (('summary.json', 'data_summary_sha256'),
                          ('context_manifest.json', 'context_manifest_sha256')):
        fingerprints['data/' + filename] = verified(data_dir / filename, config[key])
    data = read_json(data_dir / 'summary.json')
    context = read_json(data_dir / 'context_manifest.json')
    require(data.get('source_version') == 8 and data.get('build_complete') is True
            and data.get('status') == 'conditional_development' and data.get('X_slice') == [0, 12],
            'Unexpected source package identity')
    require(context.get('schema') == 'report_location_v1' and context.get('scope') ==
            'conditional_development', 'Unexpected incident schema')
    for name in ('station_ids.npy', 'train_manifest.csv'):
        fingerprints['data/' + name] = verified(data_dir / name, data['files'][name])
    fingerprints['data/train_context.npz'] = verified(
        data_dir / 'train_context.npz', context['outputs']['train_context.npz'])
    sensor_sources = [digest for path, digest in context['sources'].items()
                      if path.replace('\\', '/').split('/')[-1] == 'sensors.csv']
    require(bool(sensor_sources) and all(sha256(sensors) == digest for digest in sensor_sources),
            'Published sensor identity mismatch')
    fingerprints['sensors'] = sha256(sensors)
    meta = read_json(metadata_dir / 'manifest.json')
    require(meta.get('schema_version') == 1 and meta.get('published_sensors_sha256') ==
            fingerprints['sensors'], 'Metadata bundle identity mismatch')
    fingerprints['metadata/manifest.json'] = sha256(metadata_dir / 'manifest.json')
    source = meta['files']['source_sensor_subset.tsv']
    fingerprints['metadata/source_sensor_subset.tsv'] = verified(
        metadata_dir / 'source_sensor_subset.tsv', source['sha256'])

    multi = read_json(history_dir / 'summary.json')
    require(multi.get('status') == 'MULTICHANNEL_HISTORY_MATERIALIZATION_COMPLETE'
            and multi.get('protocol_id') == 'contra_v8_multichannel_history_materialize_v11a'
            and multi.get('engineering_check') is False
            and multi.get('acceptance', {}).get('gate_passed') is True
            and multi.get('inputs', {}).get('data_summary_sha256') == config['data_summary_sha256'],
            'Multichannel history is not a completed matching v11a package')
    require(multi.get('channel_semantics', {}).get('source_order') == CHANNELS,
            'Multichannel source order mismatch')
    require(multi['splits']['train'].get('flow_anchor_mismatches') == 0,
            'Original v11a flow anchor did not pass')
    fingerprints['history/summary.json'] = sha256(history_dir / 'summary.json')
    for filename in ('train_history.npy', 'train_multichannel_scaler.json'):
        fingerprints['history/' + filename] = verified(
            history_dir / filename, multi['outputs'][filename]['sha256'])
    ids = np.load(data_dir / 'station_ids.npy', allow_pickle=False)
    rows = read_rows(data_dir / 'train_manifest.csv')
    with np.load(data_dir / 'train_context.npz', allow_pickle=False) as stored:
        require(set(stored.files) == set(REPORT_FIELDS), 'Unexpected report fields')
        report = {key: stored[key] for key in REPORT_FIELDS}
    require(np.array_equal(report['station_ids'], ids), 'Report station order mismatch')
    dist = report['distances']
    require(dist.shape == (len(rows), len(ids), 3) and np.isfinite(dist).all()
            and (dist[..., 0] == 0).all() and ((dist[..., 1] >= 0) & (dist[..., 1] <= 1)).all()
            and np.isin(dist[..., 2], [0, 1]).all(), 'Invalid report-location features')
    labels = validate_train_clock(rows, report)
    scaler = read_json(history_dir / 'train_multichannel_scaler.json')
    require(scaler.get('fit_scope') == 'unique_train_X_12_steps_station_channel_finite_nonnegative'
            and scaler.get('channel_order') == CHANNELS
            and np.array_equal(scaler.get('station_ids'), ids)
            and scaler.get('fitted_sample_indices') == [int(r['sample_index']) for r in rows],
            'History provenance station/sample order mismatch')
    history = np.load(history_dir / 'train_history.npy', mmap_mode='r', allow_pickle=False)
    require(history.dtype == np.float32 and history.shape == (len(rows), 12, len(ids), 3)
            and list(history.shape) == multi['splits']['train']['shape']
            and data['split_counts']['train'] == len(rows) and data['station_count'] == len(ids),
            'History shape/dtype mismatch')
    return (history, ids, rows, report, labels, read_rows(sensors),
            read_rows(metadata_dir / 'source_sensor_subset.tsv', '\t'),
            multi['channel_semantics'], fingerprints, source)


def prepare(data_dir, history_dir, sensors, metadata_dir, selection, output):
    output = Path(output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Use a new output directory; existing/partial output is preserved')
    config = read_json(selection)
    (history, ids, rows, report, labels, published, raw,
     semantics, fingerprints, source) = load_inputs(
        Path(data_dir), Path(history_dir), Path(sensors), Path(metadata_dir), config)
    corridors, union, stations = select_corridors(config, ids, published, raw)
    selected = np.asarray(history[:, :, union, :])
    diagnostics = history_diagnostics(selected, labels)
    usable = np.isfinite(selected) & (selected >= 0)
    selected_report = {key: report[key] for key in REPORT_FIELDS}
    selected_report['distances'] = report['distances'][:, union, :]
    selected_report['station_ids'] = ids[union]
    summaries = []
    for corridor in corridors:
        idx = corridor['packed_node_indices']
        support = np.any(selected_report['distances'][:, idx] != 0, axis=-1)
        supported = support.any(1)
        segments = corridor['candidate_segments']
        span = corridor['covered_source_postmile_bounds']
        summaries.append({
            'id': corridor['id'], 'stations': len(idx), 'candidate_segments': len(segments),
            'covered_source_postmile_span': span[1] - span[0],
            'report_supported_train_windows': int(supported.sum()),
            'report_supported_distinct_incident_ids': len({row['incident_id'] for row, use in
                                                          zip(rows, supported) if use}),
            'segments_with_known_nonmainline': sum(bool(s['known_nonmainline']) for s in segments),
            'segments_with_coincident_postmile': sum('coincident_postmile' in s['metadata_flags']
                                                    for s in segments),
            'window_weighted_usable_fraction_by_channel': usable[:, :, idx, :].mean((0, 1, 2)).tolist(),
            'physics_ready': False})
    partial.mkdir(parents=True)
    np.save(partial / 'train_history.npy', selected, allow_pickle=False)
    np.save(partial / 'train_value_usable.npy', usable, allow_pickle=False)
    np.savez_compressed(partial / 'train_report.npz', **selected_report)
    fields = ('sample_index', 'incident_id', 'source_version', 'split', 'x_start',
              'x_end', 't0', 'report_time')
    write_rows(partial / 'train_rows.csv', [{key: row[key] for key in fields} for row in rows])
    write_rows(partial / 'stations.csv', stations)
    write_json(partial / 'corridors.json', corridors)
    write_json(partial / 'selection.json', config)
    outputs = {p.name: sha256(p) for p in sorted(partial.iterdir()) if p.is_file()}
    fingerprints['selection'] = sha256(selection)
    fingerprints['builder'] = sha256(Path(__file__))
    fingerprints['reader'] = sha256(REPO / 'src/utils/incident_corridor.py')
    result = {
        'schema': SCHEMA, 'status': 'CORRIDOR_TRAIN_HISTORY_PACK_COMPLETE',
        'scope': 'candidate_corridors_training_history_only',
        'main_training_ready': False, 'physics_ready': False,
        'model_training_performed': False, 'raw_flow_gap_or_Y_file_opened': False,
        'validation_array_opened': False, 'test_data_opened': False,
        'incident_semantic_fields_opened': False, 'network_access_performed': False,
        'samples': len(rows), 'packed_stations': len(union), 'history_shape': list(selected.shape),
        'axis_order': ['sample', 'history_step', 'packed_node', 'channel'],
        'channel_semantics': semantics, 'history_diagnostics': diagnostics,
        'mask_definition': 'finite_and_nonnegative_zero_retained_not_certified_observation_mask',
        'imputation_performed': False, 'normalization_performed': False,
        'report_support_meaning': 'original_report_location_v1_candidate_association_not_true_impact',
        'time_contract': {'interval_minutes': 5, 'history_steps': 12,
                          'x_label_offsets_from_t0_minutes': [-65, -10],
                          'interval_label': 'start_per_publisher', 'gap_minutes': 10,
                          'forecast_steps': 12, 'report_available_at': 't0',
                          'source_snapshot_is_initial_report_certified': False,
                          'online_semantics_certified': False},
        'remaining_physical_mapping': ['flow_units_and_lane_aggregation',
                                       'occupancy_speed_semantics_and_observation_operator',
                                       'directed_connections_and_physical_cell_lengths',
                                       'ramp_boundary_observations_or_bounded_model',
                                       'training_only_parameter_calibration'],
        'corridors': summaries, 'inputs_sha256': fingerprints,
        'metadata_source': source, 'outputs': outputs,
    }
    write_json(partial / 'summary.json', result)
    # Read back using the same consumer interface that the next module will use.
    loaded = CorridorHistory(partial)
    for corridor in corridors:
        first = loaded.window(0, corridor['id'])
        require(first['history_source_units'].shape == (12, len(corridor['station_ids']), 3),
                'Read-back corridor axes mismatch')
    del loaded
    partial.rename(output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--history-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path, default=Path(__file__).parent / 'physics_metadata')
    parser.add_argument('--selection', type=Path, default=Path(__file__).parent / 'incident_corridor_selection_v1.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.data_dir, args.history_dir, args.sensors, args.metadata_dir,
                     args.selection, args.output_dir)
    import json
    print(json.dumps({key: result[key] for key in ('status', 'samples', 'packed_stations',
                                                   'history_shape', 'physics_ready', 'corridors')},
                     ensure_ascii=False, indent=2, allow_nan=False))
    print('Saved corridor package: ' + str(args.output_dir), flush=True)


if __name__ == '__main__':
    main()
