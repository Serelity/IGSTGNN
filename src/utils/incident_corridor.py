"""Candidate corridor inputs in source units, independent of a physics solver.

Only train history is packaged here. Metadata ordering is a hypothesis, never
a certified CTM grid; value usability does not establish sensor observation.
"""

import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'incident_corridor_history_v1'
CHANNELS = ['flow', 'occupancy', 'speed']
REPORT_FIELDS = ('distances', 'report_age_minutes', 'forecast_tod', 'forecast_dow',
                 'sample_indices', 'station_ids')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def read_rows(path, delimiter=','):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_rows(path, rows):
    require(bool(rows), 'Cannot write an empty table')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unique_index(rows, key='station_id'):
    result = {int(row[key]): row for row in rows}
    require(len(result) == len(rows), 'Duplicate metadata station IDs')
    return result


def validate_train_clock(rows, report):
    require(bool(rows), 'Empty train manifest')
    indices = [int(row['sample_index']) for row in rows]
    require(len(set(indices)) == len(indices), 'Duplicate training sample IDs')
    require(np.array_equal(report['sample_indices'], indices), 'Report sample order mismatch')
    for key in ('report_age_minutes', 'forecast_tod', 'forecast_dow'):
        require(report[key].shape == (len(rows),) and np.isfinite(report[key]).all(),
                'Invalid report clock: ' + key)
    times = []
    for i, row in enumerate(rows):
        require(row['split'] == 'train' and int(row['source_version']) == 8,
                'Only v8 training history is allowed')
        start, end, cutoff, issued = [datetime.fromisoformat(row[k]) for k in
                                     ('x_start', 'x_end', 't0', 'report_time')]
        require(all(t.tzinfo is None for t in (start, end, cutoff, issued)),
                'Expected nominal naive source calendar')
        require(start.second == start.microsecond == 0 and start.minute % 5 == 0,
                'History is not aligned to five-minute labels')
        labels = [start + timedelta(minutes=5 * h) for h in range(12)]
        require(end == labels[-1] and start == cutoff - timedelta(minutes=65),
                'History clock must preserve original 12-step window')
        require(all(t.year == 2023 and 1 <= t.month <= 8 for t in labels),
                'History must be in January-August 2023')
        age = (cutoff - issued).total_seconds() / 60
        require(0 < age <= 5 and abs(report['report_age_minutes'][i] - age) < 1e-5,
                'Report age or availability mismatch')
        require(report['forecast_tod'][i] == cutoff.hour * 12 + cutoff.minute // 5
                and report['forecast_dow'][i] == (cutoff.weekday() + 1) % 7,
                'Forecast clock mismatch')
        times.append(labels)
    return np.asarray(times, dtype='datetime64[m]')


def select_corridors(config, station_ids, published, raw):
    """Expand fixed anchors by source postmile, without traffic-based selection."""
    require(config.get('schema') == SCHEMA and config.get('selection_scope') ==
            'fixed_anchors_source_postmile_no_outcome_selection', 'Invalid selection schema')
    require(station_ids.ndim == 1 and np.issubdtype(station_ids.dtype, np.integer)
            and len(np.unique(station_ids)) == len(station_ids), 'Invalid package station IDs')
    public, full = unique_index(published), unique_index(raw)
    require(set(public) == set(map(int, station_ids)), 'Published/package station set mismatch')
    for sid in station_ids:
        p, r = public[int(sid)], full.get(int(sid))
        require(r is not None, 'Published station absent from full metadata')
        require(p['Type'] == r['Type'] == 'Mainline', 'Expected mainline stations')
        require(p['Fwy'] == r['Fwy Name'] and p['Direction'] == r['Direction']
                and p['County'] == r['County'], 'Road identity mismatch')
        for key in ('Abs PM', 'Lat', 'Lng'):
            a, b = float(p[key]), float(r[key])
            require(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-6,
                    'Source coordinate/postmile mismatch')
    positions = {int(sid): i for i, sid in enumerate(station_ids)}
    corridors, names = [], set()
    require(bool(config.get('corridors')), 'No configured corridors')
    for spec in config['corridors']:
        name = spec['id']
        require(isinstance(name, str) and name and name not in names, 'Duplicate/empty corridor ID')
        names.add(name)
        anchors = spec['anchor_station_ids']
        require(len(anchors) == 2 and len(set(anchors)) == 2 and
                all(s in public for s in anchors), 'Unknown or repeated anchors')
        road, direction = spec['road'], spec['direction']
        require(all(public[s]['Fwy'] == road and public[s]['Direction'] == direction
                    for s in anchors), 'Anchor road/direction mismatch')
        order = spec['candidate_travel_postmile_order']
        require(order in ('ascending', 'descending'), 'Postmile order must be explicit')
        margin = spec['extend_each_end_source_postmile']
        require(isinstance(margin, (int, float)) and not isinstance(margin, bool)
                and np.isfinite(margin) and margin > 0, 'Invalid expansion extent')
        apm = [float(public[s]['Abs PM']) for s in anchors]
        lo, hi = min(apm) - margin, max(apm) + margin
        group = [r for r in published if r['Fwy'] == road and r['Direction'] == direction
                 and lo <= float(r['Abs PM']) <= hi]
        group.sort(key=lambda r: (float(r['Abs PM']), int(r['station_id'])),
                   reverse=order == 'descending')
        require(len(group) >= 3, 'Corridor needs at least three mainline stations')
        ids = [int(r['station_id']) for r in group]
        pms = [float(r['Abs PM']) for r in group]
        source_group = [r for r in raw if r['Fwy Name'] == road and r['Direction'] == direction]
        require(all(np.isfinite(float(r['Abs PM'])) for r in source_group),
                'Nonfinite full-source postmile')
        segments = []
        for i, (a, b) in enumerate(zip(pms, pms[1:])):
            low, high = sorted((a, b))
            between = [r for r in source_group
                       if low - 1e-6 <= float(r['Abs PM']) <= high + 1e-6
                       and int(r['station_id']) not in (ids[i], ids[i + 1])]
            nonmain = [r for r in between if r['Type'] != 'Mainline']
            other_main = [r for r in between if r['Type'] == 'Mainline']
            flags = []
            if high - low <= 1e-6:
                flags.append('coincident_postmile')
            if nonmain:
                flags.append('known_ramp_or_connector')
            if other_main:
                flags.append('additional_source_mainline')
            segments.append({
                'candidate_from_station_id': ids[i], 'candidate_to_station_id': ids[i + 1],
                'source_postmile_difference': high - low,
                'known_nonmainline': [{'station_id': int(r['station_id']), 'type': r['Type'],
                                      'source_postmile': float(r['Abs PM'])} for r in nonmain],
                'additional_source_mainline_ids': [int(r['station_id']) for r in other_main],
                'metadata_flags': flags, 'direct_connection_certified': False,
                'physical_length_km': None, 'exchange_observation_available': False})
        corridors.append({
            'id': name, 'road': road, 'direction': direction, 'anchor_station_ids': anchors,
            'candidate_travel_postmile_order': order, 'direction_order_certified': False,
            'requested_source_postmile_bounds': [lo, hi],
            'covered_source_postmile_bounds': [min(pms), max(pms)],
            'station_ids': ids, 'package_node_indices': [positions[s] for s in ids],
            'source_postmiles': pms, 'candidate_segments': segments,
            'physics_ready': False})
    union = sorted({i for c in corridors for i in c['package_node_indices']})
    mapping = {p: i for i, p in enumerate(union)}
    for corridor in corridors:
        corridor['packed_node_indices'] = [mapping[p] for p in corridor['package_node_indices']]
    stations = [{'station_id': int(station_ids[p]), 'package_node_index': p,
                 'packed_node_index': i, 'road': public[int(station_ids[p])]['Fwy'],
                 'direction': public[int(station_ids[p])]['Direction'],
                 'source_postmile': float(public[int(station_ids[p])]['Abs PM']),
                 'latitude_source': float(public[int(station_ids[p])]['Lat']),
                 'longitude_source': float(public[int(station_ids[p])]['Lng'])}
                for i, p in enumerate(union)]
    return corridors, np.asarray(union, dtype=np.int64), stations


def history_diagnostics(history, labels):
    """Compare overlapping X exactly and summarize unique nominal slots only."""
    require(history.ndim == 4 and history.shape[1] == 12 and history.shape[3] == 3
            and labels.shape == history.shape[:2], 'Unexpected history/clock axes')
    flat = history.reshape(-1, history.shape[2], 3)
    unique, first, inverse = np.unique(labels.reshape(-1), return_index=True, return_inverse=True)
    values = flat[first]
    for start in range(0, len(flat), 256):
        require(np.array_equal(flat[start:start + 256], values[inverse[start:start + 256]],
                               equal_nan=True), 'Overlapping history has conflicting station/time values')
    stats = {}
    for channel, name in enumerate(CHANNELS):
        x = values[..., channel]
        usable = np.isfinite(x) & (x >= 0)
        stats[name] = {'unique_cells': int(x.size), 'usable_cells': int(usable.sum()),
                       'zero_cells': int((usable & (x == 0)).sum()),
                       'negative_cells': int((np.isfinite(x) & (x < 0)).sum()),
                       'nonfinite_cells': int((~np.isfinite(x)).sum()),
                       'stations_with_no_usable_history': int((~usable.any(0)).sum())}
    return {'unique_train_history_labels': len(unique),
            'first_label': str(unique[0]), 'last_label': str(unique[-1]),
            'channel_statistics': stats, 'overlap_consistency_checked': True}


class CorridorHistory:
    """Numpy reader for model development; no targets, scaling or imputation."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.summary = read_json(self.directory / 'summary.json')
        require(self.summary.get('schema') == SCHEMA and self.summary.get('status') ==
                'CORRIDOR_TRAIN_HISTORY_PACK_COMPLETE', 'Incomplete/wrong corridor package')
        required = {'train_history.npy', 'train_value_usable.npy', 'train_report.npz',
                    'train_rows.csv', 'stations.csv', 'corridors.json', 'selection.json'}
        require(set(self.summary['outputs']) == required, 'Unexpected corridor package files')
        for name, digest in self.summary['outputs'].items():
            require(sha256(self.directory / name) == digest, 'Package checksum mismatch: ' + name)
        self.corridors = {c['id']: c for c in read_json(self.directory / 'corridors.json')}
        self.rows = read_rows(self.directory / 'train_rows.csv')
        self.stations = read_rows(self.directory / 'stations.csv')
        self.history = np.load(self.directory / 'train_history.npy', mmap_mode='r', allow_pickle=False)
        self.usable = np.load(self.directory / 'train_value_usable.npy', mmap_mode='r', allow_pickle=False)
        with np.load(self.directory / 'train_report.npz', allow_pickle=False) as stored:
            self.report = {key: stored[key] for key in REPORT_FIELDS}
        require(self.history.shape == (len(self.rows), 12, len(self.stations), 3)
                and self.history.dtype == np.float32 and self.usable.shape == self.history.shape
                and self.usable.dtype == np.bool_, 'Corridor array shape/dtype mismatch')
        require(np.array_equal(self.report['station_ids'], [int(r['station_id']) for r in self.stations]),
                'Packed station axis mismatch')
        validate_train_clock(self.rows, self.report)
        require(self.report['distances'].shape == (len(self.rows), len(self.stations), 3)
                and np.isfinite(self.report['distances']).all(), 'Packed report axes mismatch')
        for corridor in self.corridors.values():
            idx = np.asarray(corridor['packed_node_indices'])
            require(np.issubdtype(idx.dtype, np.integer) and (idx >= 0).all()
                    and (idx < len(self.stations)).all() and len(np.unique(idx)) == len(idx)
                    and np.array_equal(self.report['station_ids'][idx], corridor['station_ids']),
                    'Corridor station mapping mismatch')

    def __len__(self):
        return len(self.rows)

    def window(self, position, corridor_id):
        require(isinstance(position, (int, np.integer)) and 0 <= position < len(self),
                'Sample position out of range')
        require(corridor_id in self.corridors, 'Unknown corridor')
        corridor = self.corridors[corridor_id]
        idx = corridor['packed_node_indices']
        # Slice the sample first so advanced node indexing preserves [time,node,channel].
        history = np.asarray(self.history[position])[:, idx, :]
        return {'sample_index': int(self.report['sample_indices'][position]),
                'station_ids': np.asarray(corridor['station_ids'], dtype=np.int64),
                'history_source_units': history,
                'value_usable': np.asarray(self.usable[position])[:, idx, :],
                'report_distances': self.report['distances'][position, idx],
                'report_age_minutes': float(self.report['report_age_minutes'][position]),
                'physics_ready': False}
