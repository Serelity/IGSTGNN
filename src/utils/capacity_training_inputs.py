"""M4.2 explicit train-loss / validation-evaluation access, never test access.

Input construction and target access are separate. All normalizers and graphs
come from TRAIN. Recorded report availability remains an offline assumption.
"""
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.chronological.prepare_incident_corridors import verified
from src.utils.capacity_fusion_inputs import OriginalCapacityInputs
from src.utils.capacity_network_inputs import CapacityNetworkInputs, InformationProfile, REPORT_COLUMNS, associate_reports
from src.utils.chronological import ChronologicalDataset
from src.utils.incident_corridor import read_json, read_rows, require, sha256


PREFIX_SHIFT_MINUTES = 20


def manifest_identity(rows):
    return [[int(r['sample_index']), r['incident_id'], r['t0']] for r in rows]


def earlier_events(rows):
    return [dict(r, t0=(datetime.fromisoformat(r['t0'])-timedelta(minutes=PREFIX_SHIFT_MINUTES)).isoformat())
            for r in rows]


def prefix_inputs(inputs):
    """New cutoff = t0-20; only original slots 0:8 enter, left-padded as absent.

    Relative slots -65:-50 are missing. Available slots -45:-10 correspond to
    the earlier prefix. Original slots 10:12 become targets [5,10],[10,15].
    The two intervening slots are never used by this prefix task.
    Reports must be separately reconstructed at the earlier cutoff.
    """
    require(set(inputs) == {'history', 'valid', 'references', 'labels'}, 'Prefix requires report-free legal history')
    history, valid = inputs['history'], inputs['valid']
    require(history.ndim == 4 and history.shape[1] == 12 and valid.shape == history.shape,
            'Invalid prefix history axes')
    return dict(history=torch.cat((torch.zeros_like(history[:, :4]), history[:, :8]), 1),
                valid=torch.cat((torch.zeros_like(valid[:, :4]), valid[:, :8]), 1),
                references=inputs['references'], labels=inputs['labels'])


class CutoffReports:
    def __init__(self, reports, structure, published):
        self.rows = sorted(associate_reports(reports, structure, published), key=lambda r: r['report_time'])
        require(len({r['incident_id'] for r in self.rows}) == len(self.rows), 'Duplicate report entity')
        self.times = [datetime.fromisoformat(r['report_time']) for r in self.rows]
        self.edges = len(structure['edges'])

    def batch(self, cutoffs, device='cpu', enabled=True):
        require(type(enabled) is bool and len(cutoffs) > 0, 'Invalid report request')
        selected = [self.rows[bisect_left(self.times, c-timedelta(minutes=60)):bisect_right(self.times, c)]
                    if enabled else [] for c in cutoffs]
        b, r, e = len(cutoffs), max(map(len, selected)), self.edges
        weights, distance = np.zeros((b, r, e), np.float32), np.zeros((b, r, e), np.float32)
        ages, confidence, present = np.zeros((b, r), np.float32), np.zeros((b, r), np.float32), np.zeros((b, r), bool)
        for j, group in enumerate(selected):
            for k, row in enumerate(group):
                present[j, k] = True
                ages[j, k] = (cutoffs[j]-datetime.fromisoformat(row['report_time'])).total_seconds()/60
                edge = int(row['edge_index'])
                if edge >= 0:
                    weights[j, k, edge] = confidence[j, k] = float(row['confidence'])
                    distance[j, k, edge] = float(row['distance_km'])
        return {key: torch.as_tensor(value, device=device) for key, value in
                dict(weights=weights, distance=distance, ages=ages, confidence=confidence, present=present).items()}


class CapacityDevelopmentInputs:
    def __init__(self, original, network_dir, history_dir, report_bundle, *, ramp_exchanges=True):
        self.original = original
        self.network = CapacityNetworkInputs(network_dir, original, allow_exploratory=True,
                                             profile=InformationProfile(True, ramp_exchanges))
        self.fingerprints = dict(original.fingerprints)
        self.fingerprints['network/summary.json'] = sha256(Path(network_dir)/'summary.json')
        data, history_dir = original.data_dir, Path(history_dir)
        self.datasets = {}
        self.histories = {'train': original.values}
        self.adapters = {'train': original}
        for split in ('train', 'val'):
            for name in (f'{split}_manifest.csv', f'{split}_flow.npy'):
                self.fingerprints['data/'+name] = verified(data/name, original.frozen['files'][name])
            context = read_json(data/'context_manifest.json')
            name = f'{split}_context.npz'
            self.fingerprints['data/'+name] = verified(data/name, context['outputs'][name])
            self.datasets[split] = ChronologicalDataset(data, split)
        multi = read_json(history_dir/'summary.json')
        name = 'val_history.npy'
        self.fingerprints['history/'+name] = verified(history_dir/name, multi['outputs'][name]['sha256'])
        self.histories['val'] = np.load(history_dir/name, mmap_mode='r', allow_pickle=False)
        self.events = {s: self.datasets[s].rows for s in ('train', 'val')}
        require(manifest_identity(self.events['train']) == manifest_identity(original.events), 'Train row identity changed')
        for split in ('train', 'val'):
            values, dataset = self.histories[split], self.datasets[split]
            require(values.shape == (len(dataset), 12, len(original.station_ids), 3), 'History split axes mismatch')
            require(np.array_equal(dataset.station_ids, original.station_ids), 'Station axis changed')
            for start in range(0, len(dataset), 128):
                require(np.array_equal(values[start:start+128, :, :, 0], dataset.flow[start:start+128, :12], equal_nan=True),
                        'History flow anchor mismatch')
            for row in dataset.rows:
                t0, first, last = [datetime.fromisoformat(row[k]) for k in ('t0', 'x_start', 'x_end')]
                require(first == t0-timedelta(minutes=65) and last == t0-timedelta(minutes=10), 'History clock changed')
        val = self.datasets['val']
        self.adapters['val'] = SimpleNamespace(values=self.histories['val'], events=val.rows,
            trigger=val.context, scaler=original.scaler, references=original.references)
        bundle = Path(report_bundle)
        manifest = read_json(bundle/'manifest.json')
        from experiments.chronological.audit_incident_expansion import RAW_SHA256
        require(manifest['schema'] == 'incident_development_location_excerpt_m42_v1'
                and manifest['raw_sha256'] == RAW_SHA256
                and manifest['test_accessed'] is False and manifest['prefix_shift_minutes'] == PREFIX_SHIFT_MINUTES,
                'Unexpected development report scope')
        for split in ('train', 'val'):
            require(manifest['manifest_identity'][split] == manifest_identity(self.events[split]), 'Report split identity mismatch')
        require(manifest['policy']['lookback_minutes'] == 60
                and manifest['policy']['recorded_dt_and_location_at_first_report_assumed'] is True
                and manifest['policy']['duration_description_type_included'] is False, 'Report policy changed')
        self.fingerprints['reports/manifest.json'] = sha256(bundle/'manifest.json')
        self.fingerprints['reports/locations.tsv'] = verified(bundle/'locations.tsv', manifest['file_sha256'])
        reports = read_rows(bundle/'locations.tsv', '\t')
        require(len(reports) == manifest['rows'] and all(tuple(r) == REPORT_COLUMNS for r in reports), 'Report whitelist/count mismatch')
        self.reports = CutoffReports(reports, self.network.structure, original.published)
        by_id = {r['incident_id']: r for r in reports}
        from experiments.chronological.prepare_context import spatial_features
        public = {int(r['station_id']): r for r in original.published}
        ordered = [public[int(sid)] for sid in original.station_ids]
        for split in ('train', 'val'):
            for i, row in enumerate(self.events[split]):
                report = by_id.get(row['incident_id'])
                require(report is not None and report['report_time'] == row['report_time'], 'Missing native trigger replay')
                expected = spatial_features(dict(freeway=int(report['road_number']), direction=report['direction'],
                                                 postmile=float(report['postmile'])), ordered)
                require(np.array_equal(expected, self.datasets[split].context['distances'][i]), 'Native location replay mismatch')
        self.common_mask = np.load(Path(network_dir)/'common_structure_mask.npy', allow_pickle=False)
        self.structure_mask = np.asarray(self.network.structure['operator_mask'], bool)
        self.road_groups = {}
        for i, sid in enumerate(original.station_ids):
            row = public[int(sid)]
            self.road_groups.setdefault(str(row['Fwy']), np.zeros(len(original.station_ids), bool))[i] = True

    def batch(self, split, indices, device='cpu', *, new_reports=True, prefix=False):
        require(split in ('train', 'val') and (not prefix or split == 'train'), 'Prefix supervision is train-only')
        adapter = self.adapters[split]
        batch = OriginalCapacityInputs.batch(adapter, indices, device)
        cutoffs = [datetime.fromisoformat(self.events[split][int(i)]['t0']) for i in indices]
        if prefix:
            batch = dict(capacity_inputs=prefix_inputs(batch['capacity_inputs']))
            cutoffs = [c-timedelta(minutes=PREFIX_SHIFT_MINUTES) for c in cutoffs]
        batch['capacity_inputs']['reports'] = self.reports.batch(cutoffs, device, new_reports)
        return batch

    def targets(self, split, indices, device='cpu', *, prefix=False):
        require(split in ('train', 'val') and (not prefix or split == 'train'), 'Prefix targets are train-only')
        raw = (np.asarray(self.histories['train'][indices, 10:12, :, 0]).copy() if prefix else
               np.asarray(self.datasets[split].flow[indices, 14:26]).copy())
        valid = np.isfinite(raw) & (raw >= 0)
        return (torch.tensor(np.where(valid, raw, 0)[..., None], device=device),
                torch.tensor(valid[..., None], device=device))

    def close(self):
        for dataset in self.datasets.values():
            dataset.flow._mmap.close()
        self.histories['val']._mmap.close()
