"""Frozen train-X adapter for M2.1; no targets, future traffic or fitting."""
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.utils.incident_candidate_history import CandidateHistory
from src.utils.incident_corridor import read_json, read_rows, require, sha256


class CapacityHistoryInputs:
    def __init__(self, directory):
        directory = Path(directory)
        summary = read_json(directory / 'summary.json')
        expected = {'history_values.npy', 'history_usable.npy', 'history_labels.npy', 'history_index.npy',
                    'station_ids.npy', 'train_events.csv', 'train_report.npz', 'station_observation_diagnostics.csv'}
        require(expected.issubset(summary['outputs_sha256']), 'Missing input hashes')
        require(summary['relative_references_scope'] == 'unique_candidate_train_X_station_channel_p95_not_capacity',
                'Use frozen training-only reference scales')
        self.pack = CandidateHistory(directory)
        try:
            rows = read_rows(directory / 'station_observation_diagnostics.csv')
            require([int(r['station_id']) for r in rows] == self.pack.station_ids.tolist(), 'Reference station axis mismatch')
            self.references = np.asarray([[float(r[c + '_reference']) for c in ('flow', 'occupancy', 'speed')]
                                          for r in rows], dtype=np.float32)
            require(np.isfinite(self.references).all() and (self.references > 0).all(), 'Invalid training references')
            with np.load(directory / 'train_report.npz', allow_pickle=False) as report:
                require(np.array_equal(report['station_ids'], self.pack.station_ids)
                        and np.array_equal(report['candidate_indices'], np.arange(len(self.pack.events))), 'Report axis mismatch')
                self.report = report['distances'].copy()
                self.age = report['report_age_minutes'].copy()
            require(self.report.shape == (len(self.pack.events), len(rows), 3)
                    and self.age.shape == (len(self.pack.events),), 'Report dimensions mismatch')
            # Validate every clock before any model use, not just selected examples.
            for i, event in enumerate(self.pack.events):
                cutoff, issued = datetime.fromisoformat(event['t0']), datetime.fromisoformat(event['report_time'])
                require(event['split'] == 'train' and cutoff.year == 2023 and 1 <= cutoff.month <= 8,
                        'Expected candidate training events only')
                age = (cutoff - issued).total_seconds() / 60
                require(0 < age <= 5 and abs(float(self.age[i]) - age) < 1e-5, 'Report age/clock mismatch')
                labels = self.pack.labels[self.pack.indices[i]]
                expected_labels = np.datetime64(cutoff, 'm') + np.arange(-65, -5, 5).astype('timedelta64[m]')
                require(np.array_equal(labels, expected_labels), 'History crosses the frozen X clock')
            self.summary_sha256 = sha256(directory / 'summary.json')
        except Exception:
            self.pack.close()
            raise

    def batch(self, indices, elapsed_minutes, device='cpu'):
        indices = np.asarray(indices)
        require(indices.ndim == 1 and len(indices) > 0 and np.issubdtype(indices.dtype, np.integer)
                and indices.min() >= 0 and indices.max() < len(self.pack.events), 'Invalid batch indices')
        lookup = self.pack.indices[indices]
        tensor = lambda value, dtype: torch.as_tensor(np.asarray(value).copy(), dtype=dtype, device=device)
        return {'history': tensor(self.pack.values[lookup], torch.float32),
                'valid': tensor(self.pack.usable[lookup], torch.bool),
                'references': tensor(self.references, torch.float32),
                'report_features': tensor(self.report[indices], torch.float32),
                'report_age_minutes': tensor(self.age[indices], torch.float32),
                'elapsed_minutes': tensor(elapsed_minutes, torch.float32)}

    def close(self):
        self.pack.close()
