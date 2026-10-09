"""Compact train-X candidate pack and descriptive observation diagnostics."""
from pathlib import Path

import numpy as np

from src.utils.incident_corridor import read_json, read_rows, require, sha256

SCHEMA = 'network_incident_candidate_history_v1'


def rank_correlation(x, y):
    if len(x) < 2:
        return None
    ranks = []
    for a in (x, y):
        _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
        if len(counts) < 2:
            return None
        ranks.append((np.cumsum(counts) - (counts + 1) / 2)[inverse])
    return float(np.corrcoef(*ranks)[0, 1])


def station_diagnostics(values, labels, rules):
    """Each row is a unique source time, not an event-window repetition.

    q/(o*v) is a diagnostic hypothesis. It does not measure density or capacity.
    The later-training-block residual uses an alpha fitted to the early block.
    """
    require(values.shape == (len(labels), 3), 'Observation axes mismatch')
    usable = np.isfinite(values) & (values >= 0)
    positive = usable.all(1) & (values > 0).all(1)
    months = (labels.astype('datetime64[M]').astype(int) % 12) + 1
    references = []
    for k in range(3):
        valid = values[usable[:, k], k]
        ref = float(np.quantile(valid, rules['unit_scale_reference_quantile'])) if len(valid) else None
        references.append(ref if ref is not None and ref > 0 else None)
    joint = usable.all(1)
    result = {'unique_slots': len(labels), 'joint_usable_slots': int(joint.sum()),
              'positive_relation_slots': int(positive.sum()),
              'flow_usable_slots': int(usable[:, 0].sum()),
              'occupancy_usable_slots': int(usable[:, 1].sum()),
              'speed_usable_slots': int(usable[:, 2].sum()),
              'flow_reference': references[0], 'occupancy_reference': references[1],
              'speed_reference': references[2],
              'occupancy_speed_rank_correlation': rank_correlation(values[joint, 1], values[joint, 2]),
              'high_speed_proxy_slots': 0, 'low_speed_high_occupancy_proxy_slots': 0,
              'early_positive_slots': 0, 'late_positive_slots': 0,
              'alpha_early_source_units': None, 'ratio_p90_over_p10_early': None,
              'late_to_early_ratio_median': None, 'late_symmetric_relative_error_median': None,
              'relation_status': 'INSUFFICIENT_POSITIVE_DATA', 'physics_certified': False}
    if references[2] is not None and joint.any():
        speed = values[:, 2]
        occupancy_high = np.quantile(values[joint, 1], rules['high_occupancy_proxy_quantile'])
        result['high_speed_proxy_slots'] = int((joint & (speed >= rules['high_speed_proxy_reference_fraction'] * references[2])).sum())
        result['low_speed_high_occupancy_proxy_slots'] = int((joint & (speed < rules['low_speed_proxy_reference_fraction'] * references[2]) &
                                                            (values[:, 1] >= occupancy_high)).sum())
    early = positive & np.isin(months, rules['ratio_fit_months'])
    late = positive & np.isin(months, rules['ratio_check_months'])
    result.update(early_positive_slots=int(early.sum()), late_positive_slots=int(late.sum()))
    if min(early.sum(), late.sum()) < rules['minimum_positive_slots_per_block']:
        return result
    q, o, v = values.astype(np.float64).T
    early_ratios, late_ratios = q[early] / (o[early] * v[early]), q[late] / (o[late] * v[late])
    alpha = float(np.median(early_ratios))
    p10, p90 = np.quantile(early_ratios, [.1, .9])
    predicted = alpha * o[late] * v[late]
    result.update(alpha_early_source_units=alpha, ratio_p90_over_p10_early=float(p90 / p10),
                  late_to_early_ratio_median=float(np.median(late_ratios) / alpha),
                  late_symmetric_relative_error_median=float(np.median(2 * np.abs(predicted - q[late]) / (predicted + q[late]))),
                  relation_status='DIAGNOSTIC_COMPUTED_NOT_PHYSICAL_ACCEPTANCE')
    return result


class CandidateHistory:
    """Consumer returns [12,N,3] without storing repeated traffic windows."""
    def __init__(self, directory, verify=True):
        self.directory = Path(directory)
        self.summary = read_json(self.directory / 'summary.json')
        require(self.summary['schema'] == SCHEMA and self.summary['status'] == 'CANDIDATE_TRAIN_X_PACK_COMPLETE',
                'Incomplete or wrong candidate package')
        if verify:
            for name, digest in self.summary['outputs_sha256'].items():
                require(sha256(self.directory / name) == digest, 'Candidate package checksum mismatch: ' + name)
        self.values = np.load(self.directory / 'history_values.npy', mmap_mode='r', allow_pickle=False)
        self.usable = np.load(self.directory / 'history_usable.npy', mmap_mode='r', allow_pickle=False)
        self.indices = np.load(self.directory / 'history_index.npy', allow_pickle=False)
        self.labels = np.load(self.directory / 'history_labels.npy', allow_pickle=False)
        self.station_ids = np.load(self.directory / 'station_ids.npy', allow_pickle=False)
        self.events = read_rows(self.directory / 'train_events.csv')
        require(self.indices.shape == (len(self.events), 12) and np.issubdtype(self.indices.dtype, np.integer)
                and self.indices.min() >= 0 and self.indices.max() < len(self.labels), 'Candidate history index mismatch')
        require(self.values.shape == (len(self.labels), len(self.station_ids), 3)
                and self.usable.shape == self.values.shape and self.usable.dtype == np.bool_, 'Candidate value axes mismatch')

    def window(self, index):
        require(0 <= index < len(self.events), 'Candidate sample index outside range')
        idx = self.indices[index]
        return {'history_source_units': np.asarray(self.values[idx]),
                'value_usable': np.asarray(self.usable[idx]), 'history_labels': self.labels[idx],
                'event': self.events[index], 'station_ids': self.station_ids}

    def close(self):
        for array in (self.values, self.usable):
            array._mmap.close()
