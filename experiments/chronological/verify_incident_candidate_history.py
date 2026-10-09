"""Independent baseline-X replay and duplicate-stream QC; no target values."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.incident_candidate_history import CandidateHistory
from src.utils.incident_corridor import read_json, read_rows, require, sha256, write_json
from experiments.chronological.prepare_incident_corridors import verified


def verify(pack_dir, data_dir, baseline_history_dir, output):
    require(not output.exists(), 'Use a new verification output')
    reader = CandidateHistory(pack_dir)
    try:
        verified(data_dir / 'summary.json', reader.summary['inputs_sha256']['data_summary'])
        original = read_json(data_dir / 'summary.json')
        verified(data_dir / 'train_manifest.csv', original['files']['train_manifest.csv'])
        baseline_rows = read_rows(data_dir / 'train_manifest.csv')
        by_id = {r['incident_id']: i for i, r in enumerate(baseline_rows)}
        multi = read_json(baseline_history_dir / 'summary.json')
        require(multi['inputs']['data_summary_sha256'] == reader.summary['inputs_sha256']['data_summary']
                and multi['status'] == 'MULTICHANNEL_HISTORY_MATERIALIZATION_COMPLETE'
                and multi['acceptance']['gate_passed'] is True and multi['engineering_check'] is False,
                'Unexpected baseline history identity')
        baseline_file = baseline_history_dir / 'train_history.npy'
        verified(baseline_file, multi['outputs']['train_history.npy']['sha256'])
        baseline = np.load(baseline_file, mmap_mode='r', allow_pickle=False)
        require(baseline.shape == (len(baseline_rows), 12, len(reader.station_ids), 3), 'Baseline history axes mismatch')
        pairs = []
        for i, e in enumerate(reader.events):
            if e['incident_id'] in by_id:
                j = by_id[e['incident_id']]
                require(all(e[k] == baseline_rows[j][k] for k in ('t0', 'report_time', 'x_start', 'x_end')),
                        'Common event clock changed')
                pairs.append((i, j))
        for start in range(0, len(pairs), 32):
            a, b = zip(*pairs[start:start+32])
            actual = np.asarray(reader.values[reader.indices[list(a)]])
            expected = np.asarray(baseline[list(b)])
            require(np.array_equal(actual, expected, equal_nan=True), 'Common event X differs from baseline')
        baseline._mmap.close()
        # Check masks across the entire unique X store, not just the first window.
        for start in range(0, len(reader.labels), 512):
            x = np.asarray(reader.values[start:start+512])
            require(np.array_equal(reader.usable[start:start+512], np.isfinite(x) & (x >= 0)), 'Value mask mismatch')
        stream_groups = defaultdict(list)
        for i, sid in enumerate(reader.station_ids):
            digest = hashlib.sha256(np.ascontiguousarray(reader.values[:, i]).tobytes()).hexdigest()
            stream_groups[digest].append((i, int(sid)))
        source_rows = []
        for month in range(1, 9):
            name = f'source_month_{month:02d}.json'
            verified(data_dir / name, original['files'][name])
            source_rows.append({r['published_node_index']: r['sha256'] for r in read_json(data_dir / name)['rows']})
        duplicates = []
        for group in stream_groups.values():
            if len(group) > 1:
                duplicates.append({'station_ids': [sid for _, sid in group],
                                   'all_eight_source_month_payloads_identical': all(len({rows[i] for i, _ in group}) == 1 for rows in source_rows)})
        diagnostics = read_rows(Path(pack_dir) / 'station_observation_diagnostics.csv')
        percentiles = {}
        for field in ('ratio_p90_over_p10_early', 'late_to_early_ratio_median',
                      'late_symmetric_relative_error_median', 'occupancy_speed_rank_correlation',
                      'low_speed_high_occupancy_proxy_slots'):
            values = [float(r[field]) for r in diagnostics if r[field] != '']
            percentiles[field] = {'available_stations': len(values),
                                  'p10_p50_p90': np.quantile(values, [.1, .5, .9]).tolist() if values else None}
        result = {'status': 'CANDIDATE_HISTORY_VERIFICATION_PASS', 'common_events_exactly_equal': len(pairs),
                  'common_history_scalar_cells_compared': len(pairs)*12*len(reader.station_ids)*3,
                  'all_unique_value_masks_match': True, 'distinct_three_channel_streams': len(stream_groups),
                  'duplicate_stream_groups': duplicates, 'observation_diagnostic_percentiles': percentiles,
                  'low_speed_high_occupancy_proxy_zero_stations': sum(int(r['low_speed_high_occupancy_proxy_slots']) == 0 for r in diagnostics),
                  'low_speed_high_occupancy_proxy_under_100_stations': sum(int(r['low_speed_high_occupancy_proxy_slots']) < 100 for r in diagnostics),
                  'proxy_count_thresholds_are_descriptive_not_selection': True,
                  'physics_or_capacity_certified': False, 'new_candidate_Y_opened': False,
                  'validation_test_arrays_opened': False,
                  'inputs_sha256': {'package_summary': sha256(Path(pack_dir) / 'summary.json'),
                                   'baseline_train_history': sha256(baseline_file), 'verifier': sha256(Path(__file__))}}
        write_json(output, result)
        return result
    finally:
        reader.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pack-dir', 'data-dir', 'baseline-history-dir', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    import json
    print(json.dumps(verify(a.pack_dir, a.data_dir, a.baseline_history_dir, a.output), indent=2))


if __name__ == '__main__':
    main()
