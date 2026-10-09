import copy
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, mock_open
import unittest

import numpy as np

from experiments.chronological.prepare_incident_physics_evidence import (
    unique_history, candidate_inventory, historical_lane_audit, check_source_metadata, compare_pems_records,
)


def fixtures():
    rows = []
    start = datetime(2023, 1, 3)
    for offset in (0, 5):
        t = start + timedelta(minutes=offset)
        rows.append(dict(split='train', source_version='8', x_start=t.isoformat(),
                         x_end=(t + timedelta(minutes=55)).isoformat()))
    flow = np.full((2, 26, 2), np.nan)
    sequence = np.arange(26).reshape(13, 2)
    flow[0, :12], flow[1, :12] = sequence[:12], sequence[1:]
    published = [dict(station_id=str(i), Fwy='I80-E', Direction='E', County='Contra Costa',
                      Type='Mainline', **{'Abs PM': str(pm), 'Lat': str(37 + pm / 100), 'Lng': '-122'})
                 for i, pm in ((10, 1), (20, 2), (30, 3))]
    raw = [{**r, 'Fwy Name': r['Fwy']} for r in published]
    return flow, rows, published, raw


class PhysicsEvidenceTests(unittest.TestCase):
    def test_deduplicates_overlap_and_never_uses_gap_or_y(self):
        flow, rows, _, _ = fixtures()
        labels, values = unique_history(flow, rows)
        self.assertEqual(len(labels), 13)
        np.testing.assert_array_equal(values, np.arange(26).reshape(13, 2))
        flow[:, 12:] = -1e30
        a, b = unique_history(flow, rows)
        np.testing.assert_array_equal(a, labels)
        np.testing.assert_array_equal(b, values)
        flow[1, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            unique_history(flow, rows)

    def test_missing_values_remain_missing_and_nontraining_rows_fail(self):
        flow, rows, _, _ = fixtures()
        flow[0, 1, 0] = flow[1, 0, 0] = np.nan
        self.assertTrue(np.isnan(unique_history(flow, rows)[1][1, 0]))
        rows[0]['split'] = 'val'
        with self.assertRaisesRegex(ValueError, 'train'):
            unique_history(flow, rows)
        _, rows, _, _ = fixtures()
        rows[0]['x_start'] = '2023-09-01T00:00:00'
        with self.assertRaisesRegex(ValueError, 'January-August'):
            unique_history(flow, rows)

    def test_hidden_ramp_at_boundary_prevents_metadata_prefilter_pass(self):
        _, _, published, raw = fixtures()
        raw.append({**raw[1], 'station_id': '90', 'Type': 'On Ramp'})
        pairs = candidate_inventory(published, raw, np.ones((13, 3)), np.ones((2, 3), bool))
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(not p['metadata_prefilter_pass'] for p in pairs))
        self.assertEqual(pairs[0]['known_nonmainline_ids'], '90')
        raw[-1]['Direction'] = 'W'
        pairs = candidate_inventory(published, raw, np.ones((13, 3)), np.ones((2, 3), bool))
        self.assertTrue(all(p['metadata_prefilter_pass'] for p in pairs))
        self.assertTrue(all(not p['closed_boundary_certified'] for p in pairs))

    def test_coincident_mainline_is_ambiguous_even_without_ramps(self):
        _, _, published, raw = fixtures()
        raw.append({**raw[0], 'station_id': '91'})
        pairs = candidate_inventory(published, raw, np.ones((13, 3)), np.ones((2, 3), bool))
        self.assertFalse(pairs[0]['metadata_prefilter_pass'])
        self.assertIn('additional_mainline', pairs[0]['metadata_flags'])

    def test_lane_id_match_does_not_certify_year_or_override_coordinate_mismatch(self):
        _, _, published, _ = fixtures()
        prior = [{**published[0], 'ID': '10', 'Lanes': '3'}]
        out = historical_lane_audit(published, prior)
        self.assertTrue(out[0]['historical_road_coordinate_match'])
        self.assertFalse(out[0]['lane_count_2023_certified'])
        self.assertIsNone(out[1]['historical_lanes'])
        prior[0]['Lat'] = '39'
        self.assertFalse(historical_lane_audit(published, prior)[0]['historical_road_coordinate_match'])

    def test_source_identity_fields_must_agree(self):
        _, _, published, raw = fixtures()
        check_source_metadata(published, raw)
        bad = copy.deepcopy(raw)
        bad[0]['Direction'] = 'S'
        with self.assertRaisesRegex(ValueError, 'differs'):
            check_source_metadata(published, bad)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            check_source_metadata(published, raw + [raw[0]])

    def source_comparison(self, scale=1, zeros=False, duplicate=False):
        start = datetime(2023, 1, 3)
        labels = np.asarray([start + timedelta(minutes=5 * k) for k in range(50)], dtype='datetime64[m]')
        values = np.zeros((50, 2)) if zeros else np.arange(100).reshape(50, 2) + 1
        rows = []
        for i, label in enumerate(labels):
            for j, station in enumerate((10, 20)):
                stamp = label.astype(object).strftime('%m/%d/%Y %H:%M:%S')
                rows.append(f'{stamp},{station},4,80,E,ML,0.5,40,100,{values[i,j]},0.1,65')
        # A non-X row must not resolve or break matching even with absurd flow.
        rows.append('09/01/2023 00:00:00,10,4,80,E,ML,0.5,40,100,999999,0.1,65')
        if duplicate:
            rows.append('01/03/2023 00:00:00,10,4,80,E,ML,0.5,40,100,999,0.1,65')
        with patch('builtins.open', mock_open(read_data='\n'.join(rows))), patch(
                'experiments.chronological.prepare_incident_physics_evidence.sha256', return_value='fixture'):
            return compare_pems_records(Path('fixture.txt'), labels, values * scale, [10, 20])

    def test_raw_reference_distinguishes_count_from_hourly_rate_on_train_x(self):
        for scale, unit in ((1, 'vehicles_per_5min'), (12, 'vehicles_per_hour')):
            result = self.source_comparison(scale=scale)
            self.assertEqual(result['inferred_array_flow_unit'], unit)
            self.assertEqual(result['matched_valid_train_x_cells'], 100)
            self.assertFalse(result['official_file_provenance_independently_certified'])

    def test_all_zero_or_mismatched_reference_does_not_resolve_units(self):
        self.assertIsNone(self.source_comparison(zeros=True)['inferred_array_flow_unit'])
        self.assertIsNone(self.source_comparison(scale=2)['inferred_array_flow_unit'])

    def test_conflicting_reference_duplicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.source_comparison(duplicate=True)


if __name__ == '__main__':
    unittest.main()
