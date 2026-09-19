"""Tests for the metadata-only matched non-incident candidate audit."""

from datetime import datetime, timedelta
import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.chronological.audit_matched_controls import (
    Incident, IncidentIndex, candidate_times, duration_minutes, five_number,
    load_incidents, load_protocol, locally_clean,
)


PROTOCOL = Path(__file__).resolve().parents[1] / 'experiments/chronological/matched_nonincident_v1.json'


class ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_prohibits_y_and_test(self):
        protocol = load_protocol(PROTOCOL)
        self.assertIn('forecast_Y', protocol['prohibited_matching_inputs'])
        self.assertEqual(set(protocol['splits']), {'train', 'val'})
        self.assertFalse(protocol['main_training_ready'])

    def test_duration_fallback_and_minimum_are_conservative(self):
        self.assertEqual(duration_minutes('', 120, 120), 120)
        self.assertEqual(duration_minutes('-4', 120, 120), 120)
        self.assertEqual(duration_minutes('20', 120, 120), 120)
        self.assertEqual(duration_minutes('205', 120, 120), 205)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(PROTOCOL)

    def test_candidates_preserve_weekday_slot_and_split_support(self):
        origin = datetime(2023, 1, 8, 12, 5)
        values = candidate_times(origin, datetime(2023, 1, 1), datetime(2023, 3, 1), self.protocol)
        self.assertTrue(values)
        self.assertNotIn(origin, values)
        self.assertTrue(all(value.weekday() == origin.weekday() for value in values))
        self.assertTrue(all(value.time() == origin.time() for value in values))
        self.assertTrue(all(value - timedelta(minutes=70) >= datetime(2023, 1, 1) for value in values))

    def test_dst_nominal_date_is_not_a_control_candidate(self):
        origin = datetime(2023, 3, 5, 8, 0)
        values = candidate_times(origin, datetime(2023, 1, 1), datetime(2023, 9, 1), self.protocol)
        self.assertNotIn(datetime(2023, 3, 12, 8, 0), values)

    def test_local_overlap_uses_positive_affected_sensors(self):
        candidate = datetime(2023, 2, 1, 12, 0)
        road = (80, 'E')
        event = Incident(candidate, 30.0, 120.0, 'near-node-20')
        index = IncidentIndex({road: [event]}, {road: [event.report_time]}, {road: 120.0}, {})
        self.assertFalse(locally_clean(candidate, road, [10.0, 20.0], index, self.protocol))
        self.assertTrue(locally_clean(candidate, road, [10.0, 19.0], index, self.protocol))

    def test_incident_outside_blackout_does_not_contaminate(self):
        candidate = datetime(2023, 2, 1, 12, 0)
        road = (80, 'E')
        report = candidate - timedelta(hours=5)
        event = Incident(report, 20.0, 120.0, 'old')
        index = IncidentIndex({road: [event]}, {road: [report]}, {road: 120.0}, {})
        self.assertTrue(locally_clean(candidate, road, [20.0], index, self.protocol))

    def test_summary_quantiles_are_deterministic(self):
        self.assertEqual(five_number([8, 0, 6, 4, 2]), [0, 2, 4, 6, 8])


class OutputBoundaryTests(unittest.TestCase):
    def test_protocol_with_y_matching_is_rejected(self):
        protocol = json.loads(PROTOCOL.read_text())
        protocol['prohibited_matching_inputs'].remove('forecast_Y')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, 'forecast Y'):
                load_protocol(path)

    def test_source_index_keeps_only_incidents_capable_of_sensor_exposure(self):
        rows = [
            {'incident_id': 'near', 'duration': '', 'Abs PM': '15', 'Fwy': '80',
             'dt': '02/01/2023 12:00:00', 'Freeway_direction': 'E'},
            {'incident_id': 'far', 'duration': '30', 'Abs PM': '100', 'Fwy': '80',
             'dt': '02/01/2023 12:00:00', 'Freeway_direction': 'E'},
            {'incident_id': 'other-road', 'duration': '30', 'Abs PM': '15', 'Fwy': '4',
             'dt': '02/01/2023 12:00:00', 'Freeway_direction': 'E'},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'incidents.tsv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
            index = load_incidents(
                path, {(80, 'E'): [10.0, 20.0]},
                {'duration_missing_or_invalid_minutes': 120,
                 'minimum_post_report_minutes': 120}, 10.0, 2023)
        self.assertEqual([event.incident_id for event in index.events[(80, 'E')]], ['near'])
        self.assertEqual(index.quality['indexed_within_sensor_exposure'], 1)
        self.assertEqual(index.quality['outside_all_sensor_postmile_exposure'], 1)
        self.assertEqual(index.quality['outside_sensor_road_keys'], 1)


if __name__ == '__main__':
    unittest.main()
