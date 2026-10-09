"""Coverage must be joint; metadata coverage must not become physical readiness."""

from datetime import datetime, timedelta
import unittest

import numpy as np

from experiments.chronological.audit_incident_network_requirements import (
    inspect_network, joint_history_coverage, trace_training_reports,
)
from experiments.chronological.prepare_context import spatial_features


class NetworkRequirementsTests(unittest.TestCase):
    def make_inputs(self):
        ids = np.array([30, 10, 20, 40])
        public = [dict(station_id=s, Fwy='SR4-W' if s != 40 else 'SR160-N',
                       Direction='W' if s != 40 else 'N', County='Contra Costa', Type='Mainline',
                       **{'Abs PM': float(i), 'Lat': 38., 'Lng': -122.})
                  for i, s in enumerate(ids)]
        raw = [{('Fwy Name' if k == 'Fwy' else k): v for k, v in r.items()} for r in public]
        ramp = dict(raw[0], station_id=90, Type='On Ramp')
        ramp['Abs PM'] = .5
        raw.append(ramp)
        x = np.ones((1, 12, 4, 3), dtype=np.float32)
        labels = np.array([[datetime(2023, 1, 3) + timedelta(minutes=5*h) for h in range(12)]],
                          dtype='datetime64[m]')
        d = np.zeros((1, 4, 3), dtype=np.float32)
        adjacency = np.ones((4, 4)) - np.eye(4)
        return x, labels, ids, public, raw, d, adjacency

    def test_marginal_channels_do_not_imply_joint_same_slot_availability(self):
        x, labels, *_ = self.make_inputs()
        x[0, :6, 0, 0] = np.nan
        x[0, 6:, 0, 1] = -1
        _, _, marginal, joint = joint_history_coverage(x, labels)
        self.assertEqual(marginal[0].tolist(), [6, 6, 12])
        self.assertEqual(joint[0], 0)
        self.assertEqual(joint[1], 12)

    def test_all_groups_including_singletons_remain_and_physics_is_unestablished(self):
        args = self.make_inputs()
        summary, stations, roads, pairs = inspect_network(*args)
        self.assertEqual(summary['stations'], 4)
        self.assertEqual(summary['common_input_presence_stations'], 4)
        self.assertEqual(summary['road_groups_without_a_candidate_pair'], 1)
        self.assertEqual(summary['pairs_with_known_nonmainline'], 1)
        self.assertEqual(len(pairs), 2)
        self.assertIsNone(summary['physical_equation_joint_coverage'])
        self.assertFalse(summary['semantic_or_physical_readiness_inferred_from_numeric_coverage'])
        self.assertEqual([r['station_id'] for r in stations], args[2].tolist())
        self.assertEqual({r['road'] for r in roads}, {'SR4-W', 'SR160-N'})

    def test_repeated_slots_are_deduplicated_and_conflicts_rejected(self):
        x, labels, *_ = self.make_inputs()
        repeated = np.concatenate([x, x], axis=0)
        doubled_labels = np.concatenate([labels, labels], axis=0)
        times, _, _, joint = joint_history_coverage(repeated, doubled_labels)
        self.assertEqual(len(times), 12)
        self.assertTrue((joint == 12).all())
        repeated[1, 0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            joint_history_coverage(repeated, doubled_labels)

    def test_joint_static_graph_and_history_requirements_are_intersected(self):
        args = list(self.make_inputs())
        args[0][..., 0, 2] = np.nan
        args[-1][1, :] = args[-1][:, 1] = 0
        summary, stations, *_ = inspect_network(*args)
        self.assertEqual(summary['common_input_presence_stations'], 2)
        self.assertFalse(stations[0]['common_input_presence'])
        self.assertFalse(stations[1]['common_input_presence'])

    def test_input_presence_is_not_positive_incident_coverage(self):
        args = list(self.make_inputs())
        args[5][0, 0, 1] = 1
        summary, _, roads, _ = inspect_network(*args)
        self.assertEqual(summary['common_input_presence_stations'], 4)
        self.assertEqual(summary['stations_with_any_report_support'], 1)
        self.assertEqual(summary['road_groups_with_any_report_support'], 1)
        self.assertEqual(summary['station_report_support_fraction'], .25)
        self.assertEqual(sum(r['stations_with_any_report_support'] for r in roads), 1)

    def test_identity_replay_checks_training_scope_and_detects_mismatch(self):
        sensors = self.make_inputs()[3]
        event = dict(incident_id='a', report_time='2023-01-03T00:00:00',
                     freeway=4, direction='W', postmile=1.)
        records = [dict(sample_index=4, status='unique_metadata_candidate', candidates=[event]),
                   dict(sample_index=99, status='not_a_training_row')]
        rows = [dict(sample_index=4, incident_id='a', report_time=event['report_time'])]
        distances = spatial_features(event, sensors)[None]
        result = trace_training_reports(records, rows, sensors, distances)
        self.assertEqual(result['train_event_road_counts'], {'4-W': 1})
        self.assertEqual(result['training_rows_replayed'], 1)
        distances[0, 0, 1] = 0
        with self.assertRaisesRegex(ValueError, 'replay mismatch'):
            trace_training_reports(records, rows, sensors, distances)

    def test_zero_is_valid_and_corrupt_axes_or_graph_are_rejected(self):
        args = list(self.make_inputs())
        args[0][:] = 0
        self.assertEqual(inspect_network(*args)[0]['all_history_slots_jointly_usable_stations'], 4)
        args[-1][0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'Invalid graph'):
            inspect_network(*args)
        with self.assertRaisesRegex(ValueError, 'axes'):
            joint_history_coverage(np.zeros((1, 26, 4, 3)), args[1])


if __name__ == '__main__':
    unittest.main()
