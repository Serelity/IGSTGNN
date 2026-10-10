"""M4.1 input boundaries, graph assumptions and optional information profiles."""
import copy
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import prepare_incident_capacity_network as builder
from experiments.chronological.prepare_context import spatial_features
from src.utils.capacity_network_inputs import (REPORT_COLUMNS, CapacityNetworkInputs, InformationProfile,
    associate_reports, cutoff_collections, metadata_graph, numeric_source_groups)
from src.utils.incident_corridor import read_json, read_rows, sha256, write_json


class NetworkInputTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).absolute().parents[1]/'experiments/chronological_runs'
        scratch.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix='network_test_', dir=scratch)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ids = np.array([30, 10, 20, 40, 51, 52, 53])
        self.public = [dict(station_id=sid, Fwy=road, Direction=direction, County='Contra Costa',
                            Type='Mainline', **{'Abs PM': pm, 'Lat': 37.+pm*.005, 'Lng': -122.})
                       for sid, road, direction, pm in [(10, 'SR4-W', 'W', 0), (20, 'SR4-W', 'W', 1),
                           (30, 'SR4-W', 'W', 2), (40, 'SR4-W', 'W', 3),
                           (51, 'SR4-E', 'E', 0), (52, 'SR4-E', 'E', 1), (53, 'SR4-E', 'E', 2)]]
        self.raw = [{('Fwy Name' if k == 'Fwy' else k): v for k, v in r.items()} for r in self.public]
        self.raw.append(dict(self.raw[0], station_id=99, Type='On Ramp', **{'Abs PM': 2.5}))
        self.reports = [dict(source_row_index=i, incident_id=sid, report_time='2023-01-03T'+time,
                             road_number=4, direction=direction, postmile=pm, latitude=37.+pm*.005, longitude=-122.)
                        for i, (sid, time, direction, pm) in enumerate([
                            ('a', '08:03:00', 'W', 1.5), ('b', '08:04:00', 'E', .5),
                            ('c', '08:09:00', 'W', .5), ('older', '07:30:00', 'W', 1.5)])]
        self.events = [dict(sample_index=100+i, incident_id=r['incident_id'], report_time=r['report_time'],
                            t0='2023-01-03T'+('08:05:00' if i < 2 else '08:10:00'))
                       for i, r in enumerate(self.reports[:3])]
        values = np.random.default_rng(1).uniform(1, 2, (3, 12, 7, 3)).astype(np.float32)
        values[1] = values[0]
        by_id = {r['station_id']: r for r in self.public}
        ordered = [by_id[int(sid)] for sid in self.ids]
        distances = np.stack([spatial_features(dict(freeway=4, direction=r['direction'], postmile=r['postmile']), ordered)
                              for r in self.reports[:3]])
        self.original = SimpleNamespace(station_ids=self.ids, events=self.events, values=values,
            published=self.public, raw=self.raw, fingerprints={'history': 'fixture'},
            semantics={'source_order': ['flow', 'occupancy', 'speed']}, trigger={'distances': distances})
        def batch(indices, device='cpu'):
            indices = np.asarray(indices)
            if indices.ndim != 1 or not len(indices) or indices.min() < 0 or indices.max() >= 3:
                raise ValueError('Invalid indices')
            history = torch.tensor(values[indices], device=device)
            return dict(x=history.clone(), incident=dict(distances=torch.tensor(distances[indices], device=device)),
                        capacity_inputs=dict(history=history, valid=torch.ones_like(history, dtype=torch.bool),
                                             references=torch.ones(7, 3, device=device), labels=torch.arange(-65., -5., 5., device=device)))
        self.original.batch = batch
        self.bundle = self.root/'bundle'
        self.bundle.mkdir()
        with (self.bundle/'train_locations.tsv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS, delimiter='\t')
            writer.writeheader()
            writer.writerows(self.reports)
        write_json(self.bundle/'manifest.json', dict(schema='incident_train_location_excerpt_v1',
            raw_sha256=builder.RAW_SHA256, rows=4, conflicting_entity_ids=[],
            train_manifest_identity=[[r[k] for k in ('sample_index', 'incident_id', 't0')] for r in self.events],
            file_sha256=sha256(self.bundle/'train_locations.tsv'),
            policy=dict(lookback_minutes=60, association_radius_km=1., recorded_dt_and_location_at_first_report_assumed=True,
                        duration_description_type_included=False)))

    def build(self):
        return builder.prepare(self.original, self.bundle, self.root/'pack')

    def test_direction_gaps_full_axis_and_three_distinct_nodes(self):
        g = metadata_graph(self.ids, self.public, self.raw)
        self.assertEqual(g['admitted_chains'], [[51, 52, 53], [30, 20, 10]])
        self.assertFalse(g['operator_mask'][3])
        aliases = metadata_graph(self.ids, self.public, self.raw, [[10, 20]])
        self.assertEqual(aliases['admitted_chains'], [[51, 52, 53]])
        self.assertEqual(len(aliases['operator_mask']), 7)
        self.assertFalse(g['direct_topology_certified'])
        broken = copy.deepcopy(self.raw)
        broken.append(dict(broken[0], station_id=88, **{'Abs PM': .5}))
        self.assertTrue(any('additional_source_mainline_in_closed_interval' in p['break_reasons']
                            for p in metadata_graph(self.ids, self.public, broken)['candidate_pairs']))

    def test_known_ramps_are_explicit_budgeted_exchanges_unknown_connectors_cut(self):
        raw = copy.deepcopy(self.raw)
        raw.append(dict(raw[0], station_id=98, Type='Off Ramp', **{'Abs PM': 1.5}))
        g = metadata_graph(self.ids, self.public, raw, ramp_policy='explicit_lumped')
        self.assertTrue(all(g['operator_mask']))
        ramps = {sid: r for r in g['edges'] for sid in r['ramp_source_ids']}
        self.assertEqual(ramps[99]['destination_station'], 30)
        self.assertEqual(ramps[98]['source_station'], 30)
        outgoing = [r for r in g['edges'] if r['source_station'] == 30]
        self.assertEqual([r['weight'] for r in outgoing], [.5, .5])
        raw.append(dict(raw[0], station_id=97, Type='On Ramp', **{'Abs PM': 0.}))
        terminal = metadata_graph(self.ids, self.public, raw, ramp_policy='explicit_lumped')
        self.assertNotIn(97, terminal['unmapped_ramp_source_ids'])
        attachments = [r for r in terminal['edges'] if 97 in r['ramp_source_ids']]
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0]['destination_station'], 10)
        raw.pop()
        raw[-1]['Type'] = 'Fwy-Fwy'
        cut = metadata_graph(self.ids, self.public, raw, ramp_policy='explicit_lumped')
        self.assertTrue(any('unknown_connector_in_closed_interval' in p['break_reasons'] for p in cut['candidate_pairs']))

    def test_report_locality_half_open_direction_and_no_inlet_association(self):
        g = metadata_graph(self.ids, self.public, self.raw, ramp_policy='explicit_lumped')
        r = dict(self.reports[0], postmile=1., latitude=37.005)
        a = associate_reports([r], g, self.public)[0]
        edge = g['edges'][a['edge_index']]
        self.assertEqual((edge['source_station'], edge['destination_station']), (20, 10))
        far = associate_reports([dict(r, latitude=39.)], g, self.public)[0]
        self.assertEqual(far['edge_index'], -1)
        self.assertEqual(far['confidence'], 0.)

    def test_cutoff_lookback_inclusive_future_excluded_and_shared_keys_identical(self):
        reports = [dict(incident_id=str(i), report_time='2023-01-03T'+time) for i, time in enumerate(
            ['07:04:59', '07:05:00', '08:05:00', '08:05:01'])]
        c = cutoff_collections(self.events, reports)
        self.assertEqual(c['report_indices'][c['offsets'][0]:c['offsets'][1]], [1, 2])
        self.assertEqual(c['sample_cutoff_indices'][:2], [0, 0])
        self.assertEqual(c['unique_cutoff_sensitivity_weights'], [.5, .5, 1.])

    def test_pack_profiles_preserve_native_inputs_and_remap_edge_axes(self):
        result = self.build()
        self.assertFalse(result['main_training_ready'])
        self.assertEqual(result['jointly_certified_nodes'], 0)
        self.assertEqual(result['frozen_trigger_location_replays'], 3)
        with self.assertRaises(ValueError):
            CapacityNetworkInputs(self.root/'pack', self.original)
        native = self.original.batch(np.array([0, 1]))
        for ramps in (False, True):
            for reports in (False, True):
                reader = CapacityNetworkInputs(self.root/'pack', self.original, allow_exploratory=True,
                                               profile=InformationProfile(reports, ramps))
                batch = reader.batch(np.array([0, 1]))
                torch.testing.assert_close(batch['x'], native['x'])
                torch.testing.assert_close(batch['incident']['distances'], native['incident']['distances'])
                report = batch['capacity_inputs']['reports']
                self.assertEqual(report['weights'].shape[1] > 0, reports)
                torch.testing.assert_close(report['weights'][0], report['weights'][1])
                graph, weights = reader.graph()
                self.assertEqual(report['weights'].shape[2], graph.edges)
                self.assertFalse((report['weights'][..., graph.edge_index[0] < 0] != 0).any())
                for source in range(7):
                    outgoing = graph.edge_index[0] == source
                    if outgoing.any():
                        self.assertAlmostEqual(float(weights[outgoing].sum()), 1.)

    def test_integrity_and_rehashed_future_report_are_rejected(self):
        self.build()
        path = self.root/'pack/cutoffs.json'
        c = read_json(path)
        # c is issued at 08:09, so cannot belong to 08:05.
        c['report_indices'][0] = next(i for i, r in enumerate(read_rows(self.root/'pack/reports.csv')) if r['incident_id'] == 'c')
        write_json(path, c)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            CapacityNetworkInputs(self.root/'pack', self.original, allow_exploratory=True)
        summary = read_json(self.root/'pack/summary.json')
        summary['outputs_sha256']['cutoffs.json'] = sha256(path)
        write_json(self.root/'pack/summary.json', summary)
        with self.assertRaises(ValueError):
            CapacityNetworkInputs(self.root/'pack', self.original, allow_exploratory=True)

    def test_numeric_alias_detection_does_not_claim_physical_identity(self):
        values = self.original.values.copy()
        values[:, :, 2] = values[:, :, 0]
        self.assertEqual(numeric_source_groups(values, self.ids, batch_size=1), [[30, 20]])
        self.assertEqual(numeric_source_groups(values, self.ids, batch_size=2), [[30, 20]])

    def test_absent_optional_reports_are_empty_and_never_disable_native_trigger(self):
        summary = builder.prepare(self.original, None, self.root/'missing_reports')
        self.assertFalse(summary['information_available']['new_reports'])
        reader = CapacityNetworkInputs(self.root/'missing_reports', self.original, allow_exploratory=True)
        batch = reader.batch(np.array([0]))
        self.assertFalse(reader.information_status['new_reports']['enabled'])
        self.assertEqual(batch['capacity_inputs']['reports']['weights'].shape[1], 0)
        torch.testing.assert_close(batch['incident']['distances'], self.original.batch(np.array([0]))['incident']['distances'])

    def test_extractor_whitelists_deduplicates_and_excludes_future(self):
        raw = self.root/'raw.tsv'
        rows = [dict(incident_id=sid, dt=time, Fwy='4', Freeway_direction='W', **{'Abs PM': 1.,
                     'Latitude': 37.005, 'Longitude': -122., 'duration': 'FUTURE_FORBIDDEN', 'DESCRIPTION': 'FORBIDDEN'})
                for sid, time in [('same', '01/03/2023 08:03:00'), ('same', '01/03/2023 08:03:00'),
                                  ('conflict', '01/03/2023 08:03:00'), ('conflict', '01/03/2023 08:04:00'),
                                  ('future', '01/03/2023 08:10:01')]]
        with raw.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
            w.writeheader(); w.writerows(rows)
        with patch.object(builder, 'RAW_SHA256', sha256(raw)):
            manifest = builder.extract_reports(raw, self.events, self.public, self.root/'excerpt')
        selected = read_rows(self.root/'excerpt/train_locations.tsv', '\t')
        self.assertEqual([r['incident_id'] for r in selected], ['same'])
        self.assertEqual(tuple(selected[0]), REPORT_COLUMNS)
        self.assertEqual(manifest['conflicting_entity_ids'], ['conflict'])


if __name__ == '__main__':
    unittest.main()
