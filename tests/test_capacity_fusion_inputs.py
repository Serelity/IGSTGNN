import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_incident_corridor as corridor_fixture
from src.utils.capacity_fusion_inputs import OriginalCapacityInputs
from src.utils.incident_corridor import read_json, write_json, sha256


class FusionInputTests(unittest.TestCase):
    def setUp(self):
        f = corridor_fixture.CorridorTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        self.scaler = dict(station_ids=f.ids.tolist(), fitted_sample_indices=[100, 101],
                           fit_scope='train_X_0:12_unique_station_nominal_slot_finite_nonnegative',
                           mean=20., std=4., node_fill_mean=[10., 11., 12., 13.])
        write_json(f.data / 'scaler.json', self.scaler)
        np.save(f.data / 'adjacency.npy', np.eye(4, dtype=np.float32))
        self.refresh()

    def refresh(self):
        f = self.f
        data = read_json(f.data / 'summary.json')
        data['files'].update({name: sha256(f.data / name) for name in ('scaler.json', 'adjacency.npy')})
        write_json(f.data / 'summary.json', data)
        context = read_json(f.data / 'context_manifest.json')
        context['outputs']['adjacency.npy'] = sha256(f.data / 'adjacency.npy')
        write_json(f.data / 'context_manifest.json', context)
        multi_scaler = read_json(f.multi / 'train_multichannel_scaler.json')
        multi_scaler['mean'] = [20., 2., 10.]
        write_json(f.multi / 'train_multichannel_scaler.json', multi_scaler)
        multi = read_json(f.multi / 'summary.json')
        multi['inputs']['data_summary_sha256'] = sha256(f.data / 'summary.json')
        multi['outputs']['train_multichannel_scaler.json']['sha256'] = sha256(f.multi / 'train_multichannel_scaler.json')
        write_json(f.multi / 'summary.json', multi)
        selection = read_json(f.selection)
        selection['data_summary_sha256'] = sha256(f.data / 'summary.json')
        selection['context_manifest_sha256'] = sha256(f.data / 'context_manifest.json')
        write_json(f.selection, selection)

    def adapter(self):
        f = self.f
        return OriginalCapacityInputs(f.data, f.multi, f.sensors, f.meta, f.selection)

    def test_original_axes_flow_clock_references_and_no_target_input(self):
        adapter = self.adapter()
        self.addCleanup(adapter.close)
        batch = adapter.batch(np.array([1, 0]))
        raw = self.f.history[[1, 0]]
        torch.testing.assert_close(batch['x'][..., 0], torch.tensor((raw[..., 0]-20)/4))
        self.assertEqual(float(batch['x'][0, 0, 0, 1]), np.float32(97/288))
        torch.testing.assert_close(batch['capacity_inputs']['references'], torch.tensor([[20., 2., 10.]]*4))
        self.assertEqual(set(batch['capacity_inputs']), {'history', 'valid', 'references', 'labels'})
        self.assertEqual(batch['capacity_inputs']['labels'].tolist(), list(range(-65, -5, 5)))
        self.assertFalse(bool(batch['capacity_inputs']['valid'][1, 0, 0, 1]))

    def test_never_opens_flow_gap_Y_validation_test_or_candidate_pack(self):
        opened = []
        actual = Path.open
        def guard(path, *args, **kwargs):
            opened.append(path.name)
            if path.name.startswith(('val_', 'test_')) or path.name.endswith('_flow.npy') or 'candidate' in path.name:
                raise AssertionError('Forbidden file: '+str(path))
            return actual(path, *args, **kwargs)
        with patch.object(Path, 'open', guard):
            adapter = self.adapter()
            try:
                adapter.batch(np.array([0]))
                ledger = adapter.qualification_ledger()
            finally:
                adapter.close()
        self.assertFalse(ledger['summary']['main_training_ready'])
        self.assertNotIn('val_context.npz', opened)

    def test_pending_qualification_never_fabricates_edges_or_coverage(self):
        adapter = self.adapter()
        self.addCleanup(adapter.close)
        ledger = adapter.qualification_ledger()
        self.assertEqual([s['station_id'] for s in ledger['stations']], self.f.ids.tolist())
        self.assertTrue(all(not s['directed_exchange_qualified'] for s in ledger['stations']))
        self.assertEqual(sum(r['stations'] for r in ledger['roads']), 4)
        graph = adapter.qualified_graph()
        self.assertEqual(graph.edges, 0)
        self.assertFalse(graph.operator_mask.any())
        self.assertEqual(graph.evidence_scope, 'candidate_unverified')
        self.assertTrue(all(not p['admitted_exchange_edge'] for p in ledger['candidate_pairs']))

    def test_rejects_tampered_normalization_and_invalid_sample_indices(self):
        adapter = self.adapter()
        self.addCleanup(adapter.close)
        for indices in (np.array([-1]), np.array([2]), np.array([.5]), np.array([], dtype=int)):
            with self.assertRaises(ValueError):
                adapter.batch(indices)
        bad = copy.deepcopy(self.scaler)
        bad['station_ids'].reverse()
        write_json(self.f.data / 'scaler.json', bad)
        self.refresh()
        with self.assertRaises(ValueError):
            self.adapter()


if __name__ == '__main__':
    unittest.main()
