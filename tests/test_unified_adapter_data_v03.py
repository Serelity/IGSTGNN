"""Missing-category, availability, isolation and original-parser regressions."""
from copy import deepcopy
import logging
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.chronological import unified_adapter_data_v03 as contract
from experiments.chronological.original_igstgnn_source import get_original_modules


class UnifiedAdapterDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = get_original_modules()
        cls.logger = logging.getLogger('unified-v03-data-test')

    @staticmethod
    def sample(**updates):
        row = dict(x_data=np.zeros((12, 4, 3), np.float32),
            y_data=np.ones((12, 4, 3), np.float32),
            incident_features={'Incident Time': 900., 'Description': 0, 'Type': 0, 'Holiday': 1},
            incident_position=0, incident_distances=np.eye(4, 3, dtype=np.float32))
        row.update(updates)
        return row

    def original_loader(self, samples):
        return self.modules.loader.IncidentDataLoader(samples, True, 2, self.logger, input_dim=3,
            sensor_info={'sensor_type': np.zeros(4, np.int64), 'speed_limit': np.ones(4, np.float32)*50})

    def data(self, samples):
        return SimpleNamespace(loaders={'train_loader': self.original_loader(samples),
            'val_loader': self.original_loader(deepcopy(samples)),
            'test_loader': SimpleNamespace(samples='MUST NOT INSPECT')})

    def batch(self, samples):
        data = self.data(samples)
        return next(contract.make_loaders(data, self.modules, self.logger)['train_loader'].get_iterator())

    def test_zero_is_legal_and_bad_categories_are_distinct_unknown(self):
        for count in (6, 29):
            self.assertEqual(contract.category(0, count), (0, 'valid'))
            self.assertEqual(contract.category(np.float32(0.), count), (0, 'valid'))
            for bad in (None, np.nan, np.inf, -1, count, 1.8, '0', False, 1j):
                with self.subTest(count=count, bad=bad):
                    self.assertEqual(contract.category(bad, count)[0], -1)

    def test_missing_does_not_become_zero_before_batch_encoding(self):
        sample = self.sample(incident_features={'Holiday': 1})
        batch = self.batch([sample])
        np.testing.assert_array_equal(batch['incident_features'], [[0, -1, -1, 1]])
        self.assertEqual(batch['category_valid'].tolist(), [[False, False, True, True]])
        self.assertEqual(sample['incident_features'], {'Holiday': 1})

    def test_original_loader_and_samples_are_not_mutated(self):
        data = self.data([self.sample(), self.sample()])
        before = next(data.loaders['train_loader'].get_iterator())
        old_loader = data.loaders['train_loader']
        original_order = [id(s) for s in old_loader.samples]
        candidates = contract.make_loaders(data, self.modules, self.logger)
        after = next(data.loaders['train_loader'].get_iterator())
        self.assertIs(data.loaders['train_loader'], old_loader)
        self.assertIsNot(candidates['train_loader'], old_loader)
        self.assertEqual(set(candidates), {'train_loader', 'val_loader'})
        for key in ('incident_features', 'incident_distances', 'x_data', 'y_data'):
            np.testing.assert_array_equal(before[key], after[key])
        candidates['train_loader'].samples.reverse()
        self.assertEqual([id(s) for s in old_loader.samples], original_order)
        self.assertEqual(float(before['incident_features'][0, 0]), 900.)

    def test_unavailable_nan_placeholder_is_replaced_before_math(self):
        row = contract.encode_report(self.sample(available=False,
            incident_distances=np.full((4, 3), np.nan), incident_features=np.nan,
            incident_position=np.nan), 4)
        self.assertFalse(row['available'])
        self.assertTrue(np.isfinite(row['distances']).all())
        self.assertEqual(float(row['distances'].sum()), 0.)
        self.assertEqual(row['position'], -1)
        self.assertEqual(row['incident'][1], -1)

    def test_relation_missing_and_zero_support_remain_different(self):
        missing = self.sample()
        del missing['incident_distances']
        no_relation = contract.encode_report(missing, 4)
        zero_relation = contract.encode_report(self.sample(incident_distances=np.zeros((4, 3))), 4)
        self.assertTrue(no_relation['available'])
        self.assertFalse(no_relation['relation_available'])
        self.assertTrue(zero_relation['relation_available'])
        np.testing.assert_array_equal(no_relation['distances'], zero_relation['distances'])

    def test_explicit_missing_relation_sanitizes_nan_without_claiming_report_absent(self):
        row = contract.encode_report(self.sample(relation_available=False,
            incident_distances=np.full((1,), np.nan)), 4)
        self.assertTrue(row['available'])
        self.assertFalse(row['relation_available'])
        self.assertEqual(row['distances'].shape, (4, 3))
        self.assertEqual(float(row['distances'].sum()), 0.)

    def test_available_relation_nonfinite_and_wrong_shape_are_rejected(self):
        for bad in (np.full((4, 3), np.nan), np.full((4, 3), np.inf), np.ones((4, 2))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                contract.encode_report(self.sample(incident_distances=bad), 4)
        missing = self.sample(relation_available=True)
        del missing['incident_distances']
        with self.assertRaisesRegex(ValueError, 'absent'):
            contract.encode_report(missing, 4)

    def test_availability_flag_conflicts_and_bad_values_fail(self):
        for bad in (np.nan, 1.5, 'True', None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                contract.encode_report(self.sample(available=bad), 4)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            contract.encode_report(self.sample(available=True, report_present=False), 4)

    def test_optional_flags_reach_device_and_absent_report_masks_content(self):
        batch = self.batch([self.sample(available=False), self.sample(available=True,
            relation_available=False)])
        _, _, report, sensors = contract.device_batch_v03(batch, 'cpu')
        self.assertEqual(report['available'].tolist(), [False, True])
        self.assertEqual(report['relation_available'].tolist(), [True, False])
        self.assertEqual(report['incident'][:, 0].tolist(), [0., 0.])
        self.assertEqual(report['incident'][0].tolist(), [0., -1., -1., 1.])
        self.assertEqual(report['position'].tolist(), [-1, 0])
        self.assertEqual(float(report['distances'].sum()), 0.)
        self.assertIn('sensor_type', sensors)
        self.assertTrue(torch.equal(sensors['speed_limit'], torch.ones((2, 4))*50))

    def test_H_R_RC_isolation_keeps_public_context_equal(self):
        batch = self.batch([self.sample()])
        converted = {mode: contract.device_batch_v03(batch, 'cpu', mode)
            for mode in ('H', 'R', 'RC')}
        for mode in ('H', 'R'):
            x, y, report, sensors = converted[mode]
            self.assertEqual(report['incident'][0, 1:3].tolist(), [-1., -1.])
            self.assertEqual(report['position'].tolist(), [-1])
            self.assertEqual(report['incident'][0, 3].item(), 1.)
            self.assertTrue(torch.equal(x, converted['RC'][0]))
            self.assertTrue(torch.equal(y, converted['RC'][1]))
            self.assertTrue(torch.equal(sensors['sensor_type'], converted['RC'][3]['sensor_type']))
        self.assertFalse(converted['H'][2]['available'].any())
        self.assertFalse(converted['H'][2]['relation_available'].any())
        self.assertEqual(float(converted['H'][2]['distances'].sum()), 0.)
        self.assertTrue(torch.equal(converted['R'][2]['distances'], converted['RC'][2]['distances']))
        self.assertEqual(converted['RC'][2]['incident'][0, 1:3].tolist(), [0., 0.])
        self.assertEqual(float(batch['incident_features'][0, 0]), 0.)
        self.assertEqual(float(batch['incident_distances'].sum()), 3.)

    def test_device_contract_does_not_resurrect_nonfinite_absent_relation(self):
        batch = self.batch([self.sample(available=False)])
        batch['incident_distances'].fill(np.nan)
        _, _, incident, _ = contract.device_batch_v03(batch, 'cpu')
        self.assertTrue(torch.isfinite(incident['distances']).all())
        batch['available'][0] = True
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            contract.device_batch_v03(batch, 'cpu')

    def test_audit_never_reads_targets_or_test_samples(self):
        class NoTargetAccess(dict):
            def __getitem__(self, key):
                if key == 'y_data':
                    raise AssertionError('audit read a target')
                return super().__getitem__(key)
        rows = [NoTargetAccess(self.sample()), NoTargetAccess(self.sample(
            available=False, incident_features={'Holiday': 0}, incident_distances=np.full((4, 3), np.nan)))]
        loader = SimpleNamespace(samples=rows, num_nodes=4)
        data = SimpleNamespace(loaders={'train_loader': loader, 'val_loader': loader,
            'test_loader': property(lambda _: (_ for _ in ()).throw(AssertionError('test read')))})
        audit = contract.audit_data(data)
        self.assertEqual(set(audit['splits']), {'train', 'val'})
        self.assertFalse(audit['test_targets_read'])
        self.assertEqual(audit['splits']['train']['counts']['rows'], 2)
        self.assertEqual(audit['splits']['train']['category_histograms']['Type'], {'0': 1, '-1': 1})
        self.assertFalse(audit['evidence_boundaries']['availability_cutoff_certified'])
        self.assertFalse(audit['evidence_boundaries']['event_independence_certified'])
        self.assertEqual(audit['splits']['train']['optional_identity_field_rows']['clearance_time'], 0)

    def test_audit_retains_declared_and_inferred_flag_sources(self):
        data = self.data([self.sample(), self.sample(report_present=False, relation_available=False)])
        audit = contract.audit_data(data)['splits']['train']
        self.assertEqual(audit['flag_sources']['available']['report_present'], 1)
        self.assertEqual(audit['flag_sources']['available'][
            'inferred_from_field_presence_not_cutoff_certified'], 1)
        self.assertEqual(audit['counts']['reports_available'], 1)


if __name__ == '__main__':
    unittest.main()
