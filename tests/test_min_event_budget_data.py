"""Published X/Y fidelity and explicit event missingness contracts."""
import copy
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.utils.igstgnn_published import PublishedData
from src.utils.igstgnn_published_events import (
    DESCRIPTION_UNKNOWN, TYPE_UNKNOWN, EventPublishedData, EventPublishedLoader,
    encode_event, verify_category_mappings,
)
from tests.test_b3_r_minimal_paper import fixture


def event_fixture(root):
    samples, protocol = fixture(root)
    for i, sample in enumerate(samples):
        sample['incident_features'] = {'Type': i % 6, 'Description': i % 29,
                                       'Incident Time': object(), 'Holiday': object()}
        sample['incident_distances'] = np.array([[0, .5, 1], [0, 0, 0], [0, .2, 0], [0, 0, 0]], dtype=np.float64)
        sample['durations'] = object()
    np.save(root/'incident_all.npy', np.array(samples, dtype=object))
    for name, count in [('type_mapping.json', 6), ('desc_mapping.json', 29)]:
        (root/name).write_text(json.dumps({f'category_{i}': i for i in range(count)}), encoding='utf-8')
    return samples, protocol


class PublishedEventDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger('min_event_data_test')
        cls.logger.addHandler(logging.NullHandler())

    def test_real_class_zero_is_distinct_from_missing_and_ignored_fields_are_not_read(self):
        actual = encode_event(dict(incident_features={'Type': 0, 'Description': 0,
                                                     'Incident Time': object(), 'Holiday': object()},
                                   durations=object(), incident_position=object()), 4)
        self.assertEqual((actual['type'], actual['description']), (0, 0))
        self.assertTrue(actual['type_valid'] and actual['description_valid'] and actual['report_present'])
        unknown = encode_event(dict(incident_features={}), 4)
        self.assertEqual((unknown['type'], unknown['description']), (TYPE_UNKNOWN, DESCRIPTION_UNKNOWN))
        self.assertFalse(unknown['type_valid'] or unknown['description_valid'])
        self.assertTrue(unknown['report_present'])

    def test_missing_zero_relation_and_report_absence_are_separate(self):
        missing = encode_event({}, 4)
        self.assertFalse(missing['report_present'] or missing['relation_available'])
        np.testing.assert_array_equal(missing['relation'], np.zeros((4, 3), np.float32))
        location = encode_event({'incident_distances': np.zeros((4, 3))}, 4)
        self.assertTrue(location['report_present'] and location['relation_available'])
        self.assertFalse(location['type_valid'])
        explicit = encode_event({'incident_features': {'Type': 0},
                                 'incident_distances': np.ones((4, 3)), 'report_present': False}, 4)
        self.assertFalse(explicit['report_present'])
        self.assertTrue(explicit['relation_available'] and explicit['type_valid'])

    def test_invalid_given_codes_report_flags_and_relations_are_rejected(self):
        for key, values in [('Type', [-1, 6, .5, np.nan, np.inf, True, '0', None]),
                            ('Description', [-1, 29, .5, np.nan, True, '0', None])]:
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    encode_event({'incident_features': {key: value}}, 4)
        for features in (None, [], np.zeros(4)):
            with self.assertRaises(ValueError):
                encode_event({'incident_features': features}, 4)
        for flag in (0, 1, None, 'false'):
            with self.assertRaisesRegex(ValueError, 'Boolean'):
                encode_event({'report_present': flag}, 4)
        for relation in (np.ones((3, 3)), np.ones((4, 2)), np.full((4, 3), np.nan),
                         np.full((4, 3), np.inf), np.full((4, 3), 1e100),
                         np.ones((4, 3), dtype=complex), np.full((4, 3), '0')):
            with self.assertRaises(ValueError):
                encode_event({'incident_distances': relation}, 4)

    def test_float32_underflow_preserves_original_model_support_definition(self):
        relation = np.zeros((4, 3), np.float64)
        relation[0, 0], relation[1, 1] = 1e-100, .25
        event = encode_event({'incident_distances': relation}, 4)
        self.assertEqual(event['relation'].dtype, np.float32)
        self.assertEqual((np.abs(event['relation']).sum(-1) > 0).tolist(), [False, True, False, False])
        self.assertTrue(event['report_present'] and event['relation_available'])
        self.assertEqual(relation[0, 0], 1e-100)

    def test_xy_scaler_station_axis_order_and_batches_match_original_exactly(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            samples, protocol = event_fixture(root)
            original = PublishedData(root, protocol, self.logger)
            actual = EventPublishedData(root, protocol, self.logger)
            self.assertEqual(actual.station_ids, original.station_ids)
            self.assertEqual(float(actual.scaler.mean), float(original.scaler.mean))
            self.assertEqual(float(actual.scaler.std), float(original.scaler.std))
            order = np.arange(14)[::-1]
            for data in (actual, original):
                data.loaders['train_loader'].set_order(order)
            for key in actual.loaders:
                for batch, xy in zip(actual.loaders[key].get_iterator(), original.loaders[key].get_iterator()):
                    np.testing.assert_array_equal(batch['x_data'], xy[0])
                    np.testing.assert_array_equal(batch['y_data'], xy[1])
                    events = batch['event_data']
                    self.assertEqual(events['type'].dtype, np.int64)
                    self.assertEqual(events['description'].dtype, np.int64)
                    self.assertEqual(events['type_valid'].dtype, np.bool_)
                    self.assertEqual(events['relation'].shape, (len(xy[0]), 4, 3))
            first = next(actual.loaders['train_loader'].get_iterator())
            self.assertEqual(first['event_data']['type'][0], samples[13]['incident_features']['Type'])
            self.assertEqual(actual.audit['clock_offset_slots'], 0)
            self.assertTrue(actual.audit['test_payload_deserialized'])
            self.assertFalse(actual.audit['test_evaluated'])
            self.assertFalse(actual.audit['no_natural_no_event_observations_verified'])
            json.dumps(actual.audit)

    def test_train_val_only_validation_single_deserialization_and_explicit_test_access(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            samples, protocol = event_fixture(root)
            # Invalid test targets and metadata must not be inspected at init.
            for sample in samples[17:]:
                sample['y_data'] = object()
                sample['incident_features'] = {'Type': -1}
            with patch('src.utils.igstgnn_published.load_samples', return_value=np.array(samples, dtype=object)) as load:
                data = EventPublishedData(root, protocol, self.logger)
            load.assert_called_once()
            self.assertEqual(data.loaders['train_loader'].size, 14)
            self.assertEqual(data.loaders['val_loader'].size, 3)
            with self.assertRaisesRegex(AttributeError, 'shape'):
                data.test_loader(2, self.logger)
            for sample in samples[17:]:
                sample['y_data'] = np.zeros((12, 4, 3), np.float32)
            with self.assertRaisesRegex(ValueError, 'Type'):
                data.test_loader(2, self.logger)
            check = EventPublishedData(root, protocol, self.logger, check=True)
            self.assertEqual(check.loaders['train_loader'].size, 4)
            self.assertEqual(check.loaders['val_loader'].size, 3)
            self.assertEqual(check.loaders['train_loader'].bs, 2)
            test = next(check.test_loader(2, self.logger).get_iterator())
            np.testing.assert_array_equal(test['x_data'][0], samples[17]['x_data'])

    def test_all_used_metadata_rejected_before_training_and_vocabulary_validated(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            samples, protocol = event_fixture(root)
            samples[16]['incident_features']['Description'] = 29
            np.save(root/'incident_all.npy', np.array(samples, dtype=object))
            with self.assertRaisesRegex(ValueError, 'Description'):
                EventPublishedData(root, protocol, self.logger)
            (root/'type_mapping.json').write_text(json.dumps({'bad': 0}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'vocabulary size'):
                verify_category_mappings(root)
            (root/'type_mapping.json').write_text(json.dumps({str(i): 0 for i in range(6)}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unique contiguous'):
                verify_category_mappings(root)

    def test_event_loader_shuffle_keeps_metadata_with_traffic(self):
        samples = []
        for i in range(4):
            samples.append(dict(x_data=np.full((12, 4, 3), i, np.float32),
                                y_data=np.full((12, 4, 3), i+10, np.float32),
                                incident_features={'Type': i, 'Description': i}))
        before = copy.deepcopy(samples)
        loader = EventPublishedLoader(samples, 3, self.logger)
        loader.set_order(np.array([3, 0, 2, 1]))
        for batch in loader.get_iterator():
            np.testing.assert_array_equal(batch['x_data'][:, 0, 0, 0], batch['event_data']['type'])
            np.testing.assert_array_equal(batch['y_data'][:, 0, 0, 0]-10, batch['event_data']['description'])
        for a, b in zip(samples, before):
            np.testing.assert_array_equal(a['x_data'], b['x_data'])
            self.assertEqual(a['incident_features'], b['incident_features'])


if __name__ == '__main__':
    unittest.main()
