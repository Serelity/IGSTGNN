"""Server-only release-axis, event-vocabulary and unchanged Contra checks."""
from copy import deepcopy
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.position_state_gw16.common import LOCKED_CONFIG as GW16_CONFIG, RELEASE
from src.utils.igstgnn_published_events import encode_event as contra_encode
from src.utils.igstgnn_published_events import EventPublishedLoader as ContraLoader
from experiments.state_exchange_multidata.common import (
    DATASETS, dataset_name, load_config, protected_output, spec, validate_config)
from experiments.state_exchange_multidata.data import (
    EventPublishedData, EventPublishedLoader, authenticate_release, encode_event,
    event_counts, validation_loader, verify_category_mappings)

LOGGER = logging.getLogger(__name__)


def sample(description=0, nodes=3):
    x = np.zeros((12, nodes, 3), dtype=np.float32)
    return dict(x_data=x, y_data=x.copy(), incident_features=dict(Type=0, Description=description),
                incident_distances=np.ones((nodes, 3), dtype=np.float32))


def mappings(directory, counts):
    for filename, key in (('type_mapping.json', 'type'), ('desc_mapping.json', 'description')):
        (directory/filename).write_text(json.dumps({str(i):i for i in range(counts[key])}), encoding='utf-8')


class MultidataReleaseTests(unittest.TestCase):
    def test_registry_counts_match_author_rounding_and_nodes(self):
        expected = {'Alameda':(521, 14687, 10280, 2203, 2204),
                    'Contra_Costa':(496, 5585, 3909, 837, 839),
                    'Orange':(990, 18700, 13090, 2805, 2805)}
        for name, values in expected.items():
            row = spec(name)
            self.assertEqual(tuple(row[k] for k in
                ('nodes', 'total_samples', 'train_samples', 'val_samples', 'test_samples')), values)
            self.assertEqual(len(row['hashes']), 6)
            self.assertTrue(all(len(value) == 64 for value in row['hashes'].values()))

    def test_contra_fingerprints_and_entire_config_are_unchanged(self):
        self.assertEqual(spec('Contra_Costa')['hashes'], RELEASE)
        self.assertEqual(load_config('Contra_Costa'), GW16_CONFIG)

    def test_other_dataset_config_changes_only_required_axes(self):
        for name in DATASETS:
            actual = load_config(name)
            expected = deepcopy(GW16_CONFIG)
            expected.update(dataset=name+'_author_release', nodes=spec(name)['nodes'])
            self.assertEqual(actual, expected)
            self.assertEqual(dataset_name(actual), name)

    def test_dataset_aliases_and_case_changes_rejected(self):
        for name in ('orange', 'Contra Costa', 'alameda', '', None, True):
            with self.subTest(name=name), self.assertRaises(ValueError):
                spec(name)

    def test_dataset_specific_tuning_and_bool_numeric_overrides_rejected(self):
        for name in DATASETS:
            for key, value in (('max_epochs', 200), ('hidden', 64), ('dropout', True)):
                config = load_config(name)
                config[key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_engineering_only_changes_budget_not_model(self):
        config = load_config('Orange')
        config.update(max_epochs=3, lr_milestones=[])
        validate_config(config, engineering=True)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_config_override_cannot_change_recipe(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'config.json'
            path.write_text(json.dumps({'hidden':64}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'locked'):
                load_config('Alameda', path)
            path.write_text(json.dumps({'hidden':32}), encoding='utf-8')
            self.assertEqual(load_config('Alameda', path)['hidden'], 32)

    def test_vocabulary_cardinalities_are_dataset_local(self):
        self.assertEqual([event_counts(spec(name)) for name in DATASETS],
            [dict(type=6, description=30), dict(type=6, description=29), dict(type=6, description=32)])

    def test_last_known_category_and_unknown_have_distinct_rows(self):
        for name in DATASETS:
            counts = event_counts(spec(name))
            known = encode_event(sample(counts['description']-1), 3, counts)
            missing = encode_event({}, 3, counts)
            self.assertEqual(known['description'], counts['description']-1)
            self.assertTrue(known['description_valid'])
            self.assertEqual(missing['description'], counts['description'])
            self.assertFalse(missing['description_valid'])
            self.assertEqual(missing['type'], counts['type'])

    def test_vocab_id_range_and_noninteger_rejected(self):
        for name in DATASETS:
            counts = event_counts(spec(name))
            for value in (-1, counts['description'], 1.5, True, float('nan')):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    encode_event(sample(value), 3, counts)

    def test_contra_event_encoding_is_bit_exact(self):
        for source in (sample(28), {}, dict(incident_features={}),
                       dict(sample(0), report_present=False)):
            old, new = contra_encode(source, 3), encode_event(source, 3, event_counts(spec('Contra_Costa')))
            self.assertEqual(set(old), set(new))
            for key in old:
                np.testing.assert_array_equal(old[key], new[key])

    def test_contra_loader_batch_and_permutation_are_identical(self):
        source = [sample(0), sample(28), sample(7)]
        old, new = ContraLoader(source, 2, LOGGER), EventPublishedLoader(
            source, 2, LOGGER, event_counts(spec('Contra_Costa')))
        old.set_order([2, 0, 1])
        new.set_order([2, 0, 1])
        for first, second in zip(old.get_iterator(), new.get_iterator()):
            for key in ('x_data', 'y_data'):
                np.testing.assert_array_equal(first[key], second[key])
            for key in first['event_data']:
                np.testing.assert_array_equal(first['event_data'][key], second['event_data'][key])
        self.assertEqual(old.event_audit(), new.event_audit())

    def test_vocab_mappings_require_exact_contiguous_release_axis(self):
        counts = dict(type=6, description=32)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            mappings(directory, counts)
            self.assertEqual(len(verify_category_mappings(directory, counts)['description']), 32)
            (directory/'desc_mapping.json').write_text(json.dumps({'a':0, 'b':0}), encoding='utf-8')
            with self.assertRaises(ValueError):
                verify_category_mappings(directory, counts)

    def test_event_loader_ignores_unapproved_time_and_position_fields(self):
        source = sample(31)
        changed = dict(source, incident_position=object(), durations=object())
        changed['incident_features'] = dict(source['incident_features'],
            **{'Incident Time':object(), 'Holiday':object()})
        expected, actual = encode_event(source, 3, dict(type=6, description=32)), encode_event(
            changed, 3, dict(type=6, description=32))
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])

    def test_relation_shape_nonfinite_and_overflow_rejected(self):
        for relation in (np.ones((2, 3)), np.full((3, 3), np.nan), np.full((3, 3), 1e300)):
            with self.assertRaises(ValueError):
                encode_event(dict(sample(), incident_distances=relation), 3, dict(type=6, description=32))

    def test_training_does_not_validate_or_batch_test_events(self):
        counts = dict(type=6, description=32)
        protocol = dict(total_samples=7, train_samples=4, val_samples=1, test_samples=2,
                        stations=3, batch_size=2)
        source = [sample(31) for _ in range(5)]
        source.extend([dict(sample(0), incident_features={'Description':999}), sample(0)])
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            mappings(directory, counts)
            (directory/'sensors.csv').write_text('station_id\n1\n2\n3\n', encoding='utf-8')
            np.savez(directory/'incident_stats.npz', normalized=True, mean=0., std=1.)
            with patch('src.utils.igstgnn_published.load_samples', return_value=source):
                data = EventPublishedData(directory, protocol, LOGGER, counts)
            self.assertFalse(data.audit['test_evaluated'])
            self.assertEqual(data.audit['event_validation_scope'], 'used_train_validation_samples_only')
            with self.assertRaises(ValueError):
                data.test_loader(2, LOGGER)

    def test_validation_and_test_offsets_follow_each_author_split(self):
        for name in DATASETS:
            published = SimpleNamespace(test_loader=Mock(return_value='test'))
            data = SimpleNamespace(identity={'dataset':name}, loaders={'val_loader':'val'}, published=published)
            row = spec(name)
            self.assertEqual(validation_loader(data, 'val', 48, LOGGER), ('val', row['train_samples']))
            published.test_loader.assert_not_called()
            self.assertEqual(validation_loader(data, 'test', 48, LOGGER),
                ('test', row['train_samples']+row['val_samples']))
            published.test_loader.assert_called_once_with(48, LOGGER)

    def test_wrong_release_fingerprint_rejected_before_deserialization(self):
        with patch('experiments.state_exchange_multidata.data.sha256', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                authenticate_release('/unused/Orange', 'Orange')

    def test_output_cannot_overlap_dataset_or_old_experiment(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            data = directory/'data'
            data.mkdir()
            with self.assertRaises(ValueError):
                protected_output(data/'run', data)
            run = directory/'old_run'
            run.mkdir()
            (run/'identity.json').write_text(json.dumps({'schema':'position_state_gw16_training_v1'}))
            with self.assertRaises(ValueError):
                protected_output(run/'child', data, allow_run_children=True)


if __name__ == '__main__':
    unittest.main()
