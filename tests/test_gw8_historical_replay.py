"""Server-only regressions for certified GW7-to-GW6 historical replay.

These temporary JSON/NPZ fixtures exercise provenance and immutable cache
equality. They do not train a model or claim forecasting evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.graph_interaction_gw7 import evaluate as gw7_eval
from experiments.graph_memory_gw6 import evaluate as gw6_eval
from experiments.lag_transport_gw8 import evaluate


class GW8HistoricalReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture_count = 0

    def write_layer(self, directory, arrays, metadata):
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory/'paired_predictions.npz'
        np.savez_compressed(archive, **arrays)
        metadata['archive_sha256'] = evaluate.metrics.sha256(archive)
        path = directory/'metadata.json'
        path.write_text(json.dumps(metadata, sort_keys=True), encoding='utf-8')
        return dict(directory=str(directory.resolve()),
            metadata_sha256=evaluate.metrics.sha256(path),
            archive_sha256=metadata['archive_sha256'])

    def seal(self, fixture):
        checkpoint = fixture['checkpoint']
        reference = fixture['reference']
        ancestor = reference['gw7_source_reference']
        ancestor['archives'][checkpoint] = self.write_layer(
            fixture['gw6_directory'], fixture['gw6_arrays'], fixture['gw6_metadata'])
        fixture['gw7_metadata']['source_reference'] = deepcopy(ancestor)
        reference['archives'][checkpoint] = self.write_layer(
            fixture['gw7_directory'], fixture['gw7_arrays'], fixture['gw7_metadata'])

    def fixture(self, checkpoint='best'):
        self.fixture_count += 1
        directory = self.root/str(self.fixture_count)
        truth = np.arange(72, dtype=np.float32).reshape(2, 12, 3)+1
        values = dict(y=truth, valid=np.ones_like(truth, dtype=bool),
            sample_ids=np.array([3909, 3910], dtype=np.int64),
            node_ids=np.array([101, 102, 103], dtype=np.int64),
            road_labels=np.array(['road-a', 'road-b', 'road-a'], dtype='<U40'),
            direct_support=np.array([[True, False, True], [False, True, False]], dtype=bool))
        for index, name in enumerate(evaluate.HISTORICAL, 1):
            values['pred__'+name] = truth+np.float32(index)
        gw6_arrays = dict(deepcopy(values), schema=np.array(gw6_eval.SCHEMA),
            axis_order=np.array('sample,horizon,node'))
        gw7_arrays = dict(deepcopy(values), schema=np.array(gw7_eval.SCHEMA),
            axis_order=np.array('sample,horizon,node'))
        roles = {name:dict(selection=checkpoint, epoch=0, policy=name,
            checkpoint_sha256=str(index)*64, model_tensor_sha256=str(index+3)*64,
            historical=True, reused=True)
            for index, name in enumerate(evaluate.HISTORICAL, 1)}
        records = {name:dict(actual=dict(mae=float(index), rmse=float(index),
                mape_fraction=.1*index),
            expected=dict(mae=float(index), rmse=float(index), mape_fraction=.1*index),
            passed=True, atol=1e-5, rtol=1e-6, upstream_record_marker=name)
            for index, name in enumerate(evaluate.HISTORICAL, 1)}
        ancestor = dict(source_identity_sha256='certified-gw6-run',
            gw6_source_reference=dict(source_identity_sha256='certified-gw5-run'), archives={})
        reference = dict(source_identity_sha256='certified-gw7-run',
            gw7_source_reference=ancestor, archives={})
        gw6_metadata = dict(schema=gw6_eval.PRODUCER_SCHEMA, status='GW6_PREDICTION_PASS',
            run_identity_sha256=ancestor['source_identity_sha256'],
            source_reference=deepcopy(ancestor['gw6_source_reference']),
            checkpoint_selection=checkpoint, checkpoint_roles=deepcopy(roles),
            producer_source=gw6_eval.producer_source(), validation_replay=records,
            engineering_only=False, test_evaluated=False)
        gw7_metadata = dict(schema=gw7_eval.PRODUCER_SCHEMA, status='GW7_PREDICTION_PASS',
            run_identity_sha256=reference['source_identity_sha256'],
            checkpoint_selection=checkpoint,
            checkpoint_roles={name:dict(role, historical=True, reused=True)
                for name, role in roles.items()},
            validation_replay=dict(R_CONTINUE=dict(passed=True, new_arm_marker='preserve')),
            engineering_only=False, test_evaluated=False)
        fixture = dict(checkpoint=checkpoint, reference=reference,
            gw6_directory=directory/'gw6', gw7_directory=directory/'gw7',
            gw6_arrays=gw6_arrays, gw7_arrays=gw7_arrays,
            gw6_metadata=gw6_metadata, gw7_metadata=gw7_metadata)
        self.seal(fixture)
        return fixture

    def test_missing_gw7_replay_inherits_real_gw6_records_for_best_and_last(self):
        for checkpoint in ('best', 'last'):
            with self.subTest(checkpoint=checkpoint):
                fixture = self.fixture(checkpoint)
                reference = fixture['reference']
                values, metadata = evaluate.load_historical(reference, checkpoint)
                provenance = metadata['historical_replay_provenance']
                certified = reference['gw7_source_reference']['archives'][checkpoint]
                self.assertEqual(provenance, dict(certified,
                    checkpoint_selection=checkpoint,
                    source_run_identity_sha256='certified-gw6-run'))
                self.assertEqual(metadata['validation_replay']['R_CONTINUE'],
                    fixture['gw7_metadata']['validation_replay']['R_CONTINUE'])
                for name in evaluate.HISTORICAL:
                    record = metadata['validation_replay'][name]
                    original = fixture['gw6_metadata']['validation_replay'][name]
                    for key in ('actual', 'expected', 'atol', 'rtol', 'upstream_record_marker'):
                        self.assertEqual(record[key], original[key])
                    self.assertIs(record['passed'], True)
                    self.assertEqual(record['mode'], 'certified_historical_cache_no_new_forward')
                    self.assertIs(record['equal_new_training_budget'], False)
                    self.assertEqual(record['certification_source'], provenance)
                    np.testing.assert_array_equal(values['pred__'+name],
                        fixture['gw7_arrays']['pred__'+name])

    def test_certification_does_not_rewrite_sources_or_mutate_supplied_values(self):
        fixture = self.fixture()
        paths = [directory/name for directory in
            (fixture['gw6_directory'], fixture['gw7_directory'])
            for name in ('metadata.json', 'paired_predictions.npz')]
        digests = {path:evaluate.metrics.sha256(path) for path in paths}
        metadata = deepcopy(fixture['gw7_metadata'])
        original_metadata = deepcopy(metadata)
        values = {key:deepcopy(fixture['gw7_arrays'][key]) for key in
            (*evaluate.AXIS_KEYS, *('pred__'+name for name in evaluate.HISTORICAL))}
        original_values = deepcopy(values)
        replays, _ = evaluate.certified_historical_replay(
            fixture['reference'], 'best', values, metadata)
        self.assertEqual(metadata, original_metadata)
        for key in values:
            self.assertEqual(values[key].dtype, original_values[key].dtype)
            np.testing.assert_array_equal(values[key], original_values[key])
        replays['ORIGINAL']['passed'] = False
        self.assertIs(fixture['gw6_metadata']['validation_replay']['ORIGINAL']['passed'], True)
        evaluate.load_historical(fixture['reference'], 'best')
        self.assertEqual(digests, {path:evaluate.metrics.sha256(path) for path in paths})

    def test_missing_false_or_nonboolean_ancestor_replay_and_conflicting_gw7_fail_closed(self):
        for mutation in ('missing', 'false', 'integer', 'conflicting-gw7'):
            with self.subTest(mutation=mutation):
                fixture = self.fixture()
                records = fixture['gw6_metadata']['validation_replay']
                if mutation == 'missing':
                    del records['ORIGINAL']
                elif mutation == 'false':
                    records['ORIGINAL']['passed'] = False
                elif mutation == 'integer':
                    records['ORIGINAL']['passed'] = 1
                else:
                    fixture['gw7_metadata']['validation_replay']['ORIGINAL'] = dict(passed=False)
                self.seal(fixture)
                with self.assertRaisesRegex(ValueError, 'replay'):
                    evaluate.load_historical(fixture['reference'], 'best')

    def test_historical_prediction_value_dtype_or_shape_mismatch_is_rejected(self):
        for mutation in ('value', 'dtype', 'shape'):
            with self.subTest(mutation=mutation):
                fixture = self.fixture()
                arrays = fixture['gw6_arrays']
                key = 'pred__ORIGINAL'
                if mutation == 'value':
                    arrays[key][0, 0, 0] += np.float32(1)
                elif mutation == 'dtype':
                    arrays[key] = arrays[key].astype(np.float64)
                else:
                    arrays[key] = arrays[key][:1]
                self.seal(fixture)
                with self.assertRaisesRegex(ValueError, 'predictions differ: ORIGINAL'):
                    evaluate.load_historical(fixture['reference'], 'best')

    def test_shared_axis_value_dtype_or_shape_mismatch_is_rejected(self):
        for key, mutation in (('y', 'dtype'), ('node_ids', 'value'),
                ('sample_ids', 'shape'), ('valid', 'value'),
                ('road_labels', 'value'), ('direct_support', 'value')):
            with self.subTest(key=key, mutation=mutation):
                fixture = self.fixture()
                arrays = fixture['gw6_arrays']
                if mutation == 'dtype':
                    arrays[key] = arrays[key].astype(np.float64)
                elif mutation == 'shape':
                    arrays[key] = arrays[key][:1]
                elif key == 'road_labels':
                    arrays[key][0] = 'different-road'
                elif arrays[key].dtype.kind == 'b':
                    arrays[key].flat[0] = not bool(arrays[key].flat[0])
                else:
                    arrays[key].flat[0] += 1
                self.seal(fixture)
                with self.assertRaisesRegex(ValueError, key):
                    evaluate.load_historical(fixture['reference'], 'best')

    def test_both_generations_reject_changed_metadata_and_archive_hashes(self):
        for generation in ('gw6', 'gw7'):
            for name in ('metadata.json', 'paired_predictions.npz'):
                with self.subTest(generation=generation, name=name):
                    fixture = self.fixture()
                    path = fixture[generation+'_directory']/name
                    with path.open('ab') as stream:
                        stream.write(b'\nchanged-after-certification')
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        evaluate.load_historical(fixture['reference'], 'best')

    def test_roles_and_ancestor_certificate_fields_must_match(self):
        for mutation in ('role', 'selection', 'run-identity', 'schema', 'producer-source'):
            with self.subTest(mutation=mutation):
                fixture = self.fixture()
                metadata = fixture['gw6_metadata']
                if mutation == 'role':
                    fixture['gw7_metadata']['checkpoint_roles']['ORIGINAL']['epoch'] = 99
                elif mutation == 'selection':
                    metadata['checkpoint_selection'] = 'last'
                elif mutation == 'run-identity':
                    metadata['run_identity_sha256'] = 'another-run'
                elif mutation == 'schema':
                    metadata['schema'] = 'unrecognized-producer'
                else:
                    metadata['producer_source'] = {}
                self.seal(fixture)
                with self.assertRaisesRegex(ValueError, 'role differs|certificate differs'):
                    evaluate.load_historical(fixture['reference'], 'best')


if __name__ == '__main__':
    unittest.main()
