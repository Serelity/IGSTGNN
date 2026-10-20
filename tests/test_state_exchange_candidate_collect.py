"""Server-only candidate collection checks; no model or original cache needed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.final_test_gw16.state import payload_hash
from experiments.state_exchange_multidata.common import spec
from experiments.state_exchange_multidata_candidate import collect


def results_fixture():
    results = {}
    for name in collect.DATASETS:
        macro = dict(mae=9., rmse=11., mape_percent=13.)
        native = dict(horizon_macro=macro,
            per_horizon=[dict(lead=h, **macro) for h in range(1, 13)])
        results[name] = dict(dataset=name, split='test', samples=spec(name)['test_samples'],
            rows={collect.CASE:dict(author_metrics=native,
                common_mask_pooled=dict(mae=999., rmse=999., mape_percent=999.))},
            parameter_counts={collect.CASE:74000}, test_evaluated=True, engineering_only=False)
    return results


def contra_fixture(directory):
    native = results_fixture()['Contra_Costa']['rows'][collect.CASE]['author_metrics']
    common = dict(mae=8., rmse=12., mape_percent=14.)
    role = dict(selection='best', completed_training_epochs=100, full_protocol_reached=True,
                registered_parameters=74000, epoch=99)
    runtime = dict(threads=3)
    station_ids = list(range(496))
    payload = dict(candidate_checkpoint_role=role, candidate_weights_sha256='candidate-weights',
        runtime=runtime, metric_contract=collect.CONTRA_CONTRACT,
        source_files={'/unavailable/ORIGINAL_IGSTGNN/predictions.npz':'not-required'},
        test_protocol=dict(bounds_half_open=[4746, 5585], count=839, nodes=496,
            history=12, horizon=12, batch_size=48, split='author_order_70_15_15', station_ids=station_ids))
    frozen = dict(schema='final_test_gw16_freeze_v1', freeze_id=payload_hash(payload), payload=payload)
    result = dict(schema='final_test_gw16_paired_test_v1', status='GW16_FINAL_TEST_COMPLETE',
        freeze_id=frozen['freeze_id'], split='test', test_evaluated=True, samples=839,
        candidate_selection_used_test=False, candidate_checkpoint_role=role,
        metric_contract=collect.CONTRA_CONTRACT,
        rows={collect.CASE:dict(author_metrics=native, common_mask_pooled=common)})
    metadata = dict(runtime=runtime, metric_contract=collect.CONTRA_CONTRACT,
        author_metrics=native, common_mask_pooled=common,
        audit=dict(model_tensor_sha256='candidate-weights', optimizer_updates=0,
            forward_batches=18, future_target_passed_to_forward=False, rng_unchanged=True,
            registered_tensors_unchanged=True, raw_inputs_unchanged=True),
        data_identity=dict(hashes=spec('Contra_Costa')['hashes'], bounds_half_open=[4746, 5585],
            count=839, batch_size=48, station_ids=station_ids))
    arrays = dict(prediction=np.ones((1, 12, 1, 1), dtype=np.float32),
                  target=np.ones((1, 12, 1, 1), dtype=np.float32),
                  station_ids=np.arange(496, dtype=np.int64))
    (directory/collect.CASE).mkdir()
    for path, record in ((directory/'freeze.json', frozen), (directory/'comparison.json', result),
                         (directory/collect.CASE/'metadata.json', metadata)):
        path.write_text(json.dumps(record), encoding='utf-8')
    # cache_read is mocked to isolate collection; production cache_read itself
    # authenticates NPZ checksum, exact 839x12x496 axes, dtype and finite values.
    (directory/collect.CASE/'predictions.npz').write_bytes(b'candidate-cache')
    return frozen, result, arrays, metadata, native, common


class CandidateCollectionTests(unittest.TestCase):
    def test_primary_table_has_only_candidate_native_metrics_and_no_gain_column(self):
        rows = collect.comparison_rows(results_fixture())
        self.assertEqual([row['dataset'] for row in rows], list(collect.DATASETS))
        self.assertTrue(all(row['case'] == 'STATE_EXCHANGE' and row['mae'] == 9.
                            and row['rmse'] == 11. for row in rows))
        self.assertFalse(any('gain' in key for row in rows for key in row))

    def test_all_three_datasets_are_required(self):
        results = results_fixture()
        results.pop('Orange')
        with self.assertRaisesRegex(ValueError, 'all three'):
            collect.comparison_rows(results)

    def test_validation_engineering_and_unexposed_results_are_rejected(self):
        for key, value in (('split', 'val'), ('engineering_only', True), ('test_evaluated', False)):
            results = results_fixture()
            results['Orange'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'mix'):
                collect.comparison_rows(results)

    def test_dataset_and_sample_counts_cannot_be_relabelled(self):
        for key, value in (('dataset', 'Orange'), ('samples', 4)):
            results = results_fixture()
            results['Alameda'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                collect.comparison_rows(results)

    def test_original_row_cannot_be_inserted(self):
        results = results_fixture()
        results['Orange']['rows']['ORIGINAL_IGSTGNN'] = copy.deepcopy(
            results['Orange']['rows'][collect.CASE])
        with self.assertRaisesRegex(ValueError, 'Candidate-only'):
            collect.comparison_rows(results)

    def test_native_macro_must_equal_the_twelve_finite_leads(self):
        for value in (10., float('nan'), float('inf'), -1.):
            results = results_fixture()
            results['Alameda']['rows'][collect.CASE]['author_metrics']['horizon_macro']['mae'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'macro'):
                collect.comparison_rows(results)

    def test_horizon_order_and_parameter_count_are_checked(self):
        results = results_fixture()
        results['Orange']['rows'][collect.CASE]['author_metrics']['per_horizon'].reverse()
        with self.assertRaisesRegex(ValueError, 'ordered horizons'):
            collect.comparison_rows(results)
        for value in (0, True, 74000.):
            results = results_fixture()
            results['Orange']['parameter_counts'][collect.CASE] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'parameter'):
                collect.comparison_rows(results)

    def test_report_contains_three_rows_and_all_twelve_leads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collect.render(root, results_fixture())
            self.assertEqual(len((root/'metrics.csv').read_text().splitlines()), 4)
            self.assertEqual(len((root/'per_horizon.csv').read_text().splitlines()), 37)
            text = (root/'report.md').read_text(encoding='utf-8')
            for term in ('Alameda', 'Contra_Costa', 'Orange', '74000', '100轮', '单seed'):
                self.assertIn(term, text)

    def test_contra_only_reads_candidate_cache_and_never_requires_original_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen, _, arrays, metadata, native, common = contra_fixture(root)
            before = {str(path):path.read_bytes() for path in root.rglob('*') if path.is_file()}
            with mock.patch.object(collect, 'cache_read', return_value=(arrays, metadata)) as reader, \
                    mock.patch.object(collect, 'native_summary', return_value=native) as metric, \
                    mock.patch.object(collect, 'numpy_metrics', return_value=common), \
                    mock.patch.object(collect.torch, 'get_num_threads', return_value=7), \
                    mock.patch.object(collect.torch, 'set_num_threads') as threads:
                result = collect.load_contra(root)
            reader.assert_called_once_with(root/collect.CASE, frozen['freeze_id'], collect.CASE)
            metric.assert_called_once_with(arrays['prediction'], arrays['target'])
            self.assertEqual(threads.call_args_list, [mock.call(3), mock.call(7)])
            self.assertEqual(set(result['rows']), {collect.CASE})
            self.assertEqual(result['new_model_forward_calls'], 0)
            self.assertFalse(result['original_prediction_cache_read'])
            self.assertFalse(result['historical_source_artifacts_reopened'])
            self.assertEqual(before, {str(path):path.read_bytes() for path in root.rglob('*') if path.is_file()})

    def test_contra_future_target_and_weight_audits_are_enforced(self):
        for field, value in (('future_target_passed_to_forward', True),
                             ('model_tensor_sha256', 'different'), ('optimizer_updates', 1)):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, _, arrays, metadata, _, _ = contra_fixture(root)
                metadata['audit'][field] = value
                with mock.patch.object(collect, 'cache_read', return_value=(arrays, metadata)), \
                        self.subTest(field=field), self.assertRaisesRegex(ValueError, 'inference'):
                    collect.load_contra(root)

    def test_contra_metric_mismatch_fails_and_restores_thread_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, arrays, metadata, native, common = contra_fixture(root)
            changed = copy.deepcopy(native)
            changed['horizon_macro']['mae'] += .1
            with mock.patch.object(collect, 'cache_read', return_value=(arrays, metadata)), \
                    mock.patch.object(collect, 'native_summary', return_value=changed), \
                    mock.patch.object(collect, 'numpy_metrics', return_value=common), \
                    mock.patch.object(collect.torch, 'get_num_threads', return_value=5), \
                    mock.patch.object(collect.torch, 'set_num_threads') as threads, \
                    self.assertRaisesRegex(ValueError, 'reproduce exactly'):
                collect.load_contra(root)
            self.assertEqual(threads.call_args_list, [mock.call(3), mock.call(5)])

    def test_contra_metric_exception_also_restores_threads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, arrays, metadata, _, _ = contra_fixture(root)
            with mock.patch.object(collect, 'cache_read', return_value=(arrays, metadata)), \
                    mock.patch.object(collect, 'native_summary', side_effect=RuntimeError('metrics')), \
                    mock.patch.object(collect.torch, 'get_num_threads', return_value=6), \
                    mock.patch.object(collect.torch, 'set_num_threads') as threads, \
                    self.assertRaisesRegex(RuntimeError, 'metrics'):
                collect.load_contra(root)
            self.assertEqual(threads.call_args_list, [mock.call(3), mock.call(6)])

    def test_contra_existing_freeze_tampering_is_rejected_before_cache_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen, _, _, _, _, _ = contra_fixture(root)
            frozen['payload']['candidate_checkpoint_role']['epoch'] = 98
            (root/'freeze.json').write_text(json.dumps(frozen), encoding='utf-8')
            with mock.patch.object(collect, 'cache_read') as reader, \
                    self.assertRaisesRegex(ValueError, 'freeze differs'):
                collect.load_contra(root)
            reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
