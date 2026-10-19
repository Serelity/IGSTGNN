"""Server-only bounded runner integration tests; authenticated I/O is mocked."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np

from experiments.trajectory_diagnostic_gw13 import protocol
from experiments.trajectory_diagnostic_gw13 import run as runner


def small_data(samples=4, nodes=2):
    y = np.broadcast_to(np.where(np.arange(nodes) % 2, 400., 100.)[None, None, :],
        (samples, 12, nodes)).copy()
    valid = np.ones(y.shape, dtype=np.bool_)
    return dict(pred=y+10., y=y, valid=valid,
        sample_ids=np.arange(samples, dtype=np.int64),
        features=dict(last_flow=np.full((samples, nodes), 100., dtype=np.float32),
            mean_abs_change=np.ones((samples, nodes), dtype=np.float32),
            neighbor_change_disagreement=np.zeros((samples, nodes), dtype=np.float32),
            signed_direction=np.zeros((samples, nodes), dtype=np.float32),
            shape=(np.arange(samples*nodes).reshape(samples, nodes) % 4).astype(np.int8),
            turn_position=np.resize(np.array([-1, 2, 5, 7], dtype=np.int8),
                samples*nodes).reshape(samples, nodes)))


def audit_score_fixture(data):
    count = int(data['valid'].sum())
    records = {arm: dict(valid_count=count, mae=10., rmse=15.,
        mape_fraction=.2, signed_bias=-.1) for arm in protocol.ARMS}
    # An adequately supported but unsuccessful candidate must stop immediately.
    records[protocol.CANDIDATE]['mae'] = 10.1
    return dict(records=records,
        groups={arm: {group: dict(valid_count=1, mae=10., signed_bias=-.1)
            for group in protocol.GROUPS} for arm in protocol.ARMS},
        strata=[], residual_associations=[], paired_candidate_vs_OWN4=[])


class TrajectoryRunTests(unittest.TestCase):
    def test_slice_preserves_original_author_ids_features_and_native_mask(self):
        data = small_data()
        data['sample_ids'] += 1800
        data['valid'][1, 2, 0] = False
        sliced = runner.slice_data(data, 1, 3)
        self.assertEqual(set(sliced), {'pred', 'y', 'valid', 'sample_ids', 'features'})
        np.testing.assert_array_equal(sliced['sample_ids'], [1801, 1802])
        self.assertEqual(sliced['pred'].shape, (2, 12, 2))
        self.assertFalse(sliced['valid'][0, 2, 0])
        self.assertEqual(set(sliced['features']), set(data['features']))
        for name in data['features']:
            np.testing.assert_array_equal(sliced['features'][name], data['features'][name][1:3])
        self.assertTrue(np.shares_memory(sliced['valid'], data['valid']))

    def test_score_uses_shared_masks_and_reports_descriptive_and_paired_rows(self):
        data = small_data()
        data['valid'][0, 0, 0] = False
        fitted = object()

        def predictions(selected_model, selected_data):
            self.assertIs(selected_model, fitted)
            self.assertIs(selected_data, data)
            for index, arm in enumerate(protocol.ARMS):
                yield arm, data['pred']-index

        with patch.object(runner.model, 'predictions', side_effect=predictions):
            result = runner.score(fitted, data)
        self.assertEqual(tuple(result['records']), protocol.ARMS)
        for index, arm in enumerate(protocol.ARMS):
            self.assertEqual(result['records'][arm]['valid_count'], 95)
            self.assertAlmostEqual(result['records'][arm]['mae'], 10.-index)
            self.assertEqual(result['groups'][arm]['high_value_ge314']['valid_count'], 48)
            self.assertEqual(result['groups'][arm]['flow_50_200']['valid_count'], 47)
            self.assertEqual(result['groups'][arm]['long_horizon_7_12']['valid_count'], 48)
        self.assertEqual(len(result['strata']), 8*len(protocol.ARMS))
        self.assertEqual(len(result['residual_associations']), 8)
        for row in result['residual_associations']:
            self.assertEqual(row['desired_correction_median'], -10.)
            self.assertEqual(row['desired_correction_positive_fraction'], 0.)
            self.assertEqual(row['interpretation'],
                'descriptive_association_not_incremental_predictive_evidence')
        self.assertEqual(len(result['paired_candidate_vs_OWN4']), 13)
        for row in result['paired_candidate_vs_OWN4']:
            self.assertAlmostEqual(row['mae_gain'], 4.)

    def _mock_run(self, directory, source_changed=False):
        root = Path(directory)
        paths = {name: root/name for name in ('probe', 'cache', 'data', 'output')}
        for name in ('probe', 'cache', 'data'): paths[name].mkdir()
        (paths['cache']/'cache.json').write_text('{}\n', encoding='utf-8')
        cached = dict(train=small_data(runner.TRAIN_COUNT, nodes=1), val=small_data(2, nodes=1))
        certificate = dict(source_identity_sha256='a'*64, data_identity={'unchanged': True})
        digest_records = {'authenticated_cache': 'b'*64}
        extra = {split: dict(shape=value['features']['shape'],
            signed_direction=value['features']['signed_direction'],
            turn_position=value['features']['turn_position']) for split,value in cached.items()}
        history_meta = dict(file_digests={'authenticated_history': 'c'*64})
        audit_model = {'audit_model': True}
        coverage = {arm: dict(active_fraction=.5, eligible_cells=10, observed_cells=20)
            for arm in protocol.COVERAGE_ARMS}

        def save_model(value, path):
            self.assertIs(value, audit_model)
            Path(path).write_text('{"audit_model":true}\n', encoding='utf-8')

        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, '_export_repo_from_cache', return_value=root))
            load_source = stack.enter_context(patch.object(runner.auth, 'load_authenticated_run',
                return_value=(object(), object(), certificate)))
            stack.enter_context(patch.object(runner.auth, 'require_output_outside_sources'))
            stack.enter_context(patch.object(runner.cache_reader, 'load_cache',
                return_value=(cached, {}, digest_records)))
            stack.enter_context(patch.object(runner.features, 'load_features',
                return_value=(extra, history_meta)))
            fit = stack.enter_context(patch.object(runner.model, 'fit_model', return_value=audit_model))
            stack.enter_context(patch.object(runner.model, 'save_model', side_effect=save_model))
            score = stack.enter_context(patch.object(runner, 'score',
                side_effect=lambda fitted, data: audit_score_fixture(data)))
            stack.enter_context(patch.object(runner.model, 'coverage', return_value=coverage))
            final_source_guard = stack.enter_context(patch.object(runner.auth, 'assert_unchanged',
                side_effect=ValueError('source changed') if source_changed else None))
            final_cache_guard = stack.enter_context(patch.object(runner.cache_reader, 'rehash'))
            validation_screen = stack.enter_context(patch.object(runner.protocol, 'validation_screen',
                side_effect=AssertionError('Failed audit must not invoke validation screening')))
            outputs = stack.enter_context(patch.object(runner, 'write_outputs'))
            args = (paths['probe'], paths['cache'], paths['data'], paths['output'])
            if source_changed:
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    runner.run(*args)
                result = None
            else:
                result = runner.run(*args)
            load_source.assert_called_once()
            fit.assert_called_once()
            score.assert_called_once()
            validation_screen.assert_not_called()
            final_source_guard.assert_called_once_with(certificate)
            # Every parent/cut is fitted from the audit prefix rather than the
            # full train data. The runner then scores only the later train tail.
            fitted_data = fit.call_args.args[0]
            scored_data = score.call_args.args[1]
            np.testing.assert_array_equal(fitted_data['sample_ids'], np.arange(runner.FIT_STOP))
            np.testing.assert_array_equal(scored_data['sample_ids'],
                np.arange(runner.SCORE_START, runner.TRAIN_COUNT))
            self.assertEqual(len(fitted_data['y']), runner.FIT_STOP)
            self.assertFalse((paths['output']/'model.json').exists())
            self.assertFalse((paths['output']/'fitted_before_validation.json').exists())
            if source_changed:
                final_cache_guard.assert_not_called()
                outputs.assert_not_called()
                failure = json.loads((paths['output']/'failure.json').read_text(encoding='utf-8'))
                self.assertEqual(failure['status'], 'GW13_DIAGNOSTIC_FAILED')
                self.assertEqual(failure['error'], 'source changed')
            else:
                final_cache_guard.assert_called_once_with(digest_records)
                self.assertIn('authenticated_history', digest_records)
                outputs.assert_called_once_with(paths['output'].resolve(), result)
                self.assertTrue((paths['output']/'audit_model.json').is_file())
            return result

    def test_failed_audit_skips_full_fit_and_validation_but_checks_source_and_cache(self):
        with TemporaryDirectory() as directory:
            result = self._mock_run(directory)
        self.assertFalse(result['audit_decision']['audit_passed'])
        self.assertIsNone(result['validation'])
        self.assertFalse(result['screening']['trajectory_information_screen_passed'])
        self.assertFalse(result['screening']['candidate_for_confirmation'])
        self.assertFalse(result['test_evaluated'])

    def test_changed_source_aborts_instead_of_publishing_completed_results(self):
        with TemporaryDirectory() as directory:
            self._mock_run(directory, source_changed=True)


if __name__ == '__main__':
    unittest.main()
