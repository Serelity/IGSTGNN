"""Server-run resume, reference, metric and identical-model cache contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import train_original_receiver_match as train
from experiments.chronological import evaluate_original_receiver_match as evaluation
from experiments.chronological import receiver_match_reference as reference
from tests.test_original_isp_runner import Engine, identity_for, comparable_history
from tests.test_original_graph_window_evaluation import fixture as gw1_fixture


def fixture():
    old = gw1_fixture()
    archive = {key: value for key, value in old.items() if not key.startswith('pred__')}
    archive['schema'] = np.array(evaluation.SCHEMA)
    for i, case in enumerate(evaluation.CASES):
        archive['pred__'+case] = archive['y']+i+1.
    return archive


class ReceiverMatchWorkflowTests(unittest.TestCase):
    def identity(self, engine):
        identity = identity_for(engine)
        identity.update(schema=train.SCHEMA, design=train.DESIGN, arms=list(train.ARMS))
        initial = identity['initializations']['ORIGINAL']
        identity['initializations'] = {arm: initial for arm in ('ORIGINAL', *train.ARMS)}
        return identity

    def test_parser_requires_old_reference_for_full_run_and_total_budget(self):
        args = train.parse_args(['--run-dir', 'unused', '--gw2-run', 'existing'])
        self.assertEqual(args.epochs, 5)
        self.assertEqual(train.parse_args(['--run-dir', 'unused', '--check']).epochs, 5)
        for extra in ([], ['--epochs', '0'], ['--check', '--gw2-run', 'old'],
                      ['--check', '--only', 'MESSAGE_SELF', 'MESSAGE_SELF']):
            with self.assertRaises(SystemExit):
                train.parse_args(['--run-dir', 'unused', *extra])

    def test_exact_epoch_resume_without_optimizer_rng_order_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            full, split = Path(temp)/'full', Path(temp)/'split'
            full.mkdir(); split.mkdir()
            identity = self.identity(Engine())
            with patch('builtins.print'):
                a = train.base.run_epochs(Engine(), full, identity, 'RECEIVER_MATCH', 4)
                train.base.run_epochs(Engine(), split, identity, 'RECEIVER_MATCH', 2)
                saved = train.base.load_checkpoint(split/'last.pt')
                train.validate_saved(saved, identity, 'RECEIVER_MATCH', split, 2)
                b = train.base.run_epochs(Engine(19), split, identity, 'RECEIVER_MATCH', 4, saved)
            self.assertEqual(comparable_history(a), comparable_history(b))
            for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
                self.assertEqual(train.base.tensor_tree_hash(a[key]), train.base.tensor_tree_hash(b[key]))
            broken = deepcopy(saved); broken['global_updates'] = 999
            with self.assertRaises(ValueError):
                train.validate_saved(broken, identity, 'RECEIVER_MATCH', split, 2)

    def test_common_budget_summary_preserves_negative_gains(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = self.identity(Engine())
            with patch('builtins.print'):
                saved = train.base.run_epochs(Engine(), root, identity, 'RECEIVER_MATCH', 2)
            states = {arm: deepcopy(saved) for arm in train.ARMS}
            for arm, state in states.items():
                state['arm'] = arm
            for row in states['RECEIVER_MATCH']['history']:
                row['validation_mae'] += 10.
            summary = train.paired_summary(states, identity, 2)
            pair = summary['comparisons']['RECEIVER_MATCH_vs_MESSAGE_SELF']
            self.assertAlmostEqual(pair['independent_best_at_common_budget']['mae_gain'], -10.)
            self.assertEqual(summary['baseline_new_training_updates'], 0)
            states['RECEIVER_MATCH']['completed_epoch'] = 1
            self.assertEqual(train.paired_summary(states, identity, 2)['status'], 'GW3_TRAINING_PARTIAL')

    def test_source_certificate_rejects_budget_role_replay_and_producer_changes(self):
        identity = dict(identity_sha256='source', engineering_only=False)
        states = {arm: dict(best_epoch=3, completed_epoch=5, best_checkpoint_sha256=arm+'best',
            _last_file_sha256=arm+'last') for arm in reference.REUSED_ARMS}
        metadata = dict(schema='original_igstgnn_graph_message_reuse_producer_v1',
            status='GW2_REUSE_PREDICTION_PASS', run_identity_sha256='source', engineering_only=False,
            common_completed_epochs=5, producer_source={name:reference.metrics.sha256(
                reference.gw2.REPO/'experiments/chronological'/name) for name in reference.PRODUCERS},
            checkpoint_roles={arm:dict(selection='best', epoch=3, checkpoint_sha256=arm+'best') for arm in states},
            validation_replay={arm:dict(passed=True) for arm in states})
        reference.validate_certificate(metadata, identity, states, 'best', 5)
        for change in (lambda x:x.update(common_completed_epochs=10),
            lambda x:x['checkpoint_roles']['POST_MESSAGE'].update(epoch=5),
            lambda x:x['validation_replay']['ORIGINAL'].update(passed=False),
            lambda x:x['producer_source'].update({'evaluate_original_graph_message_reuse.py':'wrong'})):
            bad = deepcopy(metadata); change(bad)
            with self.assertRaises(ValueError):
                reference.validate_certificate(bad, identity, states, 'best', 5)

    def test_full_case_report_fixed_hard_set_masks_and_negative_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = fixture()
            result = evaluation.report(archive, Path(temp)/'report', dict(engineering_only=True))
            self.assertEqual(result['status'], 'GW3_EVALUATION_PASS')
            self.assertEqual(len(result['cases']), 11)
            self.assertEqual(result['fixed_hard_ids'], [100])
            self.assertEqual(result['comparisons']['RECEIVER_MATCH_vs_ORIGINAL']['mae_gain_pooled'], -5.)
            self.assertEqual(result['comparisons']['receiver_query_dependency']['mae_gain_pooled'], 5.)
            self.assertNotIn('STATIC_UNIFORM', result['cases'])
            self.assertIn('GW3', (Path(temp)/'report'/'report.md').read_text(encoding='utf-8'))
            broken = deepcopy(archive); broken['valid'][0, 0, 0] = True
            with self.assertRaises(ValueError):
                evaluation.validate_archive(broken)

    def test_cached_baselines_do_not_construct_models_or_run_forward(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            archive = fixture()
            source = {k:v for k,v in archive.items() if not k.startswith('pred__')}
            source['schema'] = np.array(reference.cached_eval.SCHEMA)
            for case in reference.cached_eval.CASES:
                source['pred__'+case] = source['y']+1.
            path = directory/'paired_predictions.npz'
            np.savez(path, **source)
            meta_path = directory/'metadata.json'
            meta_path.write_text('{}', encoding='utf-8')
            saved = dict(archives={'best':dict(directory=str(directory), archive_sha256=evaluation.metrics.sha256(path),
                metadata_sha256=evaluation.metrics.sha256(meta_path))})
            modules = evaluation.metrics.get_original_modules()
            native_metrics = evaluation.metrics.native_metrics(source['pred__ORIGINAL'], source['y'], modules)
            states = {arm:dict(best_epoch=1, completed_epoch=1,
                history=[{'validation_'+key:value for key, value in native_metrics.items()}]) for arm in reference.REUSED_ARMS}
            identity = dict(data={'splits':{'val':{'selected_indices':source['sample_ids'].tolist()}}})
            with patch.object(train, 'model_for_arm', side_effect=AssertionError('No baseline model permitted')):
                result, _, replay = evaluation.reference_predictions(saved, 'best', identity, states, modules)
            self.assertTrue(all(row['passed'] and row['new_inference_calls'] == 0 for row in replay.values()))
            self.assertTrue(np.array_equal(result['pred__POST_MESSAGE'], source['pred__POST_MESSAGE']))

    def test_identical_best_last_tensor_cache_skips_forward_and_rejects_axis_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            directory = run/'eval_best_old'; directory.mkdir()
            archive = fixture()
            path = directory/'paired_predictions.npz'; np.savez(path, **archive)
            metadata = dict(schema='gw3_receiver_match_producer_v1', status='GW3_PREDICTION_PASS',
                run_identity_sha256='run', archive_sha256=evaluation.metrics.sha256(path),
                checkpoint_roles={'RECEIVER_MATCH':dict(selection='best', model_tensor_sha256='same')},
                validation_replay={'RECEIVER_MATCH':dict(passed=True)},
                producer_source={name:evaluation.metrics.sha256(evaluation.REPO/'experiments/chronological'/name)
                    for name in evaluation.PRODUCERS})
            (directory/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            role = dict(selection='last', model_tensor_sha256='same')
            with patch.object(train, 'model_for_arm', side_effect=AssertionError('No repeated forward')):
                self.assertIsNotNone(evaluation.existing_cache(run, {'identity_sha256':'run'}, 'RECEIVER_MATCH', role, archive))
            self.assertIsNone(evaluation.existing_cache(run, {'identity_sha256':'run'}, 'RECEIVER_MATCH',
                dict(model_tensor_sha256='different'), archive))
            changed = deepcopy(archive); changed['sample_ids'] = changed['sample_ids'][::-1]
            with self.assertRaises(ValueError):
                evaluation.existing_cache(run, {'identity_sha256':'run'}, 'RECEIVER_MATCH', role, changed)


if __name__ == '__main__':
    unittest.main()
