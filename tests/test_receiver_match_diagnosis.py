"""Server-only tests: cached evidence diagnosis without torch/model execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import analyze_receiver_match_gw3 as diagnosis


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def signed_identity(schema):
    identity = dict(schema=schema, data={'splits':{'train':{'partition':[48, 21]}}},
        training_config={'bs':48, 'seed':2025}, source_fingerprint='native', engineering_only=False,
        initializations={'ORIGINAL':dict(original_tensor_sha256='same', post_construction_rng_sha256='same_rng')})
    identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def arrays(schema=diagnosis.GW3_SCHEMA, cases=diagnosis.GW3_CASES):
    y = np.broadcast_to(np.array([1., 10., 75., 250.], np.float32)[None, None], (5, 12, 4)).copy()
    y[0, 0, 0] = 0.
    result = dict(schema=np.array(schema), axis_order=np.array('sample,horizon,node'), y=y, valid=y != 0.,
        sample_ids=np.array([105, 102, 103, 104, 101]), node_ids=np.arange(4), road_labels=np.array(['A', 'A', 'B', 'B']),
        direct_support=np.array([[True, False, False, False]]*5))
    offsets = dict(ORIGINAL=1., STATIC=2., POST_MESSAGE=3., FIXED_UNIFORM=9., MESSAGE_SELF=8.,
        RECEIVER_MATCH=4., MESSAGE_SELF_UNIFORM=7., MESSAGE_SELF_STATIC_ONLY=7.5,
        RECEIVER_MATCH_UNIFORM=6., RECEIVER_MATCH_STATIC_ONLY=7., RECEIVER_MATCH_QUERY_SELF=5.,
        POST_LOCAL=5., POST_LOCAL_UNIFORM=6., POST_MESSAGE_UNIFORM=3.5, POST_MESSAGE_STATIC_ONLY=3.6)
    for case in cases:
        result['pred__'+case] = y+offsets[case]
    return result


def routing_stats():
    return dict(source_labels=['DYNAMIC_0'], weight_mean=[[1., 1., 1.]], weight_std=[[0., 0., 0.]], entropy_mean=[math_log_3()],
        by_phase={phase:{'DYNAMIC_0':dict(samples=5, calls=1, local_window_positions=3,
            weight_mean=[1., 1., 1.], across_sample_std_rms=[0., 0., 0.])} for phase in ('history', 'forecast_1')})


def math_log_3():
    return float(np.log(3.))


def workflow_fixture(root):
    run, gw2, gw1 = root/'gw3', root/'gw2', root/'gw1'
    identities = [signed_identity(schema) for schema in (
        'original_igstgnn_receiver_match_training_v1', 'original_igstgnn_graph_message_training_v1',
        'original_igstgnn_graph_window_training_v1')]
    for path, identity in zip((run, gw2, gw1), identities):
        dump(path/'identity.json', identity)
    reference = dict(source_run=str(gw2), source_identity_sha256=identities[1]['identity_sha256'],
        source_identity_file_sha256=diagnosis.sha256(gw2/'identity.json'), completed_epochs=2,
        gw1_reference=dict(source_run=str(gw1), source_identity_sha256=identities[2]['identity_sha256'],
            source_identity_file_sha256=diagnosis.sha256(gw1/'identity.json')),
        reused_arms={arm:dict(last_checkpoint_sha256=arm+'last') for arm in diagnosis.OLD_ARMS}, archives={})
    state_summaries = {}
    baseline = arrays()
    for arm in diagnosis.ARMS:
        row = diagnosis.metrics(baseline['pred__'+arm], baseline['y'], baseline['valid'])
        best = dict(mae=row['mae'], rmse=row['rmse'], mape_fraction=row['mape'])
        state_summaries[arm] = dict(completed_epoch=2, best_epoch=2, best_metrics=best, last_metrics=best,
            best_checkpoint_sha256=arm+'best', last_model_tensor_sha256=arm+'tensor')
        history = [dict(epoch=epoch, global_updates=epoch*2, train_mae=2.,
            validation_mae=row['mae']+(2-epoch), validation_rmse=row['rmse']+(2-epoch),
            validation_mape_fraction=row['mape']+(2-epoch)*.01, mechanisms=routing_stats()) for epoch in (1, 2)]
        source = run if arm in diagnosis.NEW_ARMS else gw2 if arm == 'POST_MESSAGE' else gw1
        dump(source/arm/'history.json', history)
    for selection in ('best', 'last'):
        directory = gw2/('eval_'+selection); directory.mkdir()
        old = arrays(diagnosis.GW2_SCHEMA, diagnosis.GW2_CASES)
        np.savez(directory/'paired_predictions.npz', **old)
        roles = {arm:dict(selection=selection, epoch=2,
            checkpoint_sha256=arm+('best' if selection == 'best' else 'last'), reused=arm != 'POST_MESSAGE') for arm in diagnosis.OLD_ARMS}
        old_meta = dict(schema='original_igstgnn_graph_message_reuse_producer_v1', status='GW2_REUSE_PREDICTION_PASS',
            run_identity_sha256=identities[1]['identity_sha256'], engineering_only=False, common_completed_epochs=2,
            reuse_reference=reference['gw1_reference'], checkpoint_roles=roles,
            validation_replay={arm:dict(passed=True) for arm in diagnosis.OLD_ARMS},
            archive_sha256=diagnosis.sha256(directory/'paired_predictions.npz'))
        dump(directory/'metadata.json', old_meta)
        dump(directory/'mechanisms.json', {'POST_MESSAGE':routing_stats()})
        reference['archives'][selection] = dict(directory=str(directory), metadata_sha256=diagnosis.sha256(directory/'metadata.json'),
            archive_sha256=old_meta['archive_sha256'])
    summary = dict(status='GW3_TRAINING_PASS', same_completed_training_budget=True, engineering_only=False,
        common_completed_epochs=2, arms=state_summaries, reference=reference)
    dump(run/'summary.json', summary)
    for selection in ('best', 'last'):
        directory = run/('eval_'+selection); directory.mkdir()
        np.savez(directory/'paired_predictions.npz', **baseline)
        old_meta = json.loads((gw2/('eval_'+selection)/'metadata.json').read_text())
        roles = {arm:dict(old_meta['checkpoint_roles'][arm], reused=True) for arm in diagnosis.OLD_ARMS}
        roles.update({arm:dict(selection=selection, epoch=2,
            checkpoint_sha256=arm+('best' if selection == 'best' else 'last'), model_tensor_sha256=arm+'tensor') for arm in diagnosis.NEW_ARMS})
        metadata = dict(schema='gw3_receiver_match_producer_v1', status='GW3_PREDICTION_PASS', engineering_only=False,
            run_identity_sha256=identities[0]['identity_sha256'], common_completed_epochs=2, reference=reference,
            checkpoint_roles=roles, validation_replay={arm:dict(passed=True) for arm in diagnosis.ARMS},
            archive_sha256=diagnosis.sha256(directory/'paired_predictions.npz'),
            producer_source={name:diagnosis.sha256(diagnosis.REPO/'experiments/chronological'/name) for name in diagnosis.PRODUCERS})
        dump(directory/'metadata.json', metadata)
        mechanisms = {case:routing_stats() for case in ('MESSAGE_SELF', 'RECEIVER_MATCH', 'RECEIVER_MATCH_UNIFORM', 'RECEIVER_MATCH_QUERY_SELF')}
        mechanisms['POST_MESSAGE'] = dict(reused=True, source_directory=str(gw2/('eval_'+selection)))
        if selection == 'last':
            mechanisms['RECEIVER_MATCH'] = dict(reused=True, source_directory=str(run/'eval_best'))
        dump(directory/'mechanisms.json', mechanisms)
    return run


class ReceiverMatchDiagnosisTests(unittest.TestCase):
    def test_mask_and_fixed_hard_ids_are_preserved_with_tie_break(self):
        archive = arrays()
        diagnosis.validate_archive(archive, diagnosis.GW3_SCHEMA, diagnosis.GW3_CASES)
        hard = diagnosis.fixed_hard(archive['pred__ORIGINAL'], archive['y'], archive['valid'], archive['sample_ids'])
        self.assertEqual(archive['sample_ids'][hard].tolist(), [101])
        for change in (lambda a:a['valid'].__setitem__((0, 0, 0), True),
            lambda a:a['sample_ids'].__setitem__(0, 102), lambda a:a.pop('pred__RECEIVER_MATCH_QUERY_SELF')):
            broken = deepcopy(archive); change(broken)
            with self.assertRaises(ValueError):
                diagnosis.validate_archive(broken, diagnosis.GW3_SCHEMA, diagnosis.GW3_CASES)

    def test_negative_gain_group_conservation_and_absolute_error_identity(self):
        data = arrays()
        predictions = {key[6:]:value for key, value in data.items() if key.startswith('pred__')}
        predictions['GW2_POST_MESSAGE_UNIFORM'] = data['y']+3.5
        with tempfile.TemporaryDirectory() as temp:
            result = diagnosis.analyze_pair('best', data, predictions, Path(temp))
        gain = next(row for row in result['comparisons'] if row['comparison'] == 'RECEIVER_MATCH_vs_POST_MESSAGE')
        self.assertAlmostEqual(gain['mae_gain'], -1.)
        self.assertEqual(gain['positive_gain'], 0.)
        self.assertEqual(gain['negative_gain'], 1.)
        self.assertLess(result['error_identity']['pointwise_residual_abs_max'], 1e-10)
        self.assertEqual(result['fixed_hard_ids'], [101])
        self.assertAlmostEqual(sum(row['global_mae_contribution'] for row in result['top_harm_groups']['flow:']), -1.)

    def test_equal_group_means_do_not_hide_opposing_point_effects(self):
        truth = np.ones((2, 12, 1))
        control = truth+2
        candidate = control.copy(); candidate[0] += 1; candidate[1] -= 1
        result = diagnosis.paired(control, candidate, truth, np.ones_like(truth, bool), truth.size)
        self.assertEqual(result['mae_gain'], 0.)
        self.assertEqual(result['cancellation'], 1.)
        self.assertEqual(result['positive_gain'], .5)
        self.assertEqual(result['negative_gain'], .5)

    def test_last_checkpoint_keeps_best_original_hard_ids(self):
        data = arrays()
        predictions = {key[6:]:value for key, value in data.items() if key.startswith('pred__')}
        predictions['GW2_POST_MESSAGE_UNIFORM'] = data['y']+3.5
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            best = diagnosis.analyze_pair('best', data, predictions, root)
            later = deepcopy(predictions)
            later['ORIGINAL'][0] = data['y'][0]+100.
            self.assertEqual(data['sample_ids'][diagnosis.fixed_hard(
                later['ORIGINAL'], data['y'], data['valid'], data['sample_ids'])].tolist(), [105])
            last = diagnosis.analyze_pair('last', data, later, root, fixed_hard_ids=best['fixed_hard_ids'])
            self.assertEqual(last['fixed_hard_ids'], [101])

    def test_cache_roles_budget_replay_and_archive_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = workflow_fixture(Path(temp))
            identity = json.loads((run/'identity.json').read_text())
            summary = json.loads((run/'summary.json').read_text())
            metadata = json.loads((run/'eval_best'/'metadata.json').read_text())
            diagnosis.certificate(metadata, identity, summary, 'best')
            for change in (lambda x:x.update(common_completed_epochs=100),
                lambda x:x['checkpoint_roles']['FIXED_UNIFORM'].update(epoch=1),
                lambda x:x['validation_replay']['RECEIVER_MATCH'].update(passed=False)):
                broken = deepcopy(metadata); change(broken)
                with self.assertRaises(ValueError):
                    diagnosis.certificate(broken, identity, summary, 'best')
            path = run/'eval_best'/'paired_predictions.npz'
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaises(ValueError):
                diagnosis.find_evaluation(run, identity, summary, 'best', diagnosis.Inputs(), run/'eval_best')

    def test_mechanism_cache_pointers_follow_source_and_reject_cycles_or_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = root/'a', root/'b'
            dump(a/'mechanisms.json', {'X':dict(reused=True, source_directory=str(b))})
            dump(b/'mechanisms.json', {'X':routing_stats()})
            stats, chain = diagnosis.resolve_mechanisms(a, 'X', diagnosis.Inputs(), (root,))
            self.assertEqual(len(chain), 2)
            summary, phases = diagnosis.flatten_mechanisms(stats)
            self.assertAlmostEqual(summary[0]['normalized_entropy'], 1.)
            self.assertEqual({row['phase'] for row in phases}, {'history', 'forecast_1'})
            dump(b/'mechanisms.json', {'X':dict(reused=True, source_directory=str(a))})
            with self.assertRaises(ValueError):
                diagnosis.resolve_mechanisms(a, 'X', diagnosis.Inputs(), (root,))
            with self.assertRaises(ValueError):
                diagnosis.resolve_mechanisms(a, 'X', diagnosis.Inputs(), (root/'unrelated',))

    def test_inputs_detect_changes_and_forbid_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'input.json'
            dump(path, {'value':1})
            ledger = diagnosis.Inputs(); ledger.json(path)
            dump(path, {'value':2})
            with self.assertRaises(ValueError):
                ledger.preserved()
            with self.assertRaises(ValueError):
                ledger.record(Path(temp)/'last.pt')

    def test_full_workflow_reads_no_torch_preserves_sources_and_retains_negative_results(self):
        import builtins
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                raise AssertionError('Diagnosis must never import models or torch')
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run = workflow_fixture(root)
            source_hashes = {str(path):diagnosis.sha256(path) for path in root.rglob('*') if path.is_file()}
            with patch('builtins.__import__', side_effect=guarded), patch('builtins.print'):
                result = diagnosis.main(['--run-dir', str(run), '--output-dir', str(root/'analysis')])
            self.assertEqual(result['status'], 'GW3_READ_ONLY_DIAGNOSIS_PASS')
            self.assertEqual(result['new_inference_calls'], 0)
            self.assertTrue(all(diagnosis.sha256(path) == value for path, value in source_hashes.items()))
            saved = json.loads((root/'analysis'/'summary.json').read_text())
            self.assertEqual(saved['mechanism_missing_cases'], 0)
            report = (root/'analysis'/'report.md').read_text(encoding='utf-8')
            self.assertIn('不是训练因果分解', report)
            self.assertIn('-1.000000', report)
            self.assertTrue((root/'analysis'/'matched_epoch_curves.csv').exists())
            with self.assertRaises(ValueError):
                diagnosis.main(['--run-dir', str(run), '--output-dir', str(root/'analysis')])

    def test_missing_mechanisms_are_explicit_without_discarding_prediction_diagnosis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run = workflow_fixture(root)
            (run/'eval_best'/'mechanisms.json').unlink()
            with patch('builtins.print'):
                diagnosis.main(['--run-dir', str(run), '--output-dir', str(root/'analysis')])
            saved = json.loads((root/'analysis'/'summary.json').read_text())
            self.assertGreater(saved['mechanism_missing_cases'], 0)
            self.assertEqual(saved['status'], 'GW3_READ_ONLY_DIAGNOSIS_PASS')


if __name__ == '__main__':
    unittest.main()
