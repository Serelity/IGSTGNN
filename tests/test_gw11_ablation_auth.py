"""Server-only read-only provenance tests; fixtures are not accuracy evidence."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.state_gate_gw11 import ablation_auth as auth


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding='utf-8')


class Fixture:
    def __init__(self, root):
        self.root = Path(root); self.repo = self.root/'repo'; self.repo.mkdir()
        self.directory = self.root/'run'; self.directory.mkdir()
        self.evaluation = self.directory/'evaluation_original'; self.evaluation.mkdir()
        self.paths = {}
        digests = {}
        for name in ('probe', 'cache', 'gw10', 'audit'):
            path = self.root/name; path.mkdir(); self.paths[name] = path
            child = path/('train.npz' if name == 'cache' else 'bound.bin')
            child.write_bytes(name.encode()); digests[name] = {str(child): auth.sha256(child)}
        (self.repo/'source.py').write_text('source bytes only\n', encoding='utf-8')
        self.identity = dict(schema=auth.SCHEMA, arms=list(auth.ARMS), candidate='STATE_GATE',
            new_epochs=3, train_batches=82, train_count=3909, seed=2025, full_horizon=12,
            optimizer_reset_at_start=True, best_selection_includes_stage_zero=True,
            test_evaluated=False, extension_allowed=False,
            execution_source={'source.py': auth.sha256(self.repo/'source.py')},
            data={'splits': {'val': {'selected_indices': [3, 8]}}},
            source_certificate=dict(probe_directory=str(self.paths['probe']),
                file_digests=digests['probe'], protected_directories=[str(self.paths['probe'])]),
            gw10_reference=dict(directory=str(self.paths['gw10']), file_digests=digests['gw10']),
            mask_gate=dict(directory=str(self.paths['audit']), file_digests=digests['audit'],
                bindings={'cache_file_digests': digests['cache']}))
        self.identity['identity_sha256'] = auth.identity_digest(self.identity)
        write(self.directory/'identity.json', self.identity)
        self.identity_sha256 = self.identity['identity_sha256']
        write(self.directory/'training_started.json', dict(schema='gw11_training_started_marker_v1',
            training_schema=auth.SCHEMA, identity_sha256=self.identity_sha256,
            registered_arms=list(auth.ARMS), fixed_new_epochs=3,
            all_stage_zero_checkpoints_committed=True, written_before_first_optimizer_update=True))
        y = np.full((2, 12, 2), 100., dtype=np.float32); y[:, :, 1] = 400.
        axes = dict(schema=np.array(auth.ARCHIVE_SCHEMA), axis_order=np.array('sample,horizon,node'),
            y=y, valid=np.ones(y.shape, dtype=bool), sample_ids=np.array([3, 8]),
            node_ids=np.array([7, 9]), road_labels=np.array(['A', 'B']),
            direct_support=np.zeros((2, 2), dtype=bool))
        self.archives = {}; self.metadata = {}; self.histories = {}; best = {}
        for arm in auth.ARMS:
            directory = self.directory/arm; directory.mkdir()
            (directory/'last.pt').write_bytes(('opaque LAST '+arm).encode())
            (directory/'best_epoch_000.pt').write_bytes(('opaque BEST '+arm).encode())
            best[arm] = dict(epoch=0, mae=10., checkpoint_file='best_epoch_000.pt',
                checkpoint_sha256=auth.sha256(directory/'best_epoch_000.pt'), model_tensor_sha256='b'*64)
        self.summary = dict(schema=auth.SCHEMA, status='GW11_TRAINING_COMPLETE',
            identity_sha256=self.identity_sha256, fixed_new_epochs=3, registered_arms=list(auth.ARMS),
            completed_epochs={arm: 3 for arm in auth.ARMS}, updates={arm: 246 for arm in auth.ARMS},
            best=best, best_selection_includes_stage_zero=True, optimizer_reset_at_start=True,
            optimizer_reset_on_resume=False, test_evaluated=False, extension_allowed=False)
        for selection in ('best', 'last'):
            (self.evaluation/selection).mkdir()
            archive = {key: value.copy() for key, value in axes.items()}
            archive.update(pred__FROZEN_SUM=y+10., pred__ORIGIN=y.astype(np.float64)+9.75,
                pred__OWN4=y.astype(np.float64)+9.5)
            for index, arm in enumerate(auth.ARMS):
                error = 10. if selection == 'best' else (11., 10.125, 12., 12.)[index]
                archive['pred__'+arm] = y+np.float32(error)
            self.archives[selection] = archive
            rows = {case: auth._record(archive['pred__'+case], y, archive['valid']) for case in auth.CASES}
            roles = {}
            for arm in auth.ARMS:
                roles[arm] = (dict(best[arm]) if selection == 'best' else dict(epoch=3,
                    mae=rows[arm]['mae'], checkpoint_file='last.pt', model_tensor_sha256='c'*64,
                    checkpoint_sha256=auth.sha256(self.directory/arm/'last.pt')))
                roles[arm]['selection'] = selection
            self.metadata[selection] = dict(schema=auth.PRODUCER_SCHEMA, checkpoint_selection=selection,
                run_identity_sha256=self.identity_sha256, checkpoint_roles=roles,
                records={arm: rows[arm] for arm in auth.ARMS},
                historical_records={case: rows[case] for case in ('FROZEN_SUM', 'ORIGIN', 'OWN4')},
                optimizer_updates_this_evaluation=0, common_completed_new_epochs=3, test_evaluated=False)
        for arm in auth.ARMS:
            self.histories[arm] = []
            for epoch in range(1, 4):
                order = auth.epoch_order(2025, epoch, 3909)
                self.histories[arm].append(dict(epoch=epoch, global_updates=82*epoch,
                    train_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                    dropout_epoch_seed=auth.epoch_seed(2025, epoch), mask_difference_points=0,
                    validation=deepcopy(self.metadata['last']['records'][arm])))
        with patch.object(auth.protocol, 'FROZEN_GW10_OWN4_MAE', 9.5):
            self.decision = auth.protocol.screen(self.metadata['best']['records'],
                self.metadata['last']['records'], {'OWN4': {'mae': 9.5}})
        self.publish()

    def publish(self):
        write(self.directory/'summary.json', self.summary)
        for arm in auth.ARMS:
            write(self.directory/arm/'history.json', self.histories[arm])
        self.pins = {}; evaluations = {}
        for selection in ('best', 'last'):
            child = self.evaluation/selection
            np.savez_compressed(child/'paired_predictions.npz', **self.archives[selection])
            archive_sha = auth.sha256(child/'paired_predictions.npz')
            self.metadata[selection]['archive_sha256'] = archive_sha
            write(child/'metadata.json', self.metadata[selection])
            write(child/'metrics.json', {**self.metadata[selection]['historical_records'],
                **self.metadata[selection]['records']})
            self.pins[selection] = dict(metadata_sha256=auth.sha256(child/'metadata.json'), archive_sha256=archive_sha)
            evaluations[selection] = dict(directory=str(child), **self.pins[selection])
        decision = dict(self.decision, run_identity_sha256=self.identity_sha256, evaluations=evaluations)
        write(self.directory/'screening.json', decision); write(self.evaluation/'screening.json', decision)
        (self.evaluation/'report.md').write_text('synthetic failed screen\n', encoding='utf-8')
        (self.directory/'report.md').write_bytes((self.evaluation/'report.md').read_bytes())
        write(self.directory/'result_pointer.json', dict(directory=str(self.evaluation),
            report_sha256=auth.sha256(self.evaluation/'report.md'),
            screening_sha256=auth.sha256(self.evaluation/'screening.json')))

    def bindings(self):
        stack = ExitStack()
        stack.enter_context(patch.object(auth, 'SOURCE_IDENTITY_SHA256', self.identity_sha256))
        stack.enter_context(patch.object(auth, 'EXPECTED_EVALUATIONS', self.pins))
        stack.enter_context(patch.object(auth, 'PRODUCTION_SHAPE', (2, 12, 2)))
        stack.enter_context(patch.object(auth.protocol, 'FROZEN_GW10_OWN4_MAE', 9.5))
        return stack

    def load(self):
        with self.bindings():
            return auth.load_run(self.directory, self.repo)


class GW11AblationAuthenticationTests(unittest.TestCase):
    def test_read_only_completed_run_returns_full_last_and_json_safe_certificate(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp); before = sorted(str(path) for path in fixture.root.rglob('*'))
            archive, identity, certificate = fixture.load()
            self.assertEqual(identity, fixture.identity)
            self.assertEqual(archive['pred__STATE_GATE'].dtype, np.float32)
            self.assertEqual(certificate['summary'], fixture.summary)
            self.assertEqual(certificate['source_paths']['cache_dir'], str(fixture.paths['cache']))
            self.assertEqual(certificate['checkpoint_deserializations'], 0)
            self.assertEqual(certificate['neural_inference_calls'], 0)
            json.dumps(certificate, allow_nan=False)
            self.assertEqual(before, sorted(str(path) for path in fixture.root.rglob('*')))

    def test_unreturned_identity_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            with fixture.bindings(), patch.object(auth, 'SOURCE_IDENTITY_SHA256', '0'*64):
                with self.assertRaisesRegex(ValueError, 'returned completed'):
                    auth.load_run(fixture.directory, fixture.repo)

    def test_current_execution_source_bytes_must_match_original_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp); (fixture.repo/'source.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'digest differs'):
                fixture.load()

    def test_incomplete_or_extended_training_budgets_are_rejected(self):
        for field, value in (('completed_epochs', {arm: 2 for arm in auth.ARMS}),
                ('updates', {arm: 328 for arm in auth.ARMS}), ('fixed_new_epochs', 4)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                fixture = Fixture(temp); fixture.summary[field] = value; fixture.publish()
                with self.assertRaisesRegex(ValueError, 'completed summary'):
                    fixture.load()

    def test_missing_or_changed_training_start_marker_is_rejected(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temp:
                fixture = Fixture(temp); path = fixture.directory/'training_started.json'
                if missing: path.unlink()
                else: write(path, {'written_before_first_optimizer_update': False})
                with self.assertRaises(ValueError): fixture.load()

    def test_original_metadata_or_archive_digest_cannot_be_replaced(self):
        for name in ('metadata.json', 'paired_predictions.npz'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                fixture = Fixture(temp); (fixture.evaluation/'last'/name).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'digest differs'):
                    fixture.load()

    def test_result_pointer_cannot_select_a_different_repeated_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            write(fixture.directory/'result_pointer.json', dict(directory=str(fixture.root/'evaluation_other')))
            with self.assertRaisesRegex(ValueError, 'result pointer'):
                fixture.load()

    def test_root_and_nested_original_failed_decisions_must_agree(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp); path = fixture.directory/'screening.json'
            decision = json.loads(path.read_text()); decision['candidate_for_confirmation'] = True; write(path, decision)
            with self.assertRaisesRegex(ValueError, 'failed screening'):
                fixture.load()

    def test_last_checkpoint_is_hashed_without_deserializing_and_changes_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp); (fixture.directory/'STATE_GATE'/'last.pt').write_bytes(b'not the same opaque bytes')
            with self.assertRaisesRegex(ValueError, 'digest differs'):
                fixture.load()

    def test_best_epoch_must_remain_zero_in_metadata_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp); fixture.metadata['best']['checkpoint_roles']['STATE_GATE']['epoch'] = 1
            fixture.publish()
            with self.assertRaisesRegex(ValueError, 'epoch/tensor role'):
                fixture.load()

    def test_last_role_cannot_escape_its_arm_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp); fixture.metadata['last']['checkpoint_roles']['STATE_GATE']['checkpoint_file'] = '../last.pt'
            fixture.publish()
            with self.assertRaisesRegex(ValueError, 'LAST must use'):
                fixture.load()

    def test_prediction_archive_requires_exact_cases_dtype_and_shape(self):
        for mutation in ('extra', 'dtype', 'shape', 'nonfinite'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                fixture = Fixture(temp); archive = fixture.archives['last']
                if mutation == 'extra': archive['pred__UNREGISTERED'] = archive['y'].copy()
                elif mutation == 'dtype': archive['pred__STATE_GATE'] = archive['pred__STATE_GATE'].astype(np.float64)
                elif mutation == 'shape': archive['direct_support'] = np.zeros((2, 12, 2), dtype=bool)
                else: archive['pred__STATE_GATE'][0, 0, 0] = np.nan
                fixture.publish()
                with self.assertRaises(ValueError): fixture.load()

    def test_target_mask_and_axis_order_are_explicit_and_shared(self):
        for mutation in ('mask', 'sample', 'order', 'node'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                fixture = Fixture(temp); archive = fixture.archives['last']
                if mutation == 'mask': archive['valid'][0, 0, 0] = False
                elif mutation == 'sample': archive['sample_ids'] = np.array([8, 3])
                elif mutation == 'order': archive['axis_order'] = np.array('node,horizon,sample')
                else: archive['node_ids'] = np.array([7, 7])
                fixture.publish()
                with self.assertRaises(ValueError): fixture.load()

    def test_metadata_metrics_are_recomputed_from_the_bound_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp); fixture.metadata['last']['records']['STATE_GATE']['mae'] += .1
            fixture.publish()
            with self.assertRaisesRegex(ValueError, 'metric replay'):
                fixture.load()

    def test_best_predictions_must_exactly_equal_the_common_frozen_sum(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp); archive = fixture.archives['best']
            archive['pred__STATE_GATE'] = archive['pred__STATE_GATE'].copy()
            archive['pred__STATE_GATE'][0, 0, 0] += np.float32(.125)
            fixture.metadata['best']['records']['STATE_GATE'] = auth._record(
                archive['pred__STATE_GATE'], archive['y'], archive['valid'])
            fixture.publish()
            with self.assertRaisesRegex(ValueError, 'frozen prediction differ'):
                fixture.load()

    def test_history_order_rng_updates_masks_and_complete_epochs_are_checked(self):
        for field, value in (('global_updates', 1), ('train_order_sha256', 'wrong'),
                ('dropout_epoch_seed', 0), ('mask_difference_points', 1)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                fixture = Fixture(temp); fixture.histories['STATE_GATE'][0][field] = value; fixture.publish()
                with self.assertRaisesRegex(ValueError, 'history shared order'):
                    fixture.load()

    def test_nested_source_changes_are_captured(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp); (fixture.paths['cache']/'train.npz').write_bytes(b'changed cache')
            with self.assertRaisesRegex(ValueError, 'digest differs'):
                fixture.load()

    def test_assert_unchanged_detects_checkpoint_and_root_report_mutation(self):
        for relative in ('STATE_GATE/last.pt', 'report.md'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temp:
                fixture = Fixture(temp); _, _, certificate = fixture.load()
                (fixture.directory/relative).write_bytes(b'changed after certification')
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    auth.assert_unchanged(certificate)

    def test_output_is_fresh_and_outside_run_sources_and_data(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp); _, _, certificate = fixture.load(); data = fixture.root/'data'; data.mkdir()
            expected = fixture.root/'new_analysis'
            self.assertEqual(auth.require_output_outside_sources(expected, fixture.directory, certificate, data), expected)
            for output in (fixture.root, fixture.directory, fixture.directory/'analysis',
                    fixture.paths['cache']/'analysis', fixture.paths['probe']/'analysis', data/'analysis'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    auth.require_output_outside_sources(output, fixture.directory, certificate, data)


if __name__ == '__main__':
    unittest.main()
