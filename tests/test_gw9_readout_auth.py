"""Server-only synthetic tests for the NumPy, opaque-byte probe reader.

These fixtures certify plumbing and rejection behavior, not model accuracy.
No test imports or deserializes a model or training checkpoint.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.branch_readout_gw9 import readout_auth as auth


def archive_fixture():
    y = np.full((2,12,3), 400., dtype=np.float32)
    y[0,0,0] = 0.
    return dict(schema=np.array(auth.ARCHIVE_SCHEMA), axis_order=np.array('sample,horizon,node'),
        registered_arms=np.asarray(auth.ARMS), y=y, valid=y != 0.,
        sample_ids=np.array([3909,3910]), node_ids=np.array([11,12,13]),
        road_labels=np.array(['A','A','B']), direct_support=np.zeros((2,3), dtype=bool),
        pred__FROZEN_R=y+10., pred__SUM_CONTROL=y+9., pred__BRANCH_READOUT=y+9.)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


class SyntheticRun:
    def __init__(self, root):
        root = Path(root)
        self.repo, self.run, self.source = root/'repo', root/'probe', root/'source'
        for directory in (self.repo,self.run,self.source): directory.mkdir()
        (self.repo/'executed.py').write_text('# immutable synthetic source\n', encoding='utf-8')
        self.archive = archive_fixture()
        self.records = auth._metrics(self.archive)
        data = dict(splits=dict(val=dict(selected_indices=self.archive['sample_ids'].tolist())))
        source_identity = dict(data=data, schema='synthetic_source')
        source_identity['identity_sha256'] = auth._identity_digest(source_identity)
        write_json(self.source/'identity.json', source_identity)
        (self.source/'selected.pt').write_bytes(b'opaque selected source; never deserialized')
        cache = self.source/'cache'; cache.mkdir()
        np.savez_compressed(cache/'paired_predictions.npz',
            **{key:self.archive[key] for key in auth.AXIS_KEYS}, pred__R_CONTINUE=self.archive['pred__FROZEN_R'])
        write_json(cache/'metadata.json', dict(synthetic=True))
        source = dict(source_run=str(self.source), source_identity_sha256=source_identity['identity_sha256'],
            source_identity_file_sha256=auth.sha256(self.source/'identity.json'),
            source_initialization=dict(model_tensor_sha256='f'*64, checkpoint_file=str(self.source/'selected.pt'),
                checkpoint_sha256=auth.sha256(self.source/'selected.pt')),
            archives=dict(best=dict(directory=str(cache), metadata_sha256=auth.sha256(cache/'metadata.json'),
                archive_sha256=auth.sha256(cache/'paired_predictions.npz'))))
        initial = {arm:dict(head_tensor_sha256='e'*64, registered_parameters=8256, trainable_parameters=8256)
            for arm in auth.ARMS}
        self.identity = dict(schema=auth.SCHEMA, arms=list(auth.ARMS), fixed_screen_epochs=3,
            streaming_features=True, best_selection_includes_stage_zero=True, optimizer_reset_at_start=True,
            frozen_R_is_equal_budget_control=False, test_evaluated=False, initializations=initial,
            execution_source={'executed.py':auth.sha256(self.repo/'executed.py')}, source_reference=source,
            backbone_tensor_sha256='f'*64, data=data)
        self.identity['identity_sha256'] = auth._identity_digest(self.identity)
        write_json(self.run/'identity.json', self.identity)
        write_json(self.run/'budget.json', dict(target_completed_epochs=3, extension_allowed=False,
            backbone_updates=0, registered_heads=list(auth.ARMS), test_evaluated=False))
        (self.run/'last.pt').write_bytes(b'opaque paired last; do not load')
        self.last_sha = auth.sha256(self.run/'last.pt')
        best = {}
        for arm in auth.ARMS:
            filename = arm+'_epoch3.pt'
            (self.run/filename).write_bytes(arm.encode())
            best[arm] = dict(epoch=3,mae_pooled=self.records[arm]['mae_pooled'], checkpoint_file=filename,
                checkpoint_sha256=auth.sha256(self.run/filename))
        history = [dict(epoch=epoch, head_updates={arm:epoch*82 for arm in auth.ARMS},
            validation=self.records) for epoch in (1,2,3)]
        write_json(self.run/'history.json', history)
        evaluation = self.run/'evaluation_synthetic'; evaluation.mkdir()
        self.expected_evaluations = {}
        for role in ('best','last'):
            child = evaluation/role; child.mkdir()
            np.savez_compressed(child/'paired_predictions.npz', **self.archive)
            roles = {arm:dict(selection=role,epoch=3,policy=arm,head_tensor_sha256='d'*64,
                checkpoint_sha256=best[arm]['checkpoint_sha256'] if role == 'best' else self.last_sha)
                for arm in auth.ARMS}
            metadata = dict(schema=auth.PRODUCER_SCHEMA,status='GW9_PROBE_PREDICTION_PASS',
                checkpoint_selection=role,run_identity_sha256=self.identity['identity_sha256'],
                source_reference=source,execution_source=self.identity['execution_source'],
                backbone_tensor_sha256='f'*64,backbone_updates=0,head_updates={arm:246 for arm in auth.ARMS},
                completed_epoch=3,frozen_R_exact_cache_replay=True,frozen_R_replay=dict(exact_equal=True),
                optimizer_updates_this_evaluation=0,test_evaluated=False,
                archive_sha256=auth.sha256(child/'paired_predictions.npz'),head_roles=roles,
                validation_replay={arm:dict(passed=True) for arm in auth.ARMS})
            write_json(child/'metadata.json', metadata)
            self.expected_evaluations[role] = dict(metadata_sha256=auth.sha256(child/'metadata.json'),
                archive_sha256=metadata['archive_sha256'])
        pointer = dict(status='GW9_PROBE_SCREENING_COMPLETE',run_identity_sha256=self.identity['identity_sha256'],
            last_checkpoint_sha256=self.last_sha,evaluations=self.expected_evaluations,checks=auth.EXPECTED_CHECKS,
            eligible_for_joint_screen=False,candidate_for_confirmation=False,joint_training_started=False,
            frozen_R_is_equal_budget_control=False,screening_failure_does_not_authorize_budget_extension=True,
            interpretation='single_seed_development_probe_not_independent_confirmation',test_evaluated=False,
            evaluation_directory=evaluation.name)
        write_json(self.run/'screening.json',pointer); write_json(evaluation/'screening.json',pointer)
        summary = dict(schema=auth.SCHEMA,status='GW9_PROBE_TRAINING_COMPLETE',completed_epoch=3,
            target_completed_epochs=3,head_updates={arm:246 for arm in auth.ARMS},backbone_updates=0,
            source_reference=source,same_budget_comparison=list(auth.ARMS),best_selection_includes_stage_zero=True,
            extension_allowed=False,joint_training_started=False,frozen_R_is_equal_budget_control=False,
            test_evaluated=False,best=best,screening=pointer)
        write_json(self.run/'summary.json',summary)

    def pins(self):
        return patch.multiple(auth,SOURCE_IDENTITY_SHA256=self.identity['identity_sha256'],
            LAST_CHECKPOINT_SHA256=self.last_sha,EXPECTED_EVALUATIONS=self.expected_evaluations,
            PRODUCTION_SHAPE=self.archive['y'].shape)


class ReadoutAuthenticationTests(unittest.TestCase):
    def test_valid_archive_has_explicit_cases_and_shared_axes(self):
        archive = archive_fixture()
        y, valid, predictions = auth.validate_archive(archive)
        self.assertIs(y,archive['y']); self.assertIs(valid,archive['valid'])
        self.assertEqual(tuple(predictions),auth.CASES)

    def test_rejects_bad_axes_masks_ids_cases_and_nonfinite_values(self):
        changes = [lambda a:a.update(axis_order=np.array('node,horizon,sample')),
            lambda a:a.update(schema=np.array('different')),
            lambda a:a.update(registered_arms=np.asarray(auth.ARMS[::-1])),
            lambda a:a.update(sample_ids=np.array([3909,3909])),
            lambda a:a.update(node_ids=np.array([11.,12.,13.])),
            lambda a:a.update(direct_support=np.zeros((2,3))),
            lambda a:a.update(road_labels=np.array([1,2,3])),
            lambda a:a.update(pred__EXTRA=a['y']),
            lambda a:a['valid'].__setitem__((0,0,0),True),
            lambda a:a['pred__BRANCH_READOUT'].__setitem__((1,1,1),np.nan)]
        for change in changes:
            with self.subTest(change=change):
                archive = deepcopy(archive_fixture()); change(archive)
                with self.assertRaises(ValueError): auth.validate_archive(archive)

    def test_global_null_mask_handles_positive_fractional_null(self):
        archive = archive_fixture(); archive['y'][0,0,0] = .5
        archive['valid'] = archive['y'] != .5
        auth.validate_archive(archive)
        archive['valid'][0,0,0] = True
        with self.assertRaisesRegex(ValueError,'global-null'): auth.validate_archive(archive)

    def test_archive_rejects_object_labels_without_deserialization(self):
        archive = archive_fixture(); archive['road_labels'] = np.array(['A','A','B'],dtype=object)
        with self.assertRaisesRegex(ValueError,'road axis'): auth.validate_archive(archive)

    def test_complete_pinned_failed_probe_is_read_without_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticRun(temp)
            with fixture.pins():
                best,last,certificate = auth.load_authenticated_run(fixture.run,fixture.repo)
                self.assertTrue(np.array_equal(best['y'],last['y']))
                self.assertFalse(certificate['eligible_for_joint_screen'])
                self.assertEqual(certificate['checkpoint_deserializations'],0)
                self.assertIn(str((fixture.source/'selected.pt').resolve()),certificate['file_digests'])
                self.assertEqual(certificate['data_identity'],fixture.identity['data'])
                auth.assert_unchanged(certificate)

    def test_all_archive_loads_disable_pickle_and_reader_preserves_source_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticRun(temp)
            originals = {str(path):auth.sha256(path) for path in Path(temp).rglob('*') if path.is_file()}
            original_load = np.load
            with fixture.pins(),patch.object(auth.np,'load',wraps=original_load) as loading:
                auth.load_authenticated_run(fixture.run,fixture.repo)
            self.assertEqual(loading.call_count,3)
            self.assertTrue(all(call.kwargs.get('allow_pickle') is False for call in loading.call_args_list))
            self.assertEqual(originals,{str(path):auth.sha256(path)
                for path in Path(temp).rglob('*') if path.is_file()})

    def test_rejects_unpinned_identity_even_when_self_digest_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticRun(temp)
            with fixture.pins():
                altered = deepcopy(fixture.identity); altered['fixed_screen_epochs'] = 4
                altered['identity_sha256'] = auth._identity_digest(altered)
                write_json(fixture.run/'identity.json',altered)
                with self.assertRaisesRegex(ValueError,'returned GW9'): auth.load_authenticated_run(fixture.run,fixture.repo)

    def test_rejects_modified_source_checkpoint_and_execution_bytes(self):
        for category in ('checkpoint','code'):
            with self.subTest(category=category),tempfile.TemporaryDirectory() as temp:
                fixture = SyntheticRun(temp)
                filename = fixture.source/'selected.pt' if category == 'checkpoint' else fixture.repo/'executed.py'
                filename.write_bytes(b'changed')
                with fixture.pins(),self.assertRaisesRegex(ValueError,'digest differs'):
                    auth.load_authenticated_run(fixture.run,fixture.repo)

    def test_rejects_modified_producer_metadata_or_archive(self):
        for filename in ('metadata.json','paired_predictions.npz'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as temp:
                fixture = SyntheticRun(temp)
                (fixture.run/'evaluation_synthetic'/'best'/filename).write_bytes(b'changed')
                with fixture.pins(),self.assertRaisesRegex(ValueError,'digest differs'):
                    auth.load_authenticated_run(fixture.run,fixture.repo)

    def test_rejects_editing_failed_gate_to_allow_joint_training(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticRun(temp)
            path = fixture.run/'screening.json'; pointer=json.loads(path.read_text())
            pointer['eligible_for_joint_screen'] = True; write_json(path,pointer)
            with fixture.pins(),self.assertRaisesRegex(ValueError,'failed screening'):
                auth.load_authenticated_run(fixture.run,fixture.repo)

    def test_final_rehash_catches_change_after_authentication(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticRun(temp)
            with fixture.pins():
                _,_,certificate = auth.load_authenticated_run(fixture.run,fixture.repo)
            (fixture.run/'last.pt').write_bytes(b'changed after read')
            with self.assertRaisesRegex(ValueError,'Authenticated source changed'):
                auth.assert_unchanged(certificate)

    def test_oldest_reference_explicit_archive_paths_are_also_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'predictions.npz';metadata=root/'certificate.json'
            archive.write_bytes(b'old archive bytes');metadata.write_bytes(b'old metadata bytes')
            old_reference=dict(path=str(archive),sha256=auth.sha256(archive),
                metadata_path=str(metadata),metadata_sha256=auth.sha256(metadata))
            digests,protected={},set()
            auth._source_tree(old_reference,digests,protected)
            self.assertEqual(set(digests),{str(archive.resolve()),str(metadata.resolve())})
            self.assertIn(str(root.resolve()),protected)
            archive.write_bytes(b'changed old cache')
            with self.assertRaisesRegex(ValueError,'Authenticated source changed'):
                auth.assert_unchanged(dict(file_digests=digests))

    def test_outputs_must_be_fresh_and_outside_all_source_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); probe=root/'probe'; source=root/'source'
            probe.mkdir();source.mkdir()
            certificate=dict(protected_directories=[str(probe),str(source)])
            for output in (probe,probe/'analysis',source/'analysis',root):
                with self.subTest(output=output),self.assertRaises(ValueError):
                    auth.require_output_outside_sources(output,probe,certificate)
            result=auth.require_output_outside_sources(root/'analysis',probe,certificate)
            self.assertEqual(result,(root/'analysis').resolve())
            self.assertFalse(result.exists())

    def test_output_symlink_is_resolved_before_source_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);probe=root/'probe';probe.mkdir();link=root/'link'
            try: link.symlink_to(probe,target_is_directory=True)
            except (OSError,NotImplementedError): self.skipTest('Directory symlinks unavailable')
            with self.assertRaisesRegex(ValueError,'outside immutable'):
                auth.require_output_outside_sources(link/'analysis',probe,dict(protected_directories=[]))


if __name__ == '__main__': unittest.main()
