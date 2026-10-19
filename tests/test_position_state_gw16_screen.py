"""Server-only decision and archive integrity tests; no model training."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.position_state_gw16 import ARMS
from experiments.position_state_gw16.common import (SCHEMA, digest_json, load_config,
    sha256, write_json)
from experiments.position_state_gw16.evaluate import (ARCHIVE_SCHEMA, read_archive,
    validation_replay)
from experiments.position_state_gw16.objective import archive_diagnostics, metric_contract
from experiments.position_state_gw16.screen import (CANDIDATE, GUARDS,
    require_same_axes, screen_records)


def fixture():
    identity = dict(identity_sha256='run-1', config=dict(max_epochs=100), engineering_only=False,
        data=dict(loader_batching=dict(train_loader=dict(batches=82))),
        initializations={arm:dict(registered_parameters=10000, model_tensor_sha256='same') for arm in ARMS})
    best, last = {}, {}
    values = dict(NATIVE_POS=11., DIRECT=10.2, FREE_RECURRENT=10.1, ORIGIN_EXCHANGE=10., STATE_EXCHANGE=9.9)
    for selection, records in (('best', best), ('last', last)):
        for arm in ARMS:
            records[arm] = dict(identity_sha256='run-1', arm=arm, split='val', test_evaluated=False,
                checkpoint_role=dict(selection=selection, stage='joint', best_includes_stage_zero=True,
                    epoch=80 if selection == 'best' else 100, completed_training_epochs=100,
                    registered_parameters=10000, training_batches_per_epoch=82, global_updates=8200,
                    full_protocol_reached=True),
                diagnostics=dict(overall=dict(mae=values[arm]),
                    groups={group:dict(mae=values[arm], valid_count=12) for group in GUARDS}))
    return best, last, identity


class GW16ScreenTests(unittest.TestCase):
    def test_complete_matched_candidate_passes_without_confirmation_claim(self):
        result = screen_records(*fixture())
        self.assertTrue(result['candidate_for_confirmation'])
        self.assertFalse(result['independent_confirmation'])
        self.assertFalse(result['test_evaluated'])
        self.assertTrue(result['screening_failure_does_not_authorize_budget_extension'])
        self.assertEqual(result['fixed_guard_reference'], 'ORIGIN_EXCHANGE')
        self.assertAlmostEqual(result['gain_pct_vs_strongest_mechanism_control'], 1.)
        self.assertFalse(result['practical_target_3pct_met'])

    def test_three_percent_goal_does_not_replace_locked_screen(self):
        best, last, identity = fixture()
        best[CANDIDATE]['diagnostics']['overall']['mae'] = 9.6
        last[CANDIDATE]['diagnostics']['overall']['mae'] = 9.6
        result = screen_records(best, last, identity)
        self.assertTrue(result['practical_target_3pct_met'])
        self.assertTrue(result['candidate_for_confirmation'])

    def test_small_gain_vs_one_mechanism_control_fails(self):
        best, last, identity = fixture()
        best['ORIGIN_EXCHANGE']['diagnostics']['overall']['mae'] = 9.92
        result = screen_records(best, last, identity)
        self.assertFalse(result['candidate_for_confirmation'])
        self.assertFalse(result['checks']['BEST_gain_vs_ORIGIN_at_least_0_005'])

    def test_native_regression_fails_even_when_mechanism_controls_are_beaten(self):
        best, last, identity = fixture()
        best['NATIVE_POS']['diagnostics']['overall']['mae'] = 9.8
        result = screen_records(best, last, identity)
        self.assertFalse(result['checks']['BEST_no_worse_than_NATIVE_POS'])
        self.assertFalse(result['candidate_for_confirmation'])

    def test_guard_uses_one_overall_comparator_and_never_per_group_minimum(self):
        best, last, identity = fixture()
        best['DIRECT']['diagnostics']['groups']['flow_50_200']['mae'] = 1.
        result = screen_records(best, last, identity)
        self.assertTrue(result['candidate_for_confirmation'])
        self.assertEqual({v['reference'] for v in result['guard_results'].values()}, {'ORIGIN_EXCHANGE'})

    def test_guard_regression_fails_despite_overall_gain(self):
        best, last, identity = fixture()
        best[CANDIDATE]['diagnostics']['groups']['flow_50_200']['mae'] = 10.01
        result = screen_records(best, last, identity)
        self.assertFalse(result['candidate_for_confirmation'])
        self.assertFalse(result['guard_results']['flow_50_200']['passed'])

    def test_empty_guard_fails_without_nan(self):
        best, last, identity = fixture()
        for arm in (CANDIDATE, 'ORIGIN_EXCHANGE'):
            best[arm]['diagnostics']['groups']['high_value_ge314'] = dict(mae=None, valid_count=0)
        result = screen_records(best, last, identity)
        self.assertFalse(result['candidate_for_confirmation'])
        json.dumps(result, allow_nan=False)

    def test_degraded_last_fails(self):
        best, last, identity = fixture()
        last[CANDIDATE]['diagnostics']['overall']['mae'] = 10.1
        result = screen_records(best, last, identity)
        self.assertFalse(result['checks']['LAST_own_degradation_at_most_0_01'])

    def test_partial_and_engineering_are_never_candidates(self):
        for engineering, epochs, maximum in ((False, 20, 100), (True, 3, 3)):
            with self.subTest(engineering=engineering):
                best, last, identity = fixture()
                identity['engineering_only'] = engineering
                identity['config']['max_epochs'] = maximum
                for selection, records in (('best', best), ('last', last)):
                    for record in records.values():
                        record['checkpoint_role'].update(completed_training_epochs=epochs,
                            epoch=0 if selection == 'best' else epochs, global_updates=epochs*82,
                            full_protocol_reached=epochs == maximum)
                self.assertFalse(screen_records(best, last, identity)['candidate_for_confirmation'])

    def test_both_directions_of_capacity_mismatch_are_gated(self):
        for count in (9700, 10300):
            with self.subTest(parameters=count):
                best, last, identity = fixture()
                identity['initializations']['DIRECT']['registered_parameters'] = count
                for records in (best, last):
                    records['DIRECT']['checkpoint_role']['registered_parameters'] = count
                result = screen_records(best, last, identity)
                self.assertFalse(result['matched_capacity_claim'])
                self.assertFalse(result['candidate_for_confirmation'])

    def test_same_count_but_unpaired_exchange_tensors_rejected(self):
        best, last, identity = fixture()
        identity['initializations']['ORIGIN_EXCHANGE']['model_tensor_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'exact tensor pair'):
            screen_records(best, last, identity)

    def test_best_stage_zero_is_eligible_but_last_must_be_completed_epoch(self):
        best, last, identity = fixture()
        best[CANDIDATE]['checkpoint_role']['epoch'] = 0
        self.assertTrue(screen_records(best, last, identity)['candidate_for_confirmation'])
        last[CANDIDATE]['checkpoint_role']['epoch'] = 0
        with self.assertRaisesRegex(ValueError, 'LAST'):
            screen_records(best, last, identity)

    def test_tie_break_is_direct_then_free_then_origin(self):
        best, last, identity = fixture()
        for arm in ('DIRECT', 'FREE_RECURRENT', 'ORIGIN_EXCHANGE'):
            best[arm]['diagnostics']['overall']['mae'] = 10.
        self.assertEqual(screen_records(best, last, identity)['fixed_guard_reference'], 'DIRECT')
        best['DIRECT']['diagnostics']['overall']['mae'] = 10.01
        self.assertEqual(screen_records(best, last, identity)['fixed_guard_reference'], 'FREE_RECURRENT')

    def test_mixed_run_or_training_budget_rejected(self):
        best, last, identity = fixture()
        last[CANDIDATE]['identity_sha256'] = 'other-run'
        with self.assertRaisesRegex(ValueError, 'identity'):
            screen_records(best, last, identity)
        best, last, identity = fixture()
        last[CANDIDATE]['checkpoint_role']['global_updates'] -= 1
        with self.assertRaisesRegex(ValueError, 'different training budgets'):
            screen_records(best, last, identity)

    def test_paired_axes_reject_label_reference_and_order_changes(self):
        left = {key:np.arange(4, dtype=np.float32) for key in
            ('target', 'reference', 'sample_ids', 'station_ids', 'assignment', 'support', 'available')}
        require_same_axes(left, deepcopy(left))
        for key in left:
            with self.subTest(key=key):
                right = deepcopy(left)
                right[key] = right[key][::-1]
                with self.assertRaisesRegex(ValueError, key):
                    require_same_axes(left, right)

    def test_validation_replay_checks_counts_and_metrics(self):
        diagnostics = dict(overall=dict(mae=2., rmse=3., mape_percent=4., valid_count=9,
            mape_valid_count=7), region_mae=1., reference=dict(mae=5.))
        expected = dict(diagnostics['overall'], region_mae=1., reference_mae=5.)
        self.assertTrue(validation_replay(diagnostics, expected)['passed'])
        expected['rmse'] += .01
        self.assertFalse(validation_replay(diagnostics, expected)['passed'])
        expected['valid_count'] = 8
        with self.assertRaisesRegex(ValueError, 'count'):
            validation_replay(diagnostics, expected)


class GW16ArchiveTests(unittest.TestCase):
    def write_fixture(self, directory):
        config = load_config()
        config.update(nodes=2, horizon=2)
        assignment = np.array([0, 0], dtype=np.int64)
        target = np.array([[[[100.], [400.]], [[120.], [420.]]]], dtype=np.float32)
        prediction = target+1
        arrays = dict(prediction=prediction, target=target, reference=np.ones_like(target)*100,
            regional=prediction.mean(axis=2, keepdims=True), alpha=np.zeros((1, 2, 1, 1), np.float32),
            beta=np.zeros_like(target), support=np.array([[True, False]]), available=np.array([True]),
            station_ids=np.array([10, 20], dtype=np.int64), sample_ids=np.array([3909], dtype=np.int64),
            assignment=assignment, state_rms_batch=np.zeros((1, 2, 2), np.float32))
        identity = dict(schema=SCHEMA, config=config, engineering_only=False,
            data=dict(loader_batching=dict(val_loader=dict(samples=1, batches=1)),
                station_ids=[10, 20], partition=dict(assignment=[0, 0])))
        identity['identity_sha256'] = digest_json(identity)
        np.savez_compressed(directory/'predictions.npz', **arrays)
        metadata = dict(schema=ARCHIVE_SCHEMA, identity=identity,
            identity_sha256=identity['identity_sha256'], metric_contract=metric_contract(config),
            engineering_only=False, split='val', test_evaluated=False, arm=CANDIDATE,
            archive_sha256=sha256(directory/'predictions.npz'), validation_replay=dict(passed=True),
            diagnostics=archive_diagnostics(arrays, config))
        write_json(directory/'metadata.json', metadata)
        return arrays, metadata

    def test_checksum_change_rejected_on_every_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_fixture(directory)
            # Isolate archive authentication from the production release-only
            # configuration validator; the fixture intentionally has two nodes.
            with patch('experiments.position_state_gw16.evaluate.validate_identity'):
                read_archive(directory)
                with (directory/'predictions.npz').open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    read_archive(directory)

    def test_recomputed_diagnostics_reject_edited_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, metadata = self.write_fixture(directory)
            metadata['diagnostics']['overall']['mae'] += 1
            write_json(directory/'metadata.json', metadata)
            with patch('experiments.position_state_gw16.evaluate.validate_identity'):
                with self.assertRaisesRegex(ValueError, 'diagnostics'):
                    read_archive(directory)

    def test_rehashed_axis_change_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arrays, metadata = self.write_fixture(directory)
            arrays['station_ids'] = arrays['station_ids'][::-1]
            np.savez_compressed(directory/'predictions.npz', **arrays)
            metadata['archive_sha256'] = sha256(directory/'predictions.npz')
            write_json(directory/'metadata.json', metadata)
            with patch('experiments.position_state_gw16.evaluate.validate_identity'):
                with self.assertRaisesRegex(ValueError, 'axis'):
                    read_archive(directory)


if __name__ == '__main__':
    unittest.main()
