"""Server-only numerical contracts; authored locally without running Python."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.trajectory_check_gw14 import matching
from experiments.trajectory_diagnostic_gw13 import model as frozen


def fixture(samples=384, nodes=4, horizon=2):
    """Every arm/state is present in every 128-sample author-order block."""
    index = np.arange(samples)
    direction = np.resize(np.asarray([-.9, -.3, .3, .9]), nodes)
    pred = np.full((samples, horizon, nodes), 20., dtype=np.float64)
    features = dict(last_flow=np.full((samples, nodes), 20., dtype=np.float64),
        mean_abs_change=np.broadcast_to((index % 16+1)[:, None], (samples, nodes)).astype(np.float64),
        signed_direction=np.broadcast_to(direction[None, :], (samples, nodes)).copy(),
        shape=np.broadcast_to(((index//16) % 4)[:, None], (samples, nodes)).copy(),
        neighbor_change_disagreement=np.ones((samples, nodes), dtype=np.float64))
    return dict(pred=pred, y=pred.copy(), valid=np.ones(pred.shape, dtype=np.bool_),
        features=features, sample_ids=index)


def one_context():
    data = fixture(nodes=1, horizon=1)
    data['features']['mean_abs_change'][:] = 1.
    return data


class MatchedTrajectoryTests(unittest.TestCase):
    def test_budget_is_exact_per_context_and_no_inherited_fit_is_called(self):
        data = fixture()
        parent = frozen.fit_model(data)
        before = json.dumps(parent, sort_keys=True, allow_nan=False)
        with patch.object(frozen, 'fit_model', side_effect=AssertionError('No GW13 refit')), \
                patch.object(frozen.old, 'fit_probe', side_effect=AssertionError('No parent refit')):
            fitted = matching.fit_matched_model(parent, data)
        self.assertEqual(before, json.dumps(parent, sort_keys=True, allow_nan=False))
        self.assertEqual(fitted['frozen_parent_model'], parent)
        self.assertTrue(fitted['match_exact'])
        retained = [row for row in fitted['contexts'] if row['retained']]
        self.assertEqual(len(retained), 8)
        for row in retained:
            self.assertEqual(row['k'], 4)
            self.assertEqual(row['common_block_ids'], [0, 1, 2])
            self.assertEqual(row['per_block_quota'], [32, 32, 32])
            self.assertEqual(row['n_per_cell'], 96)
            self.assertEqual(row['w'], 96/224)
            for arm in matching.CHILDREN:
                self.assertEqual(row[arm+'_retained_states'], [0, 1, 2, 3])
                h, c = row['horizon']-1, row['context']
                self.assertEqual(fitted['support'][arm]['valid_count'][h][c*4:c*4+4], [96]*4)
                self.assertEqual(fitted['support'][arm]['block_count'][h][c*4:c*4+4], [3]*4)
        self.assertEqual(fitted['fit_stage_counts']['inherited_parent_context_refits'], 0)
        self.assertEqual(fitted['fit_stage_counts']['child_tables_refitted'], 3)

    def test_fixed_selection_is_reproducible_and_independent_of_label_values(self):
        data = fixture(nodes=5)
        parent = frozen.fit_model(data)
        first = matching.fit_matched_model(parent, data)
        second = matching.fit_matched_model(parent, data)
        self.assertEqual(first, second)
        altered = deepcopy(data)
        altered['y'] += (np.arange(altered['y'].size).reshape(altered['y'].shape) % 7)*3.
        third = matching.fit_matched_model(parent, altered)
        for key in ('retained', 'support', 'contexts', 'selected_point_ids_sha256', 'fit_stage_counts'):
            self.assertEqual(first[key], third[key], key)
        self.assertNotEqual(first['tables'], third['tables'])
        # Unequal native state support actually triggers subsampling.
        self.assertTrue(any(row['retained'] and row['k'] < 4 for row in first['contexts']))
        selected = first['fit_stage_counts']['per_arm']
        self.assertEqual(len({row['selected_valid_points'] for row in selected.values()}), 1)
        self.assertEqual(len({row['retained_cells'] for row in selected.values()}), 1)

    def test_failed_shared_block_intersection_falls_back_for_all_arms(self):
        data = fixture(samples=512)
        index = np.arange(512)
        shape = ((index % 3)+(index//128)+1) % 4
        data['features']['shape'][:] = shape[:, None]
        parent = frozen.fit_model(data)
        fitted = matching.fit_matched_model(parent, data)
        failed = [row for row in fitted['contexts'] if row['requested_k'] > 0]
        self.assertTrue(failed)
        self.assertTrue(all(row['requested_k'] == 4 for row in failed))
        self.assertTrue(all(row['reason'] == 'fewer_than_three_shared_blocks' for row in failed))
        self.assertTrue(all(row['k'] == 0 and not row['retained'] for row in failed))
        outputs = dict(matching.predictions(fitted, data))
        for arm in matching.CHILDREN:
            np.testing.assert_array_equal(outputs[arm], outputs['CONTEXT'])

    def test_three_shared_blocks_with_only_three_matched_points_still_fall_back(self):
        data = one_context()
        index = np.arange(384)
        shape = index % 3+1
        shape[np.r_[np.arange(126), 128, 256]] = 0
        data['features']['shape'][:, 0] = shape
        negative = np.r_[0, np.arange(128, 254), 256]
        remaining = np.setdiff1d(index, negative)
        direction = np.full(384, .9)
        direction[negative] = -.9
        direction[remaining[:128]] = 0.
        data['features']['signed_direction'][:, 0] = direction
        parent = frozen.fit_model(data)
        fitted = matching.fit_matched_model(parent, data)
        row = next(row for row in fitted['contexts'] if row['requested_k'])
        self.assertEqual(row['requested_k'], 1)
        self.assertEqual(row['common_block_count'], 3)
        self.assertEqual(row['per_block_quota'], [1, 1, 1])
        self.assertEqual(row['candidate_n_per_cell'], 3)
        self.assertEqual(row['n_per_cell'], 0)
        self.assertEqual(row['reason'], 'fewer_than_64_matched_points')

    def test_no_eligible_state_produces_empty_common_mask_without_error(self):
        data = one_context()
        data['features']['shape'][:] = 0
        data['features']['signed_direction'][:, 0] = np.repeat([-.9, 0., .9], 128)
        parent = frozen.fit_model(data)
        fitted = matching.fit_matched_model(parent, data)
        self.assertFalse(any(row['retained'] for row in fitted['contexts']))
        common, report = matching.score_masks(fitted, data['features'], data['valid'])
        self.assertFalse(common.any())
        self.assertEqual(report['common_fraction'], 0.)
        self.assertEqual(report['contexts_retained'], 0)
        self.assertTrue(all(not np.asarray(fitted['retained'][arm]).any() for arm in matching.CHILDREN))

    def test_predictions_add_child_once_and_do_not_access_scoring_labels_or_mask(self):
        data = fixture()
        data['y'] += (data['features']['shape']*6.-9.)[:, None, :]
        parent = frozen.fit_model(data)
        fitted = matching.fit_matched_model(parent, data)
        scoring = dict(pred=data['pred'], features=data['features'], y=object(), valid=object(), sample_ids=object())
        outputs = dict(matching.predictions(fitted, scoring))
        self.assertEqual(tuple(outputs), matching.ARMS)
        inherited = dict(frozen.predictions(parent, scoring))
        np.testing.assert_array_equal(outputs['OWN4'], inherited['OWN4'])
        np.testing.assert_array_equal(outputs['CONTEXT'], inherited['CONTEXT'])
        _, _, context, states = frozen._bins(data['features'], parent['cuts'])
        h = np.arange(parent['horizon'])[None, :, None]
        for arm in matching.CHILDREN:
            delta = np.asarray(fitted['tables'][arm])[h, (context*4+states[arm])[:, None, :]]
            np.testing.assert_array_equal(outputs[arm], outputs['OWN4']+delta)
        self.assertTrue(any(np.any(outputs[arm] != outputs['CONTEXT']) for arm in matching.CHILDREN))

    def test_common_mask_is_the_exact_intersection_on_native_valid_points(self):
        data = fixture(nodes=5)
        parent = frozen.fit_model(data)
        fitted = matching.fit_matched_model(parent, data)
        valid = data['valid'].copy()
        valid[0, 0, 0] = False
        valid[1, 1, 1] = False
        common, report = matching.score_masks(fitted, data['features'], valid)
        _, _, context, states = frozen._bins(data['features'], parent['cuts'])
        h = np.arange(parent['horizon'])[None, :, None]
        expected = valid.copy()
        for arm in matching.CHILDREN:
            active = np.asarray(fitted['retained'][arm])[h, (context*4+states[arm])[:, None, :]] & valid
            expected &= active
            self.assertEqual(report['per_arm_active'][arm]['active_valid_count'], int(active.sum()))
        np.testing.assert_array_equal(common, expected)
        self.assertEqual(report['total_valid_count'], int(valid.sum()))
        self.assertEqual(report['common_valid_count'], int(expected.sum()))
        self.assertEqual(report['common_fraction'], expected.sum()/valid.sum())
        self.assertLess(report['common_fraction'], 1.)

    def test_wrong_fit_identity_or_support_cannot_reuse_a_frozen_parent(self):
        data = fixture()
        parent = frozen.fit_model(data)
        changed_ids = deepcopy(data)
        changed_ids['sample_ids'] += 128
        with self.assertRaisesRegex(ValueError, 'identity'):
            matching.fit_matched_model(parent, changed_ids)
        changed_features = deepcopy(data)
        changed_features['features']['shape'][:] = 0
        with self.assertRaisesRegex(ValueError, 'support'):
            matching.fit_matched_model(parent, changed_features)
        changed_mask = deepcopy(data)
        changed_mask['valid'][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, 'identity'):
            matching.fit_matched_model(parent, changed_mask)

    def test_validator_rejects_unequal_budgets_or_changed_frozen_parent(self):
        data = fixture()
        fitted = matching.fit_matched_model(frozen.fit_model(data), data)
        row_index = next(i for i, row in enumerate(fitted['contexts']) if row['retained'])
        row = fitted['contexts'][row_index]
        h, key = row['horizon']-1, row['context']*4
        mutations = (
            lambda m: m['protocol'].update(seed=2026),
            lambda m: m.update(match_exact=False),
            lambda m: m['frozen_parent_model']['tables']['CONTEXT'][h].__setitem__(row['context'], 99.),
            lambda m: m['support']['TRAJECTORY16']['valid_count'][h].__setitem__(key, 95),
            lambda m: m['support']['DIRECTION16']['block_count'][h].__setitem__(key, 4),
            lambda m: m['retained']['MAGNITUDE16'][h].__setitem__(key, False),
            lambda m: m['contexts'][row_index].update(w=.5),
            lambda m: m['contexts'][row_index].update(per_block_quota=[31, 32, 32]),
            lambda m: m['contexts'][row_index].update(TRAJECTORY16_retained_states=[0, 1, 2]),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = deepcopy(fitted)
                mutate(changed)
                with self.assertRaises(ValueError):
                    matching.validate_model(changed)

    def test_zero_residual_and_json_round_trip_preserve_predictions(self):
        data = fixture()
        fitted = matching.fit_matched_model(frozen.fit_model(data), data)
        expected = dict(matching.predictions(fitted, data))
        for value in expected.values():
            np.testing.assert_array_equal(value, data['pred'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'matched_model.json'
            matching.save_model(fitted, path)
            raw = path.read_bytes()
            self.assertTrue(raw.endswith(b'\n'))
            self.assertFalse(raw.startswith(b'\xef\xbb\xbf'))
            self.assertNotIn(b'\r', raw)
            restored = json.loads(raw.decode('utf-8'))
            for arm, value in matching.predictions(restored, data):
                np.testing.assert_array_equal(value, expected[arm])
            with self.assertRaises(FileExistsError):
                matching.save_model(fitted, path)


if __name__ == '__main__':
    unittest.main()
