"""Server-only numerical contracts for the frozen GW13 information probe.

This file is authored and statically reviewed locally; run it on the server.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.conditional_residual_gw10 import median_probe as old
from experiments.trajectory_diagnostic_gw13 import model as probe


def fixture(samples=768, horizon=2, nodes=2):
    """Each shared child state has 96 points and at least three blocks.

    Amplitude, direction and shape vary on distinct cycles. At the default
    size each amplitude group has all four states for each competing child.
    """
    index = np.arange(samples)
    pred = np.full((samples, horizon, nodes), 20., dtype=np.float64)
    amplitude = (index % 16 + 1).astype(np.float64)
    direction = np.asarray([-.9, -.3, .3, .9])[(index // 64) % 4]
    shape = ((index // 16) % 4).astype(np.int64)
    features = dict(
        last_flow=np.full((samples, nodes), 20., dtype=np.float64),
        mean_abs_change=np.repeat(amplitude[:, None], nodes, axis=1),
        signed_direction=np.repeat(direction[:, None], nodes, axis=1),
        shape=np.repeat(shape[:, None], nodes, axis=1),
        neighbor_change_disagreement=np.ones((samples, nodes), dtype=np.float64))
    return dict(pred=pred, y=pred.copy(), valid=np.ones(pred.shape, dtype=np.bool_),
                features=features, sample_ids=index)


def prefix(data, stop):
    return dict(pred=data['pred'][:stop].copy(), y=data['y'][:stop].copy(),
        valid=data['valid'][:stop].copy(), sample_ids=data['sample_ids'][:stop].copy(),
        features={name: value[:stop].copy() for name, value in data['features'].items()})


def outputs(model, data):
    return dict(probe.predictions(model, data))


class TrajectoryModelTests(unittest.TestCase):
    def test_zero_residual_recovers_each_prediction_exactly(self):
        data = fixture()
        data['pred'] += np.arange(data['pred'].size).reshape(data['pred'].shape) / 16.
        data['y'] = data['pred'].copy()
        fitted = probe.fit_model(data)
        result = outputs(fitted, data)
        self.assertEqual(tuple(result), probe.ARMS)
        self.assertEqual(len(result), 7)
        for name, value in result.items():
            self.assertEqual(value.dtype, np.float64, name)
            np.testing.assert_array_equal(value, data['pred'], err_msg=name)

    def test_constant_residual_is_added_once_by_every_corrected_arm(self):
        data = fixture()
        data['y'] += 7.
        fitted = probe.fit_model(data)
        result = outputs(fitted, data)
        np.testing.assert_array_equal(result['SUM'], data['pred'])
        for name in probe.ARMS[1:]:
            np.testing.assert_array_equal(result[name], data['pred'] + 7., err_msg=name)
        for name in ('CONTEXT', *probe.CHILDREN):
            np.testing.assert_array_equal(fitted['tables'][name],
                np.zeros_like(fitted['tables'][name]), err_msg=name)

    def test_child_replaces_shared_context_instead_of_adding_it_twice(self):
        data = fixture()
        fitted = probe.fit_model(data)
        # A deliberately nonzero parent and context make a double-add visible.
        fitted['parent_tables']['OWN4'] = (np.asarray(fitted['parent_tables']['OWN4']) + 2.).tolist()
        fitted['tables']['CONTEXT'] = (np.asarray(fitted['tables']['CONTEXT']) + 3.).tolist()
        for name in probe.CHILDREN:
            fitted['tables'][name] = (np.asarray(fitted['tables'][name]) + 5.).tolist()
        result = outputs(fitted, data)
        np.testing.assert_array_equal(result['OWN4'], data['pred'] + 2.)
        np.testing.assert_array_equal(result['CONTEXT'], data['pred'] + 5.)
        for name in probe.CHILDREN:
            np.testing.assert_array_equal(result[name], data['pred'] + 7., err_msg=name)

    def test_prediction_needs_neither_labels_nor_mask_nor_old_neighbor_feature(self):
        data = fixture()
        data['y'] += 4.
        fitted = probe.fit_model(data)
        expected = outputs(fitted, data)
        scoring = dict(pred=data['pred'], features={name: value for name, value in data['features'].items()
            if name != 'neighbor_change_disagreement'})
        before = json.dumps(fitted, allow_nan=False, sort_keys=True)
        for name, actual in outputs(fitted, scoring).items():
            np.testing.assert_array_equal(actual, expected[name])
        scoring['y'] = object()
        scoring['valid'] = object()
        scoring['sample_ids'] = object()
        for name, actual in outputs(fitted, scoring).items():
            np.testing.assert_array_equal(actual, expected[name])
        self.assertEqual(before, json.dumps(fitted, allow_nan=False, sort_keys=True))

    def test_parent_own4_is_refitted_inside_supplied_training_prefix(self):
        full = fixture()
        full['y'][:384] += 2.
        full['y'][384:] += 100.
        fit = prefix(full, 384)
        expected = old.fit_probe(fit['pred'], fit['y'], fit['valid'],
                                 fit['features'], fit['sample_ids'])
        full_parent = old.fit_probe(full['pred'], full['y'], full['valid'],
                                   full['features'], full['sample_ids'])
        with mock.patch.object(old, 'fit_probe', wraps=old.fit_probe) as refit:
            fitted = probe.fit_model(fit)
        self.assertEqual(refit.call_count, 1)
        np.testing.assert_array_equal(refit.call_args.args[0], fit['pred'])
        np.testing.assert_array_equal(refit.call_args.args[1], fit['y'])
        np.testing.assert_array_equal(refit.call_args.args[4], fit['sample_ids'])
        for name in ('ORIGIN', 'OWN4'):
            np.testing.assert_array_equal(fitted['parent_tables'][name], expected['tables'][name])
        self.assertFalse(np.array_equal(fitted['parent_tables']['OWN4'], full_parent['tables']['OWN4']))
        self.assertEqual(fitted['fit_sample_count'], 384)
        expected_hash = hashlib.sha256(fit['sample_ids'].astype('<u8').tobytes()).hexdigest()
        self.assertEqual(fitted['fit_sample_ids_sha256'], expected_hash)

    def test_all_cutpoints_use_fit_inputs_and_stay_fixed_for_arbitrary_scoring(self):
        data = fixture()
        fitted = probe.fit_model(data)
        amplitude = data['features']['mean_abs_change']
        expected = np.quantile(amplitude, [.25, .5, .75])
        np.testing.assert_array_equal(fitted['cuts']['amplitude'], expected)
        groups = np.searchsorted(expected, amplitude, side='right')
        for group in range(4):
            selected = groups == group
            for child, feature in (('MAGNITUDE16', 'mean_abs_change'),
                                   ('DIRECTION16', 'signed_direction')):
                np.testing.assert_array_equal(fitted['cuts'][child][group],
                    np.quantile(data['features'][feature][selected], [.25, .5, .75]))
        serialized = json.dumps(fitted, sort_keys=True, allow_nan=False)
        scoring = fixture()
        scoring['features']['last_flow'][:] = 10000.
        scoring['features']['mean_abs_change'] *= 10000.
        scoring['features']['signed_direction'] *= -1.
        scoring['features']['shape'] = 3 - scoring['features']['shape']
        scoring['y'][:] = -999999.
        scoring['valid'][:] = False
        outputs(fitted, scoring)
        self.assertEqual(serialized, json.dumps(fitted, sort_keys=True, allow_nan=False))

    def test_shared_children_have_equal_capacity_and_no_node_axis(self):
        data = fixture()
        data['y'][:, :, 1] += 12.
        data['y'] += (data['features']['shape'] * 4.)[:, None, :]
        fitted = probe.fit_model(data)
        result = outputs(fitted, data)
        for name in probe.CHILDREN:
            self.assertEqual(np.asarray(fitted['tables'][name]).shape, (2, 128))
            self.assertEqual(np.asarray(fitted['tables'][name]).size, 2 * 8 * 4 * 4)
            self.assertEqual(np.asarray(fitted['support'][name]['valid_count']).shape, (2, 128))
            delta = result[name] - result['OWN4']
            np.testing.assert_allclose(delta[:, :, 0], delta[:, :, 1], rtol=0., atol=1e-14)
        self.assertEqual(np.asarray(fitted['tables']['CONTEXT']).shape, (2, 32))
        self.assertEqual(np.asarray(fitted['parent_tables']['OWN4']).shape, (2, 2, 8, 4))

    def test_rare_child_falls_back_to_context_and_unseen_context_to_zero(self):
        data = fixture()
        group = (data['features']['mean_abs_change'] - 1.) // 4.
        data['y'] += (group * 10.)[:, None, :]
        data['features']['shape'][:] = 0
        data['features']['shape'][15] = 1  # Two points in one block, within the high-amplitude context.
        fitted = probe.fit_model(data)
        result = outputs(fitted, data)
        np.testing.assert_array_equal(result['TRAJECTORY16'][15], result['CONTEXT'][15])
        self.assertTrue(np.any(result['CONTEXT'][15] != result['OWN4'][15]))
        counts = np.asarray(fitted['support']['TRAJECTORY16']['valid_count'])
        blocks = np.asarray(fitted['support']['TRAJECTORY16']['block_count'])
        self.assertTrue(np.any((counts == 2) & (blocks == 1)))
        scoring = prefix(data, 3)
        scoring['features']['last_flow'][:] = 314.
        unseen = outputs(fitted, scoring)
        for name in ('CONTEXT', *probe.CHILDREN):
            np.testing.assert_array_equal(unseen[name], unseen['OWN4'], err_msg=name)

    def test_direction_only_signal_is_identified_beyond_equal_capacity_controls(self):
        data = fixture()
        direction = data['features']['signed_direction']
        residual = direction * 10.
        data['y'] += residual[:, None, :]
        fitted = probe.fit_model(data)
        result = outputs(fitted, data)
        errors = {name: float(np.mean(np.abs(value - data['y']))) for name, value in result.items()}
        self.assertLess(errors['DIRECTION16'], errors['CONTEXT'])
        self.assertLess(errors['DIRECTION16'], errors['MAGNITUDE16'])
        self.assertLess(errors['DIRECTION16'], errors['TRAJECTORY16'])
        coverage = probe.coverage(fitted, data['features'], data['valid'])
        self.assertEqual(coverage['DIRECTION16']['active_fraction'], 1.)
        self.assertGreaterEqual(coverage['DIRECTION16']['min_observed_blocks'], 3)

    def test_shape_only_signal_is_identified_beyond_direction_and_magnitude(self):
        data = fixture()
        residual = (data['features']['shape'] * 6. - 9.)[:, None, :]
        data['y'] += residual
        fitted = probe.fit_model(data)
        result = outputs(fitted, data)
        errors = {name: float(np.mean(np.abs(value - data['y']))) for name, value in result.items()}
        self.assertLess(errors['TRAJECTORY16'], errors['CONTEXT'])
        self.assertLess(errors['TRAJECTORY16'], errors['MAGNITUDE16'])
        self.assertLess(errors['TRAJECTORY16'], errors['DIRECTION16'])
        coverage = probe.coverage(fitted, data['features'], data['valid'])
        self.assertEqual(coverage['TRAJECTORY16']['active_fraction'], 1.)

    def test_coverage_counts_active_scoring_cells_instead_of_training_fraction(self):
        data = fixture()
        fitted = probe.fit_model(data)
        score = prefix(data, 2)
        score['features']['last_flow'][1] = 314.
        coverage = probe.coverage(fitted, score['features'], score['valid'])
        for name, record in coverage.items():
            self.assertEqual(record['fit_active_fraction'], 1., name)
            self.assertEqual(record['valid_count'], 8, name)
            self.assertEqual(record['active_valid_count'], 4, name)
            self.assertEqual(record['active_fraction'], .5, name)
        score['valid'][0] = False
        masked = probe.coverage(fitted, score['features'], score['valid'])
        for name, record in masked.items():
            self.assertEqual(record['valid_count'], 4, name)
            self.assertEqual(record['active_valid_count'], 0, name)
            self.assertEqual(record['active_fraction'], 0., name)
        with self.assertRaises(ValueError):
            probe.coverage(fitted, score['features'], np.zeros_like(score['valid']))

    def test_support_needs_both_minimum_points_and_distinct_blocks(self):
        data = fixture()
        # Direction state 0 still has many points, but all lie in one author block.
        data['features']['signed_direction'][:] = .9
        data['features']['signed_direction'][:128] = -.9
        data['y'][:128] += 20.
        fitted = probe.fit_model(data)
        counts = np.asarray(fitted['support']['DIRECTION16']['valid_count'])
        blocks = np.asarray(fitted['support']['DIRECTION16']['block_count'])
        self.assertTrue(np.any((counts >= 64) & (blocks == 1)))
        result = outputs(fitted, data)
        np.testing.assert_array_equal(result['DIRECTION16'][:128], result['CONTEXT'][:128])
        self.assertLess(probe.coverage(fitted, data['features'], data['valid'])['DIRECTION16']['active_fraction'], 1.)

    def test_sixty_three_points_from_three_blocks_still_fall_back(self):
        data = fixture(samples=384, horizon=1, nodes=1)
        data['features']['mean_abs_change'][:] = 1.
        data['features']['shape'][:] = 0
        selected = np.r_[np.arange(21), np.arange(128, 149), np.arange(256, 277)]
        data['features']['shape'][selected] = 1
        data['y'][selected] += 20.
        fitted = probe.fit_model(data)
        counts = np.asarray(fitted['support']['TRAJECTORY16']['valid_count'])
        blocks = np.asarray(fitted['support']['TRAJECTORY16']['block_count'])
        self.assertTrue(np.any((counts == 63) & (blocks == 3)))
        result = outputs(fitted, data)
        np.testing.assert_array_equal(result['TRAJECTORY16'][selected], result['CONTEXT'][selected])
        data['features']['shape'][277] = 1
        data['y'][277] += 20.
        enough = probe.fit_model(data)
        counts = np.asarray(enough['support']['TRAJECTORY16']['valid_count'])
        blocks = np.asarray(enough['support']['TRAJECTORY16']['block_count'])
        self.assertTrue(np.any((counts == 64) & (blocks == 3)))
        now = outputs(enough, data)
        self.assertTrue(np.all(now['TRAJECTORY16'][selected] > now['CONTEXT'][selected]))

    def test_invalid_fit_features_masks_and_ids_are_rejected(self):
        for key, replacement in (('valid', np.ones((768, 2, 2), dtype=np.int64)),
                                 ('sample_ids', np.zeros(768, dtype=np.int64))):
            with self.subTest(key=key):
                data = fixture()
                data[key] = replacement
                with self.assertRaises(ValueError):
                    probe.fit_model(data)
        for feature, value in (('mean_abs_change', -1.), ('signed_direction', 2.),
                               ('last_flow', np.nan), ('shape', 4)):
            with self.subTest(feature=feature):
                data = fixture()
                data['features'][feature][0, 0] = value
                with self.assertRaises(ValueError):
                    probe.fit_model(data)
        data = fixture()
        data['features']['shape'] = data['features']['shape'].astype(float)
        with self.assertRaises(ValueError):
            probe.fit_model(data)

    def test_tampered_protocol_tables_cuts_and_support_are_rejected(self):
        data = fixture()
        fitted = probe.fit_model(data)
        mutations = (
            lambda m: m.update(schema='other'),
            lambda m: m['protocol'].update(min_points=1),
            lambda m: m.update(arms=list(reversed(probe.ARMS))),
            lambda m: m.update(nodes=True),
            lambda m: m['cuts'].update(amplitude=[3., 2., 1.]),
            lambda m: m['cuts']['DIRECTION16'].__setitem__(0, [0., np.nan, 1.]),
            lambda m: m['tables']['TRAJECTORY16'][0].__setitem__(0, float('inf')),
            lambda m: m['tables'].__setitem__('MAGNITUDE16', [[0.]]),
            lambda m: m['parent_tables'].__setitem__('OWN4', [[0.]]),
            lambda m: m['support']['TRAJECTORY16']['valid_count'][0].__setitem__(0, -1),
            lambda m: m['support']['DIRECTION16'].__setitem__('block_count', [[1.5]]),
            lambda m: m.update(fit_sample_count=0),
            lambda m: m.update(fit_valid_count=0),
            lambda m: m.update(fit_sample_ids_sha256='invalid'),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = deepcopy(fitted)
                mutate(changed)
                with self.assertRaises(ValueError):
                    outputs(changed, data)

    def test_json_round_trip_is_exact_and_refuses_overwrite(self):
        data = fixture()
        data['y'] += (data['features']['shape'] * 6. - 9.)[:, None, :]
        fitted = probe.fit_model(data)
        expected = outputs(fitted, data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            probe.save_model(fitted, path)
            raw = path.read_bytes()
            self.assertFalse(raw.startswith(b'\xef\xbb\xbf'))
            self.assertNotIn(b'\r', raw)
            self.assertTrue(raw.endswith(b'\n'))
            restored = json.loads(raw.decode('utf-8'))
            for name, actual in outputs(restored, data).items():
                np.testing.assert_array_equal(actual, expected[name])
            with self.assertRaises(FileExistsError):
                probe.save_model(fitted, path)


if __name__ == '__main__':
    unittest.main()
