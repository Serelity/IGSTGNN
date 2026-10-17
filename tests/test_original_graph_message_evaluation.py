"""GW2 evaluation: fixed masks/axes, negative gains, replay and read-only analysis."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological import evaluate_original_graph_message as evaluation
from experiments.chronological.original_igstgnn_source import get_original_modules


def fixture():
    y = np.broadcast_to(np.array([1., 10., 75., 250.], np.float32)[None, None], (5, 12, 4)).copy()
    y[0, 0, 0] = 0.
    archive = dict(schema=np.array(evaluation.SCHEMA), axis_order=np.array('sample,horizon,node'),
        y=y, valid=y != 0., sample_ids=np.arange(100, 105), node_ids=np.arange(4),
        road_labels=np.array(['A', 'A', 'B', 'B']), direct_support=np.array([[True, False, False, False]]*5))
    for index, case in enumerate(evaluation.CASES):
        archive['pred__'+case] = y+index+1.
    return archive


class GraphMessageEvaluationTests(unittest.TestCase):
    def test_archive_rejects_wrong_axes_masks_case_set_ids_and_values(self):
        source = fixture()
        evaluation.validate_archive(source)
        mutations = [lambda a: a.update(axis_order=np.array('horizon,sample,node')),
            lambda a: a.update(schema=np.array('v03')),
            lambda a: a['valid'].__setitem__((0, 0, 0), True),
            lambda a: a['sample_ids'].__setitem__(0, 101),
            lambda a: a.pop('pred__STATIC'),
            lambda a: a['pred__STATIC'].__setitem__((0, 0, 0), np.nan),
            lambda a: a.update(direct_support=np.zeros((4, 5), bool))]
        for mutate in mutations:
            a = deepcopy(source); mutate(a)
            with self.assertRaises(ValueError):
                evaluation.validate_archive(a)

    def test_pair_report_retains_negative_gains_and_fixed_original_hard_ids(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            source = fixture()
            result = evaluation.report(source, Path(temp)/'report', dict(engineering_only=True))
            pair = result['comparisons']['POST_MESSAGE_vs_ORIGINAL']
            self.assertEqual(pair['mae_gain_pooled'], -3.)
            self.assertEqual(pair['fixed_hard_window_equal_weight_gain'], -3.)
            self.assertEqual(result['fixed_hard_ids'], [100])
            self.assertTrue((Path(temp)/'report'/'groups.csv').exists())
            with self.assertRaisesRegex(ValueError, 'preserved'):
                evaluation.report(source, Path(temp)/'report', {})

    def test_native_metrics_replay_rejects_changed_predictions(self):
        source = fixture(); y = source['y']; pred = source['pred__ORIGINAL']
        modules = get_original_modules()
        metrics = evaluation.native_metrics(pred, y, modules)
        state = dict(history=[{'validation_'+key: metrics[key] for key in ('mae', 'rmse', 'mape_fraction')}])
        self.assertTrue(evaluation.replay_metrics(pred, y, state, 1, modules)['passed'])
        self.assertFalse(evaluation.replay_metrics(pred+2, y, state, 1, modules)['passed'])

    def test_cli_archive_analysis_preserves_input_and_never_produces_new_predictions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            archive = Path(temp)/'saved.npz'
            np.savez(archive, **fixture())
            before = evaluation.sha256(archive)
            result = evaluation.main(['--archive', str(archive), '--output-dir', str(Path(temp)/'out')])
            self.assertEqual(result['inference_calls_this_analysis'], 0)
            self.assertEqual(evaluation.sha256(archive), before)
            self.assertFalse((Path(temp)/'out'/'paired_predictions.npz').exists())


if __name__ == '__main__':
    unittest.main()
