"""Server-only tests of pairing boundaries, not trained-model accuracy."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.original_baseline_gw16.compare import paired_scores, safe_output
from experiments.position_state_gw16.common import load_config


class OriginalGW16PairingTests(unittest.TestCase):
    def fixture(self):
        y = np.full((2, 12, 3, 1), 10., dtype=np.float32)
        y[0, 0, 0, 0] = 0.
        arrays = dict(target=y, prediction=y+2,
                      sample_ids=np.asarray([3909, 3910], dtype=np.int64),
                      station_ids=np.asarray([41, 51, 61], dtype=np.int64))
        base_meta = dict(data_identity=dict(hashes=dict(release='same')),
                         engineering_only=True, source=dict(training_complete=False), runtime=dict(threads=1))
        candidate_meta = dict(identity=dict(data=dict(hashes=dict(release='same')),
                                            config=load_config()), engineering_only=True,
                              arm='STATE_EXCHANGE', checkpoint_role=dict(epoch=1, full_protocol_reached=False))
        return arrays, copy.deepcopy(arrays), base_meta, candidate_meta

    def test_pairing_rejects_permuted_axis_even_when_predictions_and_shape_match(self):
        original, candidate, base_meta, candidate_meta = self.fixture()
        candidate['station_ids'] = candidate['station_ids'][::-1].copy()
        with self.assertRaisesRegex(ValueError, 'station_ids'):
            paired_scores(original, candidate, base_meta, candidate_meta)

    def test_pairing_rejects_changed_truth(self):
        original, candidate, base_meta, candidate_meta = self.fixture()
        candidate['target'][0, 1, 1, 0] += 1
        with self.assertRaisesRegex(ValueError, 'target'):
            paired_scores(original, candidate, base_meta, candidate_meta)

    def test_pairing_rejects_other_dataset_and_engineering_scope(self):
        for field in ('hashes', 'engineering'):
            original, candidate, base_meta, candidate_meta = self.fixture()
            if field == 'hashes':
                base_meta['data_identity']['hashes']['release'] = 'different'
            else:
                base_meta['engineering_only'] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                paired_scores(original, candidate, base_meta, candidate_meta)

    def test_both_predictions_use_identical_mask_and_no_test_claim(self):
        original, candidate, base_meta, candidate_meta = self.fixture()
        candidate['prediction'] = candidate['target']+1
        result = paired_scores(original, candidate, base_meta, candidate_meta)
        self.assertAlmostEqual(result['relative_mae_gain_vs_original_horizon_macro_percent'], 50.)
        self.assertEqual(result['rows']['ORIGINAL_IGSTGNN']['gw16_common_mask_pooled']['valid_count'], 71)
        self.assertEqual(result['rows']['STATE_EXCHANGE']['gw16_common_mask_pooled']['valid_count'], 71)
        self.assertFalse(result['test_evaluated'])
        self.assertFalse(result['screening_overridden'])
        self.assertFalse(result['candidate_for_confirmation'])
        self.assertFalse(result['paper_test_mae_directly_comparable_to_this_validation'])

    def test_output_cannot_be_inside_source_or_training_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.mkdir()
            with self.assertRaises(ValueError):
                safe_output(source/'overwrite', (source,))
            with self.assertRaisesRegex(ValueError, 'dataset'):
                safe_output(root/'data'/'overwrite', (source,), root/'data')
            (root/'identity.json').write_text('{}')
            with self.assertRaises(ValueError):
                safe_output(root/'compare', (source,))


if __name__ == '__main__':
    unittest.main()
